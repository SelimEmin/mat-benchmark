"""A directed message-passing neural network with three output heads.

This is the Yang et al. 2019 D-MPNN architecture (the one chemprop implements),
written directly in PyTorch rather than installed as a package: the modality flag and
the masked three-head loss both need to reach inside the model, which is awkward
through chemprop's interface, and the package pulls in a large dependency tree for a
2-CPU container. Everything characteristic of the architecture is here --

  * messages live on DIRECTED BONDS, not atoms, so a message never immediately returns
    along the edge it arrived on (the property that distinguishes D-MPNN from a plain
    MPNN and the reason it does better on molecular property prediction)
  * messages are initialised from (atom, bond) pairs, updated for `depth` steps with a
    shared weight matrix and a residual connection to the initial message
  * atom representations come from summing incoming bond messages; the molecule vector
    is their mean
  * a feed-forward head maps [molecule vector | modality flag] to THREE outputs

MULTI-TASK. The three transporters are predicted by three heads on one shared molecular
representation. Most compounds are missing at least one target, so the loss is masked:
each compound contributes only the targets it actually has. This is the form of
multi-task that the earlier one-hot GBM arm could not express -- there the model had to
spend tree capacity separating the targets, whereas here they share a trunk and the
heads are separate by construction.

Comparison is against the production GBM on the SAME global scaffold folds used in
improve_experiment.py, scored the same way (stacked single-modality keys).
"""
import json, time, warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold
from scipy import stats

warnings.filterwarnings("ignore")
RDLogger.DisableLog("rdApp.*")
torch.set_num_threads(2)
TARGETS = ("DAT", "NET", "SERT")
NFOLD, EPOCHS, HIDDEN, DEPTH, FFN, BATCH, LR = 5, 35, 300, 3, 300, 64, 1e-3

ATOMS = list(range(1, 101))
DEG = [0, 1, 2, 3, 4, 5]
CHG = [-2, -1, 0, 1, 2]
HS = [0, 1, 2, 3, 4]
HYB = [Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
       Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
       Chem.rdchem.HybridizationType.SP3D2]
BT = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE,
      Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC]
ADIM = len(ATOMS) + len(DEG) + len(CHG) + len(HS) + len(HYB) + 2
BDIM = len(BT) + 2


def onek(x, allowed):
    v = [0.0] * (len(allowed) + 0)
    if x in allowed:
        v[allowed.index(x)] = 1.0
    return v


def atom_feat(a):
    return (onek(a.GetAtomicNum(), ATOMS) + onek(a.GetTotalDegree(), DEG)
            + onek(a.GetFormalCharge(), CHG) + onek(a.GetTotalNumHs(), HS)
            + onek(a.GetHybridization(), HYB)
            + [1.0 if a.GetIsAromatic() else 0.0, a.GetMass() * 0.01])


def bond_feat(b):
    return (onek(b.GetBondType(), BT)
            + [1.0 if b.GetIsConjugated() else 0.0,
               1.0 if b.IsInRing() else 0.0])


def mol_graph(smiles):
    """Directed-bond graph: f_atoms, f_bonds, and for each directed bond the list of
    incoming bonds excluding its own reverse."""
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return None
    fa = [atom_feat(a) for a in m.GetAtoms()]
    fb, b2a, rev, a2b = [], [], [], [[] for _ in range(m.GetNumAtoms())]
    for b in m.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        f = bond_feat(b)
        for (u, v) in ((i, j), (j, i)):
            fb.append(fa[u] + f)       # message is initialised from (source atom, bond)
            b2a.append(u)
            a2b[v].append(len(fb) - 1)
        rev.extend([len(fb) - 1, len(fb) - 2])
    if not fb:                          # single-atom molecule
        fb = [[0.0] * (ADIM + BDIM)]
        b2a, rev = [0], [0]
    return dict(f_atoms=np.array(fa, np.float32), f_bonds=np.array(fb, np.float32),
                b2a=np.array(b2a, np.int64), rev=np.array(rev, np.int64),
                a2b=a2b, n_atoms=m.GetNumAtoms())


def collate(graphs):
    """Pack a batch into flat arrays with index offsets, plus padded incoming lists."""
    fa = [np.zeros((1, ADIM), np.float32)]
    fb = [np.zeros((1, ADIM + BDIM), np.float32)]
    b2a, rev, scope, a2b = [0], [0], [], [[]]
    na, nb = 1, 1
    for g in graphs:
        fa.append(g["f_atoms"]); fb.append(g["f_bonds"])
        b2a.append(g["b2a"] + na)
        rev.append(g["rev"] + nb)
        for inc in g["a2b"]:
            a2b.append([b + nb for b in inc])
        scope.append((na, g["n_atoms"]))
        na += g["n_atoms"]; nb += len(g["f_bonds"])
    mx = max(1, max(len(x) for x in a2b))
    pad = np.zeros((len(a2b), mx), np.int64)
    for i, x in enumerate(a2b):
        pad[i, :len(x)] = x
    return (torch.from_numpy(np.concatenate(fa)),
            torch.from_numpy(np.concatenate(fb)),
            torch.from_numpy(np.concatenate([np.array(b2a[:1]), *b2a[1:]])
                             if len(b2a) > 1 else np.array(b2a)),
            torch.from_numpy(np.concatenate([np.array(rev[:1]), *rev[1:]])
                             if len(rev) > 1 else np.array(rev)),
            torch.from_numpy(pad), scope)


class DMPNN(nn.Module):
    def __init__(self, hidden=HIDDEN, depth=DEPTH, ffn=FFN, n_out=3, extra=1):
        super().__init__()
        self.depth = depth
        self.W_i = nn.Linear(ADIM + BDIM, hidden, bias=False)
        self.W_h = nn.Linear(hidden, hidden, bias=False)
        self.W_o = nn.Linear(ADIM + hidden, hidden)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(0.1)
        self.ffn = nn.Sequential(nn.Linear(hidden + extra, ffn), nn.ReLU(),
                                 nn.Dropout(0.1), nn.Linear(ffn, n_out))

    def forward(self, f_atoms, f_bonds, b2a, rev, a2b, scope, extra):
        msg = self.act(self.W_i(f_bonds))
        msg0 = msg
        for _ in range(self.depth - 1):
            # sum of incoming messages at the source atom, minus the reverse bond:
            # this is what makes the passing DIRECTED
            nei = msg[a2b].sum(dim=1)              # per atom
            m = nei[b2a] - msg[rev]                # per directed bond
            msg = self.act(msg0 + self.W_h(m))
            msg = self.drop(msg)
        nei = msg[a2b].sum(dim=1)
        h = self.act(self.W_o(torch.cat([f_atoms, nei], dim=1)))
        h = self.drop(h)
        out = torch.stack([h[a:a + n].mean(dim=0) for a, n in scope])
        return self.ffn(torch.cat([out, extra], dim=1))


def scaffold_folds_seeded(scaffolds, n_splits=5, seed=0):
    rng = np.random.default_rng(seed)
    groups = {}
    for i, s in enumerate(pd.Series(scaffolds).fillna("").values):
        groups.setdefault(s, []).append(i)
    keys = list(groups)
    rng.shuffle(keys)
    keys.sort(key=lambda k: -len(groups[k]))
    folds = [[] for _ in range(n_splits)]
    order = list(range(n_splits))
    for j, k in enumerate(keys):
        if j % n_splits == 0:
            rng.shuffle(order)
        folds[order[j % n_splits]].extend(groups[k])
    return [(np.array(sorted(sum((folds[j] for j in range(n_splits) if j != i), []))),
             np.array(sorted(folds[i]))) for i in range(n_splits)]


# ------------------------------------------------------------------------ data
lab = pd.read_csv("modality_labels.csv")
wide = lab.pivot_table(index="ik", columns="target", values=["p_Ki", "p_IC50"])
wide.columns = [f"{a}_{b}" for a, b in wide.columns]
wide = wide.reset_index().merge(lab[["ik", "smiles"]].drop_duplicates("ik"), on="ik")
G = [mol_graph(s) for s in wide.smiles]
keep = [i for i, g in enumerate(G) if g is not None]
wide = wide.iloc[keep].reset_index(drop=True)
G = [G[i] for i in keep]
wide["scaffold"] = [MurckoScaffold.MurckoScaffoldSmiles(s) for s in wide.smiles]
YK = np.stack([wide[f"p_Ki_{t}"].values for t in TARGETS], 1).astype(np.float32)
YI = np.stack([wide[f"p_IC50_{t}"].values for t in TARGETS], 1).astype(np.float32)
print(f"{len(wide)} molecules; atom features {ADIM}, bond features {BDIM}", flush=True)

# training rows are (compound, modality) pairs; the row's label vector is masked
ROWS = ([(i, 1.0, YK[i]) for i in range(len(wide)) if np.isfinite(YK[i]).any()]
        + [(i, 0.0, YI[i]) for i in range(len(wide)) if np.isfinite(YI[i]).any()])
print(f"{len(ROWS)} (compound, modality) training rows", flush=True)

folds = scaffold_folds_seeded(wide.scaffold.values, NFOLD, 0)
# resumable: background runs in this container get reaped, so each fold is
# checkpointed and already-finished folds are skipped on restart
import os, sys
if os.path.exists("dmpnn_oof.npz"):
    _z = np.load("dmpnn_oof.npz")
    PK, PI = _z["PK"].copy(), _z["PI"].copy()
    DONE = set(_z["done"].tolist()) if "done" in _z else set()
    print(f"resuming; folds already done: {sorted(DONE)}", flush=True)
else:
    PK = np.full((len(wide), 3), np.nan)
    PI = np.full((len(wide), 3), np.nan)
    DONE = set()
BUDGET = int(sys.argv[1]) if len(sys.argv) > 1 else NFOLD
ran = 0

for fi, (tr, te) in enumerate(folds):
    if fi in DONE:
        continue
    if ran >= BUDGET:
        print(f"budget reached; {NFOLD - len(DONE)} fold(s) still to run", flush=True)
        break
    ran += 1
    t0 = time.time()
    trs = set(tr.tolist())
    rows = [r for r in ROWS if r[0] in trs]
    mu = np.nanmean(np.stack([r[2] for r in rows]), axis=0)
    sd = np.nanstd(np.stack([r[2] for r in rows]), axis=0)
    model = DMPNN()
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=LR, total_steps=EPOCHS * (len(rows) // BATCH + 1))
    rng = np.random.default_rng(0)
    for ep in range(EPOCHS):
        model.train()
        order = rng.permutation(len(rows))
        tot = nb = 0.0
        for s in range(0, len(order), BATCH):
            idx = order[s:s + BATCH]
            gs = [G[rows[i][0]] for i in idx]
            fa, fb, b2a, rev, a2b, scope = collate(gs)
            extra = torch.tensor([[rows[i][1]] for i in idx], dtype=torch.float32)
            y = np.stack([rows[i][2] for i in idx])
            mask = torch.from_numpy(np.isfinite(y).astype(np.float32))
            yz = torch.from_numpy(np.nan_to_num((y - mu) / sd).astype(np.float32))
            pred = model(fa, fb, b2a, rev, a2b, scope, extra)
            loss = (((pred - yz) ** 2) * mask).sum() / mask.sum().clamp(min=1)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step(); sched.step()
            tot += float(loss); nb += 1
        if ep % 10 == 0 or ep == EPOCHS - 1:
            print(f"  fold {fi+1} epoch {ep+1:>3}/{EPOCHS}  loss {tot/nb:.4f}  "
                  f"[{time.time()-t0:.0f}s]", flush=True)
    model.eval()
    with torch.no_grad():
        for mod, store in ((1.0, PK), (0.0, PI)):
            for s in range(0, te.size, 100):
                idx = te[s:s + 100]
                fa, fb, b2a, rev, a2b, scope = collate([G[i] for i in idx])
                extra = torch.full((idx.size, 1), mod)
                store[idx] = (model(fa, fb, b2a, rev, a2b, scope, extra).numpy()
                              * sd + mu)
    DONE.add(fi)
    np.savez("dmpnn_oof.npz", PK=PK, PI=PI, ik=wide.ik.values.astype("U32"),
             done=np.array(sorted(DONE)))
    print(f"fold {fi+1}/{NFOLD} done in {(time.time()-t0)/60:.1f} min", flush=True)

if len(DONE) < NFOLD:
    raise SystemExit(f"{NFOLD - len(DONE)} fold(s) remaining; run again")

res = {}
print("\n" + "=" * 70)
print("D-MPNN, three heads, stacked single-modality scoring")
print("=" * 70)
for k, t in enumerate(TARGETS):
    a, b = np.isfinite(YK[:, k]), np.isfinite(YI[:, k])
    y = np.concatenate([YK[a, k], YI[b, k]])
    p = np.concatenate([PK[a, k], PI[b, k]])
    g = np.isfinite(p)
    r2 = float(stats.pearsonr(y[g], p[g])[0] ** 2)
    res[t] = dict(n=int(g.sum()), r2=r2,
                  rho=float(stats.spearmanr(y[g], p[g]).statistic),
                  RMSE=float(np.sqrt(((y[g] - p[g]) ** 2).mean())),
                  slope=float(np.polyfit(y[g], p[g], 1)[0]),
                  bias=float((p[g] - y[g]).mean()))
    print(f"  {t:<5} n={res[t]['n']:<6} r2 {r2:.3f}  rho {res[t]['rho']:+.3f}  "
          f"RMSE {res[t]['RMSE']:.2f}  slope {res[t]['slope']:.3f}  "
          f"bias {res[t]['bias']:+.3f}")
json.dump(res, open("dmpnn.json", "w"), indent=1)
print("\nwrote dmpnn.json and dmpnn_oof.npz")
