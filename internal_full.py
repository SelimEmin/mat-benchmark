"""20-seed scaffold CV on the pre-registered model, collecting every internal number
the paper needs from ONE pass: r2, RMSE, per-compound out-of-fold predictions, and
nearest-neighbour Tanimoto to the training folds.

Featurisation, fold generation and the LightGBM call are rebuild_faithful.py verbatim,
which reproduces the manuscript's Table 2 to three decimals.

Resumable: each seed is checkpointed, so an interrupted run continues where it stopped.
"""
import json, os, pickle, time
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors as rdmd
from scipy import stats

RDLogger.DisableLog("rdApp.*")
GEN = AllChem.GetMorganGenerator(radius=2, fpSize=2048)
DESCS = [Descriptors.MolWt, Descriptors.MolLogP, Descriptors.TPSA, rdmd.CalcNumHBD,
         rdmd.CalcNumHBA, rdmd.CalcNumRotatableBonds, rdmd.CalcNumAromaticRings,
         rdmd.CalcFractionCSP3, lambda m: m.GetNumHeavyAtoms(), rdmd.CalcNumRings]
TARGETS = ("pDAT", "pNET", "pSERT")
NSEED = 20
CKPT = "internal_full.npz"

store = pickle.load(open("faithful_models.pkl", "rb"))
d = store["frame"].reset_index(drop=True)
X = np.load("faithful_X.npy")
print(f"{len(d)} compounds, {X.shape[1]} features, "
      f"{d.scaffold.nunique()} scaffolds", flush=True)
FPS = [GEN.GetFingerprint(Chem.MolFromSmiles(s)) for s in d.smiles]


def folds_for(seed, n_splits=5):
    rng = np.random.default_rng(seed)
    groups = {}
    for i, s in enumerate(pd.Series(d.scaffold.values).fillna("").values):
        groups.setdefault(s, []).append(i)
    keys = list(groups); rng.shuffle(keys)
    out = [[] for _ in range(n_splits)]
    for j, k in enumerate(sorted(keys, key=lambda k: -len(groups[k]))):
        out[int(np.argmin([len(o) for o in out]))].extend(groups[k])
    return [np.array(sorted(o)) for o in out]


def oof(y, folds):
    p = np.zeros(y.size)
    for te in folds:
        tr = np.setdiff1d(np.arange(y.size), te)
        m = LGBMRegressor(n_estimators=250, learning_rate=0.06, num_leaves=31,
                          min_child_samples=10, subsample=0.8, colsample_bytree=0.4,
                          random_state=0, verbose=-1, n_jobs=2)
        m.fit(X[tr], y[tr]); p[te] = m.predict(X[te])
    return p


def nn_for(folds):
    """Max Tanimoto from each compound to the compounds in its TRAINING folds."""
    nn = np.zeros(len(d))
    for te in folds:
        tr = np.setdiff1d(np.arange(len(d)), te)
        trfps = [FPS[i] for i in tr]
        for i in te:
            nn[i] = max(DataStructs.BulkTanimotoSimilarity(FPS[i], trfps))
    return nn


R2 = {t: [] for t in TARGETS}; RM = {t: [] for t in TARGETS}
P0, NN0, done = {}, None, []
if os.path.exists(CKPT):
    z = np.load(CKPT, allow_pickle=True)
    R2 = z["R2"].item(); RM = z["RM"].item()
    P0 = z["P0"].item(); NN0 = z["NN0"]; done = list(z["done"])
    print(f"resuming, seeds done: {done}", flush=True)

for s in range(NSEED):
    if s in done:
        continue
    t0 = time.time()
    f = folds_for(s)
    for t in TARGETS:
        y = d[t].values
        p = oof(y, f)
        R2[t].append(float(stats.pearsonr(y, p)[0] ** 2))
        RM[t].append(float(np.sqrt(((y - p) ** 2).mean())))
        if s == 0:
            P0[t] = p
    if s == 0:
        NN0 = nn_for(f)
    done.append(s)
    np.savez(CKPT, R2=R2, RM=RM, P0=P0, NN0=NN0, done=np.array(done))
    print(f"  seed {s:2d}  " + "  ".join(f"{t} {R2[t][-1]:.3f}" for t in TARGETS)
          + f"   ({time.time()-t0:.0f}s)", flush=True)

print(f"\n{'target':8}{'r2':>18}{'RMSE':>18}{'manuscript r2':>16}")
PUB = {"pDAT": 0.584, "pNET": 0.501, "pSERT": 0.541}
res = {}
for t in TARGETS:
    a, b = np.array(R2[t]), np.array(RM[t])
    res[t] = dict(r2=float(a.mean()), r2_sd=float(a.std(ddof=1)),
                  rmse=float(b.mean()), rmse_sd=float(b.std(ddof=1)))
    print(f"{t:8}{a.mean():10.3f} ± {a.std(ddof=1):.3f}"
          f"{b.mean():12.3f} ± {b.std(ddof=1):.3f}{PUB[t]:16.3f}")
json.dump({"per_target": res, "n_seeds": len(done), "n_compounds": int(len(d)),
           "r2_definition": "squared Pearson correlation"},
          open("internal_full.json", "w"), indent=1)
np.savez("internal_oof.npz",
         **{f"{t}_y": d[t].values for t in TARGETS},
         **{f"{t}_p": P0[t] for t in TARGETS}, nn=NN0)
print("\nwrote internal_full.json, internal_oof.npz")
