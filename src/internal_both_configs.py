"""The eight diagnostics for BOTH internal cross-validations, measured identically.

The published internal CV and the production internal CV have to be put on one axis
with the external sets, so they are computed the same way here: 20 seeds x 5 scaffold
folds, every diagnostic re-measured on each seed, mean and SD reported. (The earlier
internal row mixed 20-seed means for r^2 with seed-0 values for the rest; that is
avoided here.)

SCORING. Both configurations are scored against single-modality answer keys, stacked:
each held-out compound contributes its Ki value if it has one and its IC50 value if it
has one. That is the production task -- predict a value from a named assay -- and it is
what an external set actually hands you. The published model has one prediction per
compound and uses it for both; the production model uses the matching head.

NEAREST-NEIGHBOUR TANIMOTO is computed fold-wise at seed 0 only, each held-out compound
against the four folds it was not trained on. Against the whole table every compound
returns 1.0 and the number means nothing. It is seed-independent in distribution, so
one seed suffices.
"""
import json, time
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors as rdmd
from rdkit.Chem.Scaffolds import MurckoScaffold
from scipy import stats

RDLogger.DisableLog("rdApp.*")
GEN = AllChem.GetMorganGenerator(radius=2, fpSize=2048)
DESCS = [Descriptors.MolWt, Descriptors.MolLogP, Descriptors.TPSA, rdmd.CalcNumHBD,
         rdmd.CalcNumHBA, rdmd.CalcNumRotatableBonds, rdmd.CalcNumAromaticRings,
         rdmd.CalcFractionCSP3, lambda m: m.GetNumHeavyAtoms(), rdmd.CalcNumRings]
TARGETS = ("DAT", "NET", "SERT")
NSEED, NFOLD = 20, 5
# ceiling C (modality unknown) for the published config, ceiling B (modality known)
# for the production config -- they perform different tasks, see CEILING_DERIVATION
CEIL = {"published": {"DAT": 0.801, "NET": 0.759, "SERT": 0.760},
        "production": {"DAT": 0.800, "NET": 0.871, "SERT": 0.875}}


def lgbm():
    return LGBMRegressor(n_estimators=250, learning_rate=0.06, num_leaves=31,
                         min_child_samples=10, subsample=0.8, colsample_bytree=0.4,
                         random_state=0, verbose=-1, n_jobs=2)


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


def featurise(smiles):
    X, ok = [], []
    for i, s in enumerate(smiles):
        m = Chem.MolFromSmiles(s)
        if m is None:
            continue
        fp = GEN.GetFingerprint(m)
        a = np.zeros(2048, dtype=np.int8)
        DataStructs.ConvertToNumpyArray(fp, a)
        X.append(np.concatenate([a, [f(m) for f in DESCS]]))
        ok.append(i)
    return np.array(X, dtype=np.float32), np.array(ok)


def stack(yk, yi, pk, pi):
    """Each compound contributes whichever single-modality values it has."""
    a = np.isfinite(yk) & np.isfinite(pk)
    b = np.isfinite(yi) & np.isfinite(pi)
    return np.concatenate([yk[a], yi[b]]), np.concatenate([pk[a], pi[b]])


def diag(y, p, ceil):
    r = stats.pearsonr(y, p)
    r2 = float(r[0] ** 2)
    return dict(n=int(y.size), var=float(y.var(ddof=1)), r2=r2,
                R2det=float(1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()),
                slope=float(np.polyfit(y, p, 1)[0]), bias=float((p - y).mean()),
                rho=float(stats.spearmanr(y, p).statistic),
                RMSE=float(np.sqrt(((y - p) ** 2).mean())),
                pct_ceiling=float(100 * r2 / ceil))


lab = pd.read_csv("modality_labels.csv")
bench = set(pd.read_csv("mat_profiles.csv").inchikey)
out = {}

for t in TARGETS:
    t0 = time.time()
    d = lab[(lab.target == t) & lab.p_pool.notna()].reset_index(drop=True)
    X, ok = featurise(d.smiles.values)
    d = d.iloc[ok].reset_index(drop=True)
    d["scaffold"] = [MurckoScaffold.MurckoScaffoldSmiles(s) if
                     Chem.MolFromSmiles(s) is not None else "" for s in d.smiles]
    fps = [GEN.GetFingerprint(Chem.MolFromSmiles(s)) for s in d.smiles]
    inb = d.ik.isin(bench).values
    yp, yk, yi = d.p_pool.values, d.p_Ki.values, d.p_IC50.values

    acc = {"published": [], "production": []}
    nnP = np.full(len(d), np.nan)      # production training pool, seed 0
    nnB = np.full(len(d), np.nan)      # published training pool, seed 0
    for seed in range(NSEED):
        folds = scaffold_folds_seeded(d.scaffold.values, NFOLD, seed)
        pub = np.full(len(d), np.nan)
        pk = np.full(len(d), np.nan)
        pi = np.full(len(d), np.nan)
        for tr, te in folds:
            selB = tr[inb[tr] & np.isfinite(yp[tr])]
            pub[te] = lgbm().fit(X[selB], yp[selB]).predict(X[te])
            ki, ic = tr[np.isfinite(yk[tr])], tr[np.isfinite(yi[tr])]
            Xm = np.vstack([np.hstack([X[ki], np.ones((ki.size, 1), np.float32)]),
                            np.hstack([X[ic], np.zeros((ic.size, 1), np.float32)])])
            mf = lgbm().fit(Xm, np.concatenate([yk[ki], yi[ic]]))
            pk[te] = mf.predict(np.hstack([X[te], np.ones((te.size, 1), np.float32)]))
            pi[te] = mf.predict(np.hstack([X[te], np.zeros((te.size, 1), np.float32)]))
            if seed == 0:
                selA = tr[np.isfinite(yp[tr])]
                fB = [fps[i] for i in selB]
                fA = [fps[i] for i in selA]
                for i in te:
                    nnB[i] = max(DataStructs.BulkTanimotoSimilarity(fps[i], fB))
                    nnP[i] = max(DataStructs.BulkTanimotoSimilarity(fps[i], fA))
        ys, ps = stack(yk, yi, pub, pub)
        acc["published"].append(diag(ys, ps, CEIL["published"][t]))
        ys, ps = stack(yk, yi, pk, pi)
        acc["production"].append(diag(ys, ps, CEIL["production"][t]))
    for cfg in ("published", "production"):
        rec = {k: {"mean": float(np.mean([a[k] for a in acc[cfg]])),
                   "sd": float(np.std([a[k] for a in acc[cfg]], ddof=1))}
               for k in acc[cfg][0] if k != "n"}
        rec["n"] = acc[cfg][0]["n"]
        rec["var_ratio"] = 1.0
        rec["nn_tanimoto_median"] = float(np.median(nnP if cfg == "production" else nnB))
        rec["nn_pct_above_0_7"] = float(100 * np.mean(
            (nnP if cfg == "production" else nnB) > 0.7))
        rec["ceiling"] = CEIL[cfg][t]
        rec["n_train_pool"] = int(inb.sum()) if cfg == "published" else int(len(d))
        out.setdefault(cfg, {})[t] = rec
    print(f"{t}: {(time.time()-t0)/60:.1f} min   "
          f"published r2 {out['published'][t]['r2']['mean']:.3f}   "
          f"production r2 {out['production'][t]['r2']['mean']:.3f}", flush=True)
    json.dump(out, open("internal_both_configs.json", "w"), indent=1)

print("\nwrote internal_both_configs.json")
for cfg in ("published", "production"):
    print(f"\n{cfg}")
    for t in TARGETS:
        r = out[cfg][t]
        print(f"  {t:<5} n={r['n']:<6} r2 {r['r2']['mean']:.3f}+-{r['r2']['sd']:.3f}  "
              f"R2det {r['R2det']['mean']:+.3f}  slope {r['slope']['mean']:.3f}  "
              f"bias {r['bias']['mean']:+.3f}  rho {r['rho']['mean']:+.3f}  "
              f"NN {r['nn_tanimoto_median']:.2f}  {r['pct_ceiling']['mean']:.0f}%")
