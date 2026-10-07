"""Does telling the model which assay a value came from improve it?

The ceiling analysis predicts it should, and says by how much: pooling binding and
functional values without modelling the difference caps R^2 at 0.820 / 0.633 / 0.838,
whereas the measurement-noise-only ceiling is 0.918 / 0.923 / 0.940. The predicted
headroom is +0.098 at DAT, +0.290 at NET, +0.102 at SERT. This is the test.

Four arms, all on identical seeded scaffold folds over an identical held-out universe,
so the only thing that differs is what the model is trained on and told:

  pool_bench  the published setup: pooled label, and only the compounds that have all
              three transporters measured (the benchmark's intersection requirement)
  pool_all    pooled label, every compound with that one target measured
  mod_feat    one model per target, trained on expanded rows (one per compound x
              modality) with a binary Ki/IC50 feature; predicts a modality-specific value
  sep         two independent models per target, Ki-only and IC50-only

Scored against three answer keys: the Ki median, the IC50 median, and the pooled median.
The first two are what an external set actually gives you -- external measurements come
from one assay, never from a median over both -- so they are the comparison that counts.
The pooled key is what the manuscript currently reports against.
"""
import json, os, time
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
SEED, NFOLD = 0, 5
OUT = "modality_experiment.json"


def lgbm():
    return LGBMRegressor(n_estimators=250, learning_rate=0.06, num_leaves=31,
                         min_child_samples=10, subsample=0.8, colsample_bytree=0.4,
                         random_state=0, verbose=-1, n_jobs=2)


def scaffold_folds_seeded(scaffolds, n_splits=5, seed=0):   # repeated_cv.py, verbatim
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


def scores(y, p):
    m = ~(np.isnan(y) | np.isnan(p))
    y, p = y[m], p[m]
    if y.size < 5 or np.std(y) == 0 or np.std(p) == 0:
        return None
    r = stats.pearsonr(y, p)
    return {"n": int(y.size), "r2": float(r[0] ** 2), "p": float(r[1]),
            "R2det": float(1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()),
            "slope": float(np.polyfit(y, p, 1)[0]), "bias": float((p - y).mean()),
            "rho": float(stats.spearmanr(y, p).statistic),
            "RMSE": float(np.sqrt(((y - p) ** 2).mean())),
            "var": float(y.var(ddof=1))}


# ------------------------------------------------------------------------ data
lab = pd.read_csv("modality_labels.csv")
bench = set(pd.read_csv("mat_profiles.csv").inchikey)
res = json.load(open(OUT)) if os.path.exists(OUT) else {}

for t in TARGETS:
    if t in res:
        print(f"{t}: already done, skipping")
        continue
    t0 = time.time()
    d = lab[(lab.target == t) & lab.p_pool.notna()].reset_index(drop=True)
    X, ok = featurise(d.smiles.values)
    d = d.iloc[ok].reset_index(drop=True)
    d["scaffold"] = [MurckoScaffold.MurckoScaffoldSmiles(s) if
                     Chem.MolFromSmiles(s) is not None else "" for s in d.smiles]
    folds = scaffold_folds_seeded(d.scaffold.values, NFOLD, SEED)
    in_bench = d.ik.isin(bench).values
    y_pool, y_ki, y_ic = d.p_pool.values, d.p_Ki.values, d.p_IC50.values
    print(f"\n{'='*78}\n{t}: {len(d)} compounds "
          f"({int(in_bench.sum())} in the published benchmark), "
          f"Ki {int(np.isfinite(y_ki).sum())}, IC50 {int(np.isfinite(y_ic).sum())}")

    P = {k: np.full(len(d), np.nan) for k in
         ("pool_bench", "pool_all", "mf_ki", "mf_ic", "sep_ki", "sep_ic")}
    for fi, (tr, te) in enumerate(folds):
        s = time.time()
        # --- arm 1: pooled label, benchmark intersection only
        sel = tr[in_bench[tr] & np.isfinite(y_pool[tr])]
        P["pool_bench"][te] = lgbm().fit(X[sel], y_pool[sel]).predict(X[te])
        # --- arm 2: pooled label, every compound with this target
        sel = tr[np.isfinite(y_pool[tr])]
        P["pool_all"][te] = lgbm().fit(X[sel], y_pool[sel]).predict(X[te])
        # --- arm 3: one model, expanded rows, binary modality feature
        ki, ic = tr[np.isfinite(y_ki[tr])], tr[np.isfinite(y_ic[tr])]
        Xm = np.vstack([np.hstack([X[ki], np.ones((ki.size, 1), np.float32)]),
                        np.hstack([X[ic], np.zeros((ic.size, 1), np.float32)])])
        ym = np.concatenate([y_ki[ki], y_ic[ic]])
        mf = lgbm().fit(Xm, ym)
        P["mf_ki"][te] = mf.predict(np.hstack([X[te], np.ones((te.size, 1), np.float32)]))
        P["mf_ic"][te] = mf.predict(np.hstack([X[te], np.zeros((te.size, 1), np.float32)]))
        # --- arm 4: two independent models
        P["sep_ki"][te] = lgbm().fit(X[ki], y_ki[ki]).predict(X[te])
        P["sep_ic"][te] = lgbm().fit(X[ic], y_ic[ic]).predict(X[te])
        print(f"  fold {fi+1}/{NFOLD}  {time.time()-s:.0f}s  "
              f"(train pool_bench {int(in_bench[tr].sum())}, pool_all {tr.size}, "
              f"Ki {ki.size}, IC50 {ic.size})", flush=True)

    rec = {"n_compounds": int(len(d)), "n_in_benchmark": int(in_bench.sum()),
           "n_Ki": int(np.isfinite(y_ki).sum()), "n_IC50": int(np.isfinite(y_ic).sum()),
           "arms": {}}
    KEYS = {"key_Ki": y_ki, "key_IC50": y_ic, "key_pooled": y_pool}
    ARMS = {"pool_bench": {"key_Ki": "pool_bench", "key_IC50": "pool_bench",
                           "key_pooled": "pool_bench"},
            "pool_all": {"key_Ki": "pool_all", "key_IC50": "pool_all",
                         "key_pooled": "pool_all"},
            "mod_feat": {"key_Ki": "mf_ki", "key_IC50": "mf_ic"},
            "separate": {"key_Ki": "sep_ki", "key_IC50": "sep_ic"}}
    for arm, mp in ARMS.items():
        rec["arms"][arm] = {k: scores(KEYS[k], P[src]) for k, src in mp.items()}
    res[t] = rec
    json.dump(res, open(OUT, "w"), indent=1)
    np.savez(f"modality_oof_{t}.npz", ik=d.ik.values, **P,
             y_pool=y_pool, y_ki=y_ki, y_ic=y_ic)
    print(f"{t} done in {(time.time()-t0)/60:.1f} min -> {OUT}", flush=True)

print("\nALL DONE")
