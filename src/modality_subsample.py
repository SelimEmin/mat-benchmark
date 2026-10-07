"""Control 2: train on a RANDOM subsample of the full pool, matched in size to the
benchmark training pool. Isolates 'which compounds' from 'how many'."""
import json
import numpy as np, pandas as pd
from scipy import stats
exec(open("modality_experiment.py").read().split("# ---------------")[0]
     .split('"""', 2)[2])          # reuse lgbm, folds, featurise, scores

lab = pd.read_csv("modality_labels.csv")
bench = set(pd.read_csv("mat_profiles.csv").inchikey)
rng = np.random.default_rng(11)
res = {}
for t in TARGETS:
    d = lab[(lab.target == t) & lab.p_pool.notna()].reset_index(drop=True)
    X, ok = featurise(d.smiles.values)
    d = d.iloc[ok].reset_index(drop=True)
    d["scaffold"] = [MurckoScaffold.MurckoScaffoldSmiles(s) if
                     Chem.MolFromSmiles(s) is not None else "" for s in d.smiles]
    folds = scaffold_folds_seeded(d.scaffold.values, NFOLD, SEED)
    inb = d.ik.isin(bench).values
    y = d.p_pool.values
    p = np.full(len(d), np.nan)
    for tr, te in folds:
        pool = tr[np.isfinite(y[tr])]
        k = int((inb[tr] & np.isfinite(y[tr])).sum())
        sel = rng.choice(pool, min(k, pool.size), replace=False)
        p[te] = lgbm().fit(X[sel], y[sel]).predict(X[te])
    res[t] = {}
    for key, col in (("key_Ki", "p_Ki"), ("key_IC50", "p_IC50"),
                     ("key_pooled", "p_pool")):
        yy = d[col].values
        m = np.isfinite(yy) & np.isfinite(p)
        res[t][key] = float(stats.pearsonr(yy[m], p[m])[0] ** 2)
    print(t, {k: round(v, 3) for k, v in res[t].items()}, flush=True)
json.dump(res, open("modality_subsample.json", "w"), indent=1)
