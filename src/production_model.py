"""The production configuration, and a like-for-like comparison against the published one.

Two changes from the published model, both tested in modality_experiment.py:

  1. The all-three-transporter intersection requirement is dropped for TRAINING. The
     benchmark keeps only compounds measured at all three (2,849); per target ChEMBL
     has 3,367 / 3,702 / 5,338. The size-matched control showed the loss is in WHICH
     compounds the filter removes, not how many, so this is not simply a sample-size
     effect. The intersection is still required for the selectivity analyses, which
     genuinely need a three-target profile.

  2. The assay modality is kept. Training rows are expanded to one per compound x
     modality, carrying a binary Ki/IC50 flag, and the modality is specified at
     prediction time. This was worth little internally and was the only intervention
     that improved external performance significantly.

Both configurations are run over the SAME 20 seeds x 5 scaffold folds so the comparison
is paired. Scored against single-modality answer keys, because that is what an external
measurement is -- a value from one assay, never a median over both.
"""
import json, time
import numpy as np
import pandas as pd
import pickle
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


def r2(y, p):
    m = ~(np.isnan(y) | np.isnan(p))
    return float(stats.pearsonr(y[m], p[m])[0] ** 2) if m.sum() > 5 else np.nan


lab = pd.read_csv("modality_labels.csv")
bench = set(pd.read_csv("mat_profiles.csv").inchikey)
out, models = {}, {}

for t in TARGETS:
    t0 = time.time()
    d = lab[(lab.target == t) & lab.p_pool.notna()].reset_index(drop=True)
    X, ok = featurise(d.smiles.values)
    d = d.iloc[ok].reset_index(drop=True)
    d["scaffold"] = [MurckoScaffold.MurckoScaffoldSmiles(s) if
                     Chem.MolFromSmiles(s) is not None else "" for s in d.smiles]
    inb = d.ik.isin(bench).values
    yp, yk, yi = d.p_pool.values, d.p_Ki.values, d.p_IC50.values
    print(f"\n{t}: {len(d)} compounds, {int(inb.sum())} in benchmark, "
          f"Ki {int(np.isfinite(yk).sum())}, IC50 {int(np.isfinite(yi).sum())}",
          flush=True)

    acc = {k: [] for k in ("pub_Ki", "pub_IC50", "pub_pool",
                           "prod_Ki", "prod_IC50",
                           "pubB_Ki", "pubB_IC50", "prodB_Ki", "prodB_IC50")}
    for seed in range(NSEED):
        folds = scaffold_folds_seeded(d.scaffold.values, NFOLD, seed)
        pub = np.full(len(d), np.nan)
        pk = np.full(len(d), np.nan)
        pi = np.full(len(d), np.nan)
        for tr, te in folds:
            sel = tr[inb[tr] & np.isfinite(yp[tr])]
            pub[te] = lgbm().fit(X[sel], yp[sel]).predict(X[te])
            ki, ic = tr[np.isfinite(yk[tr])], tr[np.isfinite(yi[tr])]
            Xm = np.vstack([np.hstack([X[ki], np.ones((ki.size, 1), np.float32)]),
                            np.hstack([X[ic], np.zeros((ic.size, 1), np.float32)])])
            mf = lgbm().fit(Xm, np.concatenate([yk[ki], yi[ic]]))
            pk[te] = mf.predict(np.hstack([X[te], np.ones((te.size, 1), np.float32)]))
            pi[te] = mf.predict(np.hstack([X[te], np.zeros((te.size, 1), np.float32)]))
        acc["pub_Ki"].append(r2(yk, pub));     acc["pub_IC50"].append(r2(yi, pub))
        acc["pub_pool"].append(r2(yp, pub))
        acc["prod_Ki"].append(r2(yk, pk));     acc["prod_IC50"].append(r2(yi, pi))
        # restricted to the published benchmark compounds (the conservative comparison)
        acc["pubB_Ki"].append(r2(yk[inb], pub[inb]))
        acc["pubB_IC50"].append(r2(yi[inb], pub[inb]))
        acc["prodB_Ki"].append(r2(yk[inb], pk[inb]))
        acc["prodB_IC50"].append(r2(yi[inb], pi[inb]))
        if seed % 5 == 0:
            print(f"  seed {seed:>2}  pub {acc['pub_Ki'][-1]:.3f}/"
                  f"{acc['pub_IC50'][-1]:.3f}   prod {acc['prod_Ki'][-1]:.3f}/"
                  f"{acc['prod_IC50'][-1]:.3f}   [{time.time()-t0:.0f}s]", flush=True)

    out[t] = {"n": int(len(d)), "n_benchmark": int(inb.sum()),
              "n_Ki": int(np.isfinite(yk).sum()), "n_IC50": int(np.isfinite(yi).sum()),
              **{k: {"mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1))}
                 for k, v in acc.items()},
              "paired_delta": {
                  m: {"mean": float(np.mean(np.array(acc["prod_" + m])
                                            - np.array(acc["pub_" + m]))),
                      "sd": float(np.std(np.array(acc["prod_" + m])
                                         - np.array(acc["pub_" + m], ), ddof=1)),
                      "wins": int((np.array(acc["prod_" + m])
                                   > np.array(acc["pub_" + m])).sum()),
                      "t_p": float(stats.ttest_rel(acc["prod_" + m],
                                                   acc["pub_" + m]).pvalue)}
                  for m in ("Ki", "IC50")}}

    # final production model, fitted on everything
    ki, ic = np.isfinite(yk), np.isfinite(yi)
    Xm = np.vstack([np.hstack([X[ki], np.ones((ki.sum(), 1), np.float32)]),
                    np.hstack([X[ic], np.zeros((ic.sum(), 1), np.float32)])])
    models[t] = lgbm().fit(Xm, np.concatenate([yk[ki], yi[ic]]))
    print(f"{t} done in {(time.time()-t0)/60:.1f} min", flush=True)
    json.dump(out, open("production_cv.json", "w"), indent=1)

pickle.dump({"models": models, "labels": lab},
            open("production_models.pkl", "wb"))
print("\nwrote production_cv.json and production_models.pkl")
for t in TARGETS:
    o = out[t]
    print(f"\n{t}  (n={o['n']}, {o['n_benchmark']} in benchmark)")
    for m in ("Ki", "IC50"):
        dd = o["paired_delta"][m]
        print(f"  key_{m:<5} published {o['pub_'+m]['mean']:.3f}+-{o['pub_'+m]['sd']:.3f}"
              f"   production {o['prod_'+m]['mean']:.3f}+-{o['prod_'+m]['sd']:.3f}"
              f"   delta {dd['mean']:+.3f}  {dd['wins']}/{NSEED} seeds  p={dd['t_p']:.1e}")
