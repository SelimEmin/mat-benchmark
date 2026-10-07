"""Faithful rebuild, using the project's own code rather than the Methods prose.

Recovered from the project: src/baseline.py (featurize, LightGBM call),
src/repeated_cv.py (scaffold_folds_seeded, oof_reg), src/external_ci.py (external
protocol). Three things differ from the earlier reconstruction and all three matter:

  1. R^2 as reported is  pearsonr(y, p) ** 2  -- the SQUARED PEARSON CORRELATION,
     not 1 - SS_res/SS_tot. It is invariant to bias and slope in the predictions.
  2. Training uses every row of mat_profiles_censored_aware.csv. Censored values
     are kept at their bound; they are NOT dropped.
  3. subsample=0.8 is set in repeated_cv.oof_reg (and omitted in external_ci.py --
     an inconsistency in the original code, reproduced here as found).

Featurisation is baseline.featurize verbatim: Morgan radius 2 / 2048 bits as int8,
then MW, cLogP, TPSA, HBD, HBA, RotB, AromRings, FracCSP3, HeavyAtoms, Rings from
rdMolDescriptors. No salt stripping at this stage -- the table is already
standardised parents.
"""
import json, sys
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors as rdmd
from scipy import stats

RDLogger.DisableLog("rdApp.*")
GEN = AllChem.GetMorganGenerator(radius=2, fpSize=2048)
DESCS = [("MW", Descriptors.MolWt), ("cLogP", Descriptors.MolLogP),
         ("TPSA", Descriptors.TPSA), ("HBD", rdmd.CalcNumHBD),
         ("HBA", rdmd.CalcNumHBA), ("RotB", rdmd.CalcNumRotatableBonds),
         ("Arom", rdmd.CalcNumAromaticRings), ("FrCSP3", rdmd.CalcFractionCSP3),
         ("HeavyAtoms", lambda m: m.GetNumHeavyAtoms()),
         ("Rings", rdmd.CalcNumRings)]
TARGETS = ("pDAT", "pNET", "pSERT")
PUBLISHED_INTERNAL = {"pDAT": 0.584, "pNET": 0.501, "pSERT": 0.541}
PUBLISHED_EXTERNAL = {"pDAT": 0.459, "pNET": 0.015, "pSERT": 0.058}


def featurize(smiles):                       # baseline.featurize, verbatim
    X, keep = [], []
    for i, s in enumerate(smiles):
        m = Chem.MolFromSmiles(s)
        if m is None:
            continue
        fp = GEN.GetFingerprint(m)
        arr = np.zeros(2048, dtype=np.int8)
        DataStructs.ConvertToNumpyArray(fp, arr)
        X.append(np.concatenate([arr, [f(m) for _, f in DESCS]]))
        keep.append(i)
    return np.array(X, dtype=np.float32), np.array(keep)


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


def oof_reg(X, y, folds):                    # repeated_cv.oof_reg, verbatim
    p = np.full(len(y), np.nan)
    for tr, te in folds:
        m = LGBMRegressor(n_estimators=250, learning_rate=0.06, num_leaves=31,
                          min_child_samples=10, subsample=0.8,
                          colsample_bytree=0.4, random_state=0, verbose=-1, n_jobs=1)
        m.fit(X[tr], y[tr])
        p[te] = m.predict(X[te])
    return p


NSEED = int(sys.argv[1]) if len(sys.argv) > 1 else 20

d = pd.read_csv("mat_profiles.csv")          # == mat_profiles_censored_aware.csv
X, keep = featurize(d.smiles.tolist())
d = d.iloc[keep].reset_index(drop=True)
print(f"dataset {len(d)} compounds, {X.shape[1]} features, "
      f"{d.scaffold.nunique()} scaffolds   (no censoring filter, as in the original)")

print(f"\nrepeated scaffold CV, {NSEED} seeds -- R^2 reported as squared Pearson r")
cv = {t: [] for t in TARGETS}
oof0 = {}
for s in range(NSEED):
    folds = scaffold_folds_seeded(d.scaffold.values, 5, s)
    for t in TARGETS:
        y = d[t].values
        p = oof_reg(X, y, folds)
        cv[t].append(float(stats.pearsonr(y, p)[0] ** 2))
        if s == 0:
            oof0[t] = {"y": y.tolist(), "p": p.tolist()}
    if s % 5 == 0:
        print(f"  seed {s}: " + "  ".join(f"{t} {cv[t][-1]:.3f}" for t in TARGETS),
              flush=True)

print(f"\n{'target':8}{'rebuilt':>18}{'published':>11}{'difference':>12}")
internal = {}
for t in TARGETS:
    a = np.array(cv[t])
    internal[t] = {"mean": float(a.mean()), "sd": float(a.std(ddof=1)),
                   "published": PUBLISHED_INTERNAL[t]}
    print(f"{t:8}{a.mean():10.3f} +/- {a.std(ddof=1):.3f}"
          f"{PUBLISHED_INTERNAL[t]:11.3f}{a.mean()-PUBLISHED_INTERNAL[t]:+12.3f}")

print("\nfitting final models on all training rows (external_ci.py protocol)")
final = {}
for t in TARGETS:
    m = LGBMRegressor(n_estimators=250, learning_rate=0.06, num_leaves=31,
                      min_child_samples=10, colsample_bytree=0.4,
                      random_state=0, verbose=-1, n_jobs=1)   # no subsample, as found
    m.fit(X, d[t].values)
    final[t] = m

import pickle
pickle.dump({"models": final, "frame": d}, open("faithful_models.pkl", "wb"))
np.save("faithful_X.npy", X)
json.dump({"internal": internal, "n_seeds": NSEED,
           "r2_definition": "squared Pearson correlation, as in repeated_cv.py"},
          open("faithful_cv.json", "w"), indent=2)
json.dump(oof0, open("faithful_oof.json", "w"))
print("wrote faithful_models.pkl, faithful_X.npy, faithful_cv.json, faithful_oof.json")
