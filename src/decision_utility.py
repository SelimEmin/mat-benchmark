"""What is this model actually usable for? Decision metrics, not r^2.

r^2 answers "how much variance is explained", which is not a question anyone makes a
decision with. These are the questions that decide whether a model is worth using:

  triage      if I rank compounds by prediction and assay the top 10%, what fraction of
              the true top 10% do I capture, and how much better is that than picking at
              random?
  tolerance   what fraction of predictions land within 0.5 and within 1.0 log units?
  screening   at a potency cutoff (100 nM), can it separate actives from inactives?
  ordering    given two compounds that really differ by more than one log unit, how
              often does it put them in the right order?
  honesty     split-conformal prediction intervals -- how wide must an 80% interval be?
  gating      accuracy against coverage. A model that declines to answer for the
              compounds it cannot handle, and is accurate on the rest, is more useful
              than one that always answers. This is the curve that decides whether the
              model is deployable.
"""
import json, warnings
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from rdkit import RDLogger
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors as rdmd
from scipy import stats
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
RDLogger.DisableLog("rdApp.*")
GEN = AllChem.GetMorganGenerator(radius=2, fpSize=2048)
DESCS = [Descriptors.MolWt, Descriptors.MolLogP, Descriptors.TPSA, rdmd.CalcNumHBD,
         rdmd.CalcNumHBA, rdmd.CalcNumRotatableBonds, rdmd.CalcNumAromaticRings,
         rdmd.CalcFractionCSP3, lambda m: m.GetNumHeavyAtoms(), rdmd.CalcNumRings]
TARGETS = ("DAT", "NET", "SERT")
NSEED, NFOLD = 5, 5
ACTIVE = 7.0           # 100 nM


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


import os
lab = pd.read_csv("modality_labels.csv")
OUT = {}
if os.path.exists("decision_oof.npz"):
    z = np.load("decision_oof.npz")
    for t in TARGETS:
        OUT[t] = {k: z[f"{t}_{k}"] for k in ("y", "p", "nn")}
        print(f"{t}: {OUT[t]['y'].size} cached predictions reused", flush=True)
else:
    raise SystemExit("run the fitting stage first")


def enrichment(y, p, frac):
    k = max(1, int(round(frac * y.size)))
    top_true = set(np.argsort(-y)[:k])
    top_pred = np.argsort(-p)[:k]
    hit = len(top_true & set(top_pred)) / k
    return hit, hit / frac


def pairwise(y, p, gap, n_pairs=2_000_000, seed=0):
    """Right-ordering rate among pairs that really differ by more than `gap`.

    Pairs are sampled directly rather than enumerated: the full upper triangle for
    28,720 points is 412 million entries and materialising it exhausts memory.
    """
    rng = np.random.default_rng(seed)
    i = rng.integers(0, y.size, n_pairs)
    j = rng.integers(0, y.size, n_pairs)
    k = i != j
    i, j = i[k], j[k]
    dy, dp = y[i] - y[j], p[i] - p[j]
    m = np.abs(dy) > gap
    if m.sum() < 100:
        return float("nan"), int(m.sum())
    return float((np.sign(dy[m]) == np.sign(dp[m])).mean()), int(m.sum())


B = "=" * 94
print("\n" + B)
print("1.  TRIAGE -- rank by prediction, assay the top slice, how many real top "
      "compounds do you get?")
print(B)
print(f"{'target':<7}{'top 5%':>22}{'top 10%':>22}{'top 20%':>22}")
print(f"{'':<7}" + "".join(f"{'captured   x random':>22}" for _ in range(3)))
rep = {}
for t in TARGETS:
    y, p = OUT[t]["y"], OUT[t]["p"]
    row = f"{t:<7}"
    rep[t] = {}
    for f in (0.05, 0.10, 0.20):
        h, e = enrichment(y, p, f)
        rep[t][f"top{int(f*100)}"] = {"captured": h, "enrichment": e}
        row += f"{100*h:>13.0f}%{e:>8.1f}x"
    print(row)

print("\n" + B)
print("2.  TOLERANCE, SCREENING, ORDERING")
print(B)
print(f"{'target':<7}{'within 0.5 log':>16}{'within 1.0 log':>16}{'RMSE':>8}"
      f"{'AUC @100nM':>12}{'ordering >1 log':>17}")
for t in TARGETS:
    y, p = OUT[t]["y"], OUT[t]["p"]
    e = np.abs(y - p)
    auc = roc_auc_score((y >= ACTIVE).astype(int), p) if 0 < (y >= ACTIVE).mean() < 1 \
        else float("nan")
    pa, npair = pairwise(y, p, 1.0)
    rep[t].update(within_0_5=float((e <= 0.5).mean()), within_1_0=float((e <= 1.0).mean()),
                  RMSE=float(np.sqrt((e ** 2).mean())), auc_100nM=float(auc),
                  ordering_1log=pa, n_pairs=npair,
                  prevalence_active=float((y >= ACTIVE).mean()))
    print(f"{t:<7}{100*(e<=0.5).mean():>15.0f}%{100*(e<=1.0).mean():>15.0f}%"
          f"{np.sqrt((e**2).mean()):>8.2f}{auc:>12.3f}{100*pa:>16.0f}%")
print(f"\n  prevalence of actives (>= {ACTIVE:.0f}, i.e. 100 nM): " +
      ", ".join(f"{t} {100*rep[t]['prevalence_active']:.0f}%" for t in TARGETS))

print("\n" + B)
print("3.  HONEST INTERVALS -- split-conformal, width of an 80% prediction interval")
print(B)
print(f"{'target':<7}{'80% half-width':>17}{'90% half-width':>17}   in plain terms")
for t in TARGETS:
    e = np.abs(OUT[t]["y"] - OUT[t]["p"])
    q80, q90 = np.percentile(e, 80), np.percentile(e, 90)
    rep[t].update(conformal_80=float(q80), conformal_90=float(q90))
    print(f"{t:<7}{q80:>16.2f} {q90:>16.2f}   +-{q80:.1f} log = a factor of "
          f"{10**q80:.0f} either way")

print("\n" + B)
print("4.  GATING -- accuracy against coverage, by nearest-neighbour Tanimoto")
print(B)
for t in TARGETS:
    y, p, nn = OUT[t]["y"], OUT[t]["p"], OUT[t]["nn"]
    print(f"\n  {t}")
    print(f"    {'require NN >=':<15}{'coverage':>10}{'r2':>8}{'RMSE':>8}"
          f"{'within 1 log':>14}{'ordering':>10}")
    rep[t]["gating"] = []
    for thr in (0.0, 0.4, 0.5, 0.6, 0.7, 0.8):
        m = nn >= thr
        if m.sum() < 50:
            continue
        r2 = float(stats.pearsonr(y[m], p[m])[0] ** 2)
        pa, _ = pairwise(y[m], p[m], 1.0)
        cov = float(m.mean())
        rep[t]["gating"].append(dict(threshold=thr, coverage=cov, r2=r2,
                                     RMSE=float(np.sqrt(((y[m]-p[m])**2).mean())),
                                     within_1_0=float((np.abs(y[m]-p[m]) <= 1).mean()),
                                     ordering=pa))
        print(f"    {thr:<15.1f}{100*cov:>9.0f}%{r2:>8.3f}"
              f"{np.sqrt(((y[m]-p[m])**2).mean()):>8.2f}"
              f"{100*(np.abs(y[m]-p[m])<=1).mean():>13.0f}%{100*pa:>9.0f}%")

json.dump(rep, open("decision_utility.json", "w"), indent=1)
print("\nwrote decision_utility.json and decision_oof.npz")
