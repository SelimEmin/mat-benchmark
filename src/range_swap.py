"""Is the external failure caused by the training/test potency ranges differing?

Three experiments, each isolating a different part of the claim. All use the per-target
production universe with external compounds removed at InChIKey-skeleton level.

  EXP 1 -- the narrow-window internal test (the decisive control)
      Take the ordinary ChEMBL-trained model and score it only on held-out ChEMBL
      compounds that fall INSIDE the external set's potency window. Same model, same
      data source, same chemistry -- only the response range is made to match. If
      internal performance collapses to external levels, the range is the explanation.

  EXP 2 -- range-matched training (train narrow, test external)
      Train only on ChEMBL compounds inside the external set's window, then test on the
      external set. Two controls are essential:
        full      the ordinary full-range training set (the baseline)
        sizematch a RANDOM training subset of the same size, full range -- because
                  restricting by potency also shrinks the training set, and a smaller
                  training set is worse for reasons that have nothing to do with range.

  EXP 3 -- reverse transfer (train external, test ChEMBL)
      Train on the external set itself and predict ChEMBL compounds inside the matching
      window, against a ChEMBL-trained model on those same compounds. If the two
      datasets are mutually unpredictable, the problem is the relationship between them
      rather than a defect in either.

TWO THINGS TO KEEP IN MIND WHEN READING THE OUTPUT.

  r^2 falls mechanically when the test set's spread falls, even for a perfect model:
  max r^2 = 1 - sigma_e^2/var(y). So a low r^2 on a narrow window is partly arithmetic,
  not a modelling failure. RMSE does not have that property, so both are reported and
  RMSE is the one to compare across windows.

  Selecting TRAINING compounds by their potency is selection on the outcome. It
  restricts the range the model ever sees and can hurt learning for reasons unrelated
  to matching. That is exactly what the size-matched control is there to separate.
"""
import csv, json, math, warnings
import numpy as np
import pandas as pd
from collections import defaultdict
from lightgbm import LGBMRegressor
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors as rdmd
from rdkit.Chem.Scaffolds import MurckoScaffold
from scipy import stats

warnings.filterwarnings("ignore")
RDLogger.DisableLog("rdApp.*")
GEN = AllChem.GetMorganGenerator(radius=2, fpSize=2048)
DESCS = [Descriptors.MolWt, Descriptors.MolLogP, Descriptors.TPSA, rdmd.CalcNumHBD,
         rdmd.CalcNumHBA, rdmd.CalcNumRotatableBonds, rdmd.CalcNumAromaticRings,
         rdmd.CalcFractionCSP3, lambda m: m.GetNumHeavyAtoms(), rdmd.CalcNumRings]
TARGETS = ("DAT", "NET", "SERT")
NFOLD, NSEED = 5, 3
rng = np.random.default_rng(5)


def lgbm():
    return LGBMRegressor(n_estimators=250, learning_rate=0.06, num_leaves=31,
                         min_child_samples=10, subsample=0.8, colsample_bytree=0.4,
                         random_state=0, verbose=-1, n_jobs=2)


def folds_seeded(sc, n=5, seed=0):
    r = np.random.default_rng(seed)
    g = {}
    for i, s in enumerate(pd.Series(sc).fillna("").values):
        g.setdefault(s, []).append(i)
    k = list(g)
    r.shuffle(k)
    k.sort(key=lambda x: -len(g[x]))
    f = [[] for _ in range(n)]
    o = list(range(n))
    for j, kk in enumerate(k):
        if j % n == 0:
            r.shuffle(o)
        f[o[j % n]].extend(g[kk])
    return [(np.array(sorted(sum((f[j] for j in range(n) if j != i), []))),
             np.array(sorted(f[i]))) for i in range(n)]


def featurise(smiles):
    mols = [Chem.MolFromSmiles(s) for s in smiles]
    keep = [i for i, m in enumerate(mols) if m is not None]
    mols = [mols[i] for i in keep]
    A = np.zeros((len(mols), 2048), np.float32)
    for i, m in enumerate(mols):
        DataStructs.ConvertToNumpyArray(GEN.GetFingerprint(m), A[i])
    D = np.array([[f(m) for f in DESCS] for m in mols], np.float32)
    return keep, np.hstack([A, D]).astype(np.float32), mols


def skel(s):
    m = Chem.MolFromSmiles(s)
    if m is None:
        return None
    fr = Chem.GetMolFrags(m, asMols=True, sanitizeFrags=False)
    return Chem.MolToInchiKey(
        max(fr, key=lambda x: x.GetNumHeavyAtoms()) if fr else m).split("-")[0]


def score(y, p):
    if y.size < 5 or np.std(p) < 1e-9 or np.std(y) < 1e-9:
        return dict(n=int(y.size), r2=float("nan"), RMSE=float("nan"),
                    rho=float("nan"), var=float(y.var(ddof=1)) if y.size > 1 else 0.0)
    return dict(n=int(y.size), r2=float(stats.pearsonr(y, p)[0] ** 2),
                RMSE=float(np.sqrt(((y - p) ** 2).mean())),
                rho=float(stats.spearmanr(y, p).statistic),
                var=float(y.var(ddof=1)))


# ------------------------------------------------------------------ external sets
st = {r["abbrev"]: r["final"] for r in csv.DictReader(open("struct.csv"))}
v = defaultdict(dict)
for r in csv.DictReader(open("stalberga_parsed.csv")):
    v[r["compound"]][r["target"]] = (r["relation"], r["value"].strip())
STAL = [{"smiles": st[c], **{t: 9.0 - math.log10(float(d[t][1])) for t in TARGETS}}
        for c, d in v.items()
        if len(d) == 3 and all(x[0] == "=" and x[1] for x in d.values())]
PDSP = [{"smiles": r["smiles"], **{t: float(r["p" + t]) for t in TARGETS}}
        for r in csv.DictReader(open("pdsp_mat_profiles.csv"))]
CUI = [{"smiles": r["smiles"],
        **{t: (float(r["p" + t]) if r["p" + t] != "" else None) for t in TARGETS}}
       for r in csv.DictReader(open("ejmc2026_cui.csv"))]
SETS = [("Stalberga 2026", STAL), ("PDSP / BindingDB", PDSP), ("Cui et al. 2026", CUI)]
ext_skel = {skel(r["smiles"]) for _, rows in SETS for r in rows} - {None}

lab = pd.read_csv("modality_labels.csv")
report = {}

for t in TARGETS:
    d = lab[(lab.target == t) & lab.p_pool.notna()].reset_index(drop=True)
    keep, X, mols = featurise(d.smiles.values)
    d = d.iloc[keep].reset_index(drop=True)
    inside = ~d.ik.str.split("-").str[0].isin(ext_skel).values
    d, X = d[inside].reset_index(drop=True), X[inside]
    d["scaffold"] = [MurckoScaffold.MurckoScaffoldSmiles(mol=m)
                     for m, k in zip(mols, inside) if k]
    y = d.p_pool.values
    print(f"\n{t}: {len(d)} ChEMBL compounds, potency {y.min():.1f}-{y.max():.1f}, "
          f"variance {y.var(ddof=1):.2f}", flush=True)

    # out-of-fold predictions from the ordinary model, for EXP 1 and EXP 3
    oof = np.full(len(d), np.nan)
    for tr, te in folds_seeded(d.scaffold.values, NFOLD, 0):
        oof[te] = lgbm().fit(X[tr], y[tr]).predict(X[te])

    for name, rows in SETS:
        sel = [r for r in rows if r.get(t) is not None
               and Chem.MolFromSmiles(r["smiles"]) is not None]
        ye = np.array([r[t] for r in sel])
        _, Xe, _ = featurise([r["smiles"] for r in sel])
        lo, hi = ye.min(), ye.max()
        win = (y >= lo) & (y <= hi)
        rec = {"window": [float(lo), float(hi)], "n_external": int(ye.size),
               "var_external": float(ye.var(ddof=1)),
               "n_chembl_in_window": int(win.sum()),
               "var_chembl_in_window": float(y[win].var(ddof=1))}

        # ---- EXP 1: ordinary model, scored only inside the window (held out)
        rec["exp1_internal_full"] = score(y, oof)
        rec["exp1_internal_window"] = score(y[win], oof[win])

        # ---- EXP 2: train narrow / full / size-matched, test external
        full = np.arange(len(d))
        narrow = np.where(win)[0]
        sizem = rng.choice(full, min(narrow.size, full.size), replace=False)
        for arm, idx in (("full", full), ("narrow", narrow), ("sizematch", sizem)):
            if idx.size < 100:
                continue
            m = lgbm().fit(X[idx], y[idx])
            rec["exp2_" + arm] = score(ye, m.predict(Xe))
            rec["exp2_" + arm]["n_train"] = int(idx.size)

        # ---- EXP 3: train on the external set, predict ChEMBL inside the window
        if ye.size >= 15 and win.sum() >= 50:
            me = lgbm().fit(Xe, ye)
            rec["exp3_external_trained"] = score(y[win], me.predict(X[win]))
            rec["exp3_chembl_trained"] = score(y[win], oof[win])
        report.setdefault(t, {})[name] = rec
        print(f"  {name:<18} window {lo:.1f}-{hi:.1f}  "
              f"ChEMBL in window {int(win.sum()):>4}", flush=True)
    json.dump(report, open("range_swap.json", "w"), indent=2)

B = "=" * 100
print("\n" + B)
print("EXP 1 -- the same model, scored on ChEMBL held-out compounds inside the "
      "external set's potency window")
print(B)
print(f"{'set':<18}{'t':<6}{'window':>12}{'n':>7}{'var':>7}{'r2':>8}{'RMSE':>7}"
      f"   |{'external r2':>12}{'ext RMSE':>10}{'ext var':>9}")
for t in TARGETS:
    for s in report[t]:
        r = report[t][s]
        a, e = r["exp1_internal_window"], r["exp2_full"]
        print(f"{s:<18}{t:<6}{r['window'][0]:.1f}–{r['window'][1]:<7.1f}"
              f"{a['n']:>7}{a['var']:>7.2f}{a['r2']:>8.3f}{a['RMSE']:>7.2f}   |"
              f"{e['r2']:>12.3f}{e['RMSE']:>10.2f}{e['var']:>9.2f}")
print("\n  full-range internal, for reference: " + ", ".join(
    f"{t} r2 {report[t][list(report[t])[0]]['exp1_internal_full']['r2']:.3f} "
    f"RMSE {report[t][list(report[t])[0]]['exp1_internal_full']['RMSE']:.2f}"
    for t in TARGETS))

print("\n" + B)
print("EXP 2 -- training restricted to the external set's potency window, with the "
      "size-matched control")
print(B)
print(f"{'set':<18}{'t':<6}{'full train':>22}{'narrow train':>24}{'size-matched':>24}")
print(f"{'':<18}{'':<6}{'n      r2    RMSE':>22}{'n      r2    RMSE':>24}"
      f"{'n      r2    RMSE':>24}")
for t in TARGETS:
    for s in report[t]:
        r = report[t][s]
        row = f"{s:<18}{t:<6}"
        for arm in ("full", "narrow", "sizematch"):
            k = "exp2_" + arm
            if k in r:
                row += f"{r[k]['n_train']:>8}{r[k]['r2']:>8.3f}{r[k]['RMSE']:>8.2f}"
            else:
                row += f"{'--':>24}"
        print(row)

print("\n" + B)
print("EXP 3 -- reverse transfer: trained on the external set, predicting ChEMBL "
      "inside the matching window")
print(B)
print(f"{'set':<18}{'t':<6}{'n train':>9}{'n test':>8}{'ext-trained r2':>16}"
      f"{'RMSE':>7}   |{'ChEMBL-trained r2':>19}{'RMSE':>7}")
for t in TARGETS:
    for s in report[t]:
        r = report[t][s]
        if "exp3_external_trained" not in r:
            continue
        a, b = r["exp3_external_trained"], r["exp3_chembl_trained"]
        print(f"{s:<18}{t:<6}{r['n_external']:>9}{a['n']:>8}{a['r2']:>16.3f}"
              f"{a['RMSE']:>7.2f}   |{b['r2']:>19.3f}{b['RMSE']:>7.2f}")

json.dump(report, open("range_swap.json", "w"), indent=2)
print("\nwrote range_swap.json")
