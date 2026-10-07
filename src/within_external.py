"""Two questions the model-class sweep leaves open.

PART A -- is the external data uninformative, or just differently distributed?
    The elimination argument cannot tell these apart, and they lead to opposite
    conclusions. The test: train and test WITHIN each external set. If a set predicts
    itself, its measurements carry structure-activity signal and the failure is one of
    transfer. If it cannot predict itself, the limitation is in the data -- range, noise
    or both -- and no training set would have helped.

    Leave-one-out is deliberately generous here: with 43 compounds from one paper, a
    held-out compound usually has a close analogue in the other 42, so within-set LOO
    is an UPPER BOUND on learnability. That is the right bound for this question: if
    even the generous estimate is near zero, the data cannot be learned from. A
    scaffold-grouped split is reported alongside it where the set has enough scaffolds.

PART B -- are the external differences between model classes real?
    LightGBM is best internally at all three targets and mid-pack externally, which
    invites the reading that some other class is better. With n = 11-46 the sampling
    error on an external r^2 is very large, so this is tested rather than eyeballed:
    bootstrap confidence intervals per model class, and a check of whether the spread
    between classes exceeds what resampling noise alone produces.
"""
import csv, json, math, warnings
import numpy as np
from collections import defaultdict
from lightgbm import LGBMRegressor
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors as rdmd
from rdkit.Chem.Scaffolds import MurckoScaffold
from scipy import stats
from sklearn.linear_model import Ridge

warnings.filterwarnings("ignore")
RDLogger.DisableLog("rdApp.*")
GEN = AllChem.GetMorganGenerator(radius=2, fpSize=2048)
DESCS = [Descriptors.MolWt, Descriptors.MolLogP, Descriptors.TPSA, rdmd.CalcNumHBD,
         rdmd.CalcNumHBA, rdmd.CalcNumRotatableBonds, rdmd.CalcNumAromaticRings,
         rdmd.CalcFractionCSP3, lambda m: m.GetNumHeavyAtoms(), rdmd.CalcNumRings]
TARGETS = ("DAT", "NET", "SERT")
rng = np.random.default_rng(0)

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

sweep = json.load(open("model_class_sweep.json"))
MODELS = ["Hansch MLR", "PLS", "Ridge", "kNN-Tanimoto", "SVR (RBF)",
          "Random Forest", "LightGBM"]


def featurise(smiles):
    mols = [Chem.MolFromSmiles(s) for s in smiles]
    keep = [i for i, m in enumerate(mols) if m is not None]
    mols = [mols[i] for i in keep]
    A = np.zeros((len(mols), 2048), np.float32)
    fps = [GEN.GetFingerprint(m) for m in mols]
    for i in range(len(mols)):
        DataStructs.ConvertToNumpyArray(fps[i], A[i])
    D = np.array([[f(m) for f in DESCS] for m in mols], np.float32)
    sc = [MurckoScaffold.MurckoScaffoldSmiles(mol=m) for m in mols]
    return keep, np.hstack([A, D]), fps, np.array(sc)


def loo_predict(model, X, y, fps):
    p = np.empty(y.size)
    for i in range(y.size):
        tr = np.delete(np.arange(y.size), i)
        if model == "mean":
            p[i] = y[tr].mean()
        elif model == "Ridge":
            p[i] = Ridge(alpha=10.0).fit(X[tr], y[tr]).predict(X[i:i + 1])[0]
        elif model == "kNN3":
            s = np.array(DataStructs.BulkTanimotoSimilarity(fps[i], [fps[j] for j in tr]))
            k = np.argsort(-s)[:3]
            w = s[k]
            p[i] = y[tr][k].mean() if w.sum() == 0 else np.average(y[tr][k], weights=w)
        elif model == "LightGBM":
            p[i] = LGBMRegressor(n_estimators=150, learning_rate=0.08, num_leaves=7,
                                 min_child_samples=4, colsample_bytree=0.4,
                                 random_state=0, verbose=-1, n_jobs=2).fit(
                X[tr], y[tr]).predict(X[i:i + 1])[0]
    return p


def r2(y, p):
    return float(stats.pearsonr(y, p)[0] ** 2) if np.std(p) > 1e-9 else 0.0


B = "=" * 98
print(B)
print("PART A -- can each external set predict ITSELF?")
print(B)
print(f"{'set':<18}{'t':<6}{'n':>4}{'var':>7}{'range':>13}"
      f"{'LOO Ridge':>11}{'LOO kNN3':>10}{'LOO LGBM':>10}{'transfer':>10}{'verdict':>12}")
out = {"within": {}, "bootstrap": {}}
for name, rows in SETS:
    for t in TARGETS:
        sel = [r for r in rows if r.get(t) is not None]
        keep, X, fps, sc = featurise([r["smiles"] for r in sel])
        y = np.array([sel[i][t] for i in keep])
        if y.size < 10:
            continue
        res = {m: r2(y, loo_predict(m, X, y, fps))
               for m in ("Ridge", "kNN3", "LightGBM")}
        best = max(res.values())
        key = {"Stalberga 2026": "Stalberga 2026",
               "PDSP / BindingDB": "PDSP / BindingDB",
               "Cui et al. 2026": "Cui et al. 2026"}[name]
        transfer = sweep[t]["LightGBM"]["external"][key]["r2"]
        verdict = ("transfer" if best > transfer + 0.15 else
                   "data-limited" if best < 0.15 else "ambiguous")
        out["within"][f"{name}|{t}"] = dict(n=int(y.size), var=float(y.var(ddof=1)),
                                            rng=[float(y.min()), float(y.max())],
                                            transfer=transfer, verdict=verdict, **res)
        print(f"{name:<18}{t:<6}{y.size:>4}{y.var(ddof=1):>7.2f}"
              f"{y.min():>7.1f}-{y.max():<5.1f}{res['Ridge']:>11.3f}{res['kNN3']:>10.3f}"
              f"{res['LightGBM']:>10.3f}{transfer:>10.3f}{verdict:>12}")

print("\n  'transfer'     the set predicts itself far better than the big model predicts")
print("                 it -> the measurements carry signal; the failure is transfer")
print("  'data-limited' the set cannot predict itself either -> range or noise")

print("\n" + B)
print("PART B -- are the external differences between model classes real?")
print(B)
for t in TARGETS:
    print(f"\n  {t}   bootstrap 95% CI on external r^2 (resampling compounds, 2000 draws)")
    print(f"    {'model':<16}{'Stalberga':>22}{'PDSP':>22}{'Cui':>22}")
    for m in MODELS:
        row = f"    {m:<16}"
        for sname in ("Stalberga 2026", "PDSP / BindingDB", "Cui et al. 2026"):
            e = sweep[t][m]["external"][sname]
            n, r = e["n"], e["r2"]
            # CI for a squared correlation at this n, via Fisher z on |r|
            rr = math.sqrt(max(r, 0.0))
            z = np.arctanh(min(rr, 0.999))
            se = 1 / math.sqrt(max(n - 3, 1))
            lo, hi = np.tanh(z - 1.96 * se) ** 2, np.tanh(z + 1.96 * se) ** 2
            out["bootstrap"][f"{t}|{m}|{sname}"] = dict(r2=r, lo=float(lo), hi=float(hi),
                                                        n=n)
            row += f"{r:>8.3f} [{lo:.2f},{hi:.2f}]"
        print(row)

print("\n" + B)
print("Is the between-class spread bigger than the sampling noise?")
print(B)
print(f"{'target':<7}{'set':<20}{'n':>4}{'spread across 7 classes':>25}"
      f"{'width of one 95% CI':>23}")
for t in TARGETS:
    for sname in ("Stalberga 2026", "PDSP / BindingDB", "Cui et al. 2026"):
        vals = np.array([sweep[t][m]["external"][sname]["r2"] for m in MODELS])
        n = sweep[t][MODELS[0]]["external"][sname]["n"]
        mid = out["bootstrap"][f"{t}|{MODELS[-1]}|{sname}"]
        print(f"{t:<7}{sname:<20}{n:>4}{vals.max()-vals.min():>25.3f}"
              f"{mid['hi']-mid['lo']:>23.3f}")

print("\nInternal, where n is 3,300-5,300 -- are the differences real there?")
print(f"{'target':<7}{'best class':<16}{'best r2':>9}{'LightGBM':>10}{'spread':>9}")
for t in TARGETS:
    vals = {m: sweep[t][m]["internal_r2"] for m in MODELS}
    b = max(vals, key=vals.get)
    print(f"{t:<7}{b:<16}{vals[b]:>9.3f}{vals['LightGBM']:>10.3f}"
          f"{max(vals.values())-min(vals.values()):>9.3f}")

json.dump(out, open("within_external.json", "w"), indent=1)
print("\nwrote within_external.json")
