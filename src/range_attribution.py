"""Can the low external r^2 be PREDICTED from the range change alone?

The experiments so far show range matters. They do not quantify how much of the
observed external r^2 it accounts for, because "restrict the test set and see" still
leaves everything else free to vary. These two methods instead PREDICT each external
r^2 from the internal result plus the external set's spread, with no free parameters,
and treat the residual as the part range cannot explain.

METHOD A -- the classical range-restriction correction (Pearson 1903; Thorndike 1949
Case II; reviewed in Sackett & Yang 2000). A century-old psychometric standard for
exactly this question: how much of a correlation's drop is attributable to a restricted
sample. With u = SD(external) / SD(training) and R the unrestricted correlation,

    r_restricted^2  =  u^2 R^2 / (1 - R^2 + u^2 R^2)

Nothing is fitted. It assumes linearity, homoscedastic errors and selection acting on
the response alone -- so where it fails, the failure localises what else differs.

METHOD B -- a parametric simulation that drops the normality assumption. Fit the
internal relationship between prediction and truth (intercept a, slope b, residual SD
s) on out-of-fold predictions. Then, for each external set, simulate predictions for
its ACTUAL measured potencies:

    p_i = a + b * y_i + N(0, s)

This is a model that is exactly as accurate, as shrunken and as biased as the internal
model, applied to the external set's own response distribution. Everything except the
range is held at its internal value. Running it 4,000 times gives the r^2 distribution
range alone would produce. If the observed external r^2 falls inside that distribution,
range is a sufficient explanation; if it falls below, the shortfall is the residual.

Method B is the stronger of the two because it preserves the model's shrinkage and
uses the empirical potency distribution rather than assuming a normal one.
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
NDRAW = 4000
rng = np.random.default_rng(11)


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
out = {}
for t in TARGETS:
    d = lab[(lab.target == t) & lab.p_pool.notna()].reset_index(drop=True)
    keep, X, mols = featurise(d.smiles.values)
    d = d.iloc[keep].reset_index(drop=True)
    inside = ~d.ik.str.split("-").str[0].isin(ext_skel).values
    d, X = d[inside].reset_index(drop=True), X[inside]
    d["scaffold"] = [MurckoScaffold.MurckoScaffoldSmiles(mol=m)
                     for m, k in zip(mols, inside) if k]
    y = d.p_pool.values
    oof = np.full(len(d), np.nan)
    for tr, te in folds_seeded(d.scaffold.values, 5, 0):
        oof[te] = lgbm().fit(X[tr], y[tr]).predict(X[te])

    b, a = np.polyfit(y, oof, 1)                 # internal p = a + b*y
    resid = oof - (a + b * y)
    s_int = float(resid.std(ddof=2))
    R2 = float(stats.pearsonr(y, oof)[0] ** 2)
    sd_train = float(y.std(ddof=1))
    print(f"\n{t}: internal R² {R2:.3f}, slope {b:.3f}, intercept {a:.3f}, "
          f"residual SD {s_int:.3f}, SD(y) {sd_train:.3f}", flush=True)

    model = lgbm().fit(X, y)
    for name, rows in SETS:
        sel = [r for r in rows if r.get(t) is not None
               and Chem.MolFromSmiles(r["smiles"]) is not None]
        ye = np.array([r[t] for r in sel])
        _, Xe, _ = featurise([r["smiles"] for r in sel])
        pe = model.predict(Xe)
        obs = float(stats.pearsonr(ye, pe)[0] ** 2)
        u = float(ye.std(ddof=1) / sd_train)

        # --- METHOD A: classical range-restriction prediction
        predA = (u ** 2 * R2) / (1 - R2 + u ** 2 * R2)

        # --- METHOD B: simulate the internal (p | y) relationship on these potencies
        sims = np.empty(NDRAW)
        for i in range(NDRAW):
            ps = a + b * ye + rng.normal(0, s_int, ye.size)
            sims[i] = stats.pearsonr(ye, ps)[0] ** 2 if np.std(ps) > 0 else np.nan
        sims = sims[np.isfinite(sims)]
        lo, hi, med = np.percentile(sims, [2.5, 97.5]), None, float(np.median(sims))
        q05, q95 = float(np.percentile(sims, 2.5)), float(np.percentile(sims, 97.5))
        pval = float((sims <= obs).mean())

        out.setdefault(t, {})[name] = dict(
            n=int(ye.size), u=u, var_ext=float(ye.var(ddof=1)), internal_R2=R2,
            observed=obs, predicted_rangeonly=float(predA),
            sim_median=med, sim_lo=q05, sim_hi=q95,
            inside=bool(q05 <= obs <= q95), p_below=pval,
            residual_A=float(obs - predA), residual_B=float(obs - med))
        print(f"  {name:<18} u={u:.2f}  observed {obs:.3f}   "
              f"A predicts {predA:.3f}   B predicts {med:.3f} [{q05:.3f}, {q95:.3f}]"
              f"  {'INSIDE' if q05 <= obs <= q95 else 'BELOW'}", flush=True)

json.dump(out, open("range_attribution.json", "w"), indent=2)

B = "=" * 104
print("\n" + B)
print("Predicting each external r² from the internal result and the external spread, "
      "with no fitted parameters")
print(B)
print(f"{'set':<18}{'t':<6}{'n':>5}{'u':>6}{'observed':>10}"
      f"{'A: range correction':>21}{'B: simulation (95% band)':>28}{'verdict':>10}")
rows = []
for t in TARGETS:
    for s in out[t]:
        r = out[t][s]
        rows.append(r)
        print(f"{s:<18}{t:<6}{r['n']:>5}{r['u']:>6.2f}{r['observed']:>10.3f}"
              f"{r['predicted_rangeonly']:>21.3f}"
              f"{r['sim_median']:>14.3f} [{r['sim_lo']:.3f}, {r['sim_hi']:.3f}]"
              f"{'inside' if r['inside'] else 'below':>10}")

obs = np.array([r["observed"] for r in rows])
pA = np.array([r["predicted_rangeonly"] for r in rows])
pB = np.array([r["sim_median"] for r in rows])
inside = sum(r["inside"] for r in rows)
print(f"\n  Cells where the observed r² falls inside the range-only simulation band: "
      f"{inside}/9")
print(f"  Mean observed {obs.mean():.3f}   mean predicted by range alone "
      f"(A) {pA.mean():.3f}, (B) {pB.mean():.3f}")
print(f"  Mean shortfall below the range-only prediction: A {(obs - pA).mean():+.3f}, "
      f"B {(obs - pB).mean():+.3f}")
rA = stats.pearsonr(obs, pA)
print(f"  Correlation between observed and range-predicted r²: r = {rA[0]:+.3f} "
      f"(p = {rA[1]:.4f}), so range alone orders the nine cells "
      f"{'correctly' if rA[0] > 0.5 else 'poorly'}")
sl, ic, rr, pp, se = stats.linregress(pA, obs)
print(f"  Regression observed = {ic:+.3f} + {sl:.3f} x predicted   "
      f"(perfect attribution would be intercept 0, slope 1)")
print(f"  Variance in observed r² explained by the range prediction: {rr**2:.3f}")
print("\n  A shortfall means the model does WORSE externally than its own internal")
print("  accuracy on a test set of that spread would predict. That gap is the part")
print("  range cannot account for.")
print("\nwrote range_attribution.json")
