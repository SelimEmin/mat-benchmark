"""Is the internal-to-external gap explained by the response range?

TWO EXPERIMENTS, because the obvious version of the question is aimed at the wrong
side of the problem.

PART A -- the test side, which is where r^2's denominator lives.
    r^2 = 1 - SS_res/SS_tot, and SS_tot is the variance of the TEST set. A test set
    whose values barely vary gives r^2 nothing to explain, however good the
    predictions are. So the question "is the range causing the gap?" is answered by
    building INTERNAL test sets that match each external set and seeing whether
    internal performance falls to external levels. Four nested matches, each adding
    one constraint:
        1. size only
        2. + variance
        3. + mean potency
        4. + nearest-neighbour similarity to training
    If the external r^2 lands inside the distribution at step 2 or 3, the range
    explains it. Whatever gap survives step 4 is genuinely external.

PART B -- the training side, which is the experiment as literally proposed:
    restrict TRAINING to compounds inside the external set's potency range, retrain,
    and re-evaluate. Prediction: it does not help and probably hurts, because the
    test set's variance is untouched and the training set is now smaller. Run it
    anyway -- a negative result here is the cleanest way to show the mechanism is on
    the test side, and it comes with the same size-matched random control used in the
    distance experiment, so "narrower range" is separated from "less data".
"""
import csv, json, math, warnings
import numpy as np
import pandas as pd
from collections import defaultdict
from lightgbm import LGBMRegressor
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors as rdmd
from scipy import stats

warnings.filterwarnings("ignore")
RDLogger.DisableLog("rdApp.*")
GEN = AllChem.GetMorganGenerator(radius=2, fpSize=2048)
DESCS = [Descriptors.MolWt, Descriptors.MolLogP, Descriptors.TPSA, rdmd.CalcNumHBD,
         rdmd.CalcNumHBA, rdmd.CalcNumRotatableBonds, rdmd.CalcNumAromaticRings,
         rdmd.CalcFractionCSP3, lambda m: m.GetNumHeavyAtoms(), rdmd.CalcNumRings]
TARGETS = ("DAT", "NET", "SERT")
rng = np.random.default_rng(20261002)

comp = json.load(open("competency2.json"))
CK = {"Stalberga": "Stalberga 2026  (stimulants, uptake IC50)",
      "PDSP": "PDSP/BindingDB  (antidepressants, binding Ki)",
      "Cui": "Cui 2026        (SNRI series, uptake IC50)"}
z = np.load("decision_oof.npz")
OOF = {t: {k: z[f"{t}_{k}"] for k in ("y", "p", "nn")} for t in TARGETS}

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
SETS = {"Stalberga": STAL, "PDSP": PDSP, "Cui": CUI}


def skel(s):
    m = Chem.MolFromSmiles(s)
    if m is None:
        return None
    fr = Chem.GetMolFrags(m, asMols=True, sanitizeFrags=False)
    return Chem.MolToInchiKey(
        max(fr, key=lambda x: x.GetNumHeavyAtoms()) if fr else m).split("-")[0]


def feat(s):
    m = Chem.MolFromSmiles(s)
    if m is None:
        return None
    fp = GEN.GetFingerprint(m)
    a = np.zeros(2048, dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, a)
    return np.concatenate([a, [f(m) for f in DESCS]]).astype(np.float32)


def lgbm():
    return LGBMRegressor(n_estimators=250, learning_rate=0.06, num_leaves=31,
                         min_child_samples=10, subsample=0.8, colsample_bytree=0.4,
                         random_state=0, verbose=-1, n_jobs=2)



MEAN = {}
for nm, rows in SETS.items():
    for t in TARGETS:
        ys = [r[t] for r in rows if r.get(t) is not None]
        MEAN[(nm, t)] = float(np.mean(ys))

# =============================================================== PART A
def matched_null(t, n_ext, want, draws=40000, keep_min=200):
    """r^2 distribution over internal subsets matching `want` (a dict of constraints)."""
    y, p, nn = OOF[t]["y"], OOF[t]["p"], OOF[t]["nn"]
    N = y.size
    tol = dict(var=0.15, mean=0.20, nn=0.06)
    for _ in range(6):
        out = []
        for _ in range(draws // 20):
            s = rng.choice(N, n_ext, replace=False)
            ys = y[s]
            if "var" in want and abs(ys.var(ddof=1) - want["var"]) > tol["var"] * want["var"]:
                continue
            if "mean" in want and abs(ys.mean() - want["mean"]) > tol["mean"]:
                continue
            if "nn" in want and abs(np.median(nn[s]) - want["nn"]) > tol["nn"]:
                continue
            if np.std(ys) == 0 or np.std(p[s]) == 0:
                continue
            out.append(stats.pearsonr(ys, p[s])[0] ** 2)
        if len(out) >= keep_min:
            break
        tol = {k: v * 1.6 for k, v in tol.items()}          # widen and retry
    a = np.array(out)
    if a.size < 20:
        return None
    return dict(n=int(a.size), median=float(np.median(a)),
                q05=float(np.percentile(a, 5)), q95=float(np.percentile(a, 95)),
                tol_var=tol["var"])


B = "=" * 104
print(B)
print("PART A -- internal test sets matched to each external set, one constraint at a time")
print(B)
print(f"{'set':<11}{'t':<6}{'ext r2':>8}   {'size only':>20}{'+ variance':>20}"
      f"{'+ mean':>20}{'+ similarity':>20}")
rowsA = []
for nm, k in CK.items():
    for t in TARGETS:
        d = comp[k]["targets"][t]
        n_ext, r_ext = d["n"], d["r2_pearson"]
        steps = [("size only", {}),
                 ("+ variance", {"var": d["var"]}),
                 ("+ mean", {"var": d["var"], "mean": MEAN[(nm, t)]}),
                 ("+ similarity", {"var": d["var"], "mean": MEAN[(nm, t)],
                                   "nn": d["nn_tanimoto_median"]})]
        cells, verdicts = [], []
        for _, want in steps:
            res = matched_null(t, n_ext, want)
            cells.append(res)
            verdicts.append(None if res is None else
                            (res["q05"] <= r_ext <= res["q95"]))
        txt = ""
        for res, v in zip(cells, verdicts):
            txt += ("        n/a         " if res is None else
                    f"{res['median']:>7.3f} [{res['q05']:.2f},{res['q95']:.2f}]"
                    + (" IN " if v else " OUT"))
        print(f"{nm:<11}{t:<6}{r_ext:>8.3f}   {txt}")
        rowsA.append(dict(set=nm, target=t, n=n_ext, ext_r2=r_ext,
                          **{f"step{i+1}": (None if c is None else c)
                             for i, c in enumerate(cells)}))

print("\n  IN  = the external r2 falls inside the 5-95% band of matched internal subsets,")
print("        i.e. it is what an internal test set of that shape would have scored.")
print("  OUT = a gap survives that the match does not explain.")

# =============================================================== PART B
print("\n" + B)
print("PART B -- restricting the TRAINING set to the external set's potency range")
print(B)

ext_skel = {skel(r["smiles"]) for rows in SETS.values() for r in rows} - {None}
lab = pd.read_csv("modality_labels.csv")
print(f"{'set':<11}{'t':<6}{'ext range':>13}{'train full':>12}{'train in-range':>15}"
      f"{'r2 full':>9}{'r2 range-restricted':>21}{'r2 size-matched random':>24}")
rowsB = []
for t in TARGETS:
    d = lab[(lab.target == t) & lab.p_pool.notna()].reset_index(drop=True)
    d = d[~d.ik.str.split("-").str[0].isin(ext_skel)].reset_index(drop=True)
    F = [feat(s) for s in d.smiles]
    keep = [i for i, f in enumerate(F) if f is not None]
    d, X = d.iloc[keep].reset_index(drop=True), np.vstack([F[i] for i in keep])
    y = d.p_pool.values
    full = lgbm().fit(X, y)
    for nm, rows in SETS.items():
        sel = [r for r in rows if r.get(t) is not None and feat(r["smiles"]) is not None]
        if len(sel) < 5:
            continue
        ye = np.array([r[t] for r in sel])
        Xe = np.vstack([feat(r["smiles"]) for r in sel])
        lo, hi = ye.min(), ye.max()
        m = (y >= lo) & (y <= hi)
        if m.sum() < 150:
            continue
        restricted = lgbm().fit(X[m], y[m])
        idx = rng.choice(len(y), int(m.sum()), replace=False)        # size-matched
        sized = lgbm().fit(X[idx], y[idx])

        def r2(mod):
            p = mod.predict(Xe)
            return float(stats.pearsonr(ye, p)[0] ** 2) if np.std(p) > 0 else 0.0
        a, b, c = r2(full), r2(restricted), r2(sized)
        rowsB.append(dict(set=nm, target=t, lo=float(lo), hi=float(hi),
                          n_full=int(len(y)), n_range=int(m.sum()),
                          r2_full=a, r2_range=b, r2_sizematched=c))
        print(f"{nm:<11}{t:<6}{lo:>6.1f}-{hi:<6.1f}{len(y):>12}{int(m.sum()):>15}"
              f"{a:>9.3f}{b:>21.3f}{c:>24.3f}")

mean_d = np.mean([r["r2_range"] - r["r2_full"] for r in rowsB])
wins = sum(r["r2_range"] > r["r2_full"] for r in rowsB)
print(f"\n  range-restricted training vs full: mean change {mean_d:+.3f}, "
      f"better in {wins}/{len(rowsB)} cells")
mean_s = np.mean([r["r2_sizematched"] - r["r2_full"] for r in rowsB])
print(f"  size-matched random control:       mean change {mean_s:+.3f}")
print(f"  -> the two controls differ by {mean_d - mean_s:+.3f}, which is the part "
      f"attributable to the RANGE restriction rather than to the smaller training set")

json.dump({"partA": rowsA, "partB": rowsB}, open("range_experiment.json", "w"),
          indent=1, default=float)
print("\nwrote range_experiment.json")
