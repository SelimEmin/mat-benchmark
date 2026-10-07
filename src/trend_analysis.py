"""Are the patterns in the eight-diagnostic panel real relations, or eyeball artefacts?

Five questions, each answered with an exact identity or a test rather than a glance:

  1. Does the r^2 / R^2_det gap decompose exactly?  It should, into a spread term and
     a bias term:   r^2 - R^2_det = (r - k)^2 + (bias/S_y)^2,  k = S_p/S_y = slope/r.
  2. Is the bias column just shrinkage toward the training mean?  If predictions are
     p = m_tr + b(y - m_tr), then  bias = (1 - slope) * (m_tr - mean(y))  exactly.
  3. Does chemical similarity predict performance?  (The applicability-domain claim.)
  4. Does response-range coverage predict performance?  (The manuscript's claim.)
  5. Is slope < 1 everywhere, and is the shrinkage worse outside?
"""
import csv, json, math, pickle
import numpy as np
import pandas as pd
from collections import defaultdict
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors as rdmd
from scipy import stats

RDLogger.DisableLog("rdApp.*")
GEN = AllChem.GetMorganGenerator(radius=2, fpSize=2048)
DESCS = [Descriptors.MolWt, Descriptors.MolLogP, Descriptors.TPSA, rdmd.CalcNumHBD,
         rdmd.CalcNumHBA, rdmd.CalcNumRotatableBonds, rdmd.CalcNumAromaticRings,
         rdmd.CalcFractionCSP3, lambda m: m.GetNumHeavyAtoms(), rdmd.CalcNumRings]
TARGETS = ("DAT", "NET", "SERT")
COL = {t: "p" + t for t in TARGETS}
KEY = {"Stalberga 2026": "Stalberga 2026  (stimulants, uptake IC50)",
       "PDSP / BindingDB": "PDSP/BindingDB  (antidepressants, binding Ki)",
       "Cui et al. 2026": "Cui 2026        (SNRI series, uptake IC50)"}


def feat_ok(s):
    return Chem.MolFromSmiles(s) is not None


def parent_key(s):
    m = Chem.MolFromSmiles(s)
    if m is None:
        return None
    fr = Chem.GetMolFrags(m, asMols=True, sanitizeFrags=False)
    return Chem.MolToInchiKey(max(fr, key=lambda x: x.GetNumHeavyAtoms()) if fr else m)


store = pickle.load(open("faithful_models.pkl", "rb"))
frame = store["frame"]
train_keys = {parent_key(s) for s in frame.smiles}
TRAIN_MEAN = {t: float(frame[COL[t]].mean()) for t in TARGETS}

# ------------------------------------------------- set membership, matching competency2
st_smi = {r["abbrev"]: r["final"] for r in csv.DictReader(open("struct.csv"))}
vals = defaultdict(dict)
for r in csv.DictReader(open("stalberga_parsed.csv")):
    vals[r["compound"]][r["target"]] = (r["relation"], r["value"].strip())
SET_STAL = [{"smiles": st_smi[c], **{t: 9.0 - math.log10(float(d[t][1])) for t in TARGETS}}
            for c, d in vals.items()
            if len(d) == 3 and all(v[0] == "=" and v[1] for v in d.values())]
SET_PDSP = [{"smiles": r["smiles"], **{t: float(r["p" + t]) for t in TARGETS}}
            for r in csv.DictReader(open("pdsp_mat_profiles.csv"))]
SET_CUI = []
for r in csv.DictReader(open("ejmc2026_cui.csv")):
    SET_CUI.append({"smiles": r["smiles"],
                    **{t: (float(r["p" + t]) if r["p" + t] != "" else None)
                       for t in TARGETS}})
RAW = {"Stalberga 2026": SET_STAL, "PDSP / BindingDB": SET_PDSP, "Cui et al. 2026": SET_CUI}

SET_MEAN = {}
for name, rows in RAW.items():
    rows = [r for r in rows if parent_key(r["smiles"]) not in train_keys
            and feat_ok(r["smiles"])]
    for t in TARGETS:
        y = [r[t] for r in rows if r[t] is not None]
        SET_MEAN[(name, t)] = float(np.mean(y))

# ----------------------------------------------------------------- assemble the rows
comp = json.load(open("competency2.json"))
intern = json.load(open("internal_diagnostics.json"))
oof = json.load(open("faithful_oof.json"))

ROWS = []
for t in TARGETS:
    d = intern[t]
    y = np.array(oof[COL[t]]["y"], float)
    y = y[~np.isnan(y)]
    ROWS.append(dict(ev="Internal CV", t=t, n=d["n"], var=d["var_all_rows"],
                     varr=1.0, nn=d["nn_tanimoto_median"], rho=d["spearman"],
                     r2=d["r2_pearson"], r2d=d["R2_determination"], slope=d["slope"],
                     bias=d["bias"], ceil=d["ceiling"], pct=d["pct_ceiling"],
                     ymean=float(y.mean()), ext=False))
for name, key in KEY.items():
    for t in TARGETS:
        d = comp[key]["targets"][t]
        ROWS.append(dict(ev=name, t=t, n=d["n"], var=d["var"], varr=d["var_ratio"],
                         nn=d["nn_tanimoto_median"], rho=d["spearman"],
                         r2=d["r2_pearson"], r2d=d["R2_determination"],
                         slope=d["slope"], bias=d["bias"], ceil=d["ceiling"],
                         pct=100 * d["r2_pearson"] / d["ceiling"],
                         ymean=SET_MEAN[(name, t)], ext=True))

R = pd.DataFrame(ROWS)
E = R[R.ext].reset_index(drop=True)        # the nine external rows
line = "=" * 88

# ---------------------------------------------------------------- 1. exact gap identity
print(line)
print("1.  Does the r^2 / R^2_det gap decompose exactly?")
print(line)
print(f"{'evaluation':<17}{'t':<6}{'gap':>8}{'(r-k)^2':>10}{'(bias/Sy)^2':>13}"
      f"{'sum':>9}{'resid':>11}")
for _, w in R.iterrows():
    Sy2 = w["var"] * (w["n"] - 1) / w["n"]          # population variance of y
    r = math.sqrt(w["r2"]) * (1 if w["slope"] >= 0 else -1)
    k = w["slope"] / r                              # S_p / S_y
    spread, biasterm = (r - k) ** 2, w["bias"] ** 2 / Sy2
    gap = w["r2"] - w["r2d"]
    print(f"{w['ev']:<17}{w['t']:<6}{gap:>8.4f}{spread:>10.4f}{biasterm:>13.4f}"
          f"{spread + biasterm:>9.4f}{gap - spread - biasterm:>11.2e}")

# ------------------------------------------------------------ 2. bias = shrinkage?
print()
print(line)
print("2.  Is bias exactly shrinkage toward the training mean?")
print("    predicted bias = (1 - slope) x (training mean - set mean)")
print(line)
print(f"{'evaluation':<17}{'t':<6}{'train mu':>10}{'set mu':>9}{'1-slope':>9}"
      f"{'predicted':>11}{'observed':>10}{'resid':>9}")
pred, obs = [], []
for _, w in R.iterrows():
    p = (1 - w["slope"]) * (TRAIN_MEAN[w["t"]] - w["ymean"])
    pred.append(p); obs.append(w["bias"])
    print(f"{w['ev']:<17}{w['t']:<6}{TRAIN_MEAN[w['t']]:>10.3f}{w['ymean']:>9.3f}"
          f"{1 - w['slope']:>9.3f}{p:>11.3f}{w['bias']:>10.3f}{w['bias'] - p:>9.3f}")
pred, obs = np.array(pred), np.array(obs)
print(f"\n    Pearson(predicted, observed) = {stats.pearsonr(pred, obs)[0]:+.4f}"
      f"   (p = {stats.pearsonr(pred, obs)[1]:.2e})")
print(f"    mean |residual| = {np.abs(obs - pred).mean():.3f} log units;  "
      f"regression slope = {np.polyfit(pred, obs, 1)[0]:.3f}")

# ---------------------------------------------- 3/4. what predicts external performance
print()
print(line)
print("3/4.  What predicts external r^2?   (n = 9 external rows; Spearman)")
print(line)
for v, lab in [("nn", "nearest-neighbour Tanimoto (similarity)"),
               ("varr", "variance ratio (range coverage)"),
               ("var", "absolute variance of the test set"),
               ("n", "number of compounds"),
               ("ceil", "achievable ceiling"),
               ("ymean", "mean potency of the set")]:
    s = stats.spearmanr(E[v], E["r2"])
    p = stats.pearsonr(E[v], E["r2"])
    print(f"  r^2  vs {lab:<40} rho {s.statistic:+.3f} (p {s.pvalue:.3f})"
          f"   r {p[0]:+.3f}")
print()
for v, lab in [("nn", "nearest-neighbour Tanimoto"), ("varr", "variance ratio")]:
    s = stats.spearmanr(E[v], E["rho"])
    print(f"  rho  vs {lab:<40} rho {s.statistic:+.3f} (p {s.pvalue:.3f})")

print("\n  Similarity is near-constant within a set, so it has only three effective")
print("  levels. The decisive comparison is PDSP (most similar chemistry) against Cui")
print("  (least similar):")
for t in TARGETS:
    a = E[(E.ev == "PDSP / BindingDB") & (E.t == t)].iloc[0]
    b = E[(E.ev == "Cui et al. 2026") & (E.t == t)].iloc[0]
    print(f"    {t:<5} PDSP  Tanimoto {a['nn']:.2f}  r^2 {a['r2']:.3f}  rho {a['rho']:+.3f}"
          f"   |   Cui  Tanimoto {b['nn']:.2f}  r^2 {b['r2']:.3f}  rho {b['rho']:+.3f}")

# --------------------------------------------------------------- 5. shrinkage everywhere
print()
print(line)
print("5.  Slope, and whether the shrinkage is worse outside")
print(line)
print(f"  slope < 1 in {int((R.slope < 1).sum())}/{len(R)} rows   "
      f"(sign test p = {2 ** -len(R):.2e})")
print(f"  internal slope  {R[~R.ext].slope.min():.3f}-{R[~R.ext].slope.max():.3f}"
      f"  (median {R[~R.ext].slope.median():.3f})")
print(f"  external slope  {E.slope.min():.3f}-{E.slope.max():.3f}"
      f"  (median {E.slope.median():.3f})")
u = stats.mannwhitneyu(R[~R.ext].slope, E.slope, alternative="greater")
print(f"  internal > external, Mann-Whitney U p = {u.pvalue:.4f}")
print(f"  slope vs variance ratio across all 12 rows: "
      f"rho {stats.spearmanr(R.varr, R.slope).statistic:+.3f} "
      f"(p {stats.spearmanr(R.varr, R.slope).pvalue:.3f})")
print(f"  |bias| internal {np.abs(R[~R.ext].bias).max():.3f} max;  "
      f"external {np.abs(E.bias).min():.3f}-{np.abs(E.bias).max():.3f}")

# ------------------------------------------------------------------ 6. rho vs r^2
print()
print(line)
print("6.  Rank agreement against linear agreement")
print(line)
print(f"  Spearman rho vs sqrt(r^2) across all 12 rows: "
      f"r = {stats.pearsonr(R.rho, np.sqrt(R.r2))[0]:+.4f}")
print(f"  rho exceeds |Pearson r| in {int((R.rho > np.sqrt(R.r2)).sum())}/{len(R)} rows")
print("  Does normalising by the ceiling collapse the between-set spread at NET?")
for t in TARGETS:
    sub = E[E.t == t]
    print(f"    {t:<5} r^2 {list(np.round(sub.r2, 3))}  ->  "
          f"% of ceiling {list(np.round(sub.pct, 1))}"
          f"   (CV {np.std(sub.r2)/np.mean(sub.r2):.2f} -> "
          f"{np.std(sub.pct)/np.mean(sub.pct):.2f})")

R.to_csv("trend_table.csv", index=False)
print("\nwrote trend_table.csv")
