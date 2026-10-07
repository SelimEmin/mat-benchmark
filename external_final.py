"""Single, consistent recomputation of every external-set number the manuscript reports.

One convention throughout, the one the Methods now state:

  * structures desalted to the largest fragment;
  * CXSMILES annotation blocks (" |...|") stripped before parsing, so a coordinate or
    bond-geometry annotation does not silently delete a compound;
  * cocaine taken as (-)-cocaine, not its hydrolysis metabolite;
  * training overlap removed on the InChIKey SKELETON BLOCK, because ECFP4 is computed
    without chirality and a stereoisomer of a training compound has an identical
    fingerprint.

The published tables mixed conventions: Figure 13 already used the skeleton block,
while Tables 6 and 14 and the section-3.9 prose used the full InChIKey and the wrong
cocaine. This script replaces all of them from one source.

Writes external_final.json and prints the table rows.
"""
import csv, json, math, pickle
import numpy as np
from collections import defaultdict
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors as rdmd
from scipy import stats

RDLogger.DisableLog("rdApp.*")
GEN = AllChem.GetMorganGenerator(radius=2, fpSize=2048)
DESCS = [Descriptors.MolWt, Descriptors.MolLogP, Descriptors.TPSA, rdmd.CalcNumHBD,
         rdmd.CalcNumHBA, rdmd.CalcNumRotatableBonds, rdmd.CalcNumAromaticRings,
         rdmd.CalcFractionCSP3, lambda m: m.GetNumHeavyAtoms(), rdmd.CalcNumRings]
T = ("DAT", "NET", "SERT")
COL = {"DAT": "pDAT", "NET": "pNET", "SERT": "pSERT"}
FLOOR = 0.194
SIG_E = {"DAT": 0.445, "NET": 0.655, "SERT": 0.492}
COCAINE = "COC(=O)[C@H]1[C@@H](OC(=O)c2ccccc2)C[C@@H]2CC[C@H]1N2C"
TRIALS = 20000
rng = np.random.default_rng(20261003)


def mol(s):
    """Parse, stripping any CXSMILES annotation block, then desalt to largest fragment."""
    m = Chem.MolFromSmiles(s)
    if m is None and " |" in s:
        m = Chem.MolFromSmiles(s.split(" |")[0])
    if m is None:
        return None
    fr = Chem.GetMolFrags(m, asMols=True, sanitizeFrags=False)
    return max(fr, key=lambda x: x.GetNumHeavyAtoms()) if len(fr) > 1 else m


def skel(s):
    m = mol(s)
    return Chem.MolToInchiKey(m).split("-")[0] if m else None


def feat(s):
    m = mol(s)
    if m is None:
        return None
    fp = GEN.GetFingerprint(m)
    a = np.zeros(2048, dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, a)
    return np.concatenate([a, [f(m) for f in DESCS]]).astype(np.float32)


store = pickle.load(open("faithful_models.pkl", "rb"))
models, frame = store["models"], store["frame"]
TR_SKEL = {k.split("-")[0] for k in frame.inchikey}
train_fps = [GEN.GetFingerprint(mol(s)) for s in frame.smiles]
train_mean = {t: float(frame[COL[t]].mean()) for t in T}
# the variance reference is the UNCENSORED training distribution, matching Table 14
train_var = {t: float(frame[COL[t]].values[frame["cens_" + t].values == 0].var(ddof=1))
             for t in T}

# --------------------------------------------------------------------- the sets
st = {r["abbrev"]: r["final"] for r in csv.DictReader(open("struct.csv"))}
st["Cocaine"] = COCAINE
v = defaultdict(dict)
for r in csv.DictReader(open("stalberga_parsed.csv")):
    v[r["compound"]][r["target"]] = (r["relation"], r["value"].strip())
STAL = [{"name": c, "smiles": st[c],
         **{t: 9.0 - math.log10(float(d[t][1])) for t in T}}
        for c, d in v.items()
        if len(d) == 3 and all(x[0] == "=" and x[1] for x in d.values())]

PDSP = [{"name": "", "smiles": r["smiles"], **{t: float(r["p" + t]) for t in T}}
        for r in csv.DictReader(open("pdsp_mat_profiles.csv"))]

CUI = []
for r in csv.DictReader(open("ejmc2026_cui.csv")):
    CUI.append({"name": "", "smiles": r["smiles"],
                **{t: (float(r["p" + t]) if r["p" + t] != "" else None) for t in T}})

SETS = [("Stalberga", STAL), ("PDSP", PDSP), ("Cui", CUI)]

# --------------------------------------------------------- internal out-of-fold pool
oof = json.load(open("faithful_oof.json"))
iz = np.load("internal_oof.npz")


def range_matched_null(t, target_var, n_ext, draws=2000):
    y = np.array(oof[COL[t]]["y"]); p = np.array(oof[COL[t]]["p"])
    centre, out = y.mean(), []
    for _ in range(draws):
        w, idx = 1.0, np.array([], int)
        for _ in range(80):
            idx = np.where((y >= centre - w / 2) & (y <= centre + w / 2))[0]
            if idx.size >= n_ext and np.var(y[idx], ddof=1) >= target_var:
                break
            w += 0.1
        if idx.size < n_ext:
            continue
        s = rng.choice(idx, n_ext, replace=False)
        ys, ps = y[s], p[s]
        if np.std(ys) == 0 or np.std(ps) == 0:
            continue
        out.append(stats.pearsonr(ys, ps)[0] ** 2)
    a = np.array(out)
    if a.size == 0:
        return {"n": 0, "median": None, "q05": None, "q95": None}
    return {"n": int(a.size), "median": float(np.median(a)),
            "q05": float(np.percentile(a, 5)), "q95": float(np.percentile(a, 95))}


def attainable(y, nn, n):
    av = np.empty(TRIALS); as_ = np.empty(TRIALS)
    for i in range(TRIALS):
        k = rng.choice(y.size, n, replace=False)
        av[i] = y[k].var(ddof=1); as_[i] = np.median(nn[k])
    return av, as_


# --------------------------------------------------------------------- the battery
report = {}
for name, rows in SETS:
    raw = len(rows)
    parsed = [r for r in rows if feat(r["smiles"]) is not None]
    dropped = [r for r in rows if feat(r["smiles"]) is None]
    overlap = [r for r in parsed if skel(r["smiles"]) in TR_SKEL]
    keepset = [r for r in parsed if skel(r["smiles"]) not in TR_SKEL]
    rec = {"n_raw": raw, "n_unparseable": len(dropped), "n_overlap": len(overlap),
           "n": len(keepset),
           "overlap_names": sorted(r["name"] for r in overlap if r["name"]),
           "targets": {}}
    for t in T:
        keep = [r for r in keepset if r[t] is not None]
        if len(keep) < 5:
            continue
        y = np.array([r[t] for r in keep])
        X = np.vstack([feat(r["smiles"]) for r in keep])
        p = models[COL[t]].predict(X)
        r_pear, pval = stats.pearsonr(y, p)
        r2p = float(r_pear ** 2)
        var = float(y.var(ddof=1))
        ceil = 1 - SIG_E[t] ** 2 / var
        err = np.abs(y - p)
        sims = np.array([max(DataStructs.BulkTanimotoSimilarity(
            GEN.GetFingerprint(mol(r["smiles"])), train_fps)) for r in keep])
        i, j = np.triu_indices(y.size, 1)
        dy, dp = y[i] - y[j], p[i] - p[j]
        msk = np.abs(dy) > 0.5
        pa = float((np.sign(dy[msk]) == np.sign(dp[msk])).mean()) if msk.sum() else float("nan")
        # attainability of a composition-matched internal comparator
        iy, inn = iz[f"{COL[t]}_y"], iz["nn"]
        av, as_ = attainable(iy, inn, len(keep))
        pv = float((av < var).mean() * 100)
        pn = float((as_ < float(np.median(sims))).mean() * 100)
        rec["targets"][t] = {
            "n": len(keep), "var": var, "var_ratio": var / train_var[t],
            "r2_pearson": r2p,
            "R2_determination": float(1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()),
            "pearson_p": float(pval),
            "spearman": float(stats.spearmanr(y, p).statistic),
            "RMSE": float(np.sqrt(((y - p) ** 2).mean())),
            "bias": float((p - y).mean()), "slope": float(np.polyfit(y, p, 1)[0]),
            "ceiling": float(ceil),
            "pct_of_ceiling": float(100 * r2p / ceil) if ceil > 0 else None,
            "pct_within_floor": float((err <= FLOOR).mean() * 100),
            "pct_within_0_5": float((err <= 0.5).mean() * 100),
            "pairwise_acc_gap0.5": pa, "n_pairs": int(msk.sum()),
            "baseline_mean_RMSE": float(np.sqrt(((y - train_mean[t]) ** 2).mean())),
            "nn_tanimoto_median": float(np.median(sims)),
            "n_fp_identical": int((sims >= 0.9999).sum()),
            "var_percentile": pv, "nn_percentile": pn,
            "attainable": bool(1 <= pv <= 99 and 1 <= pn <= 99),
            "range_matched_null": range_matched_null(t, var, len(keep))}
    report[name] = rec

json.dump(report, open("external_final.json", "w"), indent=2)

B = "=" * 102
print(B); print("SET COMPOSITION (one convention: desalt, strip CXSMILES, skeleton-block overlap)")
print(B)
for name, _ in SETS:
    r = report[name]
    print(f"  {name:<12} raw {r['n_raw']:>3}   unparseable {r['n_unparseable']}   "
          f"skeleton overlaps {r['n_overlap']:>2}   -> n = {r['n']}")
    if r["overlap_names"]:
        print(f"               overlaps: {', '.join(r['overlap_names'])}")

print()
print(B); print("TABLE 14 ROWS  (set, target, n, var, r2, ceiling, % of ceiling)")
print(B)
for name, _ in SETS:
    for t in T:
        d = report[name]["targets"].get(t)
        if not d:
            print(f"  {name:<12}{t:<6} (fewer than 5 compounds)")
            continue
        print(f"  {name:<12}{t:<6}& {d['n']:>3} & {d['var_ratio']:.2f} & {d['r2_pearson']:.3f} "
              f"& {d['ceiling']:.3f} & {d['pct_of_ceiling']:.0f} \\\\"
              f"     (p={d['pearson_p']:.4f}, rho={d['spearman']:+.2f}, "
              f"pair-acc {100*d['pairwise_acc_gap0.5']:.0f}% of {d['n_pairs']}, "
              f"FP-identical {d['n_fp_identical']})")

print()
print(B); print("TABLE 6 ROWS  (attainability of a composition-matched internal comparator)")
print(B)
print(f"  {'set':<12}{'t':<6}{'n':>4}{'var':>8}{'pctile':>9}{'NN':>8}{'pctile':>9}   matched?")
for name, _ in SETS:
    for t in T:
        d = report[name]["targets"].get(t)
        if not d:
            continue
        def mk(p):
            return "unattain." if p < 1 or p > 99 else f"{p:.0f}th"
        print(f"  {name:<12}{t:<6}{d['n']:>4}{d['var']:>8.2f}{mk(d['var_percentile']):>9}"
              f"{d['nn_tanimoto_median']:>8.2f}{mk(d['nn_percentile']):>9}   "
              f"{'YES' if d['attainable'] else 'no'}")

print()
print(B); print("RANGE-MATCHED INTERNAL NULL, where defined")
print(B)
for name, _ in SETS:
    for t in T:
        d = report[name]["targets"].get(t)
        if not d:
            continue
        nl = d["range_matched_null"]
        if nl["median"] is None:
            print(f"  {name:<12}{t:<6} no draws satisfied the constraint")
            continue
        inside = nl["q05"] <= d["r2_pearson"] <= nl["q95"]
        print(f"  {name:<12}{t:<6} external {d['r2_pearson']:.3f}   null median "
              f"{nl['median']:.3f} [{nl['q05']:.3f}, {nl['q95']:.3f}]   "
              f"{'INSIDE' if inside else 'OUTSIDE'}"
              f"   (only meaningful if attainable: {d['attainable']})")
print("\nwrote external_final.json")
