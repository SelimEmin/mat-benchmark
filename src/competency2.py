"""Competency battery, now on the faithfully rebuilt model.

Reports BOTH definitions of R^2, because the project's own code uses the first and
most readers assume the second:

    r2_pearson  = pearsonr(y, p) ** 2      <- what the manuscript reports
    R2_deter    = 1 - SS_res / SS_tot      <- the coefficient of determination

They differ by exactly the bias and slope of the predictions. Squared Pearson is
invariant to any linear rescaling of p, so it measures ranking-and-spread agreement
only; the coefficient of determination additionally penalises a model whose
predictions are systematically shifted or compressed.
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
TARGETS = ("DAT", "NET", "SERT")
COL = {"DAT": "pDAT", "NET": "pNET", "SERT": "pSERT"}
FLOOR = 0.194
SIG_E = {"DAT": 0.445, "NET": 0.655, "SERT": 0.492}
PUBLISHED_EXT = {"DAT": 0.459, "NET": 0.015, "SERT": 0.058}
rng = np.random.default_rng(20260930)


def feat(s):
    m = Chem.MolFromSmiles(s)
    if m is None:
        return None
    fp = GEN.GetFingerprint(m)
    arr = np.zeros(2048, dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return np.concatenate([arr, [f(m) for f in DESCS]]).astype(np.float32)


def parent_key(s):
    m = Chem.MolFromSmiles(s)
    if m is None:
        return None
    fr = Chem.GetMolFrags(m, asMols=True, sanitizeFrags=False)
    return Chem.MolToInchiKey(max(fr, key=lambda x: x.GetNumHeavyAtoms()) if fr else m)


store = pickle.load(open("faithful_models.pkl", "rb"))
models, frame = store["models"], store["frame"]
train_keys = {parent_key(s) for s in frame.smiles}
train_fps = [GEN.GetFingerprint(Chem.MolFromSmiles(s)) for s in frame.smiles]
train_mean = {t: float(frame[COL[t]].mean()) for t in TARGETS}
train_var = {t: float(frame[COL[t]].var(ddof=1)) for t in TARGETS}

# ----------------------------------------------------------------- the sets
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
    rec = {"smiles": r["smiles"]}
    for t in TARGETS:
        rec[t] = float(r["p" + t]) if r["p" + t] != "" else None
    SET_CUI.append(rec)

SETS = [("Stalberga 2026  (stimulants, uptake IC50)", SET_STAL),
        ("PDSP/BindingDB  (antidepressants, binding Ki)", SET_PDSP),
        ("Cui 2026        (SNRI series, uptake IC50)", SET_CUI)]


def pairwise(y, p, gap):
    i, j = np.triu_indices(y.size, 1)
    dy, dp = y[i] - y[j], p[i] - p[j]
    m = np.abs(dy) > gap
    return (float((np.sign(dy[m]) == np.sign(dp[m])).mean()), int(m.sum())) \
        if m.sum() else (float("nan"), 0)


oof = json.load(open("faithful_oof.json"))


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
    return {"n": int(a.size), "median": float(np.median(a)),
            "q05": float(np.percentile(a, 5)), "q95": float(np.percentile(a, 95))}


report = {}
for name, rows in SETS:
    rows = [r for r in rows if parent_key(r["smiles"]) not in train_keys
            and feat(r["smiles"]) is not None]
    print("=" * 94)
    print(f"{name}    n={len(rows)} after exact-InChIKey exclusion")
    print("=" * 94)
    rec = {"n": len(rows), "targets": {}}
    for t in TARGETS:
        keep = [i for i, r in enumerate(rows)
                if r[t] is not None and feat(r["smiles"]) is not None]
        if len(keep) < 5:
            continue
        y = np.array([rows[i][t] for i in keep])
        X = np.vstack([feat(rows[i]["smiles"]) for i in keep])
        p = models[COL[t]].predict(X)
        r_pear, pval = stats.pearsonr(y, p)
        r2p = float(r_pear ** 2)
        r2d = float(1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum())
        var = float(y.var(ddof=1))
        ceil = 1 - SIG_E[t] ** 2 / var
        err = np.abs(y - p)
        sims = np.array([max(DataStructs.BulkTanimotoSimilarity(
            GEN.GetFingerprint(Chem.MolFromSmiles(rows[i]["smiles"])), train_fps))
            for i in keep])
        pa, npair = pairwise(y, p, 0.5)
        null = range_matched_null(t, var, len(keep))
        d = {"n": len(keep), "var": var, "var_ratio": var / train_var[t],
             "r2_pearson": r2p, "R2_determination": r2d, "pearson_p": float(pval),
             "spearman": float(stats.spearmanr(y, p).statistic),
             "RMSE": float(np.sqrt(((y - p) ** 2).mean())),
             "bias": float((p - y).mean()), "slope": float(np.polyfit(y, p, 1)[0]),
             "ceiling": float(ceil),
             "r2_over_ceiling": float(r2p / ceil) if ceil > 0 else None,
             "pct_within_floor": float((err <= FLOOR).mean() * 100),
             "pct_within_0_5": float((err <= 0.5).mean() * 100),
             "pairwise_acc_gap0.5": pa, "n_pairs": npair,
             "baseline_mean_RMSE": float(np.sqrt(((y - train_mean[t]) ** 2).mean())),
             "nn_tanimoto_median": float(np.median(sims)),
             "range_matched_null": null}
        rec["targets"][t] = d
        pub = (f"  [published {PUBLISHED_EXT[t]:.3f}]"
               if name.startswith("Stalberga") else "")
        print(f"\n  --- {t}  n={len(keep)} ---")
        print(f"    variance {var:.3f} ({var/train_var[t]:.2f}x training)   "
              f"NN Tanimoto median {np.median(sims):.2f}")
        print(f"    r2 (squared Pearson) {r2p:.3f}  p={pval:.4f}{pub}")
        print(f"    R2 (determination)   {r2d:+.3f}    bias {d['bias']:+.3f}  "
              f"slope {d['slope']:.3f}")
        print(f"    Spearman {d['spearman']:+.3f}   RMSE {d['RMSE']:.3f} "
              f"(baseline-mean {d['baseline_mean_RMSE']:.3f})")
        print(f"    within floor {d['pct_within_floor']:.0f}%   "
              f"within 0.5 log {d['pct_within_0_5']:.0f}%   "
              f"pairwise accuracy (>0.5 log) {100*pa:.1f}% of {npair}")
        print(f"    ceiling {ceil:.3f} -> r2 is {100*r2p/ceil:.0f}% of reachable;  "
              f"range-matched internal null {null['median']:.3f} "
              f"[{null['q05']:.3f}, {null['q95']:.3f}] -> "
              f"{'INSIDE' if null['q05'] <= r2p <= null['q95'] else 'OUTSIDE'}")
    report[name] = rec

json.dump(report, open("competency2.json", "w"), indent=2)
print("\nwrote competency2.json")
