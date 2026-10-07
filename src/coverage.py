"""Does the Comprehensive-profiling set cover the potency range the training set misses?

Model-free: compares the external set's potency distribution against the training
distribution at each target. No model involved, so no shrinkage artefact.
"""
import csv, json, math
from collections import defaultdict
import numpy as np

# ---------------------------------------------------------------- external set
rows = list(csv.DictReader(open("stalberga_parsed.csv")))
ext = defaultdict(dict)          # compound -> target -> (relation, pIC50)
for r in rows:
    v = r["value"].strip()
    if v == "":
        ext[r["compound"]][r["target"]] = (r["relation"], None)
        continue
    nM = float(v)
    assert r["unit"] == "nM"
    ext[r["compound"]][r["target"]] = (r["relation"], 9.0 - math.log10(nM))

censored = {c for c, d in ext.items() if any(x[1] is None or x[0] == ">" for x in d.values())}
usable = {c: d for c, d in ext.items() if c not in censored}

# ---------------------------------------------------------------- training set
tr = list(csv.DictReader(open("mat_profiles.csv")))
TRAIN = {}
for t, col, cc in (("DAT", "pDAT", "cens_DAT"), ("NET", "pNET", "cens_NET"),
                   ("SERT", "pSERT", "cens_SERT")):
    vals = [float(r[col]) for r in tr if r[col].strip() != "" and r[cc] == "0"]
    TRAIN[t] = np.array(vals)

out = {"external_set": {"doi": "10.1111/bph.70587",
                        "compounds_parsed": len(ext),
                        "right_censored_at_any_target": len(censored),
                        "usable_all_three": len(usable),
                        "censored_names": sorted(censored)},
       "targets": {}}

print("=" * 78)
print("EXTERNAL SET: Stalberga et al. 2026, Br J Pharmacol, DOI 10.1111/bph.70587")
print(f"  parsed {len(ext)} compounds x 3 targets; "
      f"{len(censored)} right-censored at >=1 target; {len(usable)} fully quantified")
print("=" * 78)

for t in ("DAT", "NET", "SERT"):
    train = TRAIN[t]
    q = {p: float(np.percentile(train, p)) for p in (10, 25, 50, 75, 90, 95, 99)}
    e = np.array([d[t][1] for d in usable.values()])
    enames = list(usable.keys())

    # coverage of the top of the training range
    above75 = [(n, d[t][1]) for n, d in usable.items() if d[t][1] > q[75]]
    above90 = [(n, d[t][1]) for n, d in usable.items() if d[t][1] > q[90]]
    below25 = [(n, d[t][1]) for n, d in usable.items() if d[t][1] < q[25]]

    # where each external compound sits in the training distribution
    pct = [float((train < v).mean() * 100) for v in e]

    rec = {
        "train_n": int(train.size),
        "train_mean": float(train.mean()),
        "train_sd": float(train.std(ddof=1)),
        "train_percentiles": q,
        "ext_n": int(e.size),
        "ext_mean": float(e.mean()),
        "ext_sd": float(e.std(ddof=1)),
        "ext_min": float(e.min()), "ext_max": float(e.max()),
        "ext_median_train_percentile": float(np.median(pct)),
        "n_above_train_p75": len(above75),
        "n_above_train_p90": len(above90),
        "n_below_train_p25": len(below25),
        "above_p75": sorted(above75, key=lambda x: -x[1]),
        "variance_ratio_ext_over_train": float(e.var(ddof=1) / train.var(ddof=1)),
    }
    out["targets"][t] = rec

    print(f"\n--- {t} " + "-" * (74 - len(t)))
    print(f"  training (uncensored n={train.size}): mean {train.mean():.2f}  "
          f"sd {train.std(ddof=1):.2f}   p25 {q[25]:.2f}  p50 {q[50]:.2f}  "
          f"p75 {q[75]:.2f}  p90 {q[90]:.2f}")
    print(f"  external (n={e.size}):              mean {e.mean():.2f}  "
          f"sd {e.std(ddof=1):.2f}   min {e.min():.2f}  max {e.max():.2f}")
    print(f"  external variance / training variance = {rec['variance_ratio_ext_over_train']:.2f}")
    print(f"  median external compound sits at the {np.median(pct):.0f}th training percentile")
    print(f"  external compounds above training p75: {len(above75)} / {e.size}"
          f"   above p90: {len(above90)}   below p25: {len(below25)}")
    if above75:
        print("    " + ", ".join(f"{n} {v:.2f}" for n, v in rec['above_p75'][:12]))

json.dump(out, open("coverage.json", "w"), indent=2)
print("\nwrote coverage.json")
