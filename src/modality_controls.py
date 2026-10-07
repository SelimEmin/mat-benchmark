"""Two controls on the modality experiment, plus significance.

Control 1 -- is the pool_bench arm being judged unfairly?
    It trains only on benchmark compounds but is tested on the full per-target
    universe, which includes ~1,200-3,000 compounds the benchmark excluded. The
    published internal R^2 is benchmark-trained AND benchmark-tested. Re-scoring every
    arm on the benchmark subset of each held-out fold recovers the published setting
    and shows whether the gain survives it.

Control 2 -- is the gain from MORE compounds, or from WHICH compounds?
    Subsample the full training pool, at random, to exactly the size of the benchmark
    training pool. If the subsample still beats the benchmark arm, the intersection
    requirement is selecting a bad subset, not merely a small one.

Significance: paired bootstrap over compounds, 2,000 resamples, for the differences
that matter.
"""
import json
import numpy as np
import pandas as pd
from scipy import stats

TARGETS = ("DAT", "NET", "SERT")
rng = np.random.default_rng(7)
bench = set(pd.read_csv("mat_profiles.csv").inchikey)


def r2(y, p):
    m = ~(np.isnan(y) | np.isnan(p))
    return float(stats.pearsonr(y[m], p[m])[0] ** 2) if m.sum() > 5 else np.nan


def boot_diff(y, pa, pb, n=2000):
    """Paired bootstrap on r2(pa) - r2(pb) over the compounds scored by both."""
    m = ~(np.isnan(y) | np.isnan(pa) | np.isnan(pb))
    y, pa, pb = y[m], pa[m], pb[m]
    obs = r2(y, pa) - r2(y, pb)
    d = np.empty(n)
    for i in range(n):
        s = rng.integers(0, y.size, y.size)
        d[i] = r2(y[s], pa[s]) - r2(y[s], pb[s])
    lo, hi = np.percentile(d, [2.5, 97.5])
    return obs, lo, hi, float((d <= 0).mean() if obs > 0 else (d >= 0).mean())


print("=" * 94)
print("CONTROL 1 -- every arm re-scored on the benchmark compounds only")
print("            (the published setting: benchmark-trained, benchmark-tested)")
print("=" * 94)
print(f"{'target':<6}{'key':<11}{'arm':<12}{'full universe':>15}{'benchmark only':>17}"
      f"{'n bench':>10}")
ctrl = {}
for t in TARGETS:
    z = np.load(f"modality_oof_{t}.npz", allow_pickle=True)
    inb = np.isin(z["ik"], list(bench))
    for key, ycol in (("key_Ki", "y_ki"), ("key_IC50", "y_ic"),
                      ("key_pooled", "y_pool")):
        y = z[ycol]
        for arm, src in (("pool_bench", "pool_bench"), ("pool_all", "pool_all"),
                         ("mod_feat", "mf_ki" if key == "key_Ki" else "mf_ic"),
                         ("separate", "sep_ki" if key == "key_Ki" else "sep_ic")):
            if key == "key_pooled" and arm in ("mod_feat", "separate"):
                continue
            p = z[src]
            a, b = r2(y, p), r2(y[inb], p[inb])
            ctrl[(t, key, arm)] = (a, b)
            print(f"{t:<6}{key:<11}{arm:<12}{a:>15.3f}{b:>17.3f}"
                  f"{int((inb & np.isfinite(y)).sum()):>10}")
    print()

print("=" * 94)
print("CONTROL 2 -- size-matched subsample (see modality_subsample.json)")
print("=" * 94)
try:
    sub = json.load(open("modality_subsample.json"))
    print(f"{'target':<6}{'key':<11}{'pool_bench':>12}{'size-matched':>14}"
          f"{'pool_all':>11}   interpretation")
    for t in TARGETS:
        for key in ("key_Ki", "key_IC50", "key_pooled"):
            pb = ctrl[(t, key, "pool_bench")][0]
            pa = ctrl[(t, key, "pool_all")][0]
            ps = sub[t][key]
            what = ("which compounds" if ps - pb > 0.6 * (pa - pb)
                    else "how many compounds")
            print(f"{t:<6}{key:<11}{pb:>12.3f}{ps:>14.3f}{pa:>11.3f}   {what}")
except FileNotFoundError:
    print("  (run modality_subsample.py first)")

print()
print("=" * 94)
print("SIGNIFICANCE -- paired bootstrap over compounds, 2,000 resamples")
print("=" * 94)
print(f"{'target':<6}{'key':<11}{'comparison':<26}{'d r2':>8}{'95% CI':>18}{'p':>9}")
out = {}
for t in TARGETS:
    z = np.load(f"modality_oof_{t}.npz", allow_pickle=True)
    for key, ycol, mf, sp in (("key_Ki", "y_ki", "mf_ki", "sep_ki"),
                              ("key_IC50", "y_ic", "mf_ic", "sep_ic")):
        y = z[ycol]
        for lab, a, b in (("pool_all - pool_bench", "pool_all", "pool_bench"),
                          ("mod_feat - pool_all", mf, "pool_all"),
                          ("separate - pool_all", sp, "pool_all")):
            o, lo, hi, p = boot_diff(y, z[a], z[b])
            out[f"{t}|{key}|{lab}"] = {"d": o, "lo": lo, "hi": hi, "p": p}
            star = "  *" if lo > 0 or hi < 0 else ""
            print(f"{t:<6}{key:<11}{lab:<26}{o:>+8.3f}"
                  f"   [{lo:>+6.3f},{hi:>+6.3f}]{p:>9.3f}{star}")
    print()

json.dump({"controls": {f"{k[0]}|{k[1]}|{k[2]}": {"full": v[0], "bench_only": v[1]}
                        for k, v in ctrl.items()}, "bootstrap": out},
          open("modality_controls.json", "w"), indent=1)
print("wrote modality_controls.json   (* = 95% CI excludes zero)")
