"""Can an internal test set of the same size be made to look like each external set?

The composition-matched null assumes it can: draw internal subsets matching the external
set's size, response variance, mean potency and nearest-neighbour similarity, and read
off what the model would have scored on a test set of that shape.

That assumption has to be checked, and it has not been. For each external cell this asks
where the external set's variance and similarity fall in the distribution of values
attainable by ANY random internal subset of that size. If they fall outside it, no
matched comparator exists, and a rejection sampler with a widening tolerance will quietly
return a band matched to nothing.
"""
import json
import numpy as np
from scipy import stats

rng = np.random.default_rng(20261003)
z = np.load("internal_oof.npz")
comp = json.load(open("competency2.json"))
COL = {"DAT": "pDAT", "NET": "pNET", "SERT": "pSERT"}
TRIALS = 20000


def attainable(y, nn, n):
    v = np.empty(TRIALS); s = np.empty(TRIALS)
    for i in range(TRIALS):
        k = rng.choice(y.size, n, replace=False)
        v[i] = y[k].var(ddof=1); s[i] = np.median(nn[k])
    return v, s


B = "=" * 104
print(B)
print("Is a composition-matched internal comparator attainable at all?")
print(B)
print(f"{'set':<18}{'t':<6}{'n':>4}   {'variance':>30}   {'NN similarity':>30}")
print(f"{'':<28}   {'external':>9}{'percentile':>12}{'':>9}"
      f"{'external':>12}{'percentile':>12}")
rows = []
for k, v in comp.items():
    nm = k.split("(")[0].strip()
    for t, d in v["targets"].items():
        y, nn = z[f"{COL[t]}_y"], z["nn"]
        av, as_ = attainable(y, nn, d["n"])
        pv = float((av < d["var"]).mean() * 100)
        pn = float((as_ < d["nn_tanimoto_median"]).mean() * 100)
        def mark(p):
            return "  unattainable" if p < 1 or p > 99 else f"  {p:.0f}th"
        print(f"{nm:<18}{t:<6}{d['n']:>4}   {d['var']:>9.2f}{mark(pv):>12}"
              f"{'':>9}{d['nn_tanimoto_median']:>12.2f}{mark(pn):>12}")
        rows.append(dict(set=nm, target=t, n=d["n"], var=d["var"], var_pct=pv,
                         nn=d["nn_tanimoto_median"], nn_pct=pn,
                         attainable=bool(1 <= pv <= 99 and 1 <= pn <= 99),
                         r2=d["r2_pearson"]))

ok = [r for r in rows if r["attainable"]]
print(f"\n  A matched comparator exists for {len(ok)} of {len(rows)} cells: "
      + ", ".join(f"{r['set']} {r['target']}" for r in ok))
print("  For the rest, the external set's variance or similarity lies beyond anything")
print("  a random internal subset of that size attains, so no matched null is defined.")

# -------- the matched null, computed only where it is defined ------------------
print("\n" + B)
print("Matched null, restricted to the cells where it is computable")
print(B)


def null(t, n_ext, var_t, nn_t, draws=200000):
    y, p, nn = z[f"{COL[t]}_y"], z[f"{COL[t]}_p"], z["nn"]
    out = []
    for _ in range(draws):
        s = rng.choice(y.size, n_ext, replace=False)
        ys = y[s]
        if abs(ys.var(ddof=1) - var_t) > 0.15 * var_t:
            continue
        if abs(np.median(nn[s]) - nn_t) > 0.06:
            continue
        if np.std(ys) == 0 or np.std(p[s]) == 0:
            continue
        out.append(stats.pearsonr(ys, p[s])[0] ** 2)
        if len(out) >= 4000:
            break
    a = np.array(out)
    return None if a.size < 200 else dict(
        n=int(a.size), median=float(np.median(a)),
        q05=float(np.percentile(a, 5)), q95=float(np.percentile(a, 95)))


print(f"{'set':<18}{'t':<6}{'ext r2':>9}{'draws':>8}{'null median':>13}"
      f"{'5–95% band':>16}{'verdict':>10}")
for r in ok:
    nl = null(r["target"], r["n"], r["var"], r["nn"])
    if nl is None:
        print(f"{r['set']:<18}{r['target']:<6}{r['r2']:>9.3f}"
              f"{'too few accepted draws':>47}")
        r["verdict"] = None
        continue
    inb = nl["q05"] <= r["r2"] <= nl["q95"]
    r["verdict"] = inb
    r["null"] = nl
    band = "[{:.2f}, {:.2f}]".format(nl["q05"], nl["q95"])
    print(f"{r['set']:<18}{r['target']:<6}{r['r2']:>9.3f}{nl['n']:>8}"
          f"{nl['median']:>13.3f}{band:>16}"
          f"{'INSIDE' if inb else 'OUTSIDE':>10}")

json.dump(rows, open("reachability.json", "w"), indent=1, default=float)
print("\nwrote reachability.json")
