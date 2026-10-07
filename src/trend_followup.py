"""Two follow-ups that the pooled correlations cannot settle:
   7. Within each set, does the target with the most potent compounds score best?
      (Pooling across sets confounds set identity; within-set ranking does not.)
   8. Which of the two failure modes -- compression or offset -- dominates each row?
"""
import itertools, numpy as np, pandas as pd, math
from scipy import stats

R = pd.read_csv("trend_table.csv")
E = R[R.ext].reset_index(drop=True)
print("=" * 88)
print("7.  Within-set ranking: mean potency vs r^2, three targets per set")
print("=" * 88)
rhos, conc = [], 0
for ev in E.ev.unique():
    s = E[E.ev == ev].sort_values("ymean", ascending=False)
    rho = stats.spearmanr(s.ymean, s.r2).statistic
    rhos.append(rho)
    top_pot, top_r2 = s.iloc[0]["t"], s.sort_values("r2").iloc[-1]["t"]
    conc += top_pot == top_r2
    print(f"  {ev:<18} potency order {' > '.join(s.t)}"
          f"   r^2 order {' > '.join(s.sort_values('r2', ascending=False).t)}"
          f"   rho {rho:+.2f}")
print(f"\n  most-potent target also scores highest in {conc}/3 sets; mean within-set "
      f"rho = {np.mean(rhos):+.2f}")
# exact permutation null: all 3!^3 orderings of r^2 within sets
null = []
sets = [E[E.ev == ev].reset_index(drop=True) for ev in E.ev.unique()]
for perms in itertools.product(*[list(itertools.permutations(range(3)))] * 3):
    v = [stats.spearmanr(s.ymean, s.r2.values[list(p)]).statistic
         for s, p in zip(sets, perms)]
    null.append(np.mean(v))
null = np.array(null)
print(f"  exact permutation test over all {null.size} within-set relabellings: "
      f"p = {(null >= np.mean(rhos)).mean():.4f}")

print()
print("=" * 88)
print("8.  Which failure mode dominates:  compression (r-k)^2  or  offset (bias/Sy)^2")
print("=" * 88)
print(f"{'evaluation':<18}{'t':<6}{'gap':>8}{'% compression':>16}{'% offset':>11}  verdict")
for _, w in R.iterrows():
    Sy2 = w["var"] * (w["n"] - 1) / w["n"]
    r = math.sqrt(w["r2"]) * (1 if w["slope"] >= 0 else -1)
    k = w["slope"] / r
    sp, bi = (r - k) ** 2, w["bias"] ** 2 / Sy2
    g = sp + bi
    v = "offset" if bi > 2 * sp else ("compression" if sp > 2 * bi else "both")
    print(f"{w['ev']:<18}{w['t']:<6}{g:>8.3f}{100*sp/g:>15.0f}%{100*bi/g:>10.0f}%  {v}")
