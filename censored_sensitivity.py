"""R7: tabulate the censored-objective sensitivity analysis in full.

The manuscript retains censored measurements at their bounds and trains a squared-loss
model on them, reporting only that a censored-objective check "agreed in sign across six
fold realisations". A referee is entitled to the effect sizes.

In p-scale a censored record is an UPPER bound: "IC50 > 10 uM" becomes "p < 5", so the
recorded value bounds the true potency from above. The censored objective therefore
penalises predictions ABOVE the bound and leaves predictions below it free:

    uncensored   L = (f - y)^2            g = 2(f - y)         h = 2
    censored     L = max(0, f - y)^2      g = 2*max(0, f - y)  h = 2 if f > y else 0

Everything else -- features, folds, hyperparameters, seeds -- is the pre-registered
pipeline unchanged, so the only difference between the two columns is the loss.
"""
import pickle
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from scipy import stats

NSEED = 10
TARGETS = ("pDAT", "pNET", "pSERT")
EPS = 1e-6

store = pickle.load(open("faithful_models.pkl", "rb"))
d = store["frame"].reset_index(drop=True)
X = np.load("faithful_X.npy")
print(f"{len(d)} compounds, {X.shape[1]} features", flush=True)


def folds_for(seed, n_splits=5):
    rng = np.random.default_rng(seed)
    groups = {}
    for i, s in enumerate(pd.Series(d.scaffold.values).fillna("").values):
        groups.setdefault(s, []).append(i)
    keys = list(groups); rng.shuffle(keys)
    out = [[] for _ in range(n_splits)]
    for k in sorted(keys, key=lambda k: -len(groups[k])):
        out[int(np.argmin([len(o) for o in out]))].extend(groups[k])
    return [np.array(sorted(o)) for o in out]


def make_objective(cens):
    """cens is a 0/1 array aligned with the training rows."""
    def obj(y_true, y_pred):
        r = y_pred - y_true
        g = 2.0 * r
        h = np.full_like(r, 2.0)
        over = cens == 1
        under = over & (r <= 0)          # censored and predicted below the bound: free
        g[under] = 0.0
        h[under] = EPS
        return g, h
    return obj


def run(y, cens, folds, censored_objective):
    p = np.zeros(y.size)
    for te in folds:
        tr = np.setdiff1d(np.arange(y.size), te)
        kw = dict(n_estimators=250, learning_rate=0.06, num_leaves=31,
                  min_child_samples=10, subsample=0.8, colsample_bytree=0.4,
                  random_state=0, verbose=-1, n_jobs=2)
        if censored_objective:
            kw["objective"] = make_objective(cens[tr])
        m = LGBMRegressor(**kw)
        m.fit(X[tr], y[tr])
        p[te] = m.predict(X[te])
    return p


print("\ncensored fraction per target:")
for t in TARGETS:
    c = d["cens_" + t.replace("p", "", 1)].values
    sub = d[t].values
    print(f"  {t:<6} {int(c.sum()):>4} of {len(c)} ({100*c.mean():.1f}%)   "
          f"median p censored {np.median(sub[c == 1]):.2f} vs "
          f"uncensored {np.median(sub[c == 0]):.2f}")

rows = []
for t in TARGETS:
    y = d[t].values.astype(float)
    cens = d["cens_" + t.replace("p", "", 1)].values.astype(int)
    for censored_objective in (False, True):
        r2s, rmses = [], []
        for seed in range(NSEED):
            folds = folds_for(seed)
            p = run(y, cens, folds, censored_objective)
            # evaluated on UNCENSORED compounds only, where the label is a real value
            m = cens == 0
            r2s.append(stats.pearsonr(y[m], p[m])[0] ** 2)
            rmses.append(float(np.sqrt(((y[m] - p[m]) ** 2).mean())))
        rows.append(dict(target=t, objective="censored" if censored_objective else "squared",
                         r2=float(np.mean(r2s)), r2_sd=float(np.std(r2s, ddof=1)),
                         rmse=float(np.mean(rmses)), rmse_sd=float(np.std(rmses, ddof=1))))
        print(f"  {t:<6} {'censored' if censored_objective else 'squared ':<9} "
              f"r2 {np.mean(r2s):.3f} +/- {np.std(r2s, ddof=1):.3f}   "
              f"RMSE {np.mean(rmses):.3f} +/- {np.std(rmses, ddof=1):.3f}", flush=True)

R = pd.DataFrame(rows)
R.to_csv("censored_sensitivity.csv", index=False)
print("\n" + "=" * 78)
print("INTERNAL, EVALUATED ON UNCENSORED COMPOUNDS, 10 SCAFFOLD-CV REALISATIONS")
print("=" * 78)
print(f"{'target':<7}{'squared loss':>22}{'censored objective':>24}{'difference':>14}")
for t in TARGETS:
    a = R[(R.target == t) & (R.objective == "squared")].iloc[0]
    b = R[(R.target == t) & (R.objective == "censored")].iloc[0]
    print(f"{t:<7}{a.r2:.3f} +/- {a.r2_sd:.3f}{'':>8}{b.r2:.3f} +/- {b.r2_sd:.3f}"
          f"{'':>10}{b.r2 - a.r2:+.3f}")
print("\nwrote censored_sensitivity.csv")
