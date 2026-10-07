"""Internal cross-validation measured with the SAME eight diagnostics as the
external sets, so the two can be put on one axis.

competency2.py reports variance ratio, nearest-neighbour Tanimoto, Spearman rho,
squared-Pearson r^2, the coefficient of determination, slope, bias and % of the
achievable ceiling for each external set. The manuscript reports only r^2 for the
internal cross-validation, so the other seven have to be computed here from the
faithful rebuild's out-of-fold predictions.

Protocol, matched to competency2.py exactly:
    bias   = mean(p - y)
    slope  = np.polyfit(y, p, 1)[0]
    r2p    = pearsonr(y, p) ** 2
    r2d    = 1 - SS_res / SS_tot
    ceiling = 1 - sigma_e^2 / var(y)

Two differences from an external set are structural, not choices:
  * the variance ratio is 1.00 by construction -- the held-out folds ARE the
    training distribution, so there is nothing to compare against but itself;
  * nearest-neighbour Tanimoto is computed fold-wise, each held-out compound
    against only the four folds it was NOT trained on. Computing it against the
    whole table would return 1.0 for every compound and mean nothing.

The ceiling here uses the uncensored variance, matching the manuscript's internal
ceiling table; var_all_rows is reported alongside it for transparency, since the
model is trained on all rows including censored ones.
"""
import json
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem
from scipy import stats

RDLogger.DisableLog("rdApp.*")
GEN = AllChem.GetMorganGenerator(radius=2, fpSize=2048)
TARGETS = ("DAT", "NET", "SERT")
COL = {"DAT": "pDAT", "NET": "pNET", "SERT": "pSERT"}
CENS = {"DAT": "cens_DAT", "NET": "cens_NET", "SERT": "cens_SERT"}
SIG_E = {"DAT": 0.445, "NET": 0.655, "SERT": 0.492}
SEED = 0


def scaffold_folds_seeded(scaffolds, n_splits=5, seed=0):   # repeated_cv.py, verbatim
    rng = np.random.default_rng(seed)
    groups = {}
    for i, s in enumerate(pd.Series(scaffolds).fillna("").values):
        groups.setdefault(s, []).append(i)
    keys = list(groups)
    rng.shuffle(keys)
    keys.sort(key=lambda k: -len(groups[k]))
    folds = [[] for _ in range(n_splits)]
    order = list(range(n_splits))
    for j, k in enumerate(keys):
        if j % n_splits == 0:
            rng.shuffle(order)
        folds[order[j % n_splits]].extend(groups[k])
    return [(np.array(sorted(sum((folds[j] for j in range(n_splits) if j != i), []))),
             np.array(sorted(folds[i]))) for i in range(n_splits)]


d = pd.read_csv("mat_profiles.csv")
oof = json.load(open("faithful_oof.json"))
folds = scaffold_folds_seeded(d.scaffold.values, 5, SEED)

fps = [GEN.GetFingerprint(Chem.MolFromSmiles(s)) for s in d.smiles]
nn = np.full(len(d), np.nan)
for tr, te in folds:                      # held-out compound vs its own training folds
    pool = [fps[i] for i in tr]
    for i in te:
        nn[i] = max(DataStructs.BulkTanimotoSimilarity(fps[i], pool))

print(f"internal nearest-neighbour Tanimoto to the training folds: "
      f"median {np.median(nn):.3f}, >0.7 in {100 * (nn > 0.7).mean():.1f}% of compounds")

out = {}
for t in TARGETS:
    y = np.array(oof[COL[t]]["y"], dtype=float)
    p = np.array(oof[COL[t]]["p"], dtype=float)
    m = ~(np.isnan(y) | np.isnan(p))
    y, p = y[m], p[m]

    var = float(y.var(ddof=1))
    unc = d.loc[d[CENS[t]] == 0, COL[t]].dropna()
    var_unc = float(unc.var(ddof=1))
    ceil = 1 - SIG_E[t] ** 2 / var_unc
    r2p = float(stats.pearsonr(y, p)[0] ** 2)

    out[t] = {
        "n": int(m.sum()),
        "var_ratio": 1.0,                                  # by construction
        "r2_pearson": r2p,
        "R2_determination": float(1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()),
        "slope": float(np.polyfit(y, p, 1)[0]),
        "bias": float((p - y).mean()),
        "spearman": float(stats.spearmanr(y, p).statistic),
        "nn_tanimoto_median": float(np.median(nn)),
        "nn_pct_above_0_7": float(100 * (nn > 0.7).mean()),
        "ceiling": float(ceil),
        "pct_ceiling": float(100 * r2p / ceil),
        "var_all_rows": var,
        "var_uncensored": var_unc,
    }
    o = out[t]
    print(f"{t:<5} r2 {o['r2_pearson']:.3f}  R2det {o['R2_determination']:+.3f}  "
          f"slope {o['slope']:.3f}  bias {o['bias']:+.3f}  rho {o['spearman']:+.3f}  "
          f"ceil {o['ceiling']:.3f}  {o['pct_ceiling']:.0f}%")

json.dump(out, open("internal_diagnostics.json", "w"), indent=1)
print("wrote internal_diagnostics.json")
