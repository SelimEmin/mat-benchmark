"""Is there accuracy left on the table, and where?

Internal r^2 sits at 0.51-0.56 against ceilings of 0.80-0.88, so 30-40% of what the
data permits is still unreached. Four levers are tested here on ONE shared global
scaffold split, so a compound lands in the same fold for every target and every arm and
the only thing that varies is the lever.

  multi-task   one model over all three transporters at once, with a one-hot target
               feature. DAT, NET and SERT are homologous and their SAR is correlated,
               so this is the obvious structural win -- it triples the training rows a
               single model sees. Target identity is known at prediction time, so the
               feature is legitimate rather than leakage.
  descriptors  the full RDKit descriptor block (~200) on top of the 2,048 ECFP4 bits
               and the 10 descriptors already used.
  (tuning)     run separately once the above are known, to avoid tuning on an
               architecture that is about to change.

Every arm keeps the modality feature and the per-target compound universe from the
production configuration, and is scored the same way: stacked single-modality keys.
"""
import json, time, warnings
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors as rdmd
from rdkit.Chem.Scaffolds import MurckoScaffold
from scipy import stats

warnings.filterwarnings("ignore")
RDLogger.DisableLog("rdApp.*")
GEN = AllChem.GetMorganGenerator(radius=2, fpSize=2048)
BASE_DESCS = [Descriptors.MolWt, Descriptors.MolLogP, Descriptors.TPSA, rdmd.CalcNumHBD,
              rdmd.CalcNumHBA, rdmd.CalcNumRotatableBonds, rdmd.CalcNumAromaticRings,
              rdmd.CalcFractionCSP3, lambda m: m.GetNumHeavyAtoms(), rdmd.CalcNumRings]
FULL = [f for n, f in Descriptors._descList]
TARGETS = ("DAT", "NET", "SERT")
NSEED, NFOLD = 5, 5


def lgbm(**kw):
    p = dict(n_estimators=250, learning_rate=0.06, num_leaves=31, min_child_samples=10,
             subsample=0.8, colsample_bytree=0.4, random_state=0, verbose=-1, n_jobs=2)
    p.update(kw)
    return LGBMRegressor(**p)


def scaffold_folds_seeded(scaffolds, n_splits=5, seed=0):
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


# ------------------------------------------------- one compound table for all targets
lab = pd.read_csv("modality_labels.csv")
wide = lab.pivot_table(index="ik", columns="target", values=["p_Ki", "p_IC50"])
wide.columns = [f"{a}_{b}" for a, b in wide.columns]
wide = wide.reset_index().merge(lab[["ik", "smiles"]].drop_duplicates("ik"), on="ik")
mols = [Chem.MolFromSmiles(s) for s in wide.smiles]
wide = wide[[m is not None for m in mols]].reset_index(drop=True)
mols = [m for m in mols if m is not None]
wide["scaffold"] = [MurckoScaffold.MurckoScaffoldSmiles(mol=m) for m in mols]
print(f"union compound table: {len(wide)} compounds", flush=True)

fp = np.zeros((len(mols), 2048), np.int8)
for i, m in enumerate(mols):
    DataStructs.ConvertToNumpyArray(GEN.GetFingerprint(m), fp[i])
base = np.array([[f(m) for f in BASE_DESCS] for m in mols], np.float32)
t0 = time.time()
full = np.array([[f(m) for f in FULL] for m in mols], np.float32)
full = np.nan_to_num(full, nan=0.0, posinf=0.0, neginf=0.0)
keepcol = full.std(axis=0) > 0
full = full[:, keepcol]
print(f"RDKit descriptor block: {full.shape[1]} usable of {len(FULL)} "
      f"({time.time()-t0:.0f}s)", flush=True)

F = {"ecfp": np.hstack([fp, base]).astype(np.float32),
     "ecfp+desc": np.hstack([fp, base, full]).astype(np.float32)}
Y = {t: {"Ki": wide[f"p_Ki_{t}"].values, "IC50": wide[f"p_IC50_{t}"].values}
     for t in TARGETS}


def rows_for(X, idx, t, onehot=None):
    """Expand compounds to (compound, modality) training rows for one target."""
    out_X, out_y = [], []
    for mi, mod in ((1.0, "Ki"), (0.0, "IC50")):
        k = idx[np.isfinite(Y[t][mod][idx])]
        if not k.size:
            continue
        cols = [X[k], np.full((k.size, 1), mi, np.float32)]
        if onehot is not None:
            cols.append(np.tile(onehot, (k.size, 1)))
        out_X.append(np.hstack(cols))
        out_y.append(Y[t][mod][k])
    return np.vstack(out_X), np.concatenate(out_y)


def predict_for(model, X, idx, t, mod, onehot=None):
    cols = [X[idx], np.full((idx.size, 1), 1.0 if mod == "Ki" else 0.0, np.float32)]
    if onehot is not None:
        cols.append(np.tile(onehot, (idx.size, 1)))
    return model.predict(np.hstack(cols))


def stacked_r2(t, pk, pi, idx):
    yk, yi = Y[t]["Ki"][idx], Y[t]["IC50"][idx]
    a, b = np.isfinite(yk), np.isfinite(yi)
    y = np.concatenate([yk[a], yi[b]])
    p = np.concatenate([pk[a], pi[b]])
    return float(stats.pearsonr(y, p)[0] ** 2), y, p


OH = {t: np.eye(3, dtype=np.float32)[i:i + 1] for i, t in enumerate(TARGETS)}
res = {}
for feat in ("ecfp", "ecfp+desc"):
    X = F[feat]
    for mode in ("per-target", "multi-task"):
        arm = f"{mode} / {feat}"
        acc = {t: [] for t in TARGETS}
        t0 = time.time()
        for seed in range(NSEED):
            folds = scaffold_folds_seeded(wide.scaffold.values, NFOLD, seed)
            P = {t: [np.full(len(wide), np.nan), np.full(len(wide), np.nan)]
                 for t in TARGETS}
            for tr, te in folds:
                if mode == "multi-task":
                    Xs, ys = [], []
                    for t in TARGETS:
                        a, b = rows_for(X, tr, t, OH[t])
                        Xs.append(a); ys.append(b)
                    m = lgbm().fit(np.vstack(Xs), np.concatenate(ys))
                    for t in TARGETS:
                        P[t][0][te] = predict_for(m, X, te, t, "Ki", OH[t])
                        P[t][1][te] = predict_for(m, X, te, t, "IC50", OH[t])
                else:
                    for t in TARGETS:
                        a, b = rows_for(X, tr, t)
                        m = lgbm().fit(a, b)
                        P[t][0][te] = predict_for(m, X, te, t, "Ki")
                        P[t][1][te] = predict_for(m, X, te, t, "IC50")
            for t in TARGETS:
                acc[t].append(stacked_r2(t, P[t][0], P[t][1],
                                         np.arange(len(wide)))[0])
        res[arm] = {t: {"mean": float(np.mean(acc[t])),
                        "sd": float(np.std(acc[t], ddof=1)),
                        "seeds": [float(v) for v in acc[t]]} for t in TARGETS}
        print(f"{arm:<26} " + "  ".join(
            f"{t} {res[arm][t]['mean']:.3f}±{res[arm][t]['sd']:.3f}" for t in TARGETS)
            + f"   [{(time.time()-t0)/60:.1f} min]", flush=True)
        json.dump(res, open("improve_experiment.json", "w"), indent=1)

print("\n" + "=" * 86)
print("Paired comparison against per-target / ecfp (the production configuration)")
print("=" * 86)
b = res["per-target / ecfp"]
for arm in res:
    if arm == "per-target / ecfp":
        continue
    print(f"{arm:<26}", end="")
    for t in TARGETS:
        d = np.array(res[arm][t]["seeds"]) - np.array(b[t]["seeds"])
        p = stats.ttest_rel(res[arm][t]["seeds"], b[t]["seeds"]).pvalue
        print(f"  {t} {d.mean():+.3f} (p {p:.3f})", end="")
    print()
print("\nwrote improve_experiment.json")
