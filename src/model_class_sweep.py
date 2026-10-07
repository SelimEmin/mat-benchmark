"""Seven model classes spanning sixty years of QSAR, on identical data and splits.

The argument this supports is not "our model is fine" but "the limit is not in the
learner". For that, the comparison has to span model FAMILIES rather than variants of
one family -- two gradient boosters tell you nothing. These span the history of the
field:

  mean          predict the training mean. The null. Any model that cannot beat this
                externally has transferred nothing at all.
  Hansch MLR    multiple linear regression on ten physicochemical descriptors. The
                original QSAR, Hansch & Fujita 1964.
  PLS           partial least squares on descriptors + fingerprint. The workhorse of
                1980s-90s QSAR, built precisely for collinear descriptor blocks.
  Ridge         L2-penalised linear regression on the ECFP4 bits. Linear in a very
                high-dimensional space.
  kNN-Tanimoto  predict the similarity-weighted mean of the k nearest training
                compounds. The read-across / analogue principle in its purest form,
                and the model whose assumptions the distance law directly describes.
  SVR (RBF)     kernel regression, the 2000s standard.
  Random Forest Breiman trees, the 2000s-10s standard.
  LightGBM      the project's model.
  (D-MPNN)      run separately in dmpnn.py.

If every one of these lands in the same place externally while differing internally,
the external shortfall cannot be a property of any one learner.

Internal: 5-fold scaffold CV, seed 0, pooled label, one row per compound -- the
classical QSAR setup, chosen so every model class can be run the same way.
External: trained on everything with external compounds removed at InChIKey-skeleton
level, which is the exclusion that survives the fact that ECFP4 ignores chirality.
"""
import csv, json, math, time, warnings
import numpy as np
import pandas as pd
from collections import defaultdict
from lightgbm import LGBMRegressor
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors as rdmd
from rdkit.Chem.Scaffolds import MurckoScaffold
from scipy import stats
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

warnings.filterwarnings("ignore")
RDLogger.DisableLog("rdApp.*")
GEN = AllChem.GetMorganGenerator(radius=2, fpSize=2048)
DESCS = [Descriptors.MolWt, Descriptors.MolLogP, Descriptors.TPSA, rdmd.CalcNumHBD,
         rdmd.CalcNumHBA, rdmd.CalcNumRotatableBonds, rdmd.CalcNumAromaticRings,
         rdmd.CalcFractionCSP3, lambda m: m.GetNumHeavyAtoms(), rdmd.CalcNumRings]
TARGETS = ("DAT", "NET", "SERT")
NFOLD = 5


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


def skel(s):
    m = Chem.MolFromSmiles(s)
    if m is None:
        return None
    fr = Chem.GetMolFrags(m, asMols=True, sanitizeFrags=False)
    return Chem.MolToInchiKey(
        max(fr, key=lambda x: x.GetNumHeavyAtoms()) if fr else m).split("-")[0]


# ------------------------------------------------------------------ model classes
def fit_predict(name, Xfp, Xd, ytr, tr, te, fps):
    """Each model gets the representation its family actually uses."""
    if name == "mean":
        return np.full(te.size, ytr.mean())
    if name == "Hansch MLR":
        sc = StandardScaler().fit(Xd[tr])
        return LinearRegression().fit(sc.transform(Xd[tr]), ytr).predict(
            sc.transform(Xd[te]))
    if name == "PLS":
        X = np.hstack([Xfp, Xd])
        sc = StandardScaler().fit(X[tr])
        return PLSRegression(n_components=10).fit(
            sc.transform(X[tr]), ytr).predict(sc.transform(X[te])).ravel()
    if name == "Ridge":
        return Ridge(alpha=50.0).fit(Xfp[tr], ytr).predict(Xfp[te])
    if name == "kNN-Tanimoto":
        out = np.empty(te.size)
        pool = [fps[i] for i in tr]
        for n, i in enumerate(te):
            s = np.array(DataStructs.BulkTanimotoSimilarity(fps[i], pool))
            k = np.argpartition(-s, 5)[:5]
            w = s[k]
            out[n] = ytr[k].mean() if w.sum() == 0 else np.average(ytr[k], weights=w)
        return out
    if name == "SVR (RBF)":
        X = np.hstack([Xfp, Xd])
        sc = StandardScaler().fit(X[tr])
        return SVR(C=10.0, gamma="scale", cache_size=500).fit(
            sc.transform(X[tr]), ytr).predict(sc.transform(X[te]))
    if name == "Random Forest":
        return RandomForestRegressor(n_estimators=300, min_samples_leaf=2,
                                     max_features="sqrt", random_state=0,
                                     n_jobs=2).fit(
            np.hstack([Xfp, Xd])[tr], ytr).predict(np.hstack([Xfp, Xd])[te])
    if name == "LightGBM":
        return LGBMRegressor(n_estimators=250, learning_rate=0.06, num_leaves=31,
                             min_child_samples=10, subsample=0.8, colsample_bytree=0.4,
                             random_state=0, verbose=-1, n_jobs=2).fit(
            np.hstack([Xfp, Xd])[tr], ytr).predict(np.hstack([Xfp, Xd])[te])
    raise ValueError(name)


MODELS = ["mean", "Hansch MLR", "PLS", "Ridge", "kNN-Tanimoto", "SVR (RBF)",
          "Random Forest", "LightGBM"]

# ----------------------------------------------------------------- external sets
st = {r["abbrev"]: r["final"] for r in csv.DictReader(open("struct.csv"))}
v = defaultdict(dict)
for r in csv.DictReader(open("stalberga_parsed.csv")):
    v[r["compound"]][r["target"]] = (r["relation"], r["value"].strip())
STAL = [{"smiles": st[c], **{t: 9.0 - math.log10(float(d[t][1])) for t in TARGETS}}
        for c, d in v.items()
        if len(d) == 3 and all(x[0] == "=" and x[1] for x in d.values())]
PDSP = [{"smiles": r["smiles"], **{t: float(r["p" + t]) for t in TARGETS}}
        for r in csv.DictReader(open("pdsp_mat_profiles.csv"))]
CUI = [{"smiles": r["smiles"],
        **{t: (float(r["p" + t]) if r["p" + t] != "" else None) for t in TARGETS}}
       for r in csv.DictReader(open("ejmc2026_cui.csv"))]
SETS = [("Stalberga 2026", STAL), ("PDSP / BindingDB", PDSP), ("Cui et al. 2026", CUI)]
ext_skel = {skel(r["smiles"]) for _, rows in SETS for r in rows} - {None}

lab = pd.read_csv("modality_labels.csv")
# resumable per (target, model): background jobs get reaped in this container, so the
# sweep is run one target per invocation and already-finished cells are skipped
import os, sys
report = json.load(open("model_class_sweep.json")) if os.path.exists(
    "model_class_sweep.json") else {}
ONLY = sys.argv[1] if len(sys.argv) > 1 else None
for t in TARGETS:
    if ONLY and t != ONLY:
        continue
    d = lab[(lab.target == t) & lab.p_pool.notna()].reset_index(drop=True)
    mols = [Chem.MolFromSmiles(s) for s in d.smiles]
    ok = [i for i, m in enumerate(mols) if m is not None]
    d, mols = d.iloc[ok].reset_index(drop=True), [mols[i] for i in ok]
    keep = ~d.ik.str.split("-").str[0].isin(ext_skel).values     # external out of train
    d, mols = d[keep].reset_index(drop=True), [m for m, k in zip(mols, keep) if k]
    d["scaffold"] = [MurckoScaffold.MurckoScaffoldSmiles(mol=m) for m in mols]
    fps = [GEN.GetFingerprint(m) for m in mols]
    Xfp = np.zeros((len(mols), 2048), np.float32)
    for i in range(len(mols)):
        DataStructs.ConvertToNumpyArray(fps[i], Xfp[i])
    Xd = np.array([[f(m) for f in DESCS] for m in mols], np.float32)
    y = d.p_pool.values
    folds = scaffold_folds_seeded(d.scaffold.values, NFOLD, 0)
    print(f"\n{t}: {len(d)} training compounds after removing external", flush=True)

    EX = {}
    for name, rows in SETS:
        sel = [r for r in rows if r.get(t) is not None
               and Chem.MolFromSmiles(r["smiles"]) is not None]
        em = [Chem.MolFromSmiles(r["smiles"]) for r in sel]
        efp = [GEN.GetFingerprint(m) for m in em]
        A = np.zeros((len(sel), 2048), np.float32)
        for i in range(len(sel)):
            DataStructs.ConvertToNumpyArray(efp[i], A[i])
        EX[name] = dict(y=np.array([r[t] for r in sel]), Xfp=A,
                        Xd=np.array([[f(m) for f in DESCS] for m in em], np.float32),
                        fps=efp)

    for mdl in MODELS:
        if mdl in report.get(t, {}):
            print(f"  {mdl:<15} cached", flush=True)
            continue
        t0 = time.time()
        oof = np.full(len(d), np.nan)
        for tr, te in folds:
            oof[te] = fit_predict(mdl, Xfp, Xd, y[tr], tr, te, fps)
        r2i = float(stats.pearsonr(y, oof)[0] ** 2) if np.std(oof) > 0 else 0.0
        rec = {"internal_r2": r2i,
               "internal_RMSE": float(np.sqrt(((y - oof) ** 2).mean())), "external": {}}
        allidx = np.arange(len(d))
        for name in EX:
            e = EX[name]
            Xall_fp = np.vstack([Xfp, e["Xfp"]])
            Xall_d = np.vstack([Xd, e["Xd"]])
            p = fit_predict(mdl, Xall_fp, Xall_d, y, allidx,
                            np.arange(len(d), len(d) + e["y"].size), fps + e["fps"])
            if np.std(p) == 0:
                r2, rho = 0.0, float("nan")
            else:
                r2 = float(stats.pearsonr(e["y"], p)[0] ** 2)
                rho = float(stats.spearmanr(e["y"], p).statistic)
            rec["external"][name] = {
                "n": int(e["y"].size), "r2": r2, "rho": rho,
                "RMSE": float(np.sqrt(((e["y"] - p) ** 2).mean())),
                "R2det": float(1 - ((e["y"] - p) ** 2).sum()
                               / ((e["y"] - e["y"].mean()) ** 2).sum())}
        report.setdefault(t, {})[mdl] = rec
        print(f"  {mdl:<15} internal {r2i:.3f}   external " + "  ".join(
            f"{k.split()[0][:4]} {rec['external'][k]['r2']:.3f}"
            for k in rec["external"]) + f"   [{time.time()-t0:.0f}s]", flush=True)
        json.dump(report, open("model_class_sweep.json", "w"), indent=1)

if any(m not in report.get(t, {}) for t in TARGETS for m in MODELS):
    raise SystemExit("some cells still to run; invoke again per target")

print("\n" + "=" * 94)
print("Internal versus external, by model class (mean over the three external sets)")
print("=" * 94)
print(f"{'model':<16}" + "".join(f"{t:>22}" for t in TARGETS))
print(f"{'':<16}" + "".join(f"{'internal  external':>22}" for _ in TARGETS))
for mdl in MODELS:
    row = f"{mdl:<16}"
    for t in TARGETS:
        r = report[t][mdl]
        ex = np.mean([v["r2"] for v in r["external"].values()])
        row += f"{r['internal_r2']:>13.3f}{ex:>9.3f}"
    print(row)
print("\nwrote model_class_sweep.json")
