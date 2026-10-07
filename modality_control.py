"""Size-matched control for the modality comparison.

The matching-modality models train on fewer compounds than the pooled model, so some of
their external disadvantage could be sample size rather than modality. This refits the
POOLED model on a random subset the same size as each modality model, 15 draws, and
compares. If the size-matched pooled model still beats the modality model, the loss is
about modality (or rather, about separating on it); if it falls to the same level, the
loss was just less data.
"""
import csv, json, math
import numpy as np
import pandas as pd
from collections import defaultdict
from lightgbm import LGBMRegressor
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors as rdmd
from scipy import stats

RDLogger.DisableLog("rdApp.*")
GEN = AllChem.GetMorganGenerator(radius=2, fpSize=2048)
DESCS = [Descriptors.MolWt, Descriptors.MolLogP, Descriptors.TPSA, rdmd.CalcNumHBD,
         rdmd.CalcNumHBA, rdmd.CalcNumRotatableBonds, rdmd.CalcNumAromaticRings,
         rdmd.CalcFractionCSP3, lambda m: m.GetNumHeavyAtoms(), rdmd.CalcNumRings]
T = ("DAT", "NET", "SERT"); E = ""
rng = np.random.default_rng(20261003)


def lgbm(seed=0):
    return LGBMRegressor(n_estimators=250, learning_rate=0.06, num_leaves=31,
                         min_child_samples=10, subsample=0.8, colsample_bytree=0.4,
                         random_state=seed, verbose=-1, n_jobs=2)


def feat(s):
    m = Chem.MolFromSmiles(s)
    if m is None: return None
    fp = GEN.GetFingerprint(m); a = np.zeros(2048, dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, a)
    return np.concatenate([a, [f(m) for f in DESCS]]).astype(np.float32)


def skel(s):
    m = Chem.MolFromSmiles(s)
    if m is None: return None
    fr = Chem.GetMolFrags(m, asMols=True, sanitizeFrags=False)
    parent = max(fr, key=lambda x: x.GetNumHeavyAtoms()) if fr else m
    return Chem.MolToInchiKey(parent).split("-")[0]


st = {r["abbrev"]: r["final"] for r in csv.DictReader(open(E + "struct.csv"))}
v = defaultdict(dict)
for r in csv.DictReader(open(E + "stalberga_parsed.csv")):
    v[r["compound"]][r["target"]] = (r["relation"], r["value"].strip())
STAL = [{"smiles": st[c], **{t: 9.0 - math.log10(float(d[t][1])) for t in T}}
        for c, d in v.items() if len(d) == 3 and all(x[0] == "=" and x[1] for x in d.values())]
PDSP = [{"smiles": r["smiles"], **{t: float(r["p" + t]) for t in T}}
        for r in csv.DictReader(open(E + "pdsp_mat_profiles.csv"))]
CUI = [{"smiles": r["smiles"], **{t: (float(r["p"+t]) if r["p"+t] != "" else None) for t in T}}
       for r in csv.DictReader(open(E + "ejmc2026_cui.csv"))]
SETS = [("Stalberga 2026", STAL, "uptake"), ("PDSP / BindingDB", PDSP, "binding"),
        ("Cui et al. 2026", CUI, "uptake")]
ext_skel = {skel(r["smiles"]) for _, rows, _ in SETS for r in rows} - {None}

lab = pd.read_csv("modality_labels_v2.csv"); pool = pd.read_csv("pooled_labels_v2.csv")
for df in (lab, pool):
    df["skel"] = df.smiles.map(skel)
lab = lab[~lab.skel.isin(ext_skel)].reset_index(drop=True)
pool = pool[~pool.skel.isin(ext_skel)].reset_index(drop=True)

prev = {(r["set"], r["target"]): r for r in json.load(open("modality_external_v2.json"))}
B = "=" * 104
print(B); print("Size-matched control: pooled model refit at the modality model's size")
print(B)
print(f"{'set':<18}{'t':<6}{'n train (mod)':>14}{'n train (pool)':>15}   "
      f"{'pooled full':>12}{'pooled size-matched':>21}{'modality':>10}")
rows = []
for name, data, mod in SETS:
    for t in T:
        key = (name, t)
        if key not in prev: continue
        q = lab[(lab.target == t) & (lab.modality == mod)]
        s = pool[pool.target == t]
        Xq = [feat(x) for x in q.smiles]; okq = [i for i,f in enumerate(Xq) if f is not None]
        Xs = [feat(x) for x in s.smiles]; oks = [i for i,f in enumerate(Xs) if f is not None]
        if len(okq) < 200: continue
        Xs_ = np.vstack([Xs[i] for i in oks]); ys_ = s.p_pooled.values[oks]
        sel = [r for r in data if r.get(t) is not None and feat(r["smiles"]) is not None]
        y = np.array([r[t] for r in sel]); Xe = np.vstack([feat(r["smiles"]) for r in sel])
        r2s = []
        for d in range(15):
            idx = rng.choice(len(ys_), len(okq), replace=False)
            m = lgbm(d).fit(Xs_[idx], ys_[idx])
            r2s.append(float(stats.pearsonr(y, m.predict(Xe))[0] ** 2))
        a = np.array(r2s)
        rows.append(dict(set=name, target=t, n_mod=len(okq), n_pool=len(oks),
                         r2_pool_full=prev[key]["r2_pooled"],
                         r2_pool_sized=float(a.mean()), r2_pool_sized_sd=float(a.std(ddof=1)),
                         r2_mod=prev[key]["r2_mod"]))
        print(f"{name:<18}{t:<6}{len(okq):>14}{len(oks):>15}   "
              f"{prev[key]['r2_pooled']:>12.3f}"
              f"{f'{a.mean():.3f} ± {a.std(ddof=1):.3f}':>21}{prev[key]['r2_mod']:>10.3f}")

d_size = np.array([r["r2_pool_sized"] - r["r2_pool_full"] for r in rows])
d_mod = np.array([r["r2_mod"] - r["r2_pool_full"] for r in rows])
print(f"\n  shrinking the pooled training set to the modality size costs "
      f"{d_size.mean():+.3f} r² on average")
print(f"  switching to the matching-modality model costs      {d_mod.mean():+.3f}")
print(f"  -> the part attributable to MODALITY rather than size: "
      f"{d_mod.mean()-d_size.mean():+.3f}")
json.dump(rows, open("modality_control.json", "w"), indent=1)
print("\nwrote modality_control.json")
