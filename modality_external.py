"""Does scoring an external set with its OWN modality's model beat the pooled model?

Each external set was measured one way: Stalberga and Cui by functional uptake
inhibition, PDSP by radioligand binding. If assay modality matters for transfer, the
matching-modality model should beat the model trained on both pooled together.

This is the test the manuscript says it cannot run.
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
T = ("DAT", "NET", "SERT")
E = ""


def lgbm():
    return LGBMRegressor(n_estimators=250, learning_rate=0.06, num_leaves=31,
                         min_child_samples=10, subsample=0.8, colsample_bytree=0.4,
                         random_state=0, verbose=-1, n_jobs=2)


def feat(s):
    m = Chem.MolFromSmiles(s)
    if m is None:
        return None
    fp = GEN.GetFingerprint(m)
    a = np.zeros(2048, dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, a)
    return np.concatenate([a, [f(m) for f in DESCS]]).astype(np.float32)


def skel(s):
    m = Chem.MolFromSmiles(s)
    if m is None:
        return None
    fr = Chem.GetMolFrags(m, asMols=True, sanitizeFrags=False)
    return Chem.MolToInchiKey(
        max(fr, key=lambda x: x.GetNumHeavyAtoms()) if fr else m).split("-")[0]


st = {r["abbrev"]: r["final"] for r in csv.DictReader(open(E + "struct.csv"))}
v = defaultdict(dict)
for r in csv.DictReader(open(E + "stalberga_parsed.csv")):
    v[r["compound"]][r["target"]] = (r["relation"], r["value"].strip())
STAL = [{"smiles": st[c], **{t: 9.0 - math.log10(float(d[t][1])) for t in T}}
        for c, d in v.items()
        if len(d) == 3 and all(x[0] == "=" and x[1] for x in d.values())]
PDSP = [{"smiles": r["smiles"], **{t: float(r["p" + t]) for t in T}}
        for r in csv.DictReader(open(E + "pdsp_mat_profiles.csv"))]
CUI = [{"smiles": r["smiles"],
        **{t: (float(r["p" + t]) if r["p" + t] != "" else None) for t in T}}
       for r in csv.DictReader(open(E + "ejmc2026_cui.csv"))]
SETS = [("Stalberga 2026", STAL, "uptake"),
        ("PDSP / BindingDB", PDSP, "binding"),
        ("Cui et al. 2026", CUI, "uptake")]
ext_skel = {skel(r["smiles"]) for _, rows, _ in SETS for r in rows} - {None}

lab = pd.read_csv("modality_labels_v2.csv")
pool = pd.read_csv("pooled_labels_v2.csv")
for df in (lab, pool):
    df["skel"] = df.smiles.map(skel)
lab = lab[~lab.skel.isin(ext_skel)].reset_index(drop=True)
pool = pool[~pool.skel.isin(ext_skel)].reset_index(drop=True)

MOD, POOL = {}, {}
for t in T:
    s = pool[pool.target == t]
    Xp = [feat(x) for x in s.smiles]
    ok = [i for i, f in enumerate(Xp) if f is not None]
    POOL[t] = lgbm().fit(np.vstack([Xp[i] for i in ok]), s.p.values[ok]
                         if "p" in s else s.p_pooled.values[ok])
    for mod in ("binding", "uptake"):
        q = lab[(lab.target == t) & (lab.modality == mod)]
        Xq = [feat(x) for x in q.smiles]
        okq = [i for i, f in enumerate(Xq) if f is not None]
        if len(okq) < 200:
            continue
        MOD[(t, mod)] = lgbm().fit(np.vstack([Xq[i] for i in okq]), q.p.values[okq])
    print(f"fitted {t}", flush=True)

B = "=" * 104
print("\n" + B)
print("EXTERNAL — matching-modality model versus pooled model")
print(B)
print(f"{'set':<18}{'modality':<10}{'t':<6}{'n':>4}   "
      f"{'pooled model':>24}{'matching modality':>26}{'Δr²':>8}{'ΔRMSE':>9}")
rows = []
for name, data, mod in SETS:
    for t in T:
        sel = [r for r in data if r.get(t) is not None
               and skel(r["smiles"]) in ext_skel and feat(r["smiles"]) is not None]
        if len(sel) < 5:
            continue
        y = np.array([r[t] for r in sel])
        X = np.vstack([feat(r["smiles"]) for r in sel])
        pp = POOL[t].predict(X)
        key = (t, mod)
        if key not in MOD:
            continue
        pm = MOD[key].predict(X)
        def met(p):
            return (float(stats.pearsonr(y, p)[0] ** 2),
                    float(np.sqrt(((p - y) ** 2).mean())))
        r_p, e_p = met(pp); r_m, e_m = met(pm)
        rows.append(dict(set=name, modality=mod, target=t, n=len(sel),
                         r2_pooled=r_p, r2_mod=r_m, rmse_pooled=e_p, rmse_mod=e_m))
        print(f"{name:<18}{mod:<10}{t:<6}{len(sel):>4}   "
              f"{f'r2 {r_p:.3f}  RMSE {e_p:.2f}':>24}"
              f"{f'r2 {r_m:.3f}  RMSE {e_m:.2f}':>26}{r_m-r_p:>+8.3f}{e_m-e_p:>+9.2f}")

dr = np.array([r["r2_mod"] - r["r2_pooled"] for r in rows])
de = np.array([r["rmse_mod"] - r["rmse_pooled"] for r in rows])
w = stats.wilcoxon(dr) if len(dr) > 5 else None
print(f"\n  mean Δr² {dr.mean():+.3f} (better in {int((dr>0).sum())}/{len(dr)} cells)"
      + (f", Wilcoxon p = {w.pvalue:.3f}" if w else ""))
print(f"  mean ΔRMSE {de.mean():+.3f} log units "
      f"(better in {int((de<0).sum())}/{len(de)} cells)")
json.dump(rows, open("modality_external_v2.json", "w"), indent=1)
print("\nwrote modality_external_v2.json")
