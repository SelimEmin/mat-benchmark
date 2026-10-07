"""Do the internal gains transfer to the three external sets?

Internal cross-validation says dropping the all-three-target intersection is worth
+0.12 to +0.26 R^2 and modality separation is worth almost nothing. Internal gains are
exactly the kind of thing that fails to transfer, so they have to be re-tested outside.

Each external set has ONE modality, which is what makes this a real test of the
modality arms rather than a comparison of medians:
    Stalberga 2026   uptake IC50   -> the IC50 head
    PDSP / BindingDB binding Ki    -> the Ki head
    Cui et al. 2026  uptake IC50   -> the IC50 head

Models are fitted once on everything, with external compounds removed by parent
InChIKey first. The exclusion is recomputed against the enlarged training universe,
which contains compounds the published benchmark did not.
"""
import csv, json, math
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from collections import defaultdict
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors as rdmd
from scipy import stats

RDLogger.DisableLog("rdApp.*")
GEN = AllChem.GetMorganGenerator(radius=2, fpSize=2048)
DESCS = [Descriptors.MolWt, Descriptors.MolLogP, Descriptors.TPSA, rdmd.CalcNumHBD,
         rdmd.CalcNumHBA, rdmd.CalcNumRotatableBonds, rdmd.CalcNumAromaticRings,
         rdmd.CalcFractionCSP3, lambda m: m.GetNumHeavyAtoms(), rdmd.CalcNumRings]
TARGETS = ("DAT", "NET", "SERT")
SIG_E = {"DAT": 0.445, "NET": 0.655, "SERT": 0.492}


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


def pkey(s):
    m = Chem.MolFromSmiles(s)
    if m is None:
        return None
    fr = Chem.GetMolFrags(m, asMols=True, sanitizeFrags=False)
    return Chem.MolToInchiKey(max(fr, key=lambda x: x.GetNumHeavyAtoms()) if fr else m)


# ------------------------------------------------------------------ external sets
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
SETS = [("Stalberga 2026", STAL, "IC50"), ("PDSP / BindingDB", PDSP, "Ki"),
        ("Cui et al. 2026", CUI, "IC50")]

lab = pd.read_csv("modality_labels.csv")
bench = set(pd.read_csv("mat_profiles.csv").inchikey)
ext_keys = {pkey(r["smiles"]) for _, rows, _ in SETS for r in rows}
ext_keys.discard(None)

report = {}
for t in TARGETS:
    d = lab[(lab.target == t) & lab.p_pool.notna()].reset_index(drop=True)
    F = [feat(s) for s in d.smiles]
    keep = [i for i, f in enumerate(F) if f is not None]
    d, X = d.iloc[keep].reset_index(drop=True), np.vstack([F[i] for i in keep])
    drop = d.ik.isin(ext_keys).values
    d, X = d[~drop].reset_index(drop=True), X[~drop]
    inb = d.ik.isin(bench).values
    yp, yk, yi = d.p_pool.values, d.p_Ki.values, d.p_IC50.values
    print(f"{t}: training universe {len(d)} after removing "
          f"{int(drop.sum())} external compounds ({int(inb.sum())} in benchmark)")

    mb = lgbm().fit(X[inb & np.isfinite(yp)], yp[inb & np.isfinite(yp)])
    ma = lgbm().fit(X[np.isfinite(yp)], yp[np.isfinite(yp)])
    ki, ic = np.isfinite(yk), np.isfinite(yi)
    Xm = np.vstack([np.hstack([X[ki], np.ones((ki.sum(), 1), np.float32)]),
                    np.hstack([X[ic], np.zeros((ic.sum(), 1), np.float32)])])
    mf = lgbm().fit(Xm, np.concatenate([yk[ki], yi[ic]]))
    sk = lgbm().fit(X[ki], yk[ki])
    si = lgbm().fit(X[ic], yi[ic])

    for name, rows, modality in SETS:
        sel = [r for r in rows if r.get(t) is not None and feat(r["smiles"]) is not None
               and pkey(r["smiles"]) not in set(d.ik)]
        if len(sel) < 5:
            continue
        y = np.array([r[t] for r in sel])
        Xe = np.vstack([feat(r["smiles"]) for r in sel])
        flag = np.ones((len(sel), 1), np.float32) if modality == "Ki" \
            else np.zeros((len(sel), 1), np.float32)
        preds = {"pool_bench": mb.predict(Xe), "pool_all": ma.predict(Xe),
                 "mod_feat": mf.predict(np.hstack([Xe, flag])),
                 "separate": (sk if modality == "Ki" else si).predict(Xe)}
        var = float(y.var(ddof=1))
        ceil = 1 - SIG_E[t] ** 2 / var
        for arm, p in preds.items():
            r = stats.pearsonr(y, p)
            report.setdefault(name, {}).setdefault(t, {})[arm] = {
                "n": len(sel), "modality": modality, "r2": float(r[0] ** 2),
                "p": float(r[1]),
                "R2det": float(1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()),
                "slope": float(np.polyfit(y, p, 1)[0]), "bias": float((p - y).mean()),
                "rho": float(stats.spearmanr(y, p).statistic),
                "RMSE": float(np.sqrt(((y - p) ** 2).mean())),
                "pct_ceiling": float(100 * r[0] ** 2 / ceil)}

json.dump(report, open("modality_external.json", "w"), indent=1)
print()
for name, rows, modality in SETS:
    print("=" * 92)
    print(f"{name}   ({modality} -- matched to the {modality} head)")
    print("=" * 92)
    print(f"  {'target':<6}{'n':>4}{'arm':<13}{'r2':>8}{'p':>9}{'R2det':>9}"
          f"{'slope':>8}{'bias':>8}{'rho':>8}{'% ceil':>8}")
    for t in TARGETS:
        if t not in report.get(name, {}):
            continue
        for arm, s in report[name][t].items():
            print(f"  {t:<6}{s['n']:>4}{arm:<13}{s['r2']:>8.3f}{s['p']:>9.4f}"
                  f"{s['R2det']:>9.3f}{s['slope']:>8.3f}{s['bias']:>8.3f}"
                  f"{s['rho']:>8.3f}{s['pct_ceiling']:>8.0f}")
        print()
print("wrote modality_external.json")
