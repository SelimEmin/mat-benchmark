"""The eight-diagnostic battery on the production model, internal and external.

Same diagnostics as competency2.py so the two are directly comparable, with one
change forced by the new configuration: every external set is scored against the
matching modality head, because that is the modality its measurements were made in.

Ceilings come from ceiling_correction.json. The published model is scored against
ceiling C (modality unknown) and the production model against ceiling B (modality
known), because those are the tasks they respectively perform.
"""
import csv, json, math, pickle
import numpy as np
import pandas as pd
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


def feat(s):
    m = Chem.MolFromSmiles(s)
    if m is None:
        return None
    fp = GEN.GetFingerprint(m)
    a = np.zeros(2048, dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, a)
    return np.concatenate([a, [f(m) for f in DESCS]]).astype(np.float32)


def pkey(s):
    """InChIKey SKELETON block (first 14 chars) of the largest fragment.

    The full InChIKey distinguishes stereoisomers and protonation states; ECFP4 as
    generated here does not, so (S)-citalopram and racemic citalopram carry different
    InChIKeys but IDENTICAL fingerprints. Excluding on the full key therefore leaves
    Tanimoto-1.00 near-duplicates in training, which is leakage. The skeleton block
    collapses them.
    """
    m = Chem.MolFromSmiles(s)
    if m is None:
        return None
    fr = Chem.GetMolFrags(m, asMols=True, sanitizeFrags=False)
    return Chem.MolToInchiKey(
        max(fr, key=lambda x: x.GetNumHeavyAtoms()) if fr else m).split("-")[0]


def diagnostics(y, p, sims, ceil, train_var):
    r = stats.pearsonr(y, p)
    r2 = float(r[0] ** 2)
    return {"n": int(y.size), "var": float(y.var(ddof=1)),
            "var_ratio": float(y.var(ddof=1) / train_var),
            "nn_tanimoto_median": float(np.median(sims)),
            "rho": float(stats.spearmanr(y, p).statistic),
            "r2": r2, "p": float(r[1]),
            "R2det": float(1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()),
            "slope": float(np.polyfit(y, p, 1)[0]), "bias": float((p - y).mean()),
            "RMSE": float(np.sqrt(((y - p) ** 2).mean())),
            "ceiling": float(ceil), "pct_ceiling": float(100 * r2 / ceil)}


store = pickle.load(open("production_models.pkl", "rb"))
models, lab = store["models"], store["labels"]
cc = json.load(open("ceiling_correction.json"))
bench = set(pd.read_csv("mat_profiles.csv").inchikey)

# ---------------------------------------------------------------- external sets
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

report = {}
for t in TARGETS:
    d = lab[(lab.target == t) & lab.p_pool.notna()].reset_index(drop=True)
    F = [feat(s) for s in d.smiles]
    keep = [i for i, f in enumerate(F) if f is not None]
    d = d.iloc[keep].reset_index(drop=True)
    train_fps = [GEN.GetFingerprint(Chem.MolFromSmiles(s)) for s in d.smiles]
    train_keys = {k.split("-")[0] for k in d.ik}
    train_var = float(d.p_pool.var(ddof=1))
    ceilB = cc[t]["ceiling_modality_known"]

    for name, rows, modality in SETS:
        sel = [r for r in rows if r.get(t) is not None and feat(r["smiles"]) is not None
               and pkey(r["smiles"]) not in train_keys]
        if len(sel) < 5:
            continue
        sims_all = np.array([max(DataStructs.BulkTanimotoSimilarity(
            GEN.GetFingerprint(Chem.MolFromSmiles(r["smiles"])), train_fps))
            for r in sel])
        nd = int((sims_all >= 0.99).sum())
        if nd:
            print(f"  [{name} {t}] dropping {nd} compound(s) with a Tanimoto-0.99+ "
                  f"training neighbour that survived the skeleton match")
            sel = [r for r, s_ in zip(sel, sims_all) if s_ < 0.99]
            if len(sel) < 5:
                continue
        y = np.array([r[t] for r in sel])
        X = np.vstack([feat(r["smiles"]) for r in sel])
        flag = np.full((len(sel), 1), 1.0 if modality == "Ki" else 0.0, np.float32)
        p = models[t].predict(np.hstack([X, flag]))
        sims = np.array([max(DataStructs.BulkTanimotoSimilarity(
            GEN.GetFingerprint(Chem.MolFromSmiles(r["smiles"])), train_fps))
            for r in sel])
        rec = diagnostics(y, p, sims, ceilB, train_var)
        rec["modality"] = modality
        report.setdefault(name, {})[t] = rec

json.dump(report, open("production_external.json", "w"), indent=1)

B = "=" * 100
print(B)
print("Production model on the three external sets, each scored against its own modality")
print(B)
print(f"  {'set':<18}{'t':<6}{'n':>4}{'mod':>6}{'var÷tr':>8}{'NN':>6}{'rho':>8}"
      f"{'r2':>8}{'p':>9}{'R2det':>8}{'slope':>7}{'bias':>7}{'%ceil':>7}")
for name, _, _ in SETS:
    for t in TARGETS:
        if t not in report.get(name, {}):
            continue
        s = report[name][t]
        print(f"  {name:<18}{t:<6}{s['n']:>4}{s['modality']:>6}{s['var_ratio']:>8.2f}"
              f"{s['nn_tanimoto_median']:>6.2f}{s['rho']:>+8.3f}{s['r2']:>8.3f}"
              f"{s['p']:>9.4f}{s['R2det']:>+8.3f}{s['slope']:>7.3f}{s['bias']:>+7.3f}"
              f"{s['pct_ceiling']:>7.0f}")
    print()
print("wrote production_external.json")
