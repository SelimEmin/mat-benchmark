"""Reproduce both published external results exactly, from the deposited artefacts.

Two numbers in the manuscript could not be regenerated from the deposit. Both are now
accounted for, and both reduce to a statement that was missing from Methods.

(1) THE PRE-REGISTERED TEST, n = 42.
    A rebuild gives 43. The cause is a single structure: struct.csv records Cocaine as
    benzoylecgonine, its inactive hydrolysis metabolite. The training set contains
    (-)-cocaine under the stereo-specific InChIKey ZPUCINDJVBIVPJ-LJISPDSOSA-N, so the
    correct structure is excluded as a training overlap and the wrong one is not. One
    data error therefore produced both the wrong molecule and the wrong compound count.
    Correcting the structure restores n = 42 and the published R^2 exactly.

    Note the exclusion must be applied on the InChIKey SKELETON BLOCK, not the full key:
    ECFP4 ignores stereochemistry, so a flat-drawn cocaine has a different full key from
    the training entry and identical fingerprints. The project met this failure mode once
    before, with (S)-citalopram.

(2) THE EXTENDED SET, R^2 = 0.183.
    Two things were unstated. The DAT/SERT log-ratio is modelled DIRECTLY -- a single
    LightGBM trained on the log_DAT_SERT column -- not differenced from the two
    per-target models, which gives 0.408. And the pooled statistic is computed on the
    107 compounds from the three Basel/Linkoping sources; the seven Eshleman compounds
    are reported separately by source and excluded from the pooled figure. The manuscript
    describes the set as 114 compounds at that point, which is the set size, not the n
    behind the pooled number.
"""
import csv, math, pickle
from collections import defaultdict
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors as rdmd
from scipy import stats

RDLogger.DisableLog("rdApp.*")
E = "../"
GEN = AllChem.GetMorganGenerator(radius=2, fpSize=2048)
DESCS = [Descriptors.MolWt, Descriptors.MolLogP, Descriptors.TPSA, rdmd.CalcNumHBD,
         rdmd.CalcNumHBA, rdmd.CalcNumRotatableBonds, rdmd.CalcNumAromaticRings,
         rdmd.CalcFractionCSP3, lambda m: m.GetNumHeavyAtoms(), rdmd.CalcNumRings]
T = ("DAT", "NET", "SERT"); COL = {"DAT": "pDAT", "NET": "pNET", "SERT": "pSERT"}
COCAINE = "COC(=O)[C@H]1[C@@H](OC(=O)c2ccccc2)C[C@@H]2CC[C@H]1N2C"   # (-)-cocaine


def feat(s):
    m = Chem.MolFromSmiles(s)
    if m is None:
        return None
    fp = GEN.GetFingerprint(m); a = np.zeros(2048, dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, a)
    return np.concatenate([a, [f(m) for f in DESCS]]).astype(np.float32)


def skel(s):
    m = Chem.MolFromSmiles(s)
    return Chem.MolToInchiKey(m).split("-")[0] if m else None


store = pickle.load(open(E + "faithful_models.pkl", "rb"))
M, fr = store["models"], store["frame"]
X_tr = np.load(E + "faithful_X.npy")
TR_SKEL = {k.split("-")[0] for k in fr.inchikey}

# ------------------------------------------------------------ (1) pre-registered test
st = {r["abbrev"]: r["final"] for r in csv.DictReader(open(E + "struct.csv"))}
st["Cocaine"] = COCAINE                                   # the correction
v = defaultdict(dict)
for r in csv.DictReader(open(E + "stalberga_parsed.csv")):
    v[r["compound"]][r["target"]] = (r["relation"], r["value"].strip())
cand = [(c, {t: 9.0 - math.log10(float(d[t][1])) for t in T}) for c, d in v.items()
        if len(d) == 3 and all(x[0] == "=" and x[1] for x in d.values())]
excluded = [c for c, _ in cand if skel(st[c]) in TR_SKEL]
rows = [(c, d) for c, d in cand if skel(st[c]) not in TR_SKEL]

B = "=" * 78
print(B); print("(1) PRE-REGISTERED EXTERNAL TEST"); print(B)
print(f"  candidates with complete uncensored triples : {len(cand)}")
print(f"  excluded as training overlaps ({len(excluded)})            : "
      f"{', '.join(sorted(excluded))}")
print(f"  retained                                    : {len(rows)}\n")
X = np.vstack([feat(st[c]) for c, _ in rows])
PUB = {"DAT": 0.459, "NET": 0.015, "SERT": 0.058}
print(f"  {'target':<7}{'rebuilt':>10}{'published':>12}{'difference':>12}")
for t in T:
    y = np.array([d[t] for _, d in rows]); p = M[COL[t]].predict(X)
    r2 = stats.pearsonr(y, p)[0] ** 2
    print(f"  {t:<7}{r2:>10.3f}{PUB[t]:>12.3f}{r2 - PUB[t]:>+12.4f}")

# ------------------------------------------------------------ (2) extended set
print("\n" + B); print("(2) EXTENDED EXTERNAL SET, DAT/SERT LOG-RATIO"); print(B)
d = pd.read_csv("extended_set.csv")
Xl = [feat(s) for s in d.smiles]; ok = [i for i, f in enumerate(Xl) if f is not None]
d = d.iloc[ok].reset_index(drop=True); Xe = np.vstack([Xl[i] for i in ok])
ratio = LGBMRegressor(n_estimators=250, learning_rate=0.06, num_leaves=31,
                      min_child_samples=10, colsample_bytree=0.4,
                      random_state=0, verbose=-1, n_jobs=2)
ratio.fit(X_tr, fr.log_DAT_SERT.values)
pred, y = ratio.predict(Xe), d.log_DAT_SERT.values
sel = (d.source != "Eshleman2017").values
yy, pp = y[sel], pred[sel]
r = stats.pearsonr(yy, pp)
ccc = 2*np.cov(yy, pp, ddof=1)[0, 1] / (yy.var(ddof=1)+pp.var(ddof=1)+(yy.mean()-pp.mean())**2)
print(f"  set size {len(y)}; pooled statistic computed on {int(sel.sum())} "
      f"(three sources, Eshleman reported separately)\n")
print(f"  {'':<10}{'rebuilt':>10}{'published':>12}")
for lab, got, pub in (("R^2", r[0]**2, 0.183), ("r", r[0], 0.427),
                      ("RMSE", float(np.sqrt(((pp-yy)**2).mean())), 1.335),
                      ("CCC", ccc, 0.304)):
    print(f"  {lab:<10}{got:>10.3f}{pub:>12.3f}")
print(f"\n  for contrast, differencing the two per-target models gives R^2 = "
      f"{stats.pearsonr(yy, (M['pDAT'].predict(Xe)-M['pSERT'].predict(Xe))[sel])[0]**2:.3f}")
