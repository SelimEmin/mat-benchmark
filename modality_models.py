"""Does separating binding from functional uptake help?

The manuscript says it cannot be tested: 'functional records are too few (2.4%) to train
a separate model.' That 2.4% is ChEMBL's assay_type field, which this project showed is
unreliable. Read from the assay text there are 5,601 functional records, and the test is
straightforward.

Two comparisons, both holding the architecture fixed at the pre-registered configuration:

  INTERNAL   scaffold-split CV predicting modality-specific values, with the training
             labels either modality-specific or pooled across modalities. If pooling
             costs accuracy, separating is worth doing.

  EXTERNAL   each external set was measured in one modality -- Stalberga and Cui by
             functional uptake, PDSP by radioligand binding. Score each with the matching
             modality model and with the pooled model.

Overlap with the external sets is removed by InChIKey SKELETON block, not the full key:
ECFP4 ignores stereochemistry, so (S)-citalopram and the racemate have different full
keys and identical fingerprints.
"""
import csv, json, math
from collections import defaultdict
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors as rdmd
from rdkit.Chem.Scaffolds import MurckoScaffold
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


def scaffold(s):
    m = Chem.MolFromSmiles(s)
    try:
        return MurckoScaffold.MurckoScaffoldSmiles(mol=m) if m else ""
    except Exception:
        return ""


def folds(scaf, k=5, seed=0):
    rng = np.random.default_rng(seed)
    gr = {}
    for i, s in enumerate(scaf):
        gr.setdefault(s, []).append(i)
    keys = list(gr); rng.shuffle(keys)
    out = [[] for _ in range(k)]
    for key in sorted(keys, key=lambda x: -len(gr[x])):
        out[int(np.argmin([len(o) for o in out]))].extend(gr[key])
    return [np.array(sorted(o)) for o in out]


def oof(X, y, f):
    p = np.zeros(y.size)
    for te in f:
        tr = np.setdiff1d(np.arange(y.size), te)
        m = lgbm().fit(X[tr], y[tr]); p[te] = m.predict(X[te])
    return p


# ------------------------------------------------------------------ external sets
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
lab["skel"] = lab.smiles.map(skel)
pool["skel"] = pool.smiles.map(skel)
lab = lab[~lab.skel.isin(ext_skel)].reset_index(drop=True)
pool = pool[~pool.skel.isin(ext_skel)].reset_index(drop=True)
print(f"after skeleton-block exclusion: {len(lab)} modality labels, {len(pool)} pooled\n")

SMI = sorted(set(lab.smiles) | set(pool.smiles))
F = {s: feat(s) for s in SMI}
SC = {s: scaffold(s) for s in SMI}
SMI = [s for s in SMI if F[s] is not None]
print(f"featurised {len(SMI)} compounds\n")

# ------------------------------------------------------------------ internal
B = "=" * 96
print(B); print("INTERNAL — predicting modality-specific values, 5-fold scaffold CV")
print(B)
print(f"{'target':<7}{'modality':<10}{'n':>6}   {'trained on that modality':>26}"
      f"{'trained on pooled labels':>26}{'gain':>8}")
res = {}
for t in T:
    for mod in ("binding", "uptake"):
        s = lab[(lab.target == t) & (lab.modality == mod)]
        s = s[s.smiles.isin(F)].reset_index(drop=True)
        if len(s) < 200:
            print(f"{t:<7}{mod:<10}{len(s):>6}   (too few)"); continue
        X = np.vstack([F[x] for x in s.smiles]); y = s.p.values
        f = folds([SC[x] for x in s.smiles])
        p_mod = oof(X, y, f)
        # same compounds and folds, but labels pooled across modalities
        pmap = dict(zip(pool[pool.target == t].smiles, pool[pool.target == t].p_pooled))
        yp = np.array([pmap.get(x, np.nan) for x in s.smiles])
        ok = ~np.isnan(yp)
        p_pool = np.full(y.size, np.nan)
        p_pool[ok] = oof(X[ok], yp[ok], folds([SC[x] for x in s.smiles[ok]]))
        r_mod = stats.pearsonr(y, p_mod)[0] ** 2
        r_pool = stats.pearsonr(y[ok], p_pool[ok])[0] ** 2
        e_mod = float(np.sqrt(((p_mod - y) ** 2).mean()))
        e_pool = float(np.sqrt(((p_pool[ok] - y[ok]) ** 2).mean()))
        res[(t, mod)] = dict(n=int(len(s)), r2_mod=float(r_mod), r2_pool=float(r_pool),
                             rmse_mod=e_mod, rmse_pool=e_pool)
        print(f"{t:<7}{mod:<10}{len(s):>6}   "
              f"{f'r2 {r_mod:.3f}  RMSE {e_mod:.3f}':>26}"
              f"{f'r2 {r_pool:.3f}  RMSE {e_pool:.3f}':>26}{r_mod-r_pool:>+8.3f}")
json.dump({f"{a}|{b}": v for (a, b), v in res.items()},
          open("modality_internal.json", "w"), indent=1)
print("\nwrote modality_internal.json")
