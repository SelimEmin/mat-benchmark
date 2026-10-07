"""Regenerate the quantities the manuscript figures plot.

Figures 1-10 in the deposit are 911-byte blank stubs; only 11-13 were ever produced.
This recomputes what each one needs, from the pre-registered model and the
pre-registered external protocol, so the figures and the tables come from one source.

Overlap exclusion is on the InChIKey *skeleton block*, matching repro/reproduce.py:
ECFP4 ignores stereochemistry, so a stereoisomer of a training compound carries a
different full key and an identical fingerprint. Cocaine takes the stereo-specific
(-)-cocaine structure established in the structure QC; with the flat SMILES the
skeleton block is -UHFFFAOYSA- and the overlap filter misses it, which is what
previously inflated the set to 43.
"""
import csv, json, math, pickle
import numpy as np
from collections import defaultdict
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors as rdmd
from scipy import stats

RDLogger.DisableLog("rdApp.*")
E = ""
GEN = AllChem.GetMorganGenerator(radius=2, fpSize=2048)
DESCS = [Descriptors.MolWt, Descriptors.MolLogP, Descriptors.TPSA, rdmd.CalcNumHBD,
         rdmd.CalcNumHBA, rdmd.CalcNumRotatableBonds, rdmd.CalcNumAromaticRings,
         rdmd.CalcFractionCSP3, lambda m: m.GetNumHeavyAtoms(), rdmd.CalcNumRings]
T = ("DAT", "NET", "SERT"); COL = {"DAT": "pDAT", "NET": "pNET", "SERT": "pSERT"}
COCAINE = "COC(=O)[C@H]1[C@@H](OC(=O)c2ccccc2)C[C@@H]2CC[C@H]1N2C"   # (-)-cocaine


def feat(s):
    m = Chem.MolFromSmiles(s)
    if m is None: return None
    fp = GEN.GetFingerprint(m); a = np.zeros(2048, dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, a)
    return np.concatenate([a, [f(m) for f in DESCS]]).astype(np.float32)


def skel(s):
    m = Chem.MolFromSmiles(s)
    return Chem.MolToInchiKey(m).split("-")[0] if m else None


store = pickle.load(open(E + "faithful_models.pkl", "rb"))
models, frame = store["models"], store["frame"]
TR_SKEL = {k.split("-")[0] for k in frame.inchikey}

st = {r["abbrev"]: r["final"] for r in csv.DictReader(open(E + "struct.csv"))}
# apply the cocaine correction established in the structure QC
st["Cocaine"] = COCAINE
v = defaultdict(dict)
for r in csv.DictReader(open(E + "stalberga_parsed.csv")):
    v[r["compound"]][r["target"]] = (r["relation"], r["value"].strip())
rows = [(c, {t: 9.0 - math.log10(float(d[t][1])) for t in T}) for c, d in v.items()
        if len(d) == 3 and all(x[0] == "=" and x[1] for x in d.values())]
excluded = sorted(c for c, _ in rows if skel(st[c]) in TR_SKEL)
rows = [(c, d) for c, d in rows
        if skel(st[c]) not in TR_SKEL and feat(st[c]) is not None]
print(f"excluded as training overlaps ({len(excluded)}): {', '.join(excluded)}")
print(f"pre-registered external set after skeleton-block exclusion: {len(rows)} compounds")


def ccc(y, p):
    sy, sp = y.std(ddof=1), p.std(ddof=1)
    return float(2 * np.cov(y, p, ddof=1)[0, 1] /
                 (sy ** 2 + sp ** 2 + (y.mean() - p.mean()) ** 2))


X = np.vstack([feat(st[c]) for c, _ in rows])
out = {"n": len(rows), "targets": {}}
for t in T:
    y = np.array([d[t] for _, d in rows]); p = models[COL[t]].predict(X)
    # the reference distribution is the UNCENSORED training set: censored values are
    # bounds, not potencies, and Table 14 and Figure 13 both normalise against it
    tr = frame[COL[t]].values[frame["cens_" + t].values == 0]
    r2 = float(stats.pearsonr(y, p)[0] ** 2)
    boot = []
    rng = np.random.default_rng(0)
    for _ in range(2000):
        i = rng.integers(0, y.size, y.size)
        if y[i].std() == 0 or p[i].std() == 0: continue
        boot.append(stats.pearsonr(y[i], p[i])[0] ** 2)
    out["targets"][t] = dict(
        y=y.tolist(), p=p.tolist(), r2=r2,
        ci=[float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
        rmse=float(np.sqrt(((p - y) ** 2).mean())), ccc=ccc(y, p),
        slope=float(np.polyfit(y, p, 1)[0]), bias=float((p - y).mean()),
        train_q=[float(np.percentile(tr, q)) for q in (25, 50, 75)],
        train=frame[COL[t]].values.tolist(), train_unc=tr.tolist())
    print(f"  {t:<5} r2 {r2:.3f} [{out['targets'][t]['ci'][0]:.3f},"
          f"{out['targets'][t]['ci'][1]:.3f}]  RMSE {out['targets'][t]['rmse']:.3f}  "
          f"CCC {out['targets'][t]['ccc']:.3f}  slope {out['targets'][t]['slope']:.3f}")

z = np.load("internal_full.npz", allow_pickle=True)
out["internal_seeds"] = {t: z["R2"].item()[COL[t]] for t in T}
out["internal_rmse"] = {t: z["RM"].item()[COL[t]] for t in T}
json.dump(out, open("figgen/figdata.json", "w"))
print("\nwrote figgen/figdata.json")
