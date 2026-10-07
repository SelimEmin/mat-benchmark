"""Evaluate the Cui 2026 set: overlap checks, then the competency battery."""
import csv, json, pickle, urllib.parse, urllib.request
import numpy as np
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, Descriptors, Crippen
from rdkit.Chem.Scaffolds import MurckoScaffold
from scipy import stats

RDLogger.DisableLog("rdApp.*")
TARGETS = ("DAT", "NET", "SERT")
FLOOR = 0.194
SIG_E = {"DAT": 0.445, "NET": 0.655, "SERT": 0.492}
OFFSET = {"DAT": -0.280, "NET": +0.550, "SERT": -0.090}   # within-document, functional - binding
DESC = [Descriptors.MolWt, Crippen.MolLogP, Descriptors.TPSA, Descriptors.NumHDonors,
        Descriptors.NumHAcceptors, Descriptors.NumRotatableBonds,
        Descriptors.NumAromaticRings, Descriptors.FractionCSP3,
        Descriptors.HeavyAtomCount, Descriptors.RingCount]
GEN = AllChem.GetMorganGenerator(radius=2, fpSize=2048)


def parent(s):
    m = Chem.MolFromSmiles(s)
    if m is None:
        return None
    fr = Chem.GetMolFrags(m, asMols=True, sanitizeFrags=False)
    return max(fr, key=lambda x: x.GetNumHeavyAtoms()) if fr else m


def featurise(s):
    m = parent(s)
    fp = np.asarray(GEN.GetFingerprintAsNumPy(m), dtype=np.float32)
    return np.concatenate([fp, np.array([f(m) for f in DESC], dtype=np.float32)])


rows = list(csv.DictReader(open("ejmc2026_cui.csv")))
store = pickle.load(open("rebuild_models.pkl", "rb"))
models, meta = store["models"], store["meta"]

# ------------------------------------------------------------------- overlap
train_full, train_conn, train_scaf = set(), set(), set()
train_fps, train_y = [], {t: [] for t in TARGETS}
for r in meta:
    m = parent(r["smiles"])
    if m is None:
        continue
    k = Chem.MolToInchiKey(m)
    train_full.add(k)
    train_conn.add(k.split("-")[0])
    train_scaf.add(r["scaffold"])
    train_fps.append(GEN.GetFingerprint(m))
    for t, col, cc in (("DAT", "pDAT", "cens_DAT"), ("NET", "pNET", "cens_NET"),
                       ("SERT", "pSERT", "cens_SERT")):
        train_y[t].append(float(r[col]) if (r[col].strip() and r[cc] == "0") else np.nan)
train_y = {t: np.array(v) for t, v in train_y.items()}
train_mean = {t: float(np.nanmean(train_y[t])) for t in TARGETS}
train_var = {t: float(np.nanvar(train_y[t], ddof=1)) for t in TARGETS}

print("=" * 80)
print("OVERLAP WITH THE TRAINING SET")
print("=" * 80)
n_exact = n_conn = 0
scafs, sims = [], []
for r in rows:
    m = parent(r["smiles"])
    k = Chem.MolToInchiKey(m)
    if k in train_full:
        n_exact += 1
    elif k.split("-")[0] in train_conn:
        n_conn += 1
    scafs.append(MurckoScaffold.MurckoScaffoldSmiles(mol=m))
    s = DataStructs.BulkTanimotoSimilarity(GEN.GetFingerprint(m), train_fps)
    sims.append(max(s))
sims = np.array(sims)
print(f"  exact InChIKey matches      : {n_exact} of {len(rows)}")
print(f"  stereo/salt variants        : {n_conn}")
print(f"  distinct Murcko scaffolds   : {len(set(scafs))}")
print(f"  scaffolds also in training  : {len(set(scafs) & train_scaf)} of {len(set(scafs))}")
print(f"  nearest-neighbour Tanimoto  : min {sims.min():.2f}  median {np.median(sims):.2f}"
      f"  max {sims.max():.2f}")
print(f"  compounds with NN Tanimoto > 0.7 (close analogue in training): "
      f"{int((sims > 0.7).sum())}")

# ------------------------------------------------------- ChEMBL document check
try:
    url = ("https://www.ebi.ac.uk/chembl/api/data/document.json?"
           + urllib.parse.urlencode({"doi": "10.1016/j.ejmech.2026.118771"}))
    with urllib.request.urlopen(url, timeout=60) as fh:
        tc = json.loads(fh.read().decode())["page_meta"]["total_count"]
    print(f"\n  ChEMBL_37 document.json?doi=10.1016/j.ejmech.2026.118771 "
          f"-> total_count {tc}  ({'ABSENT' if tc == 0 else 'PRESENT'})")
except Exception as exc:
    print(f"\n  ChEMBL document check failed: {exc}")

# ------------------------------------------------------------------ evaluate
def pairwise(y, p, gap=0.0):
    i, j = np.triu_indices(y.size, 1)
    dy, dp = y[i] - y[j], p[i] - p[j]
    m = np.abs(dy) > gap
    if m.sum() == 0:
        return float("nan"), 0
    return float((np.sign(dy[m]) == np.sign(dp[m])).mean()), int(m.sum())


def ccc(y, p):
    return 2 * ((y - y.mean()) * (p - p.mean())).mean() / \
        (y.var() + p.var() + (y.mean() - p.mean()) ** 2)


X = np.vstack([featurise(r["smiles"]) for r in rows])
print("\n" + "=" * 80)
print("PERFORMANCE  (rebuilt model -- see the fidelity caveat)")
print("=" * 80)
res = {}
for t in TARGETS:
    keep = [i for i, r in enumerate(rows) if r[f"p{t}"] != ""]
    if len(keep) < 5:
        continue
    y = np.array([float(rows[i][f"p{t}"]) for i in keep])
    p = models[t].predict(X[keep])
    var = y.var(ddof=1)
    r2 = 1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    rmse = float(np.sqrt(((y - p) ** 2).mean()))
    err = np.abs(y - p)
    ceiling = 1 - SIG_E[t] ** 2 / var
    # the model predicts a binding-dominated median; these are functional values,
    # so apply the measured within-document modality offset as a constant correction
    pc = p + OFFSET[t]
    r2c = 1 - ((y - pc) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    rmsec = float(np.sqrt(((y - pc) ** 2).mean()))
    base = float(np.sqrt(((y - train_mean[t]) ** 2).mean()))
    pa, npair = pairwise(y, p, 0.5)
    res[t] = {"n": len(keep), "var": float(var), "var_ratio": float(var / train_var[t]),
              "R2": float(r2), "RMSE": rmse, "bias": float((p - y).mean()),
              "slope": float(np.polyfit(y, p, 1)[0]), "ceiling": float(ceiling),
              "R2_over_ceiling": float(r2 / ceiling) if ceiling > 0 else None,
              "spearman": float(stats.spearmanr(y, p).statistic), "CCC": float(ccc(y, p)),
              "pct_within_floor": float((err <= FLOOR).mean() * 100),
              "pct_within_0_5": float((err <= 0.5).mean() * 100),
              "pairwise_acc_gap0.5": pa, "n_pairs": npair,
              "baseline_mean_RMSE": base,
              "R2_modality_corrected": float(r2c), "RMSE_modality_corrected": rmsec,
              "modality_offset_applied": OFFSET[t]}
    print(f"\n  --- {t}  (n={len(keep)}) ---")
    print(f"    variance {var:.3f} ({var/train_var[t]:.2f}x training)   "
          f"ceiling {ceiling:.3f}")
    print(f"    R^2 {r2:+.3f}   RMSE {rmse:.3f}   bias {(p-y).mean():+.3f}   "
          f"slope {np.polyfit(y,p,1)[0]:.3f}")
    print(f"    Spearman {stats.spearmanr(y,p).statistic:+.3f}   CCC {ccc(y,p):.3f}   "
          f"pairwise accuracy (>0.5 log) {100*pa:.1f}% of {npair}")
    print(f"    within floor {err.le if False else (err<=FLOOR).mean()*100:.0f}%   "
          f"within 0.5 log {(err<=0.5).mean()*100:.0f}%   "
          f"baseline-mean RMSE {base:.3f}")
    print(f"    applying the measured {t} modality offset ({OFFSET[t]:+.3f}): "
          f"R^2 {r2:+.3f} -> {r2c:+.3f},  RMSE {rmse:.3f} -> {rmsec:.3f}")

json.dump({"overlap": {"n": len(rows), "exact": n_exact, "stereo": n_conn,
                       "scaffolds": len(set(scafs)),
                       "scaffolds_in_training": len(set(scafs) & train_scaf),
                       "nn_tanimoto_median": float(np.median(sims)),
                       "n_nn_above_0.7": int((sims > 0.7).sum())},
           "performance": res},
          open("ejmc2026_eval.json", "w"), indent=2)
print("\nwrote ejmc2026_eval.json")
