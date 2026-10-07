"""Is the internal gain from dropping the intersection real, or near-duplicate leakage?

pool_all trains on ~1,200-3,000 compounds that pool_bench does not. If some of those
are near-duplicates of HELD-OUT compounds -- stereoisomers, salts, close analogues that
Murcko scaffold splitting does not separate -- then pool_all is partly being tested on
chemistry it has already seen, and the gain is an artefact.

Test: for every held-out compound, the nearest-neighbour Tanimoto to each arm's own
training pool. Then recompute the gain on the subset with NO near neighbour in either.
"""
import numpy as np, pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold
from scipy import stats
RDLogger.DisableLog("rdApp.*")
exec(open("production_model.py").read().split("lab = pd.read_csv")[0].split('"""',2)[2])

lab = pd.read_csv("modality_labels.csv")
bench = set(pd.read_csv("mat_profiles.csv").inchikey)
print(f"{'target':<6}{'NN>=0.99 to pool_bench':>24}{'NN>=0.99 to pool_all':>22}"
      f"{'median NN bench':>17}{'median NN all':>15}")
RES = {}
for t in TARGETS:
    d = lab[(lab.target == t) & lab.p_pool.notna()].reset_index(drop=True)
    X, ok = featurise(d.smiles.values)
    d = d.iloc[ok].reset_index(drop=True)
    d["scaffold"] = [MurckoScaffold.MurckoScaffoldSmiles(s) if
                     Chem.MolFromSmiles(s) is not None else "" for s in d.smiles]
    fps = [GEN.GetFingerprint(Chem.MolFromSmiles(s)) for s in d.smiles]
    inb = d.ik.isin(bench).values
    yp, yk, yi = d.p_pool.values, d.p_Ki.values, d.p_IC50.values
    folds = scaffold_folds_seeded(d.scaffold.values, NFOLD, 0)
    nnB = np.full(len(d), np.nan); nnA = np.full(len(d), np.nan)
    pub = np.full(len(d), np.nan); pk = np.full(len(d), np.nan); pi = np.full(len(d), np.nan)
    for tr, te in folds:
        selB = tr[inb[tr] & np.isfinite(yp[tr])]
        selA = tr[np.isfinite(yp[tr])]
        fB = [fps[i] for i in selB]; fA = [fps[i] for i in selA]
        for i in te:
            nnB[i] = max(DataStructs.BulkTanimotoSimilarity(fps[i], fB))
            nnA[i] = max(DataStructs.BulkTanimotoSimilarity(fps[i], fA))
        pub[te] = lgbm().fit(X[selB], yp[selB]).predict(X[te])
        ki, ic = tr[np.isfinite(yk[tr])], tr[np.isfinite(yi[tr])]
        Xm = np.vstack([np.hstack([X[ki], np.ones((ki.size,1),np.float32)]),
                        np.hstack([X[ic], np.zeros((ic.size,1),np.float32)])])
        mf = lgbm().fit(Xm, np.concatenate([yk[ki], yi[ic]]))
        pk[te] = mf.predict(np.hstack([X[te], np.ones((te.size,1),np.float32)]))
        pi[te] = mf.predict(np.hstack([X[te], np.zeros((te.size,1),np.float32)]))
    print(f"{t:<6}{100*(nnB>=0.99).mean():>23.1f}%{100*(nnA>=0.99).mean():>21.1f}%"
          f"{np.median(nnB):>17.3f}{np.median(nnA):>15.3f}")
    RES[t] = (yk, yi, pub, pk, pi, nnA, nnB)

def r2(y,p,m):
    k = m & np.isfinite(y) & np.isfinite(p)
    return float(stats.pearsonr(y[k],p[k])[0]**2), int(k.sum())
print("\n" + "="*86)
print("Internal gain, restricted to held-out compounds with NO near neighbour")
print("="*86)
print(f"{'target':<6}{'subset':<22}{'n':>6}{'published':>11}{'production':>12}{'gain':>9}")
for t in TARGETS:
    yk, yi, pub, pk, pi, nnA, nnB = RES[t]
    for lab_, m in (("all held-out", np.ones(len(nnA), bool)),
                    ("NN < 0.99", nnA < 0.99),
                    ("NN < 0.90", nnA < 0.90),
                    ("NN < 0.70", nnA < 0.70)):
        a, n1 = r2(yk, pub, m); b, _ = r2(yk, pk, m)
        c, n2 = r2(yi, pub, m); e, _ = r2(yi, pi, m)
        print(f"{t:<6}{lab_+' (Ki)':<22}{n1:>6}{a:>11.3f}{b:>12.3f}{b-a:>+9.3f}")
        print(f"{'':<6}{lab_+' (IC50)':<22}{n2:>6}{c:>11.3f}{e:>12.3f}{e-c:>+9.3f}")
    print()
