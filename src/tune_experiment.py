"""The last accuracy lever: hyperparameters, which were never tuned.

The published settings (250 trees, lr 0.06, 31 leaves, colsample 0.4) appear to be
defaults carried through the whole project. A random search over 15 configurations,
5-fold scaffold CV at seed 0, then the winner re-run over 5 seeds so the comparison
against the production configuration is paired and honest.
"""
import json, time, warnings
import numpy as np, pandas as pd
from lightgbm import LGBMRegressor
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors as rdmd
from rdkit.Chem.Scaffolds import MurckoScaffold
from scipy import stats
warnings.filterwarnings("ignore"); RDLogger.DisableLog("rdApp.*")
GEN = AllChem.GetMorganGenerator(radius=2, fpSize=2048)
DESCS = [Descriptors.MolWt, Descriptors.MolLogP, Descriptors.TPSA, rdmd.CalcNumHBD,
         rdmd.CalcNumHBA, rdmd.CalcNumRotatableBonds, rdmd.CalcNumAromaticRings,
         rdmd.CalcFractionCSP3, lambda m: m.GetNumHeavyAtoms(), rdmd.CalcNumRings]
TARGETS = ("DAT","NET","SERT"); NFOLD = 5
BASE = dict(n_estimators=250, learning_rate=0.06, num_leaves=31, min_child_samples=10,
            subsample=0.8, colsample_bytree=0.4, random_state=0, verbose=-1, n_jobs=2)

def folds_seeded(sc, n=5, seed=0):
    rng = np.random.default_rng(seed); g = {}
    for i, s in enumerate(pd.Series(sc).fillna("").values): g.setdefault(s, []).append(i)
    k = list(g); rng.shuffle(k); k.sort(key=lambda x: -len(g[x]))
    f = [[] for _ in range(n)]; o = list(range(n))
    for j, kk in enumerate(k):
        if j % n == 0: rng.shuffle(o)
        f[o[j % n]].extend(g[kk])
    return [(np.array(sorted(sum((f[j] for j in range(n) if j != i), []))),
             np.array(sorted(f[i]))) for i in range(n)]

lab = pd.read_csv("modality_labels.csv"); D = {}
for t in TARGETS:
    d = lab[(lab.target==t)&lab.p_pool.notna()].reset_index(drop=True)
    mols=[Chem.MolFromSmiles(s) for s in d.smiles]
    ok=[i for i,m in enumerate(mols) if m is not None]
    d=d.iloc[ok].reset_index(drop=True); mols=[mols[i] for i in ok]
    d["scaffold"]=[MurckoScaffold.MurckoScaffoldSmiles(mol=m) for m in mols]
    A=np.zeros((len(mols),2048),np.int8)
    for i,m in enumerate(mols): DataStructs.ConvertToNumpyArray(GEN.GetFingerprint(m),A[i])
    X=np.hstack([A,np.array([[f(m) for f in DESCS] for m in mols],np.float32)]).astype(np.float32)
    D[t]=(d, X, d.p_Ki.values, d.p_IC50.values)

def run(t, params, seed):
    d, X, yk, yi = D[t]
    pk=np.full(len(d),np.nan); pi=np.full(len(d),np.nan)
    for tr,te in folds_seeded(d.scaffold.values,NFOLD,seed):
        ki,ic=tr[np.isfinite(yk[tr])],tr[np.isfinite(yi[tr])]
        Xm=np.vstack([np.hstack([X[ki],np.ones((ki.size,1),np.float32)]),
                      np.hstack([X[ic],np.zeros((ic.size,1),np.float32)])])
        m=LGBMRegressor(**params).fit(Xm,np.concatenate([yk[ki],yi[ic]]))
        pk[te]=m.predict(np.hstack([X[te],np.ones((te.size,1),np.float32)]))
        pi[te]=m.predict(np.hstack([X[te],np.zeros((te.size,1),np.float32)]))
    a,b=np.isfinite(yk)&np.isfinite(pk), np.isfinite(yi)&np.isfinite(pi)
    y=np.concatenate([yk[a],yi[b]]); p=np.concatenate([pk[a],pi[b]])
    return float(stats.pearsonr(y,p)[0]**2)

rng=np.random.default_rng(3); cands=[]
for _ in range(15):
    cands.append(dict(BASE, n_estimators=int(rng.choice([250,500,800,1200])),
        learning_rate=float(rng.choice([0.02,0.03,0.05,0.08])),
        num_leaves=int(rng.choice([15,31,63,127])),
        min_child_samples=int(rng.choice([5,10,20,40])),
        colsample_bytree=float(rng.choice([0.2,0.3,0.4,0.6])),
        subsample=float(rng.choice([0.7,0.8,1.0])),
        reg_lambda=float(rng.choice([0.0,1.0,5.0]))))
best={}; t0=time.time()
for t in TARGETS:
    sc=[(run(t,c,0),i) for i,c in enumerate(cands)]
    b=max(sc); best[t]=cands[b[1]]
    print(f"{t}: baseline(seed0) {run(t,BASE,0):.3f}  best-of-15 {b[0]:.3f}  "
          f"cfg {{n={best[t]['n_estimators']}, lr={best[t]['learning_rate']}, "
          f"leaves={best[t]['num_leaves']}, mcs={best[t]['min_child_samples']}, "
          f"cs={best[t]['colsample_bytree']}, L2={best[t]['reg_lambda']}}}  "
          f"[{(time.time()-t0)/60:.1f} min]", flush=True)
print("\npaired over 5 seeds:")
out={}
for t in TARGETS:
    bb=[run(t,BASE,s) for s in range(5)]; tt=[run(t,best[t],s) for s in range(5)]
    p=stats.ttest_rel(tt,bb).pvalue
    out[t]={"baseline":bb,"tuned":tt,"delta":float(np.mean(tt)-np.mean(bb)),"p":float(p),
            "params":{k:v for k,v in best[t].items() if k not in ("verbose","n_jobs")}}
    print(f"  {t}: baseline {np.mean(bb):.3f}+-{np.std(bb,ddof=1):.3f}   "
          f"tuned {np.mean(tt):.3f}+-{np.std(tt,ddof=1):.3f}   "
          f"delta {np.mean(tt)-np.mean(bb):+.3f}  p={p:.3f}", flush=True)
json.dump(out, open("tune_experiment.json","w"), indent=1)
print("wrote tune_experiment.json")
