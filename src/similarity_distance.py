"""Is there a quantifiable relationship between distance-to-training and accuracy?

The earlier gating analysis sliced held-out compounds by their nearest-neighbour
Tanimoto AFTER the fact. That confounds two things: compounds far from training may
also be odd in other ways (unusual scaffolds, extreme potency, noisier measurements),
so the drop in r^2 need not be caused by the distance.

This builds the test sets instead of selecting them. For a similarity ceiling tau, a
random scaffold-grouped test set is drawn and then every training compound within
Tanimoto > tau of ANY test compound is REMOVED. The test set is held fixed while the
training set is pushed away from it, so the only thing that changes is distance.

THE CONTROL THAT MAKES IT INTERPRETABLE. Pruning by similarity also shrinks the
training set, and a smaller training set is worse for reasons that have nothing to do
with similarity. So every similarity-pruned arm is paired with a RANDOM-REMOVAL arm
that drops exactly the same NUMBER of compounds, chosen at random. The gap between the
two curves is the similarity effect; the random curve is the data-loss effect. Without
this control the experiment cannot distinguish them.
"""
import json, warnings
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors as rdmd
from rdkit.Chem.Scaffolds import MurckoScaffold
from scipy import stats

warnings.filterwarnings("ignore")
RDLogger.DisableLog("rdApp.*")
GEN = AllChem.GetMorganGenerator(radius=2, fpSize=2048)
DESCS = [Descriptors.MolWt, Descriptors.MolLogP, Descriptors.TPSA, rdmd.CalcNumHBD,
         rdmd.CalcNumHBA, rdmd.CalcNumRotatableBonds, rdmd.CalcNumAromaticRings,
         rdmd.CalcFractionCSP3, lambda m: m.GetNumHeavyAtoms(), rdmd.CalcNumRings]
TARGETS = ("DAT", "NET", "SERT")
TAUS = (1.00, 0.90, 0.80, 0.70, 0.60, 0.50, 0.40)
NREP, NTEST = 6, 300


def lgbm():
    return LGBMRegressor(n_estimators=250, learning_rate=0.06, num_leaves=31,
                         min_child_samples=10, subsample=0.8, colsample_bytree=0.4,
                         random_state=0, verbose=-1, n_jobs=2)


lab = pd.read_csv("modality_labels.csv")
rows = []
for t in TARGETS:
    d = lab[(lab.target == t) & lab.p_pool.notna()].reset_index(drop=True)
    mols = [Chem.MolFromSmiles(s) for s in d.smiles]
    ok = [i for i, m in enumerate(mols) if m is not None]
    d, mols = d.iloc[ok].reset_index(drop=True), [mols[i] for i in ok]
    scaf = np.array([MurckoScaffold.MurckoScaffoldSmiles(mol=m) for m in mols])
    fps = [GEN.GetFingerprint(m) for m in mols]
    A = np.zeros((len(mols), 2048), np.int8)
    for i, m in enumerate(mols):
        DataStructs.ConvertToNumpyArray(fps[i], A[i])
    X = np.hstack([A, np.array([[f(m) for f in DESCS] for m in mols],
                               np.float32)]).astype(np.float32)
    yk, yi = d.p_Ki.values, d.p_IC50.values
    print(f"\n{t}: {len(d)} compounds", flush=True)

    for rep in range(NREP):
        rng = np.random.default_rng(1000 * rep + 7)
        # scaffold-grouped random test set, so the test set has no internal leakage
        uq = rng.permutation(np.unique(scaf))
        test, i = [], 0
        while len(test) < NTEST and i < uq.size:
            test.extend(np.where(scaf == uq[i])[0])
            i += 1
        test = np.array(test[:NTEST])
        pool0 = np.setdiff1d(np.arange(len(d)), test)
        # max similarity of each POOL compound to the test set
        tf = [fps[i] for i in test]
        simpool = np.array([max(DataStructs.BulkTanimotoSimilarity(fps[i], tf))
                            for i in pool0])

        for tau in TAUS:
            keep_sim = pool0[simpool <= tau]
            n_removed = pool0.size - keep_sim.size
            keep_rnd = rng.choice(pool0, pool0.size - n_removed, replace=False)
            for arm, keep in (("similarity", keep_sim), ("random", keep_rnd)):
                if keep.size < 200:
                    continue
                ki = keep[np.isfinite(yk[keep])]
                ic = keep[np.isfinite(yi[keep])]
                if ki.size < 50 or ic.size < 50:
                    continue
                Xm = np.vstack([np.hstack([X[ki], np.ones((ki.size, 1), np.float32)]),
                                np.hstack([X[ic], np.zeros((ic.size, 1), np.float32)])])
                m = lgbm().fit(Xm, np.concatenate([yk[ki], yi[ic]]))
                pk = m.predict(np.hstack([X[test], np.ones((test.size, 1), np.float32)]))
                pi = m.predict(np.hstack([X[test], np.zeros((test.size, 1), np.float32)]))
                a, b = np.isfinite(yk[test]), np.isfinite(yi[test])
                y = np.concatenate([yk[test][a], yi[test][b]])
                p = np.concatenate([pk[a], pi[b]])
                kf = [fps[i] for i in keep]
                realised = float(np.median([
                    max(DataStructs.BulkTanimotoSimilarity(fps[i], kf)) for i in test]))
                rows.append(dict(target=t, rep=rep, tau=tau, arm=arm,
                                 n_train=int(keep.size), n_removed=int(n_removed),
                                 realised_nn=realised, n_test=int(y.size),
                                 r2=float(stats.pearsonr(y, p)[0] ** 2),
                                 RMSE=float(np.sqrt(((y - p) ** 2).mean())),
                                 within_1=float((np.abs(y - p) <= 1).mean()),
                                 rho=float(stats.spearmanr(y, p).statistic)))
        print(f"  rep {rep+1}/{NREP} done", flush=True)
        pd.DataFrame(rows).to_csv("similarity_distance.csv", index=False)

df = pd.DataFrame(rows)
df.to_csv("similarity_distance.csv", index=False)

B = "=" * 98
print("\n" + B)
print("Accuracy against distance to training, with the data-loss control")
print(B)
for t in TARGETS:
    s = df[df.target == t]
    print(f"\n{t}")
    print(f"  {'tau':<6}{'removed':>9}{'n train':>9}{'realised NN':>13}"
          f"{'r2 similarity':>15}{'r2 random':>12}{'gap':>8}{'RMSE sim':>10}")
    for tau in TAUS:
        a = s[(s.tau == tau) & (s.arm == "similarity")]
        b = s[(s.tau == tau) & (s.arm == "random")]
        if a.empty:
            continue
        print(f"  {tau:<6.2f}{a.n_removed.mean():>9.0f}{a.n_train.mean():>9.0f}"
              f"{a.realised_nn.mean():>13.3f}{a.r2.mean():>15.3f}"
              f"{(b.r2.mean() if not b.empty else np.nan):>12.3f}"
              f"{(a.r2.mean()-b.r2.mean() if not b.empty else np.nan):>+8.3f}"
              f"{a.RMSE.mean():>10.2f}")

print("\n" + B)
print("The relationship, fitted on the similarity arm")
print(B)
for t in TARGETS:
    s = df[(df.target == t) & (df.arm == "similarity")]
    sl, ic, r, p, se = stats.linregress(s.realised_nn, s.r2)
    slr, icr, rr, pr, ser = stats.linregress(
        df[(df.target == t) & (df.arm == "random")].n_train,
        df[(df.target == t) & (df.arm == "random")].r2)
    print(f"  {t:<5} r2 = {ic:+.3f} + {sl:.3f} x NN      (r={r:+.3f}, p={p:.1e})"
          f"   -> each 0.10 drop in NN costs {sl*0.10:.3f} of r^2")
json.dump({"n_rows": len(df)}, open("similarity_distance.json", "w"))
print("\nwrote similarity_distance.csv")
