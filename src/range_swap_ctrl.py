"""Exp 3 was confounded: 43 training compounds against 3,000 is a sample-size
comparison, not a distribution one. This adds the control — a ChEMBL-trained model
cut to the SAME tiny training size, inside the same window."""
import csv, json, math, warnings
import numpy as np, pandas as pd
from collections import defaultdict
from rdkit import Chem, RDLogger
warnings.filterwarnings("ignore"); RDLogger.DisableLog("rdApp.*")
exec(open("range_swap.py").read().split("# ------------------------------------------------------------------ external sets")[0].split('"""',2)[2])

st={r["abbrev"]:r["final"] for r in csv.DictReader(open("struct.csv"))}
v=defaultdict(dict)
for r in csv.DictReader(open("stalberga_parsed.csv")): v[r["compound"]][r["target"]]=(r["relation"],r["value"].strip())
STAL=[{"smiles":st[c],**{t:9.0-math.log10(float(d[t][1])) for t in TARGETS}} for c,d in v.items()
      if len(d)==3 and all(x[0]=="=" and x[1] for x in d.values())]
PDSP=[{"smiles":r["smiles"],**{t:float(r["p"+t]) for t in TARGETS}} for r in csv.DictReader(open("pdsp_mat_profiles.csv"))]
SETS=[("Stalberga 2026",STAL),("PDSP / BindingDB",PDSP)]
ext_skel={skel(r["smiles"]) for _,rows in SETS for r in rows}-{None}
lab=pd.read_csv("modality_labels.csv"); out={}
print(f"{'set':<18}{'t':<6}{'n train':>8}{'external-trained':>18}{'ChEMBL-trained, same n':>24}{'ChEMBL-trained, all':>21}")
for t in TARGETS:
    d=lab[(lab.target==t)&lab.p_pool.notna()].reset_index(drop=True)
    keep,X,mols=featurise(d.smiles.values); d=d.iloc[keep].reset_index(drop=True)
    ins=~d.ik.str.split("-").str[0].isin(ext_skel).values
    d,X=d[ins].reset_index(drop=True),X[ins]; y=d.p_pool.values
    for name,rows in SETS:
        sel=[r for r in rows if r.get(t) is not None and Chem.MolFromSmiles(r["smiles"]) is not None]
        ye=np.array([r[t] for r in sel]); _,Xe,_=featurise([r["smiles"] for r in sel])
        lo,hi=ye.min(),ye.max(); win=np.where((y>=lo)&(y<=hi))[0]
        if win.size<200: continue
        ext_r2=score(y[win], lgbm().fit(Xe,ye).predict(X[win]))
        small=[]
        for s in range(5):
            r=np.random.default_rng(100+s)
            tr=r.choice(win, ye.size, replace=False); te=np.setdiff1d(win,tr)
            small.append(score(y[te], lgbm().fit(X[tr],y[tr]).predict(X[te]))["r2"])
        big=[]
        for tr,te in folds_seeded(d.scaffold.values if "scaffold" in d else
                                  pd.Series([""]*len(d)), NFOLD, 0): pass
        oof=np.full(len(d),np.nan)
        from rdkit.Chem.Scaffolds import MurckoScaffold
        sc=[MurckoScaffold.MurckoScaffoldSmiles(s) for s in d.smiles]
        for tr,te in folds_seeded(sc,NFOLD,0): oof[te]=lgbm().fit(X[tr],y[tr]).predict(X[te])
        full=score(y[win],oof[win])["r2"]
        out[f"{t}|{name}"]={"n_train":int(ye.size),"external_trained":ext_r2["r2"],
                            "chembl_same_n":float(np.mean(small)),"chembl_same_n_sd":float(np.std(small,ddof=1)),
                            "chembl_full":full}
        print(f"{name:<18}{t:<6}{ye.size:>8}{ext_r2['r2']:>18.3f}"
              f"{np.mean(small):>19.3f}±{np.std(small,ddof=1):.3f}{full:>21.3f}", flush=True)
json.dump(out, open("range_swap_ctrl.json","w"), indent=1)
