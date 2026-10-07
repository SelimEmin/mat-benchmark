"""Follow-up: is the NET offset target-specific, radioligand-specific, or an interaction?

The stratified medians show the NET offset is large with [125I]RTI-55 (+0.475) and absent
with [3H]nisoxetine (-0.080), while RTI-55 at DAT and SERT gives ~0. An additive model
cannot separate "ligand" from "target x ligand" here, because nisoxetine is used almost
only at NET. Four checks:

  1. the interaction model, fitted explicitly;
  2. compounds measured at NET by BOTH radioligands -- the within-compound contrast;
  3. which radioligand the five within-document papers of the manuscript use;
  4. whether chemotype confounds radioligand (do the two strata hold different compounds?).
"""
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from radioligand import ligand          # reuse the extractor

d = pd.read_csv("classified.csv")
d = d[d.modality.isin(["binding", "uptake"])].copy()
d["ligand"] = [ligand(x) for x in d.assay_description]
B = "=" * 92

# ---------------------------------------------------------------- 1. interaction model
L = pd.read_csv("radioligand_pairs.csv")
keep = L.groupby(["target", "ligand"]).cid.transform("size") >= 10
Lk = L[keep].copy()
print(B); print("1. INTERACTION MODEL  (target x radioligand, document-clustered SEs)"); print(B)
print(f"   strata with n>=10 retained: {len(Lk)} rows, "
      f"{Lk.groupby(['target','ligand']).ngroups} strata")
Lk["cell"] = Lk.target + " / " + Lk.ligand
m = smf.ols("offset ~ C(cell)", data=Lk).fit(
    cov_type="cluster", cov_kwds={"groups": Lk.doc})
base = sorted(Lk.cell.unique())[0]
print(f"\n   reference cell: {base}")
print(f"   {'cell':<30}{'estimate':>10}{'SE':>8}{'p':>12}   {'n':>5}")
for k in m.params.index:
    nm = k.replace("C(cell)[T.", "").replace("]", "") if k != "Intercept" else base
    n = int((Lk.cell == nm).sum()) if k != "Intercept" else int((Lk.cell == base).sum())
    print(f"   {nm:<30}{m.params[k]:>+10.3f}{m.bse[k]:>8.3f}{m.pvalues[k]:>12.3g}   {n:>5}")

# ---------------------------------------------------------------- 2. within-compound
print()
print(B); print("2. COMPOUNDS MEASURED AT NET BY BOTH RADIOLIGANDS"); print(B)
net = d[(d._target == "NET")]
nb = net[net.modality == "binding"]
piv = (nb[nb.ligand.isin(["[125I]RTI-55", "[3H]nisoxetine"])]
       .groupby(["molecule_chembl_id", "ligand"]).pchembl_value.median().unstack())
both = piv.dropna()
print(f"   compounds with a NET binding value from BOTH ligands: {len(both)}")
if len(both) >= 5:
    diff = both["[125I]RTI-55"] - both["[3H]nisoxetine"]
    w = stats.wilcoxon(diff)
    print(f"   median RTI-55 minus nisoxetine binding pKi: {np.median(diff):+.3f} "
          f"(IQR {np.percentile(diff,25):+.2f} to {np.percentile(diff,75):+.2f}), "
          f"Wilcoxon p={w.pvalue:.3g}")
    print("   -> a positive value means RTI-55 assays report the compound as MORE potent,")
    print("      i.e. the two radioligands are NOT on the same scale at NET.")
    up = net[net.modality == "uptake"].groupby("molecule_chembl_id").pchembl_value.median()
    c = both.index.intersection(up.index)
    if len(c) >= 5:
        print(f"\n   of those, {len(c)} also have a functional uptake value:")
        o_r = up[c] - both.loc[c, "[125I]RTI-55"]
        o_n = up[c] - both.loc[c, "[3H]nisoxetine"]
        print(f"      offset against RTI-55 binding    : median {np.median(o_r):+.3f}")
        print(f"      offset against nisoxetine binding: median {np.median(o_n):+.3f}")
        print("      (same compounds, same uptake values -- only the comparator changes)")

# ---------------------------------------------------------------- 3. the five papers
print()
print(B); print("3. WHICH RADIOLIGAND DO THE WITHIN-DOCUMENT NET PAPERS USE?"); print(B)
DOCS = ["CHEMBL1138144", "CHEMBL1146223", "CHEMBL1135775",
        "CHEMBL1132179", "CHEMBL1136239"]
for doc in DOCS:
    g = d[(d.document_chembl_id == doc) & (d._target == "NET") & (d.modality == "binding")]
    if not len(g):
        print(f"   {doc}: no NET binding records")
        continue
    vc = g.ligand.value_counts().to_dict()
    print(f"   {doc}: {vc}")

# ---------------------------------------------------------------- 4. chemotype confound
print()
print(B); print("4. DO THE TWO NET STRATA HOLD DIFFERENT CHEMISTRY?"); print(B)
try:
    from rdkit import Chem, DataStructs, RDLogger
    from rdkit.Chem import AllChem, Descriptors
    RDLogger.DisableLog("rdApp.*")
    GEN = AllChem.GetMorganGenerator(radius=2, fpSize=2048)
    smi = d.drop_duplicates("molecule_chembl_id").set_index("molecule_chembl_id")\
           .canonical_smiles
    sets = {}
    for lig in ("[125I]RTI-55", "[3H]nisoxetine"):
        ids = L[(L.target == "NET") & (L.ligand == lig)].cid.unique()
        sets[lig] = [s for s in smi.reindex(ids).dropna()]
    a, b = sets["[125I]RTI-55"], sets["[3H]nisoxetine"]
    print(f"   RTI-55 stratum {len(a)} compounds, nisoxetine stratum {len(b)}")
    ia = set(L[(L.target=='NET')&(L.ligand=='[125I]RTI-55')].cid)
    ib = set(L[(L.target=='NET')&(L.ligand=='[3H]nisoxetine')].cid)
    print(f"   compounds appearing in both strata: {len(ia & ib)}")
    for nm, ss in (("RTI-55", a), ("nisoxetine", b)):
        mw = [Descriptors.MolWt(Chem.MolFromSmiles(s)) for s in ss
              if Chem.MolFromSmiles(s)]
        print(f"   {nm:<12} median MW {np.median(mw):6.1f}   n={len(mw)}")
    fa = [GEN.GetFingerprint(Chem.MolFromSmiles(s)) for s in a if Chem.MolFromSmiles(s)]
    fb = [GEN.GetFingerprint(Chem.MolFromSmiles(s)) for s in b if Chem.MolFromSmiles(s)]
    cross = [max(DataStructs.BulkTanimotoSimilarity(x, fb)) for x in fa]
    print(f"   median nearest-neighbour Tanimoto, RTI-55 stratum to nisoxetine stratum: "
          f"{np.median(cross):.2f}")
    print("   (a low value means the strata are different chemistry and the comparison")
    print("    between them is confounded by chemotype as well as by radioligand)")
except ImportError:
    print("   rdkit unavailable")
