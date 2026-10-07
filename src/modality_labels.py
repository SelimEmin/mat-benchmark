"""Per-modality labels: separate Ki and IC50 medians for every compound x target.

The benchmark table carries one pooled value per compound x target. The ceiling
analysis argues that the pooling itself is what caps achievable R^2, because the
binding/functional offset is structure the model is never shown. To test that, the
labels have to be rebuilt with the modality kept.

Note on provenance: ChEMBL's `assay_type` field does NOT separate the modalities here
-- 8,420 of the 8,801 IC50 records are tagged B. `standard_type` (Ki vs IC50) is the
field that actually carries the distinction, and it is what the within-document
analysis is implicitly using. This script groups on standard_type.
"""
import json
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")
TARGETS = ("DAT", "NET", "SERT")
SRC = "data/chembl_rederivation_activities.csv"

d = pd.read_csv(SRC)
d = d[d.standard_type.isin(["Ki", "IC50"])].copy()

sub = d.drop_duplicates("molecule_chembl_id")[["molecule_chembl_id", "canonical_smiles"]]
key, smi = {}, {}
for mid, s in zip(sub.molecule_chembl_id, sub.canonical_smiles):
    m = Chem.MolFromSmiles(s) if isinstance(s, str) else None
    if m is not None:
        key[mid] = Chem.MolToInchiKey(m)
        smi[mid] = s
d["ik"] = d.molecule_chembl_id.map(key)
d = d[d.ik.notna()]

# median per compound x target x modality, plus record counts
g = (d.groupby(["ik", "_target", "standard_type"])
     .agg(p=("pchembl_value", "median"), nrec=("pchembl_value", "size"),
          ndoc=("document_chembl_id", "nunique"))
     .reset_index())
pool = (d.groupby(["ik", "_target"])
        .agg(p=("pchembl_value", "median"), nrec=("pchembl_value", "size"))
        .reset_index())

wide = g.pivot_table(index=["ik", "_target"], columns="standard_type",
                     values=["p", "nrec"]).reset_index()
wide.columns = ["ik", "target", "n_IC50", "n_Ki", "p_IC50", "p_Ki"]
wide = wide.merge(pool.rename(columns={"_target": "target", "p": "p_pool",
                                       "nrec": "n_pool"}), on=["ik", "target"])
wide["smiles"] = wide.ik.map({key[m]: smi[m] for m in key})

print("=" * 78)
print("Coverage by modality (compounds with a median of that kind)")
print("=" * 78)
print(f"{'target':<7}{'Ki':>8}{'IC50':>8}{'both':>8}{'Ki only':>10}{'IC50 only':>11}"
      f"{'pooled':>9}")
OFFSET = {}
for t in TARGETS:
    s = wide[wide.target == t]
    hk, hi = s.p_Ki.notna(), s.p_IC50.notna()
    both = s[hk & hi]
    print(f"{t:<7}{int(hk.sum()):>8}{int(hi.sum()):>8}{len(both):>8}"
          f"{int((hk & ~hi).sum()):>10}{int((hi & ~hk).sum()):>11}{len(s):>9}")
    OFFSET[t] = float((both.p_Ki - both.p_IC50).median())

print()
print("=" * 78)
print("Binding - functional offset, from compounds measured both ways")
print("(positive = the Ki value is the more potent of the two)")
print("=" * 78)
print(f"{'target':<7}{'n':>6}{'median':>9}{'mean':>8}{'SD':>8}{'IQR':>16}"
      f"{'Wilcoxon p':>13}")
from scipy import stats
for t in TARGETS:
    s = wide[(wide.target == t) & wide.p_Ki.notna() & wide.p_IC50.notna()]
    diff = (s.p_Ki - s.p_IC50).values
    w = stats.wilcoxon(diff)
    q1, q3 = np.percentile(diff, [25, 75])
    print(f"{t:<7}{len(diff):>6}{np.median(diff):>9.3f}{diff.mean():>8.3f}"
          f"{diff.std(ddof=1):>8.3f}   [{q1:>6.2f},{q3:>6.2f}]{w.pvalue:>13.2e}")

print("\n  manuscript's within-document offsets for comparison: "
      "DAT -0.280, NET +0.540, SERT -0.080")
print("  (those are same-paper pairs only; these pool across papers, so they carry")
print("   the between-laboratory term the within-document design removes)")

wide.to_csv("modality_labels.csv", index=False)
json.dump({"offset_Ki_minus_IC50": OFFSET}, open("modality_offsets.json", "w"), indent=1)
print(f"\nwrote modality_labels.csv ({len(wide)} compound x target rows)")
