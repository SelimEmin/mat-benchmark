"""Is the NET offset a publication/laboratory artefact rather than an assay-modality one?

With releasers ruled out, the remaining competing explanation is provenance: if the
binding values and the functional values come from systematically different papers --
especially if a handful of documents supply most of the functional side -- the "offset"
could be a between-laboratory calibration difference. This tests that by:

  1. counting the distinct ChEMBL documents behind each modality
  2. leave-one-document-out on the functional side
  3. restricting to compounds where BOTH modalities come from the SAME document
     (the within-paper comparison, which removes between-laboratory variation entirely)
"""
import csv, json
from collections import defaultdict, Counter
import numpy as np
from scipy import stats

acts = list(csv.DictReader(open("net81_mat_activities.csv")))
paired = {r["molecule_chembl_id"]: r for r in
          csv.DictReader(open("data/net_paired_compounds.csv"))}

# NET only; same curation rules as the re-derivation
net = [a for a in acts if a["target_name"] == "NET"
       and a["molecule_chembl_id"] in paired
       and a["standard_relation"] == "="
       and a["pchembl_value"] not in (None, "", "None")]
B = [a for a in net if a["assay_type"] == "B" and a["standard_type"] == "Ki"]
F = [a for a in net if a["assay_type"] == "F" and a["standard_type"] == "IC50"]

print("=" * 86)
print("PROVENANCE OF THE TWO MODALITIES  (NET, the 81 paired compounds)")
print("=" * 86)
nB = len({a["document_chembl_id"] for a in B})
nF = len({a["document_chembl_id"] for a in F})
print(f"  binding records    {len(B):5}   distinct documents {nB}")
print(f"  functional records {len(F):5}   distinct documents {nF}")

docB = Counter(a["document_chembl_id"] for a in B)
docF = Counter(a["document_chembl_id"] for a in F)
print("\n  top functional documents (records, compounds):")
compF = defaultdict(set)
for a in F:
    compF[a["document_chembl_id"]].add(a["molecule_chembl_id"])
for d, n in docF.most_common(10):
    print(f"    {d}  {n:4} records  {len(compF[d]):3} compounds")

# ------------------------------------------------- per-compound medians by doc
def med(xs):
    return float(np.median([float(x["pchembl_value"]) for x in xs]))

bind = defaultdict(list); func = defaultdict(list)
for a in B:
    bind[a["molecule_chembl_id"]].append(a)
for a in F:
    func[a["molecule_chembl_id"]].append(a)

# ------------------------------------------------- 2. leave-one-document-out
print("\n" + "=" * 86)
print("LEAVE-ONE-DOCUMENT-OUT on the functional side")
print("=" * 86)
def offset(exclude_doc=None):
    ds = []
    for m in paired:
        f = [a for a in func.get(m, []) if a["document_chembl_id"] != exclude_doc]
        b = bind.get(m, [])
        if f and b:
            ds.append(med(f) - med(b))
    return np.array(ds)

base = offset()
print(f"  all documents:            n={base.size:3}  median {np.median(base):+.3f}")
worst = None
for d, _ in docF.most_common(12):
    o = offset(d)
    if o.size < 5:
        print(f"  drop {d}: n={o.size} -- too few left, skipped")
        continue
    print(f"  drop {d}:  n={o.size:3}  median {np.median(o):+.3f}"
          f"   ({len(compF[d])} compounds removed)")
    if worst is None or np.median(o) < worst[1]:
        worst = (d, float(np.median(o)), int(o.size))
if worst:
    print(f"\n  worst case: dropping {worst[0]} leaves median {worst[1]:+.3f} (n={worst[2]})")

# ------------------------------------------------- 3. within-document pairs
print("\n" + "=" * 86)
print("WITHIN-DOCUMENT PAIRS  (both modalities from the same paper)")
print("=" * 86)
within = []
for m in paired:
    bdocs = {a["document_chembl_id"] for a in bind.get(m, [])}
    fdocs = {a["document_chembl_id"] for a in func.get(m, [])}
    shared = bdocs & fdocs
    for d in shared:
        bb = [a for a in bind[m] if a["document_chembl_id"] == d]
        ff = [a for a in func[m] if a["document_chembl_id"] == d]
        within.append({"molecule_chembl_id": m, "document_chembl_id": d,
                       "p_binding": med(bb), "p_functional": med(ff),
                       "delta": med(ff) - med(bb)})
# one value per compound (median across shared documents)
bycomp = defaultdict(list)
for w in within:
    bycomp[w["molecule_chembl_id"]].append(w["delta"])
wd = np.array([float(np.median(v)) for v in bycomp.values()])

print(f"  compounds with both modalities in the SAME document: {wd.size}")
if wd.size:
    boot = [np.median(np.random.choice(wd, wd.size, replace=True)) for _ in range(5000)]
    print(f"  within-document offset: median {np.median(wd):+.3f}  mean {wd.mean():+.3f}"
          f"  95% CI [{np.percentile(boot,2.5):+.3f}, {np.percentile(boot,97.5):+.3f}]")
    if wd.size >= 6:
        print(f"  Wilcoxon p = {stats.wilcoxon(wd)[1]:.3g}")
    print(f"  pooled-across-documents offset for comparison: {np.median(base):+.3f}")
    print(f"  distinct documents contributing: "
          f"{len({w['document_chembl_id'] for w in within})}")
    for d, c in Counter(w['document_chembl_id'] for w in within).most_common():
        dd = [w['delta'] for w in within if w['document_chembl_id'] == d]
        print(f"    {d}  {len(dd):3} compounds  median {np.median(dd):+.3f}")

json.dump({"n_binding_records": len(B), "n_functional_records": len(F),
           "n_binding_docs": len(docB), "n_functional_docs": len(docF),
           "pooled_offset_median": float(np.median(base)), "pooled_n": int(base.size),
           "loo_worst": worst,
           "within_document_n": int(wd.size),
           "within_document_median": float(np.median(wd)) if wd.size else None,
           "within_document_docs": sorted({w["document_chembl_id"] for w in within})},
          open("document_confound.json", "w"), indent=2)
with open("within_document_pairs.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=["molecule_chembl_id", "document_chembl_id",
                                       "p_binding", "p_functional", "delta"])
    w.writeheader(); w.writerows(within)
print("\nwrote document_confound.json / within_document_pairs.csv")
