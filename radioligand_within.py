"""The decisive test: the WITHIN-DOCUMENT offset, stratified by radioligand.

The pooled stratification confounds radioligand with chemotype (the RTI-55 and nisoxetine
NET strata share 5 compounds and have a cross-stratum nearest-neighbour Tanimoto of 0.28)
and with laboratory. The within-document contrast removes the laboratory, so asking it
separately within each radioligand is the cleanest available separation.

A document qualifies for a radioligand stratum if it reports, for the same compound, a
functional uptake value and a binding value measured with that radioligand.
"""
import numpy as np
import pandas as pd
from scipy import stats
from radioligand import ligand

d = pd.read_csv("classified.csv")
d = d[d.modality.isin(["binding", "uptake"])].copy()
d["ligand"] = [ligand(x) for x in d.assay_description]
B = "=" * 92


def within(target, lig=None, min_n=3):
    """Per-document paired offsets; returns (per-doc records, all pair offsets)."""
    sub = d[d._target == target]
    out, allpairs = [], []
    for doc, g in sub.groupby("document_chembl_id"):
        b = g[g.modality == "binding"]
        if lig is not None:
            b = b[b.ligand == lig]
        u = g[g.modality == "uptake"]
        if b.empty or u.empty:
            continue
        bm = b.groupby("molecule_chembl_id").pchembl_value.median()
        um = u.groupby("molecule_chembl_id").pchembl_value.median()
        c = bm.index.intersection(um.index)
        if len(c) < min_n:
            continue
        diff = (um[c] - bm[c]).values
        out.append(dict(doc=doc, n=len(c), median=float(np.median(diff))))
        allpairs.extend(diff)
    return pd.DataFrame(out), np.array(allpairs)


print(B); print("WITHIN-DOCUMENT OFFSET AT NET, BY BINDING RADIOLIGAND"); print(B)
print("Each document contributes only compounds it measured both ways itself.\n")
print(f"{'stratum':<22}{'docs':>5}{'pairs':>7}{'median':>9}{'IQR':>18}{'Wilcoxon p':>13}")
rows = []
for lig in [None, "[125I]RTI-55", "[3H]nisoxetine"]:
    D, pairs = within("NET", lig)
    if not len(pairs):
        continue
    w = stats.wilcoxon(pairs) if len(pairs) > 5 else None
    nm = "all radioligands" if lig is None else lig
    print(f"{nm:<22}{len(D):>5}{len(pairs):>7}{np.median(pairs):>+9.3f}"
          f"   [{np.percentile(pairs,25):>+6.2f},{np.percentile(pairs,75):>+6.2f}]"
          f"{(w.pvalue if w else float('nan')):>13.3g}")
    rows.append((nm, D, pairs))

print("\nper-document detail:")
for nm, D, _ in rows:
    if nm == "all radioligands":
        continue
    print(f"  {nm}:")
    for _, r in D.sort_values("n", ascending=False).iterrows():
        print(f"     {r.doc:<16} n={int(r.n):<4} median {r['median']:+.3f}")

print()
print(B); print("THE SAME TEST AT DAT AND SERT, FOR CONTRAST"); print(B)
print(f"{'target':<7}{'stratum':<20}{'docs':>5}{'pairs':>7}{'median':>9}{'Wilcoxon p':>13}")
for t, ligs in (("DAT", [None, "[125I]RTI-55", "[3H]WIN35428"]),
                ("SERT", [None, "[125I]RTI-55", "[3H]citalopram"])):
    for lig in ligs:
        D, pairs = within(t, lig)
        if len(pairs) < 6:
            continue
        w = stats.wilcoxon(pairs)
        nm = "all radioligands" if lig is None else lig
        print(f"{t:<7}{nm:<20}{len(D):>5}{len(pairs):>7}{np.median(pairs):>+9.3f}"
              f"{w.pvalue:>13.3g}")

print()
print(B); print("SUMMARY"); print(B)
_, pr = within("NET", "[125I]RTI-55")
_, pn = within("NET", "[3H]nisoxetine")
if len(pr) and len(pn):
    u = stats.mannwhitneyu(pr, pn)
    print(f"  NET within-document offset, RTI-55 strata     : {np.median(pr):+.3f} "
          f"({len(pr)} pairs)")
    print(f"  NET within-document offset, nisoxetine strata : {np.median(pn):+.3f} "
          f"({len(pn)} pairs)")
    print(f"  difference between the two, Mann-Whitney p    : {u.pvalue:.3g}")
    print()
    print("  The pooled cross-document comparison gave +0.475 and -0.080 for the same two")
    print("  strata. Whether the within-document contrast closes that gap is the question")
    print("  of whether the offset is a property of the format or of the radioligand.")
