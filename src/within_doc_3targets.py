"""Within-document binding-vs-functional offset at DAT, NET and SERT.

Same curation as replicate_assay_offset.py (human targets, Ki from assay_type B,
IC50 from assay_type F, pchembl present, uncensored, no data-validity comment), but
restricted to compounds whose two modality values come from the SAME ChEMBL document.
That removes between-laboratory calibration differences by construction.

Eshleman et al. 2017 (J Pharmacol Exp Ther 360:33-47) is added as an independent
within-document source. Its Table 2 publishes Ki/IC50 per compound at all three
transporters, so log10(ratio) is exactly the offset used here, and its Table 3 release
assay classifies each compound as blocker or substrate WITHOUT reference to that ratio.
"""
import collections, csv, json, math
import numpy as np
from scipy import stats

ACT = "data/chembl_rederivation_activities.csv"
TARGETS = ("DAT", "NET", "SERT")
rng = np.random.default_rng(20260930)


def boot_ci(d, n=5000):
    if d.size < 3:
        return None
    b = [np.median(rng.choice(d, d.size, replace=True)) for _ in range(n)]
    return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


# ----------------------------------------------------------------- ChEMBL side
rows = [r for r in csv.DictReader(open(ACT)) if r["data_validity_comment"] in ("", "None")]
store = collections.defaultdict(lambda: collections.defaultdict(
    lambda: collections.defaultdict(lambda: collections.defaultdict(list))))
for r in rows:
    st, at = r["standard_type"], r["assay_type"]
    mod = "bind" if (st == "Ki" and at == "B") else ("func" if (st == "IC50" and at == "F") else None)
    if mod is None:
        continue
    try:
        p = float(r["pchembl_value"])
    except (TypeError, ValueError):
        continue
    store[r["_target"]][r["molecule_chembl_id"]][r["document_chembl_id"]][mod].append(p)

pooled, within, bydoc = {}, {}, {}
for t in TARGETS:
    # pooled across documents (reproduces the published paired set)
    pool = []
    for mid, docs in store[t].items():
        b = [v for d in docs.values() for v in d.get("bind", [])]
        f = [v for d in docs.values() for v in d.get("func", [])]
        if b and f:
            pool.append(float(np.median(f)) - float(np.median(b)))
    pooled[t] = np.array(pool)

    # within-document
    pairs, perdoc = [], collections.defaultdict(list)
    for mid, docs in store[t].items():
        deltas = []
        for doc, d in docs.items():
            if d.get("bind") and d.get("func"):
                dl = float(np.median(d["func"])) - float(np.median(d["bind"]))
                deltas.append(dl)
                perdoc[doc].append({"molecule_chembl_id": mid, "delta": dl})
        if deltas:
            pairs.append({"molecule_chembl_id": mid, "delta": float(np.median(deltas)),
                          "n_docs": len(deltas)})
    within[t] = np.array([p["delta"] for p in pairs])
    bydoc[t] = {k: v for k, v in perdoc.items()}

    with open(f"within_doc_{t}.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["molecule_chembl_id", "delta", "n_docs"])
        w.writeheader(); w.writerows(pairs)

# ------------------------------------------------------- Eshleman et al. 2017
# Table 2: ratio of binding Ki to uptake IC50. log10(ratio) = pIC50 - pKi = offset.
# class from Table 3 (release assay), which is independent of this ratio.
# "b" = value carried over from Eshleman et al. 2013, a companion paper.
ESH = [
    # name,                class,       DAT,   SERT,  NET,   from_2013
    ("alpha-PPP",          "blocker",   2.4,   0.9,   6.7,   False),
    ("alpha-PBP",          "blocker",   1.9,   2.4,   3.6,   False),
    ("alpha-PVP",          "blocker",   1.1,   1.2,   2.7,   False),
    ("alpha-PHP",          "blocker",   0.7,   0.8,   9.3,   False),
    ("PV-8",               "blocker",   1.0,   1.5,  11.4,   False),
    ("4-MePPP",            "blocker",   2.0,   4.8,   5.5,   False),
    ("4-MeO-alpha-PVP",    "blocker",   2.8,   1.5,   5.0,   False),
    ("3,4-MDPPP",          "blocker",   2.7,   1.5,   5.6,   False),
    ("3,4-MDPBP",          "blocker",   1.2,   3.3,   5.2,   False),
    ("alpha-PVT",          "blocker",   1.8,   0.7,  12.9,   False),
    ("MDPV",               "blocker",   1.5,   0.91,  5.7,   True),
    ("Naphyrone",          "blocker",   0.85,  0.25,  1.7,   True),
    ("Pentedrone",         "blocker",   1.9,   1.8,   8.2,   False),
    ("Pentylone",          "blocker",   2.4,   4.2,  12.6,   False),
    ("Cocaine",            "blocker",   1.9,   1.9,   7.0,   False),
    ("Mazindol",           "blocker",   2.0,   2.0,   5.0,   False),
    ("MCAT",               "releaser", 31.4,  15.2, 120.9,   False),
    ("3-F MCAT",           "releaser", 15.2,  16.4, 137.7,   False),
    ("4-Cl MCAT",          "releaser", 45.2,  42.8, 259.6,   False),
    ("4-Br MCAT",          "releaser", 16.7,  20.5, 244.3,   False),
    ("4-MEC",              "releaser",  3.1,  57.8,  17.0,   False),
    ("5-MeO-methylone",    "releaser",  None, 14.7,   4.0,   False),   # DAT "<1.6"
    ("Ethylone",           "releaser",  2.9,  51.1,  13.2,   False),
    ("4-F MCAT",           "releaser", 38.0,  None, 276.0,   True),    # SERT ND
    ("Butylone",           "releaser",  8.2,   5.3,   7.5,   True),
    ("Mephedrone",         "releaser", 49.0,  41.0, 220.0,   True),
    ("Methylone",          "releaser", 15.0,  50.0,  71.0,   True),
    ("MDAI",               "releaser", 11.9, 201.9, 250.0,   False),
    ("4-FA",               "releaser", 369.2, 24.3, 434.3,   False),
    ("METH",               "releaser", 68.7,  28.8, 110.3,   False),
    ("MDMA",               "releaser", 79.0,  66.0, 246.3,   False),
]
esh = {t: {"blocker": [], "releaser": []} for t in TARGETS}
for name, cls, dat, sert, net, old in ESH:
    for t, v in (("DAT", dat), ("SERT", sert), ("NET", net)):
        if v is not None:
            esh[t][cls].append(math.log10(v))

# --------------------------------------------------------------------- report
out = {"chembl": {}, "eshleman": {}}
print("=" * 92)
print("WITHIN-DOCUMENT OFFSET AT ALL THREE TARGETS  (ChEMBL re-retrieval)")
print("=" * 92)
print(f"{'target':<7}{'pooled n':>9}{'pooled':>9}   {'within n':>9}{'within':>9}"
      f"{'95% CI':>22}{'Wilcoxon p':>13}{'docs':>6}")
for t in TARGETS:
    w, p = within[t], pooled[t]
    ci = boot_ci(w)
    wp = stats.wilcoxon(w)[1] if w.size >= 6 else float("nan")
    nd = len([d for d, v in bydoc[t].items() if len(v) >= 5])
    print(f"{t:<7}{p.size:9d}{np.median(p):+9.3f}   {w.size:9d}{np.median(w):+9.3f}"
          f"   [{ci[0]:+.3f}, {ci[1]:+.3f}]{wp:13.2g}{nd:6d}")
    out["chembl"][t] = {"pooled_n": int(p.size), "pooled_median": float(np.median(p)),
                        "within_n": int(w.size), "within_median": float(np.median(w)),
                        "within_ci95": ci, "within_wilcoxon_p": float(wp),
                        "documents_ge5": nd}

print("\nContributing documents with >=5 compounds, per target:")
for t in TARGETS:
    print(f"  {t}:")
    for d, v in sorted(bydoc[t].items(), key=lambda kv: -len(kv[1])):
        if len(v) >= 5:
            dl = np.array([x["delta"] for x in v])
            print(f"    {d}  n={len(v):3}  median {np.median(dl):+.3f}")
            out["chembl"][t].setdefault("documents", []).append(
                {"document_chembl_id": d, "n": len(v), "median": float(np.median(dl))})

print("\n" + "=" * 92)
print("ESHLEMAN ET AL. 2017  --  one paper, one laboratory, both modalities")
print("  class assigned by their [3H]neurotransmitter RELEASE assay, not by the ratio")
print("=" * 92)
print(f"{'target':<7}{'blockers n':>12}{'median':>9}{'95% CI':>22}"
      f"{'releasers n':>13}{'median':>9}")
for t in TARGETS:
    b = np.array(esh[t]["blocker"]); r = np.array(esh[t]["releaser"])
    ci = boot_ci(b)
    print(f"{t:<7}{b.size:12d}{np.median(b):+9.3f}   [{ci[0]:+.3f}, {ci[1]:+.3f}]"
          f"{r.size:13d}{np.median(r):+9.3f}")
    out["eshleman"][t] = {"blocker_n": int(b.size), "blocker_median": float(np.median(b)),
                          "blocker_ci95": ci,
                          "releaser_n": int(r.size),
                          "releaser_median": float(np.median(r))}

print("\nNET minus the mean of DAT and SERT (the NET-specific excess):")
for src, lab in ((out["chembl"], "ChEMBL within-document"),
                 (out["eshleman"], "Eshleman blockers")):
    key = "within_median" if "within_median" in src["NET"] else "blocker_median"
    excess = src["NET"][key] - 0.5 * (src["DAT"][key] + src["SERT"][key])
    print(f"  {lab:<26} {excess:+.3f}")
    out.setdefault("net_excess", {})[lab] = float(excess)

# Kruskal-Wallis across targets, within-document only
kw = stats.kruskal(within["DAT"], within["NET"], within["SERT"])
print(f"\nKruskal-Wallis across the three within-document sets: H={kw.statistic:.1f}, "
      f"p={kw.pvalue:.2g}")
out["kruskal_within"] = {"H": float(kw.statistic), "p": float(kw.pvalue)}
kw2 = stats.kruskal(esh["DAT"]["blocker"], esh["NET"]["blocker"], esh["SERT"]["blocker"])
print(f"Kruskal-Wallis across Eshleman blockers:            H={kw2.statistic:.1f}, "
      f"p={kw2.pvalue:.2g}")
out["kruskal_eshleman_blockers"] = {"H": float(kw2.statistic), "p": float(kw2.pvalue)}

json.dump(out, open("within_doc_3targets.json", "w"), indent=2)
with open("eshleman_table2.csv", "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["compound", "class", "ratio_DAT", "ratio_SERT", "ratio_NET",
                "offset_DAT", "offset_SERT", "offset_NET", "from_eshleman_2013"])
    for name, cls, dat, sert, net, old in ESH:
        lg = [f"{math.log10(v):.3f}" if v else "" for v in (dat, sert, net)]
        w.writerow([name, cls, dat or "", sert or "", net or "", *lg, int(old)])
print("\nwrote within_doc_3targets.json, within_doc_{DAT,NET,SERT}.csv, eshleman_table2.csv")
