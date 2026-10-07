"""Strengthening tests for Finding 3 (the NET offset is neither a laboratory nor a
releaser artefact).

Six analyses, all on the within-document pairs:

  1. proportion positive, with exact binomial CI and sign test
  2. sign-flip permutation test -- the exact null for a paired design
  3. paper as the unit of analysis, not the compound
  4. Cliff's delta, NET against DAT and SERT
  5. leave-one-document-out at all three targets
  6. a structural census of the 81 NET paired compounds
"""
import collections, csv, json
import numpy as np
from scipy import stats
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors

RDLogger.DisableLog("rdApp.*")
ACT = "data/chembl_rederivation_activities.csv"
rng = np.random.default_rng(20260930)
TARGETS = ("DAT", "NET", "SERT")
DOCNAME = {"CHEMBL1132179": "Choi 1999", "CHEMBL1135775": "Houlihan 2002",
           "CHEMBL1136239": "Axford 2003", "CHEMBL1138144": "Meltzer 2006",
           "CHEMBL1146223": "Gong 2007"}

# ------------------------------------------------- within-document pairs
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

pairs = {}          # target -> list of (compound, document, delta)
for t in TARGETS:
    out = []
    for mid, docs in store[t].items():
        for doc, d in docs.items():
            if d.get("bind") and d.get("func"):
                out.append((mid, doc, float(np.median(d["func"])) - float(np.median(d["bind"]))))
    pairs[t] = out

percomp = {}        # target -> {compound: delta}
for t in TARGETS:
    agg = collections.defaultdict(list)
    for mid, doc, d in pairs[t]:
        agg[mid].append(d)
    percomp[t] = {m: float(np.median(v)) for m, v in agg.items()}

res = {}

# ------------------------------------------------- 1. proportion positive
print("=" * 84)
print("1.  HOW MANY COMPOUNDS MOVE IN THE SAME DIRECTION?")
print("=" * 84)
print(f"  {'target':7}{'n':>5}{'positive':>10}{'%':>7}{'95% CI (exact)':>20}{'sign test p':>14}")
for t in TARGETS:
    d = np.array(list(percomp[t].values()))
    k = int((d > 0).sum())
    n = int(d.size)
    ci = stats.binomtest(k, n, 0.5).proportion_ci(method="exact")
    p = stats.binomtest(k, n, 0.5).pvalue
    print(f"  {t:7}{n:5}{k:10}{100*k/n:7.1f}   [{100*ci.low:4.1f}%, {100*ci.high:4.1f}%]{p:14.2g}")
    res.setdefault("proportion_positive", {})[t] = {
        "n": n, "positive": k, "pct": 100 * k / n,
        "ci95": [float(ci.low), float(ci.high)], "sign_test_p": float(p)}

# ------------------------------------------------- 2. sign-flip permutation
print("\n" + "=" * 84)
print("2.  SIGN-FLIP PERMUTATION TEST")
print("    Under the null the two modality labels are exchangeable within a compound,")
print("    so each delta is equally likely to be +d or -d. 20,000 relabellings.")
print("=" * 84)
NPERM = 20000
for t in TARGETS:
    d = np.array(list(percomp[t].values()))
    obs = float(np.median(d))
    flips = rng.choice([-1.0, 1.0], size=(NPERM, d.size))
    null = np.median(flips * d, axis=1)
    p = float((np.abs(null) >= abs(obs)).mean())
    p = max(p, 1 / NPERM)
    print(f"  {t:7} observed median {obs:+.3f}   null median range "
          f"[{null.min():+.3f}, {null.max():+.3f}]   two-sided p "
          f"{'< ' + format(1/NPERM, '.0e') if p <= 1/NPERM else format(p, '.4f')}")
    res.setdefault("permutation", {})[t] = {"observed": obs, "p": p,
                                            "null_abs_max": float(np.abs(null).max())}

# ------------------------------------------------- 3. paper as the unit
print("\n" + "=" * 84)
print("3.  PAPER AS THE UNIT OF ANALYSIS")
print("    Addresses the objection that 75 compounds come from only 5 papers.")
print("=" * 84)
esh = json.load(open("within_doc_3targets.json"))["eshleman"]
for t in TARGETS:
    bydoc = collections.defaultdict(list)
    for mid, doc, d in pairs[t]:
        bydoc[doc].append(d)
    meds = {DOCNAME[k]: float(np.median(v)) for k, v in bydoc.items()
            if k in DOCNAME and len(v) >= 5}
    meds["Eshleman 2017"] = esh[t]["blocker_median"]
    vals = np.array(list(meds.values()))
    k = int((vals > 0).sum())
    n = vals.size
    p = stats.binomtest(k, n, 0.5).pvalue
    print(f"  {t:7} {k}/{n} sources positive   median of source medians "
          f"{np.median(vals):+.3f}   sign test p={p:.3f}")
    print("           " + "  ".join(f"{a} {b:+.2f}" for a, b in meds.items()))
    res.setdefault("paper_level", {})[t] = {"n_sources": n, "positive": k,
                                            "median_of_medians": float(np.median(vals)),
                                            "sign_test_p": float(p), "sources": meds}

# ------------------------------------------------- 4. Cliff's delta
def cliffs(a, b):
    a, b = np.asarray(a), np.asarray(b)
    gt = sum((x > b).sum() for x in a)
    lt = sum((x < b).sum() for x in a)
    return (gt - lt) / (a.size * b.size)

print("\n" + "=" * 84)
print("4.  EFFECT SIZE BETWEEN TARGETS  (Cliff's delta; |d|>0.47 is 'large')")
print("=" * 84)
for other in ("DAT", "SERT"):
    dl = cliffs(list(percomp["NET"].values()), list(percomp[other].values()))
    u = stats.mannwhitneyu(list(percomp["NET"].values()), list(percomp[other].values()))
    print(f"  NET vs {other:5} Cliff's delta {dl:+.3f}   Mann-Whitney p={u.pvalue:.2g}")
    res.setdefault("cliffs_delta", {})[f"NET_vs_{other}"] = {
        "delta": float(dl), "mannwhitney_p": float(u.pvalue)}

# ------------------------------------------------- 5. leave-one-document-out
print("\n" + "=" * 84)
print("5.  LEAVE-ONE-DOCUMENT-OUT, ALL THREE TARGETS")
print("=" * 84)
for t in TARGETS:
    base = float(np.median(list(percomp[t].values())))
    print(f"  {t}  (all documents: {base:+.3f})")
    worst = None
    for doc, name in DOCNAME.items():
        keep = [d for mid, dc, d in pairs[t] if dc != doc]
        if len(keep) < 5:
            print(f"    drop {name:<16} -- too few remaining")
            continue
        m = float(np.median(keep))
        print(f"    drop {name:<16} n={len(keep):3}  median {m:+.3f}  "
              f"shift {m - base:+.3f}")
        if worst is None or abs(m) < abs(worst[1]):
            worst = (name, m)
    if worst:
        print(f"    -> closest to zero: dropping {worst[0]} leaves {worst[1]:+.3f}")
        res.setdefault("lodo", {})[t] = {"base": base, "worst_source": worst[0],
                                         "worst_value": worst[1]}

# ------------------------------------------------- 6. structural census
print("\n" + "=" * 84)
print("6.  WHAT THE 81 NET PAIRED COMPOUNDS ACTUALLY ARE")
print("=" * 84)
CLASSES = [
    ("alpha-pyrrolidinophenone", "[c]C(=O)[CX4;!R]([#6])[NX3;R]1[CH2][CH2][CH2][CH2]1"),
    ("tropane / azabicyclooctane", "C1CC2CCC1N2"),
    ("fused isoindole (mazindol-type)", "c1ccc2c(c1)C(O)N1CCN=C21"),
    ("mazindol-type, any ring size", "c1ccc2c(c1)[CX4,CX3]N1CC[NX2]=C21"),
    ("benzhydryl / diarylmethyl", "[CX4]([c])([c])[#6,#8]"),
    ("piperazine", "N1CCNCC1"),
    ("acyclic arylethylamine (releaser)", "[c]-[CX4;!R]-[CX4;!R]-[NX3;!R;!$(N-C=O)]"),
]
pats = [(n, Chem.MolFromSmarts(s)) for n, s in CLASSES]
rec = list(csv.DictReader(open("releaser_blocker.csv")))
counts = collections.Counter()
mws = []
for r in rec:
    m = Chem.MolFromSmiles(r["smiles"])
    if m is None:
        counts["unparseable"] += 1
        continue
    mws.append(Descriptors.MolWt(m))
    hit = [n for n, p in pats if p is not None and m.HasSubstructMatch(p)]
    counts[hit[0] if hit else "other scaffold"] += 1
mws = np.array(mws)
for k, v in counts.most_common():
    print(f"    {k:<36}{v:4}")
print(f"\n  molecular weight: min {mws.min():.0f}  p25 {np.percentile(mws,25):.0f}  "
      f"median {np.median(mws):.0f}  p75 {np.percentile(mws,75):.0f}  max {mws.max():.0f}")
print(f"  below 230 Da (amphetamine-type size range): {(mws < 230).sum()} of {mws.size}")
res["structural_census"] = {"counts": dict(counts), "mw_median": float(np.median(mws)),
                            "mw_min": float(mws.min()), "mw_max": float(mws.max()),
                            "n_below_230Da": int((mws < 230).sum())}

json.dump(res, open("finding3_detail.json", "w"), indent=2)
print("\nwrote finding3_detail.json")
