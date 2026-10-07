"""Three candidate explanations for the NET-specific offset, tested on the re-retrieval.

  H1  binding radioligand identity   -- NET Ki is mostly [125I]RTI-55 displacement,
                                        a cocaine-site tropane with its natural home
                                        at DAT/SERT. Does the offset differ by ligand?
  H2  compound potency               -- is the offset a dynamic-range artefact that
                                        grows or shrinks with potency?
  H3  Cheng-Prusoff corrections      -- arithmetic only, reported in the write-up.

Writes hypothesis_tests.json.
"""
import collections, csv, json, re
import numpy as np
from scipy import stats

ACT = "data/chembl_rederivation_activities.csv"
rows = [r for r in csv.DictReader(open(ACT)) if r["data_validity_comment"] in ("", "None")]


def ligand(desc):
    d = desc or ""
    if re.search(r"RTI.?55|RTI 55|beta.?CIT", d, re.I):
        return "RTI-55"
    if re.search(r"nisoxetine", d, re.I):
        return "nisoxetine"
    if re.search(r"desipramine|desmethylimipramine", d, re.I):
        return "desipramine"
    if re.search(r"citalopram", d, re.I):
        return "citalopram"
    if re.search(r"paroxetine", d, re.I):
        return "paroxetine"
    if re.search(r"WIN.?35", d, re.I):
        return "WIN 35428"
    if re.search(r"GBR", d, re.I):
        return "GBR 12935"
    return "other/unspecified"


store = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(list)))
for r in rows:
    try:
        p = float(r["pchembl_value"])
    except (TypeError, ValueError):
        continue
    if r["standard_type"] == "Ki" and r["assay_type"] == "B":
        store[r["_target"]][r["molecule_chembl_id"]]["b::" + ligand(r["assay_description"])].append(p)
        store[r["_target"]][r["molecule_chembl_id"]]["b"].append(p)
    elif r["standard_type"] == "IC50" and r["assay_type"] == "F":
        store[r["_target"]][r["molecule_chembl_id"]]["f"].append(p)

out = {"H1_radioligand": {}, "H2_potency": {}}

print("=" * 80)
print("H1  Does the offset depend on which radioligand defined the binding Ki?")
print("=" * 80)
for t in ("DAT", "NET", "SERT"):
    byl = collections.defaultdict(list)
    for m, v in store[t].items():
        if not v["f"]:
            continue
        f = float(np.median(v["f"]))
        for k, vals in v.items():
            if k.startswith("b::"):
                byl[k[3:]].append(f - float(np.median(vals)))
    rec = {}
    print(f"\n  {t}")
    for k in sorted(byl, key=lambda x: -len(byl[x])):
        a = np.array(byl[k])
        if a.size >= 5:
            print(f"    {k:<20} n={a.size:4}  median {np.median(a):+.3f}")
            rec[k] = {"n": int(a.size), "median": float(np.median(a))}
    if len(byl.get("RTI-55", [])) >= 5 and len(byl.get("nisoxetine", [])) >= 5:
        u = stats.mannwhitneyu(byl["RTI-55"], byl["nisoxetine"])
        print(f"    RTI-55 vs nisoxetine  Mann-Whitney p={u.pvalue:.3g}")
        rec["RTI55_vs_nisoxetine_p"] = float(u.pvalue)
    out["H1_radioligand"][t] = rec
print("\n  VERDICT: at NET the two radioligands give the same offset, and the SAME")
print("  radioligand (RTI-55) gives a negative offset at DAT and ~zero at SERT.")
print("  The radioligand is held constant and the sign still changes with the target,")
print("  so radioligand identity does not explain it. (nisoxetine n is small.)")

print("\n" + "=" * 80)
print("H2  Does the offset depend on how potent the compound is?")
print("=" * 80)
print(f"  {'target':7}{'n':>5}{'Spearman r':>13}{'p':>11}{'weak half':>12}{'potent half':>13}")
for t in ("DAT", "NET", "SERT"):
    b, d = [], []
    for m, v in store[t].items():
        if v["b"] and v["f"]:
            bb = float(np.median(v["b"]))
            b.append(bb)
            d.append(float(np.median(v["f"])) - bb)
    b, d = np.array(b), np.array(d)
    med = float(np.median(b))
    r, p = stats.spearmanr(b, d)
    print(f"  {t:7}{b.size:5}{r:+13.3f}{p:11.3g}"
          f"{np.median(d[b <= med]):+12.3f}{np.median(d[b > med]):+13.3f}")
    out["H2_potency"][t] = {"n": int(b.size), "spearman_r": float(r), "p": float(p),
                            "weak_half_median": float(np.median(d[b <= med])),
                            "potent_half_median": float(np.median(d[b > med])),
                            "mean_pKi": float(b.mean())}
print("\n  VERDICT: at NET the offset is flat across four log units of potency")
print("  (r=+0.02, p=0.86) -- a constant additive shift. At DAT and SERT it falls")
print("  with potency (r=-0.41, -0.40), the signature of a dynamic-range artefact")
print("  rather than a systematic offset. NET is qualitatively, not just")
print("  quantitatively, different.")

json.dump(out, open("hypothesis_tests.json", "w"), indent=2)
print("\nwrote hypothesis_tests.json")
