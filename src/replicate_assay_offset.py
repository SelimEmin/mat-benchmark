#!/usr/bin/env python3
"""
Independent re-retrieval and chemotype stratification of the
binding-versus-functional assay-modality offset at DAT, NET and SERT.

Added in revision, in response to the question of whether the NET offset
reported in the frozen ChEMBL_37 analysis is a property of the transporter or
of a single chemotype.

This script is deliberately INDEPENDENT of the main pipeline. It re-retrieves
from the live ChEMBL REST API with its own curation rules, so the compound set
it builds is non-nested with the frozen one. It is a robustness check, not a
reproduction, and is reported as such in the manuscript (Section 3.5).

Curation applied here
    targets      human DAT CHEMBL238, NET CHEMBL222, SERT CHEMBL228
    binding      standard_type Ki   from an assay ChEMBL types 'B'
    functional   standard_type IC50 from an assay ChEMBL types 'F'
    filters      target_organism = Homo sapiens; pchembl_value present;
                 standard_relation '=' (uncensored only);
                 records carrying a ChEMBL data-validity comment dropped
    aggregation  per compound, median pchembl_value within each modality
    offset       delta = functional - binding  (positive = functional reports
                 higher potency)

Usage
    python replicate_assay_offset.py fetch     # ~3 min, writes activities.csv
    python replicate_assay_offset.py analyse   # writes results + figure

Requires: rdkit, scipy, numpy, matplotlib
"""
import sys, json, time, csv, collections, urllib.parse, urllib.request
import numpy as np

BASE = "https://www.ebi.ac.uk/chembl/api/data/activity.json"
TARGETS = {"CHEMBL238": "DAT", "CHEMBL222": "NET", "CHEMBL228": "SERT"}
FIELDS = ["molecule_chembl_id", "canonical_smiles", "standard_type",
          "standard_relation", "pchembl_value", "assay_type",
          "target_chembl_id", "target_pref_name", "target_organism",
          "data_validity_comment", "assay_description", "document_chembl_id"]
ACT_CSV = "activities.csv"
SEED = 0


# --------------------------------------------------------------- fetch ----
def _get(url, tries=4):
    for i in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=90) as r:
                return json.loads(r.read().decode())
        except Exception:
            if i == tries - 1:
                raise
            time.sleep(3 * (i + 1))


def fetch():
    rows = []
    for tid, short in TARGETS.items():
        for stype in ("Ki", "IC50"):
            q = {"target_chembl_id": tid, "standard_type": stype,
                 "target_organism": "Homo sapiens",
                 "pchembl_value__isnull": "false",
                 "standard_relation": "=", "limit": "1000",
                 "only": ",".join(FIELDS)}
            url = BASE + "?" + urllib.parse.urlencode(q)
            n = 0
            while url:
                d = _get(url)
                for a in d["activities"]:
                    a["_target"] = short
                    rows.append(a)
                    n += 1
                nxt = d["page_meta"].get("next")
                url = ("https://www.ebi.ac.uk" + nxt) if nxt else None
            print(f"  {short:4s} {stype:5s} {n:6d} records", flush=True)
    with open(ACT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS + ["_target"],
                           extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"wrote {ACT_CSV} ({len(rows)} rows)")


# ------------------------------------------------------------- analyse ----
def paired_sets():
    rows = [r for r in csv.DictReader(open(ACT_CSV))
            if r["data_validity_comment"] in ("", "None")]
    store = collections.defaultdict(
        lambda: collections.defaultdict(lambda: collections.defaultdict(list)))
    smi = {}
    for r in rows:
        st, at = r["standard_type"], r["assay_type"]
        if st == "Ki" and at == "B":
            mod = "bind"
        elif st == "IC50" and at == "F":
            mod = "func"
        else:
            continue
        try:
            p = float(r["pchembl_value"])
        except (TypeError, ValueError):
            continue
        store[r["molecule_chembl_id"]][r["_target"]][mod].append(p)
        smi.setdefault(r["molecule_chembl_id"], r["canonical_smiles"])

    out = {}
    for tgt in ("DAT", "NET", "SERT"):
        recs = []
        for mid, per_t in store.items():
            d = per_t.get(tgt)
            if d and "bind" in d and "func" in d:
                b, f = float(np.median(d["bind"])), float(np.median(d["func"]))
                recs.append({"molecule_chembl_id": mid, "smiles": smi[mid],
                             "p_binding": b, "p_functional": f,
                             "delta": f - b})
        out[tgt] = recs
    return out, len(rows)


def analyse():
    from scipy import stats
    from rdkit import Chem, RDLogger
    from rdkit.Chem import AllChem, DataStructs
    from rdkit.Chem.Scaffolds import MurckoScaffold
    from rdkit.ML.Cluster import Butina
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    RDLogger.DisableLog("rdApp.*")
    rng = np.random.default_rng(SEED)

    def boot_ci(d, n=4000):
        b = np.array([np.median(rng.choice(d, d.size, replace=True))
                      for _ in range(n)])
        return [float(x) for x in np.percentile(b, [2.5, 97.5])]

    paired, n_rows = paired_sets()
    res = {"records_used": n_rows, "seed": SEED, "per_target": {}}

    print(f"{'target':6s}{'n':>6s}{'median':>10s}{'95% CI':>22s}{'Wilcoxon p':>13s}")
    for tgt in ("DAT", "NET", "SERT"):
        d = np.array([r["delta"] for r in paired[tgt]])
        lo, hi = boot_ci(d)
        res["per_target"][tgt] = {
            "n": int(d.size), "median_offset": float(np.median(d)),
            "ci95": [lo, hi], "sd": float(np.std(d, ddof=1)),
            "wilcoxon_p": float(stats.wilcoxon(d).pvalue)}
        print(f"{tgt:6s}{d.size:6d}{np.median(d):+10.3f}"
              f"   [{lo:+.3f}, {hi:+.3f}]{stats.wilcoxon(d).pvalue:13.2e}")

    dd = {t: np.array([r["delta"] for r in paired[t]]) for t in paired}
    res["kruskal_across_targets_p"] = float(
        stats.kruskal(dd["DAT"], dd["NET"], dd["SERT"]).pvalue)
    res["DAT_vs_SERT_mannwhitney_p"] = float(
        stats.mannwhitneyu(dd["DAT"], dd["SERT"]).pvalue)
    print(f"\nKruskal-Wallis across targets: p = {res['kruskal_across_targets_p']:.2e}")
    print(f"DAT vs SERT (Mann-Whitney):    p = {res['DAT_vs_SERT_mannwhitney_p']:.4f}")

    # ---- chemotype stratification of NET ---------------------------------
    recs = [r for r in paired["NET"] if Chem.MolFromSmiles(r["smiles"])]
    mols = [Chem.MolFromSmiles(r["smiles"]) for r in recs]
    dl = np.array([r["delta"] for r in recs])
    gen = AllChem.GetMorganGenerator(radius=2, fpSize=2048)
    fps = [gen.GetFingerprint(m) for m in mols]
    dists = []
    for i in range(1, len(fps)):
        dists.extend(1 - s for s in
                     DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i]))
    clusters = Butina.ClusterData(dists, len(fps), 0.4, isDistData=True)

    strata = []
    for c in clusters:
        if len(c) < 5:
            continue
        idx = np.array(list(c))
        d = dl[idx]
        lo, hi = boot_ci(d)
        strata.append({
            "n": len(c), "median_offset": float(np.median(d)), "ci95": [lo, hi],
            "centroid_murcko_scaffold":
                MurckoScaffold.MurckoScaffoldSmiles(mol=mols[list(c)[0]])})
    strata.sort(key=lambda s: s["median_offset"])
    res["NET_butina_strata_tanimoto_0.6"] = strata

    # leave-one-stratum-out
    base = float(np.median(dl))
    loo = []
    for c in clusters:
        if len(c) < 5:
            continue
        keep = np.array([dl[i] for i in range(len(dl)) if i not in set(c)])
        loo.append({"dropped_n": len(c), "pooled_offset_without": float(np.median(keep)),
                    "shift": float(np.median(keep) - base)})
    res["NET_leave_one_stratum_out"] = {"pooled_all": base, "runs": loo}
    res["NET_kruskal_across_strata_p"] = float(stats.kruskal(
        *[dl[np.array(list(c))] for c in clusters if len(c) >= 5]).pvalue)

    # Murcko scaffold strata as an independent cut
    scaf = collections.defaultdict(list)
    for i, m in enumerate(mols):
        scaf[MurckoScaffold.MurckoScaffoldSmiles(mol=m)].append(i)
    ms = [{"n": len(ix), "median_offset": float(np.median(dl[np.array(ix)])),
           "scaffold": s} for s, ix in scaf.items() if len(ix) >= 5]
    ms.sort(key=lambda x: x["median_offset"])
    res["NET_murcko_strata"] = ms

    json.dump(res, open("chemotype_stratification.json", "w"), indent=2)
    with open("net_paired_compounds.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(recs[0].keys()))
        w.writeheader()
        w.writerows(recs)
    print("\nwrote chemotype_stratification.json, net_paired_compounds.csv")

    # ---- figure (manuscript palette) -------------------------------------
    CB_BLUE, CB_VERM, GREY = "#0072B2", "#D55E00", "#666666"
    plt.rcParams.update({"font.family": "sans-serif",
                         "font.sans-serif": ["Liberation Sans", "DejaVu Sans"],
                         "font.size": 9})
    LABEL = {
        "O=C(CN1CCCC1)c1ccccc1": "Pyrrolidinophenones",
        "c1ccc(-c2cnc(C3C4CCC(CC3c3ccccc3)N4)s2)cc1": "3-Aryltropanes",
        "c1ccc(C2c3ccccc3C3=NCCN32)cc1": "Fused imidazoline tricyclics",
        "c1ccc2cc(C3CC4CCC3C4)ccc2c1": "Naphthyl norbornanes",
        "O=C(CCc1ccccc1)NCCNCCOC(c1ccccc1)c1ccccc1": "Benzhydryl ether amines",
    }
    items = [(LABEL.get(s["centroid_murcko_scaffold"], "other"), s["n"],
              s["median_offset"], *s["ci95"]) for s in strata]
    items.append(("All compounds", int(dl.size), base, *boot_ci(dl)))

    fig, ax = plt.subplots(figsize=(6.3, 2.9))
    for i, (name, n, med, lo, hi) in enumerate(items):
        overall = name.startswith("All")
        col = CB_VERM if overall else CB_BLUE
        ax.plot([lo, hi], [i, i], color=col, lw=1.8, solid_capstyle="round", zorder=3)
        ax.plot([med], [i], "o", ms=6.5 if overall else 5.5, color=col,
                markeredgecolor="white", markeredgewidth=1.0, zorder=4)
        ax.text(hi + 0.05, i, f"{med:+.2f}", va="center", ha="left",
                fontsize=8.5, fontweight="bold", color=col)
    ax.axvline(0, color="#333333", lw=0.9, zorder=2)
    ax.axvline(base, color=CB_VERM, lw=0.7, ls=":", zorder=2)
    ax.set_yticks(range(len(items)))
    ax.set_yticklabels([f"{n}  (n = {c})" for n, c, *_ in items], fontsize=8.5)
    ax.set_xlabel("functional − binding (log units)", fontsize=9)
    ax.set_xlim(-0.6, 1.75)
    ax.xaxis.grid(True, color="#DDDDDD", lw=0.6)
    ax.set_axisbelow(True)
    for s_ in ("top", "right", "left"):
        ax.spines[s_].set_visible(False)
    ax.spines["bottom"].set_color("#999999")
    ax.tick_params(length=0, labelsize=8.5)
    fig.text(0.99, 0.015, "bars: 95% percentile bootstrap, 4000 resamples",
             fontsize=7, color=GREY, ha="right")
    fig.tight_layout(rect=(0, 0.04, 1, 1), pad=0.3)
    fig.savefig("fig11_chemotype.pdf")
    fig.savefig("fig11_chemotype.png", dpi=300)
    print("wrote fig11_chemotype.pdf / .png")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "analyse"
    {"fetch": fetch, "analyse": analyse}[cmd]()
