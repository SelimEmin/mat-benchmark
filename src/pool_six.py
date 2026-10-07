"""Pool the within-document offset across six sources: the five ChEMBL papers plus
Eshleman et al. 2017's uptake blockers.

Definition kept consistent with the five-paper version: one value per COMPOUND, equal to
the median of its within-document (functional - binding) differences across whichever
sources report it. Compounds appearing in more than one source are therefore counted
once, so the overlap between Eshleman's cathinones and the ChEMBL series does not
double-count. Overlap is detected by ChEMBL preferred name and synonym, and reported.
"""
import collections, csv, json, math, re, time, urllib.parse, urllib.request
import numpy as np
from scipy import stats

ACT = "data/chembl_rederivation_activities.csv"
BASE = "https://www.ebi.ac.uk/chembl/api/data"
rng = np.random.default_rng(20260930)


def get(path, **params):
    url = f"{BASE}/{path}?" + urllib.parse.urlencode(params)
    for attempt in range(5):
        try:
            with urllib.request.urlopen(url, timeout=90) as r:
                return json.loads(r.read().decode())
        except Exception:
            if attempt == 4:
                raise
            time.sleep(2 * (attempt + 1))


# ------------------------------------------------- 1. ChEMBL within-document values
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

chembl = {}            # target -> {molecule_chembl_id: delta}
for t in ("DAT", "NET", "SERT"):
    out = {}
    for mid, docs in store[t].items():
        ds = [float(np.median(d["func"])) - float(np.median(d["bind"]))
              for d in docs.values() if d.get("bind") and d.get("func")]
        if ds:
            out[mid] = float(np.median(ds))
    chembl[t] = out
    print(f"ChEMBL within-document {t}: {len(out)} compounds")

# ------------------------------------------------- 2. names, to detect overlap
ids = sorted(set().union(*[set(v) for v in chembl.values()]))
print(f"\nresolving names for {len(ids)} distinct compounds")
names = {}
for i in range(0, len(ids), 25):
    chunk = ids[i:i + 25]
    d = get("molecule.json", molecule_chembl_id__in=",".join(chunk), limit=100,
            only="molecule_chembl_id,pref_name,molecule_synonyms")
    for m in d["molecules"]:
        syn = [s["molecule_synonym"] for s in (m.get("molecule_synonyms") or [])]
        if m.get("pref_name"):
            syn.append(m["pref_name"])
        names[m["molecule_chembl_id"]] = {re.sub(r"[^a-z0-9]", "", s.lower()) for s in syn}

# ------------------------------------------------- 3. Eshleman blockers
ESH_BLOCKERS = [
    # name, aliases, DAT ratio, SERT ratio, NET ratio
    ("alpha-PPP",       ["alphappp", "ppp", "pyrrolidinopropiophenone"],      2.4,  0.9,   6.7),
    ("alpha-PBP",       ["alphapbp", "pbp", "pyrrolidinobutiophenone"],       1.9,  2.4,   3.6),
    ("alpha-PVP",       ["alphapvp", "pvp", "pyrrolidinovalerophenone"],      1.1,  1.2,   2.7),
    ("alpha-PHP",       ["alphaphp", "php", "pyrrolidinohexiophenone"],       0.7,  0.8,   9.3),
    ("PV-8",            ["pv8"],                                             1.0,  1.5,  11.4),
    ("4-MePPP",         ["4meppp", "meppp"],                                 2.0,  4.8,   5.5),
    ("4-MeO-alpha-PVP", ["4meoalphapvp", "meopvp"],                          2.8,  1.5,   5.0),
    ("3,4-MDPPP",       ["34mdppp", "mdppp"],                                2.7,  1.5,   5.6),
    ("3,4-MDPBP",       ["34mdpbp", "mdpbp"],                                1.2,  3.3,   5.2),
    ("alpha-PVT",       ["alphapvt", "pvt"],                                 1.8,  0.7,  12.9),
    ("MDPV",            ["mdpv", "methylenedioxypyrovalerone"],              1.5,  0.91,  5.7),
    ("Naphyrone",       ["naphyrone", "naphthylpyrovalerone"],               0.85, 0.25,  1.7),
    ("Pentedrone",      ["pentedrone"],                                      1.9,  1.8,   8.2),
    ("Pentylone",       ["pentylone", "methylenedioxypentedrone"],           2.4,  4.2,  12.6),
    ("Cocaine",         ["cocaine", "cocainehydrochloride"],                 1.9,  1.9,   7.0),
    ("Mazindol",        ["mazindol"],                                        2.0,  2.0,   5.0),
]
ESH = {"DAT": {}, "SERT": {}, "NET": {}}
for nm, aliases, dat, sert, net in ESH_BLOCKERS:
    for t, v in (("DAT", dat), ("SERT", sert), ("NET", net)):
        ESH[t][nm] = math.log10(v)

alias_map = {nm: set(a) | {re.sub(r"[^a-z0-9]", "", nm.lower())}
             for nm, a, *_ in ESH_BLOCKERS}

# ------------------------------------------------- 4. overlap
overlap = {}
for t in ("DAT", "NET", "SERT"):
    hits = {}
    for mid in chembl[t]:
        syns = names.get(mid, set())
        for nm, al in alias_map.items():
            if syns & al:
                hits[nm] = mid
    overlap[t] = hits

print("\nCompound overlap between the five ChEMBL papers and Eshleman's 16 blockers:")
for t in ("DAT", "NET", "SERT"):
    print(f"  {t}: {len(overlap[t])}  {sorted(overlap[t])}")

# ------------------------------------------------- 5. pool, one value per compound
print("\n" + "=" * 86)
print("POOLED ACROSS SIX SOURCES  (one value per compound; overlaps merged)")
print("=" * 86)
res = {}
for t in ("DAT", "NET", "SERT"):
    vals = collections.defaultdict(list)       # key -> list of deltas
    for mid, d in chembl[t].items():
        vals[mid].append(d)
    for nm, d in ESH[t].items():
        key = overlap[t].get(nm, "ESH::" + nm)  # merge onto the ChEMBL id when shared
        vals[key].append(d)
    arr = np.array([float(np.median(v)) for v in vals.values()])
    boot = [np.median(rng.choice(arr, arr.size, replace=True)) for _ in range(5000)]
    ci = (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)))
    wp = float(stats.wilcoxon(arr)[1])
    five = np.array(list(chembl[t].values()))
    res[t] = {"n": int(arr.size), "median": float(np.median(arr)), "ci95": ci,
              "wilcoxon_p": wp, "n_five_papers": int(five.size),
              "median_five_papers": float(np.median(five)),
              "n_eshleman": len(ESH[t]), "n_overlap": len(overlap[t]),
              "overlap_names": sorted(overlap[t])}
    print(f"  {t:<6} n={arr.size:4}  median {np.median(arr):+.3f}"
          f"  95% CI [{ci[0]:+.3f}, {ci[1]:+.3f}]  Wilcoxon p={wp:.2g}"
          f"   (five papers alone: n={five.size}, {np.median(five):+.3f})")

kw = stats.kruskal(*[np.array([float(np.median(v)) for v in
                               (lambda t: {**{m: [d] for m, d in chembl[t].items()},
                                           **{overlap[t].get(nm, 'E' + nm): [d]
                                              for nm, d in ESH[t].items()}})(t).values()])
                     for t in ("DAT", "NET", "SERT")])
print(f"\n  Kruskal-Wallis across the three pooled sets: H={kw.statistic:.1f}, p={kw.pvalue:.2g}")
res["kruskal"] = {"H": float(kw.statistic), "p": float(kw.pvalue)}

json.dump(res, open("pool_six.json", "w"), indent=2)
print("\nwrote pool_six.json")
