"""Independent releaser-vs-blocker labelling for the 81 NET paired compounds.

The Ki/IC50 ratio is itself the field's standard releaser/blocker discriminator
(Eshleman et al. 2017), so using it here would be circular. This script instead
pulls, for each compound, every ChEMBL activity record at DAT/NET/SERT and keeps
the assay descriptions, so release/efflux assays can be identified by assay text
rather than by the ratio under test.
"""
import csv, json, time, urllib.parse, urllib.request

BASE = "https://www.ebi.ac.uk/chembl/api/data"
TARGETS = {"CHEMBL238": "DAT", "CHEMBL222": "NET", "CHEMBL228": "SERT"}


def get(path, **params):
    url = f"{BASE}/{path}?" + urllib.parse.urlencode(params)
    for attempt in range(5):
        try:
            with urllib.request.urlopen(url, timeout=90) as r:
                return json.loads(r.read().decode())
        except Exception as exc:
            if attempt == 4:
                raise
            time.sleep(2 * (attempt + 1))
            last = exc
    raise last


rows = list(csv.DictReader(open("data/net_paired_compounds.csv")))
ids = [r["molecule_chembl_id"] for r in rows]
print(f"{len(ids)} compounds")

# ---------------------------------------------------------------- 1. names
names = {}
for i in range(0, len(ids), 25):
    chunk = ids[i:i + 25]
    d = get("molecule.json", molecule_chembl_id__in=",".join(chunk), limit=100,
            only="molecule_chembl_id,pref_name,molecule_synonyms")
    for m in d["molecules"]:
        syn = [s["molecule_synonym"] for s in (m.get("molecule_synonyms") or [])]
        names[m["molecule_chembl_id"]] = {"pref_name": m.get("pref_name"),
                                          "synonyms": sorted(set(syn))}
    print(f"  names {i + len(chunk)}/{len(ids)}")
json.dump(names, open("net81_names.json", "w"), indent=2)

# ------------------------------------------------- 2. all MAT assay records
acts = []
for i in range(0, len(ids), 10):
    chunk = ids[i:i + 10]
    for tid in TARGETS:
        offset = 0
        while True:
            d = get("activity.json", molecule_chembl_id__in=",".join(chunk),
                    target_chembl_id=tid, limit=1000, offset=offset,
                    only=("molecule_chembl_id,target_chembl_id,assay_chembl_id,"
                          "assay_type,assay_description,standard_type,"
                          "standard_relation,standard_value,standard_units,"
                          "pchembl_value,document_chembl_id,bao_label"))
            batch = d["activities"]
            for a in batch:
                a["target_name"] = TARGETS[tid]
            acts.extend(batch)
            if len(batch) < 1000:
                break
            offset += 1000
    print(f"  activities {i + len(chunk)}/{len(ids)}  cumulative {len(acts)}")

with open("net81_mat_activities.csv", "w", newline="") as fh:
    cols = ["molecule_chembl_id", "target_name", "target_chembl_id", "assay_chembl_id",
            "assay_type", "standard_type", "standard_relation", "standard_value",
            "standard_units", "pchembl_value", "document_chembl_id", "bao_label",
            "assay_description"]
    w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
    w.writeheader()
    w.writerows(acts)
print(f"wrote net81_mat_activities.csv  ({len(acts)} records)")
