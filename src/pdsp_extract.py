"""Independent extraction of the human monoamine-transporter slice of the PDSP Ki
database, as redistributed by BindingDB.

BindingDB only ingests rows it can map to a UniProt accession, and the three accessions
used here are human-specific, so species is guaranteed upstream rather than parsed from
a free-text field.

    Q01959  SLC6A3  DAT
    P23975  SLC6A2  NET
    P31645  SLC6A4  SERT
"""
import csv, math
from collections import defaultdict

csv.field_size_limit(10 ** 7)
UNI = {"Q01959": "DAT", "P23975": "NET", "P31645": "SERT"}
SRC = "BindingDB_PDSPKi.tsv"

f = open(SRC, encoding="utf-8", errors="replace")
hdr = f.readline().rstrip("\n").split("\t")
I = {h: i for i, h in enumerate(hdr)}
C_SMI = I["Ligand SMILES"]
C_ORG = I["Target Source Organism According to Curator or DataSource"]
C_KI = I["Ki (nM)"]
C_IC50 = I["IC50 (nM)"]
C_SRC = I["Curation/DataSource"]
C_PMID = I["PMID"]
C_TGT = I["Target Name"]
UNICOLS = [I[h] for h in hdr if h.startswith("UniProt (SwissProt) Primary ID of Target Chain")]

rows, nline, censored = [], 0, 0
for line in f:
    nline += 1
    p = line.rstrip("\n").split("\t")
    if len(p) < max(UNICOLS) + 1:
        continue
    tgt = None
    for c in UNICOLS:
        if p[c].strip() in UNI:
            tgt = UNI[p[c].strip()]
            break
    if tgt is None:
        continue
    raw = (p[C_KI] or "").strip() or (p[C_IC50] or "").strip()
    kind = "Ki" if (p[C_KI] or "").strip() else "IC50"
    if not raw:
        continue
    if raw[0] in "<>":            # censored
        censored += 1
        continue
    try:
        nM = float(raw)
    except ValueError:
        continue
    if nM <= 0:
        continue
    rows.append({"smiles": p[C_SMI].strip(), "target": tgt, "kind": kind,
                 "nM": nM, "p": 9.0 - math.log10(nM),
                 "organism": p[C_ORG].strip(), "source": p[C_SRC].strip(),
                 "pmid": p[C_PMID].strip(), "target_name": p[C_TGT].strip()})

print(f"scanned {nline} rows; kept {len(rows)} human MAT measurements "
      f"({censored} censored dropped)")
from collections import Counter
print("  organism field:", Counter(r["organism"] for r in rows).most_common(4))
print("  per target:", Counter(r["target"] for r in rows))
print("  measurement kind:", Counter(r["kind"] for r in rows))
print("  distinct SMILES:", len({r["smiles"] for r in rows}))
print("  distinct PMIDs:", len({r["pmid"] for r in rows if r["pmid"]}))

# ---------------------------------------------------- per compound x target median
import statistics
prof = defaultdict(dict)
agg = defaultdict(list)
for r in rows:
    agg[(r["smiles"], r["target"])].append(r["p"])
for (smi, t), vs in agg.items():
    prof[smi][t] = float(statistics.median(vs))

all3 = {s: d for s, d in prof.items() if len(d) == 3}
print(f"\ncompounds with a value at all three targets: {len(all3)} "
      f"(of {len(prof)} with any)")

P75 = {"DAT": 7.44, "NET": 7.72, "SERT": 8.05}      # training 75th percentiles
hi = {s: d for s, d in all3.items()
      if d["NET"] > P75["NET"] or d["SERT"] > P75["SERT"]}
print(f"of those, above the training p75 at NET or SERT: {len(hi)}")
for t in ("DAT", "NET", "SERT"):
    vals = [d[t] for d in all3.values()]
    print(f"  {t:5} n={len(vals):4}  min {min(vals):.2f}  median "
          f"{statistics.median(vals):.2f}  max {max(vals):.2f}  "
          f"above p75: {sum(v > P75[t] for v in vals)}")

with open("pdsp_mat_raw.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)
with open("pdsp_mat_profiles.csv", "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["smiles", "pDAT", "pNET", "pSERT"])
    for s, d in sorted(all3.items(), key=lambda kv: -max(kv[1].values())):
        w.writerow([s, f"{d['DAT']:.3f}", f"{d['NET']:.3f}", f"{d['SERT']:.3f}"])
print("\nwrote pdsp_mat_raw.csv, pdsp_mat_profiles.csv")
