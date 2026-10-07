"""Cui et al., Eur J Med Chem 310 (2026) 118771 — N-aryl-N-acyl diamines.

Transcribed from Tables 1 and 2 of the PDF, cross-checked against the drawn R-group
structures rendered from pages 5-6 and against the compound names in experimental
section 4.2.

Assay, verbatim from section 4.3.3 ("The monoamine transporter assay (SERT, DAT and
NET)"): "HEK293 cells stably expressing hSERT, hNET, or hDAT were seeded into 384-well
plates ... Test compounds and the reference standard were dissolved in assay buffer ...
serially diluted 4-fold to generate 10-point concentration curves. The top concentrations
were 20 uM for test compounds". So: HUMAN transporters, functional uptake inhibition,
IC50 by nonlinear regression. The 139 occurrences of "rat" in the paper are the
pharmacokinetics and the 8-OH-DPAT behavioural model, not this assay.

Scaffold: R2-C(=O)-N(Ar)-(cyclic amine). In Table 1 the amine is piperidin-4-yl and Ar
varies; in Table 2 Ar is fixed as benzothiophen-4-yl and the amine varies.
Compounds 18, 19 and 20 were tested as racemates, so no stereochemistry is assigned.
"""
import csv, json, math
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdMolDescriptors

RDLogger.DisableLog("rdApp.*")

PIP = "C1CCNCC1"                       # piperidin-4-yl, attachment at C4
BT4 = "c1cccc2sccc12"                  # benzo[b]thiophen-4-yl
BT5 = "c1cc2ccsc2cc1"                  # placeholder, replaced below
BT7 = "c1cccc2ccsc12"                  # benzo[b]thiophen-7-yl

# built explicitly so each attachment point is unambiguous
ROWS = [
    # n, SMILES, SERT, NET, DAT   ('>' prefix marks a censored bound)
    (1,  "CCC(=O)N(C1CCNCC1)c1ccc(Cl)c(OC)c1",          "9.8",   "64.5",  "10509.0"),
    (2,  "CCC(=O)N(C1CCNCC1)c1cccc2ccccc12",            "21.7",  "140.9", "63463.0"),
    (3,  "CCC(=O)N(C1CCNCC1)c1ccc2sccc2c1",             "1.4",   "38.9",  "400.0"),
    (4,  "CCC(=O)N(C1CCNCC1)c1cccc2ccsc12",             "2.5",   "5.3",   "4827.0"),
    (5,  "CC(=O)N(C1CCNCC1)c1cccc2ccsc12",              "10.9",  "80.0",  ">20000"),
    (6,  "CC(C)C(=O)N(C1CCNCC1)c1cccc2ccsc12",          "1.9",   "2.6",   "1383.0"),
    (7,  "CCC(=O)N(C1CCNCC1)c1cccc2sccc12",             "11.2",  "32.0",  ">20000"),
    (8,  "CC(=O)N(C1CCNCC1)c1cccc2sccc12",              "42.6",  "390.6", ">20000"),
    (9,  "CCCC(=O)N(C1CCNCC1)c1cccc2sccc12",            "12.6",  "44.6",  ">20000"),
    (10, "CC(C)C(=O)N(C1CCNCC1)c1cccc2sccc12",          "6.4",   "3.3",   "8165.0"),
    (11, "O=C(C1CC1)N(C1CCNCC1)c1cccc2sccc12",          "14.4",  "49.1",  "19233.0"),
    (12, "COC(=O)N(C1CCNCC1)c1cccc2sccc12",             "7.8",   "16.4",  "8413.0"),
    (13, "COCC(=O)N(C1CCNCC1)c1cccc2sccc12",            "43.12", "193.4", ">100000"),
    (14, "O=C(c1ccccc1)N(C1CCNCC1)c1cccc2sccc12",       "60.6",  "2392.0", ">20000"),
    (15, "CS(=O)(=O)N(C1CCNCC1)c1cccc2sccc12",          ">10000", "3575.0", ">100000"),
    (16, "CCC(=O)N(C1CC2(CNC2)C1)c1cccc2sccc12",        "1003.0", "874.9", "6557.0"),
    (17, "CCC(=O)N(C1CNC1)c1cccc2sccc12",               ">10000", "14559", ">100000"),
    (18, "CCC(=O)N(C1CCNC1)c1cccc2sccc12",              "3.5",   "2.6",   "2662.0"),
    (19, "CC(C)C(=O)N(C1CCNC1)c1cccc2sccc12",           "2.0",   "2.9",   "1292.0"),
    (20, "CCC(=O)N(C1CCNC(C)C1)c1cccc2sccc12",          "2721.0", ">20000", ">20000"),
]

# structural checks against statements in the paper
CHECK_FORMULA = {6: "C17H22N2OS"}      # HRMS in section 4.2.6
P75 = {"DAT": 7.44, "NET": 7.72, "SERT": 8.05}

out = []
print(f"{'#':>3} {'formula':<14}{'MW':>7}  {'pSERT':>6}{'pNET':>7}{'pDAT':>7}   smiles")
for n, smi, s, net, dat in ROWS:
    m = Chem.MolFromSmiles(smi)
    assert m is not None, f"compound {n} unparseable"
    f = rdMolDescriptors.CalcMolFormula(m).replace("+", "").replace("-", "")
    if n in CHECK_FORMULA:
        assert f == CHECK_FORMULA[n], f"compound {n}: {f} != {CHECK_FORMULA[n]}"
        print(f"  [formula check] compound {n}: {f} matches the paper's HRMS")
    rec = {"compound": n, "smiles": Chem.MolToSmiles(m),
           "inchikey": Chem.MolToInchiKey(m), "formula": f,
           "mw": round(Descriptors.MolWt(m), 2)}
    for t, v in (("SERT", s), ("NET", net), ("DAT", dat)):
        cens = v.startswith(">")
        nm = float(v.lstrip(">"))
        rec[f"IC50_{t}_nM"] = v
        rec[f"p{t}"] = "" if cens else round(9.0 - math.log10(nm), 3)
        rec[f"cens_{t}"] = int(cens)
    out.append(rec)
    print(f"{n:>3} {f:<14}{rec['mw']:7.1f}  "
          f"{rec['pSERT'] if rec['pSERT']!='' else '  cens':>6}"
          f"{rec['pNET'] if rec['pNET']!='' else '  cens':>7}"
          f"{rec['pDAT'] if rec['pDAT']!='' else '  cens':>7}   {rec['smiles']}")

assert len({r["inchikey"] for r in out}) == 20, "duplicate structures"

with open("ejmc2026_cui.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(out[0].keys()))
    w.writeheader(); w.writerows(out)

print("\n" + "=" * 78)
print("COVERAGE AGAINST THE TRAINING 75th PERCENTILE")
print("=" * 78)
summary = {}
for t in ("DAT", "NET", "SERT"):
    vals = [r[f"p{t}"] for r in out if r[f"p{t}"] != ""]
    hi = [r["compound"] for r in out if r[f"p{t}"] != "" and r[f"p{t}"] > P75[t]]
    print(f"  {t:5} uncensored n={len(vals):2}  max {max(vals):.2f}  "
          f"above p75 ({P75[t]}): {len(hi)}  -> compounds {hi}")
    summary[t] = {"n_uncensored": len(vals), "max": max(vals),
                  "n_above_p75": len(hi), "compounds_above_p75": hi}
json.dump(summary, open("ejmc2026_coverage.json", "w"), indent=2)
print("\nwrote ejmc2026_cui.csv, ejmc2026_coverage.json")
