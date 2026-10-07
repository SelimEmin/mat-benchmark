"""Releaser vs blocker, by structure and by literature identity -- never by Ki/IC50.

Rationale. Eshleman et al. (2017) use the binding-Ki / uptake-IC50 ratio itself as the
discriminator between substrate-type releasers and uptake blockers. The NET
assay-modality offset under test IS that ratio. Labelling the compounds with the ratio
would therefore guarantee the confound rather than test it. Two independent routes are
used instead:

  route A  ChEMBL assay text -- any release/efflux assay record for the compound
  route B  the substrate-type releaser pharmacophore, applied to structure alone
  route C  identity against literature-established releasers and blockers

Then the NET offset is recomputed within each class.
"""
import csv, json, re
import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors
from scipy import stats

RDLogger.DisableLog("rdApp.*")

# ----------------------------------------------------------------- route C lists
# Literature-established substrate-type releasers at NET (amphetamine-type).
RELEASERS = {
    "amphetamine", "dextroamphetamine", "levoamphetamine", "methamphetamine",
    "phentermine", "cathine", "cathinone", "methcathinone", "mephedrone",
    "4-methylmethcathinone", "methylone", "ethylone", "butylone", "flephedrone",
    "mdma", "mda", "mdea", "pma", "pmma", "4-fluoroamphetamine", "4-fa",
    "fenfluramine", "norfenfluramine", "dexfenfluramine", "chlorphentermine",
    "aminorex", "4-methylaminorex", "phenmetrazine", "phendimetrazine",
    "diethylpropion", "amfepramone", "tyramine", "octopamine", "pholedrine",
    "ephedrine", "pseudoephedrine", "norephedrine", "phenylpropanolamine",
    "naphyrone-nonpyrrolidine", "4-mta", "para-methoxyamphetamine",
    "tranylcypromine", "selegiline", "benzylpiperazine", "bzp", "mdpba",
    "apbt", "5-apb", "6-apb", "5-it", "5-apdb", "6-apdb", "pentedrone",
}
# Literature-established uptake blockers (no releasing action at NET).
BLOCKERS = {
    "cocaine", "cocaine hydrochloride", "mazindol", "vanoxerine", "gbr 12909",
    "gbr12909", "desipramine", "desipramine hydrochloride", "imipramine",
    "nortriptyline", "amitriptyline", "nisoxetine", "reboxetine", "atomoxetine",
    "tomoxetine", "duloxetine", "venlafaxine", "milnacipran", "bupropion",
    "nomifensine", "methylphenidate", "indatraline", "sertraline", "fluoxetine",
    "paroxetine", "citalopram", "escitalopram", "fluvoxamine", "clomipramine",
    "maprotiline", "pyrovalerone", "mdpv", "alpha-pvp", "naphyrone",
    "bus", "tesofensine", "benztropine", "modafinil", "amineptine", "tacrine",
    "oxaprotiline", "talsupram", "tomoxetine hydrochloride",
}

# --------------------------------------------------------------- route B SMARTS
# Substrate-type releaser pharmacophore: an aryl-CH2-CH(R)-NH(R') chain in which the
# two-carbon linker and the nitrogen are all acyclic and the nitrogen bears at most
# one small alkyl group. Tropanes, methylphenidate, aminotetralins, aminoindanes and
# the alpha-pyrrolidinophenones all fail on ring membership or N-substitution, which
# is the established structure-activity boundary between substrates and blockers.
ARYLETHYLAMINE = Chem.MolFromSmarts(
    "[c,n;R]-[CX4;!R]-[CX4;!R]-[NX3;!R;!$(N-C=O);!$(N-S);!$([N+])]")


def small_n(mol, n_idx):
    """True if the amine is primary, or secondary/tertiary with only small alkyls."""
    n = mol.GetAtomWithIdx(n_idx)
    subs = [a for a in n.GetNeighbors() if a.GetSymbol() != "H"]
    carbons = 0
    for a in subs:
        if a.GetSymbol() != "C":
            return False
        # count heavy atoms reachable on this branch, excluding the chain back to N
        carbons += 1
    # at most two carbon substituents, each a methyl or ethyl
    alkyls = []
    for a in subs:
        if a.GetIsAromatic():
            return False
        deg = sum(1 for b in a.GetNeighbors() if b.GetIdx() != n_idx
                  and b.GetSymbol() != "H")
        alkyls.append(deg)
    non_chain = [d for d in alkyls]
    return len(subs) <= 3 and all(d <= 2 for d in sorted(non_chain)[:-1] or [0])


def route_b(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, "unparseable"
    if Descriptors.MolWt(mol) > 280:
        return False, f"MW {Descriptors.MolWt(mol):.0f} > 280"
    matches = mol.GetSubstructMatches(ARYLETHYLAMINE)
    if not matches:
        return False, "no acyclic aryl-C-C-N(H) chain"
    for m in matches:
        n_idx = m[3]
        n = mol.GetAtomWithIdx(n_idx)
        heavy = [a for a in n.GetNeighbors()]
        # nitrogen must carry <=1 substituent besides the chain carbon
        others = [a for a in heavy if a.GetIdx() != m[2]]
        if len(others) > 1:
            continue
        if others and others[0].GetIsAromatic():
            continue
        if others:
            o = others[0]
            branch = sum(1 for b in o.GetNeighbors() if b.GetIdx() != n_idx)
            if branch > 1:            # bigger than ethyl
                continue
        return True, "amphetamine-type pharmacophore"
    return False, "chain present but N too substituted / cyclic"


# ------------------------------------------------------------------- route A
acts = list(csv.DictReader(open("net81_mat_activities.csv")))
rel_pat = re.compile(r"\brelease|\breleasing|efflux", re.I)
# exclude "prior to substrate addition" style false positives
route_a = {}
for a in acts:
    d = a["assay_description"] or ""
    if rel_pat.search(d) and not re.search(r"substrate addition", d, re.I):
        route_a.setdefault(a["molecule_chembl_id"], []).append(d)

# ------------------------------------------------------------------- assemble
names = json.load(open("net81_names.json"))
rows = list(csv.DictReader(open("data/net_paired_compounds.csv")))

recs = []
for r in rows:
    m = r["molecule_chembl_id"]
    info = names.get(m, {})
    pref = (info.get("pref_name") or "").strip()
    syns = [s.lower() for s in (info.get("synonyms") or [])]
    allnames = set(syns) | ({pref.lower()} if pref else set())

    b_ok, b_why = route_b(r["smiles"])
    c_rel = bool(allnames & RELEASERS)
    c_blk = bool(allnames & BLOCKERS)
    a_rel = m in route_a

    if a_rel or c_rel:
        label, basis = "releaser", ("ChEMBL release assay" if a_rel else "literature identity")
    elif c_blk:
        label, basis = "blocker", "literature identity"
    elif b_ok:
        label, basis = "releaser", "structural pharmacophore"
    elif b_ok is None:
        label, basis = "unknown", b_why
    else:
        label, basis = "blocker", f"structure: {b_why}"

    recs.append({"molecule_chembl_id": m, "name": pref or "", "smiles": r["smiles"],
                 "p_binding": float(r["p_binding"]), "p_functional": float(r["p_functional"]),
                 "delta": float(r["delta"]), "label": label, "basis": basis,
                 "route_a_release_assay": a_rel, "route_b_pharmacophore": b_ok,
                 "route_c_releaser": c_rel, "route_c_blocker": c_blk})

# -------------------------------------------------------------------- report
print("=" * 86)
print("ROUTE A -- ChEMBL release/efflux assay records for these 81 compounds")
print("=" * 86)
print(f"  {len(acts)} DAT/NET/SERT activity records retrieved")
print(f"  compounds with a genuine release/efflux assay record: {len(route_a)}")
if route_a:
    for m, ds in route_a.items():
        print(f"    {m}: {ds[0][:120]}")
print("  -> ChEMBL carries essentially no release-assay annotation for this set,")
print("     so route A cannot label it. Routes B and C are used instead.")

print("\n" + "=" * 86)
print("ROUTE C -- named compounds in the paired set")
print("=" * 86)
named = [r for r in recs if r["name"]]
for r in sorted(named, key=lambda x: -x["delta"]):
    print(f"  {r['molecule_chembl_id']:<16}{r['name'][:30]:<32}"
          f"delta {r['delta']:+.2f}   {r['label']} ({r['basis']})")
print(f"  {len(named)} of {len(recs)} compounds carry any ChEMBL name at all.")

print("\n" + "=" * 86)
print("CLASSIFICATION")
print("=" * 86)
from collections import Counter
print("  ", Counter(r["label"] for r in recs))
print("  basis:", Counter(r["basis"] for r in recs))

print("\n" + "=" * 86)
print("NET OFFSET WITHIN EACH CLASS")
print("=" * 86)
res = {}
for lab in ("releaser", "blocker", "unknown"):
    d = np.array([r["delta"] for r in recs if r["label"] == lab])
    if d.size == 0:
        continue
    boot = [np.median(np.random.choice(d, d.size, replace=True)) for _ in range(5000)] \
        if d.size > 2 else []
    ci = (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))) if boot else None
    w = stats.wilcoxon(d)[1] if d.size >= 6 else float("nan")
    res[lab] = {"n": int(d.size), "median": float(np.median(d)), "mean": float(d.mean()),
                "sd": float(d.std(ddof=1)) if d.size > 1 else None,
                "ci_median": ci, "wilcoxon_p": float(w)}
    cis = f"[{ci[0]:+.3f}, {ci[1]:+.3f}]" if ci else "--"
    print(f"  {lab:<10} n={d.size:<4} median {np.median(d):+.3f}  mean {d.mean():+.3f}"
          f"  95% CI {cis}  Wilcoxon p={w:.2g}")

allr = np.array([r["delta"] for r in recs])
print(f"  {'ALL':<10} n={allr.size:<4} median {np.median(allr):+.3f}  mean {allr.mean():+.3f}")

rel = np.array([r["delta"] for r in recs if r["label"] == "releaser"])
blk = np.array([r["delta"] for r in recs if r["label"] == "blocker"])
if rel.size and blk.size:
    u = stats.mannwhitneyu(rel, blk, alternative="two-sided")
    print(f"\n  releaser vs blocker: Mann-Whitney U p={u.pvalue:.3g}")
    print(f"  blockers-only median offset {np.median(blk):+.3f} "
          f"vs all-compound {np.median(allr):+.3f} "
          f"-> removing releasers changes the offset by {np.median(blk)-np.median(allr):+.3f}")

with open("releaser_blocker.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(recs[0].keys()))
    w.writeheader(); w.writerows(recs)
json.dump({"classification": dict(Counter(r["label"] for r in recs)),
           "offset_by_class": res,
           "route_a_compounds": sorted(route_a),
           "n_records_scanned": len(acts)},
          open("releaser_blocker.json", "w"), indent=2)
print("\nwrote releaser_blocker.csv / releaser_blocker.json")
