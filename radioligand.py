"""Is the NET binding-versus-functional offset a property of the assay FORMAT, or of the
NET radioligand?

Human NET binding is dominated by [3H]nisoxetine, which differs from the DAT and SERT
radioligands in nonspecific binding and in the K_D used for the Cheng-Prusoff conversion.
A systematically weak NET binding K_i caused by one radioligand would produce exactly the
observed pattern -- an offset at NET and none at DAT or SERT -- without any general
property of "assay format" being involved.

The assay descriptions are already parsed for modality, and they name the radioligand.
This extracts it and re-estimates the offset within each radioligand stratum. If the
offset holds within strata, the radioligand explanation is excluded; if it is carried by
one ligand, the effect must be renamed.
"""
import re
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

SRC = "classified.csv"

# Ordered, specific first. The label is the radioligand actually displaced.
LIGANDS = [
    ("[3H]nisoxetine",   r"\[?\s*3\s*H\s*\]?\s*-?\s*nisoxetin"),
    ("[3H]desipramine",  r"\[?\s*3\s*H\s*\]?\s*-?\s*desipramin"),
    ("[125I]RTI-55",     r"\[?\s*125\s*I\s*\]?\s*-?\s*(RTI-?55|beta-?CIT|β-?CIT)"),
    ("[3H]mazindol",     r"\[?\s*3\s*H\s*\]?\s*-?\s*mazindol"),
    ("[3H]WIN35428",     r"\[?\s*3\s*H\s*\]?\s*-?\s*(WIN\s*-?\s*35,?\s*428|CFT)"),
    ("[3H]citalopram",   r"\[?\s*3\s*H\s*\]?\s*-?\s*citalopram"),
    ("[3H]paroxetine",   r"\[?\s*3\s*H\s*\]?\s*-?\s*paroxetin"),
    ("[3H]imipramine",   r"\[?\s*3\s*H\s*\]?\s*-?\s*imipramin"),
    ("[3H]cocaine",      r"\[?\s*3\s*H\s*\]?\s*-?\s*cocaine"),
    ("[3H]GBR12935",     r"\[?\s*3\s*H\s*\]?\s*-?\s*GBR\s*-?\s*129"),
    ("[3H]NE",           r"\[?\s*3\s*H\s*\]?\s*-?\s*(noradrenalin|norepinephrin|NA\b)"),
    ("[3H]5-HT",         r"\[?\s*3\s*H\s*\]?\s*-?\s*(5-?HT|serotonin)"),
    ("[3H]DA",           r"\[?\s*3\s*H\s*\]?\s*-?\s*dopamin"),
]
PAT = [(n, re.compile(p, re.I)) for n, p in LIGANDS]


def ligand(desc):
    if not isinstance(desc, str):
        return "unstated"
    for name, rx in PAT:
        if rx.search(desc):
            return name
    return "unstated"


d = pd.read_csv(SRC)
d = d[d.modality.isin(["binding", "uptake"])].copy()
d["ligand"] = [ligand(x) for x in d.assay_description]

B = "=" * 92
print(B); print("RADIOLIGAND RECOVERY FROM ASSAY TEXT"); print(B)
for t in ("DAT", "NET", "SERT"):
    sub = d[(d._target == t) & (d.modality == "binding")]
    vc = sub.ligand.value_counts()
    named = int((sub.ligand != "unstated").sum())
    print(f"\n{t} binding records: {len(sub)}   radioligand named in {named} "
          f"({100*named/len(sub):.0f}%)")
    for k, v in vc.items():
        print(f"    {k:<18} {v:>5}  ({100*v/len(sub):.1f}%)")

# ------------------------------------------------- the paired set, per target
print()
print(B); print("OFFSET STRATIFIED BY BINDING RADIOLIGAND"); print(B)
print("Per compound x target: median binding pChEMBL (within one radioligand) against")
print("median functional uptake pChEMBL. Positive = more potent in the functional assay.")

rows = []
for t in ("DAT", "NET", "SERT"):
    sub = d[d._target == t]
    up = (sub[sub.modality == "uptake"].groupby("molecule_chembl_id")
          .pchembl_value.median())
    bi = sub[sub.modality == "binding"]
    for lig, g in bi.groupby("ligand"):
        if lig == "unstated":
            continue
        b = g.groupby("molecule_chembl_id").pchembl_value.median()
        common = b.index.intersection(up.index)
        if len(common) < 5:
            continue
        diff = (up[common] - b[common]).values
        docs = g[g.molecule_chembl_id.isin(common)].document_chembl_id.nunique()
        rows.append(dict(target=t, ligand=lig, n=len(common), n_docs=docs,
                         median=float(np.median(diff)), mean=float(diff.mean()),
                         sd=float(diff.std(ddof=1)) if len(diff) > 1 else np.nan,
                         lo=float(np.percentile(diff, 25)),
                         hi=float(np.percentile(diff, 75))))
R = pd.DataFrame(rows).sort_values(["target", "n"], ascending=[True, False])
print(f"\n{'target':<6}{'radioligand':<18}{'n':>5}{'docs':>6}{'median':>9}{'mean':>8}"
      f"{'SD':>7}{'IQR':>18}")
for _, r in R.iterrows():
    print(f"{r.target:<6}{r.ligand:<18}{r.n:>5}{r.n_docs:>6}{r['median']:>+9.3f}"
          f"{r['mean']:>+8.3f}{r['sd']:>7.3f}   [{r.lo:>+6.2f},{r.hi:>+6.2f}]")

# ------------------------------------------------- is NET still NET, within ligand?
print()
print(B); print("DOES THE NET OFFSET SURVIVE WITHIN EACH RADIOLIGAND STRATUM?"); print(B)
net = R[R.target == "NET"]
print("NET strata:")
for _, r in net.iterrows():
    print(f"   {r.ligand:<18} n={r.n:<5} median {r['median']:+.3f}")
if len(net) > 1:
    print("\n   Spread across NET radioligands: "
          f"{net['median'].min():+.3f} to {net['median'].max():+.3f}")
print("\nComparator -- the other two targets, pooled over their own radioligands:")
for t in ("DAT", "SERT"):
    s = R[R.target == t]
    if len(s):
        w = np.average(s["median"], weights=s["n"])
        print(f"   {t:<5} n-weighted median offset {w:+.3f} over {int(s.n.sum())} "
              f"compound-ligand pairs")

# ------------------------------------------------- formal test: ligand vs target
print()
print(B); print("FIXED-EFFECTS TEST: IS IT THE TARGET OR THE LIGAND?"); print(B)
print("One row per (compound, target, binding radioligand) with a matching uptake value.")
long = []
for t in ("DAT", "NET", "SERT"):
    sub = d[d._target == t]
    up = sub[sub.modality == "uptake"].groupby("molecule_chembl_id").pchembl_value.median()
    bi = sub[sub.modality == "binding"]
    for lig, g in bi.groupby("ligand"):
        if lig == "unstated":
            continue
        b = g.groupby("molecule_chembl_id").pchembl_value.median()
        dc = g.groupby("molecule_chembl_id").document_chembl_id.first()
        for cid in b.index.intersection(up.index):
            long.append(dict(cid=cid, target=t, ligand=lig,
                             offset=float(up[cid] - b[cid]), doc=dc[cid]))
L = pd.DataFrame(long)
print(f"rows: {len(L)}   targets: {L.target.nunique()}   ligands: {L.ligand.nunique()}"
      f"   documents: {L.doc.nunique()}")
if len(L) > 30:
    m1 = smf.ols("offset ~ C(target)", data=L).fit(
        cov_type="cluster", cov_kwds={"groups": L.doc})
    print("\n  offset ~ target            (NET contrast against DAT):")
    for k in m1.params.index:
        print(f"     {k:<34}{m1.params[k]:+8.3f}  p={m1.pvalues[k]:.4g}")
    m2 = smf.ols("offset ~ C(target) + C(ligand)", data=L).fit(
        cov_type="cluster", cov_kwds={"groups": L.doc})
    print("\n  offset ~ target + radioligand   (does the NET term survive?):")
    for k in m2.params.index:
        if "target" in k or k == "Intercept":
            print(f"     {k:<34}{m2.params[k]:+8.3f}  p={m2.pvalues[k]:.4g}")
    print("\n     radioligand terms:")
    for k in m2.params.index:
        if "ligand" in k:
            print(f"     {k:<34}{m2.params[k]:+8.3f}  p={m2.pvalues[k]:.4g}")
    L.to_csv("radioligand_pairs.csv", index=False)
    print("\nwrote radioligand_pairs.csv")
R.to_csv("radioligand_strata.csv", index=False)
print("wrote radioligand_strata.csv")
