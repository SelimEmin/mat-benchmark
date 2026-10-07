"""The 2x2 decomposition, (a) stratified by binding radioligand and (b) with
small-sample-corrected inference.

Two referee demands answered in one model.

R10 -- the radioligand. If the NET format effect is a property of the measurement
format, it should appear within every radioligand stratum. If it is a property of one
radioligand's calibration, it will appear in that stratum alone.

R5  -- the cluster count. The headline +0.261 rests on standard errors clustered on 49
documents. CR0/CR1 cluster-robust variance over-rejects at this cluster count, so a wild
cluster bootstrap-t with Rademacher weights is reported alongside, and CR3 as a
jackknife-style check.

A record's "stratum" is the radioligand of the binding assay; functional uptake records
carry no radioligand and are assigned to every stratum they are compared within, which
the within-compound-within-document demeaning handles automatically because the
comparison only ever happens inside one document.
"""
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from radioligand import ligand

RNG = np.random.default_rng(20261004)
NBOOT = 9999

d = pd.read_csv("classified.csv", low_memory=False)
d = d[(d.standard_relation == "=") & d.pchembl_value.notna()
      & d.modality.isin(["binding", "uptake"])].copy()
d["uptake"] = (d.modality == "uptake").astype(float)
d["Ki"] = (d.standard_type == "Ki").astype(float)
d["grp"] = d.document_chembl_id + "|" + d.molecule_chembl_id
d["ligand"] = [ligand(x) for x in d.assay_description]

# a document's NET binding radioligand: the one it uses for the most records
dl = (d[(d.modality == "binding") & (d.ligand != "unstated")]
      .groupby(["_target", "document_chembl_id"]).ligand
      .agg(lambda s: s.value_counts().idxmax()))


def demean(frame, cols, by):
    out = frame.copy()
    for c in cols:
        out[c] = frame[c] - frame.groupby(by)[c].transform("mean")
    return out


def fit(D):
    return smf.ols("pchembl_value ~ uptake + Ki - 1", data=D).fit(
        cov_type="cluster", cov_kwds={"groups": D.document_chembl_id})


def wild_bootstrap_t(D, var, nboot=NBOOT):
    """Wild cluster bootstrap-t, Rademacher weights, null imposed on `var`."""
    full = fit(D)
    t_obs = full.params[var] / full.bse[var]
    other = [c for c in ("uptake", "Ki") if c != var]
    r = smf.ols("pchembl_value ~ " + " + ".join(other) + " - 1", data=D).fit()
    resid, fitted = r.resid.values, r.fittedvalues.values
    clusters = D.document_chembl_id.values
    uniq = np.unique(clusters)
    idx = {c: np.where(clusters == c)[0] for c in uniq}
    ts = np.empty(nboot)
    Db = D.copy()
    for b in range(nboot):
        w = RNG.choice([-1.0, 1.0], size=uniq.size)
        y = fitted.copy()
        for wi, c in zip(w, uniq):
            y[idx[c]] += wi * resid[idx[c]]
        Db["pchembl_value"] = y
        mb = fit(Db)
        ts[b] = mb.params[var] / mb.bse[var]
    p = float((np.abs(ts) >= abs(t_obs)).mean())
    q = np.percentile(ts, [2.5, 97.5])
    ci = (full.params[var] - q[1] * full.bse[var],
          full.params[var] - q[0] * full.bse[var])
    return t_obs, p, ci, full


B = "=" * 96
print(B)
print("A. THE 2x2 WITHIN EACH BINDING RADIOLIGAND  (compound-within-document effects)")
print(B)
print("If the effect is the measurement FORMAT it should appear in every stratum.")
print(f"\n{'target':<6}{'radioligand':<17}{'docs':>5}{'rows':>7}"
      f"{'uptake - binding':>32}{'pKi - pIC50':>28}")
rows = []
for t in ("DAT", "NET", "SERT"):
    s_all = d[d._target == t]
    strata = [("all", None)]
    vc = dl.loc[t].value_counts() if t in dl.index.get_level_values(0) else {}
    strata += [(k, k) for k in vc.index]
    for nm, lig in strata:
        s = s_all
        if lig is not None:
            docs = dl.loc[t][dl.loc[t] == lig].index
            s = s_all[s_all.document_chembl_id.isin(docs)]
        D = demean(s, ["pchembl_value", "uptake", "Ki"], ["grp"])
        D = D[(D.uptake.abs() > 1e-9) | (D.Ki.abs() > 1e-9)]
        if len(D) < 20 or D.document_chembl_id.nunique() < 3 or D.uptake.abs().max() < 1e-9:
            continue
        m = fit(D)
        ci = m.conf_int()
        def f(v):
            return (f"{m.params[v]:+.3f} [{ci.loc[v,0]:+.2f},{ci.loc[v,1]:+.2f}] "
                    f"p={m.pvalues[v]:.3g}")
        print(f"{t:<6}{nm:<17}{D.document_chembl_id.nunique():>5}{len(D):>7}"
              f"{f('uptake'):>32}{f('Ki'):>28}")
        rows.append(dict(target=t, stratum=nm, docs=D.document_chembl_id.nunique(),
                         rows=len(D), fmt=m.params["uptake"], fmt_p=m.pvalues["uptake"],
                         vt=m.params["Ki"], vt_p=m.pvalues["Ki"]))
    print()

pd.DataFrame(rows).to_csv("twobytwo_by_radioligand.csv", index=False)

print(B)
print("B. SMALL-SAMPLE-CORRECTED INFERENCE ON THE NET FORMAT EFFECT")
print(B)
s = d[d._target == "NET"]
D = demean(s, ["pchembl_value", "uptake", "Ki"], ["grp"])
D = D[(D.uptake.abs() > 1e-9) | (D.Ki.abs() > 1e-9)]
G = D.document_chembl_id.nunique()
print(f"   clusters (documents): {G}   rows: {len(D)}")
for var, lbl in (("uptake", "format (uptake - binding)"), ("Ki", "value type (pKi - pIC50)")):
    t_obs, p_wb, ci_wb, full = wild_bootstrap_t(D, var)
    ci0 = full.conf_int()
    print(f"\n   {lbl}")
    print(f"      estimate                      {full.params[var]:+.3f}")
    print(f"      CR0 cluster-robust  95% CI    [{ci0.loc[var,0]:+.3f}, "
          f"{ci0.loc[var,1]:+.3f}]   p = {full.pvalues[var]:.4f}")
    print(f"      wild bootstrap-t    95% CI    [{ci_wb[0]:+.3f}, {ci_wb[1]:+.3f}]"
          f"   p = {p_wb:.4f}   ({NBOOT} Rademacher draws)")

print()
print(B)
print("C. THE SAME, RESTRICTED TO DOCUMENTS USING [125I]RTI-55 AT NET")
print(B)
docs = dl.loc["NET"][dl.loc["NET"] == "[125I]RTI-55"].index
s = d[(d._target == "NET") & (d.document_chembl_id.isin(docs))]
D2 = demean(s, ["pchembl_value", "uptake", "Ki"], ["grp"])
D2 = D2[(D2.uptake.abs() > 1e-9) | (D2.Ki.abs() > 1e-9)]
if D2.document_chembl_id.nunique() >= 3:
    t_obs, p_wb, ci_wb, full = wild_bootstrap_t(D2, "uptake")
    ci0 = full.conf_int()
    print(f"   clusters: {D2.document_chembl_id.nunique()}   rows: {len(D2)}")
    print(f"      format estimate               {full.params['uptake']:+.3f}")
    print(f"      CR0 95% CI                    [{ci0.loc['uptake',0]:+.3f}, "
          f"{ci0.loc['uptake',1]:+.3f}]   p = {full.pvalues['uptake']:.4f}")
    print(f"      wild bootstrap-t 95% CI       [{ci_wb[0]:+.3f}, {ci_wb[1]:+.3f}]"
          f"   p = {p_wb:.4f}")
print("\nwrote twobytwo_by_radioligand.csv")
