"""Correcting the ceiling: what a modality feature can and cannot recover.

The original derivation concludes that handing the model an assay-type feature should
recover everything in the cross-assay dispersion above measurement error -- +0.098 /
+0.290 / +0.102 of R^2, "the largest unexploited quantity in the whole project." The
experiment delivered +0.003 / -0.008 / +0.021. This script works out why, and every
quantity below is computed from one table so the arithmetic is internally consistent.
(Mixing a dispersion estimated on one pull with a variance estimated on another is
caveat 1 of the original derivation, and it is easy to walk into.)

THE ERROR, in one line: Var(offset) is not what a binary flag can learn.

A binary Ki/IC50 feature can shift predictions by the MEAN offset. It cannot reproduce
the compound-to-compound VARIATION in that offset, which is most of the variance. The
original derivation subtracted the whole variance. Three further reductions follow:

  (a) only the mean is learnable from a flag          -> gain scales with mean^2, not Var
  (b) the corpus-level mean is what a corpus-trained model sees, and at NET the
      within-laboratory and between-laboratory contrasts have OPPOSITE SIGNS and
      very nearly cancel
  (c) the correction can only bite on compounds that HAVE both kinds of value --
      about one in ten -- because for the rest the pooled label already IS the
      single-modality label

Three ceilings follow, for three different tasks. The point of the correction is not
that the ceiling was wrong; it is that modality separation does not RAISE the ceiling
of the published task. It swaps that task for a different one whose ceiling is higher.
"""
import json
import numpy as np
import pandas as pd
from scipy import stats

TARGETS = ("DAT", "NET", "SERT")
SRC = "data/chembl_rederivation_activities.csv"
MEASURED_GAIN = {"DAT": 0.003, "NET": -0.008, "SERT": 0.021}   # modality_experiment.py

act = pd.read_csv(SRC)
act = act[act.standard_type.isin(["Ki", "IC50"])]
lab = pd.read_csv("modality_labels.csv")

res = {}
for t in TARGETS:
    s = act[act._target == t]
    L = lab[lab.target == t]

    # --- label variances, same table as everything else
    var_pool = float(L.p_pool.var(ddof=1))
    var_ki = float(L.p_Ki.var(ddof=1))
    var_ic = float(L.p_IC50.var(ddof=1))

    # --- within-modality replicate scatter: same compound, same modality, two papers
    # the project's own method (raw records, same compound + modality, any paper),
    # so the number is comparable to the 0.212 in the original derivation. Note that
    # 25-50% of such groups are EXACT ties: ChEMBL propagates reference values between
    # papers, so any replicate-based noise estimate is a lower bound.
    reps = [g.pchembl_value.std(ddof=1) for _, g in
            s.groupby(["molecule_chembl_id", "standard_type"]) if len(g) >= 2]
    reps = np.array([r for r in reps if np.isfinite(r)])
    sd_rep = float(np.median(reps))
    frac_tied = float((reps == 0).mean())
    var_meas = sd_rep ** 2

    # --- cross-assay differences, split by whether one paper reported both
    same, cross, allp = [], [], []
    for mol, g in s.groupby("molecule_chembl_id"):
        ki, ic = g[g.standard_type == "Ki"], g[g.standard_type == "IC50"]
        if ki.empty or ic.empty:
            continue
        allp.append(ki.pchembl_value.median() - ic.pchembl_value.median())
        sh = set(ki.document_chembl_id) & set(ic.document_chembl_id)
        if sh:
            same.append(ki[ki.document_chembl_id.isin(sh)].pchembl_value.median()
                        - ic[ic.document_chembl_id.isin(sh)].pchembl_value.median())
        ko = ki[~ki.document_chembl_id.isin(sh)]
        io = ic[~ic.document_chembl_id.isin(sh)]
        if not ko.empty and not io.empty:
            cross.append(ko.pchembl_value.median() - io.pchembl_value.median())
    same, cross, allp = map(lambda x: np.array(x), (same, cross, allp))

    var_all = float(allp.var(ddof=1))
    var_same, var_cross = float(same.var(ddof=1)), float(cross.var(ddof=1))
    var_lab = max(0.0, var_cross - var_same)              # between-laboratory
    var_modvar = max(0.0, var_same - 2 * var_meas)        # offset VARIES by compound
    mean_corpus = float(allp.mean())                      # what a flag can learn
    frac_both = float(len(allp) / len(L))

    # --- three tasks, three ceilings
    ceil_pooled = 1 - var_all / var_pool                  # published task
    ceil_known = 1 - (var_lab + 2 * var_meas) / var_ki    # modality known at predict time
    ceil_median = 1 - var_meas / var_pool                 # scored against the median

    # --- what a binary flag should be worth, taking (a)-(c) seriously
    gain_naive = (var_all - 2 * var_meas) / var_pool      # the original derivation
    gain_meanonly = mean_corpus ** 2 / var_pool           # (a) + (b)
    gain_realistic = frac_both * gain_meanonly            # (a) + (b) + (c)

    res[t] = dict(
        n_same=int(same.size), n_cross=int(cross.size), n_all=int(allp.size),
        n_compounds=int(len(L)), frac_both=frac_both,
        mean_same=float(same.mean()), mean_cross=float(cross.mean()),
        mean_corpus=mean_corpus, sd_all=float(allp.std(ddof=1)),
        sd_replicate=sd_rep, frac_replicate_tied=frac_tied, var_pool=var_pool, var_Ki=var_ki, var_IC50=var_ic,
        var_crossassay=var_all, var_same=var_same, var_cross=var_cross,
        var_laboratory=var_lab, var_modality_variation=var_modvar,
        var_measurement=2 * var_meas,
        ceiling_pooled=ceil_pooled, ceiling_modality_known=ceil_known,
        ceiling_vs_median=ceil_median,
        gain_naive=gain_naive, gain_mean_only=gain_meanonly,
        gain_realistic=gain_realistic, gain_measured=MEASURED_GAIN[t],
        p_same=float(stats.wilcoxon(same).pvalue) if same.size > 10 else None,
        p_cross=float(stats.wilcoxon(cross).pvalue) if cross.size > 10 else None)

B = "=" * 94
print(B)
print("1.  The offset depends on whether ONE paper reported both values")
print(B)
print(f"{'target':<7}{'n same-doc':>12}{'mean':>8}{'p':>10}   {'n cross-doc':>13}"
       f"{'mean':>8}{'p':>10}   {'corpus mean':>13}")
for t in TARGETS:
    r = res[t]
    print(f"{t:<7}{r['n_same']:>12}{r['mean_same']:>+8.3f}{r['p_same']:>10.1e}   "
          f"{r['n_cross']:>13}{r['mean_cross']:>+8.3f}{r['p_cross']:>10.1e}   "
          f"{r['mean_corpus']:>+13.3f}")
print("\n  At NET the two contrasts have OPPOSITE SIGNS. Within a laboratory the binding")
print("  value is the weaker of the two; across laboratories it is the stronger. A model")
print("  trained on the corpus sees the average of these, which is close to nothing.")
print("  This is why Finding 3 needs the within-document design, and equally why the")
print("  offset it finds is not a quantity a corpus-trained model can exploit.")

print()
print(B)
print("2.  Variance decomposition of the cross-assay difference (one table throughout)")
print(B)
print(f"{'target':<7}{'Var(diff)':>11}{'measurement':>13}{'laboratory':>12}"
      f"{'offset varies':>15}{'offset mean^2':>15}")
for t in TARGETS:
    r = res[t]
    print(f"{t:<7}{r['var_crossassay']:>11.3f}{r['var_measurement']:>13.3f}"
          f"{r['var_laboratory']:>12.3f}{r['var_modality_variation']:>15.3f}"
          f"{r['mean_corpus']**2:>15.3f}")
rp = ", ".join("%s %.3f" % (t, res[t]["sd_replicate"]) for t in TARGETS)
print("\n  within-modality replicate SD (same compound, two papers): " + rp)
print("  Only the LAST column is what a binary flag can remove. The 'offset varies'")
print("  column would need the offset predicted per compound from structure, which is")
print("  a different model and has never been shown to work.")

print()
print(B)
print("3.  Predicted gain from a modality flag, with each correction applied in turn")
print(B)
print(f"{'target':<7}{'original claim':>16}{'mean only':>12}{'x frac. both':>14}"
      f"{'MEASURED':>11}")
for t in TARGETS:
    r = res[t]
    print(f"{t:<7}{r['gain_naive']:>+16.3f}{r['gain_mean_only']:>+12.3f}"
          f"{r['gain_realistic']:>+14.3f}{r['gain_measured']:>+11.3f}")
fb = ", ".join("%s %.0f%%" % (t, 100 * res[t]["frac_both"]) for t in TARGETS)
print("\n  fraction of compounds with BOTH kinds of value: " + fb)

print()
print(B)
print("4.  Three ceilings, because there are three different tasks")
print(B)
print(f"{'target':<7}{'A: vs median':>14}{'B: modality known':>19}"
      f"{'C: modality unknown':>21}")
for t in TARGETS:
    r = res[t]
    print(f"{t:<7}{r['ceiling_vs_median']:>14.3f}{r['ceiling_modality_known']:>19.3f}"
          f"{r['ceiling_pooled']:>21.3f}")
print("\n  A  predict the aggregated median, score against it            (internal CV)")
print("  B  predict one new measurement whose assay you are told        (production)")
print("  C  predict one new measurement whose assay you are not told    (published)")
print("\n  Modality separation does not raise ceiling C. It moves the work from task C")
print("  to task B, which has a higher ceiling because the answer key is no longer a")
print("  mixture. The published model's score should be read against C; the production")
print("  model's against B; and the two are not directly comparable.")

json.dump(res, open("ceiling_correction.json", "w"), indent=1)
print("\nwrote ceiling_correction.json")
