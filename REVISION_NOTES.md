> **Note added 7 October 2026.** This file is the revision log as it was written, kept
> unedited. Two open items it records are now closed: the Zenodo DOI is
> [10.5281/zenodo.23144306](https://doi.org/10.5281/zenodo.23144306), and the
> corresponding-author line on the title page is filled in. Where this log and the
> top-level `README.md` disagree, the README is current.

# Revision addendum — 2026-09-27

Added in response to the question of whether the NET binding-versus-functional
offset reported in Section 3.5 is a property of the transporter or an artefact of
a single chemotype, and to the related question of whether Bemis–Murcko scaffold
grouping is too coarse for this target family.

## What is in this addendum

```
src/replicate_assay_offset.py          re-retrieval + stratification (MIT)
data/chembl_rederivation_activities.csv   16,750 raw ChEMBL activity records
data/net_paired_compounds.csv          the 81 NET compounds measured both ways
results/chemotype_stratification.json  all statistics reported in Section 3.6
figures/fig11_chemotype.pdf/.png       Figure 11
```

Reproduce with:

```bash
python src/replicate_assay_offset.py fetch      # ~3 min against the live API
python src/replicate_assay_offset.py analyse    # writes results + figure
```

`fetch` is only needed to refresh the retrieval; `data/chembl_rederivation_activities.csv`
is the copy used for the reported numbers and `analyse` runs against it directly.

## What this analysis is, and is not

It is an **independent re-retrieval**, not a reproduction. Activities were pulled
from the live ChEMBL REST API on 2026-09-27, sixteen months after the frozen
ChEMBL_37 release used for the main analysis, with an independently written
curation script. The compound sets are **non-nested**: 89 / 81 / 71 compounds at
DAT / NET / SERT here, against 112 / 97 / 77 in the frozen analysis. Numbers
therefore differ, and are reported as a robustness check rather than as a
replication of the exact estimates.

Curation applied here:

| step | rule |
|---|---|
| targets | human DAT CHEMBL238, NET CHEMBL222, SERT CHEMBL228 |
| binding | `standard_type = Ki` from an assay ChEMBL types `B` |
| functional | `standard_type = IC50` from an assay ChEMBL types `F` |
| filters | `target_organism = Homo sapiens`; `pchembl_value` present; `standard_relation = '='`; no data-validity comment |
| aggregation | per compound, median `pchembl_value` within each modality |
| offset | `delta = functional − binding` (positive = functional reports higher potency) |

## Results

**Replication.** NET +0.540 (95% CI +0.390 to +0.670, n = 81, Wilcoxon
p = 1.2 × 10⁻⁸), against +0.440 in the frozen analysis. DAT −0.280, SERT −0.080.
Kruskal–Wallis across targets p = 2.7 × 10⁻¹⁶.

**Chemotype stratification.** Every stratum with n ≥ 5 is positive — five of five
Butina clusters at 0.6 Tanimoto, and four of four Bemis–Murcko scaffold strata.
Leave-one-stratum-out never moves the pooled offset by more than 0.15 or changes
its sign; deleting the largest stratum (25 pyrrolidinophenones) leaves +0.390.
No single chemotype carries the effect.

**Heterogeneity.** Magnitude differs significantly across strata (Kruskal–Wallis
p < 0.001), spanning roughly threefold from +0.31 to +0.99, largest in the
pyrrolidinophenone (α-PVP-type) cathinones. The pooled value is an average over a
heterogeneous set, which is consistent with the failure of the constant
correction reported in Section 3.5.

## One discrepancy from the frozen analysis

The frozen analysis reported DAT and SERT offsets as indistinguishable
(p = 0.39). In this re-retrieval they differ (−0.280 vs −0.080, Mann–Whitney
p = 0.005), driven by a more negative DAT estimate on a different compound set.

The manuscript has been amended accordingly: the claim is now that the offset is
specific to NET **among the three targets**, a contrast robust to retrieval and
curation, and no longer that DAT and SERT behave identically. See Section 3.6,
final paragraph, and the corresponding Limitations entry.

## New references required in `references.bib`

```bibtex
@article{Fooladi2025,
  author  = {Fooladi, Hosein and Vu, Thi Ngoc Lan and Mathea, Miriam
             and Kirchmair, Johannes},
  title   = {Evaluating Machine Learning Models for Molecular Property
             Prediction: Performance and Robustness on Out-of-Distribution Data},
  journal = {J. Chem. Inf. Model.},
  year    = {2025},
  volume  = {65},
  number  = {19},
  pages   = {9871--9891},
  doi     = {10.1021/acs.jcim.5c00475},
  pmid    = {40947919}
}

@article{Butina1999,
  author  = {Butina, Darko},
  title   = {Unsupervised Data Base Clustering Based on {Daylight's}
             Fingerprint and {Tanimoto} Similarity: A Fast and Automated Way
             To Cluster Small and Large Data Sets},
  journal = {J. Chem. Inf. Comput. Sci.},
  year    = {1999},
  volume  = {39},
  number  = {4},
  pages   = {747--750},
  doi     = {10.1021/ci9803381}
}
```

The Fooladi entry was verified against PubMed (PMID 40947919). The Butina entry's
DOI was verified; **confirm the volume, issue and page numbers against the
publisher record before submission** — they were taken from secondary indexes,
not from the article itself.

## Other manuscript changes in this revision

- Introduction: the scaffold-splitting paragraph now states what the recent
  critiques actually found, and notes that Murcko abstraction discards the ring
  substituents that determine selectivity in this target family.
- Methods §2.8: new, describing the re-retrieval and stratification protocol.
- Results §3.6: new, reporting replication, stratification and the discrepancy.
- Discussion: new paragraph "How much stringency does a scaffold split buy?"
- Limitations: four new entries (offset heterogeneity; DAT-vs-SERT sensitivity;
  non-nesting of the stratification set; coarseness of Murcko grouping here).
- Abstract and Scientific Contribution updated to match.
- Fixed: `\textalpha` was undefined in this preamble (no `textgreek` package) —
  changed to `$\alpha$`.
- Fixed: Table 8 (extended external performance by source) overflowed the text
  block by 58.7 pt; set to `\footnotesize` with reduced `\tabcolsep`.
- Removed: a duplicated `\maketitle` after the abstract, which printed the title
  block twice.

## Still outstanding

- The stratification runs on the re-retrieved set because the frozen per-modality
  records were not among the deposited files. Re-running `analyse` against the
  ChEMBL_37 per-modality table would let Section 3.6 partition the reported
  estimate rather than test the effect on independent data. That is the single
  change that would most strengthen this section.
- One pre-existing overfull hbox remains (38.8 pt, Methods §2.1, the run of
  CHEMBL target identifiers). Cosmetic.
- Zenodo DOI still to be minted and substituted for `[DOI to be assigned]`.
- Corresponding-author name and email are still `[name], [email]` on the title page.

---

# Second revision addendum — 2026-09-30

Added in response to three questions: whether the "Comprehensive in vitro profiling"
dataset is a usable new external set; whether the NET offset is a between-laboratory
or a releaser-versus-blocker artefact; and whether any antidepressant SAR series could
supply the missing NET/SERT coverage.

## What is in this addendum

```
src/coverage.py              external-set potency coverage against training (model-free)
src/ext_ceiling.py           achievable R^2 on the external set vs on the training set
src/releaser_fetch.py        pulls names + all DAT/NET/SERT records for the 81 compounds
src/releaser_classify.py     releaser-vs-blocker labelling by three independent routes
src/document_confound.py     within-document and leave-one-document-out analyses
src/figs.py                  Figure 11 and the annotated supplementary coverage figure

data/stalberga_parsed.csv         53 compounds x 3 targets, DOI 10.1111/bph.70587
data/net81_mat_activities.csv     785 ChEMBL DAT/NET/SERT records for the 81 compounds
data/releaser_blocker.csv         per-compound classification, basis recorded per route
data/within_document_pairs.csv    77 same-paper binding/functional pairs

results/coverage.json             per-target coverage statistics
results/ext_ceiling.json          Table 12
results/releaser_blocker.json     classification counts and offsets by class
results/document_confound.json    provenance counts, LODO, within-document estimate

figures/fig12_within_document.pdf/.png   Figure 11 in the typeset manuscript
figures/supp_coverage_annotated.pdf/.png supplementary; a more heavily annotated
                                         version of Figure 8, not used in the text
```

`coverage.py` and `figs.py` additionally read `mat_profiles_censored_aware.csv` from the
main deposit (the curated benchmark); it is not duplicated here. Run order:

```bash
python src/releaser_fetch.py       # ~4 min against the live ChEMBL API
python src/releaser_classify.py
python src/document_confound.py
python src/coverage.py
python src/ext_ceiling.py
python src/figs.py
```

## Finding 1 — the candidate external set was the existing one

The "Comprehensive in vitro profiling of traditional and emerging stimulants at
monoamine transporters and the 5-HT2A receptor" dataset carries DOI
**10.1111/bph.70587**, which is the DOI of `Stalberga2026` — the pre-registered
external set already used in Section 2.4. It is not an independent source. The counts
reconcile exactly: 53 compounds parsed, 7 right-censored at one or more targets, 46
remaining, 4 removed for exact InChIKey overlap, 42 evaluated.

## Finding 2 — the external set cannot reach NET or SERT, and this bounds R^2

Model-free comparison of the 46 fully quantified external compounds against the
training distribution:

| target | ext/train variance | median ext compound | above training p75 |
|---|---|---|---|
| DAT  | 0.73 | 57th percentile | 13 of 46 |
| NET  | 0.49 | 30th percentile | **0 of 46** |
| SERT | 0.25 | 13th percentile | **0 of 46** |

Because R^2 is normalised by the variance of the evaluation set, the achievable
ceiling moves with it. Using the measured cross-assay dispersion as the noise term,
the ceiling on this external set is 0.752 / 0.252 / 0.344 against 0.820 / 0.633 / 0.838
on the training distribution. The achieved 0.459 / 0.015 / 0.058 are therefore 61% /
6% / 17% of what was reachable (Table 12, Section 3.9).

## Finding 3 — the NET offset is neither a laboratory nor a releaser artefact

*Laboratory.* 75 of the 81 paired compounds have both modalities reported in the same
paper. Within those pairs the offset is **+0.550, 95% CI [+0.400, +0.710]**, Wilcoxon
p = 1.0e-9 — against +0.540 pooled across documents. Five documents contribute, from
five laboratories, 1999–2007, all positive (+0.15 to +1.12). Leave-one-document-out
worst case +0.390 on 53 compounds.

*Releaser versus blocker.* All 81 classify as uptake blockers. Zero of the 785
retrieved DAT/NET/SERT records is a release or efflux assay, and no compound matches
the substrate-type releaser pharmacophore; median MW is 303 Da with one compound below
230 Da. Cocaine (+0.78) and mazindol (+0.50) reproduce the 7.0x and 5.0x NET ratios
Eshleman et al. 2017 report for the same compounds, confirming the scale. The offset
is a blocker-class property measured within blockers.

Note the direction of what this does **not** show: because the paired set contains no
releasers, the offset is established for blockers only.

## Finding 4 — a temporal external window exists

ChEMBL_37 was released 2026-05-01 (`/chembl/api/data/status.json`). Its literature
freeze is near July 2025, inferred from the J Med Chem 2025 issue histogram: issues
1–12 fully curated, issue 13 absent, issue 14 one stray document. Any qualifying
measurement published after that date is external by construction, needs no new
experiments, and can be selected to span the potent end of NET and SERT. ChEMBL also
does not systematically curate several journals that do publish such data; spot checks
returning zero documents include ACS Chem Neurosci, Molecules (27 all-time), Bioorg
Chem (1), Neuropharmacology (2) and Br J Pharmacol (21). A shortlist of candidate
sources with per-DOI ChEMBL overlap checks is in the accompanying findings note.

## Manuscript changes in this revision

- Methods §2.9 (`sec:confounds`): new, describing the within-document design and the
  three independent releaser/blocker labelling routes.
- Results §3.7 (`sec:confounds-results`): new, reporting both tests; Figure 11.
- Results §3.9: new paragraph "The consequence for R^2" plus Table 12.
- Discussion: two new paragraphs, "Report the ceiling, not only the score" and
  "A tractable route to NET and SERT coverage".
- Limitations: the two near-duplicate releaser entries merged into one that now also
  bounds §3.7 in the opposite direction; two new entries (five-paper basis of the
  within-document analysis; assumptions behind the external-set ceilings).
- Fixed: the 38.8 pt overfull hbox in Methods §2.1, wrapped in `sloppypar`. Worst
  remaining overfull box is 5.8 pt.

## Finding 5 — two genuinely new external sets, and the model does generalise

Two qualifying sources were found and evaluated against the faithfully rebuilt model
(`src/rebuild_faithful.py`, which reproduces the published internal cross-validation
exactly: 0.584 / 0.501 / 0.541 at DAT / NET / SERT).

**PDSP via BindingDB** — 43 compounds with all three transporters, 29 after exact
InChIKey overlap removal. Binding Ki, so the modality matches the training label.
Reaches pNET 8.85 and pSERT 9.54, well past the training 75th percentile that
Stalberga cannot cross.

**Cui et al. 2026 (Eur J Med Chem)** — a 20-compound SNRI series transcribed from
Tables 1–2, SMILES hand-built and checked against the paper's HRMS. None of its ten
scaffolds appear in training; median nearest-neighbour Tanimoto 0.40. Human HEK293
uptake, per §4.3.3.

The headline is that §3.9's claim that NET and SERT performance is "structurally
unavailable" is superseded. Squared-Pearson r^2 rises monotonically with coverage:

```
                 Stalberga        PDSP           Cui 2026
NET    r^2       0.015 (p 0.44)   0.197 (0.016)  0.342 (0.0085), rho +0.707
SERT   r^2       0.047 (p 0.16)   0.352 (0.0007) 0.468 (0.0018), rho +0.670
```

Cui NET rho = +0.707 is indistinguishable from the internal NET rho of +0.709, on
chemistry at Tanimoto 0.40 rather than 0.63 — equal ranking on less familiar ground.
What does *not* transfer is calibration: external slopes run 0.075–0.407 against
0.51–0.59 internally, and external bias reaches ±0.97 log units against −0.06
internally. The coefficient of determination therefore goes negative on three external
target/set combinations while squared Pearson stays positive, which is the whole
distance between the two definitions.

```
src/rebuild_faithful.py       faithful rebuild from the project's own baseline.py,
                              repeated_cv.py and external_ci.py
src/pdsp_extract.py           PDSP/BindingDB extraction and overlap removal
src/ejmc2026_build.py         Cui 2026 transcription, SMILES construction, HRMS check
src/ejmc2026_eval.py          evaluation of the Cui set
src/competency2.py            the eight-diagnostic battery on all three external sets
src/internal_diagnostics.py   the same eight diagnostics for the internal folds
src/fig_external_sets.py      response-range coverage of the three sets
src/fig_diagnostics.py        the eight-diagnostic comparison figure

results/faithful_cv.json          reproduction of the published internal CV
results/competency2.json          all diagnostics, all three external sets
results/internal_diagnostics.json the internal-fold counterpart
results/ejmc2026_eval.json        Cui set evaluation
results/ext_ceiling.json          ceilings, now covering all three sets

figures/fig_external_sets.pdf/.png response-range coverage, three sets vs training
figures/fig_diagnostics.pdf/.png   eight diagnostics x four evaluations
```

`internal_diagnostics.py` computes the internal variance ratio as 1.00 by construction
and measures nearest-neighbour Tanimoto fold-wise — each held-out compound against only
the four folds it was not trained on. Against the whole table every compound would
return 1.0. The fold-wise median is 0.628 with 26.6% above 0.7, independently
reproducing the manuscript's reported 26.2%.

## Finding 6 — the model configuration changes, and the ceiling claim is corrected

Three changes, in order of how much they matter.

**The all-three-transporter intersection requirement is dropped for training.** The
benchmark keeps only compounds measured at all three (2,849); per target ChEMBL has
3,367 / 3,702 / 5,338. Training per target without the filter, 20 seeds on identical
scaffold folds, scored against single-modality answer keys:

```
                published          production         gain   seeds   p
DAT   (Ki)      0.399 +- 0.019     0.556 +- 0.015    +0.157   20/20  5e-22
DAT   (IC50)    0.423 +- 0.023     0.557 +- 0.014    +0.134   20/20  5e-20
NET   (Ki)      0.418 +- 0.010     0.520 +- 0.014    +0.101   20/20  4e-17
NET   (IC50)    0.409 +- 0.013     0.502 +- 0.010    +0.093   20/20  9e-18
SERT  (Ki)      0.192 +- 0.013     0.474 +- 0.016    +0.281   20/20  1e-26
SERT  (IC50)    0.319 +- 0.015     0.524 +- 0.013    +0.204   20/20  2e-23
```

A size-matched subsample recovers most of the gain, so the damage is in WHICH compounds
the filter removes, not how many. Near-duplicate leakage is excluded: compounds with a
training neighbour above Tanimoto 0.99 are 0.3-0.6% of held-out compounds, and
restricting to those with no neighbour above 0.70 still leaves +0.09 to +0.18. The
three-target profile is still required for the selectivity analyses.

**The assay modality is kept as a binary feature.** Worth +0.003 / -0.008 / +0.021
internally. It is retained on principle rather than on measured gain: the answer key
for any external measurement is single-modality, the feature costs nothing, and it
slightly improves calibration. It should not be described as an improvement.

**The ceiling headroom claim is corrected.** `CEILING_DERIVATION.md` previously argued
that an assay-type feature would recover +0.098 / +0.290 / +0.102 of R^2, "the largest
unexploited quantity in the whole project." That is falsified. The error was treating
Var(offset) as learnable by a binary feature; a flag can only move the mean, so the
recoverable quantity scales with mean^2, not variance. Two further reductions follow:
the corpus-level mean is not the within-laboratory mean -- at NET the two have OPPOSITE
SIGNS and cancel -- and only 8-12% of compounds have both kinds of value at all.
Predicted gain after all three corrections is +0.001 to +0.003, against a measured
+0.003 / -0.008 / +0.021.

The decomposition also strengthens Finding 3 rather than weakening it: the corpus-level
NET contrast is the wrong sign and near zero, so any analysis pooling across papers
would have missed the offset entirely. The within-document design is necessary, not
fastidious.

## Finding 7 — a leakage correction, and what it does to the external claim

An earlier reading of the external results had the modality feature improving external
performance (+0.054 mean, 8 of 9 cells, sign test p = 0.039). It does not survive
strict leakage control and must not be used. The exclusion had been applied on the full
InChIKey, which distinguishes stereoisomers, while the ECFP4 fingerprints generated
here do not encode chirality -- so (S)-citalopram and racemic citalopram carry
different keys and identical fingerprints. Re-running with exclusion on the InChIKey
skeleton block plus a Tanimoto < 0.99 filter:

```
arm                     mean external r^2   delta    cells won   sign test
published config                    0.211       -           -           -
drop intersection                   0.233  +0.022         6/9      p 0.51
+ modality feature                  0.232  +0.021         7/9      p 0.18
separate models                     0.232  +0.021         5/9      p 1.00
```

Neither change produces a reliable external improvement; excluding PDSP both are
slightly negative. A +0.10 to +0.28 internal gain, significant at p < 1e-16 across 20
seeds and robust to leakage control, transfers to +0.02 and nothing a referee would
accept. This is the paper's own thesis, produced deliberately on its own model, and it
is a cleaner instance than the original external comparison because the intervention
and the controls were chosen in advance.

**Side effect to plan for: expanding the training set absorbs the PDSP external set.**
PDSP antidepressants are classic, heavily characterised compounds, so once training is
not restricted to the three-target intersection most are already in it. After
skeleton-level exclusion only 12-16 of 43 remain and nothing is significant. PDSP can
evaluate the published configuration; it cannot evaluate the production one. Cui 2026
and Stalberga 2026 are unaffected.

```
src/modality_labels.py            per-modality medians and offsets
src/ceiling_correction.py         variance decomposition, the three ceilings
src/modality_experiment.py        four arms on seeded scaffold folds
src/modality_controls.py          size-matched and paired-bootstrap controls
src/modality_subsample.py         the size-matched arm
src/internal_leak_check.py        near-duplicate control on the internal gain
src/production_model.py           20-seed comparison, final fitted models
src/production_eval.py            eight-diagnostic battery on the production model
src/modality_external_strict.py   external evaluation, skeleton-level exclusion

data/modality_labels.csv          per compound x target x modality medians
results/ceiling_correction.json   decomposition and corrected ceilings
results/modality_experiment.json  the four arms
results/modality_controls.json    controls and bootstrap CIs
results/production_cv.json        20-seed published vs production
results/modality_external_strict.json   external, leakage-controlled
results/production_models.pkl     the fitted production models
```

## Still outstanding after this revision

- `Fooladi2025` and `Butina1999` BibTeX entries are still to be added to
  `references.bib`; verify Butina volume and pages against the publisher record.
- Recompute the chemotype stratification, the cross-assay dispersions and the ceilings
  on the **frozen** ChEMBL_37 per-modality table. Table 12 currently mixes a frozen
  variance with a re-retrieval dispersion; the two sets are close but not identical.
- Narrow the abstract's "first quantification" claim: Eshleman et al. 2017 Table 2
  already publishes per-compound Ki/IC50 ratios at all three transporters, and its NET
  column exceeds DAT and SERT for pure blockers.
- Zenodo DOI still to be minted; corresponding-author name and email still `[name]`,
  `[email]` on the title page.
- Decide and state in Methods which R^2 the paper reports. The code computes squared
  Pearson; most readers will assume the coefficient of determination. Every external
  table should carry both, as `results/competency2.json` now does. This is a must-fix.
- Obtain publisher PDFs for PMIDs 41067053 and 42002053; the attachments resolved to
  landing pages only.
- Verify the PDSP values against the Tatsumi 1997 and Owens 1997 primary tables.
- `data/struct.csv` records benzoylecgonine (MW 289.3) in the cocaine row; it should be
  the methyl ester, MW 303.4. Correct and re-run the Stalberga evaluation.
- Recompute the ceilings and dispersions on the FROZEN ChEMBL_37 table. The corrected
  ceilings in CEILING_DERIVATION.md use the 2026 re-retrieval; the manuscript's
  0.820 / 0.633 / 0.838 use the frozen variances. Still the one unaddressed caveat.
- Decide whether the production configuration replaces the published one in the paper
  or appears as a separate section. If it replaces it, every internal number and both
  external tables are recomputed, and PDSP leaves the external set list.
- Add to Limitations: the replicate-based noise floor is a lower bound, because 20-30%
  of same-compound cross-paper pairs are exact ties -- value propagation between
  publications, not independent measurement.
