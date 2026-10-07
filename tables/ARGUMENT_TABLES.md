# Supporting tables for the argument

1. **QSAR models fail on external data** — Table 1
2. **Not a defect of this model** — Tables 2a, 2b, 2c
3. **Not a defect of this model class** — Tables 3a, 3b
4. **Not bad external data** — Table 4
5. **Therefore the limit is the training–test relationship** — Tables 5a, 5b, 5c

---

## 1. QSAR models fail on external data

### Table 1 — The external shortfall, nine target × set cells

| evaluation | target | n | var ÷ train | r² | p | R²_det | RMSE | guess-mean RMSE | ρ | ceiling | % ceiling | retained |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Internal CV (production) | DAT | 3718 | 1.00 | 0.557 |  | +0.553 | 0.74 |  | +0.75 | 0.800 | 70% |  |
| Internal CV (production) | NET | 4150 | 1.00 | 0.511 |  | +0.507 | 0.76 |  | +0.71 | 0.871 | 59% |  |
| Internal CV (production) | SERT | 5744 | 1.00 | 0.521 |  | +0.514 | 0.82 |  | +0.73 | 0.875 | 59% |  |
| Stalberga 2026 | DAT | 43 | 0.68 | 0.452 | 0.0000 | -0.851 | 1.16 | 1.03 | +0.58 | 0.735 | 61% | 81% |
| Stalberga 2026 | NET | 43 | 0.50 | 0.015 | 0.4388 | -0.239 | 0.84 | 1.01 | +0.07 | 0.263 | 6% | 3% |
| Stalberga 2026 | SERT | 43 | 0.25 | 0.047 | 0.1635 | -0.286 | 0.69 | 1.46 | +0.11 | 0.360 | 13% | 9% |
| PDSP / BindingDB | DAT | 29 | 0.73 | 0.316 | 0.0015 | +0.294 | 0.74 | 1.00 | +0.63 | 0.754 | 42% | 57% |
| PDSP / BindingDB | NET | 29 | 0.84 | 0.197 | 0.0160 | +0.094 | 0.93 | 0.98 | +0.42 | 0.563 | 35% | 38% |
| PDSP / BindingDB | SERT | 29 | 0.82 | 0.352 | 0.0007 | +0.333 | 0.89 | 1.09 | +0.52 | 0.802 | 44% | 68% |
| Cui et al. 2026 | DAT | 11 | 0.34 | 0.089 | 0.3722 | -0.611 | 0.74 | 1.31 | +0.45 | 0.466 | 19% | 16% |
| Cui et al. 2026 | NET | 19 | 1.06 | 0.342 | 0.0085 | +0.149 | 1.00 | 1.15 | +0.71 | 0.655 | 52% | 67% |
| Cui et al. 2026 | SERT | 18 | 0.52 | 0.468 | 0.0018 | +0.038 | 0.84 | 1.22 | +0.67 | 0.687 | 68% | 90% |

*Internal row is the production configuration, 20 seeds × 5 scaffold folds. `retained` is external r² as a percentage of the internal r² for that target. **RMSE is in log units and does not depend on the test set's variance, so it separates two things r² fuses: whether the model got worse, and whether the test set gave r² anything to explain.** `guess-mean RMSE` is the error from ignoring the molecule and predicting the training mean — the model beats it in 8 of 9 cells. Across the nine cells r² spans 32-fold while RMSE spans 1.7-fold; at Stalberga SERT the RMSE of 0.69 is *better* than the model's own internal 0.82 while r² reads 0.047. For scale, independent public measurements of the same compound disagree by σ ≈ 0.54 (Kᵢ) to 0.68 (IC₅₀) log units. Each ceiling is 1 − σ²ₑ/var(y) on that row's own response variance. Source: `competency2.json`, `internal_both_configs.json`.*

---

## 2. Not a defect of this model

### Table 2a — Improving the model: internal effect of the configuration changes

| target | answer key | published config | ± SD | production config | ± SD  | Δ internal | seeds won | p |
|---|---|---|---|---|---|---|---|---|
| DAT | binding Kᵢ key | 0.399 | 0.019 | 0.556 | 0.015 | +0.157 | 20/20 | 5e-22 |
| DAT | uptake IC₅₀ key | 0.423 | 0.023 | 0.557 | 0.014 | +0.134 | 20/20 | 5e-20 |
| NET | binding Kᵢ key | 0.418 | 0.010 | 0.520 | 0.014 | +0.101 | 20/20 | 4e-17 |
| NET | uptake IC₅₀ key | 0.409 | 0.013 | 0.502 | 0.010 | +0.093 | 20/20 | 9e-18 |
| SERT | binding Kᵢ key | 0.192 | 0.013 | 0.474 | 0.016 | +0.281 | 20/20 | 1e-26 |
| SERT | uptake IC₅₀ key | 0.319 | 0.015 | 0.524 | 0.013 | +0.204 | 20/20 | 2e-23 |

*Production configuration drops the all-three-transporter intersection requirement for training and adds a binary assay-modality feature. 20 seeds × 5 scaffold folds, identical folds for both arms, paired t-test. Source: `production_cv.json`.*

### Table 2b — The same changes, measured externally

| set | target | n | published config | drop intersection | + modality feature | separate models | Δ best vs published |
|---|---|---|---|---|---|---|---|
| Stalberga 2026 | DAT | 46 | 0.373 | 0.454 | 0.458 | 0.492 | 0.119 |
| Stalberga 2026 | NET | 46 | 0.062 | 0.079 | 0.092 | 0.020 | 0.030 |
| Stalberga 2026 | SERT | 46 | 0.032 | 0.012 | 0.040 | 0.038 | 0.008 |
| PDSP / BindingDB | DAT | 43 | 0.247 | 0.371 | 0.354 | 0.182 | 0.124 |
| PDSP / BindingDB | NET | 43 | 0.090 | 0.192 | 0.202 | 0.146 | 0.112 |
| PDSP / BindingDB | SERT | 43 | 0.164 | 0.375 | 0.372 | 0.361 | 0.212 |
| Cui et al. 2026 | DAT | 11 | 0.426 | 0.189 | 0.117 | 0.048 | -0.237 |
| Cui et al. 2026 | NET | 19 | 0.242 | 0.328 | 0.313 | 0.233 | 0.086 |
| Cui et al. 2026 | SERT | 18 | 0.266 | 0.094 | 0.144 | 0.571 | 0.305 |

*Leakage controlled at InChIKey-skeleton level plus a Tanimoto < 0.99 filter, because ECFP4 as generated here does not encode chirality and the full InChIKey therefore leaves identical-fingerprint stereoisomers in training. Aggregate across the nine cells: drop-intersection +0.022 (6/9 cells, sign test p = 0.51); + modality +0.021 (7/9, p = 0.18). Source: `modality_external_strict.json`.*

### Table 2c — Architecture and feature levers, change in internal r²

| lever | DAT | NET | SERT |
|---|---|---|---|
| multi-task (one-hot GBM) | -0.073 (p 0.000) | -0.049 (p 0.000) | -0.056 (p 0.000) |
| + 204 RDKit descriptors | -0.007 (p 0.129) | -0.011 (p 0.010) | +0.011 (p 0.009) |
| multi-task + descriptors | -0.060 (p 0.000) | -0.051 (p 0.001) | -0.032 (p 0.002) |
| hyperparameter tuning | +0.012 (p 0.043) | +0.012 (p 0.001) | +0.034 (p 0.000) |
| D-MPNN, 3 heads | -0.027 (1 seed) | -0.039 (1 seed) | -0.036 (1 seed) |

*Baseline is the production configuration. All arms share one global scaffold split so a compound lands in the same fold throughout; 5 seeds except the D-MPNN, which is a single seed, 35 epochs, untuned and un-ensembled. The D-MPNN is worse on r² but better calibrated — slope +0.05 and bias ≈ 0 against −0.065 for the booster. Source: `improve_experiment.json`, `tune_experiment.json`, `dmpnn.json`.*

---

## 3. Not a defect of this model class

### Table 3a — Eight model classes, internal and external

| model class | era | DAT int | DAT ext | NET int | NET ext | SERT int | SERT ext |
|---|---|---|---|---|---|---|---|
| mean | null | 0.034 |  | 0.016 |  | 0.012 |  |
| Hansch MLR | 1964 | 0.074 | 0.029 | 0.074 | 0.170 | 0.000 | 0.034 |
| PLS | 1980s | 0.422 | 0.179 | 0.392 | 0.251 | 0.343 | 0.072 |
| Ridge | — | 0.522 | 0.356 | 0.469 | 0.296 | 0.427 | 0.149 |
| kNN-Tanimoto | read-across | 0.499 | 0.262 | 0.461 | 0.142 | 0.455 | 0.113 |
| SVR (RBF) | 2000s | 0.466 | 0.216 | 0.400 | 0.237 | 0.385 | 0.142 |
| Random Forest | 2000s | 0.534 | 0.184 | 0.500 | 0.274 | 0.476 | 0.347 |
| LightGBM | 2010s | 0.555 | 0.338 | 0.519 | 0.200 | 0.499 | 0.161 |

*Internal: 5-fold scaffold CV, seed 0, pooled label, one row per compound — the classical QSAR setup, chosen so every family can be run identically. External: trained on everything with external compounds removed at skeleton level; the external column is the mean over the three sets. The null row has no external value because a constant prediction has zero variance. Mean internal 0.403 → mean external 0.198, i.e. 49% retained. Source: `model_class_sweep.json`.*

### Table 3b — Are the external differences between classes resolvable?

| target | external set | n | worst class | best class | spread across 7 classes | width of one 95% CI | spread > noise? |
|---|---|---|---|---|---|---|---|
| DAT | Stalberga 2026 | 46 | 0.003 | 0.498 | 0.495 | 0.423 | yes |
| DAT | PDSP / BindingDB | 43 | 0.057 | 0.371 | 0.314 | 0.448 | no |
| DAT | Cui et al. 2026 | 11 | 0.005 | 0.503 | 0.498 | 0.624 | no |
| NET | Stalberga 2026 | 46 | 0.008 | 0.123 | 0.115 | 0.280 | no |
| NET | PDSP / BindingDB | 43 | 0.178 | 0.393 | 0.215 | 0.401 | no |
| NET | Cui et al. 2026 | 19 | 0.088 | 0.651 | 0.563 | 0.639 | no |
| SERT | Stalberga 2026 | 46 | 0.003 | 0.147 | 0.144 | 0.115 | yes |
| SERT | PDSP / BindingDB | 43 | 0.071 | 0.375 | 0.304 | 0.448 | no |
| SERT | Cui et al. 2026 | 18 | 0.015 | 0.617 | 0.602 | 0.423 | yes |

*The 95% interval is for a squared correlation at that n, via the Fisher z-transform of |r|. In six of nine cells the entire spread between seven model classes is narrower than the confidence interval on any single one of them, so no class can be distinguished from another externally. Internally, where n is 3,300–5,300, LightGBM is top of the table at all three targets. Source: `model_class_sweep.json`, `within_external.json`.*

---

## 4. Not bad external data

### Table 4 — Can each external set predict itself?

| set | target | n | variance | range | LOO Ridge | LOO kNN(3) | LOO LightGBM | best within-set | transferred | ratio | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Stalberga 2026 | DAT | 46 | 0.80 | 5.3–9.0 | 0.453 | 0.381 | 0.376 | 0.453 | 0.454 | 1.0× | ambiguous |
| Stalberga 2026 | NET | 46 | 0.57 | 4.6–7.6 | 0.200 | 0.128 | 0.241 | 0.241 | 0.079 | 3.0× | transfer |
| Stalberga 2026 | SERT | 46 | 0.37 | 4.6–7.1 | 0.132 | 0.033 | 0.270 | 0.270 | 0.012 | 22.8× | transfer |
| PDSP / BindingDB | DAT | 43 | 0.71 | 5.0–7.8 | 0.328 | 0.478 | 0.480 | 0.480 | 0.371 | 1.3× | ambiguous |
| PDSP / BindingDB | NET | 43 | 0.93 | 5.1–8.9 | 0.424 | 0.356 | 0.295 | 0.424 | 0.192 | 2.2× | transfer |
| PDSP / BindingDB | SERT | 43 | 1.60 | 5.4–9.9 | 0.405 | 0.203 | 0.334 | 0.405 | 0.375 | 1.1× | ambiguous |
| Cui et al. 2026 | DAT | 11 | 0.37 | 4.2–6.4 | 0.074 | 0.172 | 0.286 | 0.286 | 0.189 | 1.5× | ambiguous |
| Cui et al. 2026 | NET | 19 | 1.24 | 4.8–8.6 | 0.009 | 0.011 | 0.290 | 0.290 | 0.328 | 0.9× | ambiguous |
| Cui et al. 2026 | SERT | 18 | 0.77 | 5.6–8.9 | 0.184 | 0.046 | 0.270 | 0.270 | 0.094 | 2.9× | transfer |

*Leave-one-out within the set, same features as the main model. `transferred` is the production model's r² on the same compounds. LOO within a single-paper series is generous — a held-out compound usually has a close analogue among the rest — so these are an UPPER BOUND on learnability, which is the right bound for asking whether the data contains signal at all. No cell is data-limited. The two worst cells in the paper, Stalberga NET and SERT, predict themselves 3× and 23× better than the transferred model does. Source: `within_external.json`.*

---

## 5. The limit is the training–test relationship

### Table 5a — Pushing the training set away from a fixed test set

| τ (similarity ceiling) | DAT NN | DAT r² sim | DAT r² rand | NET NN | NET r² sim | NET r² rand | SERT NN | SERT r² sim | SERT r² rand |
|---|---|---|---|---|---|---|---|---|---|
| 1.000 | 0.643 | 0.532 | 0.532 | 0.640 | 0.552 | 0.552 | 0.658 | 0.507 | 0.507 |
| 0.900 | 0.638 | 0.523 | 0.529 | 0.632 | 0.552 | 0.563 | 0.647 | 0.505 | 0.510 |
| 0.800 | 0.622 | 0.500 | 0.522 | 0.620 | 0.552 | 0.563 | 0.631 | 0.499 | 0.508 |
| 0.700 | 0.585 | 0.474 | 0.530 | 0.589 | 0.520 | 0.571 | 0.601 | 0.459 | 0.501 |
| 0.600 | 0.507 | 0.422 | 0.519 | 0.515 | 0.457 | 0.550 | 0.525 | 0.360 | 0.496 |
| 0.500 | 0.435 | 0.331 | 0.505 | 0.441 | 0.336 | 0.517 | 0.449 | 0.281 | 0.495 |
| 0.400 | 0.355 | 0.280 | 0.497 | 0.353 | 0.269 | 0.517 | 0.367 | 0.206 | 0.476 |

*A scaffold-grouped test set is drawn, then every training compound within Tanimoto > τ of any test compound is removed; six replicates. The `rand` column is the control that removes the same NUMBER of compounds at random, and it stays flat — at τ = 0.40, 48% of the training set is gone and NET falls only 0.552 → 0.517, against 0.552 → 0.269 when the same number is removed by similarity. The effect is distance, not training-set size. Source: `similarity_distance.csv`.*

### Table 5b — The fitted law, inside the training distribution

| target | fitted relationship | r | p | cost per 0.10 of NN | n runs |
|---|---|---|---|---|---|
| DAT | r² = -0.023 + 0.851 × NN | +0.741 | 2.0e-08 | 0.085 | 42 |
| NET | r² = -0.089 + 1.019 × NN | +0.810 | 8.2e-11 | 0.102 | 42 |
| SERT | r² = -0.170 + 1.033 × NN | +0.787 | 6.5e-10 | 0.103 | 42 |

*At NET and SERT the slope is indistinguishable from 1.0: r² is the median nearest-neighbour Tanimoto minus a constant. Source: `similarity_distance.csv`.*

### Table 5c — The same law applied across the boundary, where it fails

| set | target | NN Tanimoto | var ÷ train | law predicts | observed | residual |
|---|---|---|---|---|---|---|
| Stalberga | DAT | 0.51 | 0.681 | 0.412 | 0.452 | +0.040 |
| Stalberga | NET | 0.51 | 0.497 | 0.431 | 0.015 | -0.416 |
| Stalberga | SERT | 0.51 | 0.252 | 0.358 | 0.047 | -0.311 |
| PDSP | DAT | 0.63 | 0.731 | 0.516 | 0.316 | -0.200 |
| PDSP | NET | 0.63 | 0.840 | 0.556 | 0.197 | -0.360 |
| PDSP | SERT | 0.63 | 0.816 | 0.484 | 0.352 | -0.132 |
| Cui | DAT | 0.41 | 0.338 | 0.324 | 0.089 | -0.234 |
| Cui | NET | 0.40 | 1.061 | 0.315 | 0.342 | +0.027 |
| Cui | SERT | 0.40 | 0.517 | 0.245 | 0.468 | +0.222 |

*Across these nine cells, external r² correlates with nearest-neighbour Tanimoto at r = -0.026 (p = 0.95) — nothing — and with response-range coverage at r = +0.547 (p = 0.13). Inside the training distribution similarity is the binding constraint; outside it, coverage is. This is why similarity-based applicability-domain metrics are internally calibrated and externally uninformative. Source: `law_vs_external.csv`.*
