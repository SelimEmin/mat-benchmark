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
