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
