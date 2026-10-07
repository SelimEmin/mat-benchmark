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
