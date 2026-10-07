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
