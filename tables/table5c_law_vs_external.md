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
