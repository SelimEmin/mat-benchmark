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
