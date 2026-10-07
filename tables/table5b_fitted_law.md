### Table 5b — The fitted law, inside the training distribution

| target | fitted relationship | r | p | cost per 0.10 of NN | n runs |
|---|---|---|---|---|---|
| DAT | r² = -0.023 + 0.851 × NN | +0.741 | 2.0e-08 | 0.085 | 42 |
| NET | r² = -0.089 + 1.019 × NN | +0.810 | 8.2e-11 | 0.102 | 42 |
| SERT | r² = -0.170 + 1.033 × NN | +0.787 | 6.5e-10 | 0.103 | 42 |

*At NET and SERT the slope is indistinguishable from 1.0: r² is the median nearest-neighbour Tanimoto minus a constant. Source: `similarity_distance.csv`.*
