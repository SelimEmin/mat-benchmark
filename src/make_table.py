"""The eight-diagnostic comparison as a table: markdown for the manuscript, CSV for reuse.

Same fifteen rows as fig_diagnostics_v2.py. Internal rows carry the 20-seed standard
deviation on r^2, which the figure cannot show.
"""
import json
import numpy as np
import pandas as pd

TARGETS = ("DAT", "NET", "SERT")
ibc = json.load(open("internal_both_configs.json"))
comp = json.load(open("competency2.json"))
KEY = {"Stalberga 2026": "Stalberga 2026  (stimulants, uptake IC50)",
       "PDSP / BindingDB": "PDSP/BindingDB  (antidepressants, binding Ki)",
       "Cui et al. 2026": "Cui 2026        (SNRI series, uptake IC50)"}
MOD = {"Stalberga 2026": "IC50", "PDSP / BindingDB": "Ki", "Cui et al. 2026": "IC50"}
SIG = {"DAT": 0.445, "NET": 0.655, "SERT": 0.492}
TV = {"DAT": 1.099, "NET": 1.170, "SERT": 1.497}

rows = []
for cfg, label in (("published", "Internal CV — published"),
                   ("production", "Internal CV — production")):
    for t in TARGETS:
        d = ibc[cfg][t]
        rows.append(dict(evaluation=label, target=t, n=d["n_train_pool"],
                         n_scored=d["n"], modality="both",
                         var_ratio=1.00, r2=d["r2"]["mean"], r2_sd=d["r2"]["sd"],
                         R2det=d["R2det"]["mean"], slope=d["slope"]["mean"],
                         bias=d["bias"]["mean"], rho=d["rho"]["mean"],
                         nn_tanimoto=d["nn_tanimoto_median"],
                         ceiling=d["ceiling"], pct_ceiling=d["pct_ceiling"]["mean"]))
for name, key in KEY.items():
    for t in TARGETS:
        d = comp[key]["targets"][t]
        ceil = 1 - SIG[t] ** 2 / d["var"]
        rows.append(dict(evaluation=name, target=t, n=d["n"], n_scored=d["n"],
                         modality=MOD[name], var_ratio=d["var"] / TV[t],
                         r2=d["r2_pearson"], r2_sd=np.nan,
                         R2det=d["R2_determination"], slope=d["slope"],
                         bias=d["bias"], rho=d["spearman"],
                         nn_tanimoto=d["nn_tanimoto_median"], ceiling=ceil,
                         pct_ceiling=100 * d["r2_pearson"] / ceil))

df = pd.DataFrame(rows)
df.to_csv("diagnostics_table.csv", index=False)

H = ["evaluation", "", "var ÷ train", "r²", "R²_det", "slope", "bias", "ρ",
     "NN Tan", "ceiling", "% ceil"]
lines = ["| " + " | ".join(H) + " |",
         "|" + "---|" * len(H)]
prev = None
for _, r in df.iterrows():
    ev = "" if r.evaluation == prev else f"**{r.evaluation}**"
    prev = r.evaluation
    r2 = f"{r.r2:.3f}" + (f" ± {r.r2_sd:.3f}" if np.isfinite(r.r2_sd) else "")
    lines.append("| " + " | ".join([
        ev, f"**{r.target}**", f"{r.var_ratio:.2f}", r2, f"{r.R2det:+.3f}",
        f"{r.slope:.3f}", f"{r.bias:+.3f}", f"{r.rho:+.2f}",
        f"{r.nn_tanimoto:.2f}", f"{r.ceiling:.3f}", f"{r.pct_ceiling:.0f}%"]) + " |")
md = "\n".join(lines)
open("diagnostics_table.md", "w").write(md + "\n")
print(md)

print("\n\nTraining pools and scored counts")
print(f"{'evaluation':<26}{'target':<7}{'train pool':>12}{'scored':>9}{'modality':>10}")
for _, r in df.iterrows():
    print(f"{r.evaluation:<26}{r.target:<7}{r.n:>12,}{r.n_scored:>9,}{r.modality:>10}")
print("\nwrote diagnostics_table.csv and diagnostics_table.md")
