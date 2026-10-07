"""Eight diagnostics across the internal cross-validation and three external sets.

Read ACROSS a row for one evaluation's full profile; read DOWN a column to compare
one diagnostic across evaluations. Each panel carries the reference value that makes
its scale interpretable (1.0 for a slope that tracks reality, 0 for no bias, and so on).

Palette: Okabe-Ito categorical slots, validated with the dataviz validator (all checks
PASS; worst adjacent CVD dE 11.0 deutan, 25.8 normal vision). Marker shape varies with
colour, so identity is never carried by colour alone.
"""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.transforms import blended_transform_factory

C = {"DAT": "#0072B2", "NET": "#D55E00", "SERT": "#009E73"}
M = {"DAT": "o", "NET": "D", "SERT": "s"}
NAVY, INK, MUTED, FAINT, BAND = "#1F4E79", "#1F2328", "#6B7280", "#A8AEB6", "#F2F4F6"
plt.rcParams.update({"font.family": "sans-serif",
                     "font.sans-serif": ["Liberation Sans", "DejaVu Sans"],
                     "font.size": 9.5, "axes.linewidth": 0.8})
TARGETS = ("DAT", "NET", "SERT")

intern = json.load(open("internal_diagnostics.json"))
comp = json.load(open("competency2.json"))
KEY = {"Stalberga 2026\nstimulants, IC$_{50}$": "Stalberga 2026  (stimulants, uptake IC50)",
       "PDSP / BindingDB\nantidepressants, $K_\\mathrm{i}$":
           "PDSP/BindingDB  (antidepressants, binding Ki)",
       "Cui et al. 2026\nSNRI series, IC$_{50}$": "Cui 2026        (SNRI series, uptake IC50)"}
SIG = {"DAT": 0.445, "NET": 0.655, "SERT": 0.492}
TV = {"DAT": 1.099, "NET": 1.170, "SERT": 1.497}
# published 20-seed means; the per-fold diagnostics below come from seed 0
PUB_R2 = {"DAT": 0.584, "NET": 0.501, "SERT": 0.541}
PUB_CEIL = {"DAT": 0.820, "NET": 0.633, "SERT": 0.838}

ROWS = []            # (evaluation, target, metrics, n)
for t in TARGETS:
    d = intern[t]
    ROWS.append(("Internal scaffold CV\n5-fold, 20 seeds", t, {
        "var": 1.00, "nn": d["nn_tanimoto_median"], "rho": d["spearman"],
        "r2": PUB_R2[t], "r2d": d["R2_determination"], "slope": d["slope"],
        "bias": d["bias"], "pct": 100 * PUB_R2[t] / PUB_CEIL[t]}, d["n"]))
for name, key in KEY.items():
    for t in TARGETS:
        d = comp[key]["targets"][t]
        ceil = 1 - SIG[t] ** 2 / d["var"]
        ROWS.append((name, t, {
            "var": d["var"] / TV[t], "nn": d["nn_tanimoto_median"],
            "rho": d["spearman"], "r2": d["r2_pearson"],
            "r2d": d["R2_determination"], "slope": d["slope"], "bias": d["bias"],
            "pct": 100 * d["r2_pearson"] / ceil}, d["n"]))

PANELS = [
    ("var", "variance ÷ training", 1.0, "same spread", (0.0, 1.22)),
    ("nn", "nearest-neighbour Tanimoto", 0.7, "close analogue", (0.30, 0.80)),
    ("rho", "Spearman $\\rho$", 0.0, "no relationship", (-0.08, 0.85)),
    ("r2", "$r^2$ (squared Pearson)", 0.0, "no relationship", (-0.03, 0.66)),
    ("r2d", "$R^2$ (determination)", 0.0, "ties the mean", (-1.05, 0.72)),
    ("slope", "slope", 1.0, "tracks reality", (0.0, 1.12)),
    ("bias", "bias (log units)", 0.0, "no shift", (-1.12, 0.68)),
    ("pct", "% of achievable ceiling", 100.0, "at the ceiling", (0, 106)),
]

n = len(ROWS)
HEAD = 0.95                                   # blank rows at the top for ref labels
fig, axes = plt.subplots(1, len(PANELS), figsize=(15.4, 6.2), sharey=True)
fig.subplots_adjust(left=0.150, right=0.995, top=0.775, bottom=0.115, wspace=0.17)

groups, seen = [], None
for i, (ev, t, _, _) in enumerate(ROWS):
    if ev != seen:
        groups.append([ev, i, i])
        seen = ev
    groups[-1][2] = i

for ax, (key, title, ref, reflab, xlim) in zip(axes, PANELS):
    for gi, (ev, a, b) in enumerate(groups):           # alternating group bands
        if gi % 2 == 0:
            ax.axhspan(n - 1 - b - 0.5, n - 1 - a + 0.5, color=BAND, zorder=0)
    if ref is not None:
        ax.axvline(ref, color="#9AA0A6", lw=1.1, ls=(0, (4, 2.5)), zorder=2)
    for i, (ev, t, m, cnt) in enumerate(ROWS):
        ax.plot([m[key]], [n - 1 - i], M[t], ms=7.4, color=C[t],
                markeredgecolor="white", markeredgewidth=1.0, zorder=5)
    ax.set_xlim(*xlim)
    ax.set_ylim(-0.6, n - 1 + HEAD)
    ax.set_title(title, fontsize=9.2, color=INK, pad=8)
    ax.xaxis.grid(True, color="#EAECEE", lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color("#C8CCD1")
    ax.tick_params(length=3, colors=MUTED, labelsize=8.3)
    if reflab:
        left = ref < (xlim[0] + xlim[1]) / 2
        ax.annotate(reflab, xy=(ref, n - 1 + 0.42), xytext=(4 if left else -4, 0),
                    textcoords="offset points", fontsize=7.4, color=MUTED,
                    va="center", ha="left" if left else "right")

# ------------------------------------------------------------- row labels
ax0 = axes[0]
ax0.set_yticks([])
T = blended_transform_factory(fig.transFigure, ax0.get_yaxis_transform())
for i, (ev, t, m, cnt) in enumerate(ROWS):
    y = n - 1 - i
    fig.text(0.113, 0, t, transform=T, ha="right", va="center",
             fontsize=9.6, color=C[t], fontweight="bold").set_position((0.113, y))
    fig.text(0.147, 0, f"$n$={cnt}", transform=T, ha="right", va="center",
             fontsize=7.6, color=FAINT).set_position((0.147, y))
for ev, a, b in groups:
    yc = n - 1 - (a + b) / 2
    fig.text(0.008, 0, ev, transform=T, ha="left", va="center", fontsize=9.8,
             color=NAVY, fontweight="bold", linespacing=1.5).set_position((0.008, yc))
    if a:                                     # rule between evaluation blocks
        fig.lines.append(plt.Line2D([0.008, 0.995], [0, 0], transform=T,
                                    color="#D7DBDF", lw=0.8, zorder=0))
        fig.lines[-1].set_ydata([n - 0.5 - a] * 2)

fig.suptitle("Eight diagnostics, four evaluations: the model ranks everywhere, "
             "and calibrates only on its own data",
             fontsize=13.5, fontweight="bold", color=NAVY, x=0.008, ha="left", y=0.975)
fig.text(0.008, 0.930,
         "Read across a row for one evaluation's whole profile; down a column to compare "
         "one diagnostic across evaluations. Dashed lines mark each scale's reference "
         "value.\nInternal $r^2$ and % of ceiling are the published 20-seed means; the other internal diagnostics come from the seed-0 fold realisation. The internal\nvariance ratio is 1.00 by construction: the held-out folds are drawn from the training distribution itself.",
         fontsize=8.6, color=MUTED, ha="left", va="top", linespacing=1.45)

h = [Line2D([], [], marker=M[t], ls="", ms=7.4, color=C[t], markeredgecolor="white",
            markeredgewidth=1.0, label=t) for t in TARGETS]
fig.legend(handles=h, loc="lower center", bbox_to_anchor=(0.57, 0.005), ncol=3,
           frameon=False, fontsize=10.5, handletextpad=0.5, columnspacing=2.6)

fig.savefig("fig_diagnostics.png", dpi=250)
fig.savefig("fig_diagnostics.pdf")
print("wrote fig_diagnostics.png / .pdf")
