"""Draw the ACS table-of-contents graphic.

Size is the ACS specification enforced by achemso.cls: 3.25 x 1.75 inches.
Palette and typography follow make_figs.py so the graphic matches figures 1-13.

Values are the published Table 10 estimates with their 95% intervals, typed from the
manuscript rather than recomputed, on the same principle as fix_figs.py: no figure may
disagree with a table.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.transforms import blended_transform_factory as blend

BLUE, VERM, TEAL = "#0072B2", "#D55E00", "#009E73"
GREY = "#666666"
plt.rcParams.update({"font.family": "sans-serif",
                     "font.sans-serif": ["Liberation Sans", "DejaVu Sans"],
                     "font.size": 9, "axes.linewidth": 0.8,
                     "mathtext.fontset": "dejavusans"})

MINUS = "−"

# y, label, est, lo, hi, colour, pooled?
ROWS = [
    (0, "[$^{3}$H]paroxetine",  -1.044, -1.39, -0.70, TEAL, False),
    (1, "[$^{3}$H]WIN 35,428",  -0.308, -0.49, -0.13, BLUE, False),
    (2, "[$^{125}$I]RTI-55",    +0.183, -0.01, +0.38, BLUE, False),
    (3, "[$^{3}$H]nisoxetine",  +0.038, -0.18, +0.26, VERM, False),
    (4, "[$^{125}$I]RTI-55",    +0.507, +0.34, +0.68, VERM, False),
    (5, "pooled",               +0.261, +0.06, +0.46, VERM, True),
]
GROUPS = [("NET", 4.0, VERM), ("DAT", 1.5, BLUE), ("SERT", 0.0, TEAL)]

fig, ax = plt.subplots(figsize=(3.25, 1.75))
fig.subplots_adjust(left=0.375, right=0.995, top=0.845, bottom=0.215)

ax.axhspan(2.62, 5.45, color=VERM, alpha=0.07, lw=0, zorder=0)
ax.axvline(0, color=GREY, lw=0.7, ls=(0, (2.5, 2)), zorder=1)
# where the pooled estimate falls relative to the two strata it averages
ax.plot([0.261, 0.261], [3.0, 5.0], color=VERM, lw=0.7,
        ls=(0, (1.2, 1.6)), alpha=0.75, zorder=2)

for y, lab, est, lo, hi, col, pooled in ROWS:
    ax.plot([lo, hi], [y, y], color=col, lw=1.5,
            solid_capstyle="butt", alpha=0.85, zorder=3)
    ax.plot([est], [y], marker="D" if pooled else "o",
            ms=4.4 if pooled else 4.2,
            mfc="white" if pooled else col, mec=col,
            mew=1.4 if pooled else 0.9, zorder=4)
    ax.text(est, y + 0.26, f"{est:+.2f}".replace("-", MINUS),
            ha="center", va="bottom", fontsize=6.4,
            color=col, fontweight="bold", zorder=5)

lab_tr = blend(ax.transAxes, ax.transData)
for y, lab, *_rest in ROWS:
    ax.text(-0.035, y, lab, transform=lab_tr, ha="right", va="center",
            fontsize=6.4, color="#222222",
            fontweight="bold" if _rest[-1] else "normal")
for name, y, col in GROUPS:
    ax.text(-0.565, y, name, transform=lab_tr, ha="left", va="center",
            fontsize=6.8, color=col, fontweight="bold")

ax.set_xlim(-1.55, 0.90)
ax.set_ylim(-0.6, 5.75)
ax.set_yticks([])
ax.set_xticks([-1.0, -0.5, 0.0, 0.5])
ax.set_xticklabels([MINUS + "1.0", MINUS + "0.5", "0", "+0.5"])
ax.tick_params(axis="x", colors="#333333", labelsize=6, length=2.5, pad=1.5)
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
ax.spines["bottom"].set_color("#999999")

ax.set_xlabel(f"functional {MINUS} binding potency (log units)",
              fontsize=6.4, color="#333333", labelpad=1.5)
fig.text(0.012, 0.985, "One target, one format, two assay cells",
         ha="left", va="top", fontsize=7.6, fontweight="bold", color="#111111")

fig.savefig("figures/numbered/toc_graphic.pdf")
fig.savefig("figures/numbered/toc_graphic.png", dpi=600)
plt.close(fig)
print("wrote figures/numbered/toc_graphic.{pdf,png}")
