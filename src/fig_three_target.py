"""The NET offset is target-specific, not a laboratory artefact.

One panel. Each row is a source in which BOTH modalities were measured on the same
compounds by the same group. Three markers per row, one per transporter. NET is the
only target on which every source agrees in sign.

    python fig_three_target.py                -> six sources (default)
    python fig_three_target.py --no-eshleman  -> the five ChEMBL papers only

The pooled row is recomputed to match whichever set of sources is shown: one value per
compound, so the two compounds common to Eshleman and the ChEMBL series (cocaine,
mazindol) are counted once. See pool_six.py.
"""
import json
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Okabe-Ito categorical slots; validated with scripts/validate_palette.js
# (all checks PASS; worst adjacent CVD dE 11.0 deutan, 25.8 normal vision)
C = {"DAT": "#0072B2", "NET": "#D55E00", "SERT": "#009E73"}
M = {"DAT": "o", "NET": "D", "SERT": "s"}          # secondary encoding, not colour alone
INK, MUTED, FAINT = "#1F2328", "#6B7280", "#C9CDD3"
plt.rcParams.update({"font.family": "sans-serif",
                     "font.sans-serif": ["Liberation Sans", "DejaVu Sans"],
                     "font.size": 9.5, "axes.linewidth": 0.8})

WITH_ESHLEMAN = "--no-eshleman" not in sys.argv
SUFFIX = "" if WITH_ESHLEMAN else "_chembl_only"

per = json.load(open("per_document_3targets.json"))
agg = json.load(open("within_doc_3targets.json"))
six = json.load(open("pool_six.json"))

PAPERS = [
    ("CHEMBL1132179", "Choi et al. 1999", "non-piperazine GBR analogues"),
    ("CHEMBL1135775", "Houlihan et al. 2002", "mazindol analogues"),
    ("CHEMBL1136239", "Axford et al. 2003", "bicyclo[2.2.1]heptanes"),
    ("CHEMBL1138144", "Meltzer et al. 2006", "pyrovalerone analogues"),
    ("CHEMBL1146223", "Gong et al. 2007", "3$\\beta$-aryltropanes"),
]

rows = [(lab, sub, {t: (per[t][doc]["median"], per[t][doc]["n"], None) for t in C}, "paper")
        for doc, lab, sub in PAPERS]

if WITH_ESHLEMAN:
    e = agg["eshleman"]
    rows.append(("Eshleman et al. 2017", "cathinones, uptake blockers only",
                 {t: (e[t]["blocker_median"], e[t]["blocker_n"], e[t]["blocker_ci95"])
                  for t in C}, "paper"))
    rows.append(("All within-document pairs", "pooled across the six sources above",
                 {t: (six[t]["median"], six[t]["n"], six[t]["ci95"]) for t in C}, "pooled"))
else:
    w = agg["chembl"]
    rows.append(("All within-document pairs", "pooled across the five papers above",
                 {t: (w[t]["within_median"], w[t]["within_n"], w[t]["within_ci95"])
                  for t in C}, "pooled"))

n = len(rows)
XLO, XHI = -1.62, 1.72
NCOL = -1.15                     # left-hand column of compound counts
PLOT0 = -1.02                    # where the data region starts

fig, ax = plt.subplots(figsize=(7.4, 5.1) if WITH_ESHLEMAN else (7.0, 4.5))
OFF = {"DAT": +0.255, "NET": 0.0, "SERT": -0.255}

ax.axvline(0, color="#9AA0A6", lw=1.0, zorder=2)
ax.axvline(PLOT0 - 0.05, color=FAINT, lw=0.8, zorder=1)

for i, (lab, sub, vals, kind) in enumerate(rows):
    y0 = n - 1 - i
    if kind == "pooled":
        ax.axhspan(y0 - 0.47, y0 + 0.47, color="#F2F4F6", zorder=0)
    for t in ("DAT", "NET", "SERT"):
        v, cnt, ci = vals[t]
        y = y0 + OFF[t]
        ax.text(NCOL, y, "—" if v is None else str(cnt), ha="right", va="center",
                fontsize=7.3, color=MUTED)
        if v is None:
            continue
        right = v
        if ci is not None:
            ax.plot(ci, [y, y], color=C[t], lw=1.7, alpha=0.45, zorder=3,
                    solid_capstyle="round")
            right = max(ci[1], v)
        ms = 8.6 if kind == "pooled" else 7.0
        ax.plot([v], [y], M[t], ms=ms, color=C[t], markeredgecolor="white",
                markeredgewidth=1.1, zorder=5)
        if kind == "pooled" or t == "NET":
            # negatives get their label on the left so it never crosses the zero line
            anchor, dx, ha = right, 7, "left"
            if v < 0:
                anchor = min(ci[0], v) if ci is not None else v
                dx, ha = -7, "right"
            ax.annotate(f"{v:+.2f}", (anchor, y), xytext=(dx, 0),
                        textcoords="offset points", ha=ha, va="center",
                        fontsize=8.1, color=MUTED,
                        fontweight="bold" if kind == "pooled" else "normal")
    if kind == "pooled":
        ax.axhline(y0 + 0.53, color=FAINT, lw=0.9, zorder=1)

    ax.text(NCOL - 0.16, y0 + 0.19, lab, ha="right", va="center", fontsize=9.3,
            color=INK, fontweight="bold" if kind == "pooled" else "normal")
    ax.text(NCOL - 0.16, y0 - 0.21, sub, ha="right", va="center", fontsize=7.8,
            color=MUTED, style="italic")

ax.text(NCOL, n - 0.44, "n", ha="right", va="center", fontsize=7.6, color=FAINT)
ax.set_xlim(XLO, XHI)
ax.set_ylim(-0.95, n + 0.12)
ax.set_yticks([])
ax.set_xticks([-1.0, -0.5, 0, 0.5, 1.0, 1.5])
ax.xaxis.grid(True, color="#F0F1F3", lw=0.9)
ax.set_axisbelow(True)
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
ax.spines["bottom"].set_bounds(PLOT0 - 0.05, XHI)
ax.spines["bottom"].set_color("#C8CCD1")
ax.tick_params(length=0, colors=MUTED, pad=4)

ax.set_xlabel("functional pIC$_{50}$ $-$ binding p$K_\\mathrm{i}$   (log units)",
              fontsize=10, labelpad=22, color=INK, x=0.62)
ax.annotate("← functional assay reads less potent", (0, 0),
            xycoords=("data", "axes fraction"), xytext=(-7, -24),
            textcoords="offset points", ha="right", va="top",
            fontsize=8, color=MUTED, annotation_clip=False)
ax.annotate("functional assay reads more potent →", (0, 0),
            xycoords=("data", "axes fraction"), xytext=(7, -24),
            textcoords="offset points", ha="left", va="top",
            fontsize=8, color=MUTED, annotation_clip=False)

h = [Line2D([], [], marker=M[t], ls="", ms=7, color=C[t], markeredgecolor="white",
            markeredgewidth=1.0, label=t) for t in ("DAT", "NET", "SERT")]
leg = ax.legend(handles=h, loc="upper right", frameon=False, fontsize=9.5,
                handletextpad=0.4, ncol=3, columnspacing=1.4,
                bbox_to_anchor=(1.005, 1.035))
for txt in leg.get_texts():
    txt.set_color(INK)

fig.suptitle("Within a single paper, the binding–functional gap appears at NET "
             "and almost nowhere else",
             fontsize=11.6, fontweight="bold", color="#1F4E79", x=0.012, ha="left",
             y=0.99)
fig.text(0.012, 0.938,
         "Each row is a source that measured both assay types on the same compounds in "
         "the same laboratory, so no between-study\ncalibration difference can "
         "contribute. Bars are 95% bootstrap intervals where the underlying values were "
         "available.",
         fontsize=8.3, color=MUTED, ha="left", va="top")
fig.tight_layout(rect=(0.0, 0.0, 1, 0.925))
fig.savefig(f"fig_three_target{SUFFIX}.png", dpi=250, bbox_inches="tight")
fig.savefig(f"fig_three_target{SUFFIX}.pdf", bbox_inches="tight")
print(f"wrote fig_three_target{SUFFIX}.png / .pdf")
