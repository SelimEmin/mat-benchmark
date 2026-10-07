"""Distance to training predicts accuracy — inside the training distribution only.

Left: test sets held fixed while the training set is pushed away from them by pruning
every compound within Tanimoto > tau. Solid lines are the similarity-pruned arm; dashed
lines are the control that removes the same NUMBER of compounds at random. The control
is flat, so the fall is caused by distance and not by the smaller training set.

Right: the fitted law extended to the three external sets. If distance were the whole
story the external points would sit on their target's line. They do not.

Palette: Okabe-Ito categorical slots, as used throughout this project's figures.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy import stats

C = {"DAT": "#0072B2", "NET": "#D55E00", "SERT": "#009E73"}
M = {"DAT": "o", "NET": "D", "SERT": "s"}
SETM = {"Stalberga": "v", "PDSP": "P", "Cui": "X"}
NAVY, INK, MUTED = "#1F4E79", "#1F2328", "#6B7280"
plt.rcParams.update({"font.family": "sans-serif",
                     "font.sans-serif": ["Liberation Sans", "DejaVu Sans"],
                     "font.size": 10, "axes.linewidth": 0.8})
TARGETS = ("DAT", "NET", "SERT")

sd = pd.read_csv("similarity_distance.csv")
ext = pd.read_csv("law_vs_external.csv")

fig, (axA, axB) = plt.subplots(1, 2, figsize=(11.6, 6.4))
fig.subplots_adjust(left=0.072, right=0.985, top=0.745, bottom=0.122, wspace=0.235)

law = {}
for t in TARGETS:
    s = sd[(sd.target == t) & (sd.arm == "similarity")]
    law[t] = stats.linregress(s.realised_nn, s.r2)
    g = s.groupby("tau").agg(nn=("realised_nn", "mean"), r2=("r2", "mean"),
                             sdv=("r2", "std")).sort_values("nn")
    axA.errorbar(g.nn, g.r2, yerr=g.sdv, marker=M[t], ms=6.5, lw=2.0, color=C[t],
                 capsize=2.5, mec="white", mew=0.9, zorder=4, label=t)
    r = sd[(sd.target == t) & (sd.arm == "random")]
    gr = r.groupby("tau").agg(nn=("realised_nn", "mean"), r2=("r2", "mean")).sort_values("nn")
    # the control is plotted against the similarity arm's distance, since removing at
    # random does not move the test set away -- it is the same x, a different training set
    axA.plot(g.nn, gr.r2.values, ls=(0, (4, 2.5)), lw=1.6, color=C[t], alpha=0.55,
             marker=M[t], ms=4.5, mfc="white", zorder=3)

axA.set_xlabel("median nearest-neighbour Tanimoto of the test set to its training set")
axA.set_ylabel("$r^2$")
axA.set_title("Pushing the training set away from a fixed test set",
              fontsize=11, color=INK, pad=9)
axA.set_xlim(0.32, 0.69)
axA.set_ylim(0.115, 0.665)
axA.grid(color="#EDEEF0", lw=0.8)
axA.set_axisbelow(True)
for sp in ("top", "right"):
    axA.spines[sp].set_visible(False)
axA.spines["bottom"].set_color("#C8CCD1")
axA.spines["left"].set_color("#C8CCD1")
axA.tick_params(colors=MUTED, length=3)
axA.annotate("same number of compounds\nremoved at random (control)",
             xy=(0.447, 0.513), xytext=(0.337, 0.628), fontsize=8.6, color=MUTED,
             ha="left", va="top",
             arrowprops=dict(arrowstyle="-", color="#B6BBC1", lw=0.9,
                             connectionstyle="arc3,rad=0.25"))
axA.annotate("pruned by similarity", xy=(0.452, 0.300), xytext=(0.497, 0.205),
             fontsize=8.6, color=MUTED, ha="left", va="center",
             arrowprops=dict(arrowstyle="-", color="#B6BBC1", lw=0.9,
                             connectionstyle="arc3,rad=0.25"))

x = np.linspace(0.33, 0.70, 50)
for t in TARGETS:
    axB.plot(x, law[t].intercept + law[t].slope * x, lw=1.7, color=C[t], alpha=0.85,
             zorder=2)
for _, r in ext.iterrows():
    axB.plot([r.nn], [r.obs], SETM[r["set"]], ms=10, color=C[r.t],
             markeredgecolor="white", markeredgewidth=1.1, zorder=5)
axB.set_xlabel("median nearest-neighbour Tanimoto to training")
axB.set_ylabel("$r^2$")
axB.set_title("The same law, with the three external sets on it",
              fontsize=11, color=INK, pad=9)
axB.set_xlim(0.32, 0.72)
axB.set_ylim(-0.03, 0.62)
axB.grid(color="#EDEEF0", lw=0.8)
axB.set_axisbelow(True)
for sp in ("top", "right"):
    axB.spines[sp].set_visible(False)
axB.spines["bottom"].set_color("#C8CCD1")
axB.spines["left"].set_color("#C8CCD1")
axB.tick_params(colors=MUTED, length=3)
axB.annotate("lines: the law fitted on the left panel", xy=(0.57, 0.50),
             xytext=(0.455, 0.575), fontsize=8.6, color=MUTED,
             arrowprops=dict(arrowstyle="-", color="#B6BBC1", lw=0.9,
                             connectionstyle="arc3,rad=0.2"))

h1 = [Line2D([], [], marker=M[t], ls="-", lw=2, ms=6.5, color=C[t],
             mec="white", label=t) for t in TARGETS]
h2 = [Line2D([], [], marker=SETM[k], ls="", ms=9, color="#55595F",
             mec="white", label=v) for k, v in
      (("Stalberga", "Stalberga 2026"), ("PDSP", "PDSP / BindingDB"),
       ("Cui", "Cui et al. 2026"))]
fig.legend(handles=h1 + h2, loc="lower center", bbox_to_anchor=(0.5, 0.002), ncol=6,
           frameon=False, fontsize=9.3, handletextpad=0.5, columnspacing=1.9)

fig.suptitle("Distance to training predicts accuracy — until the data is really external",
             fontsize=13.5, fontweight="bold", color=NAVY, x=0.008, ha="left", y=0.968)
fig.text(0.008, 0.905,
         "Left: a scaffold-grouped test set is drawn, then every training compound "
         "within Tanimoto > τ of any test compound is removed, for τ from 1.00 down to "
         "0.40; six replicates, error bars 1 SD. Each 0.10 of lost\nsimilarity costs "
         "0.085–0.103 of $r^2$ ($r$ = 0.74–0.81, $p$ < 1e-7). The dashed control removes "
         "the same number of compounds at random and stays flat, so the effect is "
         "distance, not training-set size.\nRight: across the boundary into genuinely "
         "external data the law breaks — $r^2$ against nearest-neighbour Tanimoto is "
         "$r$ = −0.03, while against response-range coverage it is $r$ = +0.55. Inside "
         "the\ntraining distribution similarity is what limits the model; outside it, "
         "coverage is.",
         fontsize=8.6, color=MUTED, ha="left", va="top", linespacing=1.45)

fig.savefig("fig_distance_law.png", dpi=250)
fig.savefig("fig_distance_law.pdf")
print("wrote fig_distance_law.png / .pdf")
