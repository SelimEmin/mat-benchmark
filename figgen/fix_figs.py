"""Rebuild three figures so that every plotted value is one the manuscript states.

fig7 and fig10 are re-made from the published table values rather than from the
rebuild, so figures and tables cannot disagree.

fig8 loses its panel (a). The extended-set scatter requires predictions from the
model with the four contributing source documents excluded from training; that model
is not in the deposit, and a rebuild without the document-level exclusion gives
R^2 = 0.408 against the published 0.183 -- optimistic, because it is the leakage the
exclusion was there to prevent. Plotting it would contradict the text. The panel is
therefore dropped rather than approximated, and the deposit gap is recorded.
"""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BLUE, VERM, NAVY = "#0072B2", "#D55E00", "#1F4E79"
TEAL, GOLD, GREY = "#009E73", "#E69F00", "#666666"
plt.rcParams.update({"font.family": "sans-serif",
                     "font.sans-serif": ["Liberation Sans", "DejaVu Sans"],
                     "font.size": 9, "axes.linewidth": 0.8})
D = json.load(open("figgen/figdata.json"))
T = ("DAT", "NET", "SERT")
COLS = {"DAT": BLUE, "NET": VERM, "SERT": TEAL}
OUT = "tex/figures/"


def tidy(ax, left=True):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_visible(left)
    ax.tick_params(colors="#333333", labelsize=8)


def save(fig, n):
    fig.savefig(OUT + n + ".pdf", bbox_inches="tight"); plt.close(fig); print("  ", n)


# ---- fig 7, from the published external values (Tables 3 and 5)
PUB = {"DAT": dict(r2=0.459, rmse=1.174), "NET": dict(r2=0.015, rmse=0.848),
       "SERT": dict(r2=0.058, rmse=0.687)}
for t in T:                       # CCC is not tabulated; recomputed on the same points
    PUB[t]["ccc"] = D["targets"][t]["ccc"]
fig, ax = plt.subplots(figsize=(5.2, 2.7))
x = np.arange(3); w = .26
ax.bar(x - w, [PUB[t]["r2"] for t in T], w, color=NAVY, zorder=3, label="$R^2$")
ax.bar(x, [PUB[t]["ccc"] for t in T], w, color=BLUE, zorder=3, label="CCC")
ax.bar(x + w, [PUB[t]["rmse"] for t in T], w, color=VERM, zorder=3,
       label="RMSE (log units)")
ax.axhline(0.127, ls=":", color="#222222", lw=1.0, zorder=4)
ax.text(2.46, 0.147, "measurement floor", fontsize=7, ha="right", color="#222222")
ax.set_xticks(x); ax.set_xticklabels(T, fontsize=8.5)
ax.set_ylabel("value", fontsize=8.6); ax.set_ylim(0, 1.45); tidy(ax)
ax.legend(frameon=False, fontsize=7.6, loc="upper left", handlelength=1.1)
ax.text(.52, .93, "$R^2$ ranks DAT best and NET worst;\nRMSE puts DAT last",
        transform=ax.transAxes, fontsize=7.4, ha="center", color=GREY)
save(fig, "fig7_metrics")

# ---- fig 8, panel (b) only
BLK = [("response-range\nposition", .444, .308), ("chemical\nsimilarity", .024, .022),
       ("source\nidentity", .178, .021)]
fig, ax = plt.subplots(figsize=(4.6, 2.8))
x = np.arange(3); w = .34
ax.bar(x - w / 2, [b[1] for b in BLK], w, color="#C9D4DF", zorder=3, label="alone")
ax.bar(x + w / 2, [b[2] for b in BLK], w, color=NAVY, zorder=3, label="unique")
for i, b in enumerate(BLK):
    ax.text(i - w / 2, b[1] + .012, f"{b[1]:.3f}", ha="center", fontsize=7.2)
    ax.text(i + w / 2, b[2] + .012, f"{b[2]:.3f}", ha="center", fontsize=7.2)
ax.set_xticks(x); ax.set_xticklabels([b[0] for b in BLK], fontsize=7.6)
ax.set_ylabel("$R^2$ of |prediction error|", fontsize=8.2); ax.set_ylim(0, .56); tidy(ax)
ax.legend(frameon=False, fontsize=7.6, loc="upper right", handlelength=1.1)
ax.text(.52, .70, "source identity adds only 2.1%\nonce range is accounted for",
        transform=ax.transAxes, fontsize=7.2, ha="center", color=GREY)
save(fig, "fig8_source")

# ---- fig 10, with the published above-percentile counts
ABOVE = {"DAT": 11, "NET": 0, "SERT": 0}   # against the UNCENSORED training 75th pct
fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.5), sharey=True,
                         gridspec_kw=dict(wspace=.12))
for ax, t in zip(axes, T):
    d = D["targets"][t]
    tr = np.array(d["train_unc"]); ex = np.array(d["y"]); q75 = d["train_q"][2]
    ax.hist(tr, bins=40, density=True, color="#D6DDE4", zorder=2, label="training")
    ax.hist(ex, bins=16, density=True, color=COLS[t], alpha=.72, zorder=3,
            label="external")
    ax.axvline(q75, ls="--", color="#222222", lw=1.0, zorder=4)
    n = ABOVE[t]
    ax.text(.04, .95, f"{n if n else 'NONE'} of 42 above the\ntraining 75th pct",
            transform=ax.transAxes, fontsize=7.4, va="top")
    ax.set_xlim(3.6, 10.4); ax.set_xlabel(f"p{t}", fontsize=8.4)
    ax.set_title(f"{'abc'[T.index(t)]}  {t}", fontsize=8.6, loc="left", pad=5)
    tidy(ax, left=(t == "DAT"))
axes[0].set_ylabel("density", fontsize=8.6)
axes[2].legend(frameon=False, fontsize=7.4, loc="upper right")
save(fig, "fig10_coverage")
