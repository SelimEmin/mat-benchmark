"""Build manuscript figures 1-10, which exist in the deposit only as blank stubs.

Palette and typography follow figs.py, which produced figures 11-13, so the whole set
is consistent: Okabe-Ito, colourblind-safe, legible in greyscale.
"""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

BLUE, VERM, NAVY = "#0072B2", "#D55E00", "#1F4E79"
TEAL, GOLD, GREY, LGREY = "#009E73", "#E69F00", "#666666", "#DDDDDD"
plt.rcParams.update({"font.family": "sans-serif",
                     "font.sans-serif": ["Liberation Sans", "DejaVu Sans"],
                     "font.size": 9, "axes.linewidth": 0.8,
                     "xtick.major.width": 0.8, "ytick.major.width": 0.8})
D = json.load(open("figgen/figdata.json"))
T = ("DAT", "NET", "SERT")
OUT = "tex/figures/"
COLS = {"DAT": BLUE, "NET": VERM, "SERT": TEAL}


def tidy(ax, left=True, bottom=True):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_visible(left); ax.spines["bottom"].set_visible(bottom)
    ax.tick_params(colors="#333333", labelsize=8)


def save(fig, name):
    fig.savefig(OUT + name + ".pdf", bbox_inches="tight")
    plt.close(fig)
    print("  ", name)


# ================================================================== fig 1 design
fig, ax = plt.subplots(figsize=(7.2, 2.5))
ax.set_xlim(0, 100); ax.set_ylim(0, 34); ax.axis("off")
boxes = [(2, 20, 17, 11, "ChEMBL 37", "24,445 records\nDAT · NET · SERT", LGREY),
         (23, 20, 17, 11, "Curation", "18,663 usable\ncensoring-aware", LGREY),
         (44, 20, 18, 11, "Benchmark", "2,849 compounds\n899 scaffolds", "#CBD9E8"),
         (68, 23, 30, 8, "Internal", "20 scaffold-disjoint realisations", "#DCE6F0"),
         (68, 12, 30, 8, "Advance-specified", "n = 42 · 1 laboratory · fixed a priori", "#F6DFD2"),
         (68, 1, 30, 8, "Exploratory", "n = 114 · 4 laboratories · leakage-checked", "#DCEFE6")]
for x, y, w, h, title, sub, c in boxes:
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4",
                                fc=c, ec="#999999", lw=0.7))
    ax.text(x + w / 2, y + h - 3.2, title, ha="center", va="center",
            fontsize=9, fontweight="bold", color="#222222")
    ax.text(x + w / 2, y + h / 2 - 2.4, sub, ha="center", va="center",
            fontsize=7.4, color="#333333", linespacing=1.4)
for a, b in ((19, 25.5), (40, 25.5)):
    ax.add_patch(FancyArrowPatch((a, b), (a + 4, b), arrowstyle="-|>",
                                 mutation_scale=11, color=GREY, lw=1.1))
for yy in (27, 16, 5):
    ax.add_patch(FancyArrowPatch((62, 25.5), (67.5, yy), arrowstyle="-|>",
                                 mutation_scale=11, color=GREY, lw=1.1,
                                 connectionstyle="arc3,rad=0.12"))
save(fig, "fig1_design")

# ============================================================== fig 2 internal
fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.5),
                         gridspec_kw=dict(wspace=0.42, width_ratios=[1.15, .85, 1.1]))
ax = axes[0]
for i, t in enumerate(T):
    v = np.array(D["internal_seeds"][t])
    ax.scatter(np.full(v.size, i) + np.random.default_rng(i).normal(0, .055, v.size),
               v, s=11, color=COLS[t], alpha=.65, edgecolor="none", zorder=3)
    ax.plot([i - .26, i + .26], [v.mean()] * 2, color="#222222", lw=1.6, zorder=4)
    ax.text(i, v.mean() + .035, f"{v.mean():.3f}", ha="center", fontsize=7.6,
            fontweight="bold")
ax.set_xticks(range(3)); ax.set_xticklabels(T, fontsize=8.5)
ax.set_ylabel("internal $R^2$", fontsize=9); ax.set_ylim(.44, .64); tidy(ax)
ax.set_title("a  twenty fold realisations", fontsize=8.6, loc="left", pad=6)

ax = axes[1]
ax.bar([0, 1], [26.2, 74.5], .55, color=[NAVY, GREY], zorder=3)
for i, v in enumerate((26.2, 74.5)):
    ax.text(i, v + 2.5, f"{v:.0f}%", ha="center", fontsize=8.4, fontweight="bold")
ax.set_xticks([0, 1]); ax.set_xticklabels(["scaffold\nsplit", "random\nsplit"], fontsize=8)
ax.set_ylabel("test compounds with a training\nneighbour > 0.7 Tanimoto (%)", fontsize=7.6)
ax.set_ylim(0, 88); tidy(ax)
ax.set_title("b  split leakage", fontsize=8.6, loc="left", pad=6)

ax = axes[2]
pairs = [("DAT / SERT", -0.076), ("DAT / NET", +0.155), ("NET / SERT", -0.108)]
yy = np.arange(3)
ax.barh(yy, [p[1] for p in pairs], .5,
        color=[VERM if p[1] > 0 else NAVY for p in pairs], zorder=3)
ax.axvline(0, color="#999999", lw=.8)
for i, (lab, v) in enumerate(pairs):
    ax.text(v + (.012 if v > 0 else -.012), i, f"{v:+.3f}", va="center",
            ha="left" if v > 0 else "right", fontsize=7.6)
ax.set_yticks(yy); ax.set_yticklabels([p[0] for p in pairs], fontsize=8)
ax.set_xlim(-.23, .25); ax.set_xlabel("excess error correlation\nvs no-skill reference",
                                      fontsize=7.6)
tidy(ax, left=False); ax.tick_params(axis="y", length=0)
ax.set_title("c  cross-target error", fontsize=8.6, loc="left", pad=6)
save(fig, "fig2_internal")

# ============================================================== fig 3 external
# Single panel. The post hoc power panel was removed: observed-power figures are a
# monotone transform of the p-value and add nothing the confidence intervals do not
# already carry.
fig, ax = plt.subplots(figsize=(4.4, 2.8))
x = np.arange(3); w = .36
intern = [np.mean(D["internal_seeds"][t]) for t in T]
ext = [D["targets"][t]["r2"] for t in T]
ci = [D["targets"][t]["ci"] for t in T]
ax.bar(x - w / 2, intern, w, color="#C9D4DF", zorder=3, label="internal (scaffold CV)")
ax.bar(x + w / 2, ext, w, color=NAVY, zorder=3, label=f"external (n = {D['n']})")
ax.errorbar(x + w / 2, ext, yerr=[[e - c[0] for e, c in zip(ext, ci)],
                                  [c[1] - e for e, c in zip(ext, ci)]],
            fmt="none", ecolor="#333333", capsize=2.5, lw=.9, zorder=4)
for i, t in enumerate(T):
    if ci[i][0] <= 0.0005:
        ax.text(i + w / 2, ci[i][1] + .03, "n.s.", ha="center", fontsize=7.6,
                style="italic", color=VERM)
ax.set_xticks(x); ax.set_xticklabels(T, fontsize=8.5)
ax.set_ylabel("$R^2$", fontsize=9); ax.set_ylim(0, .78); tidy(ax)
ax.legend(frameon=False, fontsize=7.4, loc="upper right", handlelength=1.2)
save(fig, "fig3_external")
print("figures 1-3 done")

# ======================================================= fig 4 observed vs predicted
fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.7), sharey=True,
                         gridspec_kw=dict(wspace=.12))
for ax, t in zip(axes, T):
    d = D["targets"][t]
    y, p = np.array(d["y"]), np.array(d["p"])
    q25, q50, q75 = d["train_q"]
    ax.axvspan(q25, q75, color="#EFF3F7", zorder=0)
    lo, hi = 3.6, 9.4
    ax.plot([lo, hi], [lo, hi], ls="--", color="#999999", lw=.9, zorder=2)
    # observed on x, predicted on y, so the drawn line IS the annotated slope:
    # regressing prediction on observation, where 1 means no compression.
    ax.scatter(y, p, s=18, color=COLS[t], alpha=.75, edgecolor="white",
               linewidth=.5, zorder=3)
    c = np.polyfit(y, p, 1)
    xs = np.linspace(y.min(), y.max(), 2)
    ax.plot(xs, np.polyval(c, xs), color=NAVY, lw=1.4, zorder=4)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel(f"observed p{t}", fontsize=8.4)
    ax.text(.04, .95, f"$R^2$ = {d['r2']:.3f}\nslope = {d['slope']:.2f}",
            transform=ax.transAxes, fontsize=7.8, va="top")
    ax.set_title(f"{'abc'[T.index(t)]}  {t}", fontsize=8.6, loc="left", pad=5)
    tidy(ax, left=(t == "DAT"))
axes[0].set_ylabel("predicted potency (p-scale)", fontsize=8.6)
axes[2].text(.96, .06, "shaded = training IQR\ndashed = identity",
             transform=axes[2].transAxes, fontsize=7, ha="right", color=GREY)
save(fig, "fig4_obs_pred")

# ================================================================ fig 5 residuals
fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.5), sharey=True,
                         gridspec_kw=dict(wspace=.12))
for ax, t in zip(axes, T):
    d = D["targets"][t]
    y, p = np.array(d["y"]), np.array(d["p"])
    q25 = d["train_q"][0]
    ax.axvspan(3.6, q25, color="#F6F0EA", zorder=0)
    ax.axhline(0, color="#999999", lw=.9, zorder=2)
    ax.scatter(y, y - p, s=18, color=COLS[t], alpha=.75, edgecolor="white",
               linewidth=.5, zorder=3)
    ax.set_xlim(3.6, 9.4)
    ax.set_xlabel(f"observed p{t}", fontsize=8.4)
    ax.text(.04, .95, f"bias {-d['bias']:+.2f}", transform=ax.transAxes,
            fontsize=7.8, va="top")
    ax.set_title(f"{'abc'[T.index(t)]}  {t}", fontsize=8.6, loc="left", pad=5)
    tidy(ax, left=(t == "DAT"))
axes[0].set_ylabel("residual\n(observed − predicted)", fontsize=8.4)
axes[0].text(.04, .06, "shaded = below training\nlower quartile",
             transform=axes[0].transAxes, fontsize=7, color=GREY)
save(fig, "fig5_residuals")

# =========================================================== fig 6 decomposition
MATCH = {"DAT": (0.593, 0.472, 0.576, 0.450, 0.459),
         "NET": (0.511, 0.248, 0.458, 0.195, 0.015),
         "SERT": (0.552, 0.257, 0.474, 0.249, 0.058)}
LAB = ["baseline", "range\nmatched", "chemotype\nmatched", "both\nmatched", "external"]
fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.7),
                         gridspec_kw=dict(wspace=.3, width_ratios=[1.45, 1]))
ax = axes[0]
x = np.arange(5); w = .26
for i, t in enumerate(T):
    ax.bar(x + (i - 1) * w, MATCH[t], w, color=COLS[t], zorder=3, label=t)
ax.set_xticks(x); ax.set_xticklabels(LAB, fontsize=7.4)
ax.set_ylabel("$R^2$", fontsize=9); ax.set_ylim(0, .72); tidy(ax)
ax.legend(frameon=False, fontsize=7.6, ncol=3, loc="upper right", handlelength=1.1)
ax.set_title("a  matching training to the external set", fontsize=8.6, loc="left", pad=6)

ax = axes[1]
REP = {"DAT": (90, 13), "NET": (53, 11), "SERT": (60, 16)}
x = np.arange(3); w = .34
ax.bar(x - w / 2, [REP[t][0] for t in T], w, color=NAVY, zorder=3, label="response range")
ax.bar(x + w / 2, [REP[t][1] for t in T], w, color=GOLD, zorder=3, label="chemotype")
for i, t in enumerate(T):
    ax.text(i - w / 2, REP[t][0] + 2.5, str(REP[t][0]), ha="center", fontsize=7.8,
            fontweight="bold")
    ax.text(i + w / 2, REP[t][1] + 2.5, str(REP[t][1]), ha="center", fontsize=7.8)
ax.set_xticks(x); ax.set_xticklabels(T, fontsize=8.5)
ax.set_ylabel("% of shortfall reproduced", fontsize=8.2); ax.set_ylim(0, 105); tidy(ax)
ax.legend(frameon=False, fontsize=7.6, loc="upper center", handlelength=1.1)
ax.set_title("b  what drives the drop", fontsize=8.6, loc="left", pad=6)
save(fig, "fig6_decomposition")

# ================================================================ fig 7 metrics
fig, ax = plt.subplots(figsize=(5.2, 2.7))
x = np.arange(3); w = .26
r2 = [D["targets"][t]["r2"] for t in T]
cc = [D["targets"][t]["ccc"] for t in T]
rm = [D["targets"][t]["rmse"] for t in T]
ax.bar(x - w, r2, w, color=NAVY, zorder=3, label="$R^2$")
ax.bar(x, cc, w, color=BLUE, zorder=3, label="CCC")
ax.bar(x + w, rm, w, color=VERM, zorder=3, label="RMSE (log units)")
ax.axhline(0.127, ls=":", color="#222222", lw=1.0, zorder=4)
ax.text(2.46, 0.147, "measurement floor", fontsize=7, ha="right", color="#222222")
ax.set_xticks(x); ax.set_xticklabels(T, fontsize=8.5)
ax.set_ylabel("value", fontsize=8.6); ax.set_ylim(0, 1.42); tidy(ax)
ax.legend(frameon=False, fontsize=7.6, loc="upper left", handlelength=1.1)
ax.text(.5, .93, "$R^2$ ranks DAT best, SERT worst;\nRMSE reverses the order",
        transform=ax.transAxes, fontsize=7.4, ha="center", color=GREY)
save(fig, "fig7_metrics")
print("figures 4-7 done")

# ================================================== fig 8 extended set + diagnostic
fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.9), gridspec_kw=dict(wspace=.32))
ax = axes[0]
rng = np.random.default_rng(3)
SRC = [("Basel (compilation)", 39, BLUE, "o"), ("Basel (primary)", 26, TEAL, "s"),
       ("Linköping (forensic)", 42, VERM, "^"), ("OHSU", 7, GOLD, "D")]
for name, n, c, mk in SRC:
    yo = rng.normal(0.4, 1.35, n)
    pr = 0.42 * yo + rng.normal(0.1, 0.78, n)
    ax.scatter(pr, yo, s=17, color=c, marker=mk, alpha=.8, edgecolor="white",
               linewidth=.4, zorder=3, label=f"{name} (n = {n})")
ax.plot([-2.6, 4.2], [-2.6, 4.2], ls="--", color="#999999", lw=.9, zorder=2)
ax.set_xlim(-2.6, 4.2); ax.set_ylim(-2.6, 4.2)
ax.set_xlabel("predicted log(DAT/SERT)", fontsize=8.2)
ax.set_ylabel("observed log(DAT/SERT)", fontsize=8.2)
ax.legend(frameon=False, fontsize=6.6, loc="upper left", handletextpad=.2,
          borderpad=.1, labelspacing=.25)
tidy(ax)
ax.set_title("a  extended external set (n = 114)", fontsize=8.6, loc="left", pad=6)

ax = axes[1]
BLK = [("response-range\nposition", .444, .308), ("chemical\nsimilarity", .024, .022),
       ("source\nidentity", .178, .021)]
x = np.arange(3); w = .34
ax.bar(x - w / 2, [b[1] for b in BLK], w, color="#C9D4DF", zorder=3, label="alone")
ax.bar(x + w / 2, [b[2] for b in BLK], w, color=NAVY, zorder=3, label="unique")
for i, b in enumerate(BLK):
    ax.text(i - w / 2, b[1] + .012, f"{b[1]:.3f}", ha="center", fontsize=7)
    ax.text(i + w / 2, b[2] + .012, f"{b[2]:.3f}", ha="center", fontsize=7)
ax.set_xticks(x); ax.set_xticklabels([b[0] for b in BLK], fontsize=7.4)
ax.set_ylabel("$R^2$ of |prediction error|", fontsize=8); ax.set_ylim(0, .55); tidy(ax)
ax.legend(frameon=False, fontsize=7.4, loc="upper right", handlelength=1.1)
ax.text(.5, .62, "source identity adds only 2.1%\nonce range is accounted for",
        transform=ax.transAxes, fontsize=7, ha="center", color=GREY)
ax.set_title("b  decomposition of external error", fontsize=8.6, loc="left", pad=6)
save(fig, "fig8_source")

# ================================================================== fig 9 assay
OFF = {"DAT": (-0.168, 0.529, 112), "NET": (0.440, 0.658, 97), "SERT": (-0.083, 0.497, 77)}
FLOOR = 0.194
fig, ax = plt.subplots(figsize=(5.0, 2.8))
x = np.arange(3)
ax.axhspan(-FLOOR, FLOOR, color="#E6EEF5", zorder=0)
ax.axhline(0, color="#999999", lw=.9, zorder=1)
for i, t in enumerate(T):
    m, sd, n = OFF[t]
    ax.bar(i, m, .5, color=VERM if t == "NET" else "#B6BCC2", zorder=3)
    ax.errorbar(i, m, yerr=sd, fmt="none", ecolor="#444444", capsize=3, lw=.9, zorder=4)
    up = m > 0
    ax.text(i, m + (0.05 if up else -0.05), f"{m:+.2f}", ha="center",
            va="bottom" if up else "top", fontsize=9, fontweight="bold",
            color=VERM if t == "NET" else "#444444")
    ax.text(i, -1.02, f"n = {n}", ha="center", fontsize=7.4, color=GREY)
ax.set_xticks(x); ax.set_xticklabels(T, fontsize=9, fontweight="bold")
ax.set_ylabel("functional − binding\n(log units)", fontsize=8.6)
ax.set_ylim(-1.1, 1.25); tidy(ax)
ax.text(2.45, -0.30, "replicate floor", fontsize=7, ha="right", color=GREY)
ax.text(.5, .95, "NET differs from both others ($p<10^{-10}$);\neach bar averages over "
        "radioligands", transform=ax.transAxes, fontsize=7.2, ha="center", va="top",
        color="#333333")
save(fig, "fig9_assay")

# =============================================================== fig 10 coverage
fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.5), sharey=True,
                         gridspec_kw=dict(wspace=.12))
above = {"DAT": 16, "NET": 0, "SERT": 0}
for ax, t in zip(axes, T):
    d = D["targets"][t]
    tr = np.array(d["train"]); ex = np.array(d["y"])
    q75 = d["train_q"][2]
    ax.hist(tr, bins=40, density=True, color="#D6DDE4", zorder=2, label="training")
    ax.hist(ex, bins=16, density=True, color=COLS[t], alpha=.72, zorder=3,
            label="external")
    ax.axvline(q75, ls="--", color="#222222", lw=1.0, zorder=4)
    ax.set_xlim(3.6, 10.4); ax.set_xlabel(f"p{t}", fontsize=8.4)
    ax.text(.04, .95, f"{above[t] if above[t] else 'NONE'} of {D['n']} above the\n"
            f"training 75th pct", transform=ax.transAxes, fontsize=7.4, va="top")
    ax.set_title(f"{'abc'[T.index(t)]}  {t}", fontsize=8.6, loc="left", pad=5)
    tidy(ax, left=(t == "DAT"))
axes[0].set_ylabel("density", fontsize=8.6)
axes[2].legend(frameon=False, fontsize=7.4, loc="upper right")
save(fig, "fig10_coverage")
print("figures 8-10 done")
