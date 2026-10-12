"""Rebuild figures 11 and 12 from current data.

Figure 11 is drawn from the chemotype table in the manuscript, and 12 from
within_doc_3targets.json plus the Eshleman blocker panel, so figure and table cannot
disagree.

Figure 13 lives in fig13.py. It was removed from this file because the version here
applied no training-overlap exclusion and used the censored training distribution, and
running the two generators in the wrong order silently replaced the correct figure with
the stale one.
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
E = ""
OUT = "figures/numbered/"
T = ("DAT", "NET", "SERT")
COLS = {"DAT": BLUE, "NET": VERM, "SERT": TEAL}
FLOOR = 0.194


def tidy(ax, left=True):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_visible(left)
    ax.tick_params(colors="#333333", labelsize=8)


def save(fig, n):
    fig.savefig(OUT + n + ".pdf", bbox_inches="tight"); plt.close(fig); print("  ", n)


# ================================================= fig 11 chemotype strata
STRATA = [("Pyrrolidinophenones ($\\alpha$-PVP-type)", 25, 0.99, 0.69, 1.10),
          ("Fused imidazoline tricyclics", 10, 0.86, 0.38, 1.38),
          ("Naphthyl norbornanes", 5, 0.64, 0.39, 0.78),
          ("Benzhydryl ether amines (GBR-type)", 5, 0.40, -0.15, 0.45),
          ("3-Aryltropanes", 17, 0.31, -0.17, 0.39)]
POOLED = 0.54
fig, ax = plt.subplots(figsize=(6.0, 2.9))
y = np.arange(len(STRATA))[::-1]
for i, (lab, n, m, lo, hi) in enumerate(STRATA):
    ax.plot([lo, hi], [y[i]] * 2, color="#B9C3CC", lw=3.0, solid_capstyle="round",
            zorder=2)
    ax.scatter([m], [y[i]], s=70, color=VERM, zorder=4, edgecolor="white", linewidth=1.2)
    ax.text(hi + 0.05, y[i], f"n = {n}", va="center", fontsize=7.6, color=GREY)
ax.axvline(POOLED, ls=":", color="#222222", lw=1.1, zorder=3)
ax.text(POOLED, len(STRATA) - 0.42, f" pooled {POOLED:+.2f}", fontsize=7.6,
        color="#222222", va="top")
ax.axvline(0, color="#999999", lw=0.9, zorder=1)
ax.axvspan(-FLOOR, FLOOR, color="#EEF2F6", zorder=0)
ax.set_yticks(y); ax.set_yticklabels([s[0] for s in STRATA], fontsize=8)
ax.set_xlabel("NET functional $-$ binding offset (log units)", fontsize=8.4)
ax.set_xlim(-0.45, 1.62); ax.set_ylim(-0.7, len(STRATA) - 0.3)
tidy(ax, left=False); ax.tick_params(axis="y", length=0, pad=6)
ax.text(0.985, 0.04, "shaded: replicate floor $\\pm0.19$", transform=ax.transAxes,
        fontsize=7, ha="right", color=GREY)
save(fig, "fig11_chemotype")

# ============================================ fig 12 within-document offsets
W = json.load(open(E + "results/within_doc_3targets.json"))["chembl"]
DOC = {"CHEMBL1132179": "Choi 1999", "CHEMBL1135775": "Houlihan 2002",
       "CHEMBL1136239": "Axford 2003", "CHEMBL1138144": "Meltzer 2006",
       "CHEMBL1146223": "Gong 2007"}
ORDER = ["CHEMBL1132179", "CHEMBL1135775", "CHEMBL1136239", "CHEMBL1138144",
         "CHEMBL1146223"]
ESH = {"DAT": 0.279, "NET": 0.752, "SERT": 0.176}          # Eshleman 2017, blockers only
ESH_N = {"DAT": 16, "NET": 16, "SERT": 16}
POOL = {"DAT": (-0.18, 100), "NET": (0.60, 89), "SERT": (-0.05, 80)}

rows = [(DOC[d], {t: next((x["median"] for x in W[t]["documents"]
                           if x["document_chembl_id"] == d), None) for t in T},
         {t: next((x["n"] for x in W[t]["documents"]
                   if x["document_chembl_id"] == d), 0) for t in T}) for d in ORDER]
rows.append(("Eshleman 2017 (blockers)", ESH, ESH_N))
rows.append(("All sources pooled", {t: POOL[t][0] for t in T},
             {t: POOL[t][1] for t in T}))

fig, axes = plt.subplots(1, 3, figsize=(7.6, 3.1), sharey=True,
                         gridspec_kw=dict(wspace=.08))
y = np.arange(len(rows))[::-1]
for ax, t in zip(axes, T):
    ax.axvspan(-FLOOR, FLOOR, color="#EEF2F6", zorder=0)
    ax.axvline(0, color="#999999", lw=0.9, zorder=1)
    for i, (lab, med, nn) in enumerate(rows):
        v = med[t]
        if v is None:
            ax.text(0, y[i], "no pair", fontsize=7, color=GREY, ha="center",
                    va="center", style="italic")
            continue
        last = i == len(rows) - 1
        ax.scatter([v], [y[i]], s=95 if last else 62,
                   marker="D" if last else "o",
                   color=COLS[t], zorder=4, edgecolor="white", linewidth=1.1)
        ax.text(v, y[i] + 0.34, f"{nn[t]}", fontsize=6.4, color=GREY, ha="center")
    ax.axhline(y[-1] + 0.5, color="#D8DDE2", lw=0.9)
    ax.set_xlim(-1.15, 1.6); ax.set_xlabel("functional $-$ binding", fontsize=8)
    ax.set_title(t, fontsize=9.5, fontweight="bold", color=COLS[t], pad=5)
    tidy(ax, left=False); ax.tick_params(axis="y", length=0, pad=6)
axes[0].set_yticks(y); axes[0].set_yticklabels([r[0] for r in rows], fontsize=8)
axes[1].text(0.5, -0.30, "small numbers are the compound count behind each marker",
             transform=axes[1].transAxes, fontsize=7, ha="center", color=GREY)
save(fig, "fig12_within_document")
