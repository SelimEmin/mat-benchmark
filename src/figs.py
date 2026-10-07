"""Two figures.

fig12  external-set potency coverage against the training distribution
fig13  the NET assay-modality offset measured within single papers
"""
import csv, json, math
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BLUE, VERM, NAVY, GREY, LGREY = "#0072B2", "#D55E00", "#1F4E79", "#666666", "#DDDDDD"
TEAL, GOLD = "#009E73", "#E69F00"
plt.rcParams.update({"font.family": "sans-serif",
                     "font.sans-serif": ["Liberation Sans", "DejaVu Sans"],
                     "font.size": 10})

# ============================================================ fig 12: coverage
cov = json.load(open("coverage.json"))
rows = list(csv.DictReader(open("stalberga_parsed.csv")))
ext = defaultdict(dict)
for r in rows:
    v = r["value"].strip()
    ext[r["compound"]][r["target"]] = None if v == "" or r["relation"] == ">" \
        else 9.0 - math.log10(float(v))
usable = {c: d for c, d in ext.items() if all(v is not None for v in d.values())}

tr = list(csv.DictReader(open("mat_profiles.csv")))
TRAIN = {t: np.array([float(r[c]) for r in tr if r[c].strip() != "" and r[cc] == "0"])
         for t, c, cc in (("DAT", "pDAT", "cens_DAT"), ("NET", "pNET", "cens_NET"),
                          ("SERT", "pSERT", "cens_SERT"))}

fig, axes = plt.subplots(3, 1, figsize=(7.4, 6.2), sharex=True)
TOP = 0.205          # headroom so annotation never sits over a bar (max bar ~0.09)
for ax, t in zip(axes, ("DAT", "NET", "SERT")):
    train = TRAIN[t]
    e = np.array([d[t] for d in usable.values()])
    p75 = float(np.percentile(train, 75))
    bins = np.arange(3.0, 11.01, 0.25)
    ax.hist(train, bins=bins, color=LGREY, edgecolor="white", linewidth=0.4,
            weights=np.full(train.size, 1 / train.size), zorder=1)
    ax.axvline(p75, color=NAVY, ls="--", lw=1.4, zorder=3, ymax=0.52)
    n_above = int((e > p75).sum())
    colour = TEAL if n_above else VERM
    ax.plot(e, np.full(e.size, -0.016), "|", ms=11, mew=1.7, color=VERM, zorder=4,
            clip_on=False)
    ax.set_ylim(0, TOP)
    ax.set_yticks([])
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color("#AAAAAA")
    ax.tick_params(length=3, color="#AAAAAA")
    ax.text(0.010, 0.97, t, transform=ax.transAxes, fontsize=13.5, fontweight="bold",
            color=NAVY, va="top")
    msg = (f"{n_above} of {e.size} external compounds above training p75"
           if n_above else f"NONE of {e.size} external compounds above training p75")
    ax.text(0.010, 0.76, msg, transform=ax.transAxes, fontsize=10, color=colour,
            va="top", fontweight="bold")
    ax.text(0.010, 0.615,
            f"external/training variance {cov['targets'][t]['variance_ratio_ext_over_train']:.2f}"
            f"   ·   median external compound at the "
            f"{cov['targets'][t]['ext_median_train_percentile']:.0f}th training percentile",
            transform=ax.transAxes, fontsize=8.8, color=GREY, va="top")

axes[-1].set_xlim(3.6, 10.4)
axes[-1].tick_params(axis="x", pad=11)          # clear the rug on the bottom panel
axes[-1].set_xlabel("potency  (p$K_\\mathrm{i}$ / pIC$_{50}$)", fontsize=11, labelpad=6)
h = [plt.Rectangle((0, 0), 1, 1, fc=LGREY, ec="white", label="training set, uncensored"),
     plt.Line2D([], [], marker="|", ls="", ms=10, mew=1.8, color=VERM,
                label="pre-registered external set (Stalberga et al. 2026, $n=46$)"),
     plt.Line2D([], [], ls="--", lw=1.4, color=NAVY, label="training 75th percentile")]
fig.legend(handles=h, loc="lower center", bbox_to_anchor=(0.5, -0.045), ncol=3,
           frameon=False, fontsize=9, handletextpad=0.6, columnspacing=2.0)
fig.suptitle("The external set occupies the weak tail of the response range at NET and SERT",
             fontsize=12, fontweight="bold", color=NAVY, y=0.995)
fig.tight_layout(rect=(0, 0.012, 1, 0.975), h_pad=0.8)
fig.savefig("fig12_coverage.png", dpi=240, bbox_inches="tight")
fig.savefig("fig12_coverage.pdf", bbox_inches="tight")
print("wrote fig12_coverage.png / .pdf")

# =================================================== fig 13: within-document
wd = list(csv.DictReader(open("within_document_pairs.csv")))
LABS = {
    "CHEMBL1138144": ("Meltzer 2006", "pyrovalerone analogues", "J Med Chem"),
    "CHEMBL1135775": ("Houlihan 2002", "mazindol analogues", "J Med Chem"),
    "CHEMBL1146223": ("Gong 2007", "3$\\beta$-aryltropanes", "J Med Chem"),
    "CHEMBL1132179": ("Choi 1999", "non-piperazine GBR", "J Med Chem"),
    "CHEMBL1136239": ("Axford 2003", "bicyclo[2.2.1]heptanes", "BMCL"),
}
bydoc = defaultdict(list)
for w in wd:
    bydoc[w["document_chembl_id"]].append(float(w["delta"]))
order = sorted(bydoc, key=lambda d: np.median(bydoc[d]))

fig, ax = plt.subplots(figsize=(7.2, 3.9))
percomp = defaultdict(list)
for w in wd:
    percomp[w["molecule_chembl_id"]].append(float(w["delta"]))
pooled = float(np.median([np.median(v) for v in percomp.values()]))
ax.axvline(0, color="#444444", lw=1.2, zorder=2)
ax.axvline(pooled, color=VERM, lw=1.6, ls="--", zorder=2)

rng = np.random.default_rng(7)
for i, d in enumerate(order):
    v = np.array(bydoc[d])
    jit = rng.uniform(-0.17, 0.17, v.size)
    ax.plot(v, np.full(v.size, i) + jit, "o", ms=4.6, color=BLUE, alpha=0.5,
            markeredgecolor="none", zorder=3)
    m = float(np.median(v))
    ax.plot([m], [i], "D", ms=8.5, color=NAVY, markeredgecolor="white", mew=1.3, zorder=5)
    ax.text(m, i + 0.30, f"{m:+.2f}", ha="center", va="bottom", fontsize=9,
            fontweight="bold", color=NAVY)
    ax.text(-1.22, i, f"n={v.size}", ha="left", va="center", fontsize=8.5, color=GREY)

ax.set_yticks(range(len(order)))
ax.set_yticklabels([f"{LABS[d][0]}\n{LABS[d][1]}" for d in order], fontsize=9)
ax.set_xlim(-1.25, 2.1)
ax.set_ylim(-0.7, len(order) - 0.25)
ax.set_xlabel("functional pIC$_{50}$ $-$ binding p$K_\\mathrm{i}$  at NET  (log units)",
              fontsize=10.5, labelpad=6)
ax.xaxis.grid(True, color="#EEEEEE", lw=0.9)
ax.set_axisbelow(True)
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
ax.spines["bottom"].set_color("#AAAAAA")
ax.tick_params(length=0)
ax.text(pooled + 0.04, len(order) - 0.42,
        f"pooled median {pooled:+.2f}", fontsize=9, color=VERM, fontweight="bold")
ax.set_title("Both modalities from the same paper: the NET offset is positive in all five\n"
             "contributing laboratories, 1999$-$2007",
             fontsize=11, fontweight="bold", color=NAVY, loc="left", pad=10)
fig.tight_layout(pad=0.5)
fig.savefig("fig13_within_document.png", dpi=240, bbox_inches="tight")
fig.savefig("fig13_within_document.pdf", bbox_inches="tight")
print("wrote fig13_within_document.png / .pdf")
