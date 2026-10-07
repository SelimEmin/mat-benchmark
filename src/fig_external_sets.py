"""Response-range coverage of three external sets against the training distribution.

Original layout, with the two overlap problems fixed:
  * ticks within a row no longer overprint -- a small swarm pass stacks
    near-coincident compounds across up to three sub-levels
  * the bottom panel's tick labels are padded clear of the lowest rug row

Palette: Okabe-Ito categorical slots, validated with the dataviz validator
(all checks PASS; worst adjacent CVD dE 11.0 deutan, 25.8 normal vision).
"""
import csv, math
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")
VERM, BLUE, TEAL, NAVY = "#D55E00", "#0072B2", "#009E73", "#1F4E79"
INK, MUTED, GREY = "#1F2328", "#6B7280", "#DDDDDD"
plt.rcParams.update({"font.family": "sans-serif",
                     "font.sans-serif": ["Liberation Sans", "DejaVu Sans"],
                     "font.size": 10})
TARGETS = ("DAT", "NET", "SERT")


def parent(s):
    m = Chem.MolFromSmiles(s)
    if m is None:
        return None
    fr = Chem.GetMolFrags(m, asMols=True, sanitizeFrags=False)
    return max(fr, key=lambda x: x.GetNumHeavyAtoms()) if fr else m


def swarm_levels(x, width, max_levels=3):
    """Level index per point so that points closer than `width` do not overlap."""
    lv = np.zeros(x.size, dtype=int)
    placed = []
    for i in np.argsort(x):
        used = {l for xx, l in placed if abs(xx - x[i]) < width}
        l = 0
        while l in used and l < max_levels - 1:
            l += 1
        placed.append((x[i], l))
        lv[i] = l
    return lv


# --------------------------------------------------------------------- data
tr = list(csv.DictReader(open("mat_profiles.csv")))
TRAIN = {t: np.array([float(r[c]) for r in tr if r[c].strip() and r[cc] == "0"])
         for t, c, cc in (("DAT", "pDAT", "cens_DAT"), ("NET", "pNET", "cens_NET"),
                          ("SERT", "pSERT", "cens_SERT"))}
train_keys = {Chem.MolToInchiKey(parent(r["smiles"])) for r in tr
              if parent(r["smiles"]) is not None}

st_smi = {r["abbrev"]: r["final"] for r in csv.DictReader(open("struct.csv"))}
vals = defaultdict(dict)
for r in csv.DictReader(open("stalberga_parsed.csv")):
    vals[r["compound"]][r["target"]] = (r["relation"], r["value"].strip())
SET1 = []
for c, d in vals.items():
    if len(d) == 3 and all(v[0] == "=" and v[1] for v in d.values()):
        m = parent(st_smi[c])
        if m is not None and Chem.MolToInchiKey(m) not in train_keys:
            SET1.append({t: 9.0 - math.log10(float(d[t][1])) for t in TARGETS})

SET2 = []
for r in csv.DictReader(open("pdsp_mat_profiles.csv")):
    m = parent(r["smiles"])
    if m is not None and Chem.MolToInchiKey(m) not in train_keys:
        SET2.append({t: float(r["p" + t]) for t in TARGETS})

SET3 = [{t: (float(r["p" + t]) if r["p" + t] != "" else None) for t in TARGETS}
        for r in csv.DictReader(open("ejmc2026_cui.csv"))]

SETS = [("Stalberga 2026 — forensic stimulants, uptake IC$_{50}$", SET1, VERM),
        ("PDSP / BindingDB — antidepressant class, binding $K_\\mathrm{i}$", SET2, BLUE),
        ("Cui et al. 2026 — SNRI series, uptake IC$_{50}$", SET3, TEAL)]

# ------------------------------------------------------------------- layout
TOP = 0.235            # panel data range
ROW0, ROWDY = -0.020, 0.030      # first rug row, spacing between rows
LVDY = 0.0062                    # vertical step between stacked sub-levels

fig, axes = plt.subplots(3, 1, figsize=(7.8, 8.8), sharex=True)
for ax, t in zip(axes, TARGETS):
    train = TRAIN[t]
    p75 = float(np.percentile(train, 75))
    bins = np.arange(3.0, 11.01, 0.25)
    ax.hist(train, bins=bins, color=GREY, edgecolor="white", linewidth=0.4,
            weights=np.full(train.size, 1 / train.size), zorder=1)
    ax.axvline(p75, color=NAVY, ls="--", lw=1.4, zorder=3, ymax=0.46)

    for k, (lab, rows, col) in enumerate(SETS):
        e = np.array([r[t] for r in rows if r.get(t) is not None])
        lv = swarm_levels(e, width=0.14)
        y = ROW0 - ROWDY * k - lv * LVDY
        ax.plot(e, y, "|", ms=8, mew=1.7, color=col, zorder=4, clip_on=False,
                linestyle="none")

    ax.set_ylim(0, TOP)
    ax.set_yticks([])
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color("#AAAAAA")
    ax.tick_params(length=3, color="#AAAAAA")

    ax.text(0.010, 0.97, t, transform=ax.transAxes, fontsize=13.5, fontweight="bold",
            color=NAVY, va="top")
    for i, (lab, rows, col) in enumerate(SETS):
        e = np.array([r[t] for r in rows if r.get(t) is not None])
        n_hi = int((e > p75).sum())
        vr = e.var(ddof=1) / train.var(ddof=1)
        txt = (f"{n_hi if n_hi else 'NONE'} of {e.size} above training p75   ·   "
               f"variance {vr:.2f}× training   ·   max {e.max():.2f}")
        ax.text(0.010, 0.80 - 0.135 * i, txt, transform=ax.transAxes, fontsize=9.1,
                color=col, va="top", fontweight="bold")

axes[-1].set_xlim(3.6, 10.6)
axes[-1].tick_params(axis="x", pad=46)       # clear of the lowest rug row
axes[-1].set_xlabel("potency   (p$K_\\mathrm{i}$ / pIC$_{50}$)", fontsize=11, labelpad=6)

h = [plt.Rectangle((0, 0), 1, 1, fc=GREY, ec="white", label="training set, uncensored")]
h += [Line2D([], [], marker="|", ls="", ms=10, mew=1.8, color=c, label=lab)
      for lab, r, c in SETS]
h += [Line2D([], [], ls="--", lw=1.4, color=NAVY, label="training 75th percentile")]
fig.legend(handles=h, loc="lower center", bbox_to_anchor=(0.5, -0.062), ncol=2,
           frameon=False, fontsize=9, handletextpad=0.6, columnspacing=2.2,
           labelspacing=0.55)
fig.suptitle("Two new external sets reach the part of the range the original cannot",
             fontsize=12, fontweight="bold", color=NAVY, y=0.995)
fig.tight_layout(rect=(0, 0.02, 1, 0.975), h_pad=1.4)
fig.savefig("fig_external_sets.png", dpi=250, bbox_inches="tight")
fig.savefig("fig_external_sets.pdf", bbox_inches="tight")
print("wrote fig_external_sets.png / .pdf")
