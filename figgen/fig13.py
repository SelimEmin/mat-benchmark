"""Figure 13: response-range coverage of the three external sets.

Built from the rule the Methods now state -- overlap removed on the InChIKey skeleton
block -- against the uncensored training distribution, and the caption is written from
what the figure computes rather than from an earlier build.
"""
import csv, math, pickle
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")

BLUE, VERM, TEAL, GREY = "#0072B2", "#D55E00", "#009E73", "#666666"
plt.rcParams.update({"font.family": "sans-serif",
                     "font.sans-serif": ["Liberation Sans", "DejaVu Sans"],
                     "font.size": 9, "axes.linewidth": 0.8})
E = ""; T = ("DAT", "NET", "SERT")
COLS = {"DAT": BLUE, "NET": VERM, "SERT": TEAL}
fr = pickle.load(open(E + "faithful_models.pkl", "rb"))["frame"]
TRs = {k.split("-")[0] for k in fr.inchikey}


def mol(s):
    """Parse, stripping any CXSMILES annotation block, then desalt to largest fragment.

    Two PDSP rows carry a trailing " |...|" extension; one of them (maprotiline) is
    rejected by the default parser, which would silently drop a genuine external
    compound. Stripping the annotation recovers it.
    """
    m = Chem.MolFromSmiles(s)
    if m is None and " |" in s:
        m = Chem.MolFromSmiles(s.split(" |")[0])
    if m is None:
        return None
    fr = Chem.GetMolFrags(m, asMols=True, sanitizeFrags=False)
    return max(fr, key=lambda x: x.GetNumHeavyAtoms()) if len(fr) > 1 else m


def skel(s):
    m = mol(s)
    return Chem.MolToInchiKey(m).split("-")[0] if m else None


st = {r["abbrev"]: r["final"] for r in csv.DictReader(open(E + "struct.csv"))}
st["Cocaine"] = "COC(=O)[C@H]1[C@@H](OC(=O)c2ccccc2)C[C@@H]2CC[C@H]1N2C"
v = defaultdict(dict)
for r in csv.DictReader(open(E + "stalberga_parsed.csv")):
    v[r["compound"]][r["target"]] = (r["relation"], r["value"].strip())
stal = [(c, d) for c, d in v.items()
        if len(d) == 3 and all(x[0] == "=" and x[1] for x in d.values())
        and skel(st[c]) not in TRs]
STAL = {t: np.array([9.0 - math.log10(float(d[t][1])) for _, d in stal]) for t in T}

pd_ = [r for r in csv.DictReader(open(E + "pdsp_mat_profiles.csv"))
       if skel(r["smiles"]) not in TRs]
PDSP = {t: np.array([float(r["p" + t]) for r in pd_]) for t in T}
cu = [r for r in csv.DictReader(open(E + "ejmc2026_cui.csv"))
      if skel(r["smiles"]) not in TRs]
CUI = {t: np.array([float(r["p" + t]) for r in cu if r["p" + t] != ""]) for t in T}
SETS = [("Stålberga 2026", STAL, VERM), ("PDSP / BindingDB", PDSP, BLUE),
        ("Cui et al. 2026", CUI, TEAL)]
print(f"after skeleton-block exclusion: Stalberga {len(stal)}, PDSP {len(pd_)}, Cui {len(cu)}")

fig, axes = plt.subplots(3, 1, figsize=(7.0, 5.0), sharex=True,
                         gridspec_kw=dict(hspace=.34))
counts = {}
for ax, t in zip(axes, T):
    tr = fr["p" + t].values[fr["cens_" + t].values == 0]
    q75 = float(np.percentile(tr, 75))
    ax.hist(tr, bins=46, density=True, color="#DDE3E9", zorder=1)
    ax.axvline(q75, ls="--", color="#222222", lw=1.0, zorder=4)
    for k, (nm, D, c) in enumerate(SETS):
        yy = -0.075 - 0.085 * k
        ax.plot(D[t], np.full(D[t].size, yy), "|", color=c, ms=8, mew=1.2, zorder=3)
        n = int((D[t] > q75).sum())
        counts[(nm, t)] = n
        ax.text(10.62, yy, f"{n if n else 'none'}", fontsize=7.4, color=c,
                va="center", ha="right", fontweight="bold")
    ax.text(10.62, -0.02, "above p75", fontsize=6.6, color=GREY, ha="right", va="center")
    ax.set_ylabel(t, fontsize=9.5, fontweight="bold", color=COLS[t], rotation=0,
                  ha="right", va="center", labelpad=12)
    ax.set_ylim(-0.33, None); ax.set_yticks([]); ax.set_xlim(3.4, 10.7)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors="#333333", labelsize=8)
axes[-1].set_xlabel("potency (p$K_i$ / pIC$_{50}$)", fontsize=8.6)
h = [plt.Line2D([], [], color=c, marker="|", ls="none", ms=8, mew=1.4, label=nm)
     for nm, _, c in SETS]
h += [plt.Line2D([], [], color="#DDE3E9", lw=6, label="training (uncensored)"),
      plt.Line2D([], [], color="#222222", ls="--", lw=1.0, label="training 75th pct")]
axes[0].legend(handles=h, frameon=False, fontsize=7.2, ncol=3, loc="lower center",
               bbox_to_anchor=(0.5, 1.03), columnspacing=1.4, handletextpad=.5)
fig.savefig("tex/figures/fig13_newsets.pdf", bbox_inches="tight")
print("\ncounts above the training 75th percentile:")
for nm, _, _ in SETS:
    print(f"   {nm:<20}" + "  ".join(f"{t} {counts[(nm,t)]}" for t in T))
