"""Classify assay MODALITY (binding vs functional uptake) from the free-text
assay description, independently of standard_type.

This is the whole point of the 2x2. Every earlier analysis in this project used
standard_type as a proxy for modality -- Ki meaning binding, IC50 meaning uptake --
which is exactly the confound Lee et al. (2024) warn about. Here modality comes only
from what the assay says it measured, and value type stays a separate axis.

Rules are ordered and deliberately high-precision. Anything that cannot be decided
goes to 'unclear' and is dropped rather than guessed; a 'conflict' bucket holds the
descriptions that assert both, so their effect can be tested by moving them either way.
"""
import re
import pandas as pd

SRC = "data/chembl_rederivation_activities.csv"

# "uptake site" / "reuptake site" names the TARGET, not the measurement -- neutralise it
SITE = re.compile(r"(re-?\s?uptake)\s+(site|carrier|recognition)", re.I)

# the measurement really is transport
UPTAKE = [
    re.compile(r"assessed as .{0,60}?(re-?)?uptake", re.I),
    re.compile(r"(inhibit\w*|reduc\w*|decreas\w*|block\w*|attenuat\w*)\s+(of|in)?\s*"
               r"[^.]{0,60}?(re-?)?uptake", re.I),
    re.compile(r"\[[^\]]{1,20}\][\s-]*\w[\w\s,\-]{0,40}?(re-?)?uptake", re.I),
    re.compile(r"(re-?)?uptake\s+(assay|inhibition|of|was|in\b)", re.I),
    re.compile(r"\bfunctional\b.{0,30}(re-?)?uptake", re.I),
    re.compile(r"\basp\+", re.I),
    re.compile(r"transporter[- ]mediated\s+\w+\s+(re-?)?uptake", re.I),
]

# the measurement really is occupancy
BINDING = [
    re.compile(r"\bdisplac\w*", re.I),
    re.compile(r"radioligand", re.I),
    re.compile(r"\bbinding\b", re.I),
    re.compile(r"inhibition of \[[^\]]{1,20}\][^.]{0,40}?\bfrom\b", re.I),
]


def classify(desc):
    if not isinstance(desc, str) or not desc.strip():
        return "unclear"
    txt = SITE.sub(r"\2", desc)                   # "5-HT uptake site" -> "5-HT site"
    u = any(p.search(txt) for p in UPTAKE)
    b = any(p.search(txt) for p in BINDING)
    if u and b:
        return "conflict"
    if u:
        return "uptake"
    if b:
        return "binding"
    return "unclear"


if __name__ == "__main__":
    d = pd.read_csv(SRC, low_memory=False)
    d["modality"] = d.assay_description.map(classify)
    d.to_csv("classified.csv", index=False)
    print(d.modality.value_counts(), "\n")
    print(pd.crosstab(d.modality, d.standard_type), "\n")
    for cls in ("binding", "uptake", "conflict", "unclear"):
        print("=" * 100)
        print(cls.upper(), "— 12 distinct descriptions")
        for s in d.assay_description[d.modality == cls].drop_duplicates().sample(
                12, random_state=0):
            print("   ", s[:150])
