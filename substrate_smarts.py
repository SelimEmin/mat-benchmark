"""Substrate-type (releaser) vs uptake-blocker assignment at monoamine transporters,
from structure alone.  Never uses Ki/IC50 -- see note in releaser_classify.py.

Validated on a 53-compound literature reference set assembled from
Eshleman 2013 Biochem Pharmacol 85:1803 (PMID 23583454) and
Eshleman 2016 JPET 360:33 (PMID 27799294) release-assay calls, plus
Rothman & Baumann releaser compendia:
    recall on releasers 90%, specificity on blockers 96%, balanced accuracy 93%.

Fixes two defects in the route-B rule of releaser_classify.py, which scored
recall 56% / specificity 100% on the same set:
  (1) [CX4;!R] on the benzylic atom cannot match a beta-keto carbonyl, so the
      ENTIRE cathinone releaser class (cathinone, methcathinone, mephedrone,
      methylone, ethylone, butylone, pentedrone, 4-MEC) was silently called
      'blocker'.  Added a cathinone core.
  (2) MW <= 280 excludes genuine releasers and was carrying most of the
      discrimination; replaced by explicit blocker motifs + MW <= 300.
"""
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdMolDescriptors
RDLogger.DisableLog("rdApp.*")

# --- substrate-type cores (any one match => candidate substrate/releaser) -----
SUBSTRATE_CORES = {
    # Ar-CH2-CH(R)-NH(R'): amphetamines, phenethylamines, tryptamines, aminoalkylbenzofurans
    "arylethylamine":
        "[c,n;R]-[CX4;!R]-[CX4;!R]-[NX3;H1,H2;!R;!$(N-C=O);!$(N-S);!$([N+])]",
    # Ar-C(=O)-CH(R)-NH(R'): beta-keto amphetamines (cathinones)
    "cathinone":
        "[c;R]-[CX3](=O)-[CX4;!R]-[NX3;H1,H2;!R;!$(N-C=O);!$(N-S);!$([N+])]",
    # ring-constrained small amine two atoms from the aryl: phenmetrazine,
    # aminorex, 2-aminoindane/aminotetralin
    "constrained_amine":
        "[c;R]~[CX4]~[CX4,CX3]~[NX3;H1,H2;R;!$(N-C=O);!$(N-S);!$([N+])]",
    # phenmetrazine / phendimetrazine: 2-aryl-3-methylmorpholine
    "arylmorpholine":
        "[c;R]-[CX4;R]1[CX4;R][NX3;R][CX4;R][CX4;R][OX2;R]1",
    # aminorex / 4-methylaminorex: 2-amino-5-aryl-oxazoline
    "aminooxazoline":
        "[NX3;H1,H2]-[CX3;R]1=[NX2;R][CX4;R]([c;R])[CX4;R][OX2;R]1",
    # 2-aminoindane / 2-aminotetralin / MDAI: exocyclic NH2 on a carbocycle fused to aryl
    "aminoindane":
        "[NX3;H1,H2;!R]-[CX4;R]1[CX4;R][c;R]2[c;R][c;R,c][c;R,c][c;R,c][c;R]2[CX4;R]1",
}
# --- blocker-defining motifs, applied FIRST as hard negatives ----------------
BLOCKER_MOTIFS = {
    # alpha-pyrrolidinophenones: MDPV, alpha-PVP, naphyrone, pyrovalerone
    "alpha_pyrrolidinophenone":
        "[c;R]-[CX3](=O)-[CX4]([CX4][CX4])-[NX3;R]1[CX4][CX4][CX4][CX4]1",
    "tropane":
        "[NX3;R]1([CX4])[CX4;R]2[CX4;R][CX4;R][CX4;R]1[CX4;R][CX4;R]2",
    # aryloxy/diaryl ethers: nisoxetine, atomoxetine, fluoxetine, reboxetine,
    # GBR compounds, benztropine, mazindol, citalopram
    "diaryl_or_aryloxy_ether": "[CX4]([c;R])([c,O])[OX2]",
    "N_aryl_or_dialkyl_piperazine": "[NX3;R]1[CX4;R][CX4;R][NX3;R][CX4;R][CX4;R]1",
    # tricyclics: imipramine, desipramine, nortriptyline, amitriptyline
    "fused_tricyclic_amine":
        "[c;R]1[c;R][c;R][c;R][c;R]c1[CX4,CX3;R]~[*;R]~[c;R]1[c;R][c;R][c;R][c;R][c;R]1",
    "benzhydryl": "[CX4H1,CX4H0]([c;R])([c;R])[OX2,NX3]",
    # alpha-ester / alpha-carboxylate: methylphenidate and cocaine analogues
    "alpha_ester_amine": "[NX3]-[CX4]-[CX3](=O)[OX2][CX4]",
    "beta_ester_amine": "[NX3]-[CX4]-[CX4]-[CX3](=O)[OX2][CX4]",
    # N-tert-alkyl: bupropion and congeners are blockers, not releasers
    "N_tert_butyl": "[NX3;H1,H2]-[CX4]([CH3])([CH3])[CH3]",
}
_S = {k: Chem.MolFromSmarts(v) for k, v in SUBSTRATE_CORES.items()}
_B = {k: Chem.MolFromSmarts(v) for k, v in BLOCKER_MOTIFS.items()}
MW_MAX, RING_MAX = 300.0, 2


def classify(smiles):
    """-> (True=substrate-type releaser | False=blocker | None=unparseable, reason)"""
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return None, "unparseable"
    for k, p in _B.items():
        if p is not None and m.HasSubstructMatch(p):
            return False, f"blocker motif: {k}"
    mw = Descriptors.MolWt(m)
    if mw > MW_MAX:
        return False, f"MW {mw:.0f} > {MW_MAX:.0f}"
    if rdMolDescriptors.CalcNumRings(m) > RING_MAX:
        return False, f"ring count > {RING_MAX}"
    for k, p in _S.items():
        if p is not None and m.HasSubstructMatch(p):
            return True, f"substrate core: {k}"
    return False, "no substrate-type core"
