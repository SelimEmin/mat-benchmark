"""Independent check of substrate_smarts.py against a hand-built reference set.

Releaser/blocker status here is standard monoamine-transporter pharmacology
(Rothman & Baumann release assays; Eshleman et al.; Simmler et al.). Compounds whose
status is genuinely disputed are left out rather than guessed.
"""
from substrate_smarts import classify

RELEASER = {
    "amphetamine": "CC(N)Cc1ccccc1",
    "methamphetamine": "CNC(C)Cc1ccccc1",
    "MDMA": "CNC(C)Cc1ccc2OCOc2c1",
    "MDA": "CC(N)Cc1ccc2OCOc2c1",
    "cathinone": "CC(N)C(=O)c1ccccc1",
    "methcathinone": "CNC(C)C(=O)c1ccccc1",
    "mephedrone": "CNC(C)C(=O)c1ccc(C)cc1",
    "methylone": "CNC(C)C(=O)c1ccc2OCOc2c1",
    "ethylone": "CCNC(C)C(=O)c1ccc2OCOc2c1",
    "butylone": "CCC(NC)C(=O)c1ccc2OCOc2c1",
    "4-fluoromethcathinone": "CNC(C)C(=O)c1ccc(F)cc1",
    "pentedrone": "CCCC(NC)C(=O)c1ccccc1",
    "phentermine": "CC(C)(N)Cc1ccccc1",
    "4-methylamphetamine": "CC(N)Cc1ccc(C)cc1",
    "para-chloroamphetamine": "CC(N)Cc1ccc(Cl)cc1",
    "fenfluramine": "CCNC(C)Cc1cccc(C(F)(F)F)c1",
    "norfenfluramine": "CC(N)Cc1cccc(C(F)(F)F)c1",
    "tyramine": "NCCc1ccc(O)cc1",
    "dopamine": "NCCc1ccc(O)c(O)c1",
    "serotonin": "NCCc1c[nH]c2ccc(O)cc12",
    "phenmetrazine": "CC1NCCOC1c1ccccc1",
    "aminorex": "NC1=NC(c2ccccc2)CO1",
    "4-methylaminorex": "CC1OC(N)=NC1c1ccccc1",
    "2-aminoindane": "NC1Cc2ccccc2C1",
    "MDAI": "NC1Cc2cc3OCOc3cc2C1",
    "5-APB": "CC(N)Cc1ccc2occc2c1",
    "cathine": "CC(N)C(O)c1ccccc1",
    "PAL-287": "CC(N)Cc1ccc2ccccc2c1",
}
BLOCKER = {
    "cocaine": "COC(=O)C1C(OC(=O)c2ccccc2)CC2CCC1N2C",
    "methylphenidate": "COC(=O)C(c1ccccc1)C1CCCCN1",
    "GBR12909": "Fc1ccc(cc1)C(OCCN1CCN(CCCc2ccccc2)CC1)c1ccc(F)cc1",
    "mazindol": "OC1(c2ccc(Cl)cc2)CC2=NCCN2c2ccccc21",
    "bupropion": "CC(NC(C)(C)C)C(=O)c1cccc(Cl)c1",
    "nisoxetine": "CNCCC(Oc1ccccc1OC)c1ccccc1",
    "atomoxetine": "CNCCC(Oc1ccccc1C)c1ccccc1",
    "reboxetine": "CCOc1ccccc1OC(c1ccccc1)C1OCCNC1",
    "desipramine": "CNCCCN1c2ccccc2CCc2ccccc21",
    "imipramine": "CN(C)CCCN1c2ccccc2CCc2ccccc21",
    "nortriptyline": "CNCCC=C1c2ccccc2CCc2ccccc21",
    "fluoxetine": "CNCCC(Oc1ccc(cc1)C(F)(F)F)c1ccccc1",
    "paroxetine": "Fc1ccc(cc1)C1CCNCC1COc1ccc2OCOc2c1",
    "citalopram": "CN(C)CCCC1(OCc2cc(ccc12)C#N)c1ccc(F)cc1",
    "sertraline": "CNC1CCC(c2ccccc2)c2cc(Cl)c(Cl)cc12",
    "venlafaxine": "COc1ccc(cc1)C(CN(C)C)C1(O)CCCCC1",
    "duloxetine": "CNCCC(Oc1cccc2ccccc12)c1cccs1",
    "MDPV": "CCCC(C(=O)c1ccc2OCOc2c1)N1CCCC1",
    "alpha-PVP": "CCCC(C(=O)c1ccccc1)N1CCCC1",
    "naphyrone": "CCCC(C(=O)c1ccc2ccccc2c1)N1CCCC1",
    "pyrovalerone": "CCCC(C(=O)c1ccc(C)cc1)N1CCCC1",
    "benztropine": "CN1C2CCC1CC(C2)OC(c1ccccc1)c1ccccc1",
    "WIN35428": "COC(=O)C1C(c2ccc(F)cc2)CC2CCC1N2C",
    "indatraline": "CNC1CC(c2ccc(Cl)cc2Cl)c2ccccc21",
    "modafinil": "NC(=O)CS(=O)C(c1ccccc1)c1ccccc1",
    "amitriptyline": "CN(C)CCC=C1c2ccccc2CCc2ccccc21",
    "clomipramine": "CN(C)CCCN1c2ccccc2CCc2ccc(Cl)cc21",
}

tp = fn = tn = fp = 0
errs = []
for nm, smi in RELEASER.items():
    v, why = classify(smi)
    if v is True: tp += 1
    else:
        fn += 1; errs.append(("MISSED RELEASER", nm, why))
for nm, smi in BLOCKER.items():
    v, why = classify(smi)
    if v is False: tn += 1
    else:
        fp += 1; errs.append(("BLOCKER CALLED RELEASER", nm, why))

n = tp + fn + tn + fp
print(f"reference set: {len(RELEASER)} releasers, {len(BLOCKER)} blockers")
print(f"  recall (releasers found)   {tp}/{tp+fn} = {tp/(tp+fn)*100:.0f}%")
print(f"  specificity (blockers kept) {tn}/{tn+fp} = {tn/(tn+fp)*100:.0f}%")
print(f"  balanced accuracy           {(tp/(tp+fn)+tn/(tn+fp))/2*100:.0f}%")
if errs:
    print("\n  misclassified:")
    for kind, nm, why in errs:
        print(f"    {kind:<26}{nm:<24}{why}")
