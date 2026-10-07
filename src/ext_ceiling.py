"""The R-squared ceiling ON the pre-registered external set.

R-squared is normalised by the variance of the set it is computed on. The external
set's variance is much smaller than the training set's, so the same measurement
noise eats a much larger fraction of it. This computes what a perfect model could
have scored on this external set.
"""
import json

cov = json.load(open("coverage.json"))

# noise definitions, as derived in CEILING_DERIVATION.md
SIG = {
    "aggregation only (0.127)": {"DAT": 0.127, "NET": 0.127, "SERT": 0.127},
    "single measurement (0.212)": {"DAT": 0.212, "NET": 0.212, "SERT": 0.212},
    "modality pooled": {"DAT": 0.445, "NET": 0.655, "SERT": 0.492},
}
ACHIEVED_EXT = {"DAT": 0.459, "NET": 0.015, "SERT": 0.058}   # Table 2, n=42
ACHIEVED_INT = {"DAT": 0.584, "NET": 0.501, "SERT": 0.541}
CEIL_INT = {"DAT": 0.820, "NET": 0.633, "SERT": 0.838}       # modality pooled, training variance

print("=" * 86)
print("CEILING ON THE EXTERNAL SET  (R^2 is normalised by the test set's own variance)")
print("=" * 86)
rows = []
for t in ("DAT", "NET", "SERT"):
    vt = cov["targets"][t]["train_sd"] ** 2
    ve = cov["targets"][t]["ext_sd"] ** 2
    line = {"target": t, "var_train": vt, "var_ext": ve,
            "ceil_internal": CEIL_INT[t], "achieved_ext": ACHIEVED_EXT[t]}
    print(f"\n{t}:  training variance {vt:.3f}   external variance {ve:.3f}"
          f"   (ratio {ve/vt:.2f})")
    for name, s in SIG.items():
        c_tr = 1 - s[t] ** 2 / vt
        c_ex = 1 - s[t] ** 2 / ve
        print(f"    noise {name:<28} ceiling on training {c_tr:6.3f}"
              f"   ceiling on external {c_ex:7.3f}")
        line[f"ceil_ext::{name}"] = c_ex
    ce = line["ceil_ext::modality pooled"]
    frac = ACHIEVED_EXT[t] / ce if ce > 0 else float("nan")
    line["frac_of_ext_ceiling"] = frac
    print(f"    -> model scored {ACHIEVED_EXT[t]:.3f} externally = "
          f"{100*frac:.0f}% of the {ce:.3f} that was reachable")
    rows.append(line)

print("\n" + "=" * 86)
print("SUMMARY TABLE")
print("=" * 86)
print(f"{'':6} {'int R2':>7} {'int ceil':>9} {'% int':>6}   "
      f"{'ext R2':>7} {'ext ceil':>9} {'% ext':>6}")
for r in rows:
    t = r["target"]
    ce = r["ceil_ext::modality pooled"]
    print(f"{t:6} {ACHIEVED_INT[t]:7.3f} {CEIL_INT[t]:9.3f} "
          f"{100*ACHIEVED_INT[t]/CEIL_INT[t]:5.0f}%   "
          f"{ACHIEVED_EXT[t]:7.3f} {ce:9.3f} {100*r['frac_of_ext_ceiling']:5.0f}%")

json.dump(rows, open("ext_ceiling.json", "w"), indent=2)
print("\nwrote ext_ceiling.json")
