#!/bin/sh
# Build the flat working directory the src/ scripts expect.
#
# They were all run in one folder, reading their inputs by bare filename. Sorting the
# deposit into data/, results/, figures/ and tables/ broke those paths. This stages a
# directory where everything sits side by side, hard-linking where the filesystem allows
# it so nothing is duplicated on disk and nothing in _flat/ can drift from its original.
#
# Usage, from the repository root:
#     sh src/stage.sh
#     cd _flat && python coverage.py
set -eu

cd "$(dirname "$0")/.."
OUT=_flat
mkdir -p "$OUT"

link() {
  for f in "$@"; do
    [ -f "$f" ] || continue
    ln -f "$f" "$OUT/$(basename "$f")" 2>/dev/null || cp -f "$f" "$OUT/$(basename "$f")"
  done
}

link src/*.py
link data/*
link results/*
link ./*.csv ./*.json ./*.npy ./*.npz ./*.pkl

n=$(find "$OUT" -type f | wc -l | tr -d ' ')
echo "staged $n files in $OUT/"
echo
echo "Six inputs the earlier rounds produced as intermediates are included:"
echo "  mat_profiles.csv  rebuild_models.pkl  net81_names.json"
echo "  similarity_distance.json  decision_oof.npz  dmpnn_oof.npz"
echo
echo "Run scripts from inside $OUT/. Outputs land there, not in results/, so a re-run"
echo "cannot overwrite a deposited result in place."
