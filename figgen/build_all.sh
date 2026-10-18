#!/bin/sh
# Rebuild every numbered figure, in the one order that is correct.
#
# Order matters and is not arbitrary:
#   prep.py       recomputes figdata.json from the pre-registered model and protocol
#   make_figs.py  draws figures 1-10 from it
#   fix_figs.py   redraws 7, 8 and 10 from the published table values, so that no
#                 figure can disagree with a table; it must run after make_figs.py
#   figs_11_12.py draws 11 and 12
#   fig13.py      draws 13 under the skeleton-block overlap rule
#   toc_graphic.py draws the ACS table-of-contents graphic from the Table 10 values
#
# Run from the repository root (the parent of figgen/), not from inside figgen/.
set -e
cd "$(dirname "$0")/.."
mkdir -p figures/numbered

# figs_11_12.py reads within_doc_3targets.json from results/. An earlier version read it
# from the root, which made this script abort after figure 11; fail loudly if it regresses.
grep -q '"results/within_doc_3targets.json"' figgen/figs_11_12.py

python figgen/prep.py
python figgen/make_figs.py
python figgen/fix_figs.py
python figgen/figs_11_12.py
python figgen/fig13.py
python figgen/toc_graphic.py
echo
echo "figures written to figures/numbered/"
ls -l figures/numbered/*.pdf | awk '{print "  " $9, $5 " bytes"}'
