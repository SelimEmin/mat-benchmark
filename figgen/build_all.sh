#!/bin/sh
# Rebuild every manuscript figure, in the one order that is correct.
#
# Order matters and is not arbitrary:
#   prep.py       recomputes figdata.json from the pre-registered model and protocol
#   make_figs.py  draws figures 1-10 from it
#   fix_figs.py   redraws 7, 8 and 10 from the published table values, so that no
#                 figure can disagree with a table; it must run after make_figs.py
#   figs_11_12.py draws 11 and 12
#   fig13.py      draws 13 under the skeleton-block overlap rule
#
# Run from the manuscript directory (the parent of figgen/), not from inside figgen/.
set -e
cd "$(dirname "$0")/.."
python figgen/prep.py
python figgen/make_figs.py
python figgen/fix_figs.py
python figgen/figs_11_12.py
python figgen/fig13.py
echo
echo "figures written to tex/figures/"
ls -l tex/figures/*.pdf | awk '{print "  " $9, $5 " bytes"}'
