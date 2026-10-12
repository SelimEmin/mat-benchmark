# `src/` — the earlier-round analysis scripts

These are the scripts behind the first and second rounds of the analysis. They are kept
**unmodified** apart from one mechanical change: paths that pointed at the machine they
were written on now read `data/…`. The manuscript's Declarations enumerate the errors
found during development and during revision, and that audit trail is only worth
something if the code behind it has not been quietly tidied afterwards.

## They need a flat working directory

Most scripts here read their inputs by bare filename from the current directory, because
they were all run in one flat folder; a few read `data/<file>` with the prefix. Sorting the deposit into `data/`, `results/`,
`figures/` and `tables/` made the repository legible and broke those paths. Rather than
rewrite twenty-odd scripts, stage the directory they expect:

```
sh src/stage.sh          # builds _flat/ (and _flat/data/) from data/, results/ and the root artefacts
cd _flat && python coverage.py
```

`stage.sh` hard-links where it can, so `_flat/` costs almost no disk and nothing in it is
a second copy that can drift from the original. `_flat/` is in `.gitignore`.

## Three scripts reuse a sibling's header

`internal_leak_check.py`, `modality_subsample.py` and `range_swap_ctrl.py` pull in the
featuriser, the fold splitter and the model wrapper by `exec`-ing the top of a sibling
script instead of importing it:

```python
exec(open("production_model.py").read().split("lab = pd.read_csv")[0].split('"""', 2)[2])
```

It works, and it is the reason `ruff` reports `F821 undefined-name` for every injected
name in those three files — static analysis does not follow `exec`. Those three are the
only `F821` waivers in `ruff.toml`, listed by filename so the exception cannot quietly
widen. If you are writing anything new, import; do not copy this.

## One script's input is not deposited

`pdsp_extract.py` builds the PDSP/BindingDB slice from the full BindingDB TSV download,
which is about 300 MB and is not deposited. Its outputs, `data/pdsp_mat_raw.csv` and
`data/pdsp_mat_profiles.csv`, are.

## What this directory is not

It is not the reproduction path. Two things reproduce the published numbers end to end,
both documented in the top-level `README.md`:

- `repro/reproduce.py` — the two external results, to four decimal places
- the scripts at the repository root — the radioligand stratification, the modality
  labelling, the censored-objective sensitivity table and the external-set recomputation

`external_final.py` at the root supersedes `src/competency2.py`, and the superseded
full-InChIKey overlap filter is kept as `reachability_fullkey_superseded.py` so the
correction can be audited. Where a root script and a `src/` script disagree, the root
script is the one the manuscript reports.
