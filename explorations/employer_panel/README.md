# Employer panel exploration

Marimo notebooks behind `markdowns/employer_design_agent_plan.md` and
`markdowns/employer_panel_step1.md`. They are not pipeline stages. They read
retained artifacts and the raw H-2A workbooks (one per fiscal year; the
latest release when a year has two), and write only to the scratch
directory named by `EMPLOYER_PANEL_SCRATCH`, which must be outside the
repository. Nothing here changes the v2 linkage, a producer or a retained
artifact.

Run from the repository root, in order:

```sh
export UV_CACHE_DIR=/tmp/h2a-uv-cache
export EMPLOYER_PANEL_SCRATCH=/path/outside/repo/employer_panel
mkdir -p "$EMPLOYER_PANEL_SCRATCH"
for nb in 01_filer_records 02_rule_harness 03_exit_diagnosis 04_usage_split; do
  uv run --no-sync marimo export script "explorations/employer_panel/$nb.py" \
    --output "$EMPLOYER_PANEL_SCRATCH/$nb.flat.py" --force
  uv run --no-sync python "$EMPLOYER_PANEL_SCRATCH/$nb.flat.py"
done
```

| Notebook | What it does | Scratch outputs |
| --- | --- | --- |
| `01_filer_records.py` | One row per filer case (the employer named on the case; every case-level row that is not a pre-2020 sub-employer), with four name keys, the mailing-address key and the agent or attorney name read from the raw workbooks. The same name keys for worksite-business rows. Non-missing shares of identity fields by raw file | `filer_cases.parquet`, `business_rows.parquet`, `raw_field_inventory.csv` |
| `02_rule_harness.py` | Scores linkage rules that read only the name, mailing address and agent against the FEIN (FY2024 on) and the phone (FY2015 on); nested rule sets R0–R9 closed over all years; year-to-year return rates by identifier | `harness_single_rules.csv`, `harness_rule_sets.csv`, `harness_return_errors.csv`, `harness_return_rates.csv`, `harness_fuzzy_bins.csv`, `filer_ids.parquet`, `filer_units.parquet` |
| `03_exit_diagnosis.py` | Splits v2 worksite businesses present in FY2024 and absent in FY2025 by whether their filer (FEIN) filed again and whether it named the same or a similar business | `exit_diagnosis.csv`, `exit_diagnosis_pairs.csv` |
| `04_usage_split.py` | Splits each year's change in certified workers into continuing filers, entry and exit under the FEIN and rule sets R0, R5, R7 and R8; worker-weighted missed returns; the ten largest exiting filers by year | `usage_split.csv`, `usage_missed_return.csv`, `usage_largest_exits.csv` |

Name keys, each built from the v2 `normalize_name`:

- `k_exact`: the v2 normalized name.
- `k_legal`: initials joined ("L L C" becomes "LLC") and legal-form words
  dropped.
- `k_core`: also drops parentheticals, anything after "DBA", numbers, site
  words and "AND", and makes common farm plurals singular.
- `k_sorted`: the words of `k_core`, sorted.
- `k_distinct` (harness only): `k_core` without generic farm words.

`filer_ids.parquet` holds one identifier per rule set (`filer_R0` to
`filer_R9`) for every filer case. These are scratch identifiers for scoring,
not a published linkage.

Ruff reports only `PLR1711` (the bare `return` that closes a Marimo cell) on
these files.
