# Three chapter directions

The current 2025 wage-offer and earnings analyses are now owned by the
[separate policy design](../../code/designs/policy_2025/README.md), run with
`./scripts/run_policy_2025.sh`. The November 16 migration preserves the
specifications and inspected results. This folder, its frozen declarations
and reports remain the historical three-direction exploration; the 2013
administrative-interruption work remains here.

This isolated exploration owns scratch builds, estimates, checks and the
generated adviser report. Read [design and limitations](../../content/designs/chapter-directions.md),
[registry](registry.json), and [dated measurement amendment](measurement_amendment_2026_11_14.json)
before changing a specification. The original episode registry and its producers
remain frozen as historical records. The wage extensions were developed after
related FY2026 results were known. The user subsequently authorized partial-window
earnings estimates; the [interim amendment](interim_amendment_2026_11_14.json)
supersedes the earlier no-interim rule. Current earnings results are generated in
`markdowns/chapter_directions_earnings_interim.md`; the original report is a
historical pre-interim assessment.

## Current interim earnings run

No new 2025 upload is needed. These public inputs are already cached under
`data/raw/qcew_episode_2025_2026/`:

| File | Required use |
|---|---|
| `2025_qtrly_singlefile.zip` | 2025Q1/Q2 for the prior-window lead and 2025Q4 for the first post-policy quarter |
| `2026_qtrly_singlefile.zip` | 2026Q1 for the second available post-policy quarter |
| `MANIFEST.csv` | Acquisition URLs, source dates and expected hashes |

These are quarterly single-file archives; the annual single-file product does
not supply quarter-by-quarter outcomes. Each archive includes county private
crop production (111), animal production (112) and all-private totals (10).
The 2023/2024 prior-year files and pre-policy extract are already available.
After preparing the base scratch below, run:

```sh
bash explorations/chapter_directions/run_interim.sh /tmp/h2a-chapter-directions
```

Stages `09_extract_interim.py`, `10_estimate_interim.R`, and `11_report_interim.R`
verify the archive hashes, reuse the frozen extractor, and estimate the
two-quarter window, separate quarters, common-sample checks, sensitivities,
matched pre-policy windows and the complete prior-window lead.
`tests/check_interim.R` validates period completeness, suppression, keys,
lead equivalence and independent OLS reproduction. QWI runs only if its
existing source-owned panel covers the same interim window. Repeated runs
reproduce the same recorded vintage; no downloads or schedules are invoked.

## Original current-data reproduction

Run from the repository root in the existing environment:

```sh
DRY_RUN=1 bash explorations/chapter_directions/run.sh /tmp/h2a-chapter-directions
bash explorations/chapter_directions/run.sh /tmp/h2a-chapter-directions
```

The runner uses the existing devenv profile and renv library when they are not
already on the environment paths; it installs nothing. It does not acquire
sources. Scratch must be outside the repository. Stage ownership is sequential:

| Stage | Inputs and responsibility | Scratch output |
|---|---|---|
| `00_prepare.py` | Existing county panel, disclosure/allocation/linkage/rates/QWI; reconstruct scratch dependencies through their original owners; verify frozen exposure | `inputs.json`, `filer/`, `benchmark/` |
| Original decomposition and earnings estimators | Original frozen specification and scratch inputs | `benchmark/t1emp_*`, original tau2024 estimate |
| `01_build_wages.R` | Allocated hours, case occupation, R7/R5/R8 and FEIN, statutory dose | Filer-state-occupation panel; conservation and floor audits |
| `02_estimate_wages.R` | Continuing cells; all declared weights, linkage, composition and selection checks | `wage_estimates.csv`, state deletions and coverage |
| `03_build_administration.R` | Disclosure case records and baseline-only projected dates | Case ledger, recurring cohort panel, receipt and duplication audits |
| `04_estimate_administration.R` | Cohort panel, both weights and declared comparisons | All administrative estimates, support, balance and deletions |
| `05_extract_earnings.py` | Existing public 2014–2024 QCEW ZIPs; existing canonical geography mapper | Three-series pre-policy quarterly panel and vintage hashes |
| `06_estimate_earnings.R` | Frozen exposure, pre-policy QCEW and QWI, sourced UI review | 2016–2024 diagnostics, coverage, queue; no post-policy estimate |
| `07_validate.R` | All scratch artifacts and hashes | `validation.csv/json`, reproduction checks |
| `08_report.R` | Validated estimates, audits and provenance | `markdowns/chapter_directions_exploration.md` and complete specification appendix |

`validate.sh` runs stage 07 plus R parsing, changed-file Ruff, Marimo exports,
Python edge tests, shell syntax, pipeline dry runs, grounding tests and Zola.
Each command's status and log are recorded in `execution_checks.csv`.

Optional `CD_FILER_SOURCE`, `CD_BENCHMARK_SOURCE`, and `CD_EARNINGS_SOURCE` point
to prior scratch directories. Only named dependencies are copied. When absent,
preparation invokes the existing employer-panel, identification and episode
owners. It restores a missing frozen exposure file only if reconstruction has
the original exact SHA-256. It never changes the linkage rules.

Before the interim amendment, the no-outcome-access rule prevented opening the
cached 2025 archive, and separate pre-policy CSV slices could not be retrieved.
The authorized interim runner now completes the lead from the cached archive.
The older slice-only route remains available for original-workflow reproduction:

```sh
export CHAPTER_SCRATCH=/tmp/h2a-chapter-directions
export UV_CACHE_DIR=/tmp/h2a-uv-cache
uv run --no-sync marimo export script explorations/chapter_directions/05_extract_earnings.py -o /tmp/h2a-chapter-directions/05_extract_earnings.flat.py --force
CD_EARNINGS_MODE=lead CD_ACQUIRE_LEAD=1 uv run --no-sync python /tmp/h2a-chapter-directions/05_extract_earnings.flat.py
Rscript --vanilla explorations/chapter_directions/06_estimate_earnings.R /tmp/h2a-chapter-directions lead
```

Alternatively place the exact public quarter slices at
`qcew_2025Q1_111.csv` and `qcew_2025Q2_111.csv` in scratch and omit acquisition.
The reader rejects unexpected years/quarters. Separate slice downloads are
unnecessary for the current authorized analysis.

## Deferred full-window earnings run

After actual publication (scheduled December 2, 2026, 10 a.m. New York time),
verify the [BLS release](https://www.bls.gov/schedule/news_release/cew.htm) and
create `/tmp/h2a-chapter-directions/release_metadata.json` with:

```json
{
  "status": "published",
  "published_quarters": ["2024Q4", "2025Q1", "2025Q2", "2025Q4", "2026Q1", "2026Q2"],
  "source_url": "https://www.bls.gov/news.release/cewqtr.htm",
  "checked_at": "REPLACE_WITH_ACTUAL_VERIFICATION_TIMESTAMP"
}
```

This is a schema example, not evidence of publication. Then run exactly:

```sh
bash explorations/chapter_directions/resume_earnings.sh /tmp/h2a-chapter-directions /tmp/h2a-chapter-directions/release_metadata.json
```

It verifies frozen hashes and the calendar/publication gate, downloads a
consistent current vintage for 2023–2026 to scratch, checks all six quarters
for all three industries, runs the entire registered queue, and writes a
completion marker that prevents a second full-window run. The separate interim
runner is the user-authorized partial-window path. No automatic scheduling.
Later vintages must be labeled updates to already inspected outcomes; any
specification change requires a dated amendment and preserved earlier artifacts.

Optional QWI requires `CD_QWI_COMPLETE_EXTRACT` and
`CD_QWI_RELEASE_METADATA`. The Parquet has string `county_fips`, integer `year`
and `qtr`, and the source-owned `qwi_stable_monthly_earnings`,
`qwi_stable_employment`, `qwi_stable_monthly_earnings_status` columns for private
NAICS 111. Metadata must confirm published 2026Q2, an official census.gov
`source_url`, `checked_at`, and `extract_sha256`. Status 1 and complete six-quarter
county windows are required. Missing QWI stays in the queue.

## Checks

```sh
UV_CACHE_DIR=/tmp/h2a-uv-cache uv run --no-sync ruff check explorations/chapter_directions/00_prepare.py explorations/chapter_directions/05_extract_earnings.py explorations/chapter_directions/tests/test_earnings.py
CHAPTER_SCRATCH=/tmp/h2a-chapter-directions CD_EXPORTED_EARNINGS=/tmp/h2a-chapter-directions/05_extract_earnings.flat.py UV_CACHE_DIR=/tmp/h2a-uv-cache uv run --no-sync python explorations/chapter_directions/tests/test_earnings.py
Rscript --vanilla explorations/chapter_directions/07_validate.R /tmp/h2a-chapter-directions
bash -n explorations/chapter_directions/run.sh explorations/chapter_directions/resume_earnings.sh
```

Export each changed Marimo application to `/tmp` before executing it. The report
records actual executed validation, not merely these instructions. Source
hashes, dependency vintages, formulae, failed fits and state deletions are kept
alongside estimates. Do not edit generated reports to change a result.
