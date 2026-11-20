# 2025 H-2A policy shift

Standalone design for wage offers and covered agricultural earnings around the
October 2, 2025 policy change. Read the [canonical design](../../../content/designs/policy-2025.md)
and [design registry](design.json) before changing behavior.

This is the November 16, 2026 organizational migration of the inspected wage
and earnings analyses from `explorations/chapter_directions`. It preserves
their specifications and source vintage. The original code, registries and
reports remain historical records; the new branch is the owner for subsequent
development. It does not run the 2013 shutdown analysis or import another
design's numerical settings. Historical county decomposition/utilization and
state-border power exercises remain separately documented references.

## Run

From the repository root, in the existing devenv/renv environment:

```sh
DRY_RUN=1 ./scripts/run_policy_2025.sh
./scripts/run_policy_2025.sh
```

The standalone runner uses only cached public inputs. It is outside
`run_all.sh`, manuscript and welfare runners. It uses the shared pipeline
helper, exports Marimo applications into temporary `/tmp` storage, runs R
with `--vanilla`, validates results, publishes artifacts and cleans temporary
files. A dry run does not read data, import R/Python, or access the network.

On the initial migration, `POLICY2025_LEGACY_SCRATCH` may point to the earlier
chapter scratch directory to reuse exactly three hash-verified dependencies:
the two filer Parquets and the pre-policy QCEW extract. It does not import
estimates or administrative artifacts. Subsequent runs use the retained design
dependencies. If caches are absent, the design rebuilds them from local inputs.
`sources/input_vintage.json` checks all 22 source files; drift stops the run
and requires a documented vintage/specification update.

## Order and ownership

The original stage numbers are retained to make the migration inspectable.

| Stage | Responsibility |
| --- | --- |
| `00_prepare.py` | Verify registry, exposure/source vintage and reusable caches; rebuild filer inputs through `inputs/01_filer_records.py` and `inputs/02_rule_harness.py` when needed |
| `01_build_wages.R` | Build filer–state–occupation and certification panels, state dose, allocation and floor audits |
| `02_estimate_wages.R` | All 347 wage-offer fits, state deletions, composition, selection, heterogeneity and distribution checks |
| `05_extract_earnings.py` | Pre-policy three-industry QCEW extraction; preserve full-window calendar/publication gate |
| `06_estimate_earnings.R` | All 135 pre-policy fits, disclosure/quarter audit and frozen UI classification |
| `09_extract_interim.py` | Verify and extract cached 2025/2026 archives using the same corrected aggregation |
| `10_estimate_interim.R` | Interim, quarter, common-sample, lead, sensitivity and matched-placebo queue |
| `tests/check_interim.R`, `07_validate.R` | Period/suppression/geography checks, independent OLS/WLS, full-ledger migration equivalence |
| `08_report.R` | Generated current report and complete specification appendix |
| `12_publish.R` | Retain validated products, record input/code/output hashes and verify published bytes |

`inputs/` owns this design's unchanged R5/R7/R8 filer reconstruction. It uses
shared name normalizers without changing the canonical worksite-business
crosswalk. Frozen declarations under `history/` retain their original dates and
paths as historical evidence, not runtime imports or new preregistrations.
`registry.sha256` pins declarations, input vintage, exposure, UI review and
migration comparison ledgers. Runtime implementation hashes are recorded in
the generated provenance. Do not automatically refresh frozen hashes.

## Inputs and retained products

The branch consumes the shared county panel, allocated worksites, disclosure
records, membership records, AEWR schedules, agricultural minimum wages, QWI,
cached QCEW 2014–2024 quarterly ZIPs, and
`data/raw/qcew_episode_2025_2026/{2025,2026}_qtrly_singlefile.zip` with their
acquisition manifest. Filer reconstruction also uses the local raw disclosure
workbooks. No source acquisition or credentials are part of the default run.

| Location | Contents |
| --- | --- |
| `data/intermediate/policy_2025/filer/` | `filer_cases.parquet`, `filer_ids.parquet` |
| `data/processed/policy_2025/` | Wage occupation/certification panels; pre-policy and interim earnings panels |
| `outputs/tables/policy_2025/` | Wage/earnings coefficient ledgers, state deletions, distributions, coverage, floor, allocation, quarter and validation audits |
| `outputs/reports/policy_2025.md` | Current integrated findings |
| `outputs/reports/policy_2025_specifications.md` | Every wage, pre-policy and interim coefficient row |
| `outputs/reports/policy_2025/` | Input/vintage metadata, validation JSON and `provenance.json` |

All regressions preserve nominal units, timing, sample rules, weights and
inference. The interim primary remains **2025Q4–2026Q1**, requiring both
current/prior pairs; no missing quarter is imputed. The full three-quarter
target is not substituted for this window. QWI remains unavailable if its
declared periods are missing. The original guarded full-window workflow is
documented in [the historical exploration README](../../../explorations/chapter_directions/README.md).

## Validate

The runner validates all 528 existing coefficient rows against immutable
reference ledgers at absolute tolerance `1e-10`, including sample sizes,
statuses, formulas, weights, standard errors and intervals. It also runs the
12 interim checks and an independent wage WLS. Source-vintage changes must
preserve earlier results and declare new comparison expectations explicitly.

```sh
UV_CACHE_DIR=/tmp/h2a-uv-cache uv run --no-sync ruff check code/designs/policy_2025
bash -n scripts/run_policy_2025.sh
python scripts/agent_docs.py verify
python scripts/test_agent_docs.py
zola check --skip-external-links
zola build
```

`tests/test_earnings.py` retains the extractor's seven edge checks. Export
`05_extract_earnings.py` under `/tmp`, set `POLICY2025_WORK` to external scratch
and `CD_EXPORTED_EARNINGS` to that exported file, then run the test through
`uv run --no-sync python`. Validation demonstrates computational consistency;
it does not establish causal identification or applicable legal wage floors.
