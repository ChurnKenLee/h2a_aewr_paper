# H-2A Paper

The supported workflow separates source preparation, a design-neutral
county-year panel, and multiple empirical designs. Each entry point is standalone,
uses repository-root paths from `code/paths.R`, and writes a declared artifact.

## Dependency graph

```text
code/a01_sources ──> code/b01_derived ──> code/c01_clean
                                              │
                                              v
                                       county_year_merged
                                              │
                                              v
                                       code/c02_build
                                              │
                                              v
                                       county_year_panel
                         /       |        |        |        |        \
                        v        v        v        v        v         v
              descriptives     DiD   panel IV   MC dose   border ring   welfare
```

| Branch | Owner | Principal artifact or output |
| --- | --- | --- |
| Source data | [`code/a01_sources`](code/a01_sources/) | Normalized source Parquets |
| Derived data | [`code/b01_derived`](code/b01_derived/) | Reusable upstream measures |
| Shared clean | [`code/c01_clean`](code/c01_clean/) | `data/intermediate/county_year_merged.parquet` |
| Shared build | [`code/c02_build`](code/c02_build/) | `data/processed/county_year_panel.parquet` |
| Descriptives | [`code/descriptives`](code/descriptives/) | Two shared figures |
| DiD | [`code/designs/did`](code/designs/did/) | `data/processed/did_county_year_panel.parquet` and manuscript tables |
| Panel IV | [`code/designs/panel_iv`](code/designs/panel_iv/) | Original moving-wage IV, deterministic R calibration, retained estimates and diagnostics |
| Wooldridge–Mundlak–Chamberlain | [`code/designs/mundlak_chamberlain`](code/designs/mundlak_chamberlain/) | DuckDB/Polars linear-history OLS registry, constructed estimands, and analytic inference diagnostics |
| AEWR-border ring | [`code/designs/border_discontinuity`](code/designs/border_discontinuity/) | Standalone bounded local redesign, preserved ring benchmarks, dependence/power audit and report |
| Historical wage reforms | [`code/designs/wage_reforms`](code/designs/wage_reforms/) | Reversal and herder wage, participation, assignment-support and linkage diagnostics |
| Narrow welfare | [`code/welfare`](code/welfare/) | Certification-space producer incidence from one frozen certified-hours elasticity |

The shared panel owns normalized source measures, reusable outcomes and
controls, 2011 farm employment, current and lagged AEWR-p25 wage gaps, the
BEA employment and farm wage-bill totals, separate QCEW NAICS 111/112 and
all-sector employment/wage-bill totals with disclosure flags, nominal OEWS
Big-Six hourly- and annual-wage proxies and their support fields, cropland
eligibility, border-CZ status, and stable geographic identifiers. It does not
own treatment classifications, post indicators, fixed-effect factors, year
dummies, target-cluster assignments, or instruments.

## Commands

Run the complete dependency chain:

```sh
./scripts/run_all.sh
```

Run only a supported C-side branch:

```sh
./scripts/run_shared_panel.sh
./scripts/run_descriptives.sh
./scripts/run_did.sh
./scripts/run_panel_iv.sh
./scripts/run_border_discontinuity.sh
./scripts/run_wage_reforms.sh
./scripts/run_mundlak_chamberlain.sh
./scripts/run_welfare.sh
```

The AEWR-border runner is a standalone bounded investigation and is not called by
`run_all.sh` or the manuscript workflow. The welfare runner is also standalone
and intentionally stops until its elasticity contract is frozen.
`./scripts/run_border_data_inputs.sh` separately reproduces the cached input
corrections and affected consumer rebuilds, including the input-comparison
report `outputs/reports/border_data_inputs.md`, a public-data plan and an unsent
DOL archival draft. Non-public NASS access/custom tabulations are unavailable.
See the [border README](code/designs/border_discontinuity/README.md) for scope
and the explicit public-geocoding option.
The historical wage-reform runner is standalone and consumes cached source
microdata; it writes `outputs/reports/wage_reforms.md` with explicit causal
identification and data-recovery limits.
The panel-IV runner executes stages 01--11 in R and retains the original
moving-wage estimator with deterministic calibration; stages 10--11 add a
diagnostic Kitagawa decomposition of its instruments.

Set `DRY_RUN=1` before a command to print its execution order without running
it. See [`scripts/README.md`](scripts/README.md) for source/derived runners and
expected outputs.

## Conventions

- `data/raw` contains source files and hand-maintained crosswalks.
- `data/intermediate` contains exchange artifacts owned by one producer.
- `data/processed` contains analysis-ready panels.
- `outputs/figures` and `outputs/tables` contain retained empirical products.
- `outputs/reports/border_redesign.md` is a generated methodological report;
  its findings do not automatically enter the manuscript or welfare.
- Geographic identifiers follow
  [`content/contracts/geographic-codes.md`](content/contracts/geographic-codes.md).
- H-2A employer IDs and counts follow the reviewed FY2008--2024
  [worksite-business linkage contract](content/contracts/employer-linkage.md).
- Old local Parquets may remain on disk, but only the artifacts documented
  above are supported.

## Grounded agent documentation

Cross-cutting contracts and empirical-design explanations live under
[`content`](content/). They are optimized as working context for coding agents
and remain ordinary readable Markdown for researchers. Before changing a
scope, inspect its active instructions and canonical pages:

```sh
python scripts/agent_docs.py snapshot --scope code/designs/panel_iv
```

The pages are rendered with Zola and checked against named source regions.
Inside the Devenv shell, use:

```sh
agent-docs-check
agent-docs-serve
```

Directory READMEs remain operational entry points. The Zola pages are the
canonical references for pipeline architecture, artifact semantics, and
research-design claims. `static/llms.txt` and
`static/grounding-manifest.json` are generated LLM entry points; do not edit
them directly.
