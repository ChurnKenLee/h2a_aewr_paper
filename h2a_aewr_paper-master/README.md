# H-2A Paper

The supported workflow separates source preparation, a design-neutral
county-year panel, and multiple empirical designs. Each entry point is standalone,
uses repository-root paths from `code/paths.R`, and writes a declared artifact.

## Dependency graph

<!-- grounding-fence: illustrative -->
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
                                       /        |        |        \
                                      v         v        v         v
                            descriptives      DiD    panel IV   MC dose response
                            historical border proposal: not executable
```

| Branch | Owner | Principal artifact or output |
| --- | --- | --- |
| Source data | [`code/a01_sources`](code/a01_sources/) | Normalized source Parquets |
| Derived data | [`code/b01_derived`](code/b01_derived/) | Reusable upstream measures |
| Shared clean | [`code/c01_clean`](code/c01_clean/) | `data/intermediate/county_year_merged.parquet` |
| Shared build | [`code/c02_build`](code/c02_build/) | `data/processed/county_year_panel.parquet` |
| Descriptives | [`code/descriptives`](code/descriptives/) | Two shared figures |
| DiD | [`code/designs/did`](code/designs/did/) | `data/processed/did_county_year_panel.parquet` and manuscript tables |
| Panel IV | [`code/designs/panel_iv`](code/designs/panel_iv/) | `data/processed/panel_iv_county_year.parquet`, four first stages, twelve four-column 2SLS outcome tables, summary statistics, and diagnostics |
| Multilevel Mundlak–Chamberlain | [`code/designs/mundlak_chamberlain`](code/designs/mundlak_chamberlain/) | Version-3 specification registry and compact default queue, finite-design continuous-treatment CCV inference, tables, coefficient graphs, and diagnostics |
| Historical border-enforcement proposal | [`documentation/H2A Estimation Strategy.docx`](documentation/H2A%20Estimation%20Strategy.docx) | Historical proposal only; no supported executable branch |

The shared panel owns normalized source measures, reusable outcomes and
controls, 2011 farm employment, current and lagged AEWR-p25 wage gaps,
cropland eligibility, border-CZ status, and stable geographic identifiers. It
does not own treatment classifications, post indicators, fixed-effect factors,
year dummies, target-cluster assignments, or instruments.

## Commands

Run the complete dependency chain:

<!-- grounding-fence: illustrative -->
```sh
./scripts/run_all.sh
```

Run only a supported C-side branch:

<!-- grounding-fence: illustrative -->
```sh
./scripts/run_shared_panel.sh
./scripts/run_descriptives.sh
./scripts/run_did.sh
./scripts/run_panel_iv.sh
./scripts/run_mundlak_chamberlain.sh
```

Set `DRY_RUN=1` before a command to print its execution order without running
it. See [`scripts/README.md`](scripts/README.md) for source/derived runners and
expected outputs.

## Agent grounding and documentation

Coding-agent instructions are layered from the root [`AGENTS.md`](AGENTS.md)
into stage- and design-specific files. The searchable Zola knowledge base lives
under [`agent-docs`](agent-docs/) and separates curated research contracts from
machine-generated repository facts.

<!-- grounding-fence: illustrative -->
```sh
python scripts/agent_grounding.py snapshot --scope code/designs/panel_iv
python scripts/agent_grounding.py verify
./scripts/run_tests.sh
```

Inside `devenv shell`, use `agent-docs-serve` for the local site and
`agent-docs-generate` after changing code, runners, locks, assumptions, or
agent documentation. CI rejects stale generated grounding. The source-linked
grounding page draws an extensive excerpt registry across runtime orchestration,
geography/data contracts, DiD, panel IV, and the Mundlak-Chamberlain
specification program. Its generated index reports the current count. A changed
excerpt blocks ordinary generation until its digest is explicitly reviewed and
accepted.

## Conventions

- `data/raw` contains source files and hand-maintained crosswalks.
- `data/intermediate` contains exchange artifacts owned by one producer.
- `data/processed` contains analysis-ready panels.
- `outputs/figures` and `outputs/tables` contain retained manuscript products.
- Geographic identifiers follow
  [`documentation/geographic_code_contract.md`](documentation/geographic_code_contract.md).
- Old local Parquets may remain on disk, but only the artifacts documented
  above are supported.
