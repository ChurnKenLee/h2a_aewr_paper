# H-2A Paper

The supported workflow separates source preparation, a design-neutral
county-year panel, and two empirical designs. Each entry point is standalone,
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
                                       /        |        \
                                      v         v         v
                            descriptives      DiD      panel IV
                            border discontinuity: documented future branch
```

| Branch | Owner | Principal artifact or output |
| --- | --- | --- |
| Source data | [`code/a01_sources`](code/a01_sources/) | Normalized source Parquets |
| Derived data | [`code/b01_derived`](code/b01_derived/) | Reusable upstream measures |
| Shared clean | [`code/c01_clean`](code/c01_clean/) | `data/intermediate/county_year_merged.parquet` |
| Shared build | [`code/c02_build`](code/c02_build/) | `data/processed/county_year_panel.parquet` |
| Descriptives | [`code/descriptives`](code/descriptives/) | Two shared figures |
| DiD | [`code/designs/did`](code/designs/did/) | `data/processed/did_county_year_panel.parquet` and manuscript tables |
| Panel IV | [`code/designs/panel_iv`](code/designs/panel_iv/) | `data/processed/panel_iv_cluster_year.parquet`, IV results, and retained diagnostic figures |
| Future border design | [`code/designs/border_discontinuity`](code/designs/border_discontinuity/) | Documentation only |

The shared panel owns normalized source measures, reusable outcomes and
controls, 2011 farm employment, current and lagged AEWR-p25 wage gaps,
cropland eligibility, border-CZ status, and stable geographic identifiers. It
does not own treatment classifications, post indicators, fixed-effect factors,
year dummies, target-cluster assignments, or instruments.

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
```

Set `DRY_RUN=1` before a command to print its execution order without running
it. See [`scripts/README.md`](scripts/README.md) for source/derived runners and
expected outputs.

## Conventions

- `data/raw` contains source files and hand-maintained crosswalks.
- `data/intermediate` contains exchange artifacts owned by one producer.
- `data/processed` contains analysis-ready panels.
- `outputs/figures` and `outputs/tables` contain retained manuscript products.
- Geographic identifiers follow
  [`documentation/geographic_code_contract.md`](documentation/geographic_code_contract.md).
- Old local Parquets may remain on disk, but only the artifacts documented
  above are supported.
