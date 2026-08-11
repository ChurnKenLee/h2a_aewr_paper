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
                                       /        |        |        \
                                      v         v        v         v
                            descriptives      DiD    panel IV   MC dose response
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
| Panel IV | [`code/designs/panel_iv`](code/designs/panel_iv/) | Panel-IV artifacts, retained tables, and diagnostics |
| Multilevel Mundlak–Chamberlain | [`code/designs/mundlak_chamberlain`](code/designs/mundlak_chamberlain/) | Audited dose-response specification program, retained tables, figures, and diagnostics |
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
./scripts/run_mundlak_chamberlain.sh
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
  [`content/contracts/geographic-codes.md`](content/contracts/geographic-codes.md).
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
