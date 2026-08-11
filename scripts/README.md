# Pipeline runners

All runners execute from the repository root. Set `DRY_RUN=1` to print their
steps without running them.

R steps run with `Rscript --vanilla`. The shared runner helper selects the
project-local `renv` library matching the active R major/minor version and
platform, so stale libraries from an earlier R version are ignored. If no
matching library exists after an R upgrade, run `r-renv-restore` inside the
Devenv shell before rerunning the pipeline.

| Runner | Responsibility |
| --- | --- |
| `run_sources.sh` | Required A-stage source acquisition and normalization |
| `run_optional_sources.sh` | Optional source refreshes |
| `run_derived.sh` | Supported B-stage transformations |
| `run_shared_panel.sh` | C01 normalization/merge followed by C02 shared build |
| `run_descriptives.sh` | Two retained shared figures |
| `run_did.sh` | DiD panel, four-column results, summary table, one coefficient plot |
| `run_panel_iv.sh` | County-year panel IV: four first stages, seven four-column 2SLS tables, summary statistics, and diagnostic figures |
| `run_mundlak_chamberlain.sh` | Multilevel Mundlak–Chamberlain AEWR-change dose response: panel, 28-model ladder, finite-design CCV inference, ggplot2 coefficient graphs, Great Tables outputs, diagnostics, and validation |
| `run_all.sh` | Sources, derived data, shared panel, descriptives, DiD, panel IV, Mundlak–Chamberlain |

Examples:

```sh
DRY_RUN=1 ./scripts/run_all.sh
./scripts/run_shared_panel.sh
./scripts/run_did.sh
./scripts/run_panel_iv.sh
./scripts/run_mundlak_chamberlain.sh
```
