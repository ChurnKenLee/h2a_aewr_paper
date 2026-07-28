# Pipeline runners

All runners execute from the repository root. Set `DRY_RUN=1` to print their
steps without running them.

| Runner | Responsibility |
| --- | --- |
| `run_sources.sh` | Required A-stage source acquisition and normalization |
| `run_optional_sources.sh` | Optional source refreshes |
| `run_derived.sh` | Supported B-stage transformations |
| `run_shared_panel.sh` | C01 normalization/merge followed by C02 shared build |
| `run_descriptives.sh` | Two retained shared figures |
| `run_did.sh` | DiD panel, four-column results, summary table, one coefficient plot |
| `run_panel_iv.sh` | Panel-IV construction, four retained models, first-stage plot, and six design figures with plotting data |
| `run_all.sh` | Sources, derived data, shared panel, descriptives, DiD, panel IV |
| `run_tests.sh` | Parse, syntax, ownership, dry-run, and artifact smoke checks |

Examples:

```sh
DRY_RUN=1 ./scripts/run_all.sh
./scripts/run_shared_panel.sh
./scripts/run_did.sh
./scripts/run_panel_iv.sh
./scripts/run_tests.sh
```
