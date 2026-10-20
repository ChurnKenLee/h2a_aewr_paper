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
| `run_panel_iv.sh` | Original moving-wage panel-IV construction, estimation, reports and validation through stage 09, all in R |
| `run_border_discontinuity.sh` | Standalone bounded border redesign: retained v5 audit, all-member local screen, four new comparisons, dependence/size/power evidence, report and validation |
| `run_border_data_inputs.sh` | Cached wage/Census corrections, audited allocation/calendar, geocoding preparation, affected shared/design rebuilds, denominator comparisons, public-data plan/DOL archival draft and validation |
| `run_wage_reforms.sh` | Standalone local-data investigation of the 2009–2010 reversal and herder phase-in: source/cohort panels, response and support diagnostics, report and validation |
| `run_mundlak_chamberlain.sh` | Version-4 Wooldridge–Mundlak–Chamberlain continuous-dose registry: DuckDB/Polars panel, full-history and one-lag OLS, constructed estimands, HC3/cluster inference, experimental scalar CCV comparator, and validation |
| `run_welfare.sh` | Standalone narrow certification-space producer-incidence calculation; requires a frozen, source-traced certified-hours elasticity contract |
| `run_all.sh` | Sources, derived data, shared panel, descriptives, DiD, panel IV, Mundlak–Chamberlain |

Examples:

```sh
DRY_RUN=1 ./scripts/run_all.sh
./scripts/run_shared_panel.sh
./scripts/run_did.sh
./scripts/run_panel_iv.sh
./scripts/run_border_discontinuity.sh
./scripts/run_border_data_inputs.sh
./scripts/run_wage_reforms.sh
./scripts/run_mundlak_chamberlain.sh
./scripts/run_welfare.sh
```

`run_border_discontinuity.sh` is an experimental design branch and remains
outside `run_all.sh` and the manuscript workflow. `run_welfare.sh` remains
outside `run_all.sh` until its elasticity contract is frozen.
The historical wage-reform runner also remains outside `run_all.sh`. It reads
cached H-2A source and reviewed linkage artifacts, exports four Marimo stages
to the runner's temporary directory, and performs no source refresh.

Panel IV runs its original stages 01--09 entirely with `Rscript --vanilla`.
Stages03/04 construct the frame and deterministic weights; stage09 validates
sources, weights, instruments and retained results. The experimental extensions
and simulated calibration draws have been retired. The border runner orders `01,02,03,04,05,07,08,06`; its final validator
checks both the preserved benchmark and new power/report evidence.

The standalone input-upgrade runner orders cached A01 wage/Census preparation,
B01 allocation and `01b` monthly exposure, A01 point-geocoding preparation,
shared panel, descriptives, DiD, panel IV, bounded MC, ordinary border, and
border09--11. It does not call source acquisition or optional refresh runners.
`H2A_RUN_PUBLIC_GEOCODER=1` explicitly enables public Census requests; the
default only prepares/reads caches. The public-data plan excludes non-public
NASS access/tabulations; no institutional application details are required.
The separate DOL archival draft is not submitted by any runner.
The normal derived runner now includes `01b` immediately after `01`, so the
new calendar retains a single producer in the dependency chain.

The source runner's H-2A worksite-business step always materializes its seeded
review sample, but it stops before promoting the v2 crosswalk unless
`data/raw/h2a/employer_linkage_review_labels.csv` completely labels all 200
pairs. No live source refresh is needed to rerun that linkage step against the
cached with-FIPS inputs.
