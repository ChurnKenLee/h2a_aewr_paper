# 2025–27 policy episode: pre-period checks and draft registries

Current wage-offer and interim earnings development lives in the
[2025 policy-shift design](../../code/designs/policy_2025/README.md).
This folder preserves the original prechecks, frozen exposure/registry and
full-window workflow as historical evidence. Its pre-outcome statements
below describe that original workflow, not the subsequently authorized
interim analysis.

Exploratory R scripts behind `markdowns/episode_2025_prechecks.md`. They are
not pipeline stages. They read retained artifacts and write only to the scratch
directory given as the first argument, which must be outside the repository.
Nothing here selects or changes a retained specification, table or producer.

No script reads an H-2A outcome after FY2025 or a pay outcome after 2024.
Case-level reads go through `id_read_disclosures()` or an Arrow filter on
`fiscal_year <= 2025`.

Run from the repository root. `$SH` is the output directory of
`explorations/identification_search/shared/01_outcomes.R` (about 8 minutes to
rebuild; see that folder's README).

```sh
export R_LIBS_USER=$(ls -d "$PWD"/renv/library/*/R-*/* | head -1)
ep_scratch=/path/outside/repo/episode_2025
mkdir -p "$ep_scratch"
Rscript --vanilla explorations/episode_2025/01_usage_full_cut_precheck.R "$ep_scratch" "$SH"
Rscript --vanilla explorations/episode_2025/02_usage_untouched_jobs_precheck.R "$ep_scratch" "$SH"
Rscript --vanilla explorations/episode_2025/03_event_support.R "$ep_scratch" "$SH"
Rscript --vanilla explorations/episode_2025/04_domestic_wage_support.R "$ep_scratch"
Rscript --vanilla explorations/episode_2025/05_h2a_in_qcew_state.R "$ep_scratch"
Rscript --vanilla explorations/episode_2025/06_qcew_quarterly_extract.R "$ep_scratch" 2014 2024   # about 10 minutes
Rscript --vanilla explorations/episode_2025/04_domestic_wage_support.R "$ep_scratch" \
  "$ep_scratch/qcew_crop_county_quarter_2014_2024.parquet"                                     # adds the QCEW quarterly window
Rscript --vanilla explorations/episode_2025/07_usage_extension_precheck.R "$ep_scratch" "$SH"
Rscript --vanilla explorations/episode_2025/09_domestic_wage_exposure.R "$ep_scratch"
# placebo run of the declared estimator (code-path check; tau <= 2024 only)
Rscript --vanilla explorations/episode_2025/08_domestic_wage_estimate.R "$ep_scratch" \
  "$ep_scratch/qcew_crop_county_quarter_2014_2024.parquet" 2024
git checkout -- Rplots.pdf
```

| File | What it does | Report section |
| --- | --- | --- |
| `ep_helpers.R` | Loads `explorations/identification/id_helpers.R`; state frame with former region, FY2025 rate and both statewide schedules; placebo regression and summary helpers | all |
| `01_usage_full_cut_precheck.R` | Check A: placebo-year design error of the within-region design against full-dose designs with trend, lagged-growth and factor-loading adjustments; reproduces the reported FY2024-vs-FY2025 PPML placebo | §1 |
| `02_usage_untouched_jobs_precheck.R` | Check B: range herders and piece-rate orders as within-state controls: size, co-movement with treated jobs, placebo-year error of the triple difference | §2 |
| `03_event_support.R` | Dose support for the 2025-10-02 and 2026-08-03 schedules and their correlation | §3 |
| `04_domestic_wage_support.R` | Domestic-wage design: exposure and coverage, placebo-year precision on QCEW annual pay, QWI annual earnings and the QWI Q4–Q2 window | §4 |
| `05_h2a_in_qcew_state.R` | State totals of H-2A full-time equivalents against QCEW crop and crop-support employment | §4 |
| `06_qcew_quarterly_extract.R` | County-quarter private crop-production pay and employment from raw QCEW quarterly files, for the years asked for | §4 |
| `07_usage_extension_precheck.R` | Late-season FY2026 windows and a second season of the usage design: hours shares and placebo-year error | §3 |
| `08_domestic_wage_estimate.R` | The declared domestic-wage estimator. For 2026 it runs only if the registry and exposure hashes match and all three quarters exist; for a year up to 2024 it is a placebo run | §4 |
| `09_domestic_wage_exposure.R` | Builds the county exposure and state dose table that the registry freezes | §4 |
| `usage_episode_registry_draft.json` | Draft registry for later windows and events of the usage design. Not frozen | §3 |
| `domestic_wage_registry.json`, `domestic_wage_exposure_frozen.csv`, `domestic_wage_registry.sha256` | Frozen registry for the domestic-wage design, its exposure table, and their hashes | §4 |
| `sources/` | The 2026-08-26 order, the OFLC notice of 2026-09-02, and sourced notes on the rule's legal status | §3 |

The usage registry is a draft: it carries no hash, and nothing may be
estimated under it. The domestic-wage registry is frozen; change it only by a
dated amendment, and only before any pay file for 2025 Q4 or later is opened.

Post-policy QCEW files are kept in `data/raw/qcew_episode_2025_2026/`, apart
from `data/raw/qcew`, so that the shared QCEW producer and the county panel
are unaffected. No script here opens them except
`06_qcew_quarterly_extract.R` when given that directory.

Placebo-year figures are 2.8 times the root mean squared placebo estimate (or
the larger of the mean clustered standard error and the spread of placebo
estimates, for the domestic-wage design). They compare designs on the same
footing. They are not the size-validated MDEs of
`explorations/identification_search/shared/power_harness.R`.
