# Wooldridge–Mundlak–Chamberlain continuous-dose design

This branch consumes `data/processed/county_year_panel.parquet` and runs the
version-4 DuckDB/Polars specification registry. Version 4 replaces the former
three-lag polynomial Mundlak–Chamberlain program; the R v2.3/v3 files remain
only as a historical compatibility record and are not called by the supported
runner.

## Run

From the repository root:

```sh
./scripts/run_mundlak_chamberlain.sh
```

The default `MC_SPEC_STAGE=compact` queue is bounded and deliberately spans
the archive's unresolved choices. `MC_SPEC_STAGE=exhaustive` is an explicit
opt-in. `MC_SPEC_IDS`, `MC_SPEC_MAX`, and `MC_SPEC_FORCE=1` narrow or refresh
the queue. `MC_SPEC_MAX_DENSE_GIB` (default 1.25) and
`MC_SPEC_MAX_PEAK_GIB` (default 6) guard dense dictionaries and the estimated
OLS working set before expensive allocation. Data-dependent causal-rank guards
remain active in both stages;
the 2008–09 full exposure history is estimable with county-plus-year effects
but not with the stronger region/state-by-year sets, while the 2008–10 window
supports the declared region-by-year full-history sensitivity.

## Program stages

| Script | Responsibility |
| --- | --- |
| `01_build_panel.py` | DuckDB scan; frozen-baseline bite/exposure construction; Polars panel and deterministic cluster partitions |
| `02_build_registry.py` | Compile the bounded or opt-in specification registry |
| `03_estimate.py` | Exact nested-FE FWL, causal-first rank selection, common-design six-outcome OLS, full leverage, and provenance caches |
| `04_report.py` | Construct identified 2013–2022 average current-coordinate effects and compare HC3, CR0/CR1, and explicitly experimental scalar CCV-HC3 |
| `05_validate.py` | Check artifact coverage, input hashes, inference labels, and rejected-method flags |

The executable contract is `mcw/design.py`. Treatment history begins in 2011;
outcome rows cover 2012–2022. Full history means separate linear coordinates
from 2011 through each outcome year. The pooled candidate retains all 77 cells.
County fixed effects absorb the common
levels of the full model's 2011 and 2012 paths, so their later coefficients are
explicit differences from the 2012 outcome cells. The one-lag benchmark keeps
all current and preceding coordinates, including its identified 2012 current
cell; that coefficient remains in the detailed artifact while cross-model
current-effect summaries use the common 2013–2022 target. Neither model uses polynomial
dose terms, polynomial lag profiles, a three-lag truncation, cross-dose
products, imposed trends, nonlinear links, randomization inference, or a
bootstrap.

The compact queue begins with the archive-motivated, explicitly candidate
pooled/no-county architecture: region and calendar contrasts, all hierarchical
pre-period components, their unrestricted calendar interactions, region by
non-region-component interactions, and the full predetermined causal moderator
set, centered within AEWR region for causal interactions. County plus
global/state/region-year models are retained as sensitivities and for
lower-geography treatments. Causal-first rank selection never drops a named
history coordinate; it removes only redundant projection columns.

Six primitive outcomes are fitted on one common sample and design:
applications, balanced-linkage employers, requested positions, certified
positions, certified hours, and any application. Rates, ratios, and
mean-normalized effects are constructed after estimation with named gradients;
no post-outcome-selected sample is used by the supported fit. Only raw
`aewr_log_level` and raw `aewr_log_change` coordinates produce observed-mean
elasticities. Dollar, bite, exposure, binary, rank, and standardized coordinates
instead report percentage of the observed outcome mean per one declared
treatment unit; that normalization is not labeled an elasticity.
Every result persists its target population, observation count, weight sum,
and weighting rule. Constructed estimands receive joint cross-outcome HC3,
CR0, and CR1 delta-method inference. The experimental scalar CCV comparator is
also reported when every nonzero outcome loading is proportional to one common
coefficient contrast, including the declared ratio-of-aggregates derivatives.
It is not extended to arbitrary multivariate gradients without a justified
cross-outcome assignment kernel.

The frozen-distribution bite uses a declared five-quantile approximation to
the unavailable county wage microdistribution. The 2008–09 window is the
candidate definition and 2008–10 is a sensitivity. Both retain support-count
metadata, including Rhode Island's incomplete 2008–09 history. The analysis
eligibility/normalization input is mean 2008–10 farm employment, not a special
2011 denominator. The frozen exposure share is the corresponding pre-period
mean fraction affected, and the baseline bite moderator is the 2008–2010 mean
real AEWR-minus-p25 gap; neither uses 2011 as a special baseline coordinate.

## Inference status

The Lean work supports general continuous covariance-kernel identities,
binary reduction, and convex-mixture bounds. It does **not** establish that
the scalar residualized-dose second-moment mixture is exact for arbitrary
continuous assignments. Accordingly, `ccv_hc3_scalar_mixture_experimental`
is a transparently labeled comparator. It uses `q=1`, residualized-dose
cluster moments, and full-model HC3 leverage. HC3, CR0, and conventionally
adjusted CR1 are reported separately; dense CR2 and literal deletion-refit CV3
are verification/diagnostic oracles.

Identification, treatment families, unresolved choices, estimands, inference,
and retained diagnostics are documented in the grounded
[design page](../../../content/designs/mundlak-chamberlain.md).
