# Bite and identification explorations

Exploratory R scripts behind `markdowns/bite_designs_exploration.md`,
`markdowns/aewr_identification_notes.html` and
`markdowns/fls_wage_target_mechanical_relevance.md`. They are not pipeline
stages. They read retained artifacts and write only to the output directory
given as the first argument. Nothing here selects or changes a specification.

Run from the repository root:

```sh
R_LIBS_USER=$(ls -d $PWD/renv/library/*/R-*/* | head -1) \
  Rscript --vanilla explorations/bite/<script>.R <output-dir>
git checkout -- Rplots.pdf
```

| Script | What it does | Report section |
| --- | --- | --- |
| `variation_overview.R` | Variance decomposition of AEWR, instrument and outcomes; region-year data for the identification notes | Identification notes, Part 1 |
| `partial_r2.R` | Instrument partial R²; OLS vs IV vs unweighted regional OEWS instrument | Identification notes, Part 2 §8 |
| `common_shocks.R` | Target vs donor vs non-donor subregion co-movement, region bootstrap | Identification notes, Part 2 §7 |
| `cross_test.R` | Lagged donor wages vs target outcomes net of the AEWR (loads `common_shocks.R`) | Identification notes, Part 2 §7 |
| `plain_panel.R` | Two-way fixed-effects OLS on the real AEWR, IV on the same sample, AEWR lead | Plain panel regression results |
| `nonwage_weights.R` | Recalibrates weights with and without the wage target; weights-only first stages | FLS wage-target report |
| `county_bite.R` | Designs 1(a)/1(b) with 2007 Census pay-per-worker exposure | Bite report §1 |
| `bind_check.R` | Share of H-2A hours at the AEWR and local wage predictors | Bite report §1 |
| `county_bite_acs.R` | Designs 1(a)/1(b) with the frozen AEWR-minus-local-floor bite (ACS p25 and ag minimum wage) | Bite report §1b |
| `employer_bite.R` | Design 4: employer bindingness cohort, estimates, event study; writes `employer_bite_panel.parquet` | Bite report §4 |
| `employer_robust.R` | Leave-one-region-out and wild bootstrap; reads `employer_bite_panel.parquet` from the same output directory, so run `employer_bite.R` first | Bite report §4 |
| `employer_checks.R` | Job-title types, within-type estimates, lead placebo, alternative cohorts | Bite report §4b |
| `employer_windows.R` | Sub-window estimates (loads `employer_checks.R`) | Bite report §4b |
| `employer_persistence.R` | Persistence of bound status across cohorts (loads `employer_checks.R`) | Bite report §4b |
| `timevarying_bite_check.R` | Time-varying bite of the retained DiD (AEWR minus current CZ p25): level slope, variance sources, components entered separately, region-year FE. Takes no output directory; prints to stdout | Bite report §5 |

The wild cluster bootstrap in `employer_robust.R` is a hand-rolled restricted
Webb bootstrap on fixed-effect-partialled residuals.
`fwildclusterboot::boottest` failed on these models.

The 2009–2010 reversal scripts were not copied. Their findings are in
`markdowns/wage_reversal_2009_2010_exploration.md`.

## Cross-design frozen-bite investigation

The detailed execution plan is
[`markdowns/bite_designs_agent_plan.md`](../../markdowns/bite_designs_agent_plan.md).
The generated finding is
[`markdowns/bite_designs_cross_design.md`](../../markdowns/bite_designs_cross_design.md).
This exploration reads cached inputs and leaves every retained design and
pipeline product unchanged. All numerical artifacts and figures go to an
explicit scratch directory outside the repository.

Run from the repository root in the configured environment:

```sh
export R_LIBS_USER=$(ls -d "$PWD"/renv/library/*/R-*/* | head -1)
bite_scratch=/tmp/h2a-bite-cross-design
mkdir -p "$bite_scratch"
Rscript --vanilla explorations/bite/cross_design_exposure.R "$bite_scratch"
Rscript --vanilla explorations/bite/cross_design_panel.R "$bite_scratch"
Rscript --vanilla explorations/bite/cross_design_border.R "$bite_scratch"
Rscript --vanilla explorations/bite/cross_design_diagnostics.R "$bite_scratch"
Rscript --vanilla explorations/bite/cross_design_placebos.R "$bite_scratch"
Rscript --vanilla explorations/bite/county_bite_acs.R "$bite_scratch" > "$bite_scratch/prior_bite.log"
Rscript --vanilla explorations/bite/partial_r2.R > "$bite_scratch/prior_panel.log"
Rscript --vanilla explorations/bite/test_cross_design.R > "$bite_scratch/numerical_tests.log"
Rscript --vanilla explorations/bite/cross_design_report.R "$bite_scratch"
Rscript --vanilla explorations/bite/validate_cross_design.R "$bite_scratch"
cp "$bite_scratch/bite_designs_cross_design.md" markdowns/bite_designs_cross_design.md
```

| Script | Ownership |
| --- | --- |
| `cross_design_helpers.R` | Shared scratch setup, key guards, CRV1 summaries, full-nuisance restricted Webb bootstrap, KP/SW diagnostics |
| `cross_design_exposure.R` | Frozen county table, early ACS and OEWS alternatives, distribution/level diagnostics, input hashes |
| `cross_design_panel.R` | Both control windows; four outcomes; OLS/IV models, three cluster partitions, AR sets, events and 500 OLS permutations |
| `cross_design_border.R` | Original/screened one-ring comparisons, pair exposure/adjacency, frozen-floor identity, inherited WCR and signed power |
| `cross_design_diagnostics.R` | Exposure-stratified donor co-movement and numerical MC unit/estimand reconciliation |
| `cross_design_placebos.R` | 500 primary-outcome permutations for each IV and border comparison |
| `cross_design_report.R` | Generated report, conditional average-effect conversion, summary tables and figures |
| `test_cross_design.R` | Synthetic bootstrap versus literal FE refits; KP scalar-Wald and invariance checks |
| `validate_cross_design.R` | Artifact coverage, identities, benchmark reproduction and unchanged-input checks |

The primary gap uses 2008/2010 real AEWR minus the maximum of real CZ p25
and the agricultural minimum wage. Alternatives use MC point-mass weights
in consistent real units, 2005–2007 ACS, baseline OEWS Big-Six hourly means,
and within-region ranks. The existing MC fraction and direct-bite builder
mixes nominal policy wages with real quantiles; the report diagnoses that
issue without changing its producer or retained results. The OEWS measure is
an area/occupation proxy, not an observed county farm-worker distribution.

Panel 1(b)/1(a) and region-year IV omit controls but use the corresponding
complete controlled-IV sample. OLS average-effect and two-endogenous IV models
include the declared BEA or continuing controls. Events retain observed 2009
outcomes without fabricating statutory wages. Border fits use 2011–2025 and
the branch's frozen memberships. Early exposure loses an entire border side;
its signed power is explicitly unavailable. Other signed power uses unchanged
branch seeds, resource guards, DGPs and size gates. Permutations are placebo
distributions, not assignment-based causal p-values.

The helper routes any automatic graphics device to scratch. The generated
report links to scratch figures and complete CSVs; keep that directory to
retain the evidence links. Integration proposals remain proposals in the
report and require a separately authorized empirical revision.
