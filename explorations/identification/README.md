# Alternative identification explorations

Exploratory R scripts behind
`markdowns/alternative_identification_exploration.md`, which executes
`markdowns/alternative_identification_agent_plan.md`. They are not pipeline
stages. They read retained artifacts and write only to the scratch directory
given as the first argument, which must be outside the repository. Nothing
here selects or changes a retained specification, table or producer.

Run from the repository root:

```sh
export R_LIBS_USER=$(ls -d "$PWD"/renv/library/*/R-*/* | head -1)
id_scratch=/path/outside/repo/identification   # any directory outside the repo
mkdir -p "$id_scratch"
Rscript --vanilla explorations/identification/00_fls_revision.R "$id_scratch"
Rscript --vanilla explorations/identification/01_common_unit.R "$id_scratch" /tmp/h2a-bite-cross-design
Rscript --vanilla explorations/identification/02_precision.R "$id_scratch"
Rscript --vanilla explorations/identification/03_track1_support.R "$id_scratch"
Rscript --vanilla explorations/identification/04_track1_power.R "$id_scratch"
Rscript --vanilla explorations/identification/04b_track1_preperiod.R "$id_scratch"
Rscript --vanilla explorations/identification/04c_track1_amendment_power.R "$id_scratch"
Rscript --vanilla explorations/identification/05_track2_minwage.R "$id_scratch"
# Track 2 on the corrected coverage coding: pass a state_real_minwages.parquet
# built by the current code/c01_clean/04_state_minimum_wages.R (writes track2c_*.csv)
Rscript --vanilla explorations/identification/05_track2_minwage.R "$id_scratch" /path/to/state_real_minwages.parquet
Rscript --vanilla explorations/identification/06_track3_multiregion.R "$id_scratch"
Rscript --vanilla explorations/identification/07_synthesis.R "$id_scratch" /tmp/h2a-bite-cross-design
Rscript --vanilla explorations/identification/08_track1_estimate.R "$id_scratch"   # reads FY2026; needs 04 outputs
Rscript --vanilla explorations/identification/09_track4_overtime.R "$id_scratch"
# employer decomposition of the Track 1 wage first stage; third argument (optional) is the scratch directory holding 04 and 08 outputs
Rscript --vanilla explorations/identification/10_track1_employer_decomposition.R "$id_scratch" 2025 "$id_scratch"   # placebo year, reads nothing after FY2025
Rscript --vanilla explorations/identification/10_track1_employer_decomposition.R "$id_scratch" 2026 "$id_scratch"   # reads FY2026; hash-checked
git checkout -- Rplots.pdf
```

`01_common_unit.R` and `07_synthesis.R` take an optional second argument: the
scratch directory of the earlier cross-design bite run
(`explorations/bite/README.md`), which holds `exposures.parquet`,
`average_effect_conversion.csv`, `border_estimates.csv` and `iv_ar_sets.csv`.
Run 01 and 02, 05 and 06 before 07.

| Script | What it does | Report section |
| --- | --- | --- |
| `id_helpers.R` | Loads `explorations/bite/cross_design_helpers.R`; adds FY2026-safe case readers, elasticity rows, a cluster-agnostic wrapper of the restricted Webb bootstrap, and the region-year residual-variance share | all |
| `00_fls_revision.R` | FLS preliminary vs revised field-and-livestock wage; release provenance; AEWR equals preliminary | §2a |
| `01_common_unit.R` | Refits plain panel, panel IV and bite 1(a)/1(b); converts them, the border, statewide-border power and employer design 4 to elasticities | §1 |
| `02_precision.R` | Phase 0 §4b: baseline, PPML (and control-function PPML-IV), farm-employment weights, region-year aggregate, baseline-employer panels; region-year residual-variance shares | §2b |
| `03_track1_support.R` | Track 1 support: within-former-region dispersion of the 2025 statewide rates, single-state regions, pre-period decision-date coverage | §2c, §3 |
| `track1_registry.json` | Track 1 declared design (frozen before any FY2026 outcome is read); its SHA-256 is in the report | §3 |
| `04_track1_power.R` | Track 1 design-stage power on all counties of multi-state former regions; Q1–Q3 pre-period outcomes rebuilt with the b01 fiscal-year logic and validated against `h2a_data_year` | §3 |
| `04b_track1_preperiod.R` | Registry-declared pre-period diagnostics, run after the registry was frozen: dose × FY event study FY2012–FY2025, linear drift and its implied static-DiD bias, FY2025-growth warning check, state-year variance share | §3 |
| `04c_track1_amendment_power.R` | Power for two candidate amendments prompted by 04b (state linear trends; FY2025-vs-FY2026 contrast); same DGPs, seed and gate as 04 | §3 |
| `05_track2_minwage.R` | Track 2: agricultural-exemption audit, support, four minimum-wage specifications, leads, event study, relative-price test and translation | §4 |
| `06_track3_multiregion.R` | Track 3: multi-region employer support and within-employer regional allocation on the AEWR | §5 |
| `07_synthesis.R` | Track 5: elasticity intervals and exclusions by design | §7 |
| `track1_registry_amendment1.json` | Amendment 1 (user decision 2026-11-02, frozen before any FY2026 read): FY2025-vs-FY2026 contrast primary, state jackknife primary inference, base registry and state trends as sensitivities; records the user's legal-status confirmation | §3 |
| `08_track1_estimate.R` | Track 1 post-policy estimation: hash-checks both registries, rebuilds FY2012–FY2026 Q1–Q3 outcomes (asserts equality with the frozen pre panel), primary, placebo and every declared sensitivity with jackknife, CRV1, wild bootstrap and within-region randomization inference | §3 |
| `09_track4_overtime.R` | Track 4: frozen baseline-hours overtime exposure, support check, specifications (i)–(iii), lead check, state-schedule randomization inference | §6 |
| `10_track1_employer_decomposition.R` | Splits the Track 1 offered-wage first stage into within-employer, reweighting, entry and exit parts (exact county identity); continuer pass-through, exit and entry on the dose, a pre-change selection test and trimming bounds, for balanced and high-recall employer identifiers | `markdowns/track1_employer_decomposition.md` |
| `track1_employer_decomposition_registry.json`, `track1_employer_decomposition_registry.sha256` | Frozen registry for the decomposition (2026-11-13, after the placebo-year run and before the FY2026 run) and the hashes of the registry, the estimator, both helper files and the base Track 1 registry | same |
| `sources/state_ag_overtime_schedule.csv`, `sources/state_ag_overtime_notes.md` | Sourced state farm-worker overtime thresholds 2008–2027 (web research 2026-11-02; citation, URL and confidence per row) | §6 |

Scripts 00–07 and 09 read cases only through `id_read_worksites()` or
`id_read_disclosures()`, which filter `fiscal_year <= 2025` inside the Arrow
scan. `08_track1_estimate.R` reads FY2026, after the user's approval on
2026-11-02, and it refuses to run unless both Track 1 registries match their
recorded SHA-256 hashes. `10_track1_employer_decomposition.R` reads FY2026
only when its second argument is 2026, and then only if every hash in
`track1_employer_decomposition_registry.sha256` matches; changing the
estimator or a helper file therefore needs a dated amendment and new hashes.

The wild bootstrap is the hand-rolled restricted Webb bootstrap of
`explorations/bite/cross_design_helpers.R` (`fwildclusterboot::boottest`
fails on these models). `id_wild()` only swaps the cluster column. It
supports unweighted OLS/IV; weighted and PPML fits report CRV1 (PPML-IV uses a
region pairs bootstrap) and say so.
