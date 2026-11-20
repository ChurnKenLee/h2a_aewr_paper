# POST-OUTCOME (not declared in registry.json). Identification search, strategy A1.
# Written after 05_estimation.R returned a first-stage F of 0.10, to describe the
# first-stage frame. Descriptive only: no test and no change to any declared quantity.
# PPML-weighted mean FY2025-to-FY2026 change in the county offered wage (omega),
# the predicted U.S.-rate binding-floor change d1_u = b x (F^U - e25) and the
# unweighted effective-floor change (F^U - e25), overall and by the state's
# within-state binding half (b_half, fixed at design: hours-weighted state median of b).
#
# Usage (repository root, after 05_estimation.R):
#   Rscript --vanilla explorations/identification_search/A1/07_post_outcome_first_stage_descriptive.R "$S"
# Writes $S/A1/a1_post_outcome_first_stage_descriptive.csv.
source("explorations/identification_search/A1/03_helpers.R")
a1_check_registry()
args <- commandArgs(trailingOnly = TRUE)
S <- normalizePath(args[1]); out <- a1_out_dir(S)
fs <- read_csv(file.path(out, "a1_first_stage_frame.csv"), col_types = cols(county_fips = "c", state_fips = "c"))
doses <- a1_doses(S) %>% select(county_fips, b, b_half, d1_u, eff_cut_u)
ch <- fs %>% select(county_fips, state_fips, year, omega, ppml_w) %>%
  tidyr::pivot_wider(names_from = year, values_from = omega, names_prefix = "omega_") %>%
  mutate(d_omega = omega_2026 - omega_2025) %>% left_join(doses, by = "county_fips")
summ <- function(d, grp) d %>% summarise(group = grp, counties = n(), weighted_mean_b = weighted.mean(b, ppml_w),
  weighted_mean_omega_2025 = weighted.mean(omega_2025, ppml_w), weighted_mean_omega_2026 = weighted.mean(omega_2026, ppml_w),
  weighted_mean_d_omega = weighted.mean(d_omega, ppml_w), weighted_mean_d1_u = weighted.mean(d1_u, ppml_w),
  weighted_mean_eff_cut_u = weighted.mean(eff_cut_u, ppml_w))
res <- bind_rows(summ(ch, "all"), summ(ch %>% filter(b_half == "hi"), "within-state binding half: hi"),
  summ(ch %>% filter(b_half == "lo"), "within-state binding half: lo")) %>%
  mutate(label = "POST-OUTCOME descriptive (not in registry)")
a1_write(res, out, "a1_post_outcome_first_stage_descriptive.csv"); print(as.data.frame(res), digits = 4)
