here::i_am("code/paths.R")
source(here::here("code", "paths.R"))
source(path_code("designs", "panel_iv", "design.R"))
source(path_code("designs", "panel_iv", "specification.R"))
spec <- panel_iv_read_specification(path_code("designs", "panel_iv", "estimation_spec.json"))
stopifnot(nrow(spec$outcomes) == 12L, identical(spec$fixed_effects, c("county_fips", "year")),
  spec$primary_inference_cluster == "aewr_iv_cluster_id",
  spec$instrument == "z_wage_seasonal_composition_real",
  identical(spec$policy_years, c(2011L, 2023L)),
  identical(spec$controls, DISSIMILARITY_IV_CONTROL_TERMS),
  identical(spec$models$controls, c(FALSE, TRUE)), spec$preferred_column == 2L)
panel <- data.frame(year = 2011:2013, farm_cashandinc_ppi = c(2, 3, NA), emp_farm = c(4, 0, 2),
  nbr_employers_balanced_start_year = c(8, 9, NA), emp_farm_2011 = c(2, -1, 2))
result <- panel_iv_prepare_outcomes(panel, spec)
stopifnot(identical(result$farm_cashandinc_ppi_per_farm_worker, c(500, NA_real_, NA_real_)),
  identical(result$h2a_linked_worksite_businesses_per_farm_worker_2011_start_year, c(4, NA_real_, NA_real_)))
spec$derived_outcomes$farm_cashandinc_ppi_per_farm_worker$scale <- 1L
stopifnot(inherits(tryCatch(panel_iv_prepare_outcomes(panel, spec), error = identity), "error"))
message("Specification tests passed")
