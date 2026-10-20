here::i_am("code/paths.R")
source(here::here("code", "paths.R"))
source(path_code("c00_shared", "geography.R"))
source(path_code("designs", "panel_iv", "io.R"))
source(path_code("designs", "panel_iv", "calibration.R"))
source(path_code("designs", "panel_iv", "frame.R"))
fails <- function(expression) stopifnot(inherits(tryCatch({ force(expression); NULL }, error = identity), "error"))

basis <- iv_helmert(4L)
stopifnot(max(abs(crossprod(basis) - diag(3))) < 1e-12, max(abs(colSums(basis))) < 1e-12,
  identical(dim(iv_helmert(3L)), c(3L, 2L)))
prior <- c(.2, .3, .5)
equal_rates <- (14.25 - 14.25) * c(100, 20, 5) + (14.25 - 14.25) * c(10, 40, 8)
stopifnot(!iv_standardize(equal_rates, 0, prior)$active)
rounded <- (12.34 - 12.34) * c(100, 20, 5) + (12.35 - 12.34) * c(10, 40, 8)
stopifnot(iv_standardize(rounded, 0, prior)$active, max(abs(rounded - .01 * c(10, 40, 8))) < 1e-12)
design <- cbind(c(-1, 0, 1), c(.5, -1, .4))
solution <- iv_solve_entropy(prior, design, as.numeric(crossprod(prior, design)))
stopifnot(solution$success, max(abs(solution$weights[1, ] - prior)) < 1e-12)
shifted <- iv_solve_entropy(prior, design, c(.5, -.2))
stopifnot(shifted$success, abs(sum(shifted$weights) - 1) < 1e-12,
  shifted$calibrated_imbalance_norm <= shifted$input_imbalance_norm + 1e-9)
fails(iv_solve_entropy(c(-1, 1, 1), design, c(0, 0)))
fails(iv_solve_entropy(prior, design * NA_real_, c(0, 0)))
fails(iv_impute_under_prior(c(1, NA), c(FALSE, FALSE), c(.5, .5)))
fails(iv_require_unique(data.frame(county_fips = c("01001", "01001")), "county_fips", "duplicate"))
fails(assert_geo_columns(data.frame(county_fips = 1001L), "county_fips"))

moment <- c(list(moment_id = "test", moment_family = "annual_wage", raw_values = c(2, 4, 9),
  raw_target = 3, observed_prior_mass = 1, quarter = NA_character_), iv_standardize(c(2, 4, 9), 3, c(.4, .6, 0)))
cell <- list(region = "1", year = 2010L, county_codes = c("01001", "01003", "01005"),
  prior = c(.4, .6, 0), moments = list(moment))
recovered <- iv_recover_cells(list(cell))
stopifnot(all(recovered$weights$calibrated_center_weight[recovered$weights$county_fips == "01005"] == 0),
  !any(grepl("draw|simulation|kappa", names(recovered$weights))))
original_guard <- IV_MAX_ITERATIONS
IV_MAX_ITERATIONS <- 0L
fails(iv_recover_cells(list(cell)))
IV_MAX_ITERATIONS <- original_guard

# Strict QWI completeness: duplicate, suppressed, and absent quarterly cells
# cannot silently become a complete annual employment signal.
qwi <- expand.grid(county_fips = "01001", year = 2010L, qtr = 1:4,
  industry_code = c("111", "112"), stringsAsFactors = FALSE)
qwi$qwi_beginning_quarter_employment <- 10
stopifnot(iv_qwi_annual(qwi)$qwi_strict_complete, iv_qwi_annual(qwi)$qwi_ag_employment == 20)
stopifnot(!iv_qwi_annual(rbind(qwi, qwi[1, ]))$qwi_strict_complete)
stopifnot(!iv_qwi_annual(qwi[-1, ])$qwi_strict_complete)
qwi$qwi_beginning_quarter_employment[1] <- NA_real_
stopifnot(!iv_qwi_annual(qwi)$qwi_strict_complete)

# Explicit source choices survive reordering of duplicate-domain values only
# when their recorded source rows and source fingerprint still match.
source_path <- tempfile("iv-census-source-", tmpdir = "/tmp")
writeLines("synthetic Census source", source_path)
farms <- data.frame(source_county_fips = c("01001", "01001", "01003"), year = 2007L,
  census_eligible_farms = c(72, 47, 5))
selection <- farms[c(2, 3), ]; selection$source_row <- c(2L, 3L)
metadata <- list(sha256 = digest::digest(file = source_path, algo = "sha256"), source_rows = 3L, selected_rows = 2L)
stopifnot(identical(iv_select_farm_floor(farms, selection, metadata, source_path)$census_eligible_farms, c(47, 5)))
fails(iv_select_farm_floor(farms[c(2, 1, 3), ], selection, metadata, source_path))
writeLines("refreshed source", source_path)
fails(iv_select_farm_floor(farms, selection, metadata, source_path))
unlink(source_path)

# Missing benchmarks obey the retained state residual rule. Structural zeros
# stay zero, intervening years use the preceding benchmark, and the selected
# farm floor is added only after employment projection.
counties <- data.frame(county_fips = c("01001", "01003", "01005"), state_fips = "01",
  state_abbrev = "AL", aewr_region_id = "1")
census <- tidyr::expand_grid(county_fips = counties$county_fips, year = IV_BENCHMARK_YEARS)
census$census_hired_workers_total <- c(10, NA, 30, 40, rep(20, 4), rep(NA_real_, 4))
state <- data.frame(state_fips = "01", year = IV_BENCHMARK_YEARS,
  state_census_hired_workers_reported = c(30, 42, 50, 60))
farms <- census[c("county_fips", "year")]
farms$census_eligible_farms <- ifelse(farms$county_fips == "01005", 0, 2)
qcew <- tidyr::expand_grid(county_fips = counties$county_fips, year = IV_FRAME_YEARS)
for (sector in c("crop", "animal")) {
  qcew[[paste0("qcew_", sector, "_sector_annual_avg_emplvl")]] <- ifelse(qcew$county_fips == "01005", 0, 10)
  qcew[[paste0("qcew_", sector, "_sector_disclosed")]] <- TRUE
}
bea <- qcew[c("county_fips", "year")]; bea$emp_farm <- 0; bea$emp_farm_propr <- 0
frame <- iv_build_frame(counties, census, state, farms, qcew, qwi, bea)
benchmark <- frame[frame$county_fips == "01001" & frame$source_year == 2012L, ]
stopifnot(abs(benchmark$census_hired_workers_benchmark_filled - 22) < 1e-12,
  abs(benchmark$frame_employment_mass - 24) < 1e-12,
  grepl("county_log1p_interpolation_state_residual_scaled", benchmark$census_benchmark_fill_method),
  all(frame$frame_employment_mass[frame$county_fips == "01005"] == 0),
  all(frame$state_rake_factor == 1), all(frame$two_sided_log1p_drift == 0),
  is.integer(frame$source_year),
  abs(frame$frame_employment_mass[frame$county_fips == "01001" & frame$source_year == 2011L] - 12) < 1e-12)
fails(iv_build_frame(counties, census, state[-1, ], farms, qcew, qwi, bea))
message("Calibration and frame guard tests passed")
