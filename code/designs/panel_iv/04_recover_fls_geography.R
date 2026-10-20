# Recover deterministic county weights. Optional --years/--regions rebuild
# selected cells without changing other cells of a compatible R cache.
here::i_am("code/paths.R")
source(here::here("code", "paths.R"))
source(path_code("c00_shared", "geography.R"))
source(path_code("designs", "panel_iv", "io.R"))
source(path_code("designs", "panel_iv", "calibration.R"))

arguments <- commandArgs(trailingOnly = TRUE)
options <- list(years = IV_SOURCE_YEARS, regions = NULL)
while (length(arguments)) {
  flag <- arguments[1]
  if (!flag %in% c("--years", "--regions")) stop("Unknown option: ", flag)
  next_flag <- which(startsWith(arguments[-1], "--"))
  count <- if (length(next_flag)) next_flag[1] else length(arguments)
  if (count < 2L) stop("Missing values for ", flag)
  values <- arguments[2:count]
  if (any(!grepl("^[0-9]+$", values))) stop("Invalid values for ", flag)
  options[[substring(flag, 3)]] <- values
  arguments <- if (count < length(arguments)) arguments[(count + 1L):length(arguments)] else character()
}
years <- sort(unique(as.integer(options$years)))
if (!length(years) || any(!years %in% IV_SOURCE_YEARS)) stop("Supported source years are 2010--2021")
regions <- options$regions
if (!is.null(regions)) {
  regions <- normalize_fixed_width_code(regions, 2L, "AEWR region")
  regions <- sub("^0", "", regions)
  if (any(!regions %in% as.character(1:17))) stop("Supported AEWR regions are 1--17")
  regions <- sort(unique(regions))
}
annual <- iv_annual_targets(iv_read(path_int("fls_region.parquet")), years)
quarterly <- iv_paired_targets(iv_read(path_int("fls_region_quarterly_workers.parquet")),
  iv_read(path_int("fls_region_quarterly_wages.parquet")), years)
features <- iv_build_features(iv_read(path_int("panel_iv_fls_frame.parquet")),
  iv_read(path_processed("county_year_panel.parquet")), iv_read(path_int("qcew_county_ag_quarterly_employment.parquet")),
  annual, quarterly, years, regions)
expected <- length(years) * if (is.null(regions)) 17L else length(regions)
if (length(features$cells) != expected) stop("Incomplete county recovery grid")
recovery <- iv_recover_cells(features$cells)
outputs <- list(panel_iv_fls_county_features = features$features,
  panel_iv_fls_county_weight_summary = recovery$weights,
  panel_iv_fls_county_calibration_diagnostics = recovery$calibration_diagnostics,
  panel_iv_fls_county_moment_diagnostics = recovery$moment_diagnostics)
keys <- list(c("aewr_region_id", "source_year", "county_fips", "moment_id"),
  c("aewr_region_id", "source_year", "specification", "county_fips"),
  c("aewr_region_id", "source_year", "specification"),
  c("aewr_region_id", "source_year", "specification", "moment_id"))
# Check every replacement before writing any file.
for (i in seq_along(outputs)) {
  outputs[[i]]$calibration_version <- IV_CALIBRATION_VERSION
  outputs[[i]] <- iv_merge_selected(path_int(paste0(names(outputs)[i], ".parquet")),
    outputs[[i]], years, regions, keys[[i]])
}
for (i in seq_along(outputs)) iv_atomic_parquet(outputs[[i]], path_int(paste0(names(outputs)[i], ".parquet")))
message("Wrote deterministic calibration for ", expected, " region-year cells")
