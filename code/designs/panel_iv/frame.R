# Original rolling Census/QCEW/QWI/BEA prior. These functions preserve the
# implemented forward-growth path, including the historical benchmark fills.
IV_BENCHMARK_YEARS <- c(2007L, 2012L, 2017L, 2022L)
IV_FRAME_YEARS <- 2007:2022

iv_nonnegative <- function(x) is.finite(x) & x >= 0

# The former Polars sort was unstable among duplicate Census domains. Record
# its actual source-row choices explicitly so R preserves the original prior.
# A refreshed source must be reviewed rather than silently choosing new rows.
iv_select_farm_floor <- function(farms, selection, metadata, source_path) {
  keys <- c("source_county_fips", "year")
  fields <- c(keys, "census_eligible_farms")
  iv_require_columns(farms, fields, "Census farm candidates")
  iv_require_columns(selection, c(fields, "source_row"), "Census farm selection")
  iv_require_unique(selection, keys, "Census farm selection")
  if (!is.character(selection$source_county_fips) ||
      !identical(county_fips(selection$source_county_fips), selection$source_county_fips))
    stop("Census source county identifiers must be five-digit strings")
  if (!identical(digest::digest(file = source_path, algo = "sha256"), metadata$sha256) ||
      nrow(farms) != metadata$source_rows || nrow(selection) != metadata$selected_rows)
    stop("Census source changed; review the recorded farm-floor selection before rebuilding")
  positions <- selection$source_row
  if (anyNA(positions) || any(positions < 1 | positions > nrow(farms) | positions != floor(positions)) ||
      anyDuplicated(positions)) stop("Invalid Census source-row selection")
  chosen <- as.data.frame(farms[positions, fields])
  same_values <- (is.na(chosen$census_eligible_farms) & is.na(selection$census_eligible_farms)) |
    (!is.na(chosen$census_eligible_farms) & !is.na(selection$census_eligible_farms) &
      chosen$census_eligible_farms == selection$census_eligible_farms)
  if (!all(iv_key(chosen$source_county_fips, chosen$year) == iv_key(selection$source_county_fips, selection$year)) ||
      !all(same_values) || !setequal(iv_key(farms$source_county_fips, farms$year), iv_key(selection$source_county_fips, selection$year)))
    stop("Recorded farm-floor rows no longer match the Census source")
  iv_sort(chosen, keys)
}

iv_qcew_annual <- function(panel) {
  fields <- c("county_fips", "year", paste0("qcew_", rep(c("crop", "animal"), each = 2),
    "_sector_", rep(c("annual_avg_emplvl", "disclosed"), 2)))
  iv_require_columns(panel, fields, "Annual QCEW")
  data <- as.data.frame(panel[fields])
  names(data)[names(data) == "year"] <- "source_year"
  iv_require_unique(data, c("county_fips", "source_year"), "Annual QCEW")
  crop <- !is.na(data$qcew_crop_sector_disclosed) & data$qcew_crop_sector_disclosed &
    iv_nonnegative(data$qcew_crop_sector_annual_avg_emplvl)
  animal <- !is.na(data$qcew_animal_sector_disclosed) & data$qcew_animal_sector_disclosed &
    iv_nonnegative(data$qcew_animal_sector_annual_avg_emplvl)
  data.frame(data[c("county_fips", "source_year")],
    qcew_observed_cells = as.integer(!is.na(data$qcew_crop_sector_disclosed)) +
      as.integer(!is.na(data$qcew_animal_sector_disclosed)),
    qcew_valid_cells = as.integer(crop) + as.integer(animal),
    qcew_strict_complete = crop & animal,
    qcew_ag_employment = ifelse(crop & animal,
      data$qcew_crop_sector_annual_avg_emplvl + data$qcew_animal_sector_annual_avg_emplvl, NA_real_))
}

iv_qwi_annual <- function(qwi) {
  iv_require_columns(qwi, c("county_fips", "year", "qtr", "industry_code",
    "qwi_beginning_quarter_employment"), "Quarterly QWI")
  cells <- qwi |>
    dplyr::filter(industry_code %in% c("111", "112"), qtr %in% 1:4) |>
    dplyr::mutate(source_year = as.integer(year),
      valid = iv_nonnegative(qwi_beginning_quarter_employment)) |>
    dplyr::group_by(county_fips, source_year, qtr, industry_code) |>
    dplyr::summarise(valid_cell = dplyr::n() == 1L & sum(valid) == 1L,
      cell_employment = dplyr::first(ifelse(valid, qwi_beginning_quarter_employment, NA_real_)),
      .groups = "drop")
  annual <- cells |>
    dplyr::group_by(county_fips, source_year) |>
    dplyr::summarise(qwi_observed_cells = dplyr::n(), qwi_valid_cells = sum(valid_cell),
      qwi_observed_quarters = dplyr::n_distinct(qtr),
      qwi_observed_industries = dplyr::n_distinct(industry_code),
      employment = sum(ifelse(valid_cell, cell_employment, 0)), .groups = "drop") |>
    dplyr::mutate(qwi_strict_complete = qwi_observed_cells == 8L & qwi_valid_cells == 8L &
      qwi_observed_quarters == 4L & qwi_observed_industries == 2L,
      qwi_ag_employment = ifelse(qwi_strict_complete, employment / 4, NA_real_)) |>
    dplyr::select(-employment)
  iv_require_unique(annual, c("county_fips", "source_year"), "Annual QWI")
  annual
}

iv_bea_annual <- function(bea) {
  iv_require_columns(bea, c("county_fips", "year", "emp_farm", "emp_farm_propr"), "BEA jobs")
  data <- bea |>
    dplyr::filter(year %in% IV_FRAME_YEARS) |>
    dplyr::transmute(county_fips, source_year = as.integer(year),
      bea_hired_farm_jobs = ifelse(iv_nonnegative(emp_farm) & iv_nonnegative(emp_farm_propr),
        pmax(emp_farm - emp_farm_propr, 0), NA_real_))
  iv_require_unique(data, c("county_fips", "source_year"), "BEA jobs")
  data
}

iv_employment_updates <- function(counties, qcew, qwi, bea) {
  grid <- tidyr::expand_grid(county_fips = counties$county_fips, source_year = IV_FRAME_YEARS)
  for (data in list(iv_qcew_annual(qcew), iv_qwi_annual(qwi), iv_bea_annual(bea)))
    grid <- dplyr::left_join(grid, data, by = c("county_fips", "source_year"), relationship = "one-to-one")
  grid <- iv_sort(as.data.frame(grid), c("county_fips", "source_year"))
  for (name in c("qcew_strict_complete", "qwi_strict_complete")) grid[[name]][is.na(grid[[name]])] <- FALSE
  grid$annual_update_source <- "unavailable"
  grid$annual_update_employment <- NA_real_
  grid$annual_growth_source <- "unit_growth"
  grid$annual_log1p_growth <- 0
  first <- !duplicated(grid$county_fips)
  grid$annual_growth_source[first] <- "not_applicable"
  previous <- c(NA_integer_, seq_len(nrow(grid) - 1L))
  consecutive <- !first
  for (source in c("qcew", "qwi", "bea")) {
    value <- grid[[if (source == "bea") "bea_hired_farm_jobs" else paste0(source, "_ag_employment")]]
    valid <- iv_nonnegative(value)
    if (source != "bea") valid <- valid & grid[[paste0(source, "_strict_complete")]]
    selected <- grid$annual_update_source == "unavailable" & valid
    grid$annual_update_source[selected] <- source
    grid$annual_update_employment[selected] <- value[selected]
    selected <- consecutive & grid$annual_growth_source == "unit_growth" & valid & valid[previous]
    selected[is.na(selected)] <- FALSE
    grid$annual_growth_source[selected] <- source
    grid$annual_log1p_growth[selected] <- log1p(value[selected]) - log1p(value[previous[selected]])
  }
  grid$annual_growth_factor_log1p <- exp(grid$annual_log1p_growth)
  grid$extreme_annual_change <- !grid$annual_growth_source %in% c("not_applicable", "unit_growth") &
    abs(grid$annual_log1p_growth) > log(2)
  grid$qwi_annual_fallback_used <- grid$annual_update_source == "qwi"
  grid$bea_annual_fallback_used <- grid$annual_update_source == "bea"
  grid$unit_growth_fallback_used <- grid$annual_growth_source == "unit_growth"
  grid
}

iv_frame_flags <- function(row) {
  flags <- character()
  add <- function(condition, flag) if (isTRUE(condition)) flags <<- c(flags, flag)
  add(row$census_benchmark_reported, "census_reported")
  add(row$census_published_zero, "census_published_zero")
  add(!is.na(row$census_benchmark_fill_method) && !row$census_benchmark_reported, "census_benchmark_imputed")
  add(row$structural_zero, "structural_zero")
  for (name in c("qwi_annual_fallback", "bea_annual_fallback", "unit_growth_fallback"))
    add(row[[paste0(name, "_used")]], name)
  add(row$annual_update_source == "unavailable", "annual_update_unavailable")
  add(row$extreme_annual_change, "extreme_annual_change")
  add(row$nonnegative_floor_applied, "nonnegative_floor")
  if (length(flags)) paste(flags, collapse = "|") else "none"
}

iv_build_frame <- function(counties, census, state, farms, qcew, qwi, bea) {
  counties <- iv_sort(unique(as.data.frame(counties[c("county_fips", "state_fips", "state_abbrev", "aewr_region_id")])), "county_fips")
  iv_require_unique(counties, "county_fips", "County universe")
  assert_geo_columns(counties, c("county_fips", "state_fips", "aewr_region_id"))
  census <- as.data.frame(census[census$county_fips %in% counties$county_fips & census$year %in% IV_BENCHMARK_YEARS, ])
  state <- as.data.frame(state[state$state_fips %in% counties$state_fips & state$year %in% IV_BENCHMARK_YEARS, ])
  iv_require_unique(census, c("county_fips", "year"), "County Census")
  iv_require_unique(state, c("state_fips", "year"), "State Census")
  if (nrow(state) != length(unique(counties$state_fips)) * length(IV_BENCHMARK_YEARS) ||
      any(!iv_nonnegative(state$state_census_hired_workers_reported))) stop("Incomplete state Census benchmarks")
  updates <- iv_employment_updates(counties, qcew, qwi, bea)
  update_keys <- iv_key(updates$county_fips, updates$source_year)
  county_updates <- split(seq_len(nrow(updates)), updates$county_fips)
  bea_annual <- iv_bea_annual(bea)
  positive <- unique(c(census$county_fips[iv_nonnegative(census$census_hired_workers_total) & census$census_hired_workers_total > 0],
    qcew$county_fips[which((qcew$qcew_crop_sector_disclosed & iv_nonnegative(qcew$qcew_crop_sector_annual_avg_emplvl) & qcew$qcew_crop_sector_annual_avg_emplvl > 0) |
      (qcew$qcew_animal_sector_disclosed & iv_nonnegative(qcew$qcew_animal_sector_annual_avg_emplvl) & qcew$qcew_animal_sector_annual_avg_emplvl > 0))],
    qwi$county_fips[iv_nonnegative(qwi$qwi_beginning_quarter_employment) & qwi$qwi_beginning_quarter_employment > 0],
    bea_annual$county_fips[iv_nonnegative(bea_annual$bea_hired_farm_jobs) & bea_annual$bea_hired_farm_jobs > 0]))
  reported <- census[iv_nonnegative(census$census_hired_workers_total), ]
  reported_value <- setNames(reported$census_hired_workers_total, iv_key(reported$county_fips, reported$year))
  targets <- setNames(state$state_census_hired_workers_reported, iv_key(state$state_fips, state$year))
  state$aewr_region_id <- counties$aewr_region_id[match(state$state_fips, counties$state_fips)]
  regional_totals <- stats::aggregate(state_census_hired_workers_reported ~ aewr_region_id + year, state, sum)
  national_totals <- stats::aggregate(state_census_hired_workers_reported ~ year, state, sum)
  benchmarks <- dplyr::left_join(updates[updates$source_year %in% IV_BENCHMARK_YEARS, ], counties, by = "county_fips", relationship = "many-to-one")
  ratios <- list()
  for (level in c("state_fips", "aewr_region_id", "national")) {
    grouping <- c(if (level != "national") level, "source_year")
    denominator <- benchmarks |>
      dplyr::group_by(dplyr::across(dplyr::all_of(grouping))) |>
      dplyr::summarise(total = sum(annual_update_employment[iv_nonnegative(annual_update_employment)]), .groups = "drop")
    numerator <- switch(level, state_fips = state, aewr_region_id = regional_totals, national = national_totals)
    nkeys <- if (level == "national") as.character(numerator$year) else iv_key(numerator[[level]], numerator$year)
    dkeys <- if (level == "national") as.character(denominator$source_year) else iv_key(denominator[[level]], denominator$source_year)
    values <- numerator$state_census_hired_workers_reported[match(dkeys, nkeys)] / denominator$total
    ratios[[level]] <- setNames(values[denominator$total > 0], dkeys[denominator$total > 0])
  }
  filled <- list()
  for (i in seq_len(nrow(counties))) {
    county <- counties[i, ]; id <- county$county_fips
    history <- updates[county_updates[[id]], ]
    cumulative <- function(start, end) {
      if (start == end) return(0)
      value <- sum(history$annual_log1p_growth[history$source_year > min(start, end) & history$source_year <= max(start, end)])
      if (start < end) value else -value
    }
    observed <- sort(reported$year[reported$county_fips == id])
    for (year in IV_BENCHMARK_YEARS) {
      value <- unname(reported_value[iv_key(id, year)])
      is_reported <- is.finite(value)
      structural <- FALSE
      method <- "reported"
      if (!is_reported) {
        lower <- observed[observed < year]; upper <- observed[observed > year]
        if (!id %in% positive) {
          value <- 0; method <- "structural_zero"; structural <- TRUE
        } else if (length(lower) && length(upper)) {
          lower <- max(lower); upper <- min(upper); share <- (year - lower) / (upper - lower)
          value <- expm1((1 - share) * log1p(reported_value[iv_key(id, lower)]) + share * log1p(reported_value[iv_key(id, upper)]))
          method <- "county_log1p_interpolation"
        } else if (length(observed)) {
          neighbor <- observed[order(abs(observed - year), observed)][1]
          value <- expm1(log1p(reported_value[iv_key(id, neighbor)]) + cumulative(neighbor, year))
          method <- paste0("county_growth_projection_from_", neighbor)
          if (value < 0) { value <- 0; method <- paste0(method, "_nonnegative_floor") }
        } else {
          available <- history[iv_nonnegative(history$annual_update_employment), ]
          proxy_year <- year; proxy <- 1
          if (nrow(available)) {
            available <- available[order(abs(available$source_year - year), available$source_year), ]
            proxy_year <- available$source_year[1]; proxy <- available$annual_update_employment[1]
          }
          keys <- c(iv_key(county$state_fips, year), iv_key(county$aewr_region_id, year), as.character(year))
          for (j in seq_along(ratios)) {
            ratio <- unname(ratios[[j]][keys[j]])
            if (is.finite(ratio)) break
          }
          if (!is.finite(ratio)) stop("Missing Census/employment ratio")
          value <- ratio * proxy
          method <- c("state", "aewr_region", "national")[j]
          method <- paste0(method, "_census_to_employment_ratio")
          if (proxy_year != year) method <- paste0(method, "_nearest_employment_", proxy_year)
        }
      }
      filled[[length(filled) + 1L]] <- data.frame(county, year = year, value = max(value, 0),
        method = method, reported = is_reported, published_zero = is_reported && value == 0,
        structural_zero = structural, state_residual_factor = if (is_reported) 1 else NA_real_)
    }
  }
  filled <- dplyr::bind_rows(filled)
  state_year_keys <- iv_key(filled$state_fips, filled$year)
  for (key in unique(state_year_keys)) {
    indexes <- which(state_year_keys == key)
    cells <- filled[indexes, ]; target <- targets[key]; tolerance <- 1e-8 * max(1, target)
    residual <- target - sum(cells$value[cells$reported])
    if (residual < -tolerance) stop("County Census workers exceed state total")
    residual <- max(residual, 0); missing <- which(!cells$reported)
    eligible <- which(!cells$reported & !cells$structural_zero)
    initial <- sum(cells$value[missing])
    if (!length(missing)) {
      if (residual > tolerance) stop("State residual has no unreported counties")
    } else if (initial > 0) {
      cells$value[missing] <- cells$value[missing] * (residual / initial)
      cells$state_residual_factor[missing] <- residual / initial
      cells$method[missing] <- paste0(cells$method[missing], "_state_residual_scaled")
    } else if (residual > tolerance) {
      if (!length(eligible)) stop("Positive state residual violates structural zeros")
      employment <- updates$annual_update_employment[match(iv_key(cells$county_fips[eligible], cells$year[eligible]), update_keys)]
      total <- sum(employment[is.finite(employment) & employment > 0])
      for (j in seq_along(eligible)) {
        index <- eligible[j]; weighted <- total > 0 && is.finite(employment[j]) && employment[j] > 0
        cells$value[index] <- residual * if (weighted) employment[j] / total else 1 / length(eligible)
        cells$method[index] <- paste0(cells$method[index], if (weighted) "_state_residual_employment_allocated" else "_state_residual_equal_allocated")
      }
    } else {
      cells$value[missing] <- 0; cells$state_residual_factor[missing] <- 0
      cells$method[missing] <- paste0(cells$method[missing], "_zero_state_residual")
    }
    if (abs(sum(cells$value) - target) > max(1e-7, 1e-10 * max(abs(sum(cells$value)), abs(target)))) stop("Filled Census misses state total")
    filled[indexes, ] <- cells
  }
  iv_require_unique(farms, c("county_fips", "year"), "Selected Census farm floor")
  farm_keys <- iv_key(farms$county_fips, farms$year)
  county_benchmarks <- split(seq_len(nrow(filled)), filled$county_fips)
  result <- list()
  for (i in seq_len(nrow(counties))) {
    county <- counties[i, ]; id <- county$county_fips
    history <- updates[county_updates[[id]], ]
    benchmarks <- filled[county_benchmarks[[id]], ]
    for (j in seq_len(nrow(history))) {
      update <- history[j, ]; year <- update$source_year
      lower <- max(IV_BENCHMARK_YEARS[IV_BENCHMARK_YEARS <= year])
      baseline <- benchmarks[benchmarks$year == lower, ]
      cell <- benchmarks[benchmarks$year == year, ]
      projected <- expm1(log1p(baseline$value) + sum(history$annual_log1p_growth[history$source_year > lower & history$source_year <= year]))
      floored <- projected < 0; projected <- max(projected, 0)
      farm_index <- match(iv_key(id, lower), farm_keys)
      farm_mass <- if (is.na(farm_index)) 0 else farms$census_eligible_farms[farm_index]
      observed_value <- unname(reported_value[iv_key(id, year)])
      benchmark <- nrow(cell) == 1L
      row <- data.frame(county, source_year = as.integer(year), weight_spec = IV_PRIOR_SPEC,
        annual_update_spec = IV_UPDATE_SPEC,
        census_benchmark_status = if (!benchmark) "non_benchmark_year" else if (!is.finite(observed_value)) "suppressed_or_absent" else if (observed_value == 0) "reported_zero" else "reported_positive",
        census_benchmark_reported = benchmark && cell$reported,
        census_published_zero = benchmark && cell$published_zero,
        census_hired_workers_reported = observed_value,
        census_hired_workers_benchmark_filled = if (benchmark) cell$value else NA_real_,
        census_benchmark_fill_method = if (benchmark) cell$method else NA_character_,
        census_benchmark_prefill = if (benchmark) cell$value else NA_real_,
        census_benchmark_state_residual_factor = if (benchmark) cell$state_residual_factor else NA_real_,
        structural_zero = benchmark && cell$structural_zero,
        lower_census_benchmark_year = lower, upper_census_benchmark_year = lower,
        lower_census_hired_workers_filled = baseline$value, upper_census_hired_workers_filled = baseline$value,
        interval_share = 0)
      update_fields <- c("qcew_ag_employment", "qcew_strict_complete", "qcew_observed_cells", "qcew_valid_cells",
        "qwi_ag_employment", "qwi_strict_complete", "qwi_observed_cells", "qwi_valid_cells", "bea_hired_farm_jobs",
        "annual_update_employment", "annual_update_source", "annual_growth_source", "annual_log1p_growth", "annual_growth_factor_log1p")
      row <- cbind(row, update[update_fields])
      row$two_sided_log1p_drift <- 0; row$frame_employment_mass_prerake <- projected
      row$state_census_hired_workers_target <- 0; row$state_frame_employment_prerake <- 0; row$state_rake_factor <- 1
      row$frame_employment_mass <- projected + farm_mass
      for (field in c("qwi_annual_fallback_used", "bea_annual_fallback_used", "unit_growth_fallback_used", "extreme_annual_change")) row[[field]] <- update[[field]]
      row$nonnegative_floor_applied <- floored
      row$update_imputation_source <- if (benchmark) cell$method else update$annual_growth_source
      row$quality_flags <- iv_frame_flags(row)
      result[[length(result) + 1L]] <- row
    }
  }
  result <- iv_sort(dplyr::bind_rows(result), c("county_fips", "source_year"))
  for (name in c("qcew_observed_cells", "qcew_valid_cells", "qwi_observed_cells", "qwi_valid_cells"))
    result[[name]] <- as.integer(replace(result[[name]], is.na(result[[name]]), 0))
  iv_require_unique(result, c("county_fips", "source_year"), "Frame")
  if (any(!iv_nonnegative(result$frame_employment_mass))) stop("Invalid frame mass")
  result
}
