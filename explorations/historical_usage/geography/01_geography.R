# Run from the repository root. Prints aggregate results to stdout.
source("code/paths.R")
source(path_code("c00_shared", "geography.R"))
library(data.table)
library(Matrix)
setDTthreads(2L)

cols <- c("county_fips", "state_fips", "year", "nbr_workers_certified_start_year",
          "nbr_workers_certified_fiscal_year", "man_hours_certified_start_year",
          "census_cropland_2007", "emp_farm_2011", "h2a_predicted_share_2011")
panel <- as.data.table(arrow::read_parquet(path_processed("county_year_panel.parquet"), col_select = tidyselect::all_of(cols)))
edges <- as.data.table(arrow::read_parquet(path_int("county_adjacency2010.parquet"),
  col_select = c("county_fips", "neighbor_county_fips")))
assert_geo_columns(panel, c("county_fips", "state_fips"))
assert_geo_columns(edges, c("county_fips", "neighbor_county_fips"))
setorder(panel, county_fips, year)
stopifnot(!anyDuplicated(panel[, .(county_fips, year)]), !anyDuplicated(edges),
          all(panel[, identical(as.integer(year), 2008:2025), by = county_fips]$V1),
          all(is.finite(panel$nbr_workers_certified_start_year)),
          all(is.finite(panel$nbr_workers_certified_fiscal_year)),
          all(is.finite(panel$man_hours_certified_start_year)),
          min(panel$nbr_workers_certified_start_year) >= 0)
counties <- panel[year == 2011]
county_ids <- counties$county_fips
years <- 2008:2025
positions <- matrix(panel$nbr_workers_certified_start_year, ncol = length(years), byrow = TRUE)
fiscal <- matrix(panel$nbr_workers_certified_fiscal_year, ncol = length(years), byrow = TRUE)
hours <- matrix(panel$man_hours_certified_start_year, ncol = length(years), byrow = TRUE)
edges <- edges[county_fips != neighbor_county_fips & county_fips %in% county_ids &
               neighbor_county_fips %in% county_ids]
graph <- sparseMatrix(i = match(edges$county_fips, county_ids),
                      j = match(edges$neighbor_county_fips, county_ids), x = 1,
                      dims = rep(length(county_ids), 2L))
stopifnot(isSymmetric(graph))
degree <- rowSums(graph)
cat("Calendar coverage and positive start-year position allocations\n")
print(data.table(year = years, counties = nrow(positions), positive = colSums(positions > 0),
                 at_least_ten = colSums(positions >= 10), positions = round(colSums(positions))))
cat("Isolated counties within panel:", sum(degree == 0), "\n")

# Remove the possibility that two neighboring counties share a single case.
# Every positive allocation in a retained case must name the same exact county.
allocation_cols <- c("case_number", "fiscal_year", "county_fips_list", "period_begin_date",
                     "adjusted_nbr_workers_certified")
sites <- as.data.table(arrow::read_parquet(path_int("h2a_allocated_worksites.parquet"),
                                           col_select = tidyselect::all_of(allocation_cols)))
audit <- as.data.table(arrow::read_parquet(path_int("h2a_case_allocation_audit.parquet"),
  col_select = c("case_number", "fiscal_year", "case_allocation_reconciled")))
stopifnot(!anyDuplicated(audit[, .(case_number, fiscal_year)]))
sites <- sites[audit[(case_allocation_reconciled)], on = .(case_number, fiscal_year), nomatch = NULL]
sites <- sites[is.finite(adjusted_nbr_workers_certified) & adjusted_nbr_workers_certified > 0]
sites[, footprint := {
  if (all(county_fips_list %in% county_ids)) {
    if (uniqueN(county_fips_list) == 1L) "one" else "several"
  } else "unresolved"
}, by = .(fiscal_year, case_number)]
places <- sites[footprint == "one", .(county_fips = county_fips_list,
  year = as.integer(substr(period_begin_date, 1, 4)), value = adjusted_nbr_workers_certified)]
places <- places[year %in% years, .(value = sum(value)), by = .(county_fips, year)]
assert_geo_columns(places, "county_fips")
strict <- matrix(0, nrow = nrow(positions), ncol = ncol(positions))
strict[cbind(match(places$county_fips, county_ids), match(places$year, years))] <- places$value
stopifnot(all(strict >= 0), all(strict <= positions + 1e-6))
multi <- sites[footprint == "several", .(value = sum(adjusted_nbr_workers_certified)),
               by = .(year = as.integer(substr(period_begin_date, 1, 4)))]
multi <- multi[year %in% years]
multi_total <- numeric(length(years))
multi_total[match(multi$year, years)] <- multi$value
unresolved_total <- colSums(positions - strict) - multi_total
stopifnot(all(is.finite(multi_total)), min(unresolved_total) >= -1e-6)
cat("Single-county case sensitivity coverage\n")
print(data.table(year = years, percent_positions = 100 * colSums(strict) / colSums(positions),
                 at_least_ten = colSums(strict >= 10), multiple_resolved = round(multi_total),
                 residual = round(unresolved_total)))

# Classify baseline participation once, before measuring later growth.
baseline <- years <= 2010
late <- years >= 2023
early_any <- rowSums(positions[, baseline] > 0) > 0
for (measure in c("start_positions", "start_hours", "fiscal_positions", "single_county_positions")) {
  value <- switch(measure, start_positions = positions, start_hours = hours, fiscal_positions = fiscal,
                  single_county_positions = strict)
  early_group <- if (measure == "start_hours") early_any else rowSums(value[, baseline] > 0) > 0
  totals <- data.table(early_group, start = rowMeans(value[, baseline]), end = rowMeans(value[, late]))
  sums <- totals[, .(counties = .N, early = sum(start), late = sum(end)), by = early_group]
  sums[, net_growth := late - early]
  sums[, growth_share := net_growth / sum(net_growth)]
  stopifnot(isTRUE(all.equal(sum(sums$net_growth), mean(colSums(value[, late])) -
                                      mean(colSums(value[, baseline])))))
  cat("Growth decomposition for", measure, "\n")
  print(sums)
}
for (early_years in list(2008:2010, 2009:2010)) {
  early <- years %in% early_years
  footprint <- data.table(
    early_group = rowSums(positions[, early] > 0) > 0,
    single_county = rowMeans(strict[, late]) - rowMeans(strict[, early]),
    other = rowMeans((positions - strict)[, late]) - rowMeans((positions - strict)[, early]))
  totals <- footprint[, .(counties = .N, single_county = sum(single_county), other = sum(other)),
                      by = early_group]
  delta <- sum(totals$single_county + totals$other)
  totals[, group_share := (single_county + other) / delta]
  stopifnot(isTRUE(all.equal(delta, mean(colSums(positions[, late])) - mean(colSums(positions[, early])))))
  cat("Growth by all-position baseline county group and recorded case geography; baseline",
      paste(early_years, collapse = ","), "\n")
  print(totals)
  cat("Fraction of total net growth outside single-county subset:", sum(totals$other) / delta, "\n")
  cat("Net gain fractions for single resolved, multiple resolved, and remaining positions:\n")
  print(sapply(list(single = colSums(strict), multiple = multi_total, remaining = unresolved_total),
               function(total) (mean(total[late]) - mean(total[early])) / delta))
}

# Forecast first recorded crossing after 2008. Lagged state prevalence uses all
# other counties, including counties no longer eligible for their own first crossing.
county_state <- counties$state_fips
state_rows <- split(seq_along(county_ids), county_state)
train_years <- 2012:2018
test_years <- 2019:2025
set.seed(20251120)
# Generate each placebo by relocating full neighbor-exposure histories among focal
# counties within a state. This preserves the state-year exposure distribution.
perms <- replicate(40L, {
  ord <- seq_along(county_ids)
  for (idx in state_rows) ord[idx] <- idx[sample.int(length(idx))]
  ord
})

score_forecasts <- function(data, predictions) {
  y <- data$event
  stopifnot(length(predictions) == length(y), all(is.finite(predictions)))
  p <- pmin(1 - 1e-12, pmax(1e-12, predictions))
  n1 <- sum(y)
  n0 <- length(y) - n1
  data.table(rows = length(y), events = n1, avg_prediction = mean(p),
             brier = mean((y - p)^2),
             log_loss = -mean(y * log(p) + (1 - y) * log1p(-p)),
             auc = if (n1 > 0 && n0 > 0) (sum(rank(p)[y == 1]) - n1 * (n1 + 1) / 2) / (n1 * n0) else NA_real_)
}

for (target in c("any_position", "ten_positions", "ten_single_county_positions")) {
  active <- switch(target, any_position = positions > 0, ten_positions = positions >= 10,
                   ten_single_county_positions = strict >= 10)
  first <- max.col(active, ties.method = "first")
  first[rowSums(active) == 0] <- length(years) + 1L
  neighbor <- as.matrix(graph %*% (active * 1)) / pmax(degree, 1)
  state_rate <- matrix(NA_real_, nrow = nrow(active), ncol = ncol(active))
  for (idx in state_rows) {
    other <- if (length(idx) > 1L) idx else seq_along(county_ids)
    state_rate[idx, ] <- (matrix(colSums(active[other, , drop = FALSE]),
                                nrow = length(idx), ncol = length(years), byrow = TRUE) -
                               active[idx, , drop = FALSE]) / (length(other) - 1L)
  }
  at_risk <- rbindlist(lapply(c(train_years, test_years), function(y) {
    t <- match(y, years)
    idx <- which(first >= t & degree > 0)
    data.table(county = idx, state = county_state[idx], year = y, prior = t - 1L,
               event = as.integer(first[idx] == t), state_lag = state_rate[idx, t - 1L],
               farm = log1p(counties$emp_farm_2011[idx]),
               crop = log1p(counties$census_cropland_2007[idx]),
               climate_score = log(counties$h2a_predicted_share_2011[idx]),
               neighbor_lag = neighbor[idx, t - 1L])
  }))
  # Rows with missing agricultural covariates are excluded from all compared fits.
  cat("Forecast target", target, "before covariate exclusions\n")
  print(at_risk[, .(rows = .N, events = sum(event)), by = .(test = year %in% test_years)])
  at_risk <- at_risk[is.finite(farm) & is.finite(crop) & is.finite(climate_score)]
  tr <- at_risk[year %in% train_years]
  te <- at_risk[year %in% test_years]
  base <- event ~ farm + crop + climate_score + state_lag
  spatial <- update(base, . ~ . + neighbor_lag)
  fit <- function(formula, training, test) {
    m <- glm(formula, family = binomial(), data = training)
    stopifnot(m$converged, !anyNA(coef(m)))
    predict(m, newdata = test, type = "response")
  }
  p0 <- fit(base, tr, te)
  p1 <- fit(spatial, tr, te)
  score_all <- rbind(baseline = score_forecasts(te, p0), adjacent = score_forecasts(te, p1), idcol = "model")
  print(score_all)
  print(te[, .(n = .N, events = sum(event), observed = mean(event), mean_forecast = mean(p1[.I]),
               improvement_brier = mean((event - p0[.I])^2 - (event - p1[.I])^2)), by = year])
  te[, band := cut(neighbor_lag, c(-Inf, 1 / 3, 2 / 3, Inf),
                   labels = c("low", "middle", "high"))]
  print(te[, .(rows = .N, events = sum(event), observed = mean(event)), by = band])
  null_gains <- apply(perms, 2, function(idx) {
    tr_p <- copy(tr)
    te_p <- copy(te)
    tr_p[, neighbor_lag := neighbor[cbind(idx[county], prior)]]
    te_p[, neighbor_lag := neighbor[cbind(idx[county], prior)]]
    shuffled <- fit(spatial, tr_p, te_p)
    mean((te$event - p0)^2 - (te$event - shuffled)^2)
  })
  gains <- (te$event - p0)^2 - (te$event - p1)^2
  loo <- sapply(unique(te$state), function(st) mean(gains[te$state != st]))
  cat("Training observations/events:", nrow(tr), sum(tr$event),
      "\nTrue Brier gain, range after deleting any one test state and within-state placebo 5/50/95th percentiles:\n")
  print(c(observed = mean(gains), min_loo = min(loo), max_loo = max(loo),
          quantile(null_gains, c(.05, .5, .95))))
}
