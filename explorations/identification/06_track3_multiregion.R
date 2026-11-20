# Track 3: employers placing workers in two or more AEWR regions in one year.
# Support (plan section 4c) and an employer x region x year allocation design.
# Identification remains region-year AEWR variation; this is a mechanism check,
# not an average-effect estimate. Reads only fiscal years <= 2025 through
# id_read_worksites(); writes only to the scratch directory in argument 1.
source("explorations/identification/id_helpers.R")
suppressPackageStartupMessages(library(lubridate))
out_dir <- id_output_dir()

# Geography: county -> state -> AEWR region from the shared panel ------------
panel <- read_parquet(path_processed("county_year_panel.parquet"),
  col_select = c("county_fips", "state_fips", "aewr_region_id", "year", "aewr_ppi",
    "fisher_index_ppi", "fisher_quantity_index", "emp_farm_2011"))
county_region <- panel %>% distinct(county_fips, state_fips, aewr_region_id)
stopifnot(!anyDuplicated(county_region$county_fips))
state_region <- panel %>% distinct(state_fips, aewr_region_id)
stopifnot(!anyDuplicated(state_region$state_fips))
region_year <- panel %>% group_by(aewr_region_id, year) %>%
  summarise(A = mean(aewr_ppi),
    # Region-year farm price and quantity indices: county means weighted by
    # fixed 2011 farm employment (controls, not treatments).
    price_index = weighted.mean(fisher_index_ppi, coalesce(emp_farm_2011, 0) * is.finite(fisher_index_ppi), na.rm = TRUE),
    quantity_index = weighted.mean(fisher_quantity_index, coalesce(emp_farm_2011, 0) * is.finite(fisher_quantity_index), na.rm = TRUE),
    .groups = "drop") %>%
  mutate(across(c(price_index, quantity_index), ~ ifelse(is.finite(.x), .x, NA_real_))) %>%
  arrange(aewr_region_id, year) %>% group_by(aewr_region_id) %>%
  mutate(A_lead = lead(A)) %>% ungroup()
stopifnot(!anyDuplicated(region_year[c("aewr_region_id", "year")]))

# Worksite rows: b01 position and hours construction -------------------------
w <- id_read_worksites(c("case_number", "employer_id_balanced", "employer_is_worksite_business",
  "adjusted_nbr_workers_certified", "period_begin_date", "period_end_date",
  "number_of_hours", "county_fips_list")) %>%
  filter(employer_is_worksite_business, !is.na(employer_id_balanced), employer_id_balanced != "") %>%
  mutate(begin = as.Date(period_begin_date), end = as.Date(period_end_date)) %>%
  filter(!is.na(begin))
# b01 imputes missing weekly hours with the mean over its whole frame; this
# frame is the worksite-business subset through FY2025.
hours_mean <- mean(suppressWarnings(as.numeric(w$number_of_hours)), na.rm = TRUE)
w <- w %>% mutate(
  start_year = as.integer(year(begin)),
  nh = coalesce(suppressWarnings(as.numeric(number_of_hours)), hours_mean),
  positions = coalesce(as.numeric(adjusted_nbr_workers_certified), 0),
  hours = as.numeric(end - begin) / 7 * nh * positions,
  row_id = row_number())
cat("Worksite-business rows FY<=2025:", nrow(w), " hours imputation mean:", round(hours_mean, 2), "\n")

# Equal split across listed counties (b01 convention)
wc <- w %>% select(row_id, case_number, fiscal_year, start_year, employer_id_balanced,
  positions, hours, county_fips_list) %>%
  mutate(county_fips_list = coalesce(county_fips_list, "")) %>%
  tidyr::separate_rows(county_fips_list, sep = ",") %>%
  mutate(piece = trimws(county_fips_list)) %>%
  group_by(row_id) %>% mutate(n_counties = n()) %>% ungroup() %>%
  mutate(county_fips = ifelse(nzchar(piece) & piece != "00000",
      harmonize_county_fips_2010(ifelse(nzchar(piece), piece, NA_character_)), NA_character_),
    positions = positions / n_counties, hours = hours / n_counties) %>%
  left_join(county_region %>% select(county_fips, region_c = aewr_region_id), by = "county_fips") %>%
  mutate(state_fips = substr(county_fips, 1, 2)) %>%
  left_join(state_region %>% rename(region_s = aewr_region_id), by = "state_fips") %>%
  mutate(aewr_region_id = coalesce(region_c, region_s))
unmatched <- wc %>% summarise(rows = n(), unmatched_rows = sum(is.na(aewr_region_id)),
  unmatched_position_share = sum(positions[is.na(aewr_region_id)], na.rm = TRUE) / sum(positions, na.rm = TRUE),
  state_fallback_rows = sum(is.na(region_c) & !is.na(region_s)))
print(unmatched)
id_write(unmatched, out_dir, "track3_geography_match.csv")
wc <- wc %>% filter(!is.na(aewr_region_id))

# Employer-region-year cells under two calendars ------------------------------
cells_for <- function(year_col) {
  wc %>% mutate(year = .data[[year_col]]) %>%
    group_by(employer_id_balanced, aewr_region_id, year) %>%
    summarise(positions = sum(positions, na.rm = TRUE), hours = sum(hours, na.rm = TRUE), .groups = "drop") %>%
    filter(positions > 0)
}
cells <- list(start_year = cells_for("start_year"), fiscal_year = cells_for("fiscal_year"))

# Support ------------------------------------------------------------------------
support <- bind_rows(lapply(names(cells), function(cal) {
  cells[[cal]] %>% group_by(employer_id_balanced, year) %>%
    mutate(n_regions = n_distinct(aewr_region_id)) %>% ungroup() %>%
    group_by(calendar = cal, year) %>%
    summarise(employers = n_distinct(employer_id_balanced),
      multi_region_employers = n_distinct(employer_id_balanced[n_regions >= 2]),
      employers_3plus_regions = n_distinct(employer_id_balanced[n_regions >= 3]),
      multi_share_positions = sum(positions[n_regions >= 2]) / sum(positions),
      multi_share_hours = sum(hours[n_regions >= 2], na.rm = TRUE) / sum(hours, na.rm = TRUE),
      multi_positions = sum(positions[n_regions >= 2]),
      regions_touched = n_distinct(aewr_region_id[n_regions >= 2]),
      .groups = "drop")
})) %>% filter(year >= 2008, year <= 2025) %>%
  mutate(multi_employer_share = multi_region_employers / employers,
    period = ifelse(year <= 2019, "2008-2019", "2020-2025"))
id_write(support, out_dir, "track3_support_by_year.csv")
print(as.data.frame(support), digits = 3)
support_period <- support %>% filter(year >= 2011) %>%
  mutate(period = ifelse(year <= 2019, "2011-2019", "2020-2025")) %>% group_by(calendar, period) %>%
  summarise(years = paste(range(year), collapse = "-"),
    mean_employers = mean(employers), mean_multi_region_employers = mean(multi_region_employers),
    mean_multi_employer_share = mean(multi_employer_share),
    mean_multi_share_positions = mean(multi_share_positions),
    mean_multi_share_hours = mean(multi_share_hours), .groups = "drop")
id_write(support_period, out_dir, "track3_support_by_period.csv")
print(support_period)

# Region pairs spanned by multi-region employer-years (start-year calendar)
pairs <- cells$start_year %>% filter(year >= 2011, year <= 2025) %>%
  group_by(employer_id_balanced, year) %>% filter(n_distinct(aewr_region_id) >= 2) %>%
  summarise(regs = list(sort(unique(aewr_region_id))), .groups = "drop") %>%
  mutate(pair = lapply(regs, function(r) apply(combn(r, 2), 2, paste, collapse = "_"))) %>%
  tidyr::unnest(pair) %>% mutate(period = ifelse(year <= 2019, "2011-2019", "2020-2025")) %>%
  count(period, pair, name = "employer_years") %>% arrange(period, desc(employer_years))
id_write(pairs, out_dir, "track3_region_pairs.csv")
pair_summary <- pairs %>% group_by(period) %>%
  summarise(distinct_pairs = n(), pairs_with_10plus_employer_years = sum(employer_years >= 10),
    top5_pair_share = sum(head(sort(employer_years, decreasing = TRUE), 5)) / sum(employer_years), .groups = "drop")
id_write(pair_summary, out_dir, "track3_region_pair_summary.csv")
print(pair_summary)

# Job-title types of multi-region employer-years (employer_checks.R regex)
titles <- arrow::open_dataset(path_int("h2a_with_fips.parquet")) %>%
  select(case_number, fiscal_year, job_title) %>% filter(fiscal_year <= ID_LAST_PRE_FY) %>% collect() %>%
  distinct(case_number, fiscal_year, .keep_all = TRUE) %>%
  mutate(t = toupper(job_title), type = case_when(
    grepl("EQUIP|MACHIN|OPERATOR|COMBIN|TRUCK|DRIVER|TRACTOR|CUSTOM|MECHANIC", t) ~ "equipment",
    grepl("ANIMAL|LIVESTOCK|RANCH|DAIRY|HERD|CATTLE|SHEEP|GOAT|HOG|SWINE|POULTRY|BEE|HORSE|MILK", t) ~ "livestock",
    TRUE ~ "crop")) %>% select(case_number, fiscal_year, type)
id_assert_pre(titles$fiscal_year)
multi_keys <- cells$start_year %>% group_by(employer_id_balanced, year) %>%
  filter(n_distinct(aewr_region_id) >= 2) %>% ungroup() %>% distinct(employer_id_balanced, year)
types <- wc %>% rename(year = start_year) %>% filter(year >= 2011, year <= 2025) %>%
  left_join(titles, by = c("case_number", "fiscal_year")) %>%
  mutate(type = coalesce(type, "crop"), multi = paste(employer_id_balanced, year) %in%
    paste(multi_keys$employer_id_balanced, multi_keys$year),
    period = ifelse(year <= 2019, "2011-2019", "2020-2025")) %>%
  group_by(period, multi, type) %>% summarise(positions = sum(positions), .groups = "drop") %>%
  group_by(period, multi) %>% mutate(position_share = positions / sum(positions)) %>% ungroup()
id_write(types, out_dir, "track3_title_types.csv")
print(types)

# Estimation panel ----------------------------------------------------------------
# Declared before estimation. Employer-years with positive positions in >= 2
# AEWR regions. Cells: every region the employer used in any of its
# multi-region years, crossed with its multi-region years (zeros kept), so the
# allocation can move on the extensive margin within employer. Sensitivity:
# positive cells only. Outcomes: region share of the employer-year's
# positions and hours; log1p(positions). FE: employer x year, employer x region.
build_panel <- function(cal) {
  cc <- cells[[cal]] %>% filter(year >= 2011, year <= 2025)
  my <- cc %>% group_by(employer_id_balanced, year) %>%
    filter(n_distinct(aewr_region_id) >= 2) %>% ungroup()
  regs <- my %>% distinct(employer_id_balanced, aewr_region_id)
  yrs <- my %>% distinct(employer_id_balanced, year)
  inner_join(regs, yrs, by = "employer_id_balanced", relationship = "many-to-many") %>%
    left_join(cc, by = c("employer_id_balanced", "aewr_region_id", "year")) %>%
    mutate(positions = coalesce(positions, 0), hours = coalesce(hours, 0)) %>%
    group_by(employer_id_balanced, year) %>%
    mutate(share_pos = positions / sum(positions), share_hours = if (sum(hours) > 0) hours / sum(hours) else NA_real_) %>%
    ungroup() %>% mutate(lpos = log1p(positions), positive = positions > 0) %>%
    left_join(region_year, by = c("aewr_region_id", "year")) %>%
    filter(is.finite(A)) %>% as.data.frame()
}
panels <- list(start_year = build_panel("start_year"), fiscal_year = build_panel("fiscal_year"))
panel_desc <- bind_rows(lapply(names(panels), function(cal) panels[[cal]] %>%
  mutate(period = ifelse(year <= 2019, "2011-2019", "2020-2025")) %>% group_by(calendar = cal, period) %>%
  summarise(cells = n(), positive_cells = sum(positive), employers = n_distinct(employer_id_balanced),
    employer_years = n_distinct(paste(employer_id_balanced, year)), regions = n_distinct(aewr_region_id),
    mean_share = mean(share_pos), mean_A = mean(A), .groups = "drop")))
id_write(panel_desc, out_dir, "track3_panel_description.csv")
print(panel_desc)

windows <- list(`2011-2019` = 2011:2019, `2020-2024` = 2020:2024, `2011-2024` = 2011:2024,
  `2020-2025` = 2020:2025, `2011-2025` = 2011:2025)
FE <- "employer_id_balanced^year + employer_id_balanced^aewr_region_id"
specs <- tribble(~spec, ~rhs, ~terms, ~needs,
  "base", "A", "A", character(),
  "lead", "A + A_lead", c("A", "A_lead"), "A_lead",
  "price_controls", "A + price_index + quantity_index", "A", c("price_index", "quantity_index"))
outcomes <- c(share_pos = "share_pos", share_hours = "share_hours", lpos = "lpos")

rows <- wild <- fails <- list()
for (cal in names(panels)) for (win in names(windows)) for (sample in c("all_cells", "positive_cells")) {
  if (cal == "fiscal_year" && !win %in% c("2011-2019", "2011-2025", "2020-2025")) next
  if (cal == "start_year" && win %in% c("2020-2025", "2011-2025")) next # start-year 2025 truncated by FY filter
  for (oc in names(outcomes)) for (k in seq_len(nrow(specs))) {
    sp <- specs[k, ]
    if (sample == "positive_cells" && sp$spec != "base") next
    d <- panels[[cal]] %>% filter(year %in% windows[[win]])
    if (sample == "positive_cells") d <- d %>% filter(positive) %>%
      group_by(employer_id_balanced, year) %>% filter(n() >= 2) %>% ungroup() %>% as.data.frame()
    d <- bite_complete(d, c(outcomes[[oc]], "A", unlist(sp$needs)))
    meta <- tibble(calendar = cal, window = win, sample = sample, outcome = oc, spec = sp$spec)
    key <- paste(cal, win, sample, oc, sp$spec, sep = "__")
    message(key)
    res <- tryCatch({
      m <- feols(as.formula(paste(outcomes[[oc]], "~", sp$rhs, "|", FE)), d,
        vcov = ~aewr_region_id, demeaned = TRUE, data.save = TRUE)
      r1 <- id_rows(m, d, unlist(sp$terms), meta, c("aewr_region_id", "employer_id_balanced"), "A")
      sm2 <- summary(m, vcov = ~aewr_region_id + employer_id_balanced)
      ct <- coeftable(sm2); tt <- intersect(unlist(sp$terms), rownames(ct))
      r2 <- r1 %>% filter(cluster == "aewr_region_id") %>%
        mutate(cluster = "twoway_region_employer", se = ct[tt, 2], p = ct[tt, 4], mde = 2.8 * se,
          mde_pct_mean = 100 * mde / abs(mean_y),
          elasticity_low = id_elasticity(estimate - qt(.975, clusters - 1) * se, mean_price, mean_y),
          elasticity_high = id_elasticity(estimate + qt(.975, clusters - 1) * se, mean_price, mean_y),
          mde_elasticity = abs(id_elasticity(mde, mean_price, mean_y)))
      rows[[key]] <- bind_rows(r1, r2) %>% mutate(
        semi_elasticity = if (oc == "lpos") estimate else estimate / mean_y,
        employers = n_distinct(d$employer_id_balanced[obs(m)]), regions = n_distinct(d$aewr_region_id[obs(m)]))
      if (sample == "all_cells" && sp$spec %in% c("base", "lead") &&
          (cal == "start_year" || sp$spec == "base")) {
        wild[[key]] <- bind_rows(lapply(unlist(sp$terms), function(term) {
          z <- tryCatch(id_wild(m, d, term, "aewr_region_id", B = 9999L),
            error = function(e) tibble(term = term, wild_p = NA_real_, wild_status = conditionMessage(e)))
          bind_cols(meta, z)
        }))
      }
      TRUE
    }, error = function(e) { fails[[key]] <<- bind_cols(meta, tibble(error = conditionMessage(e))); FALSE })
  }
}
est <- bind_rows(rows) %>%
  # For log1p(positions) the coefficient is already a semi-elasticity; the
  # elasticity at the mean price is beta * mean(A), not beta * mean(A) / mean(y).
  mutate(log_outcome = outcome == "lpos",
    elasticity = ifelse(log_outcome, estimate * mean_price, elasticity),
    elasticity_low = ifelse(log_outcome, (estimate - qt(.975, clusters - 1) * se) * mean_price, elasticity_low),
    elasticity_high = ifelse(log_outcome, (estimate + qt(.975, clusters - 1) * se) * mean_price, elasticity_high),
    mde_elasticity = ifelse(log_outcome, mde * mean_price, mde_elasticity),
    mde_pct_mean = ifelse(log_outcome, NA_real_, mde_pct_mean))
wl <- bind_rows(wild)
id_write(est, out_dir, "track3_estimates.csv")
id_write(wl, out_dir, "track3_wild.csv")
id_write(bind_rows(fails), out_dir, "track3_failures.csv")

headline <- est %>% filter(sample == "all_cells", spec %in% c("base", "lead")) %>%
  select(calendar, window, outcome, spec, term, cluster, estimate, se, p, mean_y, mean_price,
    elasticity, mde, mde_pct_mean, mde_elasticity, n, employers, regions) %>%
  left_join(wl %>% select(calendar, window, outcome, spec, term, wild_p, wild_status),
    by = c("calendar", "window", "outcome", "spec", "term"))
id_write(headline, out_dir, "track3_headline.csv")
print(as.data.frame(headline %>% filter(cluster != "employer_id_balanced")), digits = 3)
cat("Failures:", length(fails), "\n")
