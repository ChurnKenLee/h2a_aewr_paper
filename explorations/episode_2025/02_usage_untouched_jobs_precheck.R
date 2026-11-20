# Pre-period check B: can jobs the 2025 rule did not touch serve as within-state controls?
#
# Candidate control groups, classified from case attributes only:
#   range  = monthly-paid job with a herder/range title (own monthly AEWR; outside the 2025 rule);
#   piece  = non-range job whose order offers a piece rate (field exists from FY2020 only).
# The triple difference contrasts treated-minus-control hours growth across
# states by the size of the 2025 cut. State-by-year shocks common to both
# groups difference out; the cross-state dose is kept.
#
# Deciding numbers, FY2012-FY2025 only: control-group size by state, how closely
# control and treated growth co-move, and placebo-year design error of the
# triple difference against the double difference on the same states.
#
# The premise that piece-rate earnings did not move with the cut is NOT tested
# here: that needs FY2026 offers and belongs in a frozen registry.
#
# Args: 1 = scratch output dir (outside the repo); 2 = shared-outcomes dir.
source("explorations/episode_2025/ep_helpers.R")
out_dir <- id_output_dir()
shared <- ep_shared_dir()
RANGE_MIN_HOURS <- 50000      # mean FY2023-FY2025 control hours for a state to carry the contrast (about 24 FTE)
PIECE_FIRST_FY <- 2020L

frame <- ep_state_frame()
herder_regex <- jsonlite::fromJSON(path_code("designs", "wage_reforms", "design.json"))$herder_regex

attrs <- id_read_disclosures(c("case_number", "wage_unit", "job_title", "piece_rate_offer")) %>%
  filter(fiscal_year >= EP_FIRST_FY) %>%
  mutate(monthly = toupper(trimws(wage_unit)) %in% c("MONTH", "MTH"),
    herder_title = grepl(herder_regex, coalesce(job_title, ""), perl = TRUE),
    piece_rate = suppressWarnings(as.numeric(piece_rate_offer)),
    piece = is.finite(piece_rate) & piece_rate > 0) %>%
  group_by(case_number, fiscal_year) %>%
  summarise(monthly = any(monthly), herder_title = any(herder_title), piece = any(piece), .groups = "drop")
stopifnot(!anyDuplicated(attrs[c("case_number", "fiscal_year")]))

rows <- arrow::open_dataset(file.path(shared, "outcomes_case_rows.parquet")) %>%
  filter(fiscal_year >= EP_FIRST_FY, fiscal_year <= EP_LAST_PRE_FY, in_dq13) %>%
  select(case_number, fiscal_year, state_fips, cert_hours, cert_pos) %>% collect()
id_assert_pre(rows$fiscal_year)
# Rows whose county was not matched carry state "00"; they are outside every state contrast.
unmatched <- rows %>% group_by(fiscal_year) %>%
  summarise(share_hours_unmatched_county = sum(cert_hours[state_fips == "00"], na.rm = TRUE) / sum(cert_hours, na.rm = TRUE))
id_write(unmatched, out_dir, "untouched_unmatched_county_by_fy.csv")
rows <- rows %>% filter(state_fips != "00") %>% left_join(attrs, by = c("case_number", "fiscal_year"))
match_rate <- rows %>% group_by(fiscal_year) %>%
  summarise(hours_m = sum(cert_hours, na.rm = TRUE) / 1e6, share_hours_with_attributes = sum(cert_hours[!is.na(monthly)], na.rm = TRUE) / sum(cert_hours, na.rm = TRUE))
id_write(match_rate, out_dir, "untouched_attribute_match_by_fy.csv")
stopifnot(min(match_rate$share_hours_with_attributes) > 0.99)

# The rebuilt state totals must equal the shared county panel.
check <- arrow::open_dataset(file.path(shared, "outcomes_county_fy.parquet")) %>%
  filter(year >= EP_FIRST_FY, year <= EP_LAST_PRE_FY) %>% select(state_fips, year, cert_hours_dq13) %>% collect() %>%
  group_by(state_fips, year) %>% summarise(h_panel = sum(cert_hours_dq13, na.rm = TRUE), .groups = "drop") %>%
  full_join(rows %>% group_by(state_fips, year = fiscal_year) %>% summarise(h_rows = sum(cert_hours, na.rm = TRUE), .groups = "drop"),
    by = c("state_fips", "year")) %>% filter(!is.na(state_fips))
stopifnot(max(abs(coalesce(check$h_panel, 0) - coalesce(check$h_rows, 0))) < 1e-4 * max(check$h_panel, na.rm = TRUE))

rows <- rows %>% mutate(monthly = coalesce(monthly, FALSE), herder_title = coalesce(herder_title, FALSE), piece = coalesce(piece, FALSE),
  group = case_when(monthly & herder_title ~ "range", monthly ~ "other_monthly", TRUE ~ "nonrange"),
  piece_group = case_when(group != "nonrange" ~ NA_character_, fiscal_year < PIECE_FIRST_FY ~ NA_character_, piece ~ "piece", TRUE ~ "no_piece"))

national <- rows %>% group_by(fiscal_year, group) %>% summarise(hours_m = sum(cert_hours, na.rm = TRUE) / 1e6, positions = sum(cert_pos, na.rm = TRUE), .groups = "drop")
national_piece <- rows %>% filter(!is.na(piece_group)) %>% group_by(fiscal_year, piece_group) %>%
  summarise(hours_m = sum(cert_hours, na.rm = TRUE) / 1e6, positions = sum(cert_pos, na.rm = TRUE), .groups = "drop")
id_write(national, out_dir, "untouched_national_by_group.csv"); id_write(national_piece, out_dir, "untouched_national_piece.csv")

years <- EP_FIRST_FY:EP_LAST_PRE_FY
state_group <- function(d, gvar, treated, control, first_fy) {
  d %>% filter(.data[[gvar]] %in% c(treated, control), fiscal_year >= first_fy) %>%
    group_by(state_fips, year = fiscal_year, g = ifelse(.data[[gvar]] == treated, "T", "C")) %>%
    summarise(hours = sum(cert_hours, na.rm = TRUE), .groups = "drop") %>%
    tidyr::complete(state_fips, year = first_fy:EP_LAST_PRE_FY, g = c("T", "C"), fill = list(hours = 0)) %>%
    tidyr::pivot_wider(names_from = g, values_from = hours, names_prefix = "h") %>%
    inner_join(frame, by = "state_fips")
}

analyse <- function(panel, label, min_hours, placebo_years) {
  size <- panel %>% filter(year %in% 2023:2025) %>% group_by(state_fips) %>% summarise(c_mean = mean(hC), t_mean = mean(hT), .groups = "drop")
  ok <- panel %>% group_by(state_fips) %>% summarise(all_pos = all(hC > 0 & hT > 0), .groups = "drop") %>%
    inner_join(size, by = "state_fips") %>% mutate(in_support = all_pos & c_mean >= min_hours)
  s <- ok$state_fips[ok$in_support]
  tot25 <- panel %>% filter(year == EP_LAST_PRE_FY)
  fr <- tot25 %>% filter(state_fips %in% s)
  wsd <- function(x, w) sqrt(sum(w * (x - sum(w * x) / sum(w))^2) / sum(w))
  support <- tibble(design = label, control_hours25_m = sum(tot25$hC) / 1e6, treated_hours25_m = sum(tot25$hT) / 1e6,
    states_any_control = sum(tot25$hC > 0), states_in_support = length(s), regions_in_support = n_distinct(fr$aewr_region_id),
    control_hours25_in_support_m = sum(fr$hC) / 1e6, treated_hours25_in_support_m = sum(fr$hT) / 1e6,
    share_of_treated_hours_in_support = sum(fr$hT) / sum(tot25$hT),
    top3_share_of_control_hours = sum(sort(tot25$hC, decreasing = TRUE)[1:3]) / sum(tot25$hC),
    dose_sd_in_support = sd(fr$dose1_us), dose_sd_treated_weighted = wsd(fr$dose1_us, fr$hT))
  g <- panel %>% filter(state_fips %in% s) %>% arrange(state_fips, year) %>% group_by(state_fips) %>%
    mutate(gT = log(hT) - log(lag(hT)), gC = log(hC) - log(lag(hC)), w_size = lag(hT), w_equal = 1) %>% ungroup() %>% filter(is.finite(gT), is.finite(gC)) %>%
    group_by(year) %>% mutate(gT_dm = gT - mean(gT), gC_dm = gC - mean(gC)) %>% ungroup()
  comove <- tibble(design = label, state_years = nrow(g), cor_treated_control_growth = cor(g$gT_dm, g$gC_dm),
    sd_treated_growth = sd(g$gT_dm), sd_control_growth = sd(g$gC_dm), sd_difference = sd(g$gT_dm - g$gC_dm),
    variance_ratio_difference_to_treated = var(g$gT_dm - g$gC_dm) / var(g$gT_dm))
  res <- bind_rows(lapply(c("size", "equal"), function(wt) bind_rows(lapply(placebo_years, function(tau) {
    x <- g %>% filter(year == tau) %>% mutate(w = if (wt == "size") w_size else 1, dd = gT, ddd = gT - gC)
    bind_rows(ep_fit(x, "dd", "dose1_us") %>% mutate(estimator = "double difference, treated jobs only"),
      ep_fit(x, "ddd", "dose1_us") %>% mutate(estimator = "triple difference, treated minus control")) %>%
      mutate(weighting = wt, tau = tau)
  })))) %>% mutate(outcome = "hours", sample = label)
  mean_rate <- sum(fr$hT * fr$rate25) / sum(fr$hT)
  list(support = support, comove = comove, placebo = res, summary = ep_summarise(res, mean_rate),
    states = ok %>% inner_join(frame %>% select(state_fips, aewr_region_id, dose1_us), by = "state_fips") %>% mutate(design = label))
}

range_panel <- state_group(rows, "group", "nonrange", "range", EP_FIRST_FY)
piece_panel <- state_group(rows, "piece_group", "no_piece", "piece", PIECE_FIRST_FY)
a <- analyse(range_panel, "range herders as controls", RANGE_MIN_HOURS, EP_PLACEBO_FY)
b <- analyse(piece_panel, "piece-rate orders as controls", RANGE_MIN_HOURS, (PIECE_FIRST_FY + 1L):EP_LAST_PRE_FY)
id_write(bind_rows(a$support, b$support), out_dir, "untouched_support.csv")
id_write(bind_rows(a$comove, b$comove), out_dir, "untouched_comovement.csv")
id_write(bind_rows(a$placebo, b$placebo), out_dir, "untouched_placebo_by_year.csv")
id_write(bind_rows(a$summary, b$summary), out_dir, "untouched_placebo_summary.csv")
id_write(bind_rows(a$states, b$states), out_dir, "untouched_states.csv")

options(width = 220)
cat("NATIONAL HOURS BY GROUP (millions, decisions Oct-Jun)\n")
print(as.data.frame(national %>% filter(fiscal_year %in% c(2015, 2019, 2023, 2025)) %>% tidyr::pivot_wider(id_cols = fiscal_year, names_from = group, values_from = hours_m)), digits = 4)
print(as.data.frame(national_piece %>% tidyr::pivot_wider(id_cols = fiscal_year, names_from = piece_group, values_from = hours_m)), digits = 4)
cat("\nSUPPORT\n"); print(as.data.frame(bind_rows(a$support, b$support)), digits = 3)
cat("\nrange-support states:\n"); print(as.data.frame(a$states %>% filter(in_support) %>% arrange(desc(c_mean)) %>%
  transmute(state_fips, aewr_region_id, range_hours_k = round(c_mean / 1e3), nonrange_hours_k = round(t_mean / 1e3), dose1_us)))
cat("\nCO-MOVEMENT (year-demeaned one-year log growth)\n"); print(as.data.frame(bind_rows(a$comove, b$comove)), digits = 3)
cat("\nPLACEBO SUMMARY (elasticity units; true effect is zero)\n")
print(as.data.frame(bind_rows(a$summary, b$summary) %>% select(sample, weighting, estimator, placebo_years, states, dose_sd_left,
  bias_elasticity, mde_elasticity_rmse, mde_elasticity_se, share_abs_t_over_2)), digits = 3)
