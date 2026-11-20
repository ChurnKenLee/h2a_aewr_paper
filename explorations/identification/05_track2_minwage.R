# Track 2 (category 2): domestic labor cost via state minimum wages.
# Exemption audit, Phase 0 support, four specifications. Scratch only:
# writes track2_*.csv to the output directory given as argument 1.
source("explorations/identification/id_helpers.R")
out_dir <- id_output_dir()
CL <- c("state_fips", "aewr_region_id")
Y <- c(cert = "h2a_cert_share_farm_workers_2011_start_year",
       hours = "h2a_cert_hours_per_farm_worker_2011_start_year")
# Declared before any outcome regression: a "large" increase is a nominal
# year-over-year rise in the applicable floor of at least 5%. Nominal, so that
# farm-PPI swings in the deflator do not create events.
LARGE_PCT <- 0.05

# 1. Agricultural-exemption audit -------------------------------------------
src <- read_parquet(path_int("state_year_min_wage.parquet"))
exempt_list <- sort(unique(src$state_abbreviation[src$agriculture_exemption]))
varying <- src %>% group_by(state_fips) %>% summarise(k = n_distinct(agriculture_exemption), .groups = "drop")
missing_2024 <- setdiff(unique(src$state_abbreviation[src$year == 2023]),
  unique(src$state_abbreviation[src$year == 2024]))
p <- read_parquet(path_processed("county_year_panel.parquet")) %>% filter(year <= 2025L)
# Optional argument 2: a state_real_minwages.parquet built by the corrected
# c01/04 producer (dated coverage table). Its agricultural floor replaces the
# panel's, and outputs are written as track2c_*.csv. Without it: track2_*.csv.
mw_override <- commandArgs(trailingOnly = TRUE)[2]
T2 <- if (is.na(mw_override)) "track2" else "track2c"
if (!is.na(mw_override)) {
  ov_all <- read_parquet(mw_override) %>% mutate(year = as.integer(year))
  stopifnot(!anyDuplicated(ov_all[c("state_fips", "year")]), "ag_floor_rule" %in% names(ov_all))
  p <- p %>% select(-agriculture_exemption, -prevailing_ag_min_wage) %>%
    left_join(ov_all %>% select(state_fips, year, agriculture_exemption, prevailing_ag_min_wage),
      by = c("state_fips", "year"), relationship = "many-to-one")
}
bite_check_keys(p)
p <- panel_iv_prepare_outcomes(p, bite_spec)
st <- p %>% distinct(state_fips, state_abbrev, aewr_region_id, year, agriculture_exemption,
  state_min_wage, federal_min_wage, prevailing_ag_min_wage, ppi_2012)
stopifnot(!anyDuplicated(st[c("state_fips", "year")]))
# Domain-knowledge flags, NOT checked against statute or online in this run.
flag_note <- c(
  WA = "state minimum wage believed to cover farmworkers (overtime, not minimum, was the exemption); to verify against statute; not checked online",
  NJ = "state minimum believed to cover farm labor (separate slower 2019 schedule for seasonal piece-rate workers); to verify against statute; not checked online",
  MA = "separate agricultural minimum ($8.00) believed to apply rather than federal; to verify against statute; not checked online",
  VA = "2020 Virginia Minimum Wage Act amendments may have changed farmworker coverage; to verify against statute; not checked online",
  ME = "agricultural exemption believed broad but later partial coverage proposals; to verify against statute; not checked online",
  VT = "agricultural exemption believed to apply; to verify against statute; not checked online",
  DE = "agricultural exemption believed to apply; to verify against statute; not checked online",
  RI = "agricultural coverage uncertain; to verify against statute; not checked online",
  NE = "agricultural exemption believed to apply; to verify against statute; not checked online",
  WV = "small-employer exemption may be what is coded; to verify against statute; not checked online",
  NH = "state minimum tied to federal after 2011; to verify against statute; not checked online")
audit <- st %>% filter(year >= 2008) %>% group_by(state_abbrev, state_fips, aewr_region_id) %>%
  summarise(panel_exempt = first(agriculture_exemption),
    in_source_list = first(state_abbrev) %in% exempt_list,
    years_state_above_federal = sum(state_min_wage > federal_min_wage + 1e-9, na.rm = TRUE),
    first_year_above = suppressWarnings(min(year[state_min_wage > federal_min_wage + 1e-9], na.rm = TRUE)),
    max_floor_gap_moved = max(pmax(state_min_wage - federal_min_wage, 0) * agriculture_exemption, na.rm = TRUE),
    .groups = "drop") %>%
  filter(panel_exempt) %>%
  mutate(first_year_above = ifelse(is.finite(first_year_above), first_year_above, NA),
    exempt_reason = ifelse(in_source_list, "hard-coded a01 list", "absent from 2024 source rows: c01 coalesce(TRUE)"),
    classification_moves_floor = years_state_above_federal > 0,
    domain_flag = unname(flag_note[state_abbrev]))
audit_meta <- tibble(item = c("source_exemption_states", "exemption_varies_by_year_in_source",
  "states_missing_from_2024_source_rows", "source_notes_or_dates_in_producer"),
  value = c(paste(exempt_list, collapse = ";"), as.character(any(varying$k > 1)),
    paste(missing_2024, collapse = ";"), "none: a01 list is a literal with no citation or effective dates; c01 applies the 2024 value to every year"))
id_write(audit, out_dir, paste0(T2, "_exemption_audit.csv"))
id_write(audit_meta, out_dir, paste0(T2, "_exemption_audit_meta.csv"))
contested <- audit$state_fips[audit$classification_moves_floor]

# Floors ---------------------------------------------------------------------
st <- st %>% arrange(state_fips, year) %>% mutate(
  ag_nom = prevailing_ag_min_wage,
  gen_nom = pmax(coalesce(state_min_wage, federal_min_wage), federal_min_wage),
  ag_real = ag_nom / ppi_2012, gen_real = gen_nom / ppi_2012) %>%
  group_by(state_fips) %>% mutate(
    ag_change = ag_nom - lag(ag_nom), gen_change = gen_nom - lag(gen_nom),
    ag_lead = lead(ag_real), gen_lead = lead(gen_real)) %>% ungroup()
p <- p %>% left_join(st %>% select(state_fips, year, ag_nom, gen_nom, ag_real, gen_real, ag_lead, gen_lead),
  by = c("state_fips", "year"), relationship = "many-to-one") %>%
  mutate(log_ag = log(ag_real), log_gen = log(gen_real))

# 2. Support -------------------------------------------------------------------
resid_sd <- function(v, d) {
  dd <- bite_complete(d, v)
  r <- resid(feols(as.formula(paste(v, "~ 1 | county_fips + aewr_region_id^year")), dd))
  rc <- resid(feols(as.formula(paste(v, "~ 1 | county_fips + aewr_region_id^year")), dd %>% filter(!state_fips %in% contested)))
  tibble(variable = v, residual_sd = sd(r), raw_sd = sd(dd[[v]]),
    residual_ss_share_contested_states = sum(r[dd$state_fips %in% contested]^2) / sum(r^2),
    residual_sd_excluding_contested = sd(rc))
}
sp <- p %>% filter(year >= 2011)
support_sd <- bind_rows(lapply(c("ag_real", "gen_real", "log_ag", "log_gen", "ag_nom", "gen_nom"), resid_sd, d = sp))
changes <- st %>% filter(year >= 2009, year <= 2025) %>% summarise(
  state_years = n(),
  ag_changes = sum(abs(ag_change) > 1e-9, na.rm = TRUE),
  ag_changes_excluding_federal_years = sum(abs(ag_change) > 1e-9 & !year %in% 2009, na.rm = TRUE),
  ag_large = sum(ag_change / (ag_nom - ag_change) >= LARGE_PCT, na.rm = TRUE),
  gen_changes = sum(abs(gen_change) > 1e-9, na.rm = TRUE),
  gen_large = sum(gen_change / (gen_nom - gen_change) >= LARGE_PCT, na.rm = TRUE),
  gen_changes_in_contested_states = sum(abs(gen_change) > 1e-9 & state_fips %in% contested, na.rm = TRUE),
  states_ever_above_federal_ag = n_distinct(state_fips[ag_nom > federal_min_wage + 1e-9]),
  states_ever_above_federal_gen = n_distinct(state_fips[gen_nom > federal_min_wage + 1e-9]))
# Frozen 2008/2010 closeness of real CZ p10/p25 to the real applicable floor.
fz <- p %>% filter(year %in% c(2008L, 2010L)) %>%
  mutate(g10 = wage_p10 - ag_real, g25 = wage_p25 - ag_real) %>%
  group_by(county_fips, aewr_region_id) %>%
  summarise(k = sum(is.finite(g10) & is.finite(g25)), g10 = mean(g10), g25 = mean(g25), .groups = "drop") %>%
  filter(k == 2) %>%
  mutate(E_p10 = bite_center(-g10, aewr_region_id), E_p25 = bite_center(-g25, aewr_region_id))
bite_check_keys(fz, "county_fips")
support_bite <- fz %>% summarise(counties = n(),
  share_p10_within_1 = mean(abs(g10) <= 1), share_p25_within_1 = mean(abs(g25) <= 1),
  share_p10_below_floor = mean(g10 < 0), share_p25_below_floor = mean(g25 < 0),
  mean_gap_p10 = mean(g10), mean_gap_p25 = mean(g25), sd_gap_p10 = sd(g10), sd_gap_p25 = sd(g25))
id_write(support_sd, out_dir, paste0(T2, "_support_residual_sd.csv"))
id_write(changes, out_dir, paste0(T2, "_support_changes.csv"))
id_write(support_bite, out_dir, paste0(T2, "_support_bite.csv"))
print(support_sd); print(changes); print(support_bite)

# 3. Specifications -------------------------------------------------------------
p <- p %>% left_join(fz %>% select(county_fips, E_p10, E_p25), by = "county_fips")
rows <- wild <- fails <- list()
run <- function(key, y, rhs, fe, d, terms, price, clusters = CL, wild_terms = terms, meta_extra = list()) {
  meta <- tibble(spec = key, outcome = names(Y)[Y == y], !!!meta_extra)
  ok <- tryCatch({
    m <- feols(as.formula(paste(y, "~", rhs, "|", fe)), d, vcov = ~state_fips, demeaned = TRUE, data.save = TRUE)
    rr <- bind_rows(lapply(terms, function(tm) id_rows(m, d, tm, meta, clusters, price[[tm]])))
    rows[[paste(key, y)]] <<- rr
    wild[[paste(key, y)]] <<- bind_rows(lapply(wild_terms, function(tm)
      bind_cols(meta, id_wild(m, d, tm, "state_fips"))))
    m
  }, error = function(e) { fails[[paste(key, y)]] <<- bind_cols(meta, tibble(error = conditionMessage(e))); NULL })
  invisible(ok)
}
FE_R <- "county_fips + aewr_region_id^year"
base <- p %>% filter(year >= 2011, year <= 2025)
models <- list()
for (y in Y) {
  for (mw in c("ag_real", "gen_real", "log_ag", "log_gen")) {
    d <- bite_complete(base, c(y, mw))
    models[[paste("s1", mw, y)]] <- run(paste0("s1_level__", mw), y, mw, FE_R, d, mw, setNames(list(mw), mw))
  }
  # Excluding contested states: classification-free sample.
  d <- bite_complete(base %>% filter(!state_fips %in% contested), c(y, "ag_real"))
  run("s1_level__ag_real__excl_contested", y, "ag_real", FE_R, d, "ag_real", list(ag_real = "ag_real"))
  # Lead placebo, 2011-2024 (lead from 2025 panel floor).
  for (mw in c("ag_real", "gen_real")) {
    ld <- sub("_real", "_lead", mw)
    d <- bite_complete(base %>% filter(year <= 2024), c(y, mw, ld))
    run(paste0("s1_lead__", mw), y, paste(mw, "+", ld), FE_R, d, c(mw, ld), setNames(list(mw, mw), c(mw, ld)))
  }
}

# Event study: distributed leads/lags of large nominal increases (≥5%).
# Floors before 2008 come from FRED with the same exemption rule.
fred <- read_parquet(path_int("fred_state_minwages.parquet")) %>% mutate(year = as.integer(year)) %>%
  filter(year >= 2000, year <= 2026) %>%
  filter(state_fips %in% st$state_fips) %>%
  mutate(gen = pmax(coalesce(state_min_wage, federal_min_wage), federal_min_wage))
fred <- if (is.na(mw_override)) {
  fred %>% left_join(st %>% distinct(state_fips, agriculture_exemption), by = "state_fips") %>%
    filter(!is.na(agriculture_exemption)) %>%
    mutate(ag = ifelse(agriculture_exemption, federal_min_wage, gen))
} else {
  fred %>% left_join(ov_all %>% select(state_fips, year, ag = prevailing_ag_min_wage), by = c("state_fips", "year"))
}
fred <- fred %>%
  arrange(state_fips, year) %>% group_by(state_fips) %>%
  mutate(ev_ag = as.integer(ag / lag(ag) - 1 >= LARGE_PCT), ev_gen = as.integer(gen / lag(gen) - 1 >= LARGE_PCT)) %>%
  ungroup() %>% filter(year >= 2001)
evs <- list()
for (kind in c("ag", "gen")) {
  e <- fred %>% select(state_fips, year, ev = all_of(paste0("ev_", kind)))
  grid <- tidyr::crossing(state_fips = unique(e$state_fips), year = 2011:2022)
  lagged <- function(k) grid %>% mutate(yy = year - k) %>% left_join(e, by = c("state_fips", "yy" = "year")) %>% pull(ev)
  bins <- grid
  for (k in -3:3) if (k != -1) bins[[paste0("ev_", ifelse(k < 0, paste0("m", -k), k))]] <- coalesce(lagged(k), 0L)
  # Endpoint bins: events 4+ years ahead (to 2026) and 4+ years before (from 2001).
  bins$ev_m4 <- Reduce(`+`, lapply(4:(2026 - 2011), function(k) coalesce(lagged(-k), 0L)))
  bins$ev_4 <- Reduce(`+`, lapply(4:21, function(k) coalesce(lagged(k), 0L)))
  terms <- c("ev_m4", "ev_m3", "ev_m2", "ev_0", "ev_1", "ev_2", "ev_3", "ev_4")
  for (y in Y) {
    d <- base %>% filter(year <= 2022) %>% inner_join(bins, by = c("state_fips", "year")) %>% bite_complete(y)
    m <- feols(as.formula(paste(y, "~", paste(terms, collapse = " + "), "|", FE_R)), d, vcov = ~state_fips)
    ct <- coeftable(m)
    w <- wald(m, keep = "^ev_m", print = FALSE)
    evs[[paste(kind, y)]] <- tibble(floor = kind, outcome = names(Y)[Y == y], term = rownames(ct),
      estimate = ct[, 1], se = ct[, 2], p = ct[, 4], pre_F = w$stat, pre_p = w$p,
      n = nobs(m), states = n_distinct(d$state_fips), event_state_years = sum(e$ev[e$year >= 2008], na.rm = TRUE),
      mean_y = mean(d[[y]]))
  }
}
id_write(bind_rows(evs), out_dir, paste0(T2, "_event_study.csv"))

# (2) Bite triple difference: MW x frozen closeness (higher E = floor closer
# to or above local low wages). Exposure x year FE as explicit i(year, E)
# regressors so the bootstrap partials them out.
for (y in Y) for (ex in c("E_p10", "E_p25")) {
  d <- bite_complete(base, c(y, "ag_real", ex)) %>% mutate(E = .data[[ex]], X = ag_real * E)
  run(paste0("s2_bite__", ex), y, "ag_real + X", FE_R, d, c("ag_real", "X"),
    list(ag_real = "ag_real", X = "ag_real"), wild_terms = "X")
  run(paste0("s2_bite_Eyear__", ex), y, "ag_real + X + i(year, E, ref = 2011)", FE_R, d, c("ag_real", "X"),
    list(ag_real = "ag_real", X = "ag_real"), wild_terms = "X")
}

# (3) Contiguous cross-state pairs within former AEWR region, pair x year FE.
mem <- read_parquet(path_int("border_statewide_memberships.parquet"))
pair_rows <- list()
for (sid in c("statewide_h0_unscreened_within_region", "statewide_h0_screened_within_region")) {
  mm <- mem %>% filter(spec_id == sid) %>% select(county_fips, region_pair_id, former_aewr_region_id)
  stopifnot(!anyDuplicated(mm$county_fips))
  for (y in Y) for (mw in c("ag_real", "gen_real")) {
    d <- base %>% inner_join(mm, by = "county_fips") %>% bite_complete(c(y, mw))
    stopifnot(all(d$aewr_region_id == d$former_aewr_region_id))
    m <- run(paste0("s3_pairs__", sid, "__", mw), y, mw, "county_fips + region_pair_id^year", d, mw,
      setNames(list(mw), mw), clusters = c("region_pair_id", "state_fips", "aewr_region_id"))
    if (!is.null(m)) {
      tw <- summary(m, vcov = ~state_fips + region_pair_id)
      within_sd <- sd(resid(feols(as.formula(paste(mw, "~ 1 | county_fips + region_pair_id^year")), d)))
      pair_rows[[paste(sid, y, mw)]] <- tibble(spec = paste0("s3_pairs__", sid, "__", mw), outcome = names(Y)[Y == y],
        twoway_se = se(tw)[[mw]], twoway_p = pvalue(tw)[[mw]], pairs = n_distinct(d$region_pair_id),
        states = n_distinct(d$state_fips), counties = n_distinct(d$county_fips), within_pair_year_sd = within_sd,
        pair_years_with_floor_gap = nrow(distinct(d %>% group_by(region_pair_id, year) %>%
          filter(n_distinct(round(.data[[mw]], 6)) > 1), region_pair_id, year)))
    }
  }
}
id_write(bind_rows(pair_rows), out_dir, paste0(T2, "_pairs_twoway.csv"))

# (4) Relative-price restriction: county + year FE, AEWR and ag floor jointly.
rp <- list()
for (y in Y) for (mw in c("ag_real", "gen_real")) {
  d <- bite_complete(base, c(y, "aewr_ppi", mw))
  m <- run(paste0("s4_relprice__", mw), y, paste("aewr_ppi +", mw), "county_fips + year", d,
    c("aewr_ppi", mw), setNames(list("aewr_ppi", mw), c("aewr_ppi", mw)))
  if (!is.null(m)) for (cl in CL) {
    V <- vcov(m, vcov = as.formula(paste0("~", cl))); b <- coef(m)
    s <- b[["aewr_ppi"]] + b[[mw]]; v <- V["aewr_ppi", "aewr_ppi"] + V[mw, mw] + 2 * V["aewr_ppi", mw]
    G <- n_distinct(d[[cl]])
    rp[[paste(y, mw, cl)]] <- tibble(spec = paste0("s4_relprice__", mw), outcome = names(Y)[Y == y], cluster = cl,
      b_aewr = b[["aewr_ppi"]], b_mw = b[[mw]], sum = s, sum_se = sqrt(v),
      wald_F = s^2 / v, wald_p = pf(s^2 / v, 1, G - 1, lower.tail = FALSE))
  }
  # Restricted gap model: y on (AEWR - floor).
  d$gap <- d$aewr_ppi - d[[mw]]
  run(paste0("s4_gap_restricted__", mw), y, "gap", "county_fips + year", d, "gap", list(gap = "aewr_ppi"))
}
id_write(bind_rows(rp), out_dir, paste0(T2, "_relprice_wald.csv"))

est <- bind_rows(rows) %>% left_join(bind_rows(wild) %>% select(spec, outcome, term, wild_p, wild_cluster),
  by = c("spec", "outcome", "term"))
# Log specifications: elasticity = beta / mean(y) (beta per log point).
est <- est %>% mutate(is_log = grepl("^log_", term),
  elasticity = ifelse(is_log, estimate / mean_y, elasticity),
  elasticity_low = ifelse(is_log, (estimate - qt(.975, clusters - 1) * se) / mean_y, elasticity_low),
  elasticity_high = ifelse(is_log, (estimate + qt(.975, clusters - 1) * se) / mean_y, elasticity_high),
  mde_elasticity = ifelse(is_log, mde / abs(mean_y), mde_elasticity))
id_write(est, out_dir, paste0(T2, "_estimates.csv"))
id_write(bind_rows(fails), out_dir, paste0(T2, "_failures.csv"))
print(as.data.frame(est %>% filter(cluster == "state_fips") %>%
  select(spec, outcome, term, estimate, se, p, wild_p, mde_pct_mean, elasticity, mde_elasticity)), digits = 3)
print(bind_rows(fails))

# Translation under the relative-price restriction (only AEWR - floor matters,
# and a $1 floor rise raises the domestic cost of the relevant labor by $1):
# implied AEWR slope = -beta_MW, elasticity at the same sample's mean real AEWR.
tr <- est %>% filter(grepl("^s1_level__(ag|gen)_real$|^s1_level__ag_real__excl_contested$|^s3_pairs__", spec),
    term %in% c("ag_real", "gen_real")) %>%
  rowwise() %>% mutate(mean_aewr = {
    d <- if (grepl("^s3_", spec)) base %>% inner_join(mem %>% filter(spec_id == sub("^s3_pairs__(.*)__(ag|gen)_real$", "\\1", spec)) %>%
      select(county_fips), by = "county_fips") else base
    if (grepl("excl_contested", spec)) d <- d %>% filter(!state_fips %in% contested)
    d <- bite_complete(d, c(Y[[outcome]], term)); mean(d$aewr_ppi)
  }) %>% ungroup() %>%
  transmute(spec, outcome, term, cluster, clusters, beta_mw = estimate, se, mean_y, mean_aewr,
    implied_aewr_slope = -estimate,
    implied_aewr_elasticity = -estimate * mean_aewr / mean_y,
    implied_low = -(estimate + qt(.975, clusters - 1) * se) * mean_aewr / mean_y,
    implied_high = -(estimate - qt(.975, clusters - 1) * se) * mean_aewr / mean_y,
    implied_mde_elasticity = 2.8 * se * mean_aewr / abs(mean_y))
id_write(tr, out_dir, paste0(T2, "_relprice_translation.csv"))
print(as.data.frame(tr %>% filter(cluster == "state_fips")), digits = 3)
