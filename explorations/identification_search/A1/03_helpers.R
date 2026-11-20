# Identification search, strategy A1 (Phases 4-5): shared helpers sourced by
# 04_falsification.R, 05_estimation.R and 06_estimates_ledger.R.
#
# Every quantity implements explorations/identification_search/A1/registry.json
# (frozen; SHA-256 checked by a1_check_registry() before any data are read).
# Inference conventions follow the design-stage harness
# (explorations/identification_search/shared/power_harness.R): state CRV1 with
# t(G - 1) critical values, exact leave-one-state-out jackknife (CV3J, refits,
# centred at the mean of the leave-out estimates, (G - 1)/G scaling), former
# region CRV1 with t(G_region - 1).
here::i_am("code/paths.R")
source(here::here("code", "paths.R"))
source(path_code("c00_shared", "geography.R"))
suppressPackageStartupMessages({library(arrow); library(dplyr); library(tidyr); library(readr); library(fixest)})
setFixest_nthreads(1)
setFixest_notes(FALSE)

A1_DIR <- "explorations/identification_search/A1"
REG_PATH <- file.path(A1_DIR, "registry.json")
REG_SHA <- "9a1182bdcf177c3821a9a89651af7fb5e45a119ca8ced6a9c447c49cdc750eed"
a1_check_registry <- function() {
  want <- strsplit(readLines(file.path(A1_DIR, "registry.sha256")), " +")[[1]][1]
  got <- digest::digest(file = REG_PATH, algo = "sha256")
  if (!identical(want, REG_SHA) || !identical(got, REG_SHA)) stop("Registry hash mismatch: ", got)
  message("Registry SHA-256 verified: ", got)
  invisible(got)
}

a1_out_dir <- function(S) {
  S <- normalizePath(S); out <- file.path(S, "A1")
  if (startsWith(paste0(S, "/"), paste0(normalizePath(path_root()), "/"))) stop("Outputs must be outside the repository")
  dir.create(out, showWarnings = FALSE); out
}

# Registry estimand constants
W_BAR <- 15.2607
RATIO <- c(central = 1.2401, low = 1.2145, high = 1.3834, extreme = 1.76)
CBAR <- RATIO * W_BAR  # central 18.924
CAL <- 2020:2025
# Field and livestock SOCs (2023 AEWR rule list; triage convention)
FIELD_SOC <- c("45-2041", "45-2091", "45-2092", "45-2093", "53-7064", "45-2099")
# Registry F5(b): minimum wage >= U.S. skill-I rate
MW_FLOORED <- c("06", "09", "12", "23", "29", "36", "44", "53")

# ---- Data ------------------------------------------------------------------------
a1_doses <- function(S) {
  d <- read_parquet(file.path(S, "A1", "a1_county_doses.parquet"))
  stopifnot(is.character(d$county_fips), !anyDuplicated(d$county_fips), all(nchar(d$county_fips) == 5))
  d
}

a1_outcomes <- function(S, years, cols) {
  read_parquet(file.path(S, "shared", "outcomes_county_fy.parquet"), col_select = c("county_fips", "year", cols)) %>%
    filter(year %in% years)
}

# County x FY frame on the registry universe (dose file counties), zeros filled.
a1_panel <- function(S, doses, years, cols) {
  oc <- a1_outcomes(S, years, cols)
  p <- tidyr::crossing(doses, year = years) %>% left_join(oc, by = c("county_fips", "year"))
  for (v in cols) p[[v]] <- if (is.logical(p[[v]])) coalesce(p[[v]], FALSE) else coalesce(p[[v]], 0)
  p %>% mutate(state_year = paste(state_fips, year), region_year = paste(aewr_region_id, year))
}

# PPML calibration weights: FY2020-FY2025 county mean rq13d certified hours.
a1_ppml_weights <- function(S) {
  a1_outcomes(S, CAL, "cert_hours_rq13d") %>% tidyr::complete(county_fips, year = CAL, fill = list(cert_hours_rq13d = 0)) %>%
    group_by(county_fips) %>% summarise(ppml_w = mean(coalesce(cert_hours_rq13d, 0)), .groups = "drop")
}

# FY2025 rq13d county covariates (predetermined): field-SOC hours share, piece-rate
# positions share, experience (>= 3 months) hours share. SOC and piece rate come
# from h2a_disclosure_records (case_number, fiscal_year); WORK_EXPERIENCE_MONTHS
# from the raw FY2025 OFLC disclosure xlsx (case_number). Counties without FY2025
# certified rows get the state FY2025-hours-weighted share (the b_c imputation
# rule); *_known flags the observed values.
a1_fy2025_covariates <- function(S, doses) {
  cr <- read_parquet(file.path(S, "shared", "outcomes_case_rows.parquet"),
    col_select = c("case_number", "fiscal_year", "fy_receipt", "in_rq13d", "county_fips", "cert_pos", "cert_hours")) %>%
    filter(fy_receipt == 2025, in_rq13d, cert_hours > 0, county_fips != "00000")
  disc <- arrow::open_dataset(path_int("h2a_disclosure_records.parquet")) %>% filter(fiscal_year %in% 2024:2025) %>%
    select(case_number, fiscal_year, soc_code, piece_rate_offer) %>% collect() %>%
    mutate(soc7 = substr(gsub("[^0-9-]", "", coalesce(soc_code, "")), 1, 7),
      piece = coalesce(suppressWarnings(as.numeric(piece_rate_offer)), 0) > 0)
  stopifnot(!anyDuplicated(disc[c("case_number", "fiscal_year")]))
  xl <- readxl::read_excel(path_raw("h2a", "H-2A_Disclosure_Data_FY2025_Q4.xlsx"), col_types = "text")
  xl <- tibble(case_number = xl$CASE_NUMBER, exp_months = suppressWarnings(as.numeric(xl$WORK_EXPERIENCE_MONTHS))) %>%
    distinct(case_number, .keep_all = TRUE)
  cr <- cr %>% left_join(disc %>% select(case_number, fiscal_year, soc7, piece), by = c("case_number", "fiscal_year")) %>%
    left_join(xl, by = "case_number")
  cov_match <- cr %>% summarise(rows = n(), hours = sum(cert_hours), soc_matched_hours = sum(cert_hours[!is.na(soc7) & soc7 != ""]) / hours,
    exp_matched_hours = sum(cert_hours[!is.na(exp_months)]) / hours, piece_matched_hours = sum(cert_hours[!is.na(piece)]) / hours)
  cty <- cr %>% group_by(county_fips) %>% summarise(
    h25 = sum(cert_hours),
    field_share = sum(cert_hours[soc7 %in% FIELD_SOC]) / sum(cert_hours[!is.na(soc7) & soc7 != ""]),
    piece_share = sum(cert_pos[piece %in% TRUE]) / sum(cert_pos[!is.na(piece)]),
    exp3_share = sum(cert_hours[!is.na(exp_months) & exp_months >= 3]) / sum(cert_hours[!is.na(exp_months)]),
    .groups = "drop")
  out <- doses %>% select(county_fips, state_fips) %>% left_join(cty, by = "county_fips")
  imp <- function(v) {
    out %>% group_by(state_fips) %>% mutate(st = sum((h25 * .data[[v]])[is.finite(.data[[v]])], na.rm = TRUE) /
      sum(h25[is.finite(.data[[v]])], na.rm = TRUE)) %>% ungroup() %>%
      transmute(val = ifelse(is.finite(.data[[v]]), .data[[v]], st), known = is.finite(.data[[v]]))
  }
  for (v in c("field_share", "piece_share", "exp3_share")) {
    z <- imp(v); out[[paste0(v, "_imp")]] <- z$val; out[[paste0(v, "_known")]] <- z$known
  }
  list(cov = out, match = cov_match)
}

# ---- Estimation -----------------------------------------------------------------
pval_t <- function(b, se, df) 2 * pt(-abs(b / se), df)

# One fit with state CRV1 (primary), state jackknife (CV3J, refits) and former
# region CRV1. Returns one row per term plus jackknife draws.
a1_est <- function(spec, d, y, xs, fe, method = "ppml", wvar = NULL, terms = xs, ctrl = NULL,
  jk = TRUE, role = "", declared = TRUE, note = "") {
  fml <- as.formula(paste(y, "~", paste(c(xs, ctrl), collapse = " + "), "|", paste(fe, collapse = " + ")))
  fitf <- function(dd) {
    a <- list(fml = fml, data = dd, vcov = ~state_fips)
    if (!is.null(wvar)) a$weights <- as.formula(paste0("~", wvar))
    do.call(if (method == "ppml") fixest::fepois else fixest::feols, a)
  }
  m <- fitf(d)
  dd <- d[obs(m), , drop = FALSE]
  G <- n_distinct(dd$state_fips); GR <- n_distinct(dd$aewr_region_id)
  b <- coef(m)[terms]
  se_s <- se(summary(m, cluster = ~state_fips))[terms]
  se_r <- se(summary(m, cluster = ~aewr_region_id))[terms]
  B <- NULL; se_j <- rep(NA_real_, length(terms)); Vj <- NULL; jk_fail <- 0L
  if (jk) {
    st <- sort(unique(dd$state_fips))
    B <- t(vapply(st, function(s) {
      r <- tryCatch(coef(fitf(d[d$state_fips != s, , drop = FALSE]))[terms], error = function(e) rep(NA_real_, length(terms)))
      as.numeric(r)
    }, numeric(length(terms))))
    if (length(terms) == 1) B <- matrix(B, ncol = 1)
    rownames(B) <- st; colnames(B) <- terms
    ok <- stats::complete.cases(B); jk_fail <- sum(!ok); Bo <- B[ok, , drop = FALSE]; Gj <- nrow(Bo)
    Dv <- sweep(Bo, 2, colMeans(Bo)); Vj <- (Gj - 1) / Gj * crossprod(Dv)
    se_j <- sqrt(diag(Vj))
  }
  tq <- qt(0.975, G - 1)
  wts <- if (is.null(wvar)) rep(1, nrow(dd)) else dd[[wvar]]
  rows <- tibble(spec_id = spec, role = role, declared = declared, outcome = y, term = terms,
    model = paste0(ifelse(method == "ppml", "PPML", ifelse(is.null(wvar), "OLS", "WLS")), " | FE: ", paste(fe, collapse = " + "),
      if (!is.null(ctrl)) paste0(" | controls: ", paste(ctrl, collapse = " + ")) else ""),
    estimate = as.numeric(b), se = as.numeric(se_s), ci_low = as.numeric(b - tq * se_s), ci_high = as.numeric(b + tq * se_s),
    p = pval_t(as.numeric(b), as.numeric(se_s), G - 1), inference_method = "state_fips_CRV1_t(G-1)",
    se_jackknife = se_j, p_jackknife = pval_t(as.numeric(b), se_j, G - 1),
    se_region_crv1 = as.numeric(se_r), p_region_crv1 = pval_t(as.numeric(b), as.numeric(se_r), GR - 1),
    n = nobs(m), units = n_distinct(dd[[fe[1]]]), clusters = G, region_clusters = GR, jk_failed_refits = jk_fail,
    mean_y = sum(wts * dd[[y]]) / sum(wts), loglik = as.numeric(logLik(m)), note = note)
  message(sprintf("%-38s %-22s %-24s b=%9.5f se=%8.5f p=%.3f jk_se=%8.5f n=%d G=%d", spec, y, terms, rows$estimate, rows$se, rows$p,
    rows$se_jackknife, rows$n, G))
  list(rows = rows, m = m, B = B, Vj = Vj, dd = dd)
}

# Fieller / Anderson-Rubin-type set for beta = eta / pi with a 2x2 covariance V
# (order eta, pi): {beta : (eta - beta pi)^2 <= t^2 (V11 - 2 beta V12 + beta^2 V22)}.
a1_fieller <- function(eta, pi, V, tq) {
  a <- pi^2 - tq^2 * V[2, 2]; bq <- -2 * (eta * pi - tq^2 * V[1, 2]); cq <- eta^2 - tq^2 * V[1, 1]
  disc <- bq^2 - 4 * a * cq
  if (a > 0 && disc >= 0) {
    r <- sort(c((-bq - sqrt(disc)) / (2 * a), (-bq + sqrt(disc)) / (2 * a)))
    list(type = "bounded", low = r[1], high = r[2])
  } else if (a < 0 && disc >= 0) {
    r <- sort(c((-bq - sqrt(disc)) / (2 * a), (-bq + sqrt(disc)) / (2 * a)))
    list(type = "union_of_rays", low = r[1], high = r[2])  # (-Inf, low] U [high, Inf)
  } else if (a < 0) list(type = "whole_line", low = -Inf, high = Inf)
  else list(type = "empty", low = NA_real_, high = NA_real_)
}

a1_write <- function(x, out, name) { readr::write_csv(x, file.path(out, name)); invisible(x) }
