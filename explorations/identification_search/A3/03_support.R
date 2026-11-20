# Identification search, strategy A3 (Phase 3 design, step 1b): support
# statistics, treated pre-rule cells, the predetermined pass-through bound,
# pre-period residual variance and the node-period share (gate DGP rule).
# Builds the design frames used by 04_power.R.
#
# Usage (repository root, after 02_units_dose.R):
#   Rscript --vanilla explorations/identification_search/A3/03_support.R "$S"
# Writes $S/A3/a3_support.csv, a3_dose_by_fy.csv, a3_treated_cells.csv,
# a3_passthrough_bound.csv, a3_node_share.csv, a3_design_frames.rds.
#
# Integrity: outcomes enter only for FY2020-FY2022 (PPML weights mu_i and
# FE-only residual variance of the linearized outcome; pre-rule offered wages
# for the pass-through bound). The policy frames carry the dose only. No
# outcome is related to the dose.
here::i_am("code/paths.R")
source(here::here("code", "paths.R"))
source(path_code("c00_shared", "geography.R"))
suppressPackageStartupMessages({library(arrow); library(data.table); library(fixest)})
source(path_code("designs", "border_discontinuity", "power.R"))
source("explorations/identification_search/shared/power_harness.R")
setFixest_nthreads(1); setFixest_notes(FALSE)
args <- commandArgs(trailingOnly = TRUE)
S <- normalizePath(args[1]); out <- file.path(S, "A3")
if (startsWith(paste0(S, "/"), paste0(normalizePath(path_root()), "/"))) stop("Outputs must be outside the repository")
PRE <- 2020:2022

units <- as.data.table(read_parquet(file.path(out, "a3_units.parquet")))
dose <- as.data.table(read_parquet(file.path(out, "a3_unit_dose.parquet")))
pre <- as.data.table(read_parquet(file.path(out, "a3_pre_outcomes.parquet")))
stopifnot(all(pre$fy %in% PRE))
shc <- grep("^sh_", names(units), value = TRUE)

# ---- Frames ---------------------------------------------------------------------
# policy: unit x FY with dose x (log applicable floor), FE columns, clusters, node, group, weight mu.
# pre: unit x FY2020-FY2022 with the linearized outcome y/mu - 1 (PPML score linearization).
build <- function(wn, fys, dose_col = "x", mu_col = "mu", drop_fy = integer(0), controls = TRUE, outcome = "cert_hours", ctl_cols = shc) {
  u <- units[get(mu_col) > 0]
  pol <- merge(dose[window == wn, .(unit_id, year = fy, x = get(dose_col))],
               u[, c("unit_id", "employer_id_balanced", "state_fips", "type", "node", mu_col, unique(c(shc, "s_nf"))), with = FALSE], by = "unit_id")
  pol <- pol[year %in% setdiff(fys, drop_fy)]
  setnames(pol, mu_col, "w")
  ycol <- if (wn == "rq13d") paste0(outcome, "_rq13d") else outcome
  pr <- CJ(unit_id = u$unit_id, year = PRE)
  pr <- merge(pr, pre[, .(unit_id = paste(employer_id_balanced, state_fips, sep = "|"), year = fy, yy = get(ycol))], by = c("unit_id", "year"), all.x = TRUE)
  pr[, yy := fcoalesce(yy, 0)]
  pr <- merge(pr, u[, c("unit_id", "state_fips", "type", "node", mu_col), with = FALSE], by = "unit_id")
  setnames(pr, mu_col, "w")
  if (outcome != "cert_hours" || wn == "rq13d") {  # weights = this outcome's own pre-period mean
    m <- pr[, .(w2 = mean(yy)), by = unit_id]; pr <- merge(pr[, w := NULL], m, by = "unit_id")[w2 > 0]; setnames(pr, "w2", "w")
    pol <- merge(pol[, w := NULL], m, by = "unit_id")[w2 > 0]; setnames(pol, "w2", "w")
  }
  pr[, y := yy / w - 1]
  pol[, `:=`(state_year = paste(state_fips, year), group = type, unit = unit_id)]
  pr[, `:=`(state_year = paste(state_fips, year), group = type, unit = unit_id)]
  # FWL: residualize the dose on the declared share x FY controls and the FE (weighted)
  if (controls) {
    yrs <- sort(unique(pol$year)); ctl <- character(0)
    for (v in ctl_cols) for (t in yrs[-1]) { nm <- paste0(v, "_", t); pol[, (nm) := get(v) * (year == t)]; ctl <- c(ctl, nm) }
    ctl <- ctl[vapply(ctl, function(v) any(pol[[v]] != 0), logical(1))]
    fit <- feols(as.formula(paste("x ~", paste(ctl, collapse = " + "), "| unit + state_year")), data = pol, weights = ~w)
    pol[, x_raw := x]; pol[, x := resid(fit)]
    pol[, (ctl) := NULL]
  }
  setorder(pol, unit, year); setorder(pr, unit, year)
  list(pol = as.data.frame(pol), pre = as.data.frame(pr), y = "y", w = "w", window = wn)
}
FE <- c("unit", "state_year")
support <- function(tag, f, fe = FE) {
  p <- f$pol; w <- p$w / mean(p$w)
  xw <- drop(fixest::demean(cbind(x = p$x), f = lapply(fe, function(v) factor(p[[v]])), weights = w, tol = 1e-10, nthreads = 1L)) * sqrt(w)
  sxx <- sum(xw^2)
  lev_s <- tapply(xw^2, p$state_fips, sum); lev_n <- tapply(xw^2, p$node, sum)
  pr <- f$pre; wp <- pr$w / mean(pr$w)
  r <- drop(fixest::demean(cbind(y = pr$y), f = lapply(fe, function(v) factor(pr[[v]])), weights = wp, tol = 1e-10, nthreads = 1L)) * sqrt(wp)
  data.table(run = tag, window = f$window, units = uniqueN(p$unit), periods = uniqueN(p$year), rows = nrow(p),
    units_with_dose = uniqueN(p$unit[abs(xw) > 1e-9]), states = uniqueN(p$state_fips), states_with_dose = sum(lev_s > 1e-12 * sxx),
    nodes = uniqueN(p$node), dose_sd_after_fe = sqrt(sxx / sum(w)), effective_states = sxx^2 / sum(lev_s^2),
    max_state_leverage = max(lev_s) / sxx, max_state = names(lev_s)[which.max(lev_s)],
    effective_nodes = sxx^2 / sum(lev_n^2), max_node_leverage = max(lev_n) / sxx, max_node = names(lev_n)[which.max(lev_n)],
    preperiod_residual_sd = sqrt(sum(r^2) / sum(wp)), pre_rows = nrow(pr))
}

frames <- list(
  primary = build("rfy", 2020:2025),
  no_controls = build("rfy", 2020:2025, controls = FALSE),
  nonfield_share_controls = build("rfy", 2020:2025, ctl_cols = "s_nf"),
  drop_fy2023 = build("rfy", 2020:2025, drop_fy = 2023L),
  statutory_floor = build("rfy", 2020:2025, dose_col = "x_stat"),
  no_ramp = build("rfy", 2020:2025, dose_col = "x_noramp"),
  no_teche = build("rfy", 2020:2025, dose_col = "x_noteche"),
  rq13d_fy2026 = build("rq13d", 2020:2026, mu_col = "mu_rq13d"),
  rq13d_fy2026_sk2 = build("rq13d", 2020:2026, dose_col = "x_sk2", mu_col = "mu_rq13d"),
  req_pos = build("rfy", 2020:2025, outcome = "req_pos"))
sup <- rbindlist(lapply(names(frames), function(k) support(k, frames[[k]])))
sup_alt <- list()
# employer x FY FE (within-employer, across worksite states and job mixes): multi-unit employers only
f <- frames$primary; p <- as.data.table(f$pol)
multi <- p[, uniqueN(unit), by = employer_id_balanced][V1 > 1, employer_id_balanced]
fe_emp <- list(pol = as.data.frame(p[employer_id_balanced %in% multi][, emp_year := paste(employer_id_balanced, year)]),
  pre = as.data.frame(merge(as.data.table(f$pre), units[, .(unit_id, employer_id_balanced)], by.x = "unit", by.y = "unit_id")[employer_id_balanced %in% multi][, emp_year := paste(employer_id_balanced, year)]),
  window = "rfy")
sup_alt[[1]] <- support("employer_x_fy_fe", fe_emp, fe = c("unit", "emp_year", "state_year"))
sup <- rbind(sup, rbindlist(sup_alt), fill = TRUE)

# Cell secondary: employer x state x {field, nonfield}, nonfield hours assigned by order SOC in mixed units
cd <- as.data.table(read_parquet(file.path(out, "a3_cell_dose.parquet")))
pc <- as.data.table(read_parquet(file.path(out, "a3_pre_cell_outcomes.parquet")))
cpol <- cd[, .(unit = cell_id, year = fy, x, w = mu, state_fips, employer_id_balanced, kind)]
cpol <- merge(cpol, units[, .(employer_id_balanced, state_fips, node)], by = c("employer_id_balanced", "state_fips"))
cpol[, `:=`(state_year = paste(state_fips, year), emp_year = paste(employer_id_balanced, year), node = paste(state_fips, kind))]
cpr <- CJ(unit = unique(cpol$unit), year = PRE)
cpr <- merge(cpr, pc[, .(unit = paste(employer_id_balanced, state_fips, kind, sep = "|"), year = fy, yy = cert_hours)], by = c("unit", "year"), all.x = TRUE)
cpr <- merge(cpr[, yy := fcoalesce(yy, 0)], unique(cpol[, .(unit, w, state_fips, node, employer_id_balanced)]), by = "unit")
cpr[, `:=`(y = yy / w - 1, state_year = paste(state_fips, year), emp_year = paste(employer_id_balanced, year))]
cf <- list(pol = as.data.frame(cpol), pre = as.data.frame(cpr), window = "rfy")
sup <- rbind(sup, support("cells_state_fy", cf), fill = TRUE)
multi_c <- cpol[, uniqueN(kind), by = employer_id_balanced][V1 > 1, employer_id_balanced]
cf2 <- list(pol = as.data.frame(cpol[employer_id_balanced %in% multi_c]), pre = as.data.frame(cpr[employer_id_balanced %in% multi_c]), window = "rfy")
sup <- rbind(sup, support("cells_employer_x_fy_fe", cf2, fe = c("unit", "emp_year")), fill = TRUE)
fwrite(sup, file.path(out, "a3_support.csv"))
print(sup[, .(run, units, units_with_dose, states_with_dose, dose_sd_after_fe, effective_states, max_state_leverage, max_state, effective_nodes, max_node_leverage, max_node, preperiod_residual_sd)])

# ---- Dose description by FY (nonfield units, PPML-weighted) --------------------------
wsd <- function(x, w) sqrt(sum(w * (x - sum(w * x) / sum(w))^2) / sum(w))
dd <- merge(dose, units[, .(unit_id, mu, mu_rq13d, s_nf)], by = "unit_id")[s_nf > 0]
dfy <- dd[, .(units = .N, D_mean = weighted.mean(D, mu), D_sd = wsd(D, mu), gap_mean = weighted.mean(D / s_nf, mu * s_nf),
              gap_sd = wsd(D / s_nf, mu * s_nf), gap_p10 = quantile(D / s_nf, 0.1), gap_p90 = quantile(D / s_nf, 0.9),
              share_neg_gap = weighted.mean(D < -1e-9, mu)), by = .(window, fy)][order(window, fy)]
fwrite(dfy, file.path(out, "a3_dose_by_fy.csv")); print(dfy)

# ---- Treated pre-rule cells (employer x state x nonfield SOC) and their hours -----------
cells <- as.data.table(read_parquet(file.path(out, "a3_cells.parquet")))
fl <- as.data.table(read_parquet(file.path(out, "a3_floors_daily.parquet")))
g24 <- fl[d == as.IDate("2024-01-15"), .(gap24 = log(pmax(oews, F)) - log(F), oews_na = is.na(oews)), by = .(state_fips, soc)]
nc <- merge(cells[field == FALSE], g24, by = c("state_fips", "soc"), all.x = TRUE)
nc[, gap24 := fcoalesce(gap24, 0)]
tc <- rbind(
  nc[, .(group = "all nonfield cells", cells = .N, employers = uniqueN(employer_id_balanced), states = uniqueN(state_fips), state_soc_shock_cells = uniqueN(paste(state_fips, soc)), hours_fy2020_22 = sum(hours))],
  nc[gap24 > 0.005, .(group = "positive 2024 gap (treated)", cells = .N, employers = uniqueN(employer_id_balanced), states = uniqueN(state_fips), state_soc_shock_cells = uniqueN(paste(state_fips, soc)), hours_fy2020_22 = sum(hours))],
  nc[gap24 <= 0.005, .(group = "zero gap (OEWS <= FLS or no rate; placebo group)", cells = .N, employers = uniqueN(employer_id_balanced), states = uniqueN(state_fips), state_soc_shock_cells = uniqueN(paste(state_fips, soc)), hours_fy2020_22 = sum(hours))],
  nc[, .(cells = .N, employers = uniqueN(employer_id_balanced), states = uniqueN(state_fips), state_soc_shock_cells = uniqueN(paste(state_fips, soc)), hours_fy2020_22 = sum(hours)), by = .(group = paste("SOC", soc))][order(-hours_fy2020_22)][1:12],
  cells[field == TRUE, .(group = "field cells (controls)", cells = .N, employers = uniqueN(employer_id_balanced), states = uniqueN(state_fips), state_soc_shock_cells = NA_integer_, hours_fy2020_22 = sum(hours))])
tot_hours <- cells[, sum(hours)]
tc[, hours_share := hours_fy2020_22 / tot_hours]
fwrite(tc, file.path(out, "a3_treated_cells.csv")); print(tc)

# ---- Predetermined pass-through bound from pre-rule offers ---------------------------
# Unit's FY2022 nonfield offer (hours-weighted log) carried at its FY2022 premium over the field floor
# into FY2024; predicted post offer = max(premium-carried offer, FY2024 nonfield floor). Ratio of the
# predicted log offer change (net of the field floor change) to the dose gap: the pass-through if
# employers pay max(old wage path, new floor) and code the same SOC.
of <- as.data.table(read_parquet(file.path(out, "a3_pre_offers_nonfield.parquet")))[fy == 2022]
fyN <- fl[, .(lN = mean(lN), lF = mean(log(F))), by = .(state_fips, soc, fy = year(d) + (month(d) >= 10L))][fy %in% c(2022, 2024)]
fyN <- dcast(fyN, state_fips + soc ~ fy, value.var = c("lN", "lF"))
pb <- merge(of, fyN, by = c("state_fips", "soc"))
pb[, `:=`(prem = pmax(lw - lF_2022, 0), gap = lN_2024 - lF_2024)]
pb[, pred := pmax(prem, gap)]
pt <- data.table(cells = nrow(pb), hours = sum(pb$h), share_at_floor_pre = pb[, weighted.mean(prem < 0.005, h)],
  mean_premium_pre = pb[, weighted.mean(prem, h)], mean_gap = pb[, weighted.mean(gap, h)],
  predicted_pass_through = pb[gap > 0, sum(h * (pred - prem)) / sum(h * gap)])
fwrite(pt, file.path(out, "a3_passthrough_bound.csv")); print(pt)

# ---- Node-period share (gate DGP rule) ---------------------------------------------
ns <- rbindlist(lapply(c("primary", "rq13d_fy2026"), function(k) {
  f <- frames[[k]]
  cbind(run = k, node = "state x unit type", ish_node_period_share(f$pre, "y", "unit", "year", FE, "node", weights = "w"))
}))
fwrite(ns, file.path(out, "a3_node_share.csv")); print(ns)
saveRDS(c(frames, list(cells = cf, employer_fe = fe_emp)), file.path(out, "a3_design_frames.rds"))
