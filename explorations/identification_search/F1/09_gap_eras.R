# Identification search, strategy F1 (Phase 5): G2 (cross-state check of the
# gap-model premise) and G3 (era descriptives). Registry gap_model_implications.
#
#   G2: hours-weighted correlation across states with >= 10 above-floor M26 orders of
#       the state b26(a25) with gap = a25_s - ACS 2024 state agricultural mean hourly wage;
#       also with the P4 relative shortfall E[(a25 - w*)+]/a25 (confirmation states with
#       >= 10 above-floor orders; state values from the single confirmation fit). Weights:
#       state M26 hours. 999-draw bootstrap over states (seed 20261102), percentile CI.
#       Premise supported only if the CI excludes 0 with a positive sign.
#   G3: eras FY2008 (all hourly non-piece), FY2011-13 (all hourly non-piece; SOC not reported),
#       FY2014-18, FY2019-22, FY2023-25 (field SOC), 48 dose-file states, classes above / at_floor
#       (as M25). Hours share above the AEWR (applicable old-rule floor) and S(m x AEWR),
#       m in {1.05, 1.10, 1.20}; normal Tobit (intercept and common scale, r = log(w / AEWR),
#       hours weights) scale and b(AEWR / 1.1) [EXTRAPOLATION]. 999-draw state bootstrap.
#
# Usage: R_LIBS_USER=... Rscript --vanilla explorations/identification_search/F1/09_gap_eras.R "$S"
source("explorations/identification_search/F1/03_est_helpers.R")
f1_check_registry()
args <- commandArgs(trailingOnly = TRUE)
S <- normalizePath(args[1]); out <- f1_out_dir(S)
o <- read_parquet(file.path(out, "f1_orders.parquet"))
doses <- f1_doses(S); ST48 <- sort(unique(doses$state_fips)); g <- f1_g_frozen(out)
o <- f1_attach_doses(o, doses)
M26 <- f1_M26(o, ST48)

# ---- G2 -------------------------------------------------------------------------------
acs <- read_parquet(path_int("acs_state_ag_wage.parquet")) %>% filter(year == 2024) %>% transmute(state_fips, acs_wage = acs_ag_mean_hourly_wage)
stopifnot(is.character(acs$state_fips))
st <- M26 %>% group_by(state_fips) %>% summarise(n_above = sum(cls == "above"),
  b26_a25 = sum(hours[wage <= pmax(a25d, F_app) + TOL]) / sum(hours), a25 = first(a25d), hours = sum(hours), .groups = "drop") %>%
  filter(n_above >= 10) %>% left_join(acs, by = "state_fips") %>% mutate(gap = a25 - acs_wage)
stopifnot(!anyNA(st$gap))
sf <- read_csv(file.path(out, "f1_p4_state_shortfall.csv"), col_types = cols(state_fips = "c", .default = col_double()))
wcor <- function(x, y, w) { w <- w / sum(w); mx <- sum(w * x); my <- sum(w * y); sum(w * (x - mx) * (y - my)) / sqrt(sum(w * (x - mx)^2) * sum(w * (y - my)^2)) }
g2 <- list()
for (v in c("b26_a25", "p4_rel_shortfall")) {
  d <- if (v == "b26_a25") st %>% transmute(state_fips, x = gap, y = b26_a25, w = hours) else
    st %>% inner_join(sf %>% select(state_fips, rel_shortfall), by = "state_fips") %>% transmute(state_fips, x = gap, y = rel_shortfall, w = hours)
  e <- wcor(d$x, d$y, d$w); set.seed(SEED); cnt <- boot_counts(d$state_fips)
  dr <- apply(cnt, 2, function(cc) { i <- rep(seq_len(nrow(d)), cc); wcor(d$x[i], d$y[i], d$w[i]) })
  ci <- pci(dr)
  g2[[v]] <- tibble(item = paste0("G2_corr_gap_", v), estimate = e, se = sd(dr, na.rm = TRUE), ci_low = ci[1], ci_high = ci[2], n_states = nrow(d),
    premise_supported = ci[1] > 0)
}
g2 <- bind_rows(g2); print(g2)
write_csv(st, file.path(out, "f1_g2_states.csv"))

# ---- G3 -------------------------------------------------------------------------------
eras <- list(`FY2008` = list(fy = 2008, soc = c("unknown", "field", "nonfield")), `FY2011-13` = list(fy = 2011:2013, soc = c("unknown", "field", "nonfield")),
  `FY2014-18` = list(fy = 2014:2018, soc = "field"), `FY2019-22` = list(fy = 2019:2022, soc = "field"), `FY2023-25` = list(fy = 2023:2025, soc = "field"))
set.seed(SEED); cnt <- boot_counts(ST48)
g3 <- list()
for (nm in names(eras)) {
  e <- eras[[nm]]
  d <- o %>% filter(fy %in% e$fy, !new_rule, state_fips %in% ST48, soc_grp %in% e$soc, !piece, cls %in% c("above", "at_floor"))
  ind <- cbind(`share_above_AEWR` = d$cls == "above", `S(1.05 AEWR)` = d$cls == "above" & d$wage > 1.05 * d$F_app + TOL,
    `S(1.10 AEWR)` = d$cls == "above" & d$wage > 1.10 * d$F_app + TOL, `S(1.20 AEWR)` = d$cls == "above" & d$wage > 1.20 * d$F_app + TOL) * 1
  sb <- share_boot(d, ind, ST48, cnt)
  rows <- tibble(item = paste0("G3_", colnames(ind)), era = nm, estimate = sb$est, se = apply(sb$draws, 2, sd),
    ci_low = apply(sb$draws, 2, function(x) pci(x)[1]), ci_high = apply(sb$draws, 2, function(x) pci(x)[2]), n_orders = nrow(d),
    n_states = n_distinct(d$state_fips), label = "descriptive (heavily seen)")
  # Normal Tobit, intercept + common scale (F_ref = F_app under the old rule)
  # Intercept-only likelihood depends on (y, cens, weight) only: collapse to state x y x cens cells and
  # express a bootstrap draw as state multiplicities on the cell weights (identical estimator, no row copies).
  md <- p4_data(d)
  cells <- tibble(state_fips = md$state, y = round(md$y, 10), cens = md$cens, h = md$d$hours) %>%
    group_by(state_fips, y, cens) %>% summarise(h = sum(h), .groups = "drop")
  tob <- function(wt, start = NULL) {
    mc <- list(y = cells$y, cens = cells$cens, X = cbind(b0 = rep(1, nrow(cells))), w = wt / mean(wt), skill2 = rep(FALSE, nrow(cells)), center = 0)
    f <- p4_fit(mc, "normal", FALSE, start = start)
    list(par = f$par, out = c(tobit_scale = exp(f$par[["ls"]]), `b(AEWR/1.1)` = pnorm((log(1 / 1.1) - f$par[["b0"]]) / exp(f$par[["ls"]]))))
  }
  t0 <- tob(cells$h); te <- t0$out
  cc <- cnt[, , drop = FALSE]
  td <- t(vapply(seq_len(NBOOT), function(b) { m <- cc[cells$state_fips, b]; tob(cells$h * m, start = t0$par)$out }, numeric(2)))
  rows <- bind_rows(rows, tibble(item = paste0("G3_", names(te)), era = nm, estimate = unname(te), se = apply(td, 2, sd),
    ci_low = apply(td, 2, function(x) pci(x)[1]), ci_high = apply(td, 2, function(x) pci(x)[2]), n_orders = nrow(d),
    n_states = n_distinct(d$state_fips), label = "EXTRAPOLATION (normal Tobit, intercept and common scale)"))
  g3[[nm]] <- rows; message("era done: ", nm)
}
g3 <- bind_rows(g3) %>% mutate(inference = "999-draw state-cluster bootstrap, percentile")
print(g3 %>% select(era, item, estimate, ci_low, ci_high, n_orders), n = 50, width = 200)
write_csv(g2, file.path(out, "f1_g2.csv")); write_csv(g3, file.path(out, "f1_g3.csv"))
message("done ", format(Sys.time()))
