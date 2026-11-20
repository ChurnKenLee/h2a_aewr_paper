# Identification search, strategy F1 (Phase 5): P4 single fit of the chosen model on
# the 32 confirmation states, model-uncertainty band (all 4 candidates on the
# confirmation states), V2 (LOSO calibration across confirmation states), V3
# (FY2025 -> FY2026 forecast), S7 (all 48 states). Label: EXTRAPOLATION below the floor.
#
# Usage: R_LIBS_USER=... Rscript --vanilla explorations/identification_search/F1/07_p4_confirmation.R "$S"
# Requires $S/F1/f1_p4_choice.json (06_p4_tuning.R). Writes f1_p4.csv, f1_p4_band.csv,
# f1_v2.csv, f1_v2_states.csv, f1_v3.csv, f1_p4.rds.
#
# Implementation readings (fixed before running; registry silent):
#   - P4 outputs are hours-weighted over the fit sample's M26 orders; bootstrap refits
#     keep the covariate centring constant of the point fit.
#   - V2/V3 "above-floor hours share" = hours share of class 'above' (registry usage);
#     the model's prediction is P(w* > F_app + 0.05). A model-consistent alternative
#     (observed = exact observations above F_app + 0.05, i.e. incl. at_old_aewr) is
#     reported as a post-outcome reading.
#   - V3 quantiles: hours-weighted 50th/90th percentiles of offer / F_app among class
#     'above' M26 confirmation orders vs. the model's hours-weighted mixture of
#     w*/F_app conditional on w* > F_app + 0.05.
source("explorations/identification_search/F1/03_est_helpers.R")
f1_check_registry()
args <- commandArgs(trailingOnly = TRUE)
S <- normalizePath(args[1]); out <- f1_out_dir(S)
ch <- fromJSON(file.path(out, "f1_p4_choice.json")); CH <- P4_CANDS %>% filter(model == ch$chosen)
message("chosen model (from exploration tuning): ", CH$model)
o <- read_parquet(file.path(out, "f1_orders.parquet"))
doses <- f1_doses(S); ST48 <- sort(unique(doses$state_fips))
g <- f1_g_frozen(out)
o <- f1_attach_doses(o, doses)
M26all <- f1_M26(o, ST48)
CF <- M26all %>% filter(part == "confirmation"); cf_states <- sort(unique(CF$state_fips)); stopifnot(length(cf_states) == 32)

boot_fit_outputs <- function(D, fam, scale_k, states, B = NBOOT) {
  md <- p4_data(D); fit <- p4_fit(md, fam, scale_k); est <- p4_outputs(fit, md)
  set.seed(SEED); cnt <- boot_counts(states, B); sidx <- split(seq_len(nrow(D)), D$state_fips)[states]
  dr <- t(vapply(seq_len(B), function(b) {
    idx <- unlist(rep(sidx, cnt[, b]), use.names = FALSE)
    mb <- p4_data(D[idx, ], center = md$center); fb <- p4_fit(mb, fam, scale_k, start = fit$par)
    p4_outputs(fb, mb)
  }, numeric(length(est))))
  list(fit = fit, md = md, est = est, draws = dr)
}
tab_out <- function(r, scope) tibble(item = paste0("P4_", names(r$est)), scope = scope, estimate = unname(r$est),
  se = apply(r$draws, 2, sd), ci_low = apply(r$draws, 2, function(x) pci(x)[1]), ci_high = apply(r$draws, 2, function(x) pci(x)[2]),
  n_orders = nrow(r$md$d), n_states = n_distinct(r$md$d$state_fips), censored_hours_share = sum(r$md$d$hours[r$md$cens]) / sum(r$md$d$hours))

# ---- P4 chosen model on confirmation states ----------------------------------------------
t0 <- Sys.time()
P4 <- boot_fit_outputs(CF, CH$fam, CH$scale_k, cf_states)
p4 <- tab_out(P4, paste("confirmation states (32), chosen", CH$model))
message("P4 bootstrap done in ", round(difftime(Sys.time(), t0, units = "mins"), 1), " min")
print(p4, width = 200)

# ---- Model-uncertainty band: all 4 candidates on confirmation states --------------------------
band <- bind_rows(lapply(seq_len(nrow(P4_CANDS)), function(j) {
  cd <- P4_CANDS[j, ]; md <- p4_data(CF); fit <- p4_fit(md, cd$fam, cd$scale_k); r <- p4_outputs(fit, md)
  tibble(model = cd$model, stat = names(r), value = unname(r), nll = fit$nll, conv = fit$conv)
}))
bsum <- band %>% filter(grepl("^b_|^median|^shortfall", stat)) %>% group_by(stat) %>%
  summarise(band_min = min(value), band_max = max(value), .groups = "drop") %>% mutate(item = paste0("P4_", stat))
p4 <- p4 %>% left_join(bsum %>% select(item, band_min, band_max), by = "item")
print(band %>% filter(grepl("^b_|median|shortfall", stat)) %>% tidyr::pivot_wider(id_cols = stat, names_from = model, values_from = value), width = 200)

# ---- V2: LOSO over confirmation states ------------------------------------------------------
v2s <- bind_rows(lapply(cf_states, function(s) {
  tr <- p4_data(CF %>% filter(state_fips != s)); fit <- p4_fit(tr, CH$fam, CH$scale_k, start = P4$fit$par)
  te <- p4_data(CF %>% filter(state_fips == s), center = tr$center); d <- te$d
  pr <- 1 - p4_cdf(fit, te, d$F_app + TOL)
  tibble(state_fips = s, hours = sum(d$hours), pred_above = sum(d$hours * pr) / sum(d$hours),
    obs_above = sum(d$hours[d$cls == "above"]) / sum(d$hours),
    obs_exact_above_Fapp = sum(d$hours[d$cls %in% c("above", "at_old_aewr") & d$wage > d$F_app + TOL]) / sum(d$hours))
}))
cal <- function(v, ycol) { b <- wls(v$pred_above, v[[ycol]], v$hours); c(intercept = b[[1]], slope = b[[2]]) }
v2 <- list()
for (yc in c("obs_above", "obs_exact_above_Fapp")) {
  e <- cal(v2s, yc); set.seed(SEED); cnt <- boot_counts(v2s$state_fips)
  dr <- t(apply(cnt, 2, function(cc) { i <- rep(seq_len(nrow(v2s)), cc); cal(v2s[i, ], yc) }))
  ci_i <- pci(dr[, "intercept"]); ci_s <- pci(dr[, "slope"])
  v2[[yc]] <- tibble(item = c("V2_calibration_intercept", "V2_calibration_slope"), observed = yc,
    estimate = c(e[["intercept"]], e[["slope"]]), se = c(sd(dr[, "intercept"]), sd(dr[, "slope"])),
    ci_low = c(ci_i[1], ci_s[1]), ci_high = c(ci_i[2], ci_s[2]), n_states = nrow(v2s),
    pass = (ci_s[1] <= 1 & ci_s[2] >= 1) & (ci_i[1] <= 0 & ci_i[2] >= 0),
    declared = yc == "obs_above")
}
v2 <- bind_rows(v2); print(v2, width = 200)

# ---- V3: FY2025 -> FY2026 forecast ------------------------------------------------------------
C25 <- f1_M25(o, ST48) %>% filter(part == "confirmation")
wq <- function(x, w, p) { o_ <- order(x); x <- x[o_]; cw <- cumsum(w[o_]) / sum(w); x[which(cw >= p)[1]] }
v3_stats <- function(D25, D26, start = NULL) {
  m25 <- p4_data(D25); f25 <- p4_fit(m25, CH$fam, CH$scale_k, start = start)
  pd <- p4_data(D26, center = m25$center, Fref_col = "a25d"); d <- pd$d; h <- d$hours
  ms <- p4_mu_s(f25, pd); P <- FAM[[CH$fam]]
  Fr <- function(thr) P$p((log(thr / ((1 + g) * d$a25d)) - ms$mu) / ms$s)   # P(w*26 <= thr)
  pa <- 1 - Fr(d$F_app + TOL); pb <- 1 - Fr(d$a25d * (1 + g) + TOL)
  pred <- c(share_above_Fapp = sum(h * pa) / sum(h), share_above_a25_1g = sum(h * pb) / sum(h))
  # conditional mixture quantiles of w*/F_app given w* > F_app + TOL
  den <- sum(h * pa)
  cq <- function(p) uniroot(function(t) sum(h * pmax(Fr(t * d$F_app) - (1 - pa), 0)) / den - p, c(1, 20), tol = 1e-9)$root
  pred <- c(pred, q50_ratio_above = cq(0.5), q90_ratio_above = cq(0.9))
  ab <- d$cls == "above"
  obs <- c(share_above_Fapp = sum(h[ab]) / sum(h), share_above_a25_1g = sum(h[ab & d$wage > d$a25d * (1 + g) + TOL]) / sum(h),
    q50_ratio_above = wq(d$wage[ab] / d$F_app[ab], h[ab], 0.5), q90_ratio_above = wq(d$wage[ab] / d$F_app[ab], h[ab], 0.9))
  list(pred = pred, obs = obs, fit = f25)
}
V3 <- v3_stats(C25, CF)
set.seed(SEED); cnt <- boot_counts(cf_states)
i25 <- split(seq_len(nrow(C25)), C25$state_fips)[cf_states]; i26 <- split(seq_len(nrow(CF)), CF$state_fips)[cf_states]
t0 <- Sys.time()
v3d <- t(vapply(seq_len(NBOOT), function(b) {
  r <- v3_stats(C25[unlist(rep(i25, cnt[, b]), use.names = FALSE), ], CF[unlist(rep(i26, cnt[, b]), use.names = FALSE), ], start = V3$fit$par)
  c(r$pred - r$obs, r$pred)
}, numeric(8)))
message("V3 bootstrap done in ", round(difftime(Sys.time(), t0, units = "mins"), 1), " min")
v3 <- tibble(item = paste0("V3_", names(V3$pred)), predicted = unname(V3$pred), observed = unname(V3$obs),
  estimate = unname(V3$pred - V3$obs), se = apply(v3d[, 1:4], 2, sd),
  ci_low = apply(v3d[, 1:4], 2, function(x) pci(x)[1]), ci_high = apply(v3d[, 1:4], 2, function(x) pci(x)[2]),
  n_orders_fy25 = nrow(C25), n_orders_fy26 = nrow(CF), n_states = length(cf_states)) %>%
  mutate(ci_includes_0 = ci_low <= 0 & ci_high >= 0)
print(v3, width = 200)

# ---- S7: all 48 states -----------------------------------------------------------------------
S7 <- boot_fit_outputs(M26all, CH$fam, CH$scale_k, ST48)
s7 <- tab_out(S7, paste("S7: all 48 states, chosen", CH$model))
print(s7 %>% select(item, estimate, ci_low, ci_high), width = 200)

write_csv(p4, file.path(out, "f1_p4.csv")); write_csv(band, file.path(out, "f1_p4_band.csv"))
write_csv(v2, file.path(out, "f1_v2.csv")); write_csv(v2s, file.path(out, "f1_v2_states.csv"))
write_csv(v3, file.path(out, "f1_v3.csv")); write_csv(s7, file.path(out, "f1_p4_s7.csv"))
st_sf <- p4_state_shortfall(P4$fit, P4$md)
write_csv(st_sf, file.path(out, "f1_p4_state_shortfall.csv"))
saveRDS(list(p4 = p4, band = band, v2 = v2, v2s = v2s, v3 = v3, s7 = s7, fit = P4$fit, chosen = CH$model, st_sf = st_sf,
  v3_fit25 = V3$fit), file.path(out, "f1_p4.rds"))
message("done ", format(Sys.time()))
