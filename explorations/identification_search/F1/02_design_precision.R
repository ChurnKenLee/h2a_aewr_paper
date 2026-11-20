# Identification search, strategy F1 (Phase 3 design, step 3): design-stage
# precision of the declared F1 parameters. Replaces the elasticity MDE (F1 is
# model-based and is not pooled).
#
# Usage (repository root, after 01_support.R):
#   R_LIBS_USER=$(ls -d $PWD/renv/library/*/R-*/* | head -1) Rscript --vanilla \
#     explorations/identification_search/F1/02_design_precision.R "$S"
# Reads $S/F1/f1_orders.parquet and $S/A1/a1_state_doses.csv.
# Writes $S/F1/f1_design_*.csv.
#
# Integrity (what is and is not read):
#   - theta (P1): FY2025 -> FY2026 incumbent units, their FY2025 hours and the
#     predetermined state floor change are support. Offered-wage CHANGES are
#     read only for the FY2023 -> FY2024 transition (design calibration; that
#     transition is excluded from every declared estimation and falsification
#     sample), and only as residuals about their weighted mean.
#   - P2/P3: hours shares above the floor in FY2025 and FY2026 at the floor
#     itself (already computed as support by 01_support.R); FY2025 upper-tail
#     shares at higher cutoffs (FY2025 is pre-rule data) for the effect size
#     under the fixed-premium alternative. No FY2026 offer is compared with any
#     cutoff other than its own applicable floor.
#   - P4: simulated data only (structure from FY2026 support: states, hours,
#     floors). No model is fitted to observed FY2026 or FY2025 offered wages.
source("explorations/identification_search/F1/00_helpers.R")
args <- commandArgs(trailingOnly = TRUE)
S <- normalizePath(args[1]); out <- f1_out_dir(S)
SEED <- 20261102L; NSIM <- 999L; NBOOT <- 999L
o <- read_parquet(file.path(out, "f1_orders.parquet"))
doses <- read_csv(file.path(S, "A1", "a1_state_doses.csv"), col_types = cols(state_fips = "c", area_id = "c", .default = col_double()),
  progress = FALSE, col_select = c(state_fips, area_id, a24, a25, e25, US_s1, US_s2, FU, eff_cut_u))
crv1 <- function(x, y, w, cl) {
  X <- cbind(1, x); W <- w / mean(w); XtWX <- crossprod(X, X * W); b <- solve(XtWX, crossprod(X, y * W))
  e <- drop(y - X %*% b); G <- length(unique(cl)); n <- length(y)
  meat <- Reduce(`+`, lapply(split(seq_len(n), cl), function(i) { s <- crossprod(X[i, , drop = FALSE], (W * e)[i]); s %*% t(s) }))
  V <- solve(XtWX) %*% meat %*% solve(XtWX) * G / (G - 1) * (n - 1) / (n - 2)
  list(b = b[2], se = sqrt(V[2, 2]), G = G)
}
jk <- function(x, y, w, cl) {
  g <- sort(unique(cl)); bs <- vapply(g, function(k) { i <- cl != k; X <- cbind(1, x[i]); W <- w[i]
    solve(crossprod(X, X * W), crossprod(X, y[i] * W))[2] }, numeric(1))
  sqrt((length(g) - 1) / length(g) * sum((bs - mean(bs))^2))
}

# ---- P1 theta: support frame and calibration residuals ----------------------------
unit_frame <- function(base, nxt, new_only) {
  q <- o %>% filter(!piece, soc_grp == "field", !is.na(employer_id))
  b <- q %>% filter(fy == base) %>% group_by(employer_id, state_fips) %>%
    summarise(h = sum(hours), sh_above = sum(hours[cls == "above"]) / sum(hours), any_unknown = any(cls == "floor_unknown"),
      mw_area = mw_area[which.max(hours)], w_base = sum(wage * hours) / sum(hours), .groups = "drop") %>%
    filter(!any_unknown, sh_above >= 0.9999)
  n <- q %>% filter(fy == nxt, new_rule | !new_only) %>% group_by(employer_id, state_fips) %>%
    summarise(w_next = sum(wage * hours) / sum(hours), .groups = "drop")
  b %>% inner_join(n, by = c("employer_id", "state_fips"))
}
# Calibration: FY2023 -> FY2024 (residuals about the weighted mean only)
cal <- unit_frame(2023, 2024, FALSE) %>% mutate(dw = w_next - w_base)
cal <- cal %>% mutate(res = dw - sum(dw * h) / sum(h))
cal_sum <- cal %>% summarise(units = n(), states = n_distinct(state_fips), resid_sd_w = sqrt(sum(h * res^2) / sum(h)),
  resid_sd_unw = sd(res), resid_iqr = IQR(res), share_var_state = {
    m <- tapply(res * h, state_fips, sum) / tapply(h, state_fips, sum); hs <- tapply(h, state_fips, sum)
    sum(hs * m^2) / sum(h * res^2) })
print(cal_sum)
# Estimation frame (support only: units, FY2025 hours, predetermined floor change)
est <- o %>% filter(fy == 2025, !piece, soc_grp == "field", !is.na(employer_id)) %>% group_by(employer_id, state_fips) %>%
  summarise(h = sum(hours), sh_above = sum(hours[cls == "above"]) / sum(hours), any_unknown = any(cls == "floor_unknown"),
    mw_area = mw_area[which.max(hours)], .groups = "drop") %>% filter(!any_unknown, sh_above >= 0.9999) %>%
  semi_join(o %>% filter(fy == 2026, new_rule, !piece, soc_grp == "field") %>% distinct(employer_id, state_fips), by = c("employer_id", "state_fips")) %>%
  left_join(doses %>% select(state_fips, mw_area = area_id, dF = eff_cut_u), by = c("state_fips", "mw_area")) %>%
  filter(state_fips %in% doses$state_fips)  # A1 dose universe: 48 contiguous states (AK, HI, DC, PR excluded)
stopifnot(!anyNA(est$dF))
dF_sum <- est %>% summarise(units = n(), states = n_distinct(state_fips), hw_mean_dF = sum(h * dF) / sum(h),
  hw_sd_dF = sqrt(sum(h * (dF - sum(h * dF) / sum(h))^2) / sum(h)), min_dF = min(dF), max_dF = max(dF),
  max_state_hours_share = max(tapply(h, state_fips, sum)) / sum(h))
print(dF_sum)
# Simulation: dw = g + theta * dF + e, e = calibration residuals resampled by state blocks
set.seed(SEED)
cst <- split(cal$res, cal$state_fips); cst <- cst[lengths(cst) >= 2]
sts <- split(seq_len(nrow(est)), est$state_fips)
sim_theta <- function(theta) {
  t(vapply(seq_len(NSIM), function(r) {
    e <- numeric(nrow(est))
    for (s in names(sts)) { src <- cst[[sample.int(length(cst), 1)]]; e[sts[[s]]] <- sample(src, length(sts[[s]]), replace = TRUE) }
    y <- 0.5 + theta * est$dF + e
    f <- crv1(est$dF, y, est$h, est$state_fips); sj <- jk(est$dF, y, est$h / mean(est$h), est$state_fips)
    tq <- qt(0.975, f$G - 1)
    c(b = f$b, se = f$se, se_jk = sj, rej0_crv = abs(f$b / f$se) > tq, rej0_jk = abs(f$b / sj) > tq,
      rej1_crv = abs((f$b - 1) / f$se) > tq, rej1_jk = abs((f$b - 1) / sj) > tq)
  }, numeric(7)))
}
wilson_hi <- function(p, n) { z <- 1.959964; (p + z^2 / (2 * n) + z * sqrt(p * (1 - p) / n + z^2 / (4 * n^2))) / (1 + z^2 / n) }
th <- bind_rows(lapply(c(0, 0.5, 1), function(t0) {
  m <- sim_theta(t0)
  tibble(theta_true = t0, sd_theta_hat = sd(m[, "b"]), mean_theta_hat = mean(m[, "b"]), mean_se_crv1 = mean(m[, "se"]), mean_se_jk = mean(m[, "se_jk"]),
    rej_theta0_crv1 = mean(m[, "rej0_crv"]), rej_theta0_jk = mean(m[, "rej0_jk"]),
    rej_theta1_crv1 = mean(m[, "rej1_crv"]), rej_theta1_jk = mean(m[, "rej1_jk"]))
})) %>% mutate(wilson_hi_size_crv1 = ifelse(theta_true == 0, wilson_hi(rej_theta0_crv1, NSIM), ifelse(theta_true == 1, wilson_hi(rej_theta1_crv1, NSIM), NA)),
  wilson_hi_size_jk = ifelse(theta_true == 0, wilson_hi(rej_theta0_jk, NSIM), ifelse(theta_true == 1, wilson_hi(rej_theta1_jk, NSIM), NA)))
print(th, width = 250)
write_csv(bind_rows(cal_sum %>% mutate(item = "calibration FY2023->FY2024 above-floor incumbents"),
  dF_sum %>% mutate(item = "estimation frame FY2025->FY2026 (support)")), file.path(out, "f1_design_theta_frame.csv"))
write_csv(th, file.path(out, "f1_design_theta_sim.csv"))

# ---- P2/P3: shares above cutoffs; state-cluster bootstrap ----------------------------------
boot_share <- function(d, ind) {
  st <- d %>% mutate(x = ind) %>% group_by(state_fips) %>% summarise(hx = sum(hours * x), h = sum(hours), .groups = "drop")
  est <- sum(st$hx) / sum(st$h)
  b <- vapply(seq_len(NBOOT), function(r) { i <- sample.int(nrow(st), nrow(st), replace = TRUE); sum(st$hx[i]) / sum(st$h[i]) }, numeric(1))
  c(share = est, se = sd(b), states = nrow(st))
}
set.seed(SEED)
f25 <- o %>% filter(state_fips %in% doses$state_fips, fy == 2025, !piece, soc_grp == "field", cls %in% c("above", "at_floor", "between_rates", "below_floor")) %>%
  select(-any_of(c("a25"))) %>%
  left_join(doses %>% select(state_fips, mw_area = area_id, a25, FU), by = c("state_fips", "mw_area"))
f26 <- o %>% filter(state_fips %in% doses$state_fips, fy == 2026, new_rule, !piece, soc_grp == "field", cls != "floor_unknown")
# drift g: FY2025-hours-weighted AEWR25 / AEWR24 - 1 (treatment data)
g <- f25 %>% left_join(doses %>% select(state_fips, mw_area = area_id, a24), by = c("state_fips", "mw_area")) %>%
  summarise(g = sum(hours * (a25 / a24 - 1), na.rm = TRUE) / sum(hours[!is.na(a24)])) %>% pull(g)
s25 <- boot_share(f25, f25$cls == "above")
s26 <- boot_share(f26, f26$cls == "above")
# V1 statistic: Delta_S = S26(a25 (1+g)) - S25(a25). Under M1 (floor-independent market
# wage, common drift g, same composition) Delta_S = 0. Effect size under the fixed-premium
# alternative (M2), from FY2025 only: S26(a25 (1+g)) under M2 = share of FY2025 offers whose
# premium over the FY2025 floor exceeds a25 (1+g) - FU, i.e. w25 > a25 + (a25 (1+g) - FU) + TOL.
m2 <- boot_share(f25, f25$cls == "above" & f25$wage > f25$a25 + (f25$a25 * (1 + g) - f25$FU) + TOL)
s25_a25 <- boot_share(f25, f25$cls == "above" & f25$wage > f25$a25 + TOL)
p23 <- tibble(item = c("S25: FY2025 share above every floor (field non-piece)", "S26: FY2026 share above every floor (field non-piece; support)",
  "S25(a25): FY2025 share above a25 (= M1 prediction of S26(a25 (1+g)))", "M2-implied S26(a25(1+g)) from FY2025 premiums"),
  share = c(s25["share"], s26["share"], s25_a25["share"], m2["share"]), se_state_boot = c(s25["se"], s26["se"], s25_a25["se"], m2["se"]),
  states = c(s25["states"], s26["states"], s25_a25["states"], m2["states"])) %>%
  mutate(g = g)
# Design SE of Delta_S: under M1 S26(a25(1+g)) has the sampling distribution of S25(a25); SE = sqrt(2) x SE(S25(a25)).
dS_se <- sqrt(2) * s25_a25["se"]
p23 <- bind_rows(p23, tibble(item = c("V1 design SE of Delta_S", "V1 MDE (2.8 x SE)", "M2-implied Delta_S = M2 S26 - S25(a25)",
    "V1 design SE of Delta_S, conservative (SE of S26 at its own floor)"),
  share = c(NA, NA, m2["share"] - s25_a25["share"], NA), se_state_boot = c(dS_se, 2.8 * dS_se, NA, sqrt(s25_a25["se"]^2 + s26["se"]^2)), states = NA, g = g))
print(p23, width = 200)
write_csv(p23, file.path(out, "f1_design_shares.csv"))
fs <- f26 %>% group_by(state_fips) %>% summarise(k = sum(cls == "above"))
message("FY2026 field non-piece states with >= 10 above-floor orders: ", sum(fs$k >= 10), "; orders above: ", sum(fs$k))

# ---- P4: parametric censored model, simulation only -------------------------------------
# Structure from FY2026 support: field non-piece orders in confirmation states, hours,
# F_app, F_ref (state U.S. skill-I rate), state. Latent r = log(w*/F_ref) ~ family(mu, s);
# mu set so that the hours share above F_app equals the observed support share.
ms <- f26 %>% filter(part %in% "confirmation") %>% transmute(state_fips, hours, lc = log(F_app / F_ref), F_app, F_ref)
W <- ms$hours / mean(ms$hours); share_obs <- sum(ms$hours[f26$cls[f26$part %in% "confirmation"] == "above"]) / sum(ms$hours)
qf <- list(normal = list(p = pnorm, d = dnorm, q = qnorm, r = rnorm), logistic = list(p = plogis, d = dlogis, q = qlogis, r = rlogis))
tobit_fit <- function(y, cens, lc, w, fam = "normal") {
  P <- qf[[fam]]
  nll <- function(par) { mu <- par[1]; s <- exp(par[2])
    -sum(w * ifelse(cens, P$p((lc - mu) / s, log.p = TRUE), P$d((y - mu) / s, log = TRUE) - log(s))) }
  op <- optim(c(-0.2, log(0.15)), nll, method = "BFGS")
  c(mu = op$par[1], s = exp(op$par[2]))
}
b_at <- function(par, a_log, w, fam = "normal") sum(w * qf[[fam]]$p((a_log - par["mu"]) / par["s"])) / sum(w)
calib_mu <- function(s, fam) uniroot(function(mu) sum(W * (1 - qf[[fam]]$p((ms$lc - mu) / s))) / sum(W) - share_obs, c(-3, 2))$root
H2A_ratio <- f1_state_skill_rates() %>% transmute(state_fips, h2a_log = log(H2A_s1 / US_s1))
ms <- ms %>% left_join(H2A_ratio, by = "state_fips")
targets <- list(b_at_Fapp = ms$lc, b_at_Fapp_minus1 = log((ms$F_app - 1) / ms$F_ref), b_at_Fapp_minus2 = log((ms$F_app - 2) / ms$F_ref),
  b_at_H2A_s1 = ms$h2a_log, b_at_Fapp_plus1 = log((ms$F_app + 1) / ms$F_ref))
set.seed(SEED)
st_idx <- split(seq_len(nrow(ms)), ms$state_fips)
p4 <- list()
for (dgp in c("normal", "logistic", "mixture")) for (s in c(0.10, 0.20, 0.30)) {
  fam_gen <- if (dgp == "mixture") "normal" else dgp
  mu <- if (dgp == "mixture") NA else calib_mu(s, dgp)
  gen <- function() {
    if (dgp == "mixture") {  # 2-component: 85% at mu1 with scale s/2, 15% at mu1 + 2s with scale s; mu1 calibrated
      z <- runif(nrow(ms)) < 0.85
      f <- function(m1) sum(W * (0.85 * (1 - pnorm((ms$lc - m1) / (s / 2))) + 0.15 * (1 - pnorm((ms$lc - m1 - 2 * s) / s)))) / sum(W) - share_obs
      m1 <- uniroot(f, c(-3, 2))$root
      ifelse(z, rnorm(nrow(ms), m1, s / 2), rnorm(nrow(ms), m1 + 2 * s, s))
    } else qf[[dgp]]$r(nrow(ms), mu, s)
  }
  truth <- if (dgp == "mixture") {
    f <- function(m1) sum(W * (0.85 * (1 - pnorm((ms$lc - m1) / (s / 2))) + 0.15 * (1 - pnorm((ms$lc - m1 - 2 * s) / s)))) / sum(W) - share_obs
    m1 <- uniroot(f, c(-3, 2))$root
    vapply(targets, function(a) sum(W * (0.85 * pnorm((a - m1) / (s / 2)) + 0.15 * pnorm((a - m1 - 2 * s) / s))) / sum(W), numeric(1))
  } else vapply(targets, function(a) b_at(c(mu = mu, s = s), a, W, dgp), numeric(1))
  for (r in 1:20) {   # 20 simulated samples per cell; SE from a 49-draw state bootstrap of each
    rr <- gen(); cens <- rr <= ms$lc; y <- pmax(rr, ms$lc)
    par <- tobit_fit(y, cens, ms$lc, W, "normal")
    bhat <- vapply(targets, function(a) b_at(par, a, W, "normal"), numeric(1))
    bb <- t(vapply(1:49, function(k) { i <- unlist(st_idx[sample.int(length(st_idx), length(st_idx), replace = TRUE)])
      pp <- tobit_fit(y[i], cens[i], ms$lc[i], W[i], "normal"); vapply(targets, function(a) b_at(pp, a[i], W[i], "normal"), numeric(1)) }, numeric(length(targets))))
    p4[[length(p4) + 1]] <- tibble(dgp = dgp, scale = s, rep = r, target = names(targets), truth = truth, est = bhat, boot_se = apply(bb, 2, sd))
  }
  message("P4 cell done: ", dgp, " ", s)
}
p4 <- bind_rows(p4)
p4s <- p4 %>% group_by(dgp, scale, target) %>% summarise(truth = first(truth), mean_est = mean(est), bias = mean(est - truth),
  sd_est = sd(est), mean_boot_se = mean(boot_se), rmse = sqrt(mean((est - truth)^2)), .groups = "drop")
print(p4s, n = 60, width = 200)
write_csv(p4s, file.path(out, "f1_design_p4_sim.csv"))
message("done ", format(Sys.time()))
