# Identification search, strategy F1 (Phases 4-5): estimation helpers.
# Implements the declared samples, inference and models of
# explorations/identification_search/F1/registry.json (frozen; SHA-256 checked
# by f1_check_registry()). Sources 00_helpers.R unchanged.
#
# Sourced by 04_*.R onward. Nothing here reads outcomes by itself.
source("explorations/identification_search/F1/00_helpers.R")
suppressPackageStartupMessages(library(jsonlite))

SEED <- 20261102L
NBOOT <- 999L
REGISTRY_SHA <- "93cc085fff5f2302e53612e1353aa6380aaccb4e954325d5267957c558a4c2ec"
SCRIPT_SHA <- c(`00_helpers.R` = "e46e58d14ea23bffdc17b60781777b11807d60a138d0852210290ebdee46da4d",
  `01_support.R` = "765fddac54b54e92db4fcf64feebbfb7fff2d00c2a87e7c5816090de8f501505",
  `02_design_precision.R` = "456c84cd729d16d061064e3fbd34b48496a467178e6dae84ade35c022bf81f1b")

f1_check_registry <- function() {
  sha <- function(f) strsplit(system2("sha256sum", f, stdout = TRUE), " ")[[1]][1]
  if (sha(file.path(F1_DIR, "registry.json")) != REGISTRY_SHA) stop("F1 registry hash mismatch")
  for (nm in names(SCRIPT_SHA)) if (sha(file.path(F1_DIR, nm)) != SCRIPT_SHA[[nm]]) stop("F1 design script hash mismatch: ", nm)
  invisible(TRUE)
}

f1_doses <- function(S) {
  read_csv(file.path(S, "A1", "a1_state_doses.csv"), col_types = cols(state_fips = "c", area_id = "c", .default = col_double()),
    progress = FALSE, col_select = c(state_fips, area_id, a24, a25, e25, H2A_s1, US_s1, US_s2, mw_20260101, FU, eff_cut_u)) %>%
    rename(mw_area = area_id)
}

# Attach predetermined doses (state x MW area) to an order table: a24, a25d (dose-file a25), H2A_s1, ...
f1_attach_doses <- function(d, doses) {
  d %>% left_join(doses %>% select(state_fips, mw_area, a24, a25d = a25, e25, H2A_s1d = H2A_s1, US_s2d = US_s2, mw26 = mw_20260101,
    FU, eff_cut_u), by = c("state_fips", "mw_area"))
}

# ---- Declared samples -------------------------------------------------------------------
# M26 classes: above / at_floor / at_old_aewr (excluded classes counted elsewhere).
f1_M26 <- function(o, states, soc = "field", piece_ok = FALSE) {
  o %>% filter(fy == 2026, new_rule, state_fips %in% states, soc_grp %in% soc, piece_ok | !piece, cls %in% c("above", "at_floor", "at_old_aewr"))
}
f1_M25 <- function(o, states, soc = "field", piece_ok = FALSE) {
  o %>% filter(fy == 2025, state_fips %in% states, soc_grp %in% soc, piece_ok | !piece, cls %in% c("above", "at_floor"))
}

# Old-AEWR value matched by an at_old_aewr offer (for S2a censoring)
f1_old_value <- function(d) {
  old <- f1_old_aewr() %>% select(state_fips, old_rates)
  d %>% left_join(old, by = "state_fips") %>%
    mutate(c_old = mapply(function(w, r) if (length(r) == 0) NA_real_ else r[which.min(abs(w - r))], wage, old_rates)) %>% select(-old_rates)
}

# S3: 'above' offers at a state x SOC-group x FY value (to the cent) offered by >= 3
# distinct employers among non-piece 'above' offers of that FY are censored there.
f1_apply_S3 <- function(o) {
  k <- o %>% filter(cls == "above", !piece) %>% mutate(wr = round(wage, 2)) %>%
    group_by(fy, state_fips, soc_grp, wr) %>% summarise(n_emp = n_distinct(employer_id), .groups = "drop")
  o %>% mutate(wr = round(wage, 2)) %>% left_join(k, by = c("fy", "state_fips", "soc_grp", "wr")) %>%
    mutate(s3 = cls == "above" & coalesce(n_emp, 0L) >= 3L,
      c_floor = ifelse(s3, wage, c_floor), cls = ifelse(s3, "at_floor", cls)) %>% select(-wr, -n_emp)
}

# ---- Inference ---------------------------------------------------------------------------
wls <- function(x, y, w) { X <- cbind(1, x); solve(crossprod(X, X * w), crossprod(X, y * w))[, 1] }
crv1 <- function(x, y, w, cl) {
  X <- cbind(1, x); W <- w / mean(w); XtWX <- crossprod(X, X * W); b <- solve(XtWX, crossprod(X, y * W))
  e <- drop(y - X %*% b); G <- length(unique(cl)); n <- length(y)
  meat <- Reduce(`+`, lapply(split(seq_len(n), cl), function(i) { s <- crossprod(X[i, , drop = FALSE], (W * e)[i]); s %*% t(s) }))
  V <- solve(XtWX) %*% meat %*% solve(XtWX) * G / (G - 1) * (n - 1) / (n - 2)
  list(b = b[2, 1], se = sqrt(V[2, 2]), G = G)
}
# CV3J: exact leave-one-state-out jackknife, (G-1)/G scaling, centred at the mean of leave-out estimates
cv3j <- function(x, y, w, cl) {
  g <- sort(unique(cl))
  bs <- vapply(g, function(k) { i <- cl != k; wls(x[i], y[i], w[i])[2] }, numeric(1))
  sqrt((length(g) - 1) / length(g) * sum((bs - mean(bs))^2))
}
# P1-type regression: estimate with primary jackknife and CRV1 alternate; H0 theta = 0 and theta = 1
p1_fit <- function(x, y, w, cl) {
  b <- wls(x, y, w); G <- length(unique(cl)); tq <- qt(0.975, G - 1)
  sj <- cv3j(x, y, w, cl); cr <- crv1(x, y, w, cl)
  tibble(estimate = b[2], intercept = b[1], se = sj, ci_low = b[2] - tq * sj, ci_high = b[2] + tq * sj,
    p_theta0 = 2 * pt(-abs(b[2] / sj), G - 1), p_theta1 = 2 * pt(-abs((b[2] - 1) / sj), G - 1),
    se_crv1 = cr$se, ci_low_crv1 = b[2] - tq * cr$se, ci_high_crv1 = b[2] + tq * cr$se,
    p_theta0_crv1 = 2 * pt(-abs(b[2] / cr$se), G - 1), p_theta1_crv1 = 2 * pt(-abs((b[2] - 1) / cr$se), G - 1),
    n_units = length(y), n_states = G)
}

# State-cluster bootstrap index draws: a states x NBOOT matrix of multiplicities (seed fixed by caller)
boot_counts <- function(states, B = NBOOT) {
  n <- length(states)
  m <- vapply(seq_len(B), function(r) tabulate(sample.int(n, n, replace = TRUE), n), integer(n))
  rownames(m) <- states; m
}
# Ratio-of-sums statistics by state: num, den are states x K matrices; returns est and B x K draws
boot_ratio <- function(num, den, cnt) {
  est <- colSums(num) / colSums(den)
  dr <- t(crossprod(cnt, num) / crossprod(cnt, den))   # K x B -> transpose below
  list(est = est, draws = t(dr))
}
pci <- function(x) quantile(x, c(0.025, 0.975), na.rm = TRUE, names = FALSE)

# ---- P4: left-censored location-scale model ----------------------------------------------
FAM <- list(normal = list(p = pnorm, d = dnorm, q = qnorm), logistic = list(p = plogis, d = dlogis, q = qlogis))
P4_CANDS <- tibble(model = c("normal_common", "normal_skill", "logistic_common", "logistic_skill"),
  fam = c("normal", "normal", "logistic", "logistic"), scale_k = c(FALSE, TRUE, FALSE, TRUE))

# Model data: y (exact r or censoring point), cens, X, w (normalized to mean 1 over the fit sample), skill2
# at_old_aewr handling: "exact" (primary), "censored" (S2a: censored at the old AEWR), "excluded" (S2b)
p4_data <- function(d, center = NULL, nonfield = FALSE, old = "exact", Fref_col = "F_ref") {
  if (old == "excluded") d <- d %>% filter(cls != "at_old_aewr")
  if (old == "censored") d <- d %>% mutate(c_floor = ifelse(cls == "at_old_aewr", c_old, c_floor), cls = ifelse(cls == "at_old_aewr", "at_floor", cls))
  Fr <- d[[Fref_col]]
  lF <- log(Fr)
  if (is.null(center)) center <- sum(d$hours * lF) / sum(d$hours)
  X <- cbind(b0 = 1, b1_skillII = as.numeric(d$skill == "II"), b2_flc = as.numeric(d$flc), b3_logFref = lF - center)
  if (nonfield) X <- cbind(X, b4_nonfield = as.numeric(d$soc_grp == "nonfield"))
  cens <- d$cls == "at_floor"
  y <- ifelse(cens, log(d$c_floor / Fr), log(d$wage / Fr))
  stopifnot(all(is.finite(y)), all(is.finite(X)))
  list(d = d, y = y, cens = cens, X = X, w = d$hours / mean(d$hours), skill2 = d$skill == "II", center = center, state = d$state_fips)
}
p4_scale <- function(par, k, md) { if (k == 0) rep(exp(par["ls"]), length(md$y)) else ifelse(md$skill2, exp(par["ls2"]), exp(par["ls"])) }
p4_ll_i <- function(par, md, fam, scale_k) {
  P <- FAM[[fam]]; mu <- drop(md$X %*% par[colnames(md$X)]); s <- p4_scale(par, scale_k, md)
  z <- (md$y - mu) / s
  ifelse(md$cens, P$p(z, log.p = TRUE), P$d(z, log = TRUE) - log(s))
}
p4_fit <- function(md, fam, scale_k, start = NULL) {
  nm <- c(colnames(md$X), "ls", if (scale_k) "ls2")
  if (is.null(start)) {
    b <- solve(crossprod(md$X, md$X * md$w), crossprod(md$X, md$y * md$w))[, 1]
    s0 <- log(sqrt(sum(md$w * (md$y - md$X %*% b)^2) / sum(md$w)) + 0.01)
    start <- c(b, ls = s0, if (scale_k) c(ls2 = s0))
  }
  start <- start[nm]
  nll <- function(p) { names(p) <- nm; v <- -sum(md$w * p4_ll_i(p, md, fam, scale_k)); if (!is.finite(v)) 1e12 else v }
  op <- optim(start, nll, method = "BFGS", control = list(maxit = 1000, reltol = 1e-12))
  if (op$convergence != 0) op <- optim(op$par, nll, method = "Nelder-Mead", control = list(maxit = 20000, reltol = 1e-12))
  p <- op$par; names(p) <- nm
  list(par = p, nll = op$value, conv = op$convergence, fam = fam, scale_k = scale_k, center = md$center)
}
# Predictive objects for a prediction set pd (p4_data built with the fit's center)
p4_mu_s <- function(fit, pd) list(mu = drop(pd$X %*% fit$par[colnames(pd$X)]), s = p4_scale(fit$par, fit$scale_k, pd))
p4_cdf <- function(fit, pd, a) {   # P(w* <= a_i), a in dollars (vector aligned with pd)
  ms <- p4_mu_s(fit, pd); FAM[[fit$fam]]$p((log(a / pd$d$F_ref) - ms$mu) / ms$s)
}
U_GRID <- (seq_len(4000) - 0.5) / 4000
p4_outputs <- function(fit, pd) {
  d <- pd$d; h <- d$hours; ms <- p4_mu_s(fit, pd); P <- FAM[[fit$fam]]
  bw <- function(a) sum(h * P$p((log(a / d$F_ref) - ms$mu) / ms$s)) / sum(h)
  out <- c(b_Fapp_minus2 = bw(d$F_app - 2), b_Fapp_minus1 = bw(d$F_app - 1), b_H2A_s1 = bw(d$H2A_s1d), b_Fapp = bw(d$F_app), b_a25 = bw(d$a25d))
  # median of w*/F_app (hours-weighted mixture)
  fm <- function(m) sum(h * P$p((log(m * d$F_app / d$F_ref) - ms$mu) / ms$s)) / sum(h) - 0.5
  out["median_wstar_over_Fapp"] <- uniroot(fm, c(0.01, 50), tol = 1e-10)$root
  # mean shortfall E[(a25 - w*)+] over unique (mu, s, F_ref, a25) combos, quantile grid
  key <- paste(signif(ms$mu, 12), signif(ms$s, 12), d$F_ref, d$a25d)
  u <- !duplicated(key); idx <- match(key, key[u])
  q <- P$q(U_GRID)
  sf <- vapply(which(u), function(i) mean(pmax(d$a25d[i] - d$F_ref[i] * exp(ms$mu[i] + ms$s[i] * q), 0)), numeric(1))
  sf <- sf[idx]
  out["shortfall_a25_dollars"] <- sum(h * sf) / sum(h)
  out["shortfall_a25_relative"] <- sum(h * sf / d$a25d) / sum(h)
  pars <- fit$par; pars[c("ls", if (fit$scale_k) "ls2")] <- exp(pars[c("ls", if (fit$scale_k) "ls2")])
  names(pars) <- sub("^ls2$", "s_skillII", sub("^ls$", if (fit$scale_k) "s_skillI" else "s", names(pars)))
  c(out, pars)
}
# State-level relative shortfall (G2)
p4_state_shortfall <- function(fit, pd) {
  d <- pd$d; ms <- p4_mu_s(fit, pd); q <- FAM[[fit$fam]]$q(U_GRID)
  key <- paste(signif(ms$mu, 12), signif(ms$s, 12), d$F_ref, d$a25d); u <- !duplicated(key); idx <- match(key, key[u])
  sf <- vapply(which(u), function(i) mean(pmax(d$a25d[i] - d$F_ref[i] * exp(ms$mu[i] + ms$s[i] * q), 0)), numeric(1))[idx]
  tibble(state_fips = d$state_fips, h = d$hours, rel = sf / d$a25d) %>% group_by(state_fips) %>%
    summarise(rel_shortfall = sum(h * rel) / sum(h), hours = sum(h), .groups = "drop")
}

# Registry tie rule: highest LOSO log-likelihood wins; ties within 1 unit go to the simpler model.
# Simplicity order (implementation reading of "normal before logistic, common before skill-specific"):
# fewer parameters first, then normal before logistic.
P4_SIMPLICITY <- c("normal_common", "logistic_common", "normal_skill", "logistic_skill")

# ---- P2 / P3 share machinery -------------------------------------------------------------
# State sums of hours x indicator columns, aligned to `states` (zeros where absent)
st_sums <- function(d, ind, states) {
  ind <- as.matrix(ind); storage.mode(ind) <- "double"
  num <- matrix(0, length(states), ncol(ind), dimnames = list(states, colnames(ind)))
  den <- num
  if (nrow(d) > 0) {
    r <- rowsum(d$hours * ind, d$state_fips); num[rownames(r), ] <- r
    h <- rowsum(d$hours, d$state_fips); den[rownames(h), ] <- h[, 1]
  }
  list(num = num, den = den)
}
# Share estimates and bootstrap draws (B x K) under multiplicities cnt (states x B)
share_boot <- function(d, ind, states, cnt) {
  s <- st_sums(d, ind, states)
  list(est = colSums(s$num) / colSums(s$den), draws = crossprod(cnt, s$num) / crossprod(cnt, s$den))
}
# Counterfactual floor grid of P2 (a_i >= F_app,i)
p2_grid <- function(d, g) {
  cbind(F_app = d$F_app, `F_app+0.50` = d$F_app + 0.5, `F_app+1.00` = d$F_app + 1, `F_app+2.00` = d$F_app + 2,
    `midpoint(F_app,a25)` = pmax((d$F_app + d$a25d) / 2, d$F_app), a25 = pmax(d$a25d, d$F_app), `a25(1+g)` = pmax(d$a25d * (1 + g), d$F_app))
}
# b26(a) on a sample: rows per grid point with bootstrap CI (percentile)
p2_rows <- function(d, g, states, cnt, scope, tol = TOL) {
  A <- p2_grid(d, g); ind <- (d$wage <= A + tol) * 1
  sb <- share_boot(d, ind, states, cnt)
  tibble(item = "P2_b26", grid = colnames(A), scope = scope, estimate = sb$est, se = apply(sb$draws, 2, sd),
    ci_low = apply(sb$draws, 2, function(x) pci(x)[1]), ci_high = apply(sb$draws, 2, function(x) pci(x)[2]),
    n_orders = nrow(d), n_states = n_distinct(d$state_fips), n_above = sum(d$cls == "above"))
}
# P3 Delta_S(m) = S26(m a25 (1+g)) - S25(m a25), joint state bootstrap (class 'above' and offer > cutoff + tol)
p3_rows <- function(d26, d25, g, states, cnt, scope, ms = c(1, 1.05, 1.10, 1.20), tol = TOL) {
  i26 <- sapply(ms, function(m) (d26$cls == "above" & d26$wage > m * d26$a25d * (1 + g) + tol) * 1)
  i25 <- sapply(ms, function(m) (d25$cls == "above" & d25$wage > m * d25$a25d + tol) * 1)
  colnames(i26) <- colnames(i25) <- paste0("m=", ms)
  s26 <- share_boot(d26, i26, states, cnt); s25 <- share_boot(d25, i25, states, cnt)
  dd <- s26$draws - s25$draws
  tibble(item = "P3_Delta_S", grid = colnames(i26), scope = scope, g = g, S26 = s26$est, S25 = s25$est, estimate = s26$est - s25$est,
    se = apply(dd, 2, sd), ci_low = apply(dd, 2, function(x) pci(x)[1]), ci_high = apply(dd, 2, function(x) pci(x)[2]),
    S26_ci_low = apply(s26$draws, 2, function(x) pci(x)[1]), S26_ci_high = apply(s26$draws, 2, function(x) pci(x)[2]),
    S25_ci_low = apply(s25$draws, 2, function(x) pci(x)[1]), S25_ci_high = apply(s25$draws, 2, function(x) pci(x)[2]),
    n_orders = nrow(d26) + nrow(d25), n_states = n_distinct(c(d26$state_fips, d25$state_fips)))
}
f1_g_frozen <- function(out) { g <- read_csv(file.path(out, "f1_design_shares.csv"), show_col_types = FALSE)$g[1]; stopifnot(abs(g - 0.0376) < 5e-4); g }
