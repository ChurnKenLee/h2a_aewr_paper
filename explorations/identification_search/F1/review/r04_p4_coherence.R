# F1 blind review (Phase 6), POST-OUTCOME check r04: is the P4 latent distribution
# coherent as an alternative-wage distribution?
#  (a) Refit the chosen normal_common model on the 32 confirmation states (point fit only,
#      same code path as 07_p4_confirmation.R) and compute the hours-weighted latent mass
#      below the order's legal minimum (max(area MW over the contract, $7.25)) and below
#      $7.25; the implied hours-weighted mean and median latent wage in dollars; and the
#      hours-weighted ACS 2024 state agricultural mean hourly wage for comparison.
#  (b) Assumption-free bounds for b(a) below the floor: b(a) in [0, b26(F_app)] only.
#  (c) A misspecification probe: normal model fitted with the uncensored part only
#      (truncated-normal likelihood of 'above' offers, a common truncation point F_app):
#      what does the upper tail alone imply for b(F_app)?
# Usage: R_LIBS_USER=... Rscript --vanilla explorations/identification_search/F1/review/r04_p4_coherence.R "$S"
source("explorations/identification_search/F1/03_est_helpers.R")
args <- commandArgs(trailingOnly = TRUE)
S <- normalizePath(args[1]); out <- file.path(S, "F1", "review"); dir.create(out, showWarnings = FALSE)
o <- read_parquet(file.path(S, "F1", "f1_orders.parquet"))
doses <- f1_doses(S); ST48 <- sort(unique(doses$state_fips))
o <- f1_attach_doses(o, doses)
CF <- f1_M26(o, ST48) %>% filter(part == "confirmation")
md <- p4_data(CF); fit <- p4_fit(md, "normal", FALSE); est <- p4_outputs(fit, md)
print(round(est, 4))
ms <- p4_mu_s(fit, md); d <- md$d; h <- d$hours
legal <- pmax(coalesce(d$mw_max, 7.25), 7.25)
pb <- function(a) sum(h * pnorm((log(a / d$F_ref) - ms$mu) / ms$s)) / sum(h)
mean_w <- sum(h * d$F_ref * exp(ms$mu + ms$s^2 / 2)) / sum(h)
med_w <- { f <- function(x) pb(rep(x, length(h))) - 0.5; uniroot(f, c(1, 60))$root }
acs <- read_parquet(path_int("acs_state_ag_wage.parquet")) %>% filter(year == 2024) %>% transmute(state_fips, acs = acs_ag_mean_hourly_wage)
acs_w <- d %>% left_join(acs, by = "state_fips") %>% summarise(v = sum(hours * acs, na.rm = TRUE) / sum(hours[!is.na(acs)])) %>% pull(v)
a <- tibble(stat = c("latent mass below the legal minimum max(MW, 7.25)", "latent mass below $7.25", "latent mass below the H-2A skill-I rate (= P4 b_H2A_s1)",
    "implied hours-weighted mean latent wage ($)", "implied hours-weighted median latent wage ($)", "hours-weighted F_app ($)", "hours-weighted a25 ($)",
    "hours-weighted ACS 2024 state ag mean hourly wage ($)", "censored hours share"),
  value = c(pb(legal), pb(rep(7.25, length(h))), pb(d$H2A_s1d), mean_w, med_w, sum(h * d$F_app) / sum(h), sum(h * d$a25d) / sum(h), acs_w, sum(h[md$cens]) / sum(h)))
print(a); write_csv(a, file.path(out, "r04a_p4_coherence.csv"))

# (c) Upper-tail-only fit: exact 'above' (and at_old_aewr) observations, left-truncated at log(F_app/F_ref) + small tol;
# normal with the same mean covariates and common scale. Implied untruncated P(w* <= F_app) vs observed censored share.
ex <- !md$cens; tr <- log((d$F_app + TOL) / d$F_ref)
nll <- function(p) { mu <- drop(md$X %*% p[1:4]); s <- exp(p[5])
  -sum((md$w * (dnorm((md$y - mu) / s, log = TRUE) - log(s) - pnorm((mu - tr) / s, log.p = TRUE)))[ex]) }
st <- c(fit$par[1:4], fit$par["ls"])
op <- optim(st, nll, method = "BFGS", control = list(maxit = 2000, reltol = 1e-12))
mu_t <- drop(md$X %*% op$par[1:4]); s_t <- exp(op$par[5])
c_tail <- tibble(stat = c("tail-only fit: scale", "tail-only fit: implied P(w* <= F_app + 0.05)", "observed hours share at or below F_app + 0.05 (censored + at/below)",
    "censored-model (chosen) scale", "tail-only fit: implied b(H-2A skill-I rate)", "convergence"),
  value = c(s_t, sum(h * pnorm((tr - mu_t) / s_t)) / sum(h), sum(h[md$cens | d$wage <= d$F_app + TOL]) / sum(h), exp(fit$par["ls"]),
    sum(h * pnorm((log(d$H2A_s1d / d$F_ref) - mu_t) / s_t)) / sum(h), op$convergence))
print(c_tail); write_csv(c_tail, file.path(out, "r04c_tail_only_fit.csv"))
