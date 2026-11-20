here::i_am("code/paths.R"); source(here::here("code", "paths.R"))
suppressMessages({library(arrow); library(dplyr); library(fixest)})
out_dir <- commandArgs(trailingOnly = TRUE)[1]
p <- read_parquet(file.path(out_dir, "employer_bite_panel.parquet")) %>%
  filter(year >= 2011, year <= 2019, is.finite(aewr_real)) %>% mutate(X = bound * aewr_real)
f <- function(y, d) feols(as.formula(paste(y, "~ X | employer_id_balanced + region^year + year[bound]")), d, vcov = ~state)
mixed <- p %>% distinct(employer_id_balanced, region, bound) %>% group_by(region) %>%
  summarise(unb = sum(bound == 0), b = sum(bound == 1)) %>% filter(unb >= 20, b >= 20)
cat("regions with >=20 bound and >=20 unbound employers:", paste(mixed$region, collapse = ","), "\n")
for (y in c("any", "lworkers")) {
  full <- f(y, p)
  cat(sprintf("\n%s full: %.4f (%.4f)\n", y, coef(full)[1], se(full)[1]))
  for (r in mixed$region) {
    m <- f(y, p %>% filter(region != r))
    cat(sprintf("  drop region %-3s: %.4f (%.4f)\n", r, coef(m)[1], se(m)[1]))
  }
  m <- f(y, p %>% filter(region %in% mixed$region))
  cat(sprintf("  mixed regions only: %.4f (%.4f)\n", coef(m)[1], se(m)[1]))
  # Restricted wild cluster bootstrap by region (Webb), on FWL residuals.
  d <- p[obs(full), ]
  ry <- resid(feols(as.formula(paste(y, "~ 1 | employer_id_balanced + region^year + year[bound]")), d))
  rx <- resid(feols(X ~ 1 | employer_id_balanced + region^year + year[bound], d))
  b0 <- sum(rx * ry) / sum(rx^2)
  cl <- d$region; G <- unique(cl); idx <- match(cl, G)
  tstat <- function(yv) { b <- sum(rx * yv) / sum(rx^2); e <- yv - b * rx
    s <- tapply(rx * e, idx, sum); c(b, b / sqrt(sum(s^2)) * sum(rx^2) / sum(rx^2)) }
  t0 <- tstat(ry)[1] / sqrt(sum(tapply(rx * (ry - b0 * rx), idx, sum)^2)) * sum(rx^2)
  webb <- c(-sqrt(1.5), -1, -sqrt(.5), sqrt(.5), 1, sqrt(1.5))
  set.seed(1)
  tb <- replicate(9999, { v <- sample(webb, length(G), TRUE)[idx]; yb <- ry * v  # null beta = 0: residual is ry
    bb <- sum(rx * yb) / sum(rx^2); eb <- yb - bb * rx
    bb / sqrt(sum(tapply(rx * eb, idx, sum)^2)) * sum(rx^2) })
  cat(sprintf("  wild bootstrap (17 regions, null imposed) p = %.3f\n", mean(abs(tb) >= abs(t0))))
}
