# Scratch diagnostic: the time-varying bite of the retained DiD,
#   B_{c,t-1} = real AEWR_{r,t-1} - real CZ p25 wage_{c,t-1}  (aewr_cz_p25_l1),
# on the retained DiD sample and controls (table_1_main_results.tex, column 2).
# Asks what identifies the bite coefficient and whether its two components
# enter with the equal and opposite coefficients a bite model requires.
# Reads the retained DiD panel; writes nothing; prints to stdout.
here::i_am("code/paths.R")
source(here::here("code", "paths.R"))
source(path_code("designs", "did", "helpers.R"))
suppressMessages({library(arrow); library(dplyr); library(fixest)})
y <- "h2a_cert_share_farm_workers_2011_start_year"
d <- read_parquet(path_processed("did_county_year_panel.parquet")) %>% did_sample()
need <- c(y, "aewr_cz_p25_l1", "aewr_ppi_l1", "wage_p25_l1", "post", "ln_pop_census",
          "emp_pop_ratio", "county_fips", "year", "cz_id", "aewr_region_id", "state_fips")
stopifnot(all(need %in% names(d)))
d <- d[complete.cases(d[, need]), ]
stopifnot(!anyDuplicated(d[c("county_fips", "year")]))
stopifnot(max(abs(d$aewr_cz_p25_l1 - (d$aewr_ppi_l1 - d$wage_p25_l1))) < 1e-8)
cat("N", nrow(d), "counties", n_distinct(d$county_fips), "years", paste(range(d$year), collapse = "-"),
    "mean outcome", mean(d[[y]]), "\n")

cl <- did_cluster_formula
ctl <- "+ ln_pop_census + emp_pop_ratio"
f <- function(rhs, fe) as.formula(paste(y, "~", rhs, ctl, "|", fe))
show <- function(m, rows) print(coeftable(m)[rows, , drop = FALSE])
lincom <- function(m, k) c(estimate = sum(coef(m)[k]), se = sqrt(sum(vcov(m)[k, k])))

cat("\n== 1. retained column 2 (county + year FE) ==\n")
m1 <- feols(f("aewr_cz_p25_l1 * post", "county_fips + year"), d, vcov = cl, notes = FALSE)
show(m1, c("aewr_cz_p25_l1", "aewr_cz_p25_l1:postTRUE"))
cat("post-2011 slope (main + interaction):\n"); print(lincom(m1, c("aewr_cz_p25_l1", "aewr_cz_p25_l1:postTRUE")))

cat("\n== 2. one slope for all years ==\n")
m2 <- feols(f("aewr_cz_p25_l1", "county_fips + year"), d, vcov = cl, notes = FALSE)
show(m2, "aewr_cz_p25_l1")

cat("\n== 3. sources of within variation in the bite (county + year FE) ==\n")
r <- function(v, fe) resid(feols(as.formula(paste(v, "~ 1 |", fe)), d, notes = FALSE))
ra <- r("aewr_ppi_l1", "county_fips + year"); rw <- r("wage_p25_l1", "county_fips + year")
rg <- r("aewr_cz_p25_l1", "county_fips + year")
print(c(var_bite = var(rg), var_aewr = var(ra), var_wage = var(rw), cov_aewr_wage = cov(ra, rw), cor_aewr_wage = cor(ra, rw)))
cat("sd of bite after county + region-year FE (wage only):",
    sd(r("aewr_cz_p25_l1", "county_fips + aewr_region_id^year")), "\n")
cat("median distinct wage_p25_l1 values per year:", median(tapply(d$wage_p25_l1, d$year, n_distinct)),
    " CZs:", n_distinct(d$cz_id), "\n")

cat("\n== 4. components entered separately (county + year FE) ==\n")
k <- c("aewr_ppi_l1", "wage_p25_l1")
for (v in list(cl, ~aewr_region_id)) {
  m4 <- feols(f("aewr_ppi_l1 + wage_p25_l1", "county_fips + year"), d, vcov = v, notes = FALSE)
  cat("cluster:", deparse(v), "\n"); show(m4, k)
  s <- lincom(m4, k); df <- if (identical(v, cl)) Inf else n_distinct(d$aewr_region_id) - 1
  cat("bite restriction b_aewr + b_wage = 0: estimate", s["estimate"], "se", s["se"],
      "p", 2 * pt(-abs(s["estimate"] / s["se"]), df), "\n")
}
m4p <- feols(f("aewr_ppi_l1 * post + wage_p25_l1 * post", "county_fips + year"), d, vcov = cl, notes = FALSE)
cat("with post interactions:\n")
show(m4p, c("aewr_ppi_l1", "wage_p25_l1", "aewr_ppi_l1:postTRUE", "postTRUE:wage_p25_l1"))

cat("\n== 5. bite with region-year or state-year FE (only the local wage identifies) ==\n")
m5 <- feols(f("aewr_cz_p25_l1", "county_fips + aewr_region_id^year"), d, vcov = cl, notes = FALSE)
cat("region-year FE:\n"); show(m5, "aewr_cz_p25_l1")
m5p <- feols(f("aewr_cz_p25_l1 * post", "county_fips + aewr_region_id^year"), d, vcov = cl, notes = FALSE)
cat("region-year FE with post interaction:\n"); show(m5p, c("aewr_cz_p25_l1", "aewr_cz_p25_l1:postTRUE"))
cat("post-2011 slope:\n"); print(lincom(m5p, c("aewr_cz_p25_l1", "aewr_cz_p25_l1:postTRUE")))
m5s <- feols(f("aewr_cz_p25_l1", "county_fips + state_fips^year"), d, vcov = cl, notes = FALSE)
cat("state-year FE:\n"); show(m5s, "aewr_cz_p25_l1")
