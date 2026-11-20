# Which years drive the 2011-12 cohort estimate? Re-estimate on sub-windows.
script_dir <- dirname(normalizePath(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE))))
src <- readLines(file.path(script_dir, "employer_checks.R"))
eval(parse(text = src[seq_len(grep("^p11 <- build", src) - 1)]))
p11 <- build(2011:2012, 2011:2019)
p14 <- build(2014:2015, 2011:2019)
out <- bind_rows(
  est("2011-12 cohort, 2011-15", p11 %>% filter(year <= 2015), "X", FE_1A),
  est("2011-12 cohort, 2013-19", p11 %>% filter(year >= 2013), "X", FE_1A),
  est("2011-12 cohort, 2014-19", p11 %>% filter(year >= 2014), "X", FE_1A),
  est("2014-15 cohort, 2011-19 (pre-baseline years included)", p14, "X", FE_1A),
  est("2014-15 cohort, 2011-15", p14 %>% filter(year <= 2015), "X", FE_1A)
) %>% mutate(t = coef / se)
print(as.data.frame(out %>% select(check, outcome, coef, se, nobs, t)), digits = 3)
# Variance of the regional AEWR deviation by year (identifying variation in 1(a)).
rd <- region_aewr %>% filter(year >= 2011, year <= 2019)
rd$dev <- resid(feols(aewr_real ~ 1 | aewr_region_id + year, rd))
print(rd %>% group_by(year) %>% summarise(sd_dev = round(sd(dev), 2)))
