script_dir <- dirname(normalizePath(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE))))
src <- readLines(file.path(script_dir, "employer_checks.R"))
eval(parse(text = src[seq_len(grep("^p11 <- build", src) - 1)]))
a <- build(2011:2012, 2012) %>% distinct(employer_id_balanced, b11 = bound, type, region)
b <- build(2014:2015, 2014) %>% distinct(employer_id_balanced, b14 = bound)
x <- inner_join(a, b, by = "employer_id_balanced")
cat("employers in both cohorts:", nrow(x), "of", nrow(a), "\n")
print(table(bound_2011_12 = x$b11, bound_2014_15 = x$b14))
print(x %>% filter(b11 == 0) %>% count(type, b14))
# 2011-12 unbound that stay in program vs exit: participation by 2014
cat("2011-12 unbound present in 2014-15 cohort:", round(mean(a$employer_id_balanced[a$b11 == 0] %in% b$employer_id_balanced), 3),
  " bound:", round(mean(a$employer_id_balanced[a$b11 == 1] %in% b$employer_id_balanced), 3), "\n")
