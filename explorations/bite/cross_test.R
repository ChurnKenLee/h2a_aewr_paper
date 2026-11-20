# Direct test: donor-subregion wages (source year t-1) vs target-subregion
# outcomes net of the AEWR (year t), donor vs non-donor same-region pairs.
script_dir <- dirname(normalizePath(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE))))
source_lines <- readLines(file.path(script_dir, "common_shocks.R"))
stop_at <- grep("^for \\(nm in names\\(series\\)\\)", source_lines)[1] - 1
eval(parse(text = source_lines[seq_len(stop_at)]))
set.seed(20261102)
regions <- unique(pairs$aewr_region_id)
w <- series$oews_log_wage %>% transmute(aewr_region_id, other = sub, year = year + 1L, vo = v)
for (nm in c("cert_share_net_aewr", "applications_net_aewr")) {
  x <- pairs %>%
    inner_join(series[[nm]] %>% rename(target = sub, vt = v), by = c("aewr_region_id", "target"),
      relationship = "many-to-many") %>%
    inner_join(w, by = c("aewr_region_id", "other", "year"))
  sx <- split(x, x$aewr_region_id)
  draws <- replicate(999, {
    b <- bind_rows(sx[sample(regions, replace = TRUE)])
    c(cor(b$vt[b$is_donor], b$vo[b$is_donor]), cor(b$vt[!b$is_donor], b$vo[!b$is_donor]))
  })
  q <- function(v) sprintf("[%.2f, %.2f]", quantile(v, .025), quantile(v, .975))
  cat(sprintf("%-22s donor wage(t-1): %.3f %s | non-donor: %.3f %s | n=%d\n", nm,
    cor(x$vt[x$is_donor], x$vo[x$is_donor]), q(draws[1, ]),
    cor(x$vt[!x$is_donor], x$vo[!x$is_donor]), q(draws[2, ]), sum(x$is_donor)))
}
