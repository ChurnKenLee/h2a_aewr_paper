# Scratch-only helpers for the alternative-identification exploration.
# Reuses explorations/bite/cross_design_helpers.R (fixest settings, scratch
# guard, CRV1 rows, restricted Webb bootstrap). No retained artifact writes.
source("explorations/bite/cross_design_helpers.R")

# The Track 1 stop point forbids reading any FY2026 outcome, offered wage or
# case count before the user approves. Every case-level read goes through
# these helpers, which filter inside the Arrow scan.
ID_LAST_PRE_FY <- 2025L

id_output_dir <- function() bite_output_dir()

id_read_worksites <- function(cols) {
  cols <- unique(c("fiscal_year", cols))
  d <- arrow::open_dataset(path_int("h2a_allocated_worksites.parquet")) %>%
    select(all_of(cols)) %>%
    filter(fiscal_year <= ID_LAST_PRE_FY) %>%
    collect()
  stopifnot(nrow(d) > 0, max(d$fiscal_year) <= ID_LAST_PRE_FY)
  d
}

id_read_disclosures <- function(cols) {
  cols <- unique(c("fiscal_year", cols))
  d <- arrow::open_dataset(path_int("h2a_disclosure_records.parquet")) %>%
    select(all_of(cols)) %>%
    filter(fiscal_year <= ID_LAST_PRE_FY) %>%
    collect()
  stopifnot(nrow(d) > 0, max(d$fiscal_year) <= ID_LAST_PRE_FY)
  d
}

id_assert_pre <- function(years) {
  if (any(years > ID_LAST_PRE_FY, na.rm = TRUE)) stop("Post-policy year present before approval")
  invisible(TRUE)
}

# Elasticity at estimation-sample means: beta * mean(price) / mean(outcome).
id_elasticity <- function(beta, mean_price, mean_y) beta * mean_price / mean_y

# CRV1 rows for several cluster variables, plus the two common units.
id_rows <- function(m, d, terms, meta, clusters, price) {
  d <- d[obs(m), , drop = FALSE]
  yname <- all.vars(m$fml_all$linear)[1]
  w <- if (!is.null(weights(m))) weights(m) else rep(1, nrow(d))
  mean_y <- sum(w * d[[yname]]) / sum(w)
  mean_p <- sum(w * d[[price]]) / sum(w)
  bind_rows(lapply(clusters, function(cl) {
    if (!cl %in% names(d) || anyNA(d[[cl]])) return(NULL)
    sm <- summary(m, vcov = as.formula(paste0("~", cl)))
    ct <- coeftable(sm); tt <- intersect(terms, rownames(ct))
    bind_cols(meta[rep(1, length(tt)), , drop = FALSE], tibble(
      term = tt, cluster = cl, clusters = n_distinct(d[[cl]]),
      estimate = ct[tt, 1], se = ct[tt, 2], p = ct[tt, ncol(ct)],
      n = nrow(d), counties = if ("county_fips" %in% names(d)) n_distinct(d$county_fips) else NA_integer_,
      first_year = min(d$year), last_year = max(d$year),
      mean_y = mean_y, mean_price = mean_p, price = price,
      mde = 2.8 * ct[tt, 2]))
  })) %>% mutate(
    mde_pct_mean = 100 * mde / abs(mean_y),
    elasticity = id_elasticity(estimate, mean_price, mean_y),
    elasticity_low = id_elasticity(estimate - qt(.975, clusters - 1) * se, mean_price, mean_y),
    elasticity_high = id_elasticity(estimate + qt(.975, clusters - 1) * se, mean_price, mean_y),
    mde_elasticity = abs(id_elasticity(mde, mean_price, mean_y)))
}

# Restricted Webb bootstrap from cross_design_helpers.R with any cluster
# variable: bite_wild reads the cluster from d$aewr_region_id only.
# Unweighted OLS/IV only; weighted and PPML fits are not supported.
id_wild <- function(m, d, term, cluster, B = 9999L, seed = 20261102L) {
  if (!is.null(weights(m))) stop("id_wild supports unweighted fits only")
  d2 <- d
  d2$aewr_region_id <- as.character(d[[cluster]])
  bite_wild(m, d2, term, B = B, seed = seed) %>% mutate(wild_cluster = cluster)
}

# Share of residual outcome variance (after county and year FE) that lies at
# the region-year level: R^2 of the two-way residual on region-year cells.
id_region_year_share <- function(d, y, unit = "county_fips", cell = "aewr_region_id",
  weights = NULL) {
  f <- as.formula(paste(y, "~ 1 |", unit, "+ year"))
  m <- feols(f, d, weights = weights)
  r <- resid(m); dd <- d[obs(m), , drop = FALSE]
  cellid <- paste(dd[[cell]], dd$year)
  w <- if (is.null(weights)) rep(1, length(r)) else dd[[all.vars(weights)[1]]]
  cm <- ave(r * w, cellid, FUN = sum) / ave(w, cellid, FUN = sum)
  sum(w * cm^2) / sum(w * r^2)
}

id_write <- function(x, out_dir, name) {
  readr::write_csv(x, file.path(out_dir, name))
  invisible(x)
}

id_md <- function(x, digits = 4) {
  x <- as.data.frame(x)
  f <- x
  for (j in seq_along(f)) if (is.numeric(f[[j]])) f[[j]] <- ifelse(is.na(f[[j]]), "",
    formatC(f[[j]], digits = digits, format = "fg", flag = "#"))
  c(paste("|", paste(names(f), collapse = " | "), "|"),
    paste("|", paste(rep("---", ncol(f)), collapse = " | "), "|"),
    apply(f, 1, function(r) paste("|", paste(trimws(r), collapse = " | "), "|")))
}
