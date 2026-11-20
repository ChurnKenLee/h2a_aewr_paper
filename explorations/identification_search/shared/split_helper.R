# Identification search, Phase 0, step 4: confirmation-split helper.
#
# Integrity rule 2 of markdowns/identification_search_agent_plan.md: when a
# strategy needs data-driven choices, units are assigned with seed 20261102
# to an exploration third and a confirmation two-thirds. The assignment is a
# pure function of the sorted unique unit ids, the seed and the optional
# strata, so every agent reproduces it exactly.
#
# Use from R:
#   source("explorations/identification_search/shared/split_helper.R")
#   s <- ish_split(units = unique(panel$state_fips))                    # unstratified
#   s <- ish_split(units = st$state_fips, strata = st$aewr_region_id)    # within strata
# CLI (writes the standard state and county splits to the scratch dir):
#   Rscript --vanilla explorations/identification_search/shared/split_helper.R "$S/shared"
ISH_SPLIT_SEED <- 20261102L
ISH_SPLIT_MIN_UNITS <- 30L

# units: character ids (geographic ids must already be normalized strings).
# strata: optional vector aligned with units; each stratum is split separately
#   (one third rounded to nearest, at least one exploration unit when a
#   stratum has 3+ units).
# Returns a data.frame(unit, stratum, part = "exploration"/"confirmation").
ish_split <- function(units, strata = NULL, seed = ISH_SPLIT_SEED) {
  if (!is.character(units)) stop("Unit ids must be character strings")
  if (anyNA(units) || any(!nzchar(units))) stop("Unit ids must be nonempty")
  d <- data.frame(unit = units, stratum = if (is.null(strata)) "all" else as.character(strata), stringsAsFactors = FALSE)
  d <- unique(d)
  if (anyDuplicated(d$unit)) stop("Each unit must belong to one stratum")
  d <- d[order(d$stratum, d$unit), , drop = FALSE]
  old <- if (exists(".Random.seed", envir = globalenv())) get(".Random.seed", envir = globalenv()) else NULL
  on.exit(if (!is.null(old)) assign(".Random.seed", old, envir = globalenv()))
  set.seed(seed, kind = "Mersenne-Twister", normal.kind = "Inversion", sample.kind = "Rejection")
  d$part <- "confirmation"
  for (s in sort(unique(d$stratum))) {
    i <- which(d$stratum == s)
    k <- round(length(i) / 3)
    if (length(i) >= 3L) k <- max(k, 1L)
    if (k > 0) d$part[i[sample.int(length(i), k)]] <- "exploration"
  }
  if (length(unique(d$unit)) < ISH_SPLIT_MIN_UNITS)
    warning("Fewer than ", ISH_SPLIT_MIN_UNITS, " units: the plan says skip the split and flag the strategy")
  rownames(d) <- NULL
  d
}

if (sys.nframe() == 0L) {
  here::i_am("code/paths.R")
  source(here::here("code", "paths.R"))
  source(path_code("c00_shared", "geography.R"))
  suppressPackageStartupMessages({library(arrow); library(dplyr)})
  out_dir <- commandArgs(trailingOnly = TRUE)[1]
  if (is.na(out_dir) || !nzchar(out_dir)) stop("Supply a scratch output directory outside the repository")
  dir.create(out_dir, recursive = TRUE, showWarnings = FALSE); out_dir <- normalizePath(out_dir)
  if (startsWith(paste0(out_dir, "/"), paste0(normalizePath(path_root()), "/"))) stop("Outputs must be outside the repository")
  cp <- read_parquet(path_processed("county_year_panel.parquet"), col_select = c("county_fips", "state_fips", "aewr_region_id", "year")) %>%
    filter(year == 2011) %>% distinct(county_fips, state_fips, aewr_region_id)
  stopifnot(identical(cp$county_fips, harmonize_county_fips_2010(cp$county_fips)))
  st <- cp %>% distinct(state_fips, aewr_region_id)
  splits <- list(
    state_unstratified = ish_split(sort(unique(st$state_fips))),
    state_by_former_region = ish_split(st$state_fips, st$aewr_region_id),
    county_unstratified = ish_split(sort(unique(cp$county_fips))),
    county_by_state = ish_split(cp$county_fips, cp$state_fips))
  for (nm in names(splits)) write.csv(splits[[nm]], file.path(out_dir, paste0("split_", nm, ".csv")), row.names = FALSE)
  print(lapply(splits, function(s) table(s$part)))
}
