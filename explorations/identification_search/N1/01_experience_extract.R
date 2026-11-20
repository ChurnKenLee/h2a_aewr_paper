# Identification search, strategy N1 (Phase 3 design, step 1a): extract the
# job-order experience requirement from the raw OFLC H-2A disclosure files.
#
# Usage (repository root):
#   Rscript --vanilla explorations/identification_search/N1/01_experience_extract.R "$S"
# Reads data/raw/h2a/H-2A_Disclosure_Data_FY20{20..26}*.xlsx (raw inputs; not
# modified) and $S/shared/outcomes_case_rows.parquet (join coverage only).
# Writes $S/N1/n1_experience_extract.parquet (one row per disclosure case x FY)
# and $S/N1/n1_experience_coverage.csv.
#
# The field is WORK_EXPERIENCE_MONTHS in the FY2020-FY2026 layouts
# (EMP_EXP_NUM_MONTHS in FY2015-FY2019; not used: the employer unit breaks at
# FY2020). It is not carried by data/intermediate/h2a_disclosure_records.parquet,
# so this is an exploration-local extract, not a producer change.
#
# Integrity: FY2026 rows are extracted (the field is needed for declared
# Phase 5 mechanism and first-stage work) but this script reports only their
# coding coverage (non-missing and parse rates by case). It does not tabulate
# the FY2026 distribution of requirements, any FY2026 requirement by FY2025
# unit, any FY2026 offered wage, or any outcome.
suppressPackageStartupMessages({library(readxl); library(dplyr); library(arrow); library(readr)})
args <- commandArgs(trailingOnly = TRUE)
S <- normalizePath(args[1]); out <- file.path(S, "N1"); dir.create(out, showWarnings = FALSE)
if (startsWith(paste0(S, "/"), paste0(normalizePath("."), "/"))) stop("Outputs must be outside the repository")
files <- c(`2020` = "H-2A_Disclosure_Data_FY2020.xlsx", `2021` = "H-2A_Disclosure_Data_FY2021.xlsx",
  `2022` = "H-2A_Disclosure_Data_FY2022_Q4.xlsx", `2023` = "H-2A_Disclosure_Data_FY2023_Q4.xlsx",
  `2024` = "H-2A_Disclosure_Data_FY2024_Q4.xlsx", `2025` = "H-2A_Disclosure_Data_FY2025_Q4.xlsx",
  `2026` = "H-2A_Disclosure_Data_FY2026_Q3.xlsx")
keep <- c("CASE_NUMBER", "CASE_STATUS", "WORK_EXPERIENCE_MONTHS", "TRAINING_MONTHS", "EDUCATION_LEVEL",
  "SUPERVISE_OTHER_EMP", "H2A_LABOR_CONTRACTOR", "NAICS_CODE", "SOC_CODE", "PIECE_RATE_OFFER", "WORKSITE_STATE")
num <- function(x) suppressWarnings(as.numeric(trimws(x)))
rows <- list()
for (fy in names(files)) {
  f <- file.path("data", "raw", "h2a", files[[fy]])
  h <- names(read_excel(f, n_max = 0))
  miss <- setdiff(keep, h); if (length(miss)) message(fy, " missing columns: ", paste(miss, collapse = ", "))
  d <- read_excel(f, col_types = "text")[, intersect(keep, h)]
  for (m in miss) d[[m]] <- NA_character_
  rows[[fy]] <- tibble(fiscal_year = as.integer(fy), case_number = trimws(d$CASE_NUMBER), case_status_raw = d$CASE_STATUS,
    exp_raw = d$WORK_EXPERIENCE_MONTHS, exp_months = num(d$WORK_EXPERIENCE_MONTHS), training_months = num(d$TRAINING_MONTHS),
    education_level = d$EDUCATION_LEVEL, supervise = toupper(substr(trimws(d$SUPERVISE_OTHER_EMP), 1, 1)),
    flc = toupper(substr(trimws(d$H2A_LABOR_CONTRACTOR), 1, 1)), naics = gsub("[^0-9]", "", d$NAICS_CODE),
    soc_raw = d$SOC_CODE, piece_rate = num(d$PIECE_RATE_OFFER), worksite_state = d$WORKSITE_STATE)
  message(fy, ": ", nrow(d), " rows")
}
x <- bind_rows(rows)
# One row per case x FY (disclosure main rows are one per case; check).
dup <- x %>% count(fiscal_year, case_number) %>% filter(n > 1)
message("duplicate case x FY rows: ", nrow(dup))
x <- x %>% group_by(fiscal_year, case_number) %>% slice(1) %>% ungroup() %>%
  mutate(exp_valid = !is.na(exp_months) & exp_months >= 0 & exp_months <= 120 & exp_months == round(exp_months),
    exp_months = ifelse(exp_valid, exp_months, NA_real_),
    tier2 = ifelse(is.na(exp_months), NA, exp_months >= 3),
    exp_class = case_when(is.na(exp_months) ~ NA_character_, exp_months == 0 ~ "0", exp_months <= 2 ~ "1-2",
      exp_months == 3 ~ "3", TRUE ~ "4+"))
stopifnot(all(nzchar(x$case_number)))
write_parquet(x, file.path(out, "n1_experience_extract.parquet"))

# ---- Coverage --------------------------------------------------------------------
cov_case <- x %>% group_by(fiscal_year) %>% summarise(cases = n(), exp_nonmissing_raw = mean(!is.na(exp_raw) & trimws(exp_raw) != ""),
  exp_valid_share = mean(exp_valid), nonvalid_nonmissing = sum(!exp_valid & !is.na(exp_raw) & trimws(exp_raw) != ""), .groups = "drop")
# Join coverage to the shared case rows (certified hours; window = all rows of the disclosure FY). FY2026: case counts only.
cr <- read_parquet(file.path(S, "shared", "outcomes_case_rows.parquet"),
  col_select = c("case_number", "fiscal_year", "cert_hours", "in_rq13d", "fy_receipt")) %>% filter(fiscal_year >= 2020)
cr_case <- cr %>% group_by(case_number, fiscal_year) %>% summarise(cert_hours = sum(cert_hours, na.rm = TRUE),
  rq13d_hours = sum(cert_hours[in_rq13d], na.rm = TRUE), .groups = "drop")
j <- cr_case %>% left_join(x %>% select(case_number, fiscal_year, exp_valid, exp_class), by = c("case_number", "fiscal_year"))
cov_rows <- j %>% group_by(fiscal_year) %>% summarise(case_row_cases = n(), case_rows_matched = mean(!is.na(exp_valid)),
  case_rows_with_valid_exp = mean(exp_valid %in% TRUE),
  cert_hours_share_valid_exp = ifelse(first(fiscal_year) < 2026, sum(cert_hours[exp_valid %in% TRUE]) / sum(cert_hours), NA_real_),
  .groups = "drop")
dist <- j %>% filter(fiscal_year <= 2025, exp_valid %in% TRUE) %>% group_by(fiscal_year, exp_class) %>%
  summarise(h = sum(cert_hours), .groups = "drop") %>% group_by(fiscal_year) %>% mutate(h = h / sum(h)) %>%
  tidyr::pivot_wider(names_from = exp_class, values_from = h, names_prefix = "cert_hours_share_exp_")
cov <- cov_case %>% left_join(cov_rows, by = "fiscal_year") %>% left_join(dist, by = "fiscal_year") %>%
  mutate(note = ifelse(fiscal_year == 2026, "FY2026 Q3: coverage only; requirement distribution deliberately not tabulated (post-period)", ""))
write_csv(cov, file.path(out, "n1_experience_coverage.csv"))
print(as.data.frame(cov))
