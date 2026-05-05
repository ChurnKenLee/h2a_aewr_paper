folder_dir  <- paste0("C:/Users/", Sys.info()["user"], "/Dropbox/H-2A Paper/")
folder_data <- paste0(folder_dir, "Data Int/")

suppressPackageStartupMessages({ library(arrow); library(dplyr) })

county_df <- read_parquet(paste0(folder_data, "county_df_analysis_year.parquet"))

stacked_sample <- county_df %>%
  filter(any_cropland_2007 == 1, county_simple_treatment_groups != "always takers") %>%
  arrange(countyfips, year) %>%
  mutate(aewr_cz_p25_pd1 = (aewr_cz_p25 - aewr_cz_p25_l1) / aewr_cz_p25_l1)

pd1_clean <- stacked_sample %>% filter(!is.na(aewr_cz_p25_pd1), is.finite(aewr_cz_p25_pd1))

cat("=== Distribution of aewr_cz_p25_pd1 ===\n")
cat(sprintf("N (finite obs): %d\n", nrow(pd1_clean)))
cat(sprintf("N counties:     %d\n", n_distinct(pd1_clean$countyfips)))
print(quantile(pd1_clean$aewr_cz_p25_pd1, c(0.05, 0.10, 0.15, 0.20, 0.25,
                                              0.75, 0.80, 0.85, 0.90, 0.95)))

# Function: classify counties under a given pair of thresholds
classify <- function(q_low, q_high, label) {
  lo <- quantile(pd1_clean$aewr_cz_p25_pd1, q_low)
  hi <- quantile(pd1_clean$aewr_cz_p25_pd1, q_high)

  events <- stacked_sample %>%
    mutate(
      large_increase = !is.na(aewr_cz_p25_pd1) & is.finite(aewr_cz_p25_pd1) & aewr_cz_p25_pd1 >= hi,
      large_decrease = !is.na(aewr_cz_p25_pd1) & is.finite(aewr_cz_p25_pd1) & aewr_cz_p25_pd1 <= lo
    )

  ce <- events %>%
    group_by(countyfips) %>%
    summarise(
      fiy = if (any(large_increase, na.rm = TRUE)) min(year[large_increase], na.rm = TRUE) else NA_integer_,
      fdy = if (any(large_decrease, na.rm = TRUE)) min(year[large_decrease], na.rm = TRUE) else NA_integer_,
      .groups = "drop"
    ) %>%
    mutate(
      tied = !is.na(fiy) & !is.na(fdy) & fiy == fdy,
      t_inc = case_when(tied ~ 0L, !is.na(fiy) & (is.na(fdy) | fiy < fdy) ~ 1L, TRUE ~ 0L),
      t_dec = case_when(tied ~ 0L, !is.na(fdy) & (is.na(fiy) | fdy < fiy) ~ 1L, TRUE ~ 0L),
      never = (t_inc == 0L & t_dec == 0L & !tied)
    )

  n_inc   <- sum(ce$t_inc)
  n_dec   <- sum(ce$t_dec)
  n_never <- sum(ce$never)
  n_tied  <- sum(ce$tied)

  # Not-yet-treated at 2008 (increase design)
  nyt_inc_2008 <- sum(ce$t_inc == 1L & ce$fiy > 2008)
  nyt_dec_2008 <- sum(ce$t_dec == 1L & ce$fdy > 2008)

  cat(sprintf(
    "%-20s  lo=%+.3f hi=%+.3f  | inc=%4d  dec=%4d  never=%4d  tied=%2d | NTT_inc_2008=%3d  NTT_dec_2008=%3d\n",
    label, lo, hi, n_inc, n_dec, n_never, n_tied, nyt_inc_2008, nyt_dec_2008
  ))
}

cat("\n=== Treatment group sizes by threshold ===\n")
cat(sprintf("%-20s  %-18s  | %-5s  %-5s  %-7s  %-5s | %-16s  %-16s\n",
            "Threshold", "lo / hi values", "inc", "dec", "never", "tied",
            "NTT inc @ 2008", "NTT dec @ 2008"))
cat(strrep("-", 105), "\n")

classify(0.25, 0.75, "q25/q75 (current)")
classify(0.20, 0.80, "q20/q80")
classify(0.15, 0.85, "q15/q85")
classify(0.10, 0.90, "q10/q90")
classify(0.05, 0.95, "q5/q95")
