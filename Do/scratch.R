folder_dir    <- paste0("C:/Users/", Sys.info()["user"], "/Dropbox/H-2A Paper/")
folder_do     <- paste0(folder_dir, "Do/")
folder_data   <- paste0(folder_dir, "Data Int/")
folder_output <- paste0(folder_dir, "Output/")

library(arrow)
library(tidyverse)
library(scales)
library(ggthemes)
library(fixest)

county_df <- read_parquet(paste0(folder_data, "county_df_analysis_year.parquet"))

## --- Change 1: decile TS plot (% deviation, no legend) ---

aewr_cz_p25_czreg_ts_data <- county_df %>%
  filter(
    any_cropland_2007 == 1,
    county_simple_treatment_groups != "always takers",
    !is.na(cz_out10), !is.na(aewr_region_num)
  ) %>%
  mutate(cz_aewr_id = paste0(cz_out10, "_", aewr_region_num)) %>%
  group_by(cz_aewr_id, year) %>%
  summarise(aewr_cz_p25_mean = mean(aewr_cz_p25, na.rm = TRUE), .groups = "drop")

czreg_national_avg <- aewr_cz_p25_czreg_ts_data %>%
  group_by(year) %>%
  summarise(national_avg = mean(aewr_cz_p25_mean, na.rm = TRUE))

aewr_cz_p25_czreg_ts_data <- merge(aewr_cz_p25_czreg_ts_data, czreg_national_avg, by = "year")

aewr_cz_p25_czreg_ts_data <- aewr_cz_p25_czreg_ts_data %>%
  mutate(deviation = (aewr_cz_p25_mean - national_avg) / national_avg)

czreg_2022_dev <- aewr_cz_p25_czreg_ts_data %>%
  filter(year == 2022) %>%
  select(cz_aewr_id, deviation) %>%
  rename(dev_2022 = deviation) %>%
  mutate(decile = ntile(dev_2022, 10), above_trend_stable = ifelse(dev_2022 > 0, 1, 0))

aewr_cz_p25_czreg_ts_data <- merge(
  aewr_cz_p25_czreg_ts_data,
  czreg_2022_dev %>% select(cz_aewr_id, decile, above_trend_stable),
  by = "cz_aewr_id"
)

czreg_decile_ts <- aewr_cz_p25_czreg_ts_data %>%
  group_by(decile, year) %>%
  summarise(avg_deviation = mean(deviation, na.rm = TRUE), .groups = "drop")

cat("Decile TS range:", paste(round(range(czreg_decile_ts$avg_deviation), 4), collapse = " to "), "\n")

plot_czreg_decile_ts <- ggplot(
  czreg_decile_ts,
  aes(x = year, y = avg_deviation, group = as.factor(decile), color = decile)
) +
  geom_line(linewidth = 0.8) +
  scale_color_distiller(palette = "RdBu", name = "Decile\n(2022 Endpoint)") +
  scale_y_continuous(labels = scales::percent) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "grey50") +
  geom_vline(xintercept = 2012, linetype = "dashed", color = "black") +
  theme_clean() +
  theme(legend.position = "none") +
  xlab("Year") +
  ylab("AEWR p25 Bite Deviation from National Average (% of national avg)")

ggsave(
  filename = paste0(folder_output, "fig_ts_aewr_cz_p25_czregion_deviations_deciles.png"),
  plot_czreg_decile_ts, width = 8, height = 5, device = "png"
)
cat("Decile TS saved.\n")

## --- Change 6: summary stats table ---

samp_base <- county_df %>%
  filter(any_cropland_2007 == 1, county_simple_treatment_groups != "always takers")

sumstats_vars <- list(
  "H-2A share of 2011 farm employment" = "h2a_cert_share_farm_workers_2011_start_year",
  "H-2A certified workers (start year)" = "nbr_workers_certified_start_year",
  "Farm employment 2011 (baseline)"     = "emp_farm_2011",
  "AEWR p25 bite (2012 $)"             = "aewr_cz_p25",
  "Log population"                      = "ln_pop_census",
  "Employment-to-population ratio"      = "emp_pop_ratio"
)

sumstats_rows <- purrr::imap_dfr(sumstats_vars, function(col, label) {
  x <- samp_base[[col]]
  tibble(
    Variable = label,
    N        = sum(!is.na(x)),
    Mean     = mean(x, na.rm = TRUE),
    SD       = sd(x, na.rm = TRUE),
    Min      = min(x, na.rm = TRUE),
    Max      = max(x, na.rm = TRUE)
  )
})

print(sumstats_rows)

sumstats_tex <- c(
  "\\begin{table}[htbp]",
  "\\centering",
  "\\caption{Summary Statistics: Difference-in-Differences Variables}",
  "\\label{tab:sumstats}",
  "\\begin{tabular}{lrrrrr}",
  "\\hline\\hline",
  "Variable & N & Mean & SD & Min & Max \\\\",
  "\\hline",
  apply(sumstats_rows, 1, function(r) {
    sprintf("%s & %s & %.3f & %.3f & %.3f & %.3f \\\\",
      r["Variable"],
      format(as.integer(r["N"]), big.mark = ","),
      as.numeric(r["Mean"]),
      as.numeric(r["SD"]),
      as.numeric(r["Min"]),
      as.numeric(r["Max"])
    )
  }),
  "\\hline\\hline",
  "\\end{tabular}",
  "\\end{table}"
)

writeLines(sumstats_tex, con = paste0(folder_output, "table_sumstats_dd_variables.tex"))
cat("Summary stats table saved.\n")
