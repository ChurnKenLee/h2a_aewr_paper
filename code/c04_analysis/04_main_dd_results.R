# Purpose: Estimate the main DD specifications and robustness checks.
# Input: data/processed/county_df_analysis_year.parquet.
# Output: outputs/tables/table_1_main_results.tex.
# Run after: code/c02_build/04_finalize_county_panel.R.

source(if (file.exists(file.path("code", "bootstrap_paths.R"))) {
  file.path("code", "bootstrap_paths.R")
} else {
  file.path("..", "bootstrap_paths.R")
})
source(path_code("c00_shared", "analysis_helpers.R"))
library(arrow)
library(tidyverse)
library(fixest)
library(ggfixest)

county_df <- read_parquet(path_processed("county_df_analysis_year.parquet"))
samp_base <- analysis_sample(county_df)
samp_no_border <- samp_base %>% filter(border_cz == 0)

## Exhibit 13: DD Main Results -------------------------------------------------

cat("samp_base rows:", nrow(samp_base), "\n")
stopifnot(nrow(samp_base) > 1000)

# DD model 1: no controls, all CZs
dd_1 <- feols(
  h2a_cert_share_farm_workers_2011_start_year ~
    aewr_cz_p25_l1 * postdummy | county_fe + year_fe,
  data = samp_base,
  vcov = ~cz_aewr_region_fe,
  demeaned = TRUE
)

dd_1$X_demeaned
xbar = dd_1$X_demeaned
ybar = dd_1$y_demeaned
ybar

samp_base_dd1 <- samp_base %>%
  filter(
    !is.na(h2a_cert_share_farm_workers_2011_start_year),
    !is.na(aewr_cz_p25_l1),
    !is.na(postdummy)
  )

new_table <- bind_cols(samp_base_dd1, xbar) %>%
  mutate(ybar = ybar) %>%
  group_by(year) %>%
  mutate(
    treatment_decile = cut_number(
      aewr_cz_p25_l1...205,
      4,
      labels = c("1", "2", "3", "4")
    )
  ) %>%
  ungroup()

ggplot(new_table, aes(y = ybar, color = treatment_decile)) +
  stat_ecdf(geom = "step", linewidth = 1) +
  coord_cartesian(xlim = c(0.10, 0.90), ylim = c(-0.1, 0.1)) +
  facet_wrap(~year) +
  labs(
    title = "Inverse ECDF / Quantile Plot",
    x = "Quantile (Cumulative Probability)",
    y = "ybar"
  ) +
  theme_minimal()


# DD model 2: with controls, all CZs
dd_2 <- feols(
  h2a_cert_share_farm_workers_2011_start_year ~
    aewr_cz_p25_l1 *
    postdummy +
    ln_pop_census +
    emp_pop_ratio |
    county_fe + year_fe,
  data = samp_base,
  vcov = ~cz_aewr_region_fe
)

# DD model 3: no controls, no border CZs
dd_3 <- feols(
  h2a_cert_share_farm_workers_2011_start_year ~
    aewr_cz_p25_l1 * postdummy | county_fe + year_fe,
  data = samp_no_border,
  vcov = ~cz_aewr_region_fe
)

# DD model 4: with controls, no border CZs
dd_4 <- feols(
  h2a_cert_share_farm_workers_2011_start_year ~
    aewr_cz_p25_l1 *
    postdummy +
    ln_pop_census +
    emp_pop_ratio |
    county_fe + year_fe,
  data = samp_no_border,
  vcov = ~cz_aewr_region_fe
)

# DD model 5: robustness check by excluding potentially influential CZs within each AEWR region
# How many total ag workers are there within each AEWR region?
samp_no_large_cz <- samp_base %>%
  group_by(aewr_region_num, year) %>%
  mutate(aewr_region_year_total_emp_farm = sum(emp_farm, na.rm = TRUE)) %>%
  ungroup() %>%
  group_by(cz_fe, year) %>%
  mutate(cz_year_total_emp_farm = sum(emp_farm, na.rm = TRUE)) %>%
  ungroup()

cz_share_of_aewr_region_farm_emp <- samp_no_large_cz %>%
  distinct(aewr_region_num, cz_fe, year, .keep_all = TRUE) %>%
  select(
    aewr_region_num,
    cz_fe,
    year,
    aewr_region_year_total_emp_farm,
    cz_year_total_emp_farm
  ) %>%
  mutate(
    cz_share_farm_emp = cz_year_total_emp_farm / aewr_region_year_total_emp_farm
  )

plot_cz_share_of_aewr_region_farm_emp <- cz_share_of_aewr_region_farm_emp %>%
  ggplot(aes(x = cz_share_farm_emp)) +
  geom_density(fill = "#69b3a2", color = "#e9ecef", alpha = 0.8) +
  xlab("CZ share of total farm employment within AEWR region")

plot_cz_share_of_aewr_region_farm_emp %>%
  ggsave(
    filename = path_figures("cz_share_of_farm_emp_in_aewr_region.png"),
    width = 8,
    height = 5,
    device = "png"
  )

# Pick arbitrary cutoff of 0.1
cz_to_keep_share_below_10 <- cz_share_of_aewr_region_farm_emp %>%
  filter(cz_share_farm_emp < 0.1) %>%
  select(aewr_region_num, cz_fe, year)

samp_cz_share_below_10 <- samp_no_large_cz %>%
  inner_join(cz_to_keep_share_below_10)

dd_5 <- feols(
  h2a_cert_share_farm_workers_2011_start_year ~
    aewr_cz_p25_l1 * postdummy | county_fe + year_fe,
  data = samp_cz_share_below_10,
  vcov = ~cz_aewr_region_fe,
  demeaned = TRUE
)

# What if we just dropped the largest CZ within each AEWR region?
cz_to_keep_not_largest <- cz_share_of_aewr_region_farm_emp %>%
  group_by(aewr_region_num, year) %>%
  filter(cz_share_farm_emp != max(cz_share_farm_emp, na.rm = TRUE))

samp_cz_no_largest <- samp_no_large_cz %>%
  inner_join(cz_to_keep_not_largest)

dd_6 <- feols(
  h2a_cert_share_farm_workers_2011_start_year ~
    aewr_cz_p25_l1 * postdummy | county_fe + year_fe,
  data = samp_cz_no_largest,
  vcov = ~cz_aewr_region_fe,
  demeaned = TRUE
)

# Tables
table_1 <- etable(
  dd_1,
  dd_2,
  dd_3,
  dd_4,
  tex = TRUE,
  title = "The Effect of the AEWR Wage Premium on H-2A Utilization",
  headers = c(
    "No Controls",
    "Controls",
    "No Border, No Controls",
    "No Border, Controls"
  ),
  dict = c(
    "h2a_cert_share_farm_workers_2011_start_year" = "Normalized H-2A program usage",
    "aewr_cz_p25_l1" = "Lagged AEWR vs 25th pct wage gap",
    "postdummy" = "Post",
    "ln_pop_census" = "Log population",
    "emp_pop_ratio" = "Employment-to-pop ratio"
  ),
  signif.code = c("***" = 0.01, "**" = 0.05, "*" = 0.10),
  file = path_tables("table_1_main_results.tex"),
  replace = TRUE
)

summary(dd_1)
summary(dd_2)
summary(dd_3)
summary(dd_4)
summary(dd_5)
summary(dd_6)
