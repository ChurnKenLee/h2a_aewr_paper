# Purpose: Describe and estimate DD effects for the county Fisher price index.
# Input: data/processed/county_df_analysis_year.parquet and county shapefile.
# Outputs: price-index figures and table_fisher_price_dd.tex.
# Run after: code/c02_build/04_finalize_county_panel.R.

source(if (file.exists(file.path("code", "bootstrap_paths.R"))) {
  file.path("code", "bootstrap_paths.R")
} else {
  file.path("..", "bootstrap_paths.R")
})
source(path_code("c00_shared", "fips.R"))
source(path_code("c00_shared", "analysis_helpers.R"))
library(arrow)
library(tidyverse)
library(fixest)
library(sf)
library(ggspatial)
library(scales)
library(ggthemes)

county_df <- read_parquet(path_processed("county_df_analysis_year.parquet"))
samp_base <- analysis_sample(county_df)
county_map <- read_county_map(
  path_raw("county_shapefile", "tl_2020_us_county.zip")
)

#### Exhibit 17: Fisher Price Index Time Series -----------------------------------

fisher_ts <- samp_base %>%
  filter(!is.na(fisher_index_ppi)) %>%
  group_by(year) %>%
  summarise(
    mean_price = mean(fisher_index_ppi, na.rm = TRUE),
    sd_price = sd(fisher_index_ppi, na.rm = TRUE)
  )

ts_fisher_price <- ggplot(fisher_ts, aes(x = year, y = mean_price)) +
  geom_ribbon(
    aes(ymin = mean_price - sd_price, ymax = mean_price + sd_price),
    alpha = 0.2,
    fill = "steelblue"
  ) +
  geom_line(linewidth = 1.25, color = "#2166ac") +
  geom_vline(xintercept = 2011, linetype = "dashed") +
  scale_x_continuous(breaks = seq(2008, 2022, by = 2)) +
  labs(
    x = "Year",
    y = "Fisher Price Index (real, 2012=100)",
    title = "Agricultural Price Index Across Counties, 2008–2022",
    subtitle = "Mean ± 1 SD across sample counties"
  ) +
  theme_clean()

ts_fisher_price

ggsave(
  filename = path_figures("ts_fisher_price_index_ppi.png"),
  ts_fisher_price,
  device = "png"
)

#### Exhibit 18: DD Regressions — Fisher Price Index as Outcome ------------------

price_dd_1 <- feols(
  fisher_index_ppi ~
    aewr_cz_p25_l1 * postdummy | county_fe + year_fe,
  data = samp_base,
  vcov = ~cz_aewr_region_fe
)

price_dd_2 <- feols(
  fisher_index_ppi ~
    aewr_cz_p25_l1 *
    postdummy +
    ln_pop_census +
    emp_pop_ratio |
    county_fe + year_fe,
  data = samp_base,
  vcov = ~cz_aewr_region_fe
)

price_dd_3 <- feols(
  fisher_index_ppi ~
    aewr_cz_p25_l1 * postdummy | county_fe + year_fe,
  data = samp_no_border,
  vcov = ~cz_aewr_region_fe
)

price_dd_4 <- feols(
  fisher_index_ppi ~
    aewr_cz_p25_l1 *
    postdummy +
    ln_pop_census +
    emp_pop_ratio |
    county_fe + year_fe,
  data = samp_no_border,
  vcov = ~cz_aewr_region_fe
)

table_fisher_price <- etable(
  price_dd_1,
  price_dd_2,
  price_dd_3,
  price_dd_4,
  tex = TRUE,
  title = "The Effect of the AEWR Wage Premium on Agricultural Prices",
  headers = c(
    "No Controls",
    "Controls",
    "No Border, No Controls",
    "No Border, Controls"
  ),
  dict = c(
    "fisher_index_ppi" = "Fisher price index (real, 2012=100)",
    "aewr_cz_p25_l1" = "Lagged AEWR vs 25th pct wage gap",
    "postdummy" = "Post",
    "ln_pop_census" = "Log population",
    "emp_pop_ratio" = "Employment-to-pop ratio"
  ),
  signif.code = c("***" = 0.01, "**" = 0.05, "*" = 0.10),
  file = path_tables("table_fisher_price_dd.tex"),
  replace = TRUE
)

#### Exhibit 19: County Map — Fisher Price Index Ratio 2022 / 2008 ---------------

fisher_map_data <- samp_base %>%
  filter(year %in% c(2011, 2022), !is.na(fisher_index_ppi)) %>%
  select(countyfips, year, fisher_index_ppi) %>%
  distinct(countyfips, year, .keep_all = TRUE) %>%
  pivot_wider(
    id_cols = countyfips,
    names_from = year,
    values_from = fisher_index_ppi,
    names_prefix = "price_"
  ) %>%
  mutate(price_ratio_2022_2011 = price_2022 / price_2011) %>%
  filter(!is.na(price_ratio_2022_2011))

ratio_median <- median(fisher_map_data$price_ratio_2022_2011, na.rm = TRUE)

fisher_county_map <- merge(
  x = county_map,
  y = fisher_map_data,
  by = "countyfips",
  all.x = TRUE,
  all.y = FALSE
)

map_fisher_price_ratio <- ggplot(fisher_county_map) +
  geom_sf(
    aes(fill = price_ratio_2022_2011),
    color = alpha("grey", 0.3),
    linewidth = 0.1
  ) +
  theme(
    panel.grid.major = element_line(
      color = gray(0.5),
      linetype = "dashed",
      linewidth = 0.5
    ),
    panel.background = element_rect(fill = "aliceblue")
  ) +
  theme_bw() +
  annotation_north_arrow(
    location = "bl",
    which_north = "true",
    pad_x = unit(0.05, "in"),
    pad_y = unit(0.25, "in"),
    style = north_arrow_fancy_orienteering
  ) +
  annotation_scale(location = "bl", width_hint = 0.4) +
  scale_fill_gradient2(
    low = muted("#2166ac"),
    mid = "white",
    midpoint = ratio_median,
    high = muted("#b2182b"),
    name = "Price Index\nRatio\n2022 / 2011",
    na.value = "grey90"
  )

map_fisher_price_ratio

ggsave(
  filename = path_figures("map_fisher_price_ratio_2022_2011.png"),
  map_fisher_price_ratio,
  device = "png"
)
