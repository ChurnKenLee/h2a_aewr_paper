# Purpose: Produce AEWR distributions, time series, maps, and regional diagnostics.
# Inputs: processed county panel, processed AEWR panel, and the county shapefile.
# Outputs: AEWR descriptive figures and analysis_aewr_region_trends.parquet.
# Run after: code/c01_clean/08_aewr_panel.R and code/c02_build/04_finalize_county_panel.R.

source(if (file.exists(file.path("code", "bootstrap_paths.R"))) {
  file.path("code", "bootstrap_paths.R")
} else {
  file.path("..", "bootstrap_paths.R")
})
source(path_code("c00_shared", "fips.R"))
source(path_code("c00_shared", "analysis_helpers.R"))
library(arrow)
library(tidyverse)
library(sf)
library(ggspatial)
library(scales)
library(ggthemes)

county_df <- read_parquet(path_processed("county_df_analysis_year.parquet"))
aewr_data_full <- read_parquet(path_processed("aewr_data_full.parquet"))
county_map <- read_county_map(
  path_raw("county_shapefile", "tl_2020_us_county.zip")
)

save_aewr_region_ts <- function(aewr_data, y_var, y_label, filename_prefix) {
  for (region in sort(unique(aewr_data$aewr_region_num))) {
    plot <- ggplot(
      subset(aewr_data, aewr_region_num == region),
      aes(x = year, y = .data[[y_var]])
    ) +
      geom_line() +
      labs(title = paste0("AEWR Region Number: ", region)) +
      xlab(y_label)
    ggsave(
      path_figures("aewr_ts", paste0(filename_prefix, region, ".png")),
      plot,
      device = "png"
    )
  }
}

bite_long <- county_df %>%
  filter(any_cropland_2007 == 1) %>%
  select(countyfips, year, aewr_cz_p10, aewr_cz_p25, aewr_cz_p50) %>%
  pivot_longer(
    cols = c(aewr_cz_p10, aewr_cz_p25, aewr_cz_p50),
    names_to = "percentile",
    values_to = "bite"
  ) %>%
  mutate(
    percentile = recode(
      percentile,
      aewr_cz_p10 = "AEWR minus p10",
      aewr_cz_p25 = "AEWR minus p25",
      aewr_cz_p50 = "AEWR minus p50"
    )
  )

ggplot(bite_long, aes(x = bite)) +
  geom_histogram(bins = 50, fill = "steelblue", color = "white") +
  geom_vline(xintercept = 0, linetype = "dashed", color = "red") +
  facet_wrap(~percentile, ncol = 1, scales = "free_y") +
  labs(
    x = "Real AEWR minus Real CZ Wage Percentile (2012 $)",
    y = "Count",
    title = "Distribution of AEWR Bite Variables"
  ) +
  theme_clean()

ggsave(
  path_figures("hist_aewr_cz_bite_variables.png"),
  width = 7,
  height = 8
)

#### Exhibit 1: AEWR TS Real ---------------------------------------------------

aewr_data_full_ts <- aewr_data_full %>%
  group_by(year) %>%
  summarise(aewr = mean(aewr, na.rm = T), aewr_ppi = mean(aewr_ppi, na.rm = T))

aewr_ts <- ggplot(data = aewr_data_full_ts, aes(x = year, y = aewr_ppi)) +
  geom_line(linewidth = 1.25, color = "#2166ac") +
  theme_clean() +
  xlab("Year") +
  ylab("Real AEWR") +
  geom_vline(xintercept = 2011, linetype = "dashed")
aewr_ts
ggsave(
  filename = path_figures("ts_national_aewr_real.png"),
  aewr_ts,
  device = "png"
)


#### Exhibit 2: AEWR TS Nominal --------------------------------------------------

aewr_ts_nom <- ggplot(data = aewr_data_full_ts, aes(x = year, y = aewr)) +
  geom_line(linewidth = 1.25, color = "#2166ac") +
  theme_clean() +
  xlab("Year") +
  ylab("Nominal AEWR") +
  geom_vline(xintercept = 2011, linetype = "dashed")
aewr_ts_nom
ggsave(
  filename = path_figures("ts_national_aewr_nominal.png"),
  aewr_ts_nom,
  device = "png"
)

#### Exhibit 1C: Real AEWR vs p25 CZ Wage Bite TS ## ---------------------------

aewr_bite_ts_data <- county_df %>%
  filter(
    any_cropland_2007 == 1,
    county_simple_treatment_groups != "always takers"
  ) %>%
  group_by(year) %>%
  summarise(aewr_cz_p25 = mean(aewr_cz_p25, na.rm = TRUE))

aewr_bite_ts <- ggplot(
  data = aewr_bite_ts_data,
  aes(x = year, y = aewr_cz_p25)
) +
  geom_line(linewidth = 1.25, color = "#2166ac") +
  theme_clean() +
  xlab("Year") +
  ylab("Real AEWR vs p25 Wage Gap (2012 $)") +
  geom_vline(xintercept = 2011, linetype = "dashed")
aewr_bite_ts

ggsave(
  filename = path_figures("ts_national_aewr_cz_p25_bite.png"),
  aewr_bite_ts,
  device = "png"
)

#### Exhibit 6: AEWR Map: AEWR Difference from National Trend ------------------

ts_ddd <- county_df %>%
  mutate(
    ch_aewr_ppi = aewr_ppi - aewr_ppi_l1,
    pch_aewr_ppi = (aewr_ppi - aewr_ppi_l1) / aewr_ppi_l1
  )

ggplot(ts_ddd, aes(x = pch_aewr_ppi)) +
  geom_histogram()

# need to make high growth and low growth regions

aewr_reg_ts_data <- county_df %>%
  select(aewr_region_num, aewr_state_ag_ppi, year) %>%
  arrange(year, aewr_region_num) %>%
  group_by(aewr_region_num, year) %>%
  summarise(aewr_state_ag_ppi = mean(aewr_state_ag_ppi, na.rm = T))

aewr_base <- aewr_reg_ts_data %>%
  filter(year < 2011) %>%
  group_by(aewr_region_num) %>%
  summarise(aewr_ppi_base = mean(aewr_state_ag_ppi, na.rm = T))

aewr_reg_ts_data <- merge(
  x = aewr_reg_ts_data,
  y = aewr_base,
  by = "aewr_region_num",
  all = T
)

aewr_reg_ts_data <- aewr_reg_ts_data %>%
  mutate(aewr_ppi_chbase = (aewr_state_ag_ppi - aewr_ppi_base) / aewr_ppi_base)

aewr_reg_ts_avg <- aewr_reg_ts_data %>%
  group_by(year) %>%
  summarise(aewr_ppi_chbase_avg = mean(aewr_ppi_chbase))

aewr_reg_ts_data <- merge(
  x = aewr_reg_ts_data,
  y = aewr_reg_ts_avg,
  by = "year",
  all = T
)

aewr_reg_ts_data <- aewr_reg_ts_data %>%
  mutate(aewr_ppi_chbase_detrend = (aewr_ppi_chbase - aewr_ppi_chbase_avg))

aewr_reg_ts_data_color <- aewr_reg_ts_data %>%
  filter(year == 2022 | year == 2008) %>%
  select(aewr_region_num, aewr_ppi_chbase_detrend, year) %>%
  pivot_wider(
    id_cols = aewr_region_num,
    names_from = "year",
    values_from = "aewr_ppi_chbase_detrend"
  )

aewr_reg_ts_data_color <- aewr_reg_ts_data_color %>%
  mutate(aewr_high_growth = `2022` - `2008`) %>%
  select(aewr_region_num, aewr_high_growth)

aewr_reg_ts_data <- merge(
  x = aewr_reg_ts_data,
  y = aewr_reg_ts_data_color,
  by = "aewr_region_num"
)

# need to make an AEWR region to stat xwalk

aewr_state_xwalk <- unique(
  county_df %>%
    select(aewr_region_num, statefips)
) %>%
  rename(statefip = statefips)

county_map_aewr <- merge(
  x = county_map,
  y = aewr_state_xwalk,
  by = "statefip",
  all.x = T,
  all.y = F
)

county_map_aewr <- merge(
  x = county_map_aewr,
  y = aewr_reg_ts_data,
  by = "aewr_region_num",
  all.x = T,
  all.y = F
)

aewr_map_data_growth <- ggplot(county_map_aewr) +
  geom_sf(aes(fill = aewr_high_growth), color = alpha("grey", 0.5)) +
  theme(
    panel.grid.major = element_line(
      color = gray(0.5),
      linetype = "dashed",
      size = 0.5
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
    midpoint = 0,
    high = muted("#b2182b"),
    name = "Change in\nAEWR Relative\nTo National\nTrend from\n2008-2022"
  )
aewr_map_data_growth


ggsave(
  filename = path_figures("map_aewr_change_from_trend_2022_2008.png"),
  aewr_map_data_growth,
  device = "png"
)


#### Exhibit 6B: AEWR p25 Bite Change 2008–2022 (County-Level Map) ## ----------

# County-level change in AEWR p25 bite from 2008 to 2022 (in 2012 $ levels)
county_bite_change <- county_df %>%
  filter(
    any_cropland_2007 == 1,
    county_simple_treatment_groups != "always takers",
    year %in% c(2008, 2022)
  ) %>%
  select(countyfips, year, aewr_cz_p25) %>%
  distinct(countyfips, year, .keep_all = TRUE) %>%
  pivot_wider(
    names_from = year,
    values_from = aewr_cz_p25,
    names_prefix = "bite_"
  ) %>%
  mutate(bite_change_2008_2022 = bite_2022 - bite_2008) %>%
  filter(!is.na(bite_change_2008_2022))

bite_median <- median(county_bite_change$bite_change_2008_2022, na.rm = TRUE)

county_map_bite <- merge(
  x = county_map,
  y = county_bite_change,
  by = "countyfips",
  all.x = TRUE,
  all.y = FALSE
)

map_aewr_cz_p25_bite_change <- ggplot(county_map_bite) +
  geom_sf(
    aes(fill = bite_change_2008_2022),
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
    midpoint = bite_median,
    high = muted("#b2182b"),
    name = "Change in\nAEWR p25 Bite\n2008–2022\n(2012 $)",
    na.value = "grey90"
  )
map_aewr_cz_p25_bite_change

ggsave(
  filename = path_figures("map_aewr_cz_p25_change_from_trend_2022_2008.png"),
  map_aewr_cz_p25_bite_change,
  device = "png"
)

#### Exhibit 8: AEWR TS Difference from Trend by Region ------------------------

# input data is made with the map above

plot_aewr_reg_ts <- ggplot(
  aewr_reg_ts_data,
  aes(
    x = year,
    y = aewr_ppi_chbase_detrend,
    group = as.factor(aewr_region_num),
    color = aewr_high_growth
  )
) +
  geom_line() +
  theme_clean() +
  scale_color_gradient2(
    low = muted("#2166ac"),
    mid = "white",
    midpoint = 0,
    high = muted("#b2182b"),
    name = "AEWR Deviation from Trend"
  ) +
  theme(legend.position = "none") +
  xlab("Year") +
  ylab("AEWR Difference from National Trend") +
  geom_hline(yintercept = 0, linetype = "dashed", color = "grey") +
  scale_y_continuous(
    breaks = c(-1, -.5, 0, .5, 1),
    labels = c("-100%", "-50%", "0%", "+50%", "+100%"),
    limits = c(-1, 1)
  ) +
  geom_vline(xintercept = 2012, linetype = "dashed", color = "black")
plot_aewr_reg_ts

ggsave(
  filename = path_figures("fig_ts_aewr_region_change_from_trend.png"),
  plot_aewr_reg_ts,
  device = "png"
)

#### Exhibit 9: Distribution of AEWR - p10 wage --------------------------------
# Calculate diffs
distribution_of_aewr_changes <- county_df %>%
  select(
    countyfips,
    year,
    aewr_cz_p10,
    aewr_cz_p10_l1,
    aewr_cz_p25,
    aewr_cz_p25_l1,
    aewr_cz_p50,
    aewr_cz_p50_l1
  ) %>%
  mutate(
    aewr_cz_p10_d1 = aewr_cz_p10 - aewr_cz_p10_l1,
    aewr_cz_p25_d1 = aewr_cz_p25 - aewr_cz_p25_l1,
    aewr_cz_p50_d1 = aewr_cz_p50 - aewr_cz_p50_l1,
    aewr_cz_p10_pd1 = (aewr_cz_p10 - aewr_cz_p10_l1) / aewr_cz_p10_l1,
    aewr_cz_p25_pd1 = (aewr_cz_p25 - aewr_cz_p25_l1) / aewr_cz_p25_l1,
    aewr_cz_p50_pd1 = (aewr_cz_p50 - aewr_cz_p50_l1) / aewr_cz_p50_l1,
  )

# CZ 10th percentile wage
plot_aewr_cz_diff <- distribution_of_aewr_changes %>%
  ggplot(aes(x = aewr_cz_p10_d1)) +
  geom_density(fill = "#69b3a2", color = "#e9ecef", alpha = 0.8) +
  xlab("1 year change in AEWR-CZ 10th percentile wage (USD)")

plot_aewr_cz_diff %>%
  ggsave(
    filename = path_figures("aewr_cz_p10_wage_change_1year.png"),
    width = 8,
    height = 5,
    device = "png"
  )

# CZ 25th percentile wage
plot_aewr_cz_diff <- distribution_of_aewr_changes %>%
  ggplot(aes(x = aewr_cz_p25_d1)) +
  geom_density(fill = "#69b3a2", color = "#e9ecef", alpha = 0.8) +
  xlab("1 year change in AEWR-CZ 25th percentile wage (USD)")

plot_aewr_cz_diff %>%
  ggsave(
    filename = path_figures("aewr_cz_p25_wage_change_1year.png"),
    width = 8,
    height = 5,
    device = "png"
  )

# CZ 50th percentile wage
plot_aewr_cz_diff <- distribution_of_aewr_changes %>%
  ggplot(aes(x = aewr_cz_p50_d1)) +
  geom_density(fill = "#69b3a2", color = "#e9ecef", alpha = 0.8) +
  xlab("1 year change in AEWR-CZ 50th percentile wage (USD)")

plot_aewr_cz_diff %>%
  ggsave(
    filename = path_figures("aewr_cz_p50_wage_change_1year.png"),
    width = 8,
    height = 5,
    device = "png"
  )

## % CH

# CZ 10th percentile wage
plot_aewr_cz_pdiff <- distribution_of_aewr_changes %>%
  filter(
    !is.na(aewr_cz_p10_pd1) &
      !is.infinite(aewr_cz_p10_pd1) &
      !is.na(aewr_cz_p10_l1) &
      aewr_cz_p10_l1 != 0
  ) %>%
  ggplot(aes(x = aewr_cz_p10_pd1)) +
  geom_density(fill = "#69b3a2", color = "#e9ecef", alpha = 0.8) +
  xlab("1 year change in AEWR-CZ 10th percentile wage (%)") +
  scale_x_continuous(limits = c(-2, 2))

plot_aewr_cz_pdiff %>%
  ggsave(
    filename = path_figures("aewr_cz_p10_wage_pchange_1year.png"),
    width = 8,
    height = 5,
    device = "png"
  )

# CZ 25th percentile wage
plot_aewr_cz_pdiff <- distribution_of_aewr_changes %>%
  filter(
    !is.na(aewr_cz_p25_pd1) &
      !is.infinite(aewr_cz_p25_pd1) &
      !is.na(aewr_cz_p25_pd1) &
      aewr_cz_p25_pd1 != 0
  ) %>%
  ggplot(aes(x = aewr_cz_p25_pd1)) +
  geom_density(fill = "#69b3a2", color = "#e9ecef", alpha = 0.8) +
  xlab("1 year change in AEWR-CZ 25th percentile wage (%)") +
  scale_x_continuous(limits = c(-2, 2))

plot_aewr_cz_pdiff %>%
  ggsave(
    filename = path_figures("aewr_cz_p25_wage_pchange_1year.png"),
    width = 8,
    height = 5,
    device = "png"
  )

# CZ 50th percentile wage
plot_aewr_cz_pdiff <- distribution_of_aewr_changes %>%
  filter(
    !is.na(aewr_cz_p50_pd1) &
      !is.infinite(aewr_cz_p50_pd1) &
      !is.na(aewr_cz_p50_pd1) &
      aewr_cz_p50_pd1 != 0
  ) %>%
  ggplot(aes(x = aewr_cz_p50_pd1)) +
  geom_density(fill = "#69b3a2", color = "#e9ecef", alpha = 0.8) +
  xlab("1 year change in AEWR-CZ 50th percentile wage (%)") +
  scale_x_continuous(limits = c(-2, 2))

plot_aewr_cz_pdiff %>%
  ggsave(
    filename = path_figures("aewr_cz_p50_wage_pchange_1year.png"),
    width = 8,
    height = 5,
    device = "png"
  )

write_parquet(
  aewr_reg_ts_data,
  path_int("analysis_aewr_region_trends.parquet")
)

dir.create(path_figures("aewr_ts"), recursive = TRUE, showWarnings = FALSE)

aewr_data <- read_parquet(path_processed("aewr_data_full.parquet"))

save_aewr_region_ts(aewr_data, "ln_aewr", "Log AEWR (nominal)", "ts_ln_aewr_nominal_")
save_aewr_region_ts(aewr_data, "ln_aewr_l1", "Log Change AEWR (nominal)", "ts_ln_aewr_l1_nominal_")
save_aewr_region_ts(aewr_data, "aewr_diff", "AEWR (nominal) difference from LOO trend", "ts_aewr_diffloo_nominal_")
save_aewr_region_ts(aewr_data, "aewr_ppi_diff", "AEWR (real) difference from LOO trend", "ts_aewr_diffloo_real_")
save_aewr_region_ts(aewr_data, "aewr", "AEWR (nominal)", "ts_aewr_nominal_")
save_aewr_region_ts(aewr_data, "aewr_ppi", "AEWR (real)", "ts_aewr_real_")
save_aewr_region_ts(aewr_data, "ch_aewr", "Change AEWR (nominal)", "ts_change_aewr_nominal_")
save_aewr_region_ts(aewr_data, "ch_aewr_ppi", "Change AEWR (real)", "ts_change_aewr_real_")
save_aewr_region_ts(aewr_data, "pch_aewr", "Percent Change AEWR (nominal)", "ts_percentchange_aewr_nominal_")
save_aewr_region_ts(aewr_data, "pch_aewr_ppi", "Percent Change AEWR (real)", "ts_percentchange_aewr_real_")

combo_plot <- ggplot(
  data = aewr_data,
  aes(
    x = year,
    y = pch_aewr,
    group = as.factor(aewr_region_num),
    color = as.factor(aewr_region_num)
  )
) +
  geom_line() +
  geom_vline(xintercept = 2007)
combo_plot
