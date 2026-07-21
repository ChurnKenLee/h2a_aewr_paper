# Purpose: Visualize treatment exposure and DD comparison groups.
# Inputs: processed county panel, county shapefile, and AEWR regional trend summary.
# Outputs: exposure maps and DD descriptive figures.
# Run after: 01_aewr_descriptives.R and code/c02_build/04_finalize_county_panel.R.

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
county_map <- read_county_map(
  path_raw("county_shapefile", "tl_2020_us_county.zip")
)
aewr_reg_ts_data <- read_parquet(
  path_int("analysis_aewr_region_trends.parquet")
)
aewr_reg_ts_data_color <- aewr_reg_ts_data %>%
  distinct(aewr_region_num, aewr_high_growth)

#### Exhibit 7: Exposure Map: Counties by Sample Classification ----------------
## DDD Sample Map


ddd_map_data <- county_df %>%
  dplyr::select(
    countyfips,
    high_h2a_share_75,
    high_h2a_share_50,
    any_cropland_2007
  )

ddd_map <- merge(
  x = county_map,
  y = ddd_map_data,
  by = "countyfips",
  all.x = T,
  all.y = F
)

ddd_map <- ddd_map %>%
  transform(
    high_h2a_share_75 = ifelse(any_cropland_2007 == 1, high_h2a_share_75, NA),
    high_h2a_share_50 = ifelse(high_h2a_share_50 == 1, high_h2a_share_50, NA)
  )

ddd_samp_map <- ggplot(ddd_map) +
  geom_sf(aes(fill = as.factor(high_h2a_share_75))) +
  theme(
    panel.grid.major = element_line(
      color = gray(0.5),
      linetype = "dashed",
      size = 0.5
    ),
    panel.background = element_rect(fill = "aliceblue")
  ) +
  theme_bw() +
  scale_fill_manual(
    values = c("#f4a582", "#4393c3"),
    na.value = "#E5E4E2",
    name = "Top Quartile\nin H2A Use\nPre-2012"
  ) +
  annotation_north_arrow(
    location = "bl",
    which_north = "true",
    pad_x = unit(0.05, "in"),
    pad_y = unit(0.25, "in"),
    style = north_arrow_fancy_orienteering
  ) +
  annotation_scale(location = "bl", width_hint = 0.4)
ddd_samp_map

ggsave(
  filename = path_figures("map_aewr_ddd_sample_dropnocropland.png"),
  ddd_samp_map,
  device = "png"
)

#### Exhibit 10: DDD by Graph with predicted usage -----------------------
# need to make the ts graph using the trend growth as a dummy


aewr_reg_ts_data <- aewr_reg_ts_data %>%
  mutate(
    aewr_high_growth_p50 = ifelse(
      aewr_high_growth >= median(aewr_reg_ts_data_color$aewr_high_growth),
      1,
      0
    ),
    aewr_high_growth_positive = ifelse(aewr_high_growth > 0, 1, 0),
    aewr_above_trend_growth = ifelse(aewr_ppi_chbase_detrend > 0, 1, 0)
  )

# need: nbr_workers_requested_start_year, high / low dummy, growth dummy, year


aewr_reg_ts_h2a <- county_df %>%
  select(
    year,
    aewr_region_num,
    nbr_workers_certified_start_year,
    county_simple_treatment_groups
  )

aewr_reg_ts_data <- merge(
  x = aewr_reg_ts_h2a,
  y = aewr_reg_ts_data,
  by = c("aewr_region_num", "year"),
  all = T
)

# collapse

aewr_reg_ts_data_collapse <- aewr_reg_ts_data %>%
  filter(!is.na(county_simple_treatment_groups)) %>%
  group_by(county_simple_treatment_groups, aewr_above_trend_growth, year) %>%
  summarise(
    nbr_workers_certified_start_year = sum(
      nbr_workers_certified_start_year,
      na.rm = T
    )
  )

# index to 2012

aewr_reg_ts_data_base <- aewr_reg_ts_data_collapse %>%
  filter(year == 2011)

aewr_reg_ts_data_base <- aewr_reg_ts_data_base %>%
  rename(nbr_workers_certified_base = nbr_workers_certified_start_year) %>%
  select(-year)

aewr_reg_ts_data_collapse <- merge(
  x = aewr_reg_ts_data_collapse,
  y = aewr_reg_ts_data_base,
  by = c("county_simple_treatment_groups", "aewr_above_trend_growth")
)

aewr_reg_ts_data_collapse <- aewr_reg_ts_data_collapse %>%
  mutate(
    nbr_workers_certified_indx2011 = nbr_workers_certified_start_year /
      nbr_workers_certified_base
  )

aewr_reg_ts_data_collapse <- aewr_reg_ts_data_collapse %>%
  mutate(
    group_lab = case_when(
      (county_simple_treatment_groups == "always takers") &
        (aewr_above_trend_growth == 1) ~ "High AEWR Growth, Always Takers",
      (county_simple_treatment_groups == "always takers") &
        (aewr_above_trend_growth == 0) ~ "Low AEWR Growth, Always Takers",
      (county_simple_treatment_groups != "always takers") &
        (aewr_above_trend_growth == 1) ~ "High AEWR Growth, Adopters",
      (county_simple_treatment_groups != "always takers") &
        (aewr_above_trend_growth == 0) ~ "Low AEWR Growth, Adopters"
    )
  )


plot_aewr_use_ts_DDD <- ggplot(
  aewr_reg_ts_data_collapse,
  aes(
    x = year,
    y = nbr_workers_certified_indx2011,
    linetype = as.factor(group_lab),
    color = as.factor(group_lab)
  )
) +
  geom_line() +
  theme_classic() +
  scale_color_manual(
    values = c(
      "#b2182b",
      "#2166ac",

      "#b2182b",
      "#2166ac"
    )
  ) +
  scale_linetype_manual(values = c(1, 1, 2, 2)) +
  labs(color = "group_lab", linetype = "group_lab") +
  guides(
    color = guide_legend(ncol = 2, byrow = TRUE),
    linetype = guide_legend(ncol = 2, byrow = TRUE),
  ) +
  theme(legend.position = "bottom") +
  theme(legend.title = element_blank()) +
  geom_hline(yintercept = 1, alpha = 0.5) +
  ylab("Number of H2-A Workers Requested\n(Indexed to 2011)")
plot_aewr_use_ts_DDD

ggsave(
  plot_aewr_use_ts_DDD,
  filename = path_figures("fig_ts_aewr_growth_exposure_using_predicted_ddd.png")
)


#### Exhibit 11: CZ x AEWR Region Deviation from Trend (Deciles) ## -------

# Aggregate aewr_cz_p25 to CZ x AEWR region x year
aewr_cz_p25_czreg_ts_data <- county_df %>%
  filter(
    any_cropland_2007 == 1,
    county_simple_treatment_groups != "always takers",
    !is.na(cz_out10),
    !is.na(aewr_region_num)
  ) %>%
  mutate(cz_aewr_id = paste0(cz_out10, "_", aewr_region_num)) %>%
  group_by(cz_aewr_id, year) %>%
  summarise(
    aewr_cz_p25_mean = mean(aewr_cz_p25, na.rm = TRUE),
    .groups = "drop"
  )

# National average bite across all CZ x region units by year
czreg_national_avg <- aewr_cz_p25_czreg_ts_data %>%
  group_by(year) %>%
  summarise(national_avg = mean(aewr_cz_p25_mean, na.rm = TRUE))

aewr_cz_p25_czreg_ts_data <- merge(
  x = aewr_cz_p25_czreg_ts_data,
  y = czreg_national_avg,
  by = "year"
)

aewr_cz_p25_czreg_ts_data <- aewr_cz_p25_czreg_ts_data %>%
  mutate(deviation = (aewr_cz_p25_mean - national_avg) / national_avg)

# Classify CZ x region units into 10 decile bins based on 2022 endpoint deviation
czreg_2022_dev <- aewr_cz_p25_czreg_ts_data %>%
  filter(year == 2022) %>%
  select(cz_aewr_id, deviation) %>%
  rename(dev_2022 = deviation) %>%
  mutate(
    decile = ntile(dev_2022, 10),
    above_trend_stable = ifelse(dev_2022 > 0, 1, 0)
  )

aewr_cz_p25_czreg_ts_data <- merge(
  x = aewr_cz_p25_czreg_ts_data,
  y = czreg_2022_dev %>% select(cz_aewr_id, decile, above_trend_stable),
  by = "cz_aewr_id"
)

# Average deviation by decile x year for plotting
czreg_decile_ts <- aewr_cz_p25_czreg_ts_data %>%
  group_by(decile, year) %>%
  summarise(avg_deviation = mean(deviation, na.rm = TRUE), .groups = "drop")

plot_czreg_decile_ts <- ggplot(
  czreg_decile_ts,
  aes(
    x = year,
    y = avg_deviation,
    group = as.factor(decile),
    color = decile
  )
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
plot_czreg_decile_ts

ggsave(
  filename = path_figures("fig_ts_aewr_cz_p25_czregion_deviations_deciles.png"),
  plot_czreg_decile_ts,
  width = 8,
  height = 5,
  device = "png"
)

#### Exhibit 12: DD Visual - Indexed H-2A Use, Above vs Below Trend ## ---------

# Merge stable above/below classification (based on 2022 endpoint) to county_df
county_df_dd_vis <- county_df %>%
  filter(
    any_cropland_2007 == 1,
    county_simple_treatment_groups != "always takers",
    !is.na(cz_out10),
    !is.na(aewr_region_num)
  ) %>%
  mutate(cz_aewr_id = paste0(cz_out10, "_", aewr_region_num))

county_df_dd_vis <- merge(
  x = county_df_dd_vis,
  y = czreg_2022_dev %>% select(cz_aewr_id, above_trend_stable),
  by = "cz_aewr_id",
  all.x = TRUE
)

# Collapse H-2A use by above_trend_stable x year, index to 2011 = 1
dd_vis_collapse <- county_df_dd_vis %>%
  filter(!is.na(above_trend_stable)) %>%
  group_by(above_trend_stable, year) %>%
  summarise(
    h2a_use = sum(nbr_workers_certified_start_year, na.rm = TRUE),
    .groups = "drop"
  )

dd_vis_base <- dd_vis_collapse %>%
  filter(year == 2011) %>%
  rename(h2a_base = h2a_use) %>%
  select(above_trend_stable, h2a_base)

dd_vis_collapse <- merge(
  dd_vis_collapse,
  dd_vis_base,
  by = "above_trend_stable"
)

dd_vis_collapse <- dd_vis_collapse %>%
  mutate(
    h2a_indx = h2a_use / h2a_base,
    trend_lab = ifelse(
      above_trend_stable == 1,
      "Above National Trend (2022 Endpoint)",
      "Below National Trend (2022 Endpoint)"
    )
  )

plot_dd_indexed_h2a <- ggplot(
  dd_vis_collapse,
  aes(
    x = year,
    y = h2a_indx,
    color = trend_lab,
    linetype = trend_lab
  )
) +
  geom_line(linewidth = 1.25) +
  scale_color_manual(values = c("#b2182b", "#2166ac")) +
  scale_linetype_manual(values = c(1, 2)) +
  geom_hline(yintercept = 1, linetype = "dashed", alpha = 0.5) +
  geom_vline(xintercept = 2011, linetype = "dashed", color = "black") +
  theme_clean() +
  theme(legend.position = "bottom", legend.title = element_blank()) +
  xlab("Year") +
  ylab("H-2A Workers Certified (Indexed to 2011 = 1)")
plot_dd_indexed_h2a

ggsave(
  filename = path_figures("fig_ts_aewr_cz_p25_dd_indexed_h2a.png"),
  plot_dd_indexed_h2a,
  width = 7,
  height = 5,
  device = "png"
)
