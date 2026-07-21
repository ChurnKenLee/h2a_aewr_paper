# Purpose: Produce H-2A national trends, maps, and predicted-use descriptives.
# Inputs: processed H-2A panels, processed county panel, and county shapefile.
# Outputs: H-2A descriptive figures.
# Run after: code/c01_clean/05_h2a_county_panels.R and code/c02_build/04_finalize_county_panel.R.

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
h2a_data_ts <- read_parquet(path_processed("h2a_data_ts.parquet"))
h2a_data <- read_parquet(path_processed("h2a_data.parquet"))
county_map <- read_county_map(
  path_raw("county_shapefile", "tl_2020_us_county.zip")
)

#### Exhibit 3: H2A Use TS ---------------------------------------------------------

# levels

h2a_ts <- ggplot(
  data = subset(h2a_data_ts, case_year > 2007 & case_year <= 2022),
  aes(x = case_year)
) +
  geom_line(
    aes(y = h2a_nbr_workers_certified / 1000),
    color = "#4393c3",
    linewidth = 1.5
  ) +
  theme_clean() +
  xlab("Year") +
  ylab("H-2A Workers Certified (Thousands)") +
  scale_y_continuous(
    breaks = c(0, 50, 100, 150, 200, 250, 300, 350, 400),
    limits = c(0, 400)
  )
h2a_ts

# indexed

base_req <- h2a_data_ts$h2a_man_hours_requested[h2a_data_ts$case_year == 2011]
base_workers <- h2a_data_ts$h2a_nbr_workers_certified[
  h2a_data_ts$case_year == 2011
]
base_hours <- h2a_data_ts$h2a_man_hours_certified[h2a_data_ts$case_year == 2011]
base_aps <- h2a_data_ts$n_applications[h2a_data_ts$case_year == 2011]

h2a_data_ts_index <- h2a_data_ts %>%
  mutate(
    indx_workers_certified = (h2a_nbr_workers_certified / base_workers[1]) *
      100,
    indx_workers_req = (h2a_man_hours_requested / base_req[1]) * 100,
    indx_man_hours_certified = (h2a_man_hours_certified / base_hours[1]) * 100,
    indx_applications = (n_applications / base_aps[1]) * 100
  )

h2a_data_ts_index <- h2a_data_ts_index %>%
  select(
    case_year,
    indx_workers_certified,
    indx_man_hours_certified,
    indx_applications,
    indx_workers_req
  )

h2a_data_ts_index <- h2a_data_ts_index %>%
  pivot_longer(
    cols = starts_with("indx_"),
    names_to = "series",
    names_prefix = "indx_",
    values_to = "indx",
    values_drop_na = F
  )


# zeros should not be graphed #

h2a_data_ts_index <- h2a_data_ts_index %>%
  transform(indx = ifelse(indx < 50, NA, indx))

h2a_indx_ts <- ggplot(
  data = subset(h2a_data_ts_index, case_year > 2007 & case_year <= 2022),
  aes(
    x = case_year,
    y = indx,
    group = series,
    color = series,
    linetype = series
  )
) +
  geom_line(linewidth = 1.25, alpha = 0.75) +
  theme_clean() +
  scale_color_manual(
    values = c("#2166ac", "#4393c3", "#F198A2", "#b2182b"),
    labels = c(
      "Applications",
      "Man Hours Certified",
      "Workers Certified",
      "Workers Requested"
    )
  ) +
  scale_linetype_manual(
    values = c(5, 4, 3, 1),
    labels = c(
      "Applications",
      "Man Hours Certified",
      "Workers Certified",
      "Workers Requested"
    )
  ) +
  theme(legend.title = element_blank(), legend.position = "bottom") +
  xlab("Year") +
  ylab("Indexed Values (2011 = 100)") +
  geom_hline(yintercept = 100, linetype = "dashed") +
  guides(color = guide_legend(nrow = 2), linetype = guide_legend(nrow = 2))
h2a_indx_ts

ggsave(
  filename = path_figures("fig_line_ts_h2a_workers_certified.png"),
  h2a_ts,
  device = "png"
)
ggsave(
  filename = path_figures("fig_line_ts_h2a_indexes.png"),
  h2a_indx_ts,
  device = "png"
)

#### Exhibit 4: H2A Map: 2012 use ----------------------------------------------


h2a_map_data <- NULL

h2a_map_data_2012 <- h2a_data %>%
  filter(census_period == 2012) %>%
  mutate(ln_h2a_workers_cert_2012 = log(nbr_workers_certified_start_year)) %>%
  select(
    countyfips,
    ln_h2a_workers_cert_2012,
    nbr_workers_certified_start_year
  ) %>%
  rename(nbr_workers_certified_2012 = nbr_workers_certified_start_year)

h2a_map_data_2017 <- h2a_data %>%
  filter(census_period == 2017) %>%
  mutate(ln_h2a_workers_cert_2017 = log(nbr_workers_certified_start_year)) %>%
  select(
    countyfips,
    ln_h2a_workers_cert_2017,
    nbr_workers_certified_start_year
  ) %>%
  rename(nbr_workers_certified_2017 = nbr_workers_certified_start_year)

h2a_map_data_2022 <- h2a_data %>%
  filter(census_period == 2022) %>%
  mutate(ln_h2a_workers_cert_2022 = log(nbr_workers_certified_start_year)) %>%
  select(
    countyfips,
    ln_h2a_workers_cert_2022,
    nbr_workers_certified_start_year
  ) %>%
  rename(nbr_workers_certified_2022 = nbr_workers_certified_start_year)


h2a_map_data <- merge(
  x = h2a_map_data_2012,
  y = h2a_map_data_2017,
  by = "countyfips"
)
h2a_map_data <- merge(
  x = h2a_map_data,
  y = h2a_map_data_2022,
  by = "countyfips"
)

county_map_fips <- county_map

h2a_map_data <- merge(
  x = county_map_fips,
  y = h2a_map_data,
  by = "countyfips",
  all.x = T,
  all.y = F
)

# 2012

h2a_map_data_2012 <- ggplot(h2a_map_data) +
  geom_sf(aes(fill = ln_h2a_workers_cert_2012), color = alpha("grey", 0.5)) +
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
    low = "white",
    mid = muted("#C3DBF4"),
    midpoint = 5,
    high = muted("#0B243C"),
    name = "H-2A Workers\nCertified (log)"
  )
h2a_map_data_2012

ggsave(
  filename = path_figures("map_h2a_workers_2012.png"),
  h2a_map_data_2012,
  device = "png"
)

# no logs

h2a_map_data$nbr_workers_certified_2012[is.na(
  h2a_map_data$nbr_workers_certified_2012
)] <- 0

h2a_map_data_2012_nolog <- ggplot(h2a_map_data) +
  geom_sf(aes(fill = nbr_workers_certified_2012), color = alpha("grey", 0.5)) +
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
    low = "white",
    mid = muted("#C3DBF4"),
    midpoint = 3000,
    high = muted("#0B243C"),
    name = "H-2A Workers\nCertified (Levels)"
  )
h2a_map_data_2012_nolog

ggsave(
  filename = path_figures("map_h2a_workers_2012_nolog.png"),
  h2a_map_data_2012_nolog,
  device = "png"
)

## Predicted H2A Share Histogram -----------------------------------------------

predicted_hist_data <- county_df %>%
  select(
    h2a_predicted_share_2011,
    year,
    countyfips,
    state_abbrev,
    county_simple_treatment_groups
  ) %>%
  filter(year == 2011)

summary(predicted_hist_data$h2a_predicted_share_2011)

hist_pred <- ggplot(
  data = predicted_hist_data,
  aes(
    x = h2a_predicted_share_2011,
    group = county_simple_treatment_groups,
    fill = county_simple_treatment_groups
  )
) +
  geom_histogram(alpha = 0.5, position = "identity", binwidth = 0.01) +
  geom_vline(xintercept = 0.01, linetype = "dashed", color = "black") +
  scale_x_continuous(limits = c(-0.01, 1)) +
  theme_classic() +
  scale_fill_manual(values = c("#b2182b", "#2166ac")) +
  xlab("Predicted H2A Use as Share of Agricultural Employment") +
  theme(legend.position = "bottom", legend.title = element_blank())

hist_pred

ggsave(
  filename = path_figures("hist_h2a_predicted_share_2011.png"),
  hist_pred,
  device = "png"
)

## Predicted H2A map -----------------------------------------------------------

pred_h2a_Map_data <- merge(
  x = county_map,
  y = predicted_hist_data,
  by = "countyfips",
  all.x = T,
  all.y = F
)

pred_h2a_Map <- ggplot(pred_h2a_Map_data) +
  geom_sf(
    aes(fill = county_simple_treatment_groups),
    color = alpha("grey", 0.5)
  ) +
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
  scale_fill_manual(values = c("#b2182b", "#2166ac")) +
  theme(legend.position = "bottom", legend.title = element_blank())
pred_h2a_Map

ggsave(
  filename = path_figures("map_predicted_h2a_groups.png"),
  pred_h2a_Map,
  device = "png"
)


#### Exhibit 5: H2A Map: Change from 2012 use ----------------------------------

h2a_change_data <- h2a_data %>%
  filter(census_period == 2012 | census_period == 2022) %>%
  select(countyfips, census_period, nbr_workers_certified_start_year) %>%
  pivot_wider(
    id_cols = countyfips,
    names_from = census_period,
    values_from = nbr_workers_certified_start_year
  )

h2a_change_data[is.na(h2a_change_data)] <- 0 # NAs are zeros here

h2a_change_data <- h2a_change_data %>% # make change vars
  mutate(
    ch_workers = `2022` - `2012`,
    pch_workers = (`2022` - `2012`) / `2012`,
    new_h2a = ifelse(`2012` == 0 & `2022` != 0, `2022`, NA)
  )

h2a_change_map_data <- merge(
  x = county_map,
  y = h2a_change_data,
  by = "countyfips",
  all.x = T,
  all.y = F
)

# change from 2012

h2a_map_data_ch2012 <- ggplot(h2a_change_map_data) +
  geom_sf(aes(fill = log(ch_workers)), color = alpha("grey", 0.5)) +
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
    low = "white",
    mid = muted("#C3DBF4"),
    midpoint = 9,
    high = muted("#0B243C"),
    name = "Log Change in\nH2-A Workers\nCertified"
  )
h2a_map_data_ch2012

ggsave(
  filename = path_figures("map_h2a_workers_ch2022_2012.png"),
  h2a_map_data_ch2012,
  device = "png"
)

# New from 2012

h2a_map_data_ch2012_new <- ggplot(h2a_change_map_data) +
  geom_sf(aes(fill = log(new_h2a)), color = alpha("grey", 0.5)) +
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
    low = "white",
    mid = muted("#C3DBF4"),
    midpoint = 7,
    high = muted("#0B243C"),
    name = "Log H2-A\nWorkers\nCertified in\nnew counties\nsince 2012"
  )
h2a_map_data_ch2012_new

ggsave(
  filename = path_figures("map_h2a_workers_ch2022_2012_newcounties.png"),
  h2a_map_data_ch2012_new,
  device = "png"
)
