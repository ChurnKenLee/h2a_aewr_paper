## H2A: Analysis Figures and Tables
## Phil Hoxie
## 1/31/24

gc() 

library(sf)
library(tidyverse)
#library(USAboundaries)
library(ggspatial)
library(scales)
library(cowplot)
library(ggthemes)
library(fixest)
library(ggfixest)

Sys.sleep(1)

set.seed(12345)

date <- paste0(
  substr(Sys.Date(), 1, 4),
  substr(Sys.Date(), 6, 7),
  substr(Sys.Date(), 9, 10)
)

#### Load and Clean County Shape Files -------------------------------------------

cnt_shp <- st_read(paste0(folder_data, "tl_2020_us_county.shp"))

fips_codes <- read_csv(paste0(folder_data, "fips_codes.csv"))

aewr_df <- read_parquet(paste0(folder_data, "aewr.parquet"))

border_df <- read_parquet(paste0(folder_data, "border_df_analysis.parquet"))
aewr_data <- read_parquet(paste0(folder_data, "aewr_data.parquet"))
h2a_data_ts <- read_parquet(paste0(folder_data, "h2a_data_ts.parquet"))
h2a_data <- read_parquet(paste0(folder_data, "h2a_data.parquet"))

border_df_yearly <- read_parquet(paste0(
  folder_data,
  "border_df_analysis_year.parquet"
))
aewr_data_full <- read_parquet(paste0(folder_data, "aewr_data_full.parquet"))

#### County DDD Design -----------------------------------------------------------

county_df <- read_parquet(paste0(
  folder_data,
  "county_df_analysis_year.parquet"
))

names(county_df)

#### Exhibit 0: Distribution of AEWR Bite Variables (real, non-lagged) ---------

bite_long <- county_df %>%
  filter(any_cropland_2007 == 1) %>%
  select(countyfips, year, aewr_cz_p10, aewr_cz_p25, aewr_cz_p50) %>%
  pivot_longer(
    cols = c(aewr_cz_p10, aewr_cz_p25, aewr_cz_p50),
    names_to = "percentile",
    values_to = "bite"
  ) %>%
  mutate(percentile = recode(percentile,
    aewr_cz_p10 = "AEWR minus p10",
    aewr_cz_p25 = "AEWR minus p25",
    aewr_cz_p50 = "AEWR minus p50"
  ))

ggplot(bite_long, aes(x = bite)) +
  geom_histogram(bins = 50, fill = "steelblue", color = "white") +
  geom_vline(xintercept = 0, linetype = "dashed", color = "red") +
  facet_wrap(~ percentile, ncol = 1, scales = "free_y") +
  labs(
    x = "Real AEWR minus Real CZ Wage Percentile (2012 $)",
    y = "Count",
    title = "Distribution of AEWR Bite Variables"
  ) +
  theme_clean()

ggsave(
  paste0(folder_output, "hist_aewr_cz_bite_variables.png"),
  width = 7, height = 8
)

# map

county_map <- cnt_shp %>%
  mutate(statefip = as.numeric(STATEFP))

county_map <- county_map %>%
  filter(statefip <= 56 & statefip != 2 & statefip != 15)

# simplify the map
county_map <- st_simplify(
  county_map,
  preserveTopology = FALSE,
  dTolerance = 1000
)

ggplot(county_map) +
  geom_sf()
f
county_map$countyfips <- as.numeric(str_c(
  county_map$STATEFP,
  county_map$COUNTYFP
))


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
  filename = paste0(folder_output, "ts_national_aewr_real.png"),
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
  filename = paste0(folder_output, "ts_national_aewr_nominal.png"),
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
  filename = paste0(folder_output, "ts_national_aewr_cz_p25_bite.png"),
  aewr_bite_ts,
  device = "png"
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
head(h2a_data_ts_index)


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
  filename = paste0(folder_output, "fig_line_ts_h2a_workers_certified.png"),
  h2a_ts,
  device = "png"
)
ggsave(
  filename = paste0(folder_output, "fig_line_ts_h2a_indexes.png"),
  h2a_indx_ts,
  device = "png"
)

#### Exhibit 4: H2A Map: 2012 use ----------------------------------------------

head(county_map)

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
  filename = paste0(folder_output, "map_H2A_workers_2012.png"),
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
  filename = paste0(folder_output, "map_H2A_workers_2012_nolog.png"),
  h2a_map_data_2012_nolog,
  device = "png"
)

## Predicted H2A Share Histogram -----------------------------------------------

predicted_hist_data <- county_df %>% 
  select(h2a_predicted_share_2011, year, countyfips, state_abbrev, county_simple_treatment_groups) %>% 
  filter(year == 2011)

summary(predicted_hist_data$h2a_predicted_share_2011)

hist_pred <- ggplot(data = predicted_hist_data, aes(x = h2a_predicted_share_2011, 
                                                    group = county_simple_treatment_groups,
                                                    fill = county_simple_treatment_groups)) +
  geom_histogram(alpha = 0.5, position = "identity", binwidth = 0.01) +
  geom_vline(xintercept = 0.01, linetype = "dashed", color = "black") +
  scale_x_continuous(limits = c(-0.01, 1)) +
  theme_classic() +
  scale_fill_manual(values = c("#b2182b","#2166ac" ))+
  xlab("Predicted H2A Use as Share of Agricultural Employment") +
  theme(legend.position = "bottom", legend.title = element_blank())

hist_pred

ggsave(
  filename = paste0(folder_output, "hist_h2a_predicted_share_2011.png"),
  hist_pred,
  device = "png"
)

## Predicted H2A map -----------------------------------------------------------

pred_h2a_Map_data <- merge(x = county_map, 
                           y = predicted_hist_data, 
                           by = "countyfips",
                           all.x = T,
                           all.y = F
                           )

pred_h2a_Map <- ggplot(pred_h2a_Map_data) +
  geom_sf(aes(fill = county_simple_treatment_groups), color = alpha("grey", 0.5)) +
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
  scale_fill_manual(values = c("#b2182b","#2166ac" ))+
  theme(legend.position = "bottom", legend.title = element_blank())
pred_h2a_Map

ggsave(
  filename = paste0(folder_output, "map_predicted_h2a_groups.png"),
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
  filename = paste0(folder_output, "map_H2A_workers_ch2022_2012.png"),
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
  filename = paste0(
    folder_output,
    "map_H2A_workers_ch2022_2012_newcounties.png"
  ),
  h2a_map_data_ch2012_new,
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
  filename = paste0(folder_output, "map_aewr_change_from_trend_2022_2008.png"),
  aewr_map_data_growth,
  device = "png"
)

gc()
#### Exhibit 7: Exposure Map: Counties by Sample Classification ----------------

## DDD Sample Map

head(county_df)

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
  filename = paste0(folder_output, "map_aewr_ddd_sample_dropnocropland.png"),
  ddd_samp_map,
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
  )+
  geom_vline(xintercept = 2012, linetype = "dashed", color= "black")
plot_aewr_reg_ts

ggsave(
  filename = paste0(folder_output, "fig_ts_aewr_region_change_from_trend.png"),
  plot_aewr_reg_ts,
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
  pivot_wider(names_from = year, values_from = aewr_cz_p25, names_prefix = "bite_") %>%
  mutate(bite_change_2008_2022 = bite_2022 - bite_2008) %>%
  filter(!is.na(bite_change_2008_2022))

bite_median <- median(county_bite_change$bite_change_2008_2022, na.rm = TRUE)

county_map_bite <- merge(
  x = county_map,
  y = county_bite_change,
  by = "countyfips",
  all.x = TRUE, all.y = FALSE
)

map_aewr_cz_p25_bite_change <- ggplot(county_map_bite) +
  geom_sf(aes(fill = bite_change_2008_2022), color = alpha("grey", 0.3), linewidth = 0.1) +
  theme(
    panel.grid.major = element_line(
      color = gray(0.5), linetype = "dashed", linewidth = 0.5
    ),
    panel.background = element_rect(fill = "aliceblue")
  ) +
  theme_bw() +
  annotation_north_arrow(
    location = "bl", which_north = "true",
    pad_x = unit(0.05, "in"), pad_y = unit(0.25, "in"),
    style = north_arrow_fancy_orienteering
  ) +
  annotation_scale(location = "bl", width_hint = 0.4) +
  scale_fill_gradient2(
    low  = muted("#2166ac"), mid = "white", midpoint = bite_median,
    high = muted("#b2182b"),
    name = "Change in\nAEWR p25 Bite\n2008–2022\n(2012 $)",
    na.value = "grey90"
  )
map_aewr_cz_p25_bite_change

ggsave(
  filename = paste0(folder_output, "map_aewr_cz_p25_change_from_trend_2022_2008.png"),
  map_aewr_cz_p25_bite_change,
  device = "png"
)

#### Exhibit 9: DDD by Graph a la Jeff -----------------------------------------

# # need to make the ts graph using the trend growth as a dummy

# head(aewr_reg_ts_data)

# aewr_reg_ts_data <- aewr_reg_ts_data %>%
#   mutate(
#     aewr_high_growth_p50 = ifelse(
#       aewr_high_growth >= median(aewr_reg_ts_data_color$aewr_high_growth),
#       1,
#       0
#     ),
#     aewr_high_growth_positive = ifelse(aewr_high_growth > 0, 1, 0)
#   )

# # need: nbr_workers_requested_start_year, high / low dummy, growth dummy, year

# names(county_df)

# aewr_reg_ts_h2a <- county_df %>%
#   select(
#     year,
#     aewr_region_num,
#     nbr_workers_certified_start_year,
#     high_h2a_share_75,
#     high_h2a_share_66,
#     high_h2a_share_50
#   )

# aewr_reg_ts_data <- merge(
#   x = aewr_reg_ts_h2a,
#   y = aewr_reg_ts_data,
#   by = c("aewr_region_num", "year"),
#   all = T
# )

# # collapse

# aewr_reg_ts_data_collapse <- aewr_reg_ts_data %>%
#   group_by(high_h2a_share_75, aewr_high_growth_p50, year) %>%
#   summarise(
#     nbr_workers_certified_start_year = sum(
#       nbr_workers_certified_start_year,
#       na.rm = T
#     )
#   )

# # index to 2012

# aewr_reg_ts_data_base <- aewr_reg_ts_data_collapse %>%
#   filter(year == 2011)

# aewr_reg_ts_data_base <- aewr_reg_ts_data_base %>%
#   rename(nbr_workers_certified_base = nbr_workers_certified_start_year) %>%
#   select(-year)

# aewr_reg_ts_data_collapse <- merge(
#   x = aewr_reg_ts_data_collapse,
#   y = aewr_reg_ts_data_base,
#   by = c("high_h2a_share_75", "aewr_high_growth_p50")
# )

# aewr_reg_ts_data_collapse <- aewr_reg_ts_data_collapse %>%
#   mutate(
#     nbr_workers_certified_indx2011 = nbr_workers_certified_start_year /
#       nbr_workers_certified_base
#   )

# aewr_reg_ts_data_collapse <- aewr_reg_ts_data_collapse %>%
#   mutate(
#     group_lab = ifelse(
#       high_h2a_share_75 == 1 & aewr_high_growth_p50 == 1,
#       "High AEWR Growth, High Exposure",
#       ifelse(
#         high_h2a_share_75 == 1 & aewr_high_growth_p50 == 0,
#         "Low AEWR Growth, High Exposure",
#         ifelse(
#           high_h2a_share_75 == 0 & aewr_high_growth_p50 == 1,
#           "High AEWR Growth, Low Exposure",
#           "Low AEWR Growth, Low Exposure"
#         )
#       )
#     )
#   )

# plot_aewr_use_ts_DDD <- ggplot(
#   aewr_reg_ts_data_collapse,
#   aes(
#     x = year,
#     y = nbr_workers_certified_indx2011,
#     linetype = as.factor(group_lab),
#     color = as.factor(group_lab)
#   )
# ) +
#   geom_line() +
#   theme_classic() +
#   scale_color_manual(values = c("#b2182b", "#b2182b", "#2166ac", "#2166ac")) +
#   scale_linetype_manual(values = c(1, 5, 3, 4)) +
#   labs(color = "group_lab", linetype = "group_lab") +
#   guides(
#     color = guide_legend(ncol = 2, byrow = TRUE),
#     linetype = guide_legend(ncol = 2, byrow = TRUE),
#   ) +
#   theme(legend.position = "bottom") +
#   theme(legend.title = element_blank()) +
#   geom_hline(yintercept = 1, alpha = 0.5) +
#   ylab("Number of H2-A Workers Requested\n(Indexed to 2011)")
# plot_aewr_use_ts_DDD

# ggsave(
#   plot_aewr_use_ts_DDD,
#   filename = paste0(folder_output, "fig_ts_aewr_growth_exposure_DDD.png")
# )

# head(aewr_reg_ts_data_collapse)

#### Exhibit 9B: DDD by Graph with predicted usage -----------------------
# need to make the ts graph using the trend growth as a dummy

head(aewr_reg_ts_data)

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

names(county_df)

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
  group_by(county_simple_treatment_groups
    ,
    aewr_above_trend_growth,
    year
  ) %>%
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

# aewr_reg_ts_data_collapse <- aewr_reg_ts_data_collapse %>%
#   mutate(
#     group_lab = case_when(
#       (county_treatment_group_classification == "always takers") &
#         (aewr_high_growth_p50 == 1) ~ "High AEWR Growth, Always Takers",
#       (county_treatment_group_classification == "always takers") &
#         (aewr_high_growth_p50 == 0) ~ "Low AEWR Growth, Always Takers",
#       (county_treatment_group_classification == "adopters") &
#         (aewr_high_growth_p50 == 1) ~ "High AEWR Growth, Adopters",
#       (county_treatment_group_classification == "adopters") &
#         (aewr_high_growth_p50 == 0) ~ "Low AEWR Growth, Adopters",
#       (county_treatment_group_classification == "never takers") &
#         (aewr_high_growth_p50 == 1) ~ "High AEWR Growth, Never Takers",
#       (county_treatment_group_classification == "never takers") &
#         (aewr_high_growth_p50 == 0) ~ "Low AEWR Growth, Never Takers"
#     )
#   )

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

head(aewr_reg_ts_data_collapse)

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
  filename = paste0(
    folder_output,
    "fig_ts_aewr_growth_exposure_using_predicted_DDD.png"
  )
)

head(aewr_reg_ts_data_collapse)

#### Exhibit 9B New: CZ x AEWR Region Deviation from Trend (Deciles) ## -------

# Aggregate aewr_cz_p25 to CZ x AEWR region x year
aewr_cz_p25_czreg_ts_data <- county_df %>%
  filter(
    any_cropland_2007 == 1,
    county_simple_treatment_groups != "always takers",
    !is.na(cz_out10), !is.na(aewr_region_num)
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
    x = year, y = avg_deviation,
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
  filename = paste0(folder_output, "fig_ts_aewr_cz_p25_czregion_deviations_deciles.png"),
  plot_czreg_decile_ts,
  width = 8, height = 5,
  device = "png"
)

#### Exhibit 9C: DD Visual - Indexed H-2A Use, Above vs Below Trend ## ---------

# Merge stable above/below classification (based on 2022 endpoint) to county_df
county_df_dd_vis <- county_df %>%
  filter(
    any_cropland_2007 == 1,
    county_simple_treatment_groups != "always takers",
    !is.na(cz_out10), !is.na(aewr_region_num)
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

dd_vis_collapse <- merge(dd_vis_collapse, dd_vis_base, by = "above_trend_stable")

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
    x = year, y = h2a_indx,
    color = trend_lab, linetype = trend_lab
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
  filename = paste0(folder_output, "fig_ts_aewr_cz_p25_dd_indexed_h2a.png"),
  plot_dd_indexed_h2a,
  width = 7, height = 5,
  device = "png"
)

## Exhibit 10: DD Main Results -------------------------------------------------

samp_base <- county_df %>%
  filter(any_cropland_2007 == 1, county_simple_treatment_groups != "always takers")

samp_no_border <- samp_base %>% filter(border_cz == 0)

# DD model 1: no controls, all CZs
dd_1 <- feols(
  h2a_cert_share_farm_workers_2011_start_year ~
    aewr_cz_p25_l1 * postdummy |
    county_fe + year_fe,
  data = samp_base,
  vcov = ~cz_aewr_region_fe
)

# DD model 2: with controls, all CZs
dd_2 <- feols(
  h2a_cert_share_farm_workers_2011_start_year ~
    aewr_cz_p25_l1 * postdummy +
    ln_pop_census +
    emp_pop_ratio |
    county_fe + year_fe,
  data = samp_base,
  vcov = ~cz_aewr_region_fe
)

# DD model 3: no controls, no border CZs
dd_3 <- feols(
  h2a_cert_share_farm_workers_2011_start_year ~
    aewr_cz_p25_l1 * postdummy |
    county_fe + year_fe,
  data = samp_no_border,
  vcov = ~cz_aewr_region_fe
)

# DD model 4: with controls, no border CZs
dd_4 <- feols(
  h2a_cert_share_farm_workers_2011_start_year ~
    aewr_cz_p25_l1 * postdummy +
    ln_pop_census +
    emp_pop_ratio |
    county_fe + year_fe,
  data = samp_no_border,
  vcov = ~cz_aewr_region_fe
)

table_1 <- etable(
  dd_1, dd_2, dd_3, dd_4,
  tex = TRUE,
  title = "The Effect of the AEWR Wage Premium on H-2A Utilization",
  headers = c("No Controls", "Controls", "No Border, No Controls", "No Border, Controls"),
  dict = c(
    "h2a_cert_share_farm_workers_2011_start_year" = "Normalized H-2A program usage",
    "aewr_cz_p25_l1"  = "Lagged AEWR vs 25th pct wage gap",
    "postdummy"        = "Post",
    "ln_pop_census"    = "Log population",
    "emp_pop_ratio"    = "Employment-to-pop ratio"
  ),
  signif.code = c("***" = 0.01, "**" = 0.05, "*" = 0.10),
  file = paste0(folder_output, "table_1_main_results.tex"),
  replace = TRUE
)

summary(dd_1)
summary(dd_2)
summary(dd_3)
summary(dd_4)

## Exhibit 11: Event Study (Flexible DD, Base Year = 2011) --------------------

es_rhs <- paste(
  "aewr_cz_p25_l1 * yeardummy_2008",
  "aewr_cz_p25_l1 * yeardummy_2009",
  "aewr_cz_p25_l1 * yeardummy_2010",
  "aewr_cz_p25_l1 * yeardummy_2012",
  "aewr_cz_p25_l1 * yeardummy_2013",
  "aewr_cz_p25_l1 * yeardummy_2014",
  "aewr_cz_p25_l1 * yeardummy_2015",
  "aewr_cz_p25_l1 * yeardummy_2016",
  "aewr_cz_p25_l1 * yeardummy_2017",
  "aewr_cz_p25_l1 * yeardummy_2018",
  "aewr_cz_p25_l1 * yeardummy_2019",
  "aewr_cz_p25_l1 * yeardummy_2020",
  "aewr_cz_p25_l1 * yeardummy_2021",
  "aewr_cz_p25_l1 * yeardummy_2022",
  sep = " + "
)

es_fml <- as.formula(paste(
  "h2a_cert_share_farm_workers_2011_start_year ~",
  es_rhs,
  "| county_fe + year_fe"
))

es_fml_ctrl <- as.formula(paste(
  "h2a_cert_share_farm_workers_2011_start_year ~",
  es_rhs,
  "+ ln_pop_census + emp_pop_ratio | county_fe + year_fe"
))

# Event study model 1: no controls, all CZs
es_1 <- feols(es_fml, data = samp_base,      vcov = ~cz_aewr_region_fe)

# Event study model 2: with controls, all CZs
es_2 <- feols(es_fml_ctrl, data = samp_base, vcov = ~cz_aewr_region_fe)

# Event study model 3: no controls, no border CZs
es_3 <- feols(es_fml, data = samp_no_border,      vcov = ~cz_aewr_region_fe)

# Event study model 4: with controls, no border CZs
es_4 <- feols(es_fml_ctrl, data = samp_no_border, vcov = ~cz_aewr_region_fe)

# Wald test on post-2011 interactions (joint significance)
post_terms <- grep("aewr_cz_p25_l1:yeardummy_201[2-9]|aewr_cz_p25_l1:yeardummy_202",
                   names(coef(es_1)), value = TRUE)
wald(es_1, keep = post_terms)
wald(es_2, keep = post_terms)

es_dict <- c(
  "aewr_cz_p25_l1"    = "Lagged AEWR vs 25th pct wage gap",
  "yeardummy_2008"    = "2008",
  "yeardummy_2009"    = "2009",
  "yeardummy_2010"    = "2010",
  "yeardummy_2012"    = "2012",
  "yeardummy_2013"    = "2013",
  "yeardummy_2014"    = "2014",
  "yeardummy_2015"    = "2015",
  "yeardummy_2016"    = "2016",
  "yeardummy_2017"    = "2017",
  "yeardummy_2018"    = "2018",
  "yeardummy_2019"    = "2019",
  "yeardummy_2020"    = "2020",
  "yeardummy_2021"    = "2021",
  "yeardummy_2022"    = "2022"
)

table_2 <- etable(
  es_1, es_2, es_3, es_4,
  tex = TRUE,
  title = "Event Study Coefficients (Base Year = 2011)",
  keep = "%aewr_cz_p25_l1:yeardummy",
  headers = c("No Controls", "Controls", "No Border, No Controls", "No Border, Controls"),
  dict = es_dict,
  signif.code = c("***" = 0.01, "**" = 0.05, "*" = 0.10),
  file = paste0(folder_output, "table_2_event_study.tex"),
  replace = TRUE
)

## Exhibit 12: Event Study Coefficient Plots -----------------------------------

make_coefplot <- function(model) {
  coef_names <- grep("aewr_cz_p25_l1:yeardummy_", names(coef(model)), value = TRUE)
  ct <- model$coeftable[coef_names, , drop = FALSE]

  years_in_model <- as.integer(sub(".*yeardummy_", "", rownames(ct)))

  all_years <- 2008:2022
  coeff_df <- data.frame(year = all_years) %>%
    mutate(
      beta = ifelse(year == 2011, 0, ct[match(paste0("aewr_cz_p25_l1:yeardummy_", year), rownames(ct)), 1]),
      se   = ifelse(year == 2011, 0, ct[match(paste0("aewr_cz_p25_l1:yeardummy_", year), rownames(ct)), 2])
    ) %>%
    mutate(
      upper_ci = beta + 1.96 * se,
      lower_ci = beta - 1.96 * se
    )

  ggplot(coeff_df, aes(x = year)) +
    geom_hline(yintercept = 0, color = "grey40") +
    geom_vline(xintercept = 2011, linetype = "dashed", color = "grey40") +
    geom_ribbon(aes(ymin = lower_ci, ymax = upper_ci), alpha = 0.2, fill = "steelblue") +
    geom_line(aes(y = beta), color = "steelblue", lwd = 1.2) +
    geom_point(aes(y = beta), color = "steelblue", size = 2) +
    labs(
      x = "Year",
      y = "Coefficient on Lagged AEWR vs 25th pct wage gap"
    ) +
    theme_clean()
}

p_es_1 <- make_coefplot(es_1)
p_es_2 <- make_coefplot(es_2)
p_es_3 <- make_coefplot(es_3)
p_es_4 <- make_coefplot(es_4)

ggsave(p_es_1,
  filename = paste0(folder_output, "coefplot_dd_no_controls.png"),
  width = 8, height = 5, device = "png")

ggsave(p_es_2,
  filename = paste0(folder_output, "coefplot_dd_controls.png"),
  width = 8, height = 5, device = "png")

ggsave(p_es_3,
  filename = paste0(folder_output, "coefplot_dd_no_border_no_controls.png"),
  width = 8, height = 5, device = "png")

ggsave(p_es_4,
  filename = paste0(folder_output, "coefplot_dd_no_border_controls.png"),
  width = 8, height = 5, device = "png")

## Exhibit 13: Summary Statistics Table -----------------------------------------

sumstats_vars <- list(
  "H-2A share of 2011 farm employment" = "h2a_cert_share_farm_workers_2011_start_year",
  "H-2A certified workers (start year)" = "nbr_workers_certified_start_year",
  "Farm employment 2011 (baseline)"     = "emp_farm_2011",
  "AEWR p25 bite (2012 \$)"             = "aewr_cz_p25",
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
