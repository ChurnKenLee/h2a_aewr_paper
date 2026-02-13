## H2A: Analysis Figures and Tables
## Phil Hoxie
## 1/31/24

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

date <- paste0(substr(Sys.Date(), 1, 4), substr(Sys.Date(), 6, 7),  substr(Sys.Date(), 9, 10))

## Load and Clean County Shape Files -------------------------------------------

cnt_shp <- st_read(paste0(folder_data, "tl_2020_us_county.shp"))

fips_codes <- read_csv(paste0(folder_data, "fips_codes.csv"))

aewr_df <- read_parquet(paste0(folder_data, "aewr.parquet"))
  
border_df <- read_parquet(paste0(folder_data, "border_df_analysis.parquet"))
aewr_data <- read_parquet(paste0(folder_data, "aewr_data.parquet"))
h2a_data_ts <- read_parquet(paste0(folder_data, "h2a_data_ts.parquet"))
h2a_data <- read_parquet(paste0(folder_data, "h2a_data.parquet"))

border_df_yearly <- read_parquet(paste0(folder_data, "border_df_analysis_year.parquet"))
aewr_data_full <- read_parquet(paste0(folder_data, "aewr_data_full.parquet"))

## County DDD Design -----------------------------------------------------------

county_df <- read_parquet(paste0(folder_data, "county_df_analysis_year.parquet"))

names(county_df)

# map 

county_map <- cnt_shp %>% 
  mutate(statefip = as.numeric(STATEFP))

county_map <- county_map %>% 
  filter(statefip <= 56 & statefip != 2 & statefip != 15)

# simplify the map
county_map <- st_simplify(county_map, preserveTopology = FALSE, dTolerance = 1000)

ggplot(county_map) +
  geom_sf()

county_map$countyfips <- as.numeric(str_c(county_map$STATEFP, county_map$COUNTYFP))


## Exhibit 1: AEWR TS Real ---------------------------------------------------


aewr_data_full_ts <- aewr_data_full %>% 
  group_by(year) %>% 
  summarise(aewr = mean(aewr, na.rm = T), 
            aewr_ppi = mean(aewr_ppi, na.rm = T))

aewr_ts <- ggplot(data = aewr_data_full_ts, aes(x = year, y = aewr_ppi))+
  geom_line(linewidth = 1.25, color = "#2166ac")+
  theme_clean()+
  xlab("Year")+
  ylab("Real AEWR")+
  geom_vline(xintercept = 2011, linetype = "dashed")
aewr_ts
ggsave(filename = paste0(folder_output, "ts_national_aewr_real.png"), aewr_ts, device = "png")

## Exhibit 2: AEWR TS Nominal --------------------------------------------------

aewr_ts_nom <- ggplot(data = aewr_data_full_ts, aes(x = year, y = aewr))+
  geom_line(linewidth = 1.25, color = "#2166ac")+
  theme_clean()+
  xlab("Year")+
  ylab("Nominal AEWR")+
  geom_vline(xintercept = 2011, linetype = "dashed")
aewr_ts_nom
ggsave(filename = paste0(folder_output, "ts_national_aewr_nominal.png"), aewr_ts, device = "png")


## Exhibit 3: H2A Use TS ---------------------------------------------------------

# levels 


h2a_ts <- ggplot(data = subset(h2a_data_ts, case_year > 2007 & case_year <= 2022), aes(x = case_year))+
  geom_line(aes(y = h2a_nbr_workers_certified / 1000), color = "#4393c3", linewidth = 1.5)+
  theme_clean()+
  xlab("Year")+
  ylab("H-2A Workers Certified (Thousands)")+
  scale_y_continuous(breaks = c(0, 50, 100, 150, 200, 250, 300, 350, 400), 
                     limits = c(0, 400))
h2a_ts

# indexed 

base_req <- h2a_data_ts$h2a_man_hours_requested[ h2a_data_ts$case_year == 2011]
base_workers <- h2a_data_ts$h2a_nbr_workers_certified[ h2a_data_ts$case_year == 2011]
base_hours <- h2a_data_ts$h2a_man_hours_certified[ h2a_data_ts$case_year == 2011]
base_aps <- h2a_data_ts$n_applications[ h2a_data_ts$case_year == 2011]

h2a_data_ts_index <- h2a_data_ts %>% 
  mutate(indx_workers_certified = (h2a_nbr_workers_certified / base_workers[1]) * 100,
         indx_workers_req = (h2a_man_hours_requested / base_req[1]) * 100,
         indx_man_hours_certified = (h2a_man_hours_certified / base_hours[1]) * 100,
         indx_applications = (n_applications / base_aps[1]) * 100)

h2a_data_ts_index <- h2a_data_ts_index %>% 
  select(case_year, indx_workers_certified, indx_man_hours_certified, indx_applications, indx_workers_req)

h2a_data_ts_index <- h2a_data_ts_index %>% 
  pivot_longer(cols = starts_with("indx_"), names_to = "series", names_prefix = "indx_", values_to = "indx", values_drop_na = F)
head(h2a_data_ts_index)


# zeros should not be graphed # 

h2a_data_ts_index <- h2a_data_ts_index %>% 
  transform(indx = ifelse(indx < 50, NA, indx))

h2a_indx_ts <- ggplot(data = subset(h2a_data_ts_index, case_year > 2007 & case_year <= 2022), aes(x = case_year, y = indx, group = series, color = series, linetype = series))+
  geom_line(linewidth = 1.25, alpha = 0.75)+
  theme_clean()+
  scale_color_manual(values = c("#2166ac", "#4393c3", "#F198A2", "#b2182b"), labels = c("Applications", "Man Hours Certified", "Workers Certified" , "Workers Requested"))+
  scale_linetype_manual(values = c(5, 4, 3, 1), labels = c("Applications", "Man Hours Certified", "Workers Certified", "Workers Requested"))+
  theme(legend.title = element_blank(), legend.position = "bottom")+
  xlab("Year")+
  ylab("Indexed Values (2011 = 100)")+
  geom_hline(yintercept = 100, linetype = "dashed")+
  guides(color = guide_legend(nrow = 2),
         linetype = guide_legend(nrow = 2))
h2a_indx_ts

ggsave(filename = paste0(folder_output, "fig_line_ts_h2a_workers_certified.png"), h2a_ts, device = "png")
ggsave(filename = paste0(folder_output, "fig_line_ts_h2a_indexes.png"), h2a_indx_ts, device = "png")

## Exhibit 4: H2A Map: 2012 use ----------------------------------------------

head(county_map)

h2a_map_data <- NULL

h2a_map_data_2012 <- h2a_data %>% 
  filter(census_period == 2012) %>% 
  mutate(ln_h2a_workers_cert_2012 = log(nbr_workers_certified_start_year)) %>% 
  select(countyfips, ln_h2a_workers_cert_2012, nbr_workers_certified_start_year) %>% 
  rename(nbr_workers_certified_2012 = nbr_workers_certified_start_year)

h2a_map_data_2017 <- h2a_data %>% 
  filter(census_period == 2017) %>% 
  mutate(ln_h2a_workers_cert_2017 = log(nbr_workers_certified_start_year)) %>% 
  select(countyfips, ln_h2a_workers_cert_2017, nbr_workers_certified_start_year) %>% 
  rename(nbr_workers_certified_2017 = nbr_workers_certified_start_year)

h2a_map_data_2022 <- h2a_data %>% 
  filter(census_period == 2022) %>% 
  mutate(ln_h2a_workers_cert_2022 = log(nbr_workers_certified_start_year)) %>% 
  select(countyfips, ln_h2a_workers_cert_2022, nbr_workers_certified_start_year) %>% 
  rename(nbr_workers_certified_2022 = nbr_workers_certified_start_year)


h2a_map_data <- merge(x = h2a_map_data_2012, y = h2a_map_data_2017, by = "countyfips")
h2a_map_data <- merge(x = h2a_map_data, y = h2a_map_data_2022, by = "countyfips")

county_map_fips <- county_map

h2a_map_data <- merge(x = county_map_fips, y = h2a_map_data, by = "countyfips", all.x = T, all.y = F)

# 2012

h2a_map_data_2012 <- ggplot(h2a_map_data) +
  geom_sf(aes(fill = ln_h2a_workers_cert_2012), color=alpha("grey",0.5))+
  theme(panel.grid.major = element_line(color = gray(0.5), linetype = "dashed", 
                                        size = 0.5), panel.background = element_rect(fill = "aliceblue"))+
  theme_bw()+
  annotation_north_arrow(location = "bl", which_north = "true", 
                         pad_x = unit(0.05, "in"), pad_y = unit(0.25, "in"),
                         style = north_arrow_fancy_orienteering)+
  annotation_scale(location = "bl", width_hint = 0.4) +
  scale_fill_gradient2(
    low = "white",
    mid = muted("#C3DBF4"),
    midpoint = 5, 
    high = muted("#0B243C"),
    name = "H-2A Workers\nCertified (log)"
  )
h2a_map_data_2012

ggsave(filename = paste0(folder_output, "map_H2A_workers_2012.png"), h2a_map_data_2012, device = "png")

# no logs 

h2a_map_data$nbr_workers_certified_2012[is.na(h2a_map_data$nbr_workers_certified_2012)] <- 0

h2a_map_data_2012_nolog <- ggplot(h2a_map_data) +
  geom_sf(aes(fill = nbr_workers_certified_2012), color=alpha("grey",0.5))+
  theme(panel.grid.major = element_line(color = gray(0.5), linetype = "dashed", 
                                        size = 0.5), panel.background = element_rect(fill = "aliceblue"))+
  theme_bw()+
  annotation_north_arrow(location = "bl", which_north = "true", 
                         pad_x = unit(0.05, "in"), pad_y = unit(0.25, "in"),
                         style = north_arrow_fancy_orienteering)+
  annotation_scale(location = "bl", width_hint = 0.4) +
  scale_fill_gradient2(
    low = "white",
    mid = muted("#C3DBF4"),
    midpoint = 3000, 
    high = muted("#0B243C"),
    name = "H-2A Workers\nCertified (Levels)"
  )
h2a_map_data_2012_nolog

ggsave(filename = paste0(folder_output, "map_H2A_workers_2012_nolog.png"), h2a_map_data_2012, device = "png")


## Exhibit 5: H2A Map: Change from 2012 use ----------------------------------

h2a_change_data <- h2a_data %>% 
  filter(census_period == 2012 | census_period == 2022) %>% 
  select(countyfips, census_period, nbr_workers_certified_start_year) %>% 
  pivot_wider(id_cols = countyfips, names_from = census_period, values_from = nbr_workers_certified_start_year)

h2a_change_data[is.na(h2a_change_data)] <- 0 # NAs are zeros here

h2a_change_data <- h2a_change_data %>% # make change vars
  mutate(ch_workers = `2022` - `2012`, 
         pch_workers = (`2022` - `2012`)/`2012`,
         new_h2a = ifelse(`2012` == 0 & `2022` != 0, `2022`, NA))

h2a_change_map_data <- merge(x = county_map, y = h2a_change_data, by = "countyfips", all.x = T, all.y = F)

# change from 2012

h2a_map_data_ch2012 <- ggplot(h2a_change_map_data) +
  geom_sf(aes(fill = log(ch_workers)), color=alpha("grey",0.5))+
  theme(panel.grid.major = element_line(color = gray(0.5), linetype = "dashed", 
                                        size = 0.5), panel.background = element_rect(fill = "aliceblue"))+
  theme_bw()+
  annotation_north_arrow(location = "bl", which_north = "true", 
                         pad_x = unit(0.05, "in"), pad_y = unit(0.25, "in"),
                         style = north_arrow_fancy_orienteering)+
  annotation_scale(location = "bl", width_hint = 0.4) +
  scale_fill_gradient2(
    low = "white",
    mid = muted("#C3DBF4"),
    midpoint = 9,
    high = muted("#0B243C"),
    name = "Log Change in\nH2-A Workers\nCertified"
  )
h2a_map_data_ch2012

ggsave(filename = paste0(folder_output, "map_H2A_workers_ch2022_2012.png"), h2a_map_data_ch2012, device = "png")

# New from 2012

h2a_map_data_ch2012_new <- ggplot(h2a_change_map_data) +
  geom_sf(aes(fill = log(new_h2a)), color=alpha("grey",0.5))+
  theme(panel.grid.major = element_line(color = gray(0.5), linetype = "dashed", 
                                        size = 0.5), panel.background = element_rect(fill = "aliceblue"))+
  theme_bw()+
  annotation_north_arrow(location = "bl", which_north = "true", 
                         pad_x = unit(0.05, "in"), pad_y = unit(0.25, "in"),
                         style = north_arrow_fancy_orienteering)+
  annotation_scale(location = "bl", width_hint = 0.4) +
  scale_fill_gradient2(
    low = "white",
    mid = muted("#C3DBF4"),
    midpoint = 7,
    high = muted("#0B243C"),
    name = "Log H2-A\nWorkers\nCertified in\nnew counties\nsince 2012"
  )
h2a_map_data_ch2012_new

ggsave(filename = paste0(folder_output, "map_H2A_workers_ch2022_2012_newcounties.png"), h2a_map_data_ch2012_new, device = "png")

## Exhibit 6: AEWR Map: AEWR Difference from National Trend ------------------

ts_ddd <- county_df %>% 
  mutate(ch_aewr_ppi = aewr_ppi - aewr_ppi_l1,
         pch_aewr_ppi = (aewr_ppi - aewr_ppi_l1) / aewr_ppi_l1)

ggplot(ts_ddd, aes(x = pch_aewr_ppi))+
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

aewr_reg_ts_data <- merge(x = aewr_reg_ts_data, y = aewr_base, by = "aewr_region_num", all = T)

aewr_reg_ts_data <- aewr_reg_ts_data %>% 
  mutate(aewr_ppi_chbase = (aewr_state_ag_ppi - aewr_ppi_base) / aewr_ppi_base)

aewr_reg_ts_avg <- aewr_reg_ts_data %>% 
  group_by(year) %>% 
  summarise(aewr_ppi_chbase_avg = mean(aewr_ppi_chbase))

aewr_reg_ts_data <- merge(x = aewr_reg_ts_data, y = aewr_reg_ts_avg, by = "year", all = T)

aewr_reg_ts_data <- aewr_reg_ts_data %>% 
  mutate(aewr_ppi_chbase_detrend = (aewr_ppi_chbase - aewr_ppi_chbase_avg))

aewr_reg_ts_data_color <- aewr_reg_ts_data %>% 
  filter(year == 2022 | year == 2008) %>% 
  select(aewr_region_num, aewr_ppi_chbase_detrend, year) %>% 
  pivot_wider(id_cols = aewr_region_num, names_from = "year", values_from = "aewr_ppi_chbase_detrend")

aewr_reg_ts_data_color <- aewr_reg_ts_data_color %>% 
  mutate(aewr_high_growth = `2022` - `2008`) %>% 
  select(aewr_region_num, aewr_high_growth)

aewr_reg_ts_data <- merge(x = aewr_reg_ts_data, y = aewr_reg_ts_data_color, by = "aewr_region_num")

# need to make an AEWR region to stat xwalk 

aewr_state_xwalk <- unique(county_df %>% 
  select(aewr_region_num, statefips)) %>% 
  rename(statefip = statefips)

county_map_aewr <- merge(x = county_map, y = aewr_state_xwalk, by = "statefip", all.x = T, all.y = F)

county_map_aewr <- merge(x = county_map_aewr, y = aewr_reg_ts_data, by = "aewr_region_num", all.x = T, all.y = F)

aewr_map_data_growth <- ggplot(county_map_aewr) +
  geom_sf(aes(fill = aewr_high_growth), color=alpha("grey",0.5))+
  theme(panel.grid.major = element_line(color = gray(0.5), linetype = "dashed", 
                                        size = 0.5), panel.background = element_rect(fill = "aliceblue"))+
  theme_bw()+
  annotation_north_arrow(location = "bl", which_north = "true", 
                         pad_x = unit(0.05, "in"), pad_y = unit(0.25, "in"),
                         style = north_arrow_fancy_orienteering)+
  annotation_scale(location = "bl", width_hint = 0.4) +
  scale_fill_gradient2(
    low = muted("#2166ac"),
    mid = "white",
    midpoint = 0, 
    high = muted("#b2182b"),
    name = "Change in\nAEWR Relative\nTo National\nTrend from\n2008-2022"
  )
aewr_map_data_growth


ggsave(filename = paste0(folder_output, "map_aewr_change_from_trend_2022_2008.png"), aewr_map_data_growth, device = "png")

gc()
## Exhibit 7: Exposure Map: Counties by Sample Classification ----------------

## DDD Sample Map 

head(county_df)

ddd_map_data <- county_df %>% 
  dplyr::select(countyfips, high_h2a_share_75, high_h2a_share_50, any_cropland_2007)

ddd_map <- merge(x = county_map, y = ddd_map_data, by = "countyfips", all.x = T, all.y = F)

ddd_map <- ddd_map %>% 
  transform(high_h2a_share_75 = ifelse(any_cropland_2007 == 1, high_h2a_share_75, NA),
            high_h2a_share_50 = ifelse(high_h2a_share_50 == 1, high_h2a_share_50, NA))

ddd_samp_map <- ggplot(ddd_map) +
  geom_sf(aes(fill = as.factor(high_h2a_share_75)))+
  theme(panel.grid.major = element_line(color = gray(0.5), linetype = "dashed", 
                                        size = 0.5), panel.background = element_rect(fill = "aliceblue"))+
  theme_bw()+
  scale_fill_manual(values = c("#f4a582", "#4393c3" ),
                    na.value = "#E5E4E2", name = "Top Quartile\nin H2A Use\nPre-2012")+
  annotation_north_arrow(location = "bl", which_north = "true", 
                         pad_x = unit(0.05, "in"), pad_y = unit(0.25, "in"),
                         style = north_arrow_fancy_orienteering)+
  annotation_scale(location = "bl", width_hint = 0.4)
ddd_samp_map

ggsave(filename = paste0(folder_output, "map_aewr_ddd_sample_dropnocropland.png"), ddd_samp_map, device = "png")


## Exhibit 8: AEWR TS Difference from Trend by Region ------------------------

# input data is made with the map above 

plot_aewr_reg_ts <- ggplot(aewr_reg_ts_data, aes(x = year, y = aewr_ppi_chbase_detrend, group = as.factor(aewr_region_num), 
                                                 color = aewr_high_growth))+
  geom_line()+
  theme_clean()+
  scale_color_gradient2(
    low = muted("#2166ac"),
    mid = "white",
    midpoint = 0, 
    high = muted("#b2182b"),
    name = "AEWR Deviation from Trend"
  )+
  theme(legend.position = "none")+
  xlab("Year")+
  ylab("AEWR Difference from National Trend")+
  geom_hline(yintercept = 0, linetype = "dashed", color = "grey")+
  scale_y_continuous(breaks = c(-1, -.5, 0, .5, 1), labels = c("-100%", "-50%", "0%", "+50%", "+100%"), limits = c(-1, 1))
plot_aewr_reg_ts

ggsave(filename = paste0(folder_output, "fig_ts_aewr_region_change_from_trend.png"), plot_aewr_reg_ts, device = "png")

## Exhibit 9: DDD by Graph a la Jeff -----------------------------------------


# need to make the ts graph using the trend growth as a dummy 

head(aewr_reg_ts_data)

aewr_reg_ts_data <- aewr_reg_ts_data %>% 
  mutate(aewr_high_growth_p50 = ifelse(aewr_high_growth >= median(aewr_reg_ts_data_color$aewr_high_growth), 1, 0),
         aewr_high_growth_positive = ifelse(aewr_high_growth > 0, 1, 0))

# need: nbr_workers_requested_start_year, high / low dummy, growth dummy, year  

names(county_df)

aewr_reg_ts_h2a <- county_df %>% 
  select(year, aewr_region_num, nbr_workers_certified_start_year, high_h2a_share_75, high_h2a_share_66, high_h2a_share_50)

aewr_reg_ts_data <- merge(x = aewr_reg_ts_h2a, y = aewr_reg_ts_data, by = c("aewr_region_num", "year"), all = T)

# collapse 

aewr_reg_ts_data_collapse <- aewr_reg_ts_data %>% 
  group_by(high_h2a_share_75, aewr_high_growth_p50, year) %>% 
  summarise(nbr_workers_certified_start_year = sum(nbr_workers_certified_start_year, na.rm = T))

# index to 2012 

aewr_reg_ts_data_base <- aewr_reg_ts_data_collapse %>% 
  filter(year == 2011)

aewr_reg_ts_data_base <- aewr_reg_ts_data_base %>% 
  rename(nbr_workers_certified_base = nbr_workers_certified_start_year) %>% 
  select(-year)

aewr_reg_ts_data_collapse <- merge(x = aewr_reg_ts_data_collapse, y = aewr_reg_ts_data_base, 
                                   by = c("high_h2a_share_75", "aewr_high_growth_p50"))

aewr_reg_ts_data_collapse <- aewr_reg_ts_data_collapse %>% 
  mutate(nbr_workers_certified_indx2011 = nbr_workers_certified_start_year / nbr_workers_certified_base)

aewr_reg_ts_data_collapse <- aewr_reg_ts_data_collapse %>% 
  mutate(group_lab = ifelse(high_h2a_share_75 == 1 & aewr_high_growth_p50 == 1, "High AEWR Growth, High Exposure",
                            ifelse(high_h2a_share_75 == 1 & aewr_high_growth_p50 == 0, "Low AEWR Growth, High Exposure",
                                   ifelse(high_h2a_share_75 == 0 & aewr_high_growth_p50 == 1, "High AEWR Growth, Low Exposure",
                                          "Low AEWR Growth, Low Exposure"))))

plot_aewr_use_ts_DDD <- ggplot(aewr_reg_ts_data_collapse, aes(x = year, y = nbr_workers_certified_indx2011, 
                                                              linetype = as.factor(group_lab), 
                                                              color = as.factor(group_lab)))+
  geom_line()+
  theme_classic()+
  scale_color_manual(values = c("#b2182b", "#b2182b","#2166ac", "#2166ac"))+
  scale_linetype_manual(values = c(1,5,3,4))+
  labs(color = "group_lab", linetype = "group_lab") +
  guides(color = guide_legend(ncol = 2, byrow = TRUE),
         linetype = guide_legend(ncol = 2, byrow = TRUE),) +
  theme(legend.position = "bottom")+
  theme(legend.title = element_blank())+
  geom_hline(yintercept = 1, alpha = 0.5)+
  ylab("Number of H2-A Workers Requested\n(Indexed to 2011)")
plot_aewr_use_ts_DDD

ggsave(plot_aewr_use_ts_DDD, filename = paste0(folder_output, "fig_ts_aewr_growth_exposure_DDD.png"))

head(aewr_reg_ts_data_collapse)

## Exhibit 10: DDD (simple) in parts for H2A use only -------------------------
# + ln_pop_census + emp_pop_ratio
fixest_model_dd_simple <- feols(h2a_cert_share_farm_workers_2011_start_year ~ aewr_state_ag_ppi_l1 * postdummy  
                                + ln_pop_census + emp_pop_ratio | county_fe + year_fe, 
                                 data = subset(county_df, any_cropland_2007 == 1), vcov = ~statefips)

fixest_model_dd_simple$coeftable # matches stata
fixest_model_dd_simple$nobs

fixest_model_ddd_simple <- feols(h2a_cert_share_farm_workers_2011_start_year ~ aewr_state_ag_ppi_l1 * high_h2a_share_75 * postdummy  
                                 + ln_pop_census + emp_pop_ratio | county_fe + year_fe, 
                                 data = subset(county_df, any_cropland_2007 == 1 & year >= 2011), vcov = ~statefips)

fixest_model_ddd_simple$coeftable # matches stata
fixest_model_ddd_simple$nobs

fixest_model_ddd_event <- feols(h2a_cert_share_farm_workers_2011_start_year ~ 
                                  aewr_state_ag_ppi_l1 * yeardummy_2008 * high_h2a_share_75 + 
                                  aewr_state_ag_ppi_l1 * yeardummy_2009 * high_h2a_share_75 + 
                                  aewr_state_ag_ppi_l1 * yeardummy_2010 * high_h2a_share_75 + 
                                  aewr_state_ag_ppi_l1 * yeardummy_2012 * high_h2a_share_75 + 
                                  aewr_state_ag_ppi_l1 * yeardummy_2013 * high_h2a_share_75 +
                                  aewr_state_ag_ppi_l1 * yeardummy_2014 * high_h2a_share_75 + 
                                  aewr_state_ag_ppi_l1 * yeardummy_2015 * high_h2a_share_75 +
                                  aewr_state_ag_ppi_l1 * yeardummy_2016 * high_h2a_share_75 +
                                  aewr_state_ag_ppi_l1 * yeardummy_2017 * high_h2a_share_75 +
                                  aewr_state_ag_ppi_l1 * yeardummy_2018 * high_h2a_share_75 +
                                  aewr_state_ag_ppi_l1 * yeardummy_2019 * high_h2a_share_75 +
                                  aewr_state_ag_ppi_l1 * yeardummy_2020 * high_h2a_share_75 +
                                  aewr_state_ag_ppi_l1 * yeardummy_2021 * high_h2a_share_75 +
                                  aewr_state_ag_ppi_l1 * yeardummy_2022 * high_h2a_share_75 + 
                                   ln_pop_census + emp_pop_ratio | county_fe + year_fe, 
                                 data = subset(county_df, any_cropland_2007 == 1), vcov = ~statefips)

fixest_model_ddd_event$coeftable # matches stata
fixest_model_ddd_event$nobs

wald(fixest_model_ddd_event, keep = c("aewr_state_ag_ppi_l1:high_h2a_share_75:yeardummy_2012",
                                        "aewr_state_ag_ppi_l1:high_h2a_share_75:yeardummy_2013",
                                       "aewr_state_ag_ppi_l1:high_h2a_share_75:yeardummy_2014",
                                       "aewr_state_ag_ppi_l1:high_h2a_share_75:yeardummy_2015",
                                       "aewr_state_ag_ppi_l1:high_h2a_share_75:yeardummy_2016",
                                       "aewr_state_ag_ppi_l1:high_h2a_share_75:yeardummy_2017",
                                       "aewr_state_ag_ppi_l1:high_h2a_share_75:yeardummy_2018",
                                       "aewr_state_ag_ppi_l1:high_h2a_share_75:yeardummy_2019",
                                       "aewr_state_ag_ppi_l1:high_h2a_share_75:yeardummy_2020",
                                       "aewr_state_ag_ppi_l1:high_h2a_share_75:yeardummy_2021",
                                       "aewr_state_ag_ppi_l1:high_h2a_share_75:yeardummy_2022"), vcov = ~statefips)

# graph it 

coeff_df <- data.frame(year = c(2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022),
                       beta = c(fixest_model_ddd_event$coeftable[33, 1], 
                                fixest_model_ddd_event$coeftable[34, 1], 
                                fixest_model_ddd_event$coeftable[35, 1],
                                0,
                                fixest_model_ddd_event$coeftable[36, 1],
                                fixest_model_ddd_event$coeftable[37, 1],
                                fixest_model_ddd_event$coeftable[38, 1],
                                fixest_model_ddd_event$coeftable[39, 1],
                                fixest_model_ddd_event$coeftable[40, 1],
                                fixest_model_ddd_event$coeftable[41, 1],
                                fixest_model_ddd_event$coeftable[42, 1],
                                fixest_model_ddd_event$coeftable[43, 1],
                                fixest_model_ddd_event$coeftable[44, 1],
                                fixest_model_ddd_event$coeftable[45, 1],
                                fixest_model_ddd_event$coeftable[46, 1]),
                       se = c(fixest_model_ddd_event$coeftable[33, 2], 
                              fixest_model_ddd_event$coeftable[34, 2], 
                              fixest_model_ddd_event$coeftable[35, 2], 
                              0,
                              fixest_model_ddd_event$coeftable[36, 2],
                              fixest_model_ddd_event$coeftable[37, 2],
                              fixest_model_ddd_event$coeftable[38, 2],
                              fixest_model_ddd_event$coeftable[39, 2],
                              fixest_model_ddd_event$coeftable[40, 2],
                              fixest_model_ddd_event$coeftable[41, 2],
                              fixest_model_ddd_event$coeftable[42, 2],
                              fixest_model_ddd_event$coeftable[43, 2],
                              fixest_model_ddd_event$coeftable[44, 2],
                              fixest_model_ddd_event$coeftable[45, 2],
                              fixest_model_ddd_event$coeftable[46, 2])) 


coeff_df <- coeff_df %>% 
  mutate(upper_ci = beta + se * 1.96,
         lower_ci = beta - se * 1.96)

coefplot <- ggplot(data = coeff_df, aes(x = year))+
  geom_line(aes(y = beta), lwd = 1.5)+
  geom_line(aes(y = upper_ci), linetype = "dashed")+
  geom_line(aes(y = lower_ci), linetype = "dashed")+
  geom_vline(xintercept = 2011)+
  geom_hline(yintercept = 0)+
  theme_classic()
coefplot

ggsave(coefplot, filename = paste0(folder_output, "coefplot_ddd_h2a_req_share_farm_workers_2011_start_year_high_h2a_share_75_controls.png"), device = "png")

# with ggiplot 

fixest_model_ddd_full_i <- feols(
  h2a_cert_share_farm_workers_2011_start_year ~ 
    i(year, high_h2a_share_75, ref = 2011) +
    i(year, aewr_state_ag_ppi_l1, ref = 2011) +
    i(high_h2a_share_75, aewr_state_ag_ppi_l1, ref = 0)+
    i(year, I(high_h2a_share_75 * aewr_state_ag_ppi_l1), ref = 2011) + 
    ln_pop_census + emp_pop_ratio + aewr_state_ag_ppi_l1 | county_fe + year_fe, 
  data = subset(county_df, any_cropland_2007 == 1), 
  vcov = ~statefips
)
summary(fixest_model_ddd_full_i)
# Plot only the triple interaction coefficients
ddd_plot <- ggiplot(fixest_model_ddd_full_i, i.select = 4, geom_style = "ribbon")+
  xlab("")+
  theme(plot.title = element_blank())
ddd_plot

ggsave(coefplot, filename = paste0(folder_output, "coefplot_ddd_h2a_req_share_farm_workers_2011_start_year_high_h2a_share_75_controls_gg.png"), device = "png")

## Exhibit 11: Event studies for H2A use, wages, employment, prices ---------

## Exhibit 12: Pre-Trend Tests for H2A use, wages, emp, prices ---------------

## Exhibit 13: DDD tables for H2A use, wages, emp, prices --------------------

## Exhibit 14: Summary Statistics --------------------------------------------

