## H2A: Analysis Figures and Tables
## Phil Hoxie
## 1/31/24

library(sf)
library(tidyverse)
library(USAboundaries)
library(ggspatial)
library(scales)
library(cowplot)
library(ggthemes)
library(fixest)

Sys.sleep(1)

set.seed(12345)

date <- paste0(substr(Sys.Date(), 1, 4), substr(Sys.Date(), 6, 7),  substr(Sys.Date(), 9, 10))

## County DDD Design -----------------------------------------------------------

county_df <- readRDS(paste0(folder_data, "county_df_analysis_year.rds"))

names(county_df)
# DiffinDiff by group 

# count 75

# high

fixest_model_dd_high <- feols(nbr_workers_certified_start_year ~ aewr_ppi * postdummy | county_fe + year_fe, 
                              data = subset(county_df, high_h2a_count_75 == 1), vcov = ~countyfips)

test <- county_df %>% 
  filter(high_h2a_count_75 == 1)

fixest_model_dd_high$coeftable # matches stata
fixest_model_dd_high$nobs

# low

fixest_model_dd_low <- feols(nbr_workers_requested_start_year ~ aewr_ppi * postdummy | county_fe + year_fe, 
                              data = subset(county_df, high_h2a_count_75 == 0), vcov = ~countyfips)

fixest_model_dd_low$coeftable # matches stata
fixest_model_dd_low$nobs

# count 66

# high

fixest_model_dd_high <- feols(nbr_workers_requested_start_year ~ aewr_ppi * postdummy | county_fe + year_fe, 
                              data = subset(county_df, high_h2a_count_66 == 1), vcov = ~countyfips)

fixest_model_dd_high$coeftable # matches stata
fixest_model_dd_high$nobs

# low

fixest_model_dd_low <- feols(nbr_workers_requested_start_year ~ aewr_ppi * postdummy | county_fe + year_fe, 
                             data = subset(county_df, high_h2a_count_66 == 0), vcov = ~countyfips)

fixest_model_dd_low$coeftable # matches stata
fixest_model_dd_low$nobs

# count 50

# high

fixest_model_dd_high <- feols(nbr_workers_requested_start_year ~ aewr_ppi_l1 * postdummy | county_fe + year_fe, 
                              data = subset(county_df, high_h2a_count_50 == 1), vcov = ~countyfips)

fixest_model_dd_high$coeftable # matches stata
fixest_model_dd_high$nobs

# low

fixest_model_dd_low <- feols(nbr_workers_requested_start_year ~ aewr_ppi_l1 * postdummy | county_fe + year_fe, 
                             data = subset(county_df, high_h2a_count_50 == 0), vcov = ~countyfips)

fixest_model_dd_low$coeftable # matches stata
fixest_model_dd_low$nobs


# DDD (simple)

fixest_model_ddd_simple <- feols(nbr_workers_requested_start_year ~ aewr * postdummy * high_h2a_share_75 + ln_pop_census + emp_pop_ratio | county_fe + year_fe, 
                             data = subset(county_df, any_cropland_2007 == 1), vcov = ~countyfips)

fixest_model_ddd_simple$coeftable # matches stata
fixest_model_ddd_simple$nobs

fixest_model_ddd_simple <- feols(nbr_workers_requested_start_year ~ 
                                   aewr * yeardummy_2008 * high_h2a_share_75 + 
                                   aewr * yeardummy_2009 * high_h2a_share_75 + 
                                   aewr * yeardummy_2010 * high_h2a_share_75 + 
                                   aewr * yeardummy_2011 * high_h2a_share_75 + 
                                   aewr * yeardummy_2013 * high_h2a_share_75 +
                                   aewr * yeardummy_2014 * high_h2a_share_75 + 
                                   aewr * yeardummy_2015 * high_h2a_share_75 +
                                   aewr * yeardummy_2016 * high_h2a_share_75 +
                                   aewr * yeardummy_2017 * high_h2a_share_75 +
                                   aewr * yeardummy_2018 * high_h2a_share_75 +
                                   aewr * yeardummy_2019 * high_h2a_share_75 +
                                   aewr * yeardummy_2020 * high_h2a_share_75 +
                                   aewr * yeardummy_2021 * high_h2a_share_75 +
                                   aewr * yeardummy_2022 * high_h2a_share_75 + 
                                   ln_pop_census + emp_pop_ratio | county_fe + year_fe, 
                                 data = subset(county_df, any_cropland_2007 == 1), vcov = ~countyfips)

fixest_model_ddd_simple$coeftable # matches stata
fixest_model_ddd_simple$nobs

wald(fixest_model_ddd_simple, keep = c("aewr:high_h2a_share_75:yeardummy_2013",
                                       "aewr:high_h2a_share_75:yeardummy_2014",
                                       "aewr:high_h2a_share_75:yeardummy_2015",
                                       "aewr:high_h2a_share_75:yeardummy_2016",
                                       "aewr:high_h2a_share_75:yeardummy_2017",
                                       "aewr:high_h2a_share_75:yeardummy_2018",
                                       "aewr:high_h2a_share_75:yeardummy_2019",
                                       "aewr:high_h2a_share_75:yeardummy_2020",
                                       "aewr:high_h2a_share_75:yeardummy_2021",
                                       "aewr:high_h2a_share_75:yeardummy_2022"), vcov = ~countyfips)

# graph it 

coeff_df <- data.frame(year = c(2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022),
                       beta = c(fixest_model_ddd_simple$coeftable[33, 1], 
                                fixest_model_ddd_simple$coeftable[34, 1], 
                                fixest_model_ddd_simple$coeftable[35, 1], 
                                fixest_model_ddd_simple$coeftable[36, 1],
                                0,
                                fixest_model_ddd_simple$coeftable[37, 1],
                                fixest_model_ddd_simple$coeftable[38, 1],
                                fixest_model_ddd_simple$coeftable[39, 1],
                                fixest_model_ddd_simple$coeftable[40, 1],
                                fixest_model_ddd_simple$coeftable[41, 1],
                                fixest_model_ddd_simple$coeftable[42, 1],
                                fixest_model_ddd_simple$coeftable[43, 1],
                                fixest_model_ddd_simple$coeftable[44, 1],
                                fixest_model_ddd_simple$coeftable[45, 1],
                                fixest_model_ddd_simple$coeftable[46, 1]),
                       se = c(fixest_model_ddd_simple$coeftable[33, 2], 
                              fixest_model_ddd_simple$coeftable[34, 2], 
                              fixest_model_ddd_simple$coeftable[35, 2], 
                              fixest_model_ddd_simple$coeftable[36, 2],
                              0,
                              fixest_model_ddd_simple$coeftable[37, 2],
                              fixest_model_ddd_simple$coeftable[38, 2],
                              fixest_model_ddd_simple$coeftable[39, 2],
                              fixest_model_ddd_simple$coeftable[40, 2],
                              fixest_model_ddd_simple$coeftable[41, 2],
                              fixest_model_ddd_simple$coeftable[42, 2],
                              fixest_model_ddd_simple$coeftable[43, 2],
                              fixest_model_ddd_simple$coeftable[44, 2],
                              fixest_model_ddd_simple$coeftable[45, 2],
                              fixest_model_ddd_simple$coeftable[46, 2])) 


coeff_df <- coeff_df %>% 
  mutate(upper_ci = beta + se * 1.96,
         lower_ci = beta - se * 1.96)

coefplot <- ggplot(data = coeff_df, aes(x = year))+
  geom_line(aes(y = beta), lwd = 1.5)+
  geom_line(aes(y = upper_ci), linetype = "dashed")+
  geom_line(aes(y = lower_ci), linetype = "dashed")+
  geom_vline(xintercept = 2012)+
  geom_hline(yintercept = 0)+
  theme_classic()
coefplot

ggsave(coefplot, filename = paste0(folder_output, "coefplot_ddd_high_h2a_share_75.png"), device = "png")

## within CZ design ------------------------------------------------------------

fixest_model_czdesign  <- feols(nbr_workers_requested_start_year ~ aewr_ppi_l2 + emp_farm | county_fe + cztime_fe,
                                   data = subset(county_df, border_cz == 1))

fixest_model_czdesign$coeftable # matches stata
fixest_model_czdesign$nobs

county_df %>% group_by(border_cz) %>% tally()

fixest_model_czdesign  <- feols(nbr_workers_requested_start_year ~ aewr_ppi_l2 * yeardummy_2008 + aewr_ppi_l2 * yeardummy_2009 + aewr_ppi_l2 * yeardummy_2010 + aewr_ppi_l2 * yeardummy_2011 + aewr_ppi_l2 * yeardummy_2013 + aewr_ppi_l2 * yeardummy_2014 + aewr_ppi_l2 * yeardummy_2015 + aewr_ppi_l2 * yeardummy_2016 + aewr_ppi_l2 * yeardummy_2017 + aewr_ppi_l2 * yeardummy_2018 + aewr_ppi_l2 * yeardummy_2019 + aewr_ppi_l2 * yeardummy_2020 + aewr_ppi_l2 * yeardummy_2021 + aewr_ppi_l2 * yeardummy_2022 + emp_farm | county_fe + cztime_fe,
                                data = subset(county_df, border_cz == 1))

fixest_model_czdesign$coeftable # matches stata
fixest_model_czdesign$nobs

fixest_model_czdesign  <- feols(nbr_workers_requested_start_year ~ aewr_ppi_l1 * yeardummy_2008 + aewr_ppi_l1 * yeardummy_2009 + aewr_ppi_l1 * yeardummy_2010 + aewr_ppi_l1 * yeardummy_2011 + aewr_ppi_l1 * yeardummy_2013 + aewr_ppi_l1 * yeardummy_2014 + aewr_ppi_l1 * yeardummy_2015 + aewr_ppi_l1 * yeardummy_2016 + aewr_ppi_l1 * yeardummy_2017 + aewr_ppi_l1 * yeardummy_2018 + aewr_ppi_l1 * yeardummy_2019 + aewr_ppi_l1 * yeardummy_2020 + aewr_ppi_l1 * yeardummy_2021 + aewr_ppi_l1 * yeardummy_2022 + emp_farm | county_fe + cztime_fe,
                                data = subset(county_df, border_cz == 1))

fixest_model_czdesign$coeftable # matches stata
fixest_model_czdesign$nobs

county_df %>% group_by(border_cz) %>% tally()

## border pair design ----------------------------------------------------------

border_df <- readRDS(paste0(folder_data, "border_df_analysis.rds"))
aewr_data <- readRDS(paste0(folder_data, "aewr_data.rds"))
h2a_data_ts <- readRDS(paste0(folder_data, "h2a_data_ts.rds"))
h2a_data <- readRDS(paste0(folder_data, "h2a_data.rds"))

border_df_yearly <- readRDS(paste0(folder_data, "border_df_analysis_year.rds"))
aewr_data_full <- readRDS(paste0(folder_data, "aewr_data_full.rds"))
# colors: https://colorbrewer2.org/?type=diverging&scheme=RdBu&n=8
# ['#b2182b','#d6604d','#f4a582','#fddbc7','#d1e5f0','#92c5de','#4393c3','#2166ac']  

# "#b2182b", "#d6604d", "#f4a582", "#fddbc7", "#d1e5f0", "#92c5de", "#4393c3", "#2166ac", "#b2182b", "#d6604d", "#f4a582", "#fddbc7", "#d1e5f0", "#92c5de", "#4393c3", "#2166ac", "#d1e5f0" 


aewr_data_full_ts <- aewr_data_full %>% 
  group_by(year) %>% 
  summarise(aewr = mean(aewr, na.rm = T), 
            aewr_ppi = mean(aewr_ppi, na.rm = T))

aewr_ts <- ggplot(data = aewr_data_full_ts, aes(x = year, y = aewr_ppi))+
  geom_line(linewidth = 1.25, color = "#2166ac")+
  theme_clean()+
  xlab("Year")+
  ylab("Real AEWR")+
  geom_vline(xintercept = 2012, linetype = "dashed")
aewr_ts
ggsave(filename = paste0(folder_output, "ts_national_aewr_real.png"), aewr_ts, device = "png")

## Sample Map ------------------------------------------------------------------

county_map <- us_counties(map_date = NULL, resolution = "low")
class(county_map)

head(county_map$statefp)

county_map$state_fip <- as.numeric(county_map$statefp)

head(county_map$state_fip)

county_map <- county_map %>% 
  filter(state_fip <= 56 & state_fip != 2 & state_fip != 15)

ggplot(county_map) +
  geom_sf()

county_map$countyfips <- as.numeric(str_c(county_map$statefp, county_map$countyfp))

head(county_map$countyfips)

border_df_merge <- border_df %>% 
  filter(census_period == 2012 & aewr_border_sample == 1) %>% 
  select(countyfips, aewr_region_num, aewr_border_side, border_side)

county_map_data <- merge(x = county_map, y = border_df_merge, by = "countyfips", all.x = T, all.y = F)

rm(border_df_merge)

samp_map <- ggplot(county_map_data) +
  geom_sf(aes(fill = as.factor(aewr_region_num)))+
  theme(panel.grid.major = element_line(color = gray(0.5), linetype = "dashed", 
                                        size = 0.5), panel.background = element_rect(fill = "aliceblue"))+
  theme_bw()+
  scale_fill_manual(values = c("#b2182b", "#d6604d", "#f4a582", "#fddbc7", "#d1e5f0", "#92c5de", "#4393c3", "#2166ac", "#b2182b", "#d6604d", "#f4a582", "#fddbc7", "#d1e5f0", "#92c5de", "#4393c3", "#2166ac", "#d1e5f0" ),
                    na.value = "#E5E4E2")+
  annotation_north_arrow(location = "bl", which_north = "true", 
                         pad_x = unit(0.75, "in"), pad_y = unit(0.5, "in"),
                         style = north_arrow_fancy_orienteering)+
  annotation_scale(location = "bl", width_hint = 0.4) +
  theme(legend.position = "none")
samp_map

ggsave(filename = paste0(folder_output, "map_aewr_sample.png"), samp_map, device = "png")


border_df_merge <- border_df %>% 
  filter(census_period == 2012 & aewr_border_sample == 1 & no_cropland_2007_pair != 1) %>% 
  select(countyfips, aewr_region_num, aewr_border_side)

county_map_data_drop <- merge(x = county_map, y = border_df_merge, by = "countyfips", all.x = T, all.y = F)

samp_map_drop <- ggplot(county_map_data_drop) +
  geom_sf(aes(fill = as.factor(aewr_region_num)))+
  theme(panel.grid.major = element_line(color = gray(0.5), linetype = "dashed", 
                                        size = 0.5), panel.background = element_rect(fill = "aliceblue"))+
  theme_bw()+
  scale_fill_manual(values = c("#b2182b", "#d6604d", "#f4a582", "#fddbc7", "#d1e5f0", "#92c5de", "#4393c3", "#2166ac", "#b2182b", "#d6604d", "#f4a582", "#fddbc7", "#d1e5f0", "#92c5de", "#4393c3", "#2166ac", "#d1e5f0" ),
                    na.value = "#E5E4E2")+
  annotation_north_arrow(location = "bl", which_north = "true", 
                         pad_x = unit(0.75, "in"), pad_y = unit(0.5, "in"),
                         style = north_arrow_fancy_orienteering)+
  annotation_scale(location = "bl", width_hint = 0.4) +
  theme(legend.position = "none")
samp_map_drop

ggsave(filename = paste0(folder_output, "map_aewr_sample_dropnocropland.png"), samp_map, device = "png")

## H2A Use Map ---------------------------------------------------------------


head(county_map)

h2a_map_data <- NULL

h2a_map_data_2012 <- h2a_data %>% 
  filter(census_period == 2012) %>% 
  mutate(ln_h2a_workers_req_2012 = log(nbr_workers_requested_start_year)) %>% 
  select(countyfips, ln_h2a_workers_req_2012)

h2a_map_data_2017 <- h2a_data %>% 
  filter(census_period == 2017) %>% 
  mutate(ln_h2a_workers_req_2017 = log(nbr_workers_requested_start_year)) %>% 
  select(countyfips, ln_h2a_workers_req_2017)

h2a_map_data_2022 <- h2a_data %>% 
  filter(census_period == 2022) %>% 
  mutate(ln_h2a_workers_req_2022 = log(nbr_workers_requested_start_year)) %>% 
  select(countyfips, ln_h2a_workers_req_2022)

h2a_map_data <- merge(x = h2a_map_data_2012, y = h2a_map_data_2017, by = "countyfips")
h2a_map_data <- merge(x = h2a_map_data, y = h2a_map_data_2022, by = "countyfips")

county_map_fips <- county_map %>% 
  mutate(countyfips = as.numeric(geoid))

h2a_map_data <- merge(x = county_map_fips, y = h2a_map_data, by = "countyfips", all.x = T, all.y = F)

# 2012

h2a_map_data_2012 <- ggplot(h2a_map_data) +
  geom_sf(aes(fill = ln_h2a_workers_req_2012), color=alpha("grey",0.5))+
  theme(panel.grid.major = element_line(color = gray(0.5), linetype = "dashed", 
                                        size = 0.5), panel.background = element_rect(fill = "aliceblue"))+
  theme_bw()+
  annotation_north_arrow(location = "bl", which_north = "true", 
                         pad_x = unit(0.75, "in"), pad_y = unit(0.5, "in"),
                         style = north_arrow_fancy_orienteering)+
  annotation_scale(location = "bl", width_hint = 0.4) +
  scale_fill_gradient2(
    low = muted("#2166ac"),
    mid = "white",
    midpoint = 5, 
    high = muted("#b2182b"),
    name = "H-2A Workers\nRequested (log)"
  )
h2a_map_data_2012

ggsave(filename = paste0(folder_output, "map_H2A_workers_2012.png"), h2a_map_data_2012, device = "png")


# 2017


h2a_map_data_2017 <- ggplot(h2a_map_data) +
  geom_sf(aes(fill = ln_h2a_workers_req_2017), color=alpha("grey",0.5))+
  theme(panel.grid.major = element_line(color = gray(0.5), linetype = "dashed", 
                                        size = 0.5), panel.background = element_rect(fill = "aliceblue"))+
  theme_bw()+
  annotation_north_arrow(location = "bl", which_north = "true", 
                         pad_x = unit(0.75, "in"), pad_y = unit(0.5, "in"),
                         style = north_arrow_fancy_orienteering)+
  annotation_scale(location = "bl", width_hint = 0.4) +
  scale_fill_gradient2(
    low = muted("#2166ac"),
    mid = "white",
    midpoint = 5, 
    high = muted("#b2182b"),
    name = "H-2A Workers\nRequested (log)"
  )
h2a_map_data_2017

ggsave(filename = paste0(folder_output, "map_H2A_workers_2017.png"), h2a_map_data_2017, device = "png")


# 2022


h2a_map_data_2022 <- ggplot(h2a_map_data) +
  geom_sf(aes(fill = ln_h2a_workers_req_2022), color=alpha("grey",0.5))+
  theme(panel.grid.major = element_line(color = gray(0.5), linetype = "dashed", 
                                        size = 0.5), panel.background = element_rect(fill = "aliceblue"))+
  theme_bw()+
  annotation_north_arrow(location = "bl", which_north = "true", 
                         pad_x = unit(0.75, "in"), pad_y = unit(0.5, "in"),
                         style = north_arrow_fancy_orienteering)+
  annotation_scale(location = "bl", width_hint = 0.4) +
  scale_fill_gradient2(
    low = muted("#2166ac"),
    mid = "white",
    midpoint = 5, 
    high = muted("#b2182b"),
    name = "H-2A Workers\nRequested (log)"
  )
h2a_map_data_2022

ggsave(filename = paste0(folder_output, "map_H2A_workers_2022.png"), h2a_map_data_2022, device = "png")



## AEWR Heat Map ---------------------------------------------------------------

head(county_map)

aewr_map_data <- NULL

aewr_map_data_2007 <- aewr_data %>% 
    filter(census_period == 2007) %>% 
    rename(aewr_2007 = aewr, 
           aewr_ppi_2007 = aewr_ppi) %>% 
    select(-census_period)

aewr_map_data_2012 <- aewr_data %>% 
  filter(census_period == 2012) %>% 
  rename(aewr_2012 = aewr, 
         aewr_ppi_2012 = aewr_ppi) %>% 
  select(-census_period)

aewr_map_data_2017 <- aewr_data %>% 
  filter(census_period == 2017) %>% 
  rename(aewr_2017 = aewr, 
         aewr_ppi_2017 = aewr_ppi) %>% 
  select(-census_period)

aewr_map_data_2022 <- aewr_data %>% 
  filter(census_period == 2022) %>% 
  rename(aewr_2022 = aewr, 
         aewr_ppi_2022 = aewr_ppi) %>% 
  select(-census_period)

aewr_map_data <- merge(x = aewr_map_data_2007, y = aewr_map_data_2012, by = "state_abbrev")
aewr_map_data <- merge(x = aewr_map_data, y = aewr_map_data_2017, by = "state_abbrev")
aewr_map_data <- merge(x = aewr_map_data, y = aewr_map_data_2022, by = "state_abbrev")

aewr_map_data <- merge(x = county_map, y = aewr_map_data, by.x = "state_abbr", by.y = "state_abbrev", all.x = T, all.y = F)

hist(aewr_data$aewr_ppi)

min(aewr_data$aewr_ppi)
max(aewr_data$aewr_ppi)

hist(aewr_map_data$aewr_ppi_2007)

midpnt <- 13.5

aewr_map_2007 <- ggplot(aewr_map_data) +
  geom_sf(aes(fill = aewr_ppi_2007), color=alpha("grey",0.5))+
  theme(panel.grid.major = element_line(color = gray(0.5), linetype = "dashed", 
                                        size = 0.5), panel.background = element_rect(fill = "aliceblue"))+
  theme_bw()+
  annotation_north_arrow(location = "bl", which_north = "true", 
                         pad_x = unit(0.75, "in"), pad_y = unit(0.5, "in"),
                         style = north_arrow_fancy_orienteering)+
  annotation_scale(location = "bl", width_hint = 0.4) +
  scale_fill_gradient2(
    low = muted("#2166ac"),
    mid = "white",
    high = muted("#b2182b"),
    midpoint = midpnt,
    name = "Real\nAEWR",
    limits = c(min(aewr_data$aewr_ppi), max(aewr_data$aewr_ppi))
  )
aewr_map_2007

ggsave(filename = paste0(folder_output, "map_aewr_values_ppi_2007.png"), aewr_map_2007, device = "png")

hist(aewr_map_data$aewr_ppi_2012)

aewr_map_2012 <- ggplot(aewr_map_data) +
  geom_sf(aes(fill = aewr_ppi_2012), color=alpha("grey",0.5))+
  theme(panel.grid.major = element_line(color = gray(0.5), linetype = "dashed", 
                                        size = 0.5), panel.background = element_rect(fill = "aliceblue"))+
  theme_bw()+
  annotation_north_arrow(location = "bl", which_north = "true", 
                         pad_x = unit(0.75, "in"), pad_y = unit(0.5, "in"),
                         style = north_arrow_fancy_orienteering)+
  annotation_scale(location = "bl", width_hint = 0.4) +
  scale_fill_gradient2(
    low = muted("#2166ac"),
    mid = "white",
    high = muted("#b2182b"),
    midpoint = midpnt,
    name = "Real\nAEWR",
    limits = c(min(aewr_data$aewr_ppi), max(aewr_data$aewr_ppi))
  )
aewr_map_2012

ggsave(filename = paste0(folder_output, "map_aewr_values_ppi_2012.png"), aewr_map_2012, device = "png")

hist(aewr_map_data$aewr_ppi_2017)

aewr_map_2017 <- ggplot(aewr_map_data) +
  geom_sf(aes(fill = aewr_ppi_2017), color=alpha("grey",0.5))+
  theme(panel.grid.major = element_line(color = gray(0.5), linetype = "dashed", 
                                        size = 0.5), panel.background = element_rect(fill = "aliceblue"))+
  theme_bw()+
  annotation_north_arrow(location = "bl", which_north = "true", 
                         pad_x = unit(0.75, "in"), pad_y = unit(0.5, "in"),
                         style = north_arrow_fancy_orienteering)+
  annotation_scale(location = "bl", width_hint = 0.4) +
  scale_fill_gradient2(
    low = muted("#2166ac"),
    mid = "white",
    high = muted("#b2182b"),
    midpoint = midpnt,
    name = "Real\nAEWR",
    limits = c(min(aewr_data$aewr_ppi), max(aewr_data$aewr_ppi))
  )
aewr_map_2017

ggsave(filename = paste0(folder_output, "map_aewr_values_ppi_2017.png"), aewr_map_2017, device = "png")

hist(aewr_map_data$aewr_ppi_2022)

aewr_map_2022 <- ggplot(aewr_map_data) +
  geom_sf(aes(fill = aewr_ppi_2022), color=alpha("grey",0.5))+
  theme(panel.grid.major = element_line(color = gray(0.5), linetype = "dashed", 
                                        size = 0.5), panel.background = element_rect(fill = "aliceblue"))+
  theme_bw()+
  annotation_north_arrow(location = "bl", which_north = "true", 
                         pad_x = unit(0.75, "in"), pad_y = unit(0.5, "in"),
                         style = north_arrow_fancy_orienteering)+
  annotation_scale(location = "bl", width_hint = 0.4) +
  scale_fill_gradient2(
    low = muted("#2166ac"),
    mid = "white",
    high = muted("#b2182b"),
    midpoint = midpnt,
    name = "Real\nAEWR",
    limits = c(min(aewr_data$aewr_ppi), max(aewr_data$aewr_ppi))
  )
aewr_map_2022

ggsave(filename = paste0(folder_output, "map_aewr_values_ppi_2022.png"), aewr_map_2022, device = "png")

## Ag Emp Share Fig ------------------------------------------------------------

# later

## H2A Use over time fig -------------------------------------------------------

# levels 


h2a_ts <- ggplot(data = subset(h2a_data_ts, case_year > 2007 & case_year <= 2022), aes(x = case_year))+
  geom_line(aes(y = h2a_nbr_workers_certified), color = "#4393c3", linewidth = 1.5)+
  theme_clean()+
  xlab("Year")+
  ylab("H-2A Workers Certified (Thousands)")+
  scale_y_continuous(breaks = c(100000, 1500000, 200000, 250000, 300000, 350000), labels = c("100", "150", "200", "250", "300", "350"))
h2a_ts

# indexed 

base_workers <- h2a_data_ts$h2a_nbr_workers_certified[ h2a_data_ts$case_year == 2012]
base_hours <- h2a_data_ts$h2a_man_hours_certified[ h2a_data_ts$case_year == 2012]
base_aps <- h2a_data_ts$n_applications[ h2a_data_ts$case_year == 2012]

h2a_data_ts_index <- h2a_data_ts %>% 
  mutate(indx_workers_certified = (h2a_nbr_workers_certified / base_workers[1]) * 100,
        indx_man_hours_certified = (h2a_man_hours_certified / base_hours[1]) * 100,
        indx_applications = (n_applications / base_aps[1]) * 100)

h2a_data_ts_index <- h2a_data_ts_index %>% 
  select(case_year, indx_workers_certified, indx_man_hours_certified, indx_applications)

h2a_data_ts_index <- h2a_data_ts_index %>% 
  pivot_longer(cols = starts_with("indx_"), names_to = "series", names_prefix = "indx_", values_to = "indx", values_drop_na = F)
head(h2a_data_ts_index)

h2a_indx_ts <- ggplot(data = subset(h2a_data_ts_index, case_year > 2007 & case_year <= 2022), aes(x = case_year, y = indx, group = series, color = series, linetype = series))+
  geom_line(linewidth = 1.25)+
  theme_clean()+
  scale_color_manual(values = c("#b2182b", "#2166ac", "#4393c3"), labels = c("Applications", "Man Hours Certified", "Workers Certified"))+
  scale_linetype_manual(values = c("solid", "longdash", "dotted"), labels = c("Applications", "Man Hours Certified", "Workers Certified"))+
  theme(legend.title = element_blank(), legend.position = "bottom")+
  xlab("Year")+
  ylab("Indexed Values (2012 = 100)")+
  geom_hline(yintercept = 100, linetype = "dashed")
h2a_indx_ts

ggsave(filename = paste0(folder_output, "fig_line_ts_h2a_workers_certified.png"), h2a_ts, device = "png")
ggsave(filename = paste0(folder_output, "fig_line_ts_h2a_indexes.png"), h2a_indx_ts, device = "png")


## Balance ---------------------------------------------------------------------

## First Stage Regression ------------------------------------------------------

## TSCB First Stage Regression -------------------------------------------------

# try with fixest 

aewr_border_df_drop <- border_df %>%
  filter(aewr_border_sample == 1 & no_cropland_2007_pair != 1 & census_period > 2007)

aewr_border_df <- border_df %>%
  filter(aewr_border_sample == 1 & census_period > 2007) # can't use 2007 data for h2A

dim(aewr_border_df_drop)
dim(aewr_border_df)

aewr_border_df %>% group_by(aewr_border_side) %>% tally()

aewr_border_df %>% group_by(census_period) %>% tally()

aewr_border_df %>% filter(is.na(emp_pop_ratio)) %>% select(countyfips, census_period)
# 51520, bristol city virginia, all urban, so drop it. 

pairs <- str_detect(unique(aewr_border_df$pair_id), "51520") 

which(pairs == T)

unique(aewr_border_df$pair_id)[which(pairs == T)]

dim(aewr_border_df)

aewr_border_df <- aewr_border_df %>% 
  filter(pair_id != unique(aewr_border_df$pair_id)[which(pairs == T)])


# 
# aewr_border_df %>% 
#   group_by(census_period) %>% 
#   tally()

# aewr_border_df_diagnose <- aewr_border_df %>% 
#   select( countyname, census_period, countyfips, fipsneighbor, neighbor_abbrev, state_abbrev, pair_id,  aewr_region_num, aewr_region_num_neighbor, aewr_region_border_id, pair_fe, pairtime_fe, county_fe, county_unique_fe, border_pair, aewr_ppi, ln_pop_census , h2a_nbr_workers_requested)

# out vars: h2a_man_hours_requested, h2a_nbr_workers_requested, n_applications

names(aewr_border_df)

aewr_border_df %>% 
  group_by(census_period) %>% 
  tally()

fixest_model_v2_cluster  <- feols(ln1_nbr_workers_requested_all_years ~ ln_aewr_ppi,
                                  vcov = ~aewr_region_border_id, data = aewr_border_df)

fixest_model_v2_cluster$coeftable # matches stata
fixest_model_v2_cluster$nobs

fixest_model_v2_cluster  <- feols(ln1_nbr_workers_requested_all_years ~ ln_aewr  | county_unique_fe + pairtime_fe,
                          vcov = ~aewr_region_border_id, data = aewr_border_df)

fixest_model_v2_cluster$coeftable # matches stata
fixest_model_v2_cluster$nobs

fixest_model_v2_cluster  <- feols(h2a_req_share_farm_workers_start_year ~ ln_aewr_ppi + share_undoc_prime + emp_pop_ratio  ,
                                  vcov = ~aewr_region_border_id, data = aewr_border_df)

fixest_model_v2_cluster$coeftable # matches stata
fixest_model_v2_cluster$nobs



fixest_model_v2_cluster  <- feols(ln1_nbr_workers_requested_all_years ~ ln_aewr  + share_undoc_prime + emp_pop_ratio  | county_unique_fe + census_period_fe,
                                  vcov = ~aewr_region_border_id, data = aewr_border_df)

fixest_model_v2_cluster$coeftable # matches stata
fixest_model_v2_cluster$nobs

fixest_model_v2_cluster  <- feols(ln1_nbr_workers_requested_all_years ~ ln_aewr_ppi  | county_unique_fe + pairtime_fe + state_fe,
                                  vcov = ~aewr_region_border_id, data = aewr_border_df)

fixest_model_v2_cluster$coeftable # matches stata
fixest_model_v2_cluster$nobs

## Yearly Regression ----------------------------------------------------------

# try with fixest 

aewr_border_df_drop_yearly <- border_df_yearly %>%
  filter(aewr_border_sample == 1 & no_cropland_2007_pair != 1)

aewr_border_df_yearly <- border_df_yearly %>%
  filter(aewr_border_sample == 1) # can't use 2007 data for h2A

dim(aewr_border_df_drop_yearly)
dim(aewr_border_df_yearly)

pairs <- str_detect(unique(aewr_border_df_yearly$pair_id), "51520") 

which(pairs == T)

unique(aewr_border_df_yearly$pair_id)[which(pairs == T)]

dim(aewr_border_df_yearly)

aewr_border_df_yearly <- aewr_border_df_yearly %>% 
  filter(pair_id != unique(aewr_border_df_yearly$pair_id)[which(pairs == T)])

fixest_model_v2_cluster  <- feols(ln1_nbr_workers_requested_start_years ~   ln_aewr_l2  | county_unique_fe + pairtime_fe,
                                  vcov = twoway ~ aewr_region_border_id + aewr_region_num , data = aewr_border_df_yearly)

fixest_model_v2_cluster$coeftable # matches stata
fixest_model_v2_cluster$nobs

fixest_model_v2_cluster  <- feols(ln1_nbr_workers_requested_start_years ~   ln_aewr_l2  | county_unique_fe + pairtime_fe,
                                  vcov =  ~ aewr_region_border_id  , data = aewr_border_df_yearly)

fixest_model_v2_cluster$coeftable # matches stata
fixest_model_v2_cluster$nobs


fixest_model_v2_cluster  <- feols(ln1_nbr_workers_requested_start_years ~   ln_aewr_l2  | county_unique_fe + pairtime_fe,
                                  vcov = twoway ~ aewr_region_border_id + aewr_region_num , data = subset(aewr_border_df_yearly, cz_same == 1))

fixest_model_v2_cluster$coeftable # matches stata
fixest_model_v2_cluster$nobs

## Bootstrap -------------------------------------------------------

# clustdfyear, this is our base datafram
# bootreps, this is our number of reps

bootreps <- 9999

input_data <- aewr_border_df_yearly %>% 
  filter(cz_same == 1) # limits sample to only pairs in the same CZ

dim(input_data) # 750 pairs

length(unique(input_data$pair_id))


pairs_df <- input_data %>%
  group_by(pair_id) %>%
  tally() %>%
  arrange(-n)

head(pairs_df)
summary(pairs_df$n)


# duplicate_pairs <- input_data %>%
#   filter(pair_id == "34017_36047" | pair_id == "34017_36061" | pair_id == "34017_36085" )
#
# duplicate_pairs
#
# 1 34017_36047    12
# 2 34017_36061    12
# 3 34017_36085    12

# rename

names(input_data)
names(input_data)[which(names(input_data) == "aewr_region_border_id")] <- "cluster" # aewr_region_border_id
names(input_data)[which(names(input_data) == "year")] <- "time" # census_period
names(input_data)[which(names(input_data) == "pair_id")] <- "pair" # pair_id
names(input_data)[which(names(input_data) == "aewr_border_side")] <- "side" # aewr_border_side w/in {0,1}
names(input_data)

dim(input_data)

periods <- unique(input_data$time) # grab the period variable

boot_treat_ests <- NULL # this is our storage vector

# from main DF, get clusters and number of pairs per cluster

cluster_list <- unique(input_data$cluster) # string

for (b in 1:bootreps) { # main boot loop starts


  # for each cluster, sample pairs with replacement

  boot_clust_samp <- NULL # store the clustered sample here

  cnt <- 1 # counter for boot sample id

  for (i in 1:length(cluster_list)) {

    # get the unique list of pairs

    temp_clust_df <- subset(input_data, cluster == cluster_list[i]) # iterate
    temp_pair_vec <- unique(temp_clust_df$pair)
    temp_clustpairsamp <- sample(temp_pair_vec, size = length(temp_pair_vec), replace = TRUE)
    temp_bootclustid <- seq(cnt, cnt + (length(temp_pair_vec)-1)) # this is a new ID for this boot sample

    temp_clust_df <- data.frame(pair = temp_clustpairsamp, cluster = cluster_list[i], boot_id = temp_bootclustid) # iterate

    boot_clust_samp <- rbind(boot_clust_samp, temp_clust_df)

    rm(temp_clust_df, temp_pair_vec, temp_clustpairsamp)

    cnt <- max(temp_bootclustid) + 1

    rm(temp_bootclustid)

    # print(i/length(cluster_list))
  }

  # use to make dataset for this bootstrap iteration

  # match years

  temp <- boot_clust_samp

  boot_clust_samp <- NULL

  for (i in 1:length(periods)) {
    temp$time <- periods[i]
    boot_clust_samp <- rbind(boot_clust_samp, temp)
    temp <- subset(temp, select = -c(time)) # remove the column we added
    # print(i)
  }

  dim(boot_clust_samp)

  # double and add treat / control indicator (or border side indicator, both would work)

  # match w/ 1 and 2 designating the side

#  order_vec <- sort(rep(seq(1,2), dim(boot_clust_samp)[1]))
  order_vec <- sort(rep(seq(0,1), dim(boot_clust_samp)[1]))

  boot_samp_int <- rbind(boot_clust_samp, boot_clust_samp)

  boot_samp_int$side <- order_vec

  rm(boot_clust_samp)

  # merge with data

  dim(boot_samp_int)
  dim(input_data)

  # make bootpair id (handles replacements) #

  boot_samp_final <- merge(x = boot_samp_int, y = input_data, by = c("pair", "cluster", "time", "side"), all.x = T, all.y = F)
  dim(boot_samp_final)

  rm(boot_samp_int)

  # fix the FEs #

  # units will be given by boot id and side

  boot_samp_final$boot_unit_fe <- with(boot_samp_final, interaction(as.factor(boot_id),  side))

  # then pairs are marked by boot id, and we will use time for the year dimension

  boot_samp_final$boot_pairtime_fe <- with(boot_samp_final, interaction(as.factor(boot_id),  time))

  levels(boot_samp_final$boot_pairtime_fe) <- c(levels(boot_samp_final$boot_pairtime_fe),"0.0")

  boot_samp_final$boot_pairtime_fe[boot_samp_final$time == periods[1]] <- "0.0" # first period is 0

  #################
  # run the model #
  #################

  # fixest_model  <- feols(outcome ~ treatcnts | factor(boot_unit_fe) + factor(boot_pairtime_fe), data = boot_samp_final)
  #
  # boot_est <- fixest_model$coeftable[1,1]
  #
  # boot_treat_ests <- c(boot_treat_ests, boot_est)
  #
  # rm(boot_samp_final)

  # make a table, 4 models, 3 outcome vars

  # outvars: ln1_nbr_workers_requested_start_years, ln1_nbr_workers_requested_all_years, nbr_workers_requested_start_year, nbr_workers_requested_all_years
  # models: aewr alone, add pop, add share, full
  m1_o1 <- feols(ln1_nbr_workers_requested_start_years ~ ln_aewr_ppi_l1 | boot_unit_fe + boot_pairtime_fe,
                 data = boot_samp_final)
  m2_o1 <- feols(ln1_nbr_workers_requested_start_years ~ ln_aewr_ppi_l1 + ln_pop_census + emp_pop_ratio | boot_unit_fe + boot_pairtime_fe,
                 data = boot_samp_final)
  m3_o1 <- feols(ln1_nbr_workers_requested_start_years ~ ln_aewr_ppi_l2 | boot_unit_fe + boot_pairtime_fe,
                 data = boot_samp_final)
  m4_o1 <- feols(ln1_nbr_workers_requested_start_years ~ ln_aewr_ppi_l2 + ln_pop_census + emp_pop_ratio | boot_unit_fe + boot_pairtime_fe,
                 data = boot_samp_final)

  out1bootres <- data.frame(outvar = c("ln1_nbr_workers_requested_start_years", "ln1_nbr_workers_requested_start_years", "ln1_nbr_workers_requested_start_years", "ln1_nbr_workers_requested_start_years"),
                            model = c(1, 2, 3, 4),
                            aewr_ppi_coef = c(m1_o1$coefficients[1], m2_o1$coefficients[1], m3_o1$coefficients[1], m4_o1$coefficients[1]),
                            lags = c(1, 1, 2, 2),
                            logs = c("Y", "Y", "Y", "Y"),
                            controls = c("N", "ln_pop_census and emp_pop_ratio", "N", "ln_pop_census and emp_pop_ratio"),
                            bootrep = c(b, b, b, b), 
                            obs = c( m1_o1$nobs , m2_o1$nobs , m3_o1$nobs , m4_o1$nobs))

  m1_o2 <- feols(ln1_nbr_workers_requested_all_years ~ ln_aewr_ppi_l1 | boot_unit_fe + boot_pairtime_fe,
                 data = boot_samp_final)
  m2_o2 <- feols(ln1_nbr_workers_requested_all_years ~ ln_aewr_ppi_l1 + ln_pop_census + emp_pop_ratio | boot_unit_fe + boot_pairtime_fe,
                 data = boot_samp_final)
  m3_o2 <- feols(ln1_nbr_workers_requested_all_years ~ ln_aewr_ppi_l2 | boot_unit_fe + boot_pairtime_fe,
                 data = boot_samp_final)
  m4_o2 <- feols(ln1_nbr_workers_requested_all_years ~ ln_aewr_ppi_l2 + ln_pop_census + emp_pop_ratio | boot_unit_fe + boot_pairtime_fe,
                 data = boot_samp_final)
  
  out2bootres <- data.frame(outvar = c("ln1_nbr_workers_requested_all_years", "ln1_nbr_workers_requested_all_years", "ln1_nbr_workers_requested_all_years", "ln1_nbr_workers_requested_all_years"),
                            model = c(1, 2, 3, 4),
                            aewr_ppi_coef = c(m1_o2$coefficients[1], m2_o2$coefficients[1], m3_o2$coefficients[1], m4_o2$coefficients[1]),
                            lags = c(1, 1, 2, 2),
                            logs = c("Y", "Y", "Y", "Y"),
                            controls = c("N", "ln_pop_census and emp_pop_ratio", "N", "ln_pop_census and emp_pop_ratio"),
                            bootrep = c(b, b, b, b), 
                            obs = c( m1_o2$nobs , m2_o2$nobs , m3_o2$nobs , m4_o2$nobs))
  
  m1_o3 <- feols(nbr_workers_requested_start_year ~ aewr_ppi_l1 | boot_unit_fe + boot_pairtime_fe,
                 data = boot_samp_final)
  m2_o3 <- feols(nbr_workers_requested_start_year ~ aewr_ppi_l1 + ln_pop_census + emp_pop_ratio | boot_unit_fe + boot_pairtime_fe,
                 data = boot_samp_final)
  m3_o3 <- feols(nbr_workers_requested_start_year ~ aewr_ppi_l2 | boot_unit_fe + boot_pairtime_fe,
                 data = boot_samp_final)
  m4_o3 <- feols(nbr_workers_requested_start_year ~ aewr_ppi_l2 + ln_pop_census + emp_pop_ratio | boot_unit_fe + boot_pairtime_fe,
                 data = boot_samp_final)
  
  out3bootres <- data.frame(outvar = c("nbr_workers_requested_start_year", "nbr_workers_requested_start_year", "nbr_workers_requested_start_year", "nbr_workers_requested_start_year"),
                            model = c(1, 2, 3, 4),
                            aewr_ppi_coef = c(m1_o3$coefficients[1], m2_o3$coefficients[1], m3_o3$coefficients[1], m4_o3$coefficients[1]),
                            lags = c(1, 1, 2, 2),
                            logs = c("N", "N", "N", "N"),
                            controls = c("N", "ln_pop_census and emp_pop_ratio", "N", "ln_pop_census and emp_pop_ratio"),
                            bootrep = c(b, b, b, b), 
                            obs = c( m1_o3$nobs , m2_o3$nobs , m3_o3$nobs , m4_o3$nobs))
  
  m1_o4 <- feols(nbr_workers_requested_all_years ~ aewr_ppi_l1 | boot_unit_fe + boot_pairtime_fe,
                 data = boot_samp_final)
  m2_o4 <- feols(nbr_workers_requested_all_years ~ aewr_ppi_l1 + ln_pop_census + emp_pop_ratio | boot_unit_fe + boot_pairtime_fe,
                 data = boot_samp_final)
  m3_o4 <- feols(nbr_workers_requested_all_years ~ aewr_ppi_l2 | boot_unit_fe + boot_pairtime_fe,
                 data = boot_samp_final)
  m4_o4 <- feols(nbr_workers_requested_all_years ~ aewr_ppi_l2 + ln_pop_census + emp_pop_ratio | boot_unit_fe + boot_pairtime_fe,
                 data = boot_samp_final)
  
  out4bootres <- data.frame(outvar = c("nbr_workers_requested_all_years", "nbr_workers_requested_all_years", "nbr_workers_requested_all_years", "nbr_workers_requested_all_years"),
                            model = c(1, 2, 3, 4),
                            aewr_ppi_coef = c(m1_o4$coefficients[1], m2_o4$coefficients[1], m3_o4$coefficients[1], m4_o4$coefficients[1]),
                            lags = c(1, 1, 2, 2),
                            logs = c("N", "N", "N", "N"),
                            controls = c("N", "ln_pop_census and emp_pop_ratio", "N", "ln_pop_census and emp_pop_ratio"),
                            bootrep = c(b, b, b, b), 
                            obs = c( m1_o4$nobs , m2_o4$nobs , m3_o4$nobs , m4_o4$nobs))


  boot_treat_ests <- rbind(boot_treat_ests, out1bootres)
  boot_treat_ests <- rbind(boot_treat_ests, out2bootres)
  boot_treat_ests <- rbind(boot_treat_ests, out3bootres)
  boot_treat_ests <- rbind(boot_treat_ests, out4bootres)

  print("### Boot Percent Complete ###")
  print(b / bootreps)

} # end boot loop

boot_treat_ests <- boot_treat_ests %>% 
  mutate(note = "CZ Pair Smaple")

write.csv(boot_treat_ests, file = paste0(folder_output, "boot_tscb_paired_first_stage_r", bootreps, "_", date, ".csv"))

# put it in a nice table #

head(boot_treat_ests)

# cols: out vars, models
# rows: controls

hist(boot_treat_ests$obs)
summary(boot_treat_ests$obs)

boot_treat_sum_test <- boot_treat_ests %>%
  filter(outvar == "h2a_nbr_workers_requested" & model == 4)

sd(boot_treat_sum_test$aewr_ppi_coef)

boot_treat_sum <- boot_treat_ests %>%
  group_by(outvar, model, lags, logs, controls) %>%
  summarise(aewr_ppi = mean(aewr_ppi_coef, na.rm = T),
            aewr_ppi_bootse = sd(aewr_ppi_coef, na.rm = T),
            aewr_ppi_confinf_l = quantile(aewr_ppi_coef, prob = 0.025, na.rm = T),
            aewr_ppi_confinf_u = quantile(aewr_ppi_coef, prob = 0.975, na.rm = T),
            obs = mean(obs, na.rm = T)) %>% 
  mutate(note = "CZ Pair Smaple")

boot_treat_sum <- boot_treat_sum %>% 
  mutate(est_fstat = (aewr_ppi / aewr_ppi_bootse)^2,
         est_tstat = (aewr_ppi / aewr_ppi_bootse))
 
write.csv(boot_treat_sum, file = paste0(folder_output, "boot_tscb_paired_table_summary_first_stage_r", bootreps, "_", date, ".csv"))

names(boot_treat_sum)

aewr_est_fig <- ggplot(data = boot_treat_sum, aes(x = model))+
  geom_point(aes(y = aewr_ppi))+
  geom_errorbar(aes(ymin = aewr_ppi_confinf_l, ymax = aewr_ppi_confinf_u))+
  facet_wrap(~outvar, scales = "free")+
  geom_hline(yintercept = 0)
aewr_est_fig

ggsave(filename = paste0(folder_output, "fig_crude_coefplot_boot_r", bootreps, "_", date, ".png"), aewr_est_fig, device = "png")

aewr_est_fig <- ggplot(data = subset(boot_treat_sum, outvar == "ln1_nbr_workers_requested_start_years"), aes(x = model))+
  geom_point(aes(y = aewr_ppi))+
  geom_errorbar(aes(ymin = aewr_ppi_confinf_l, ymax = aewr_ppi_confinf_u))+
  geom_hline(yintercept = 0)+
  ylab("Log Real AEWR")
aewr_est_fig

ggsave(filename = paste0(folder_output, "fig_crude_coefplot_boot_ln1_nbr_workers_requested_start_years_r", bootreps, "_", date, ".png"), aewr_est_fig, device = "png")
