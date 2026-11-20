# Build the exposure and dose table that the domestic-wage registry freezes.
# Inputs are pre-policy only: 2022-2024 certified positions, 2011 farm
# employment, and the statutory rates. The output is copied into
# explorations/episode_2025/domestic_wage_exposure_frozen.csv and hashed.
# Args: 1 = scratch output dir (outside the repo).
source("explorations/episode_2025/ep_helpers.R")
out_dir <- id_output_dir()
frame <- ep_state_frame()
p <- arrow::open_dataset(path_processed("county_year_panel.parquet")) %>% filter(year %in% 2022:2024) %>%
  select(county_fips, state_fips, cz_id, year, h2a_cert_share_farm_workers_2011_start_year, bea_farm_emp_2011) %>% collect()
stopifnot(!anyDuplicated(p[c("county_fips", "year")]), is.character(p$county_fips))
e <- p %>% group_by(county_fips, state_fips, cz_id = as.character(cz_id)) %>%
  summarise(S = mean(h2a_cert_share_farm_workers_2011_start_year), farm_emp_2011 = first(bea_farm_emp_2011), years = n(), .groups = "drop") %>%
  filter(is.finite(S), years == 3)
cz <- e %>% filter(is.finite(farm_emp_2011), farm_emp_2011 > 0) %>% group_by(cz_id) %>%
  summarise(S_cz = sum(S * farm_emp_2011) / sum(farm_emp_2011), .groups = "drop")
e <- e %>% left_join(cz, by = "cz_id") %>%
  mutate(S_winsorized = pmin(S, quantile(S, .99)), S_cz_winsorized = pmin(S_cz, quantile(S_cz, .99, na.rm = TRUE))) %>%
  inner_join(frame %>% transmute(state_fips, cut_us = dose1_us, cut_h2a = dose1_h2a), by = "state_fips") %>%
  select(county_fips, state_fips, cz_id, S, S_winsorized, S_cz, S_cz_winsorized, cut_us, cut_h2a) %>% arrange(county_fips)
stopifnot(!anyDuplicated(e$county_fips), !anyNA(e[c("S_winsorized", "cut_us", "cut_h2a")]),
  identical(e$county_fips, harmonize_county_fips_2010(e$county_fips)))
id_write(e, out_dir, "domestic_wage_exposure_frozen.csv")
cat("counties", nrow(e), " states", n_distinct(e$state_fips), " S p99 cap", max(e$S_winsorized), " missing S_cz", sum(is.na(e$S_cz)), "\n")
print(summary(e[c("S", "S_winsorized", "S_cz_winsorized", "cut_us", "cut_h2a")]))
