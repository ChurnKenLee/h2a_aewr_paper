# Descriptive: are H-2A workers inside the QCEW crop series?
# Compares H-2A full-time equivalents (certified hours / 2,080; certified hours
# overstate hours worked) with state private-sector QCEW employment in crop
# production (NAICS 111) and crop support activities (NAICS 1151). State totals
# are used so that misallocating a worksite to the wrong county cannot matter.
# Years through 2024 only. Args: 1 = scratch output dir (outside the repo).
source("explorations/episode_2025/ep_helpers.R")
out_dir <- id_output_dir()
YEARS <- c(2011L, 2019L, 2024L)
q <- arrow::open_dataset(path_int("qcew.parquet")) %>%
  filter(year %in% YEARS, own_code == "5",
    (industry_code == "111" & agglvl_code == "55") | (industry_code == "1151" & agglvl_code == "56")) %>%
  select(area_fips, year, industry_code, disclosure_code, annual_avg_emplvl) %>% collect() %>%
  filter(grepl("^[0-9]{2}000$", area_fips)) %>%
  transmute(state_fips = substr(area_fips, 1, 2), year = as.integer(year), industry_code,
    emp = ifelse(disclosure_code %in% "N", NA_real_, annual_avg_emplvl))
stopifnot(!anyDuplicated(q[c("state_fips", "year", "industry_code")]))
q <- tidyr::pivot_wider(q, names_from = industry_code, values_from = emp, names_prefix = "naics_")
fte <- arrow::open_dataset(path_processed("county_year_panel.parquet")) %>% filter(year %in% YEARS) %>%
  select(state_fips, year, man_hours_certified_start_year) %>% collect() %>%
  group_by(state_fips, year) %>% summarise(h2a_fte = sum(man_hours_certified_start_year, na.rm = TRUE) / 2080, .groups = "drop")
d <- inner_join(fte, q, by = c("state_fips", "year")) %>%
  mutate(fte_over_crop = h2a_fte / naics_111, fte_over_crop_and_support = h2a_fte / (naics_111 + naics_1151))
id_write(d, out_dir, "h2a_fte_vs_qcew_state.csv")
nat <- d %>% group_by(year) %>% summarise(h2a_fte_k = sum(h2a_fte) / 1e3, crop_k = sum(naics_111, na.rm = TRUE) / 1e3,
  crop_support_k = sum(naics_1151, na.rm = TRUE) / 1e3, states_crop_suppressed = sum(is.na(naics_111)),
  states_support_suppressed = sum(is.na(naics_1151)))
last <- d %>% filter(year == max(YEARS))
excess <- tibble(
  rule = c("H-2A FTE exceeds crop employment", "H-2A FTE exceeds crop + crop-support employment"),
  states = c(sum(last$fte_over_crop > 1, na.rm = TRUE), sum(last$fte_over_crop_and_support > 1, na.rm = TRUE)),
  share_of_national_h2a_fte = c(sum(last$h2a_fte[which(last$fte_over_crop > 1)]), sum(last$h2a_fte[which(last$fte_over_crop_and_support > 1)])) / sum(last$h2a_fte),
  state_fips = c(paste(sort(last$state_fips[which(last$fte_over_crop > 1)]), collapse = " "),
    paste(sort(last$state_fips[which(last$fte_over_crop_and_support > 1)]), collapse = " ")))
w <- d %>% filter(year %in% range(YEARS)) %>% select(state_fips, year, h2a_fte, naics_111, naics_1151) %>%
  tidyr::pivot_wider(names_from = year, values_from = c(h2a_fte, naics_111, naics_1151))
chg <- tibble(state_fips = w$state_fips, d_fte = w[[3]] - w[[2]], d_crop = w[[5]] - w[[4]], d_support = w[[7]] - w[[6]]) %>%
  filter(is.finite(d_crop), is.finite(d_support))
slopes <- tibble(states = nrow(chg), slope_crop_on_h2a = coef(lm(d_crop ~ d_fte, chg))[2],
  slope_crop_and_support_on_h2a = coef(lm(I(d_crop + d_support) ~ d_fte, chg))[2])
id_write(nat, out_dir, "h2a_fte_vs_qcew_national.csv"); id_write(excess, out_dir, "h2a_fte_vs_qcew_excess_states.csv")
id_write(slopes, out_dir, "h2a_fte_vs_qcew_change_slopes.csv")
options(width = 200)
print(as.data.frame(nat), digits = 4); print(as.data.frame(excess), digits = 3); print(as.data.frame(slopes), digits = 3)
print(as.data.frame(last %>% arrange(desc(h2a_fte)) %>% head(12) %>% mutate(h2a_fte = round(h2a_fte))), digits = 3)
