# Treatment support for the episode's dated events. Rates and FY2025 hours only; no outcome after FY2025.
#   event 1: 2025-10-02 statewide skill-I rates replace the FY2025 regional rates;
#   event 2: 2026-08-03 schedule replaces the 2025-10-02 schedule.
# A third event (the replacement rule ordered by the court) has no published rates yet.
# Args: 1 = scratch output dir (outside the repo); 2 = shared-outcomes dir.
source("explorations/episode_2025/ep_helpers.R")
out_dir <- id_output_dir()
shared <- ep_shared_dir()
frame <- ep_state_frame()
h25 <- arrow::open_dataset(file.path(shared, "outcomes_county_fy.parquet")) %>% filter(year == EP_LAST_PRE_FY) %>%
  select(state_fips, cert_hours_fy) %>% collect() %>% group_by(state_fips) %>% summarise(hours25 = sum(cert_hours_fy, na.rm = TRUE), .groups = "drop")
d <- frame %>% inner_join(h25, by = "state_fips") %>% mutate(w = hours25)
wm <- function(x, w) sum(w * x) / sum(w)
wsd <- function(x, w) sqrt(sum(w * (x - wm(x, w))^2) / sum(w))
within <- function(x, d) { ms <- d$multi_state_region; sqrt(sum(d$w[ms] * ep_resid(d[ms, ], x, fe = "aewr_region_id")^2) / sum(d$w[ms])) }
row <- function(v, label) tibble(dose = label, states = nrow(d), mean = mean(d[[v]]), sd = sd(d[[v]]), min = min(d[[v]]), max = max(d[[v]]),
  hours_weighted_mean = wm(d[[v]], d$w), hours_weighted_sd = wsd(d[[v]], d$w), hours_weighted_sd_within_former_region = within(v, d),
  share_states_negative = mean(d[[v]] < 0))
tab <- bind_rows(row("dose1_us", "event 1, U.S. skill I: 2025-10-02 rate minus FY2025 rate"),
  row("dose1_h2a", "event 1, H-2A skill I cash: 2025-10-02 rate minus FY2025 rate"),
  row("dose2_us", "event 2, U.S. skill I: 2026-08-03 rate minus 2025-10-02 rate"),
  row("dose2_h2a", "event 2, H-2A skill I cash: 2026-08-03 rate minus 2025-10-02 rate"))
cors <- tibble(pair = c("event 1 and event 2 doses, U.S. skill I", "same, within former region", "event 1 and event 2 doses, H-2A cash"),
  correlation = c(cor(d$dose1_us, d$dose2_us),
    { ms <- d[d$multi_state_region, ]; cor(ep_resid(ms, "dose1_us", fe = "aewr_region_id"), ep_resid(ms, "dose2_us", fe = "aewr_region_id")) },
    cor(d$dose1_h2a, d$dose2_h2a)))
id_write(tab, out_dir, "event_dose_support.csv"); id_write(cors, out_dir, "event_dose_correlations.csv")
id_write(d %>% select(state_fips, aewr_region_id, multi_state_region, rate25, us_e1, us_e2, h2a_e1, h2a_e2, dose1_us, dose2_us, dose1_h2a, dose2_h2a, hours25),
  out_dir, "event_dose_by_state.csv")
options(width = 220)
print(as.data.frame(tab), digits = 3); print(as.data.frame(cors), digits = 3)
