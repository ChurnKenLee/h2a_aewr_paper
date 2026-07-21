# Purpose: Compare FLS wage changes with ACS, OEWS, and QCEW proxies.
# Inputs: regional/state FLS and state proxy-wage parquets plus geography crosswalks.
# Outputs: IV proxy-comparison figures in outputs/figures.
# This diagnostic is informative but is not an input to instrument construction.

source(
  if (file.exists(file.path("code", "bootstrap_paths.R"))) {
    file.path("code", "bootstrap_paths.R")
  } else {
    file.path("..", "bootstrap_paths.R")
  }
)
library(arrow)
library(tidyverse)
library(janitor)

fls_region <- read_parquet(path_int("fls_region.parquet"))
fls_state <- read_parquet(path_int("fls_state.parquet"))
acs <- read_parquet(path_int("acs_state_ag_wage.parquet"))
oews <- read_parquet(path_int("oews_state_aggregated.parquet")) %>%
  filter(occ_code == "AEWR")
qcew <- read_parquet(path_int("qcew_state_ag_wage.parquet"))

aewr_region <- read_csv(
  path_raw("geographic_crosswalks", "phil", "aewr_regions.csv"),
  show_col_types = FALSE
) %>%
  clean_names()

fips_codes <- read_csv(
  path_raw("geographic_crosswalks", "phil", "fips_codes.csv"),
  show_col_types = FALSE
) %>%
  clean_names() %>%
  transmute(
    state_fips_code = sprintf("%02d", as.integer(fips)),
    state_abbrev
  )

aewr_region <- aewr_region %>%
  inner_join(fips_codes, by = "state_abbrev") %>%
  mutate(aewr_region_num = as.integer(aewr_region_num))

fls_regions <- fls_region %>%
  distinct(aewr_region_num, region_name) %>%
  mutate(aewr_region_num = as.integer(aewr_region_num))

fls_states <- fls_state %>%
  distinct(state_fips_code, state_name) %>%
  mutate(state_fips_code = sprintf("%02d", as.integer(state_fips_code)))

fls_region_ts <- bind_rows(
  fls_region %>%
    transmute(
      year = revised_year,
      aewr_region_num = as.integer(aewr_region_num),
      source = "FLS",
      wage = field_livestock_revised,
      fls_release = "revised"
    ),
  fls_region %>%
    transmute(
      year = preliminary_year,
      aewr_region_num = as.integer(aewr_region_num),
      source = "FLS",
      wage = field_livestock_preliminary,
      fls_release = "preliminary"
    )
) %>%
  filter(!is.na(year), !is.na(wage), wage > 0) %>%
  mutate(
    year = as.integer(year),
    fls_release_rank = if_else(fls_release == "revised", 1L, 2L)
  ) %>%
  arrange(aewr_region_num, year, fls_release_rank) %>%
  distinct(aewr_region_num, year, .keep_all = TRUE) %>%
  select(year, aewr_region_num, source, wage)

fls_state_ts <- bind_rows(
  fls_state %>%
    transmute(
      year = revised_year,
      state_fips_code = sprintf("%02d", as.integer(state_fips_code)),
      source = "FLS",
      wage = field_livestock_revised,
      fls_release = "revised"
    ),
  fls_state %>%
    transmute(
      year = preliminary_year,
      state_fips_code = sprintf("%02d", as.integer(state_fips_code)),
      source = "FLS",
      wage = field_livestock_preliminary,
      fls_release = "preliminary"
    )
) %>%
  filter(!is.na(year), !is.na(wage), wage > 0) %>%
  mutate(
    year = as.integer(year),
    fls_release_rank = if_else(fls_release == "revised", 1L, 2L)
  ) %>%
  arrange(state_fips_code, year, fls_release_rank) %>%
  distinct(state_fips_code, year, .keep_all = TRUE) %>%
  select(year, state_fips_code, source, wage)

proxy_state <- bind_rows(
  acs %>%
    transmute(
      source = "ACS",
      year = as.integer(year),
      state_fips_code = sprintf("%02d", as.integer(state_fips_code)),
      wage = acs_ag_mean_hourly_wage,
      weight = acs_ag_workers_perwt
    ),
  oews %>%
    transmute(
      source = "OEWS",
      year = as.integer(year),
      state_fips_code = sprintf("%02d", as.integer(state_fips_code)),
      wage = oews_mean_hourly_wage,
      weight = oews_tot_emp
    ),
  qcew %>%
    transmute(
      source = "QCEW",
      year = as.integer(year),
      state_fips_code = sprintf("%02d", as.integer(state_fips_code)),
      wage = qcew_ag_mean_hourly_wage_40h,
      weight = qcew_ag_workers
    )
) %>%
  filter(!is.na(wage), wage > 0, !is.na(weight), weight > 0)

proxy_region_ts <- proxy_state %>%
  inner_join(aewr_region, by = "state_fips_code") %>%
  group_by(source, year, aewr_region_num) %>%
  summarise(wage = weighted.mean(wage, weight, na.rm = TRUE), .groups = "drop")

region_ts <- bind_rows(fls_region_ts, proxy_region_ts) %>%
  inner_join(fls_regions, by = "aewr_region_num") %>%
  arrange(source, aewr_region_num, year) %>%
  group_by(source, aewr_region_num) %>%
  mutate(log_change = log(wage) - lag(log(wage))) %>%
  ungroup() %>%
  filter(!is.na(log_change)) %>%
  filter(year >= 2001)

state_ts <- bind_rows(
  fls_state_ts,
  proxy_state %>% select(year, state_fips_code, source, wage)
) %>%
  inner_join(fls_states, by = "state_fips_code") %>%
  arrange(source, state_fips_code, year) %>%
  group_by(source, state_fips_code) %>%
  mutate(log_change = log(wage) - lag(log(wage))) %>%
  ungroup() %>%
  filter(!is.na(log_change)) %>%
  filter(year >= 2001 & year <= 2010)

region_change_ymax <- region_ts %>%
  filter(source != "ACS") %>%
  summarise(ymax = 1.05 * max(abs(log_change), na.rm = TRUE)) %>%
  pull(ymax)

state_change_ymax <- state_ts %>%
  filter(source != "ACS") %>%
  summarise(ymax = 1.05 * max(abs(log_change), na.rm = TRUE)) %>%
  pull(ymax)

source_colors <- c(
  FLS = "black",
  ACS = "#1b9e77",
  OEWS = "#d95f02",
  QCEW = "#7570b3"
)

dir.create(path_figures(), recursive = TRUE, showWarnings = FALSE)

for (s in c("ACS", "OEWS", "QCEW")) {
  plot_data <- region_ts %>%
    filter(source %in% c("FLS", s))

  fig_region_change <- ggplot(
    plot_data,
    aes(x = year, y = log_change, color = source, linewidth = source == "FLS")
  ) +
    geom_hline(yintercept = 0, color = "grey80") +
    geom_line(alpha = 0.9) +
    facet_wrap(~region_name, ncol = 4) +
    scale_color_manual(values = source_colors[c("FLS", s)]) +
    scale_linewidth_manual(
      values = c(`FALSE` = 0.55, `TRUE` = 0.9),
      guide = "none"
    ) +
    scale_y_continuous(labels = scales::percent_format(accuracy = 1)) +
    coord_cartesian(ylim = c(-region_change_ymax, region_change_ymax)) +
    labs(
      x = NULL,
      y = "Annual log change",
      color = NULL,
      title = paste("FLS and", s, "Wage Growth by AEWR Region")
    ) +
    theme_minimal(base_size = 10) +
    theme(legend.position = "bottom")

  ggsave(
    path_figures(paste0(
      "iv_proxy_fls_region_log_changes_",
      tolower(s),
      ".png"
    )),
    fig_region_change,
    width = 12,
    height = 9,
    dpi = 200
  )

  plot_data <- state_ts %>%
    filter(source %in% c("FLS", s))

  fig_state_change <- ggplot(
    plot_data,
    aes(x = year, y = log_change, color = source, linewidth = source == "FLS")
  ) +
    geom_hline(yintercept = 0, color = "grey80") +
    geom_line(alpha = 0.9) +
    facet_wrap(~state_name, ncol = 6) +
    scale_color_manual(values = source_colors[c("FLS", s)]) +
    scale_linewidth_manual(
      values = c(`FALSE` = 0.5, `TRUE` = 0.85),
      guide = "none"
    ) +
    scale_y_continuous(labels = scales::percent_format(accuracy = 1)) +
    coord_cartesian(ylim = c(-state_change_ymax, state_change_ymax)) +
    labs(
      x = NULL,
      y = "Annual log change",
      color = NULL,
      title = paste("FLS and", s, "Wage Growth by State")
    ) +
    theme_minimal(base_size = 9) +
    theme(legend.position = "bottom")

  ggsave(
    path_figures(paste0("iv_proxy_fls_state_log_changes_", tolower(s), ".png")),
    fig_state_change,
    width = 13,
    height = 11,
    dpi = 200
  )
}

cat("Saved annual-change figures for ACS, OEWS, and QCEW.\n")
