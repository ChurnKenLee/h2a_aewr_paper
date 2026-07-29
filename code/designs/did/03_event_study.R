# Purpose: Estimate the four retained DiD event studies with 2011 omitted.
# Outputs: table_2_event_study.tex and coefplot_dd_controls.png.

here::i_am("code/paths.R")
source(here::here("code", "paths.R"))
source(path_code("designs", "did", "helpers.R"))
library(arrow)
library(dplyr)
library(fixest)
library(ggplot2)
library(ggthemes)
library(tibble)

sample_full <- read_parquet(
  path_processed("did_county_year_panel.parquet")
) %>%
  did_sample()
sample_no_border <- sample_full %>%
  filter(!border_cz)

models <- list(
  did_event_model(sample_full),
  did_event_model(sample_full, controls = TRUE),
  did_event_model(sample_no_border),
  did_event_model(sample_no_border, controls = TRUE)
)

etable(
  models,
  tex = TRUE,
  title = "Event Study Coefficients (Base Year = 2011)",
  keep_raw = "year::.*:aewr_cz_p25_l1",
  headers = did_table_headers,
  dict = c(
    did_table_dictionary,
    "year::2008:aewr_cz_p25_l1" = "Lagged AEWR vs 25th pct wage gap $\\times$ 2008",
    "year::2009:aewr_cz_p25_l1" = "Lagged AEWR vs 25th pct wage gap $\\times$ 2009",
    "year::2010:aewr_cz_p25_l1" = "Lagged AEWR vs 25th pct wage gap $\\times$ 2010",
    "year::2012:aewr_cz_p25_l1" = "Lagged AEWR vs 25th pct wage gap $\\times$ 2012",
    "year::2013:aewr_cz_p25_l1" = "Lagged AEWR vs 25th pct wage gap $\\times$ 2013",
    "year::2014:aewr_cz_p25_l1" = "Lagged AEWR vs 25th pct wage gap $\\times$ 2014",
    "year::2015:aewr_cz_p25_l1" = "Lagged AEWR vs 25th pct wage gap $\\times$ 2015",
    "year::2016:aewr_cz_p25_l1" = "Lagged AEWR vs 25th pct wage gap $\\times$ 2016",
    "year::2017:aewr_cz_p25_l1" = "Lagged AEWR vs 25th pct wage gap $\\times$ 2017",
    "year::2018:aewr_cz_p25_l1" = "Lagged AEWR vs 25th pct wage gap $\\times$ 2018",
    "year::2019:aewr_cz_p25_l1" = "Lagged AEWR vs 25th pct wage gap $\\times$ 2019",
    "year::2020:aewr_cz_p25_l1" = "Lagged AEWR vs 25th pct wage gap $\\times$ 2020",
    "year::2021:aewr_cz_p25_l1" = "Lagged AEWR vs 25th pct wage gap $\\times$ 2021",
    "year::2022:aewr_cz_p25_l1" = "Lagged AEWR vs 25th pct wage gap $\\times$ 2022"
  ),
  signif.code = c("***" = 0.01, "**" = 0.05, "*" = 0.10),
  file = path_tables("table_2_event_study.tex"),
  replace = TRUE
)

coefficient_table <- as.data.frame(coeftable(models[[2]])) %>%
  rownames_to_column("term") %>%
  filter(grepl("^year::[0-9]{4}:aewr_cz_p25_l1$", term)) %>%
  transmute(
    year = as.integer(sub("^year::([0-9]{4}).*$", "\\1", term)),
    estimate = Estimate,
    standard_error = `Std. Error`
  ) %>%
  bind_rows(
    tibble(year = 2011L, estimate = 0, standard_error = 0)
  ) %>%
  arrange(year) %>%
  mutate(
    lower = estimate - 1.96 * standard_error,
    upper = estimate + 1.96 * standard_error
  )

coefficient_plot <- ggplot(
  coefficient_table,
  aes(x = year, y = estimate)
) +
  geom_hline(yintercept = 0, color = "grey40") +
  geom_vline(
    xintercept = 2011,
    linetype = "dashed",
    color = "grey40"
  ) +
  geom_ribbon(
    aes(ymin = lower, ymax = upper),
    alpha = 0.2,
    fill = "steelblue"
  ) +
  geom_line(color = "steelblue", linewidth = 1.2) +
  geom_point(color = "steelblue", size = 2) +
  labs(
    x = "Year",
    y = "Coefficient on lagged AEWR vs 25th pct wage gap"
  ) +
  theme_clean()

ggsave(
  path_figures("coefplot_dd_controls.png"),
  coefficient_plot,
  width = 8,
  height = 5,
  device = "png"
)
