# Purpose: Estimate and plot the flexible DD event-study specifications.
# Input: data/processed/county_df_analysis_year.parquet.
# Outputs: table_2_event_study.tex and event-study coefficient figures.
# Run after: code/c02_build/04_finalize_county_panel.R.

source(if (file.exists(file.path("code", "bootstrap_paths.R"))) {
  file.path("code", "bootstrap_paths.R")
} else {
  file.path("..", "bootstrap_paths.R")
})
source(path_code("c00_shared", "analysis_helpers.R"))
library(arrow)
library(tidyverse)
library(fixest)
library(ggfixest)

county_df <- read_parquet(path_processed("county_df_analysis_year.parquet"))
samp_base <- analysis_sample(county_df)
samp_no_border <- samp_base %>% filter(border_cz == 0)

## Exhibit 14: Event Study (Flexible DD, Base Year = 2011) --------------------

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
es_1 <- feols(es_fml, data = samp_base, vcov = ~cz_aewr_region_fe)

# Event study model 2: with controls, all CZs
es_2 <- feols(es_fml_ctrl, data = samp_base, vcov = ~cz_aewr_region_fe)

# Event study model 3: no controls, no border CZs
es_3 <- feols(es_fml, data = samp_no_border, vcov = ~cz_aewr_region_fe)

# Event study model 4: with controls, no border CZs
es_4 <- feols(es_fml_ctrl, data = samp_no_border, vcov = ~cz_aewr_region_fe)

# Wald test on post-2011 interactions (joint significance)
post_terms <- grep(
  "aewr_cz_p25_l1:yeardummy_201[2-9]|aewr_cz_p25_l1:yeardummy_202",
  names(coef(es_1)),
  value = TRUE
)
wald(es_1, keep = post_terms)
wald(es_2, keep = post_terms)

es_dict <- c(
  "aewr_cz_p25_l1" = "Lagged AEWR vs 25th pct wage gap",
  "yeardummy_2008" = "2008",
  "yeardummy_2009" = "2009",
  "yeardummy_2010" = "2010",
  "yeardummy_2012" = "2012",
  "yeardummy_2013" = "2013",
  "yeardummy_2014" = "2014",
  "yeardummy_2015" = "2015",
  "yeardummy_2016" = "2016",
  "yeardummy_2017" = "2017",
  "yeardummy_2018" = "2018",
  "yeardummy_2019" = "2019",
  "yeardummy_2020" = "2020",
  "yeardummy_2021" = "2021",
  "yeardummy_2022" = "2022"
)

table_2 <- etable(
  es_1,
  es_2,
  es_3,
  es_4,
  tex = TRUE,
  title = "Event Study Coefficients (Base Year = 2011)",
  keep = "%aewr_cz_p25_l1:yeardummy",
  headers = c(
    "No Controls",
    "Controls",
    "No Border, No Controls",
    "No Border, Controls"
  ),
  dict = es_dict,
  signif.code = c("***" = 0.01, "**" = 0.05, "*" = 0.10),
  file = path_tables("table_2_event_study.tex"),
  replace = TRUE
)

## Exhibit 15: Event Study Coefficient Plots -----------------------------------

make_coefplot <- function(model) {
  coef_names <- grep(
    "aewr_cz_p25_l1:yeardummy_",
    names(coef(model)),
    value = TRUE
  )
  ct <- model$coeftable[coef_names, , drop = FALSE]

  years_in_model <- as.integer(sub(".*yeardummy_", "", rownames(ct)))

  all_years <- 2008:2022
  coeff_df <- data.frame(year = all_years) %>%
    mutate(
      beta = ifelse(
        year == 2011,
        0,
        ct[match(paste0("aewr_cz_p25_l1:yeardummy_", year), rownames(ct)), 1]
      ),
      se = ifelse(
        year == 2011,
        0,
        ct[match(paste0("aewr_cz_p25_l1:yeardummy_", year), rownames(ct)), 2]
      )
    ) %>%
    mutate(
      upper_ci = beta + 1.96 * se,
      lower_ci = beta - 1.96 * se
    )

  ggplot(coeff_df, aes(x = year)) +
    geom_hline(yintercept = 0, color = "grey40") +
    geom_vline(xintercept = 2011, linetype = "dashed", color = "grey40") +
    geom_ribbon(
      aes(ymin = lower_ci, ymax = upper_ci),
      alpha = 0.2,
      fill = "steelblue"
    ) +
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

ggsave(
  p_es_1,
  filename = path_figures("coefplot_dd_no_controls.png"),
  width = 8,
  height = 5,
  device = "png"
)

ggsave(
  p_es_2,
  filename = path_figures("coefplot_dd_controls.png"),
  width = 8,
  height = 5,
  device = "png"
)

ggsave(
  p_es_3,
  filename = path_figures("coefplot_dd_no_border_no_controls.png"),
  width = 8,
  height = 5,
  device = "png"
)

ggsave(
  p_es_4,
  filename = path_figures("coefplot_dd_no_border_controls.png"),
  width = 8,
  height = 5,
  device = "png"
)
