# Purpose: Plot national H-2A certified workers.
# Output: outputs/figures/fig_line_ts_h2a_workers_certified.png.

here::i_am("code/paths.R")
source(here::here("code", "paths.R"))
library(arrow)
library(dplyr)
library(ggplot2)
library(ggthemes)

h2a_time_series <- read_parquet(path_int("h2a_aggregated.parquet")) %>%
  filter(year > 2007L, year <= 2022L) %>%
  group_by(year) %>%
  summarise(
    h2a_nbr_workers_certified = sum(
      nbr_workers_certified_start_year,
      na.rm = TRUE
    ),
    .groups = "drop"
  )

figure <- ggplot(
  h2a_time_series,
  aes(x = year, y = h2a_nbr_workers_certified / 1000)
) +
  geom_line(color = "#4393c3", linewidth = 1.5) +
  scale_y_continuous(
    breaks = seq(0, 400, by = 50),
    limits = c(0, 400)
  ) +
  labs(
    x = "Year",
    y = "H-2A Workers Certified (Thousands)"
  ) +
  theme_clean()

ggsave(
  path_figures("fig_line_ts_h2a_workers_certified.png"),
  figure,
  device = "png"
)
