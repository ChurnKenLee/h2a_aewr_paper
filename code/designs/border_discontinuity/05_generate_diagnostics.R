# Purpose: Retain AEWR-border ring diagnostics, maps, and audit plots.

here::i_am("code/paths.R")
source(here::here("code", "paths.R"))
source(path_code("c00_shared", "geography.R"))
library(arrow)
library(dplyr)
library(ggplot2)
library(readr)
library(sf)
library(stringr)
library(tidyr)

features <- read_parquet(
  path_int("border_discontinuity_county_features.parquet")
)
memberships <- read_parquet(
  path_int("border_discontinuity_geography_memberships.parquet")
) %>%
  as_tibble()
geography_diagnostics <- read_parquet(
  path_int("border_discontinuity_geography_diagnostics.parquet")
) %>%
  as_tibble()
exclusions <- read_parquet(
  path_int("border_discontinuity_assignment_exclusions.parquet")
) %>%
  as_tibble()
agricultural_county_pair_audit <- read_parquet(
  path_int("border_discontinuity_agricultural_border_audit.parquet")
) %>%
  as_tibble()
estimates <- readr::read_csv(
  path_tables("border_discontinuity_estimates.csv"),
  show_col_types = FALSE
)
annual_coefficients <- readr::read_csv(
  path_tables("border_discontinuity_ring_annual_coefficients.csv"),
  show_col_types = FALSE
)
leave_one_out <- readr::read_csv(
  path_tables("border_discontinuity_ring_leave_one_border_pair_out.csv"),
  col_types = cols(excluded_border_pair_id = col_character())
)
border_pair_decomposition <- readr::read_csv(
  path_tables("border_discontinuity_ring_border_pair_decomposition.csv"),
  col_types = cols(region_pair_id = col_character())
)
region_decomposition <- readr::read_csv(
  path_tables("border_discontinuity_ring_region_decomposition.csv"),
  col_types = cols(aewr_region_id = col_character())
)
county_decomposition <- readr::read_csv(
  path_tables("border_discontinuity_ring_county_decomposition.csv"),
  col_types = cols(
    county_fips = col_character(),
    aewr_region_id = col_character()
  )
)
stratum_region_year_decomposition <- readr::read_csv(
  path_tables(
    "border_discontinuity_ring_stratum_region_year_decomposition.csv"
  ),
  col_types = cols(aewr_region_id = col_character())
)

feature_diagnostics <- tibble(
  feature_start_year = unique(features$feature_start_year),
  feature_end_year = unique(features$feature_end_year),
  n_counties = nrow(features),
  n_raw_features = sum(str_starts(names(features), "raw__")),
  n_scaled_features = sum(str_starts(names(features), "scaled__")),
  complete_raw_feature_counties = sum(features$raw_feature_complete),
  median_raw_feature_missing_count = median(
    features$raw_feature_missing_count
  ),
  maximum_raw_feature_missing_count = max(
    features$raw_feature_missing_count
  ),
  n_raw_cross_border_county_pairs = nrow(agricultural_county_pair_audit),
  n_eligible_cross_border_county_pairs = sum(
    agricultural_county_pair_audit$county_pair_eligible
  ),
  n_raw_aewr_border_pairs = n_distinct(
    agricultural_county_pair_audit$region_pair_id
  ),
  n_aewr_border_pairs_with_eligible_county_pair = n_distinct(
    agricultural_county_pair_audit$region_pair_id[
      agricultural_county_pair_audit$county_pair_eligible
    ]
  )
)

exclusion_summary <- exclusions %>%
  count(spec_id, reason, name = "excluded_memberships") %>%
  left_join(
    exclusions %>%
      group_by(spec_id, reason) %>%
      summarise(excluded_counties = n_distinct(county_fips), .groups = "drop"),
    by = c("spec_id", "reason"),
    relationship = "one-to-one"
  )

readr::write_csv(
  geography_diagnostics,
  path_tables("border_discontinuity_geography_diagnostics.csv")
)
readr::write_csv(
  exclusion_summary,
  path_tables("border_discontinuity_exclusion_summary.csv")
)
readr::write_csv(
  feature_diagnostics,
  path_tables("border_discontinuity_feature_diagnostics.csv")
)
readr::write_csv(
  agricultural_county_pair_audit,
  path_tables("border_discontinuity_agricultural_county_pair_audit.csv")
)

model_labels <- c(
  ring_h0_disjoint__base = "Border core",
  ring_h1_disjoint__base = "One ring (v5 benchmark)",
  ring_h2_disjoint__base = "Two rings",
  ring_h1_stacked__base = "One ring, stacked",
  ring_h1_disjoint__equal_stratum_region = "One ring, equal sides",
  ring_h1_disjoint__baseline_trends = "One ring, baseline trends"
)
plot_data <- estimates %>%
  filter(period == "policy") %>%
  mutate(
    model_label = unname(model_labels[model_id]),
    model_label = factor(model_label, levels = rev(unname(model_labels)))
  )
if (any(is.na(plot_data$model_label))) {
  stop("Coefficient plot encountered an unregistered model.", call. = FALSE)
}

coefficient_plot <- ggplot(
  plot_data,
  aes(
    x = estimate,
    y = model_label,
    xmin = wcb_confidence_low,
    xmax = wcb_confidence_high
  )
) +
  geom_vline(xintercept = 0, color = "grey55", linewidth = 0.4) +
  geom_errorbar(orientation = "y", width = 0.15, linewidth = 0.6) +
  geom_point(size = 2, color = "#D55E00") +
  labs(
    title = "Direct real-AEWR estimates in border-ring samples",
    subtitle = "Intervals invert null-imposed wild bootstrap tests clustered by border pair",
    x = "Change in H-2A certified positions per 2011 farm worker\nper $1 increase in real hourly AEWR",
    y = NULL
  ) +
  theme_minimal(base_size = 11) +
  theme(panel.grid.major.y = element_blank())

ggsave(
  path_figures("border_discontinuity_estimates.png"),
  coefficient_plot,
  width = 9,
  height = 6,
  dpi = 300
)

primary_inference <- estimates %>%
  filter(model_id == "ring_h1_disjoint__base") %>%
  transmute(
    estimate,
    `Standard cluster-robust` = list(c(
      crv1_confidence_low,
      crv1_confidence_high
    )),
    `Two-way (pair + region)` = list(c(
      two_way_crv1_confidence_low,
      two_way_crv1_confidence_high
    )),
    `Wild cluster bootstrap` = list(c(
      wcb_confidence_low,
      wcb_confidence_high
    )),
    `Delete-one-pair jackknife` = list(c(
      crv3_jackknife_confidence_low,
      crv3_jackknife_confidence_high
    ))
  ) %>%
  tidyr::pivot_longer(
    cols = -estimate,
    names_to = "inference_method",
    values_to = "interval"
  ) %>%
  mutate(
    confidence_low = purrr::map_dbl(interval, 1L),
    confidence_high = purrr::map_dbl(interval, 2L),
    inference_method = factor(
      inference_method,
      levels = c(
        "Delete-one-pair jackknife",
        "Wild cluster bootstrap",
        "Two-way (pair + region)",
        "Standard cluster-robust"
      )
    )
  )
if (nrow(primary_inference) != 4L) {
  stop("Primary inference plot requires exactly four intervals.", call. = FALSE)
}
primary_inference_plot <- ggplot(
  primary_inference,
  aes(
    x = estimate,
    y = inference_method,
    xmin = confidence_low,
    xmax = confidence_high
  )
) +
  geom_vline(xintercept = 0, color = "grey55", linewidth = 0.4) +
  geom_errorbar(orientation = "y", width = 0.14, linewidth = 0.65) +
  geom_point(size = 2, color = "#D55E00") +
  labs(
    title = "V5 benchmark under pair and two-way inference",
    subtitle = paste(
      "V5 pair inference comparator; sensitivity also clusters by",
      "AEWR region"
    ),
    x = "Outcome change per $1 increase in real hourly AEWR",
    y = NULL
  ) +
  theme_minimal(base_size = 11) +
  theme(panel.grid.major.y = element_blank())

ggsave(
  path_figures("border_discontinuity_ring_primary_inference.png"),
  primary_inference_plot,
  width = 8,
  height = 4,
  dpi = 300
)

annual_plot <- ggplot(
  annual_coefficients,
  aes(
    x = year,
    y = estimate,
    ymin = wcb_confidence_low,
    ymax = wcb_confidence_high
  )
) +
  annotate(
    "rect",
    xmin = 2010.5,
    xmax = Inf,
    ymin = -Inf,
    ymax = Inf,
    fill = "grey92"
  ) +
  geom_hline(yintercept = 0, color = "grey50", linewidth = 0.4) +
  geom_errorbar(width = 0.18, linewidth = 0.45, color = "#D55E00") +
  geom_point(size = 1.8, color = "#D55E00") +
  scale_x_continuous(breaks = annual_coefficients$year) +
  labs(
    title = "Annual treatment coefficients for the v5 one-ring benchmark sample",
    subtitle = "Shading marks policy years; intervals use border-pair wild bootstrap tests",
    x = NULL,
    y = "Outcome change per $1 increase in real hourly AEWR"
  ) +
  theme_minimal(base_size = 11) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

ggsave(
  path_figures("border_discontinuity_ring_annual_coefficients.png"),
  annual_plot,
  width = 9,
  height = 5,
  dpi = 300
)

full_primary_estimate <- estimates %>%
  filter(model_id == "ring_h1_disjoint__base") %>%
  pull(estimate)
leave_one_out_plot <- leave_one_out %>%
  mutate(
    excluded_border_pair_id = factor(
      excluded_border_pair_id,
      levels = rev(sort(unique(excluded_border_pair_id)))
    )
  ) %>%
  ggplot(
    aes(
      x = estimate,
      y = excluded_border_pair_id,
      xmin = crv1_confidence_low,
      xmax = crv1_confidence_high
    )
  ) +
  geom_vline(
    xintercept = full_primary_estimate,
    color = "#0072B2",
    linewidth = 0.6
  ) +
  geom_vline(xintercept = 0, color = "grey55", linewidth = 0.4) +
  geom_errorbar(orientation = "y", width = 0.15, linewidth = 0.5) +
  geom_point(size = 1.8, color = "#D55E00") +
  labs(
    title = "Leave-one-border-pair-out estimates",
    subtitle = "Blue line is the full-sample estimate; intervals use standard cluster-robust inference",
    x = "Outcome change per $1 increase in real hourly AEWR",
    y = "Excluded AEWR border pair"
  ) +
  theme_minimal(base_size = 11) +
  theme(panel.grid.major.y = element_blank())

ggsave(
  path_figures("border_discontinuity_ring_leave_one_border_pair_out.png"),
  leave_one_out_plot,
  width = 8,
  height = 8,
  dpi = 300
)

border_pair_contribution_plot <- border_pair_decomposition %>%
  mutate(
    pair_label = paste0(
      "Pair ",
      region_pair_id,
      " (",
      state_abbreviations,
      ")"
    ),
    pair_label = factor(
      pair_label,
      levels = pair_label[order(coefficient_contribution)]
    )
  ) %>%
  ggplot(
    aes(
      x = coefficient_contribution,
      y = pair_label,
      fill = certified_position_share
    )
  ) +
  geom_vline(xintercept = 0, color = "grey55", linewidth = 0.4) +
  geom_col(width = 0.75) +
  scale_fill_viridis_c(
    labels = scales::label_percent(accuracy = 1),
    name = "Share of certified\npositions in sample"
  ) +
  labs(
    title = "Border-pair contributions to the v5 benchmark coefficient",
    subtitle = paste(
      "Bars sum to the fitted coefficient; color records program usage,",
      "not regression weight"
    ),
    x = "FWL coefficient contribution",
    y = NULL
  ) +
  theme_minimal(base_size = 9) +
  theme(panel.grid.major.y = element_blank(), legend.position = "bottom")

ggsave(
  path_figures("border_discontinuity_ring_border_pair_contributions.png"),
  border_pair_contribution_plot,
  width = 10,
  height = 9,
  dpi = 300
)

region_contribution_plot <- region_decomposition %>%
  mutate(
    region_label = paste0(
      "Region ",
      aewr_region_id,
      " (",
      state_abbreviations,
      ")"
    ),
    region_label = factor(
      region_label,
      levels = region_label[order(coefficient_contribution)]
    )
  ) %>%
  ggplot(
    aes(
      x = coefficient_contribution,
      y = region_label,
      fill = certified_position_share
    )
  ) +
  geom_vline(xintercept = 0, color = "grey55", linewidth = 0.4) +
  geom_col(width = 0.75) +
  scale_fill_viridis_c(
    labels = scales::label_percent(accuracy = 1),
    name = "Share of certified\npositions in sample"
  ) +
  labs(
    title = "AEWR-region contributions to the v5 benchmark coefficient",
    subtitle = paste(
      "Bars sum to the fitted coefficient; color records program usage,",
      "not regression weight"
    ),
    x = "FWL coefficient contribution",
    y = NULL
  ) +
  theme_minimal(base_size = 10) +
  theme(panel.grid.major.y = element_blank(), legend.position = "bottom")

ggsave(
  path_figures("border_discontinuity_ring_region_contributions.png"),
  region_contribution_plot,
  width = 9,
  height = 7,
  dpi = 300
)

county_contribution_plot <- county_decomposition %>%
  slice_max(
    order_by = absolute_aggregated_score_share,
    n = 20L,
    with_ties = FALSE
  ) %>%
  mutate(
    county_label = paste0(countyname, " [", region_pair_id, "]"),
    county_label = factor(
      county_label,
      levels = county_label[order(coefficient_contribution)]
    ),
    pair_group = if_else(
      region_pair_id == "09_10",
      "AEWR border pair 09_10",
      "Other border pair"
    )
  ) %>%
  ggplot(
    aes(
      x = coefficient_contribution,
      y = county_label,
      fill = pair_group
    )
  ) +
  geom_vline(xintercept = 0, color = "grey55", linewidth = 0.4) +
  geom_col(width = 0.75) +
  scale_fill_manual(
    values = c(
      "AEWR border pair 09_10" = "#D55E00",
      "Other border pair" = "grey60"
    ),
    name = NULL
  ) +
  labs(
    title = "County contributions among the largest residual-score shares",
    subtitle = paste(
      "Twenty counties selected by absolute aggregated score share;",
      "brackets give AEWR border pair"
    ),
    x = "FWL coefficient contribution",
    y = NULL
  ) +
  theme_minimal(base_size = 9) +
  theme(panel.grid.major.y = element_blank(), legend.position = "bottom")

ggsave(
  path_figures("border_discontinuity_ring_county_contributions.png"),
  county_contribution_plot,
  width = 9,
  height = 7,
  dpi = 300
)

stratum_year_contribution <- stratum_region_year_decomposition %>%
  group_by(stratum_id, year) %>%
  summarise(
    coefficient_contribution = sum(coefficient_contribution),
    .groups = "drop"
  ) %>%
  group_by(stratum_id) %>%
  mutate(total_absolute_contribution = sum(abs(coefficient_contribution))) %>%
  ungroup() %>%
  mutate(
    stratum_label = str_remove(stratum_id, "^ring_h1_disjoint__"),
    stratum_label = factor(
      stratum_label,
      levels = unique(
        stratum_label[order(total_absolute_contribution)]
      )
    )
  )
contribution_limit <- max(abs(
  stratum_year_contribution$coefficient_contribution
))
stratum_year_plot <- ggplot(
  stratum_year_contribution,
  aes(x = year, y = stratum_label, fill = coefficient_contribution)
) +
  geom_tile(color = "white", linewidth = 0.1) +
  scale_fill_gradient2(
    low = "#0072B2",
    mid = "white",
    high = "#D55E00",
    midpoint = 0,
    limits = c(-contribution_limit, contribution_limit),
    name = "Coefficient\ncontribution"
  ) +
  scale_x_continuous(breaks = seq.int(2011, 2022)) +
  labs(
    title = "Border-stratum contributions by policy year",
    subtitle = "Each cell sums the two AEWR-region sides of a retained component",
    x = NULL,
    y = "Border pair and connected component"
  ) +
  theme_minimal(base_size = 9) +
  theme(
    panel.grid = element_blank(),
    axis.text.x = element_text(angle = 45, hjust = 1),
    legend.position = "bottom"
  )

ggsave(
  path_figures("border_discontinuity_ring_stratum_year_contributions.png"),
  stratum_year_plot,
  width = 10,
  height = 9,
  dpi = 300
)

read_county_map <- function(county_zip) {
  shape_dir <- tempfile("h2a-border-counties-")
  dir.create(shape_dir)
  on.exit(unlink(shape_dir, recursive = TRUE), add = TRUE)
  utils::unzip(county_zip, exdir = shape_dir)
  sf::st_read(
    file.path(shape_dir, "tl_2020_us_county.shp"),
    quiet = TRUE
  ) %>%
    mutate(
      state_fips = state_fips(STATEFP),
      county_fips = harmonize_county_fips_2010(
        combine_county_fips(STATEFP, COUNTYFP)
      )
    ) %>%
    filter(
      as.integer(state_fips) <= 56,
      !state_fips %in% c("02", "15")
    )
}

county_map <- read_county_map(
  path_raw("county_shapefile", "tl_2020_us_county.zip")
)
region_crosswalk <- read_parquet(
  path_processed("border_discontinuity_panel.parquet"),
  col_select = c("county_fips", "aewr_region_id")
) %>%
  distinct()
aewr_boundaries <- county_map %>%
  inner_join(region_crosswalk, by = "county_fips", relationship = "one-to-one") %>%
  group_by(aewr_region_id) %>%
  summarise(
    geometry = sf::st_union(geometry),
    .groups = "drop"
  )

county_border_color <- "grey65"
county_border_linewidth <- 0.08
aewr_border_halo_color <- "white"
aewr_border_halo_linewidth <- 0.9
aewr_border_color <- "grey10"
aewr_border_linewidth <- 0.45

ring_map_data <- memberships %>%
  filter(spec_id %in% c(
    "ring_h0_disjoint",
    "ring_h1_disjoint",
    "ring_h2_disjoint"
  )) %>%
  distinct(spec_id, county_fips, ring_distance) %>%
  mutate(
    specification = factor(
      spec_id,
      levels = c(
        "ring_h0_disjoint",
        "ring_h1_disjoint",
        "ring_h2_disjoint"
      ),
      labels = c("Border core", "One ring", "Two rings")
    ),
    ring_distance = factor(ring_distance)
  ) %>%
  inner_join(county_map, by = "county_fips", relationship = "many-to-one") %>%
  st_as_sf()

ring_map <- ggplot() +
  geom_sf(data = county_map, fill = "grey96", color = NA) +
  geom_sf(
    data = ring_map_data,
    aes(fill = ring_distance),
    color = NA
  ) +
  geom_sf(
    data = county_map,
    fill = NA,
    color = county_border_color,
    linewidth = county_border_linewidth
  ) +
  geom_sf(
    data = aewr_boundaries,
    fill = NA,
    color = aewr_border_halo_color,
    linewidth = aewr_border_halo_linewidth
  ) +
  geom_sf(
    data = aewr_boundaries,
    fill = NA,
    color = aewr_border_color,
    linewidth = aewr_border_linewidth
  ) +
  facet_wrap(~ specification, ncol = 1) +
  scale_fill_brewer(palette = "YlOrRd", name = "Adjacency ring") +
  coord_sf(datum = NA) +
  labs(
    title = "Retained disjoint AEWR-border windows",
    subtitle = "Thin grey lines are counties; outlined dark lines are AEWR regions"
  ) +
  theme_void() +
  theme(legend.position = "bottom", strip.text = element_text(face = "bold"))

ggsave(
  path_figures("border_discontinuity_ring_windows.png"),
  ring_map,
  width = 8,
  height = 10,
  dpi = 300,
  bg = "white"
)

message("Wrote AEWR-border ring diagnostic tables and figures.")
