# Purpose: Cluster target units and rank dissimilar donor clusters.
# Inputs: panel_iv_county_features.parquet and processed/county_year_panel.parquet.
# Outputs: CZ features, cluster assignments, diagnostics, donor pairs, and the
# primary k = 5 map.
# Run after: 07_build_cz_features.R and code/c02_build/04_finalize_county_panel.R.

here::i_am("code/paths.R")
source(here::here("code", "paths.R"))
source(path_code("c00_shared", "geography.R"))
source(path_code("designs", "panel_iv", "design.R"))
library(arrow)
library(dplyr)
library(ggplot2)
library(sf)
library(stringr)
library(tibble)
library(tidyr)

county_features <- read_parquet(path_int("panel_iv_county_features.parquet"))
county_feature_names <- setdiff(names(county_features), "county_fips")
county_df <- read_parquet(
  path_processed("county_year_panel.parquet")
) |>
  mutate(
    panel_iv_target_unit_id = make_panel_iv_target_unit_id(
      cz_id,
      aewr_region_id
    )
  )

soil_vars <- c(
  "slope_r",
  "slopegradwta",
  "resdept_r",
  "aws025wta",
  "aws050wta",
  "aws0100wta",
  "aws0150wta",
  "wtdepannmin",
  "wtdepaprjunmin",
  "brockdepmin",
  "cropprodindex"
)

unit_xwalk <- county_df |>
  distinct(county_fips, cz_id, aewr_region_id, panel_iv_target_unit_id)

county_feature_weights <- county_df |>
  filter(
    year >= DISSIMILARITY_IV_FEATURE_START_YEAR,
    year <= DISSIMILARITY_IV_FEATURE_END_YEAR
  ) |>
  group_by(county_fips, cz_id, aewr_region_id, panel_iv_target_unit_id) |>
  summarise(feature_weight = mean(emp_farm, na.rm = TRUE), .groups = "drop") |>
  mutate(
    feature_weight = if_else(
      is.nan(feature_weight) | is.na(feature_weight) | feature_weight <= 0,
      1,
      feature_weight
    )
  )

unit_features <- unit_xwalk |>
  left_join(
    county_feature_weights,
    by = c("county_fips", "cz_id", "aewr_region_id", "panel_iv_target_unit_id")
  ) |>
  mutate(feature_weight = replace_na(feature_weight, 1)) |>
  left_join(county_features, by = "county_fips") |>
  group_by(cz_id, aewr_region_id, panel_iv_target_unit_id) |>
  summarise(
    unit_feature_weight = sum(feature_weight, na.rm = TRUE),
    across(
      all_of(county_feature_names),
      ~ weighted.mean(.x, w = feature_weight, na.rm = TRUE)
    ),
    .groups = "drop"
  )

feature_names <- setdiff(
  names(unit_features),
  c("cz_id", "aewr_region_id", "panel_iv_target_unit_id", "unit_feature_weight")
)

share_feature_names <- feature_names[
  str_detect(feature_names, "^share_cdl_|^share_soil_")
]

unit_features <- unit_features |>
  mutate(
    across(
      all_of(share_feature_names),
      # For crop/soil shares, Euclidean distance on sqrt(shares) is the
      # Hellinger distance for compositional variables. It keeps large-acreage
      # crops important while reducing dominance by a few very large shares.
      ~ sqrt(pmax(.x, 0))
    )
  )

# Retain the exact feature representation supplied to the clustering routine:
# compositional shares use their square-root transform, while continuous
# climate and soil variables remain in levels. Standardization still occurs
# separately within each AEWR region below.
write_parquet(
  unit_features,
  path_int("panel_iv_target_unit_features.parquet")
)

# Rescale each feature block to roughly equal weight
feature_blocks <- list(
  crops = feature_names[str_detect(feature_names, "^share_cdl_")],
  climate = feature_names[str_detect(feature_names, "^normal_cb_")],
  soil_continuous = intersect(feature_names, soil_vars),
  soil_categorical = feature_names[str_detect(feature_names, "^share_soil_")]
)

iv_k_values <- DISSIMILARITY_IV_K_VALUES

cluster_list <- list()
cluster_diagnostic_list <- list()
donor_cluster_list <- list()
for (r in sort(unique(unit_features$aewr_region_id))) {
  d <- unit_features |> filter(aewr_region_id == r)

  for (v in feature_names) {
    x <- d[[v]]
    x[is.nan(x)] <- NA_real_
    med <- median(x, na.rm = TRUE)
    if (is.na(med)) {
      med <- 0
    }
    x[is.na(x)] <- med
    sx <- sd(x)
    d[[v]] <- if (!is.na(sx) && sx > 0) {
      (x - mean(x)) / sx
    } else {
      0
    }
  }

  for (block_name in names(feature_blocks)) {
    block_cols <- feature_blocks[[block_name]]
    if (length(block_cols) > 0) {
      d[block_cols] <- d[block_cols] / sqrt(length(block_cols))
    }
  }

  x <- as.matrix(d[, feature_names])
  hclust_fit <- hclust(dist(x), method = "ward.D2")

  for (iv_k in iv_k_values) {
    list_key <- paste(r, iv_k, sep = "_")
    selected_cluster <- cutree(hclust_fit, k = iv_k)
    donor_cluster_counts <- DISSIMILARITY_IV_PRIMARY_DONOR_COUNT

    cluster_list[[list_key]] <- d |>
      select(panel_iv_target_unit_id, aewr_region_id) |>
      mutate(iv_k = iv_k, iv_cluster = selected_cluster)

    cluster_diagnostic_list[[list_key]] <- tibble(
      aewr_region_id = r,
      iv_k = iv_k,
      iv_cluster = selected_cluster,
      unit_feature_weight = d$unit_feature_weight
    ) |>
      group_by(aewr_region_id, iv_k, iv_cluster) |>
      summarise(
        cluster_units = n(),
        cluster_feature_weight = sum(unit_feature_weight, na.rm = TRUE),
        .groups = "drop"
      ) |>
      mutate(
        cluster_feature_weight_share = cluster_feature_weight /
          sum(cluster_feature_weight, na.rm = TRUE)
      )

    cluster_centroids <- as_tibble(x) |>
      mutate(iv_cluster = selected_cluster) |>
      group_by(iv_cluster) |>
      summarise(
        across(all_of(feature_names), ~ mean(.x)),
        .groups = "drop"
      )

    centroid_matrix <- as.matrix(cluster_centroids[, feature_names])
    rownames(centroid_matrix) <- cluster_centroids$iv_cluster
    centroid_distance <- as.matrix(dist(centroid_matrix))
    cluster_ids <- sort(unique(selected_cluster))
    donor_pair_list <- list()

    for (target_cluster_id in cluster_ids) {
      donor_pair_list[[as.character(target_cluster_id)]] <- tibble(
        aewr_region_id = r,
        iv_k = iv_k,
        target_cluster = target_cluster_id,
        donor_cluster = cluster_ids,
        donor_cluster_distance = centroid_distance[
          as.character(target_cluster_id),
          as.character(cluster_ids)
        ]
      ) |>
        filter(donor_cluster != target_cluster) |>
        arrange(desc(donor_cluster_distance), donor_cluster) |>
        mutate(donor_rank = row_number()) |>
        crossing(
          donor_cluster_count = donor_cluster_counts
        ) |>
        filter(donor_rank <= donor_cluster_count)
    }

    donor_cluster_list[[list_key]] <- bind_rows(donor_pair_list)
  }
}

iv_clusters <- bind_rows(cluster_list)
write_parquet(iv_clusters, path_int("panel_iv_target_clusters.parquet"))
iv_cluster_diagnostics <- bind_rows(cluster_diagnostic_list) |>
  group_by(aewr_region_id, iv_k) |>
  mutate(
    region_min_cluster_units = min(cluster_units),
    region_min_cluster_weight_share = min(cluster_feature_weight_share)
  ) |>
  ungroup()
iv_donor_clusters <- bind_rows(donor_cluster_list)
write_parquet(
  iv_cluster_diagnostics,
  path_int("panel_iv_cluster_diagnostics.parquet")
)
write_parquet(iv_donor_clusters, path_int("panel_iv_donor_clusters.parquet"))

# Map CZ x AEWR-region clusters ----------------------------------------------

aewr_region_labels <- county_df |>
  distinct(aewr_region_id, state_abbrev) |>
  filter(!is.na(aewr_region_id), !is.na(state_abbrev)) |>
  arrange(aewr_region_id, state_abbrev) |>
  group_by(aewr_region_id) |>
  summarise(
    aewr_region_states = paste(state_abbrev, collapse = ", "),
    .groups = "drop"
  ) |>
  mutate(
    aewr_region_label = paste0(
      "AEWR Region ",
      aewr_region_id,
      " (",
      aewr_region_states,
      ")"
    )
  )

iv_k_max <- max(iv_k_values)
iv_cluster_levels <- paste0("Cluster ", seq_len(iv_k_max))
iv_cluster_base_colors <- c(
  "#1b9e77",
  "#d95f02",
  "#7570b3",
  "#e7298a",
  "#66a61e",
  "#e6ab02"
)
if (iv_k_max > length(iv_cluster_base_colors)) {
  iv_cluster_base_colors <- grDevices::colorRampPalette(
    iv_cluster_base_colors
  )(iv_k_max)
}
iv_cluster_colors <- setNames(
  iv_cluster_base_colors[seq_len(iv_k_max)],
  iv_cluster_levels
)

county_iv_clusters <- county_df |>
  distinct(county_fips, cz_id, aewr_region_id, panel_iv_target_unit_id) |>
  # Analysis data use the pre-2015 Shannon County code; the bundled 2020
  # TIGER shapefile uses the newer Oglala Lakota County code.
  mutate(county_fips = recode(county_fips, `46113` = "46102")) |>
  left_join(
    iv_clusters,
    by = c("panel_iv_target_unit_id", "aewr_region_id"),
    relationship = "many-to-many"
  ) |>
  left_join(aewr_region_labels, by = "aewr_region_id") |>
  mutate(
    iv_cluster_id = iv_cluster,
    iv_cluster = factor(
      if_else(
        is.na(iv_cluster_id),
        NA_character_,
        paste0("Cluster ", iv_cluster_id)
      ),
      levels = iv_cluster_levels
    ),
    aewr_region_label = factor(
      aewr_region_label,
      levels = aewr_region_labels$aewr_region_label
    )
  )


county_shape_zip <- path_raw("county_shapefile", "tl_2020_us_county.zip")
unzip(county_shape_zip, exdir = tempdir())
county_map_iv_clusters <- sf::st_read(
  file.path(tempdir(), "tl_2020_us_county.shp"),
  quiet = TRUE
) |>
  mutate(
    state_fips = state_fips(STATEFP),
    county_fips = combine_county_fips(STATEFP, COUNTYFP)
  ) |>
  filter(
    as.integer(state_fips) <= 56,
    !state_fips %in% c("02", "15")
  ) |>
  sf::st_make_valid() |>
  sf::st_transform(5070) |>
  left_join(county_iv_clusters, by = "county_fips") |>
  filter(!is.na(aewr_region_id))

cz_aewr_cluster_boundaries <- county_map_iv_clusters |>
  group_by(iv_k, panel_iv_target_unit_id, aewr_region_id, aewr_region_label) |>
  summarise(geometry = sf::st_union(geometry), .groups = "drop")

aewr_region_boundaries <- county_map_iv_clusters |>
  group_by(iv_k, aewr_region_id, aewr_region_label) |>
  summarise(geometry = sf::st_union(geometry), .groups = "drop")

iv_cluster_map_theme <- theme_void(base_size = 10) +
  theme(
    legend.position = "bottom",
    legend.title = element_text(size = 9),
    legend.text = element_text(size = 9),
    plot.title = element_text(face = "bold", hjust = 0),
    plot.subtitle = element_text(hjust = 0),
    plot.caption = element_text(size = 8, hjust = 0),
    plot.margin = margin(8, 8, 8, 8),
    plot.background = element_rect(fill = "white", color = NA),
    panel.background = element_rect(fill = "white", color = NA),
    legend.background = element_rect(fill = "white", color = NA)
  )

for (iv_k in DISSIMILARITY_IV_PRIMARY_K) {
  k_cluster_levels <- paste0("Cluster ", seq_len(iv_k))
  k_map_data <- county_map_iv_clusters |>
    filter(iv_k == .env$iv_k) |>
    mutate(
      iv_cluster = factor(
        as.character(iv_cluster),
        levels = k_cluster_levels
      )
    )
  k_cz_boundaries <- cz_aewr_cluster_boundaries |>
    filter(iv_k == .env$iv_k)
  k_region_boundaries <- aewr_region_boundaries |>
    filter(iv_k == .env$iv_k)

  iv_cluster_map_all <- ggplot() +
    geom_sf(
      data = k_map_data,
      aes(fill = iv_cluster),
      color = scales::alpha("white", 0.35),
      linewidth = 0.03
    ) +
    geom_sf(
      data = k_cz_boundaries,
      fill = NA,
      color = scales::alpha("grey20", 0.65),
      linewidth = 0.12
    ) +
    geom_sf(
      data = k_region_boundaries,
      fill = NA,
      color = "white",
      linewidth = 1
    ) +
    geom_sf(
      data = k_region_boundaries,
      fill = NA,
      color = "#5e3c99",
      linewidth = 0.45
    ) +
    scale_fill_manual(
      values = iv_cluster_colors,
      drop = TRUE,
      name = "IV cluster"
    ) +
    coord_sf(datum = NA) +
    labs(
      title = "Dissimilarity IV Clusters within AEWR Regions",
      subtitle = paste0(
        "County shading shows each CZ x AEWR-region unit's cluster, k = ",
        iv_k,
        "; purple outlines are AEWR regions and grey outlines are CZ-region units."
      ),
      x = NULL,
      y = NULL
    ) +
    iv_cluster_map_theme

  ggsave(
    filename = path_figures("fig_iv_dissimilarity_clusters_k5.png"),
    iv_cluster_map_all,
    width = 12,
    height = 8,
    dpi = 300,
    device = "png",
    bg = "white"
  )
}
