# Purpose: Cluster CZ-by-AEWR-region units and rank dissimilar donor clusters.
# Inputs: iv_county_features.parquet and processed/county_df_analysis_year.parquet.
# Outputs: cluster assignments, diagnostics, donor pairs, and cluster maps.
# Run after: 07_build_cz_features.R and code/c02_build/04_finalize_county_panel.R.

source(
  if (file.exists(file.path("code", "bootstrap_paths.R"))) {
    file.path("code", "bootstrap_paths.R")
  } else {
    file.path("..", "bootstrap_paths.R")
  }
)
source(path_code("c00_shared", "fips.R"))
library(arrow)
library(tidyverse)
library(sf)

cat("Reading IV clustering inputs\n")
county_features <- read_parquet(path_int("iv_county_features.parquet"))
county_feature_names <- setdiff(names(county_features), "county_ansi")
county_df <- read_parquet(
  path_processed("county_df_analysis_year.parquet")
) %>%
  janitor::clean_names()

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

unit_xwalk <- county_df %>%
  mutate(county_ansi = countyfips) %>%
  distinct(county_ansi, cz_out10, aewr_region_num, cz_aewr_region_fe)

county_feature_weights <- county_df %>%
  mutate(county_ansi = countyfips) %>%
  filter(year >= 2008, year <= 2011) %>%
  group_by(county_ansi, cz_out10, aewr_region_num, cz_aewr_region_fe) %>%
  summarise(feature_weight = mean(emp_farm, na.rm = TRUE), .groups = "drop") %>%
  mutate(
    feature_weight = if_else(
      is.nan(feature_weight) | is.na(feature_weight) | feature_weight <= 0,
      1,
      feature_weight
    )
  )

unit_features <- unit_xwalk %>%
  left_join(
    county_feature_weights,
    by = c("county_ansi", "cz_out10", "aewr_region_num", "cz_aewr_region_fe")
  ) %>%
  mutate(feature_weight = replace_na(feature_weight, 1)) %>%
  left_join(county_features, by = "county_ansi") %>%
  group_by(cz_out10, aewr_region_num, cz_aewr_region_fe) %>%
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
  c("cz_out10", "aewr_region_num", "cz_aewr_region_fe", "unit_feature_weight")
)

share_feature_names <- feature_names[
  str_detect(feature_names, "^share_cdl_|^share_soil_")
]

unit_features <- unit_features %>%
  mutate(
    across(
      all_of(share_feature_names),
      # For crop/soil shares, Euclidean distance on sqrt(shares) is the
      # Hellinger distance for compositional variables. It keeps large-acreage
      # crops important while reducing dominance by a few very large shares.
      ~ sqrt(pmax(.x, 0))
    )
  )

# Rescale each feature block to roughly equal weight
feature_blocks <- list(
  crops = feature_names[str_detect(feature_names, "^share_cdl_")],
  climate = feature_names[str_detect(feature_names, "^normal_cb_")],
  soil_continuous = intersect(feature_names, soil_vars),
  soil_categorical = feature_names[str_detect(feature_names, "^share_soil_")]
)

# Keep the original two-cluster design as the benchmark and retain nearby
# alternatives for sensitivity analysis. Every AEWR region has at least 16
# CZ-region units, so each value is feasible in every region.
iv_k_values <- 2:5

cat(
  "Clustering each AEWR region for k =",
  paste(iv_k_values, collapse = ", "),
  "\n"
)
cluster_list <- list()
cluster_diagnostic_list <- list()
donor_cluster_list <- list()
for (r in sort(unique(unit_features$aewr_region_num))) {
  d <- unit_features %>% filter(aewr_region_num == r)

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
    donor_cluster_counts <- seq_len(iv_k - 1L)

    cluster_list[[list_key]] <- d %>%
      select(cz_aewr_region_fe, aewr_region_num) %>%
      mutate(iv_k = iv_k, iv_cluster = selected_cluster)

    cluster_diagnostic_list[[list_key]] <- tibble(
      aewr_region_num = r,
      iv_k = iv_k,
      iv_cluster = selected_cluster,
      unit_feature_weight = d$unit_feature_weight
    ) %>%
      group_by(aewr_region_num, iv_k, iv_cluster) %>%
      summarise(
        cluster_units = n(),
        cluster_feature_weight = sum(unit_feature_weight, na.rm = TRUE),
        .groups = "drop"
      ) %>%
      mutate(
        cluster_feature_weight_share = cluster_feature_weight /
          sum(cluster_feature_weight, na.rm = TRUE)
      )

    cluster_centroids <- as_tibble(x) %>%
      mutate(iv_cluster = selected_cluster) %>%
      group_by(iv_cluster) %>%
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
        aewr_region_num = r,
        iv_k = iv_k,
        target_cluster = target_cluster_id,
        donor_cluster = cluster_ids,
        donor_cluster_distance = centroid_distance[
          as.character(target_cluster_id),
          as.character(cluster_ids)
        ]
      ) %>%
        filter(donor_cluster != target_cluster) %>%
        arrange(desc(donor_cluster_distance), donor_cluster) %>%
        mutate(donor_rank = row_number()) %>%
        crossing(
          donor_cluster_count = donor_cluster_counts
        ) %>%
        filter(donor_rank <= donor_cluster_count)
    }

    donor_cluster_list[[list_key]] <- bind_rows(donor_pair_list)
  }
}

iv_clusters <- bind_rows(cluster_list)
cat("Writing cluster assignments and donor pairs\n")
write_parquet(iv_clusters, path_int("iv_cz_aewr_clusters.parquet"))
iv_cluster_diagnostics <- bind_rows(cluster_diagnostic_list) %>%
  group_by(aewr_region_num, iv_k) %>%
  mutate(
    region_min_cluster_units = min(cluster_units),
    region_min_cluster_weight_share = min(cluster_feature_weight_share)
  ) %>%
  ungroup()
iv_donor_clusters <- bind_rows(donor_cluster_list)
write_parquet(
  iv_cluster_diagnostics,
  path_int("iv_cluster_diagnostics.parquet")
)
write_parquet(iv_donor_clusters, path_int("iv_donor_clusters.parquet"))

# Map CZ x AEWR-region clusters ----------------------------------------------

cat("Rendering maps for each cluster count\n")
iv_cluster_figure_dir <- path_figures("iv_dissimilarity_clusters")
dir.create(iv_cluster_figure_dir, recursive = TRUE, showWarnings = FALSE)

aewr_region_labels <- county_df %>%
  distinct(aewr_region_num, state_abbrev) %>%
  filter(!is.na(aewr_region_num), !is.na(state_abbrev)) %>%
  arrange(aewr_region_num, state_abbrev) %>%
  group_by(aewr_region_num) %>%
  summarise(
    aewr_region_states = paste(state_abbrev, collapse = ", "),
    .groups = "drop"
  ) %>%
  mutate(
    aewr_region_label = paste0(
      "AEWR Region ",
      aewr_region_num,
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

county_iv_clusters <- county_df %>%
  distinct(countyfips, cz_out10, aewr_region_num, cz_aewr_region_fe) %>%
  # Analysis data use the pre-2015 Shannon County code; the bundled 2020
  # TIGER shapefile uses the newer Oglala Lakota County code.
  mutate(countyfips = recode(countyfips, `46113` = "46102")) %>%
  left_join(
    iv_clusters,
    by = c("cz_aewr_region_fe", "aewr_region_num"),
    relationship = "many-to-many"
  ) %>%
  left_join(aewr_region_labels, by = "aewr_region_num") %>%
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

stopifnot(all(!is.na(county_iv_clusters$iv_cluster_id)))

county_shape_zip <- path_raw("county_shapefile", "tl_2020_us_county.zip")
unzip(county_shape_zip, exdir = tempdir())
county_map_iv_clusters <- sf::st_read(
  file.path(tempdir(), "tl_2020_us_county.shp"),
  quiet = TRUE
) %>%
  mutate(
    statefip = state_fips(STATEFP),
    countyfips = combine_county_fips(STATEFP, COUNTYFP)
  ) %>%
  filter(
    as.integer(statefip) <= 56,
    !statefip %in% c("02", "15")
  ) %>%
  sf::st_make_valid() %>%
  sf::st_transform(5070) %>%
  left_join(county_iv_clusters, by = "countyfips") %>%
  filter(!is.na(aewr_region_num))

cz_aewr_cluster_boundaries <- county_map_iv_clusters %>%
  group_by(iv_k, cz_aewr_region_fe, aewr_region_num, aewr_region_label) %>%
  summarise(geometry = sf::st_union(geometry), .groups = "drop")

aewr_region_boundaries <- county_map_iv_clusters %>%
  group_by(iv_k, aewr_region_num, aewr_region_label) %>%
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

for (iv_k in iv_k_values) {
  k_cluster_levels <- paste0("Cluster ", seq_len(iv_k))
  k_map_data <- county_map_iv_clusters %>%
    filter(iv_k == .env$iv_k) %>%
    mutate(
      iv_cluster = factor(
        as.character(iv_cluster),
        levels = k_cluster_levels
      )
    )
  k_cz_boundaries <- cz_aewr_cluster_boundaries %>%
    filter(iv_k == .env$iv_k)
  k_region_boundaries <- aewr_region_boundaries %>%
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
    filename = path_figures(
      "iv_dissimilarity_clusters",
      paste0(
        "iv_dissimilarity_clusters_all_aewr_regions_k",
        iv_k,
        ".png"
      )
    ),
    iv_cluster_map_all,
    width = 12,
    height = 8,
    dpi = 150,
    device = "png",
    bg = "white"
  )

  # Preserve the original filenames for the benchmark two-cluster design.
  if (iv_k == min(iv_k_values)) {
    ggsave(
      filename = path_figures(
        "iv_dissimilarity_clusters",
        "iv_dissimilarity_clusters_all_aewr_regions.png"
      ),
      iv_cluster_map_all,
      width = 12,
      height = 8,
      dpi = 150,
      device = "png",
      bg = "white"
    )
  }

  for (r in sort(unique(k_map_data$aewr_region_num))) {
    region_map_data <- k_map_data %>%
      filter(aewr_region_num == .env$r)
    region_cz_boundaries <- k_cz_boundaries %>%
      filter(aewr_region_num == .env$r)
    region_boundary <- k_region_boundaries %>%
      filter(aewr_region_num == .env$r)

    region_label <- as.character(unique(region_map_data$aewr_region_label))
    region_bbox <- sf::st_bbox(region_map_data)
    region_aspect <- as.numeric(
      (region_bbox$ymax - region_bbox$ymin) /
        (region_bbox$xmax - region_bbox$xmin)
    )
    save_width <- if (region_aspect > 1.15) 6.5 else 8.5
    save_height <- min(max(save_width * region_aspect + 1.2, 4.8), 9)

    iv_cluster_map_region <- ggplot() +
      geom_sf(
        data = region_map_data,
        aes(fill = iv_cluster),
        color = scales::alpha("white", 0.45),
        linewidth = 0.06
      ) +
      geom_sf(
        data = region_cz_boundaries,
        fill = NA,
        color = scales::alpha("grey15", 0.75),
        linewidth = 0.2
      ) +
      geom_sf(
        data = region_boundary,
        fill = NA,
        color = "black",
        linewidth = 0.5
      ) +
      scale_fill_manual(
        values = iv_cluster_colors,
        drop = TRUE,
        name = "IV cluster"
      ) +
      coord_sf(datum = NA) +
      labs(
        title = region_label,
        subtitle = paste0(
          "CZ x AEWR-region dissimilarity clusters, k = ",
          iv_k
        ),
        x = NULL,
        y = NULL
      ) +
      iv_cluster_map_theme

    region_id <- str_pad(r, width = 2, side = "left", pad = "0")
    ggsave(
      filename = path_figures(
        "iv_dissimilarity_clusters",
        paste0(
          "iv_dissimilarity_clusters_k",
          iv_k,
          "_aewr_region_",
          region_id,
          ".png"
        )
      ),
      iv_cluster_map_region,
      width = save_width,
      height = save_height,
      dpi = 150,
      device = "png",
      bg = "white"
    )

    if (iv_k == min(iv_k_values)) {
      ggsave(
        filename = path_figures(
          "iv_dissimilarity_clusters",
          paste0(
            "iv_dissimilarity_clusters_aewr_region_",
            region_id,
            ".png"
          )
        ),
        iv_cluster_map_region,
        width = save_width,
        height = save_height,
        dpi = 150,
        device = "png",
        bg = "white"
      )
    }
  }
}

cat("Saved IV cluster maps to", iv_cluster_figure_dir, "\n")
