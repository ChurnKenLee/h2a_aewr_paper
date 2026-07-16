# Construct CZ-level AEWR IV using same-AEWR-region CZs
rm(list = ls())
if (!exists("path_code", mode = "function")) {
  source(file.path("code", "paths.R"))
}
library(arrow)
library(tidyverse)
library(tidylog, warn.conflicts = FALSE)
library(janitor)
library(foreign)
if (!exists("split_county_map", mode = "function")) {
  source(path_code("c00_setup.R"))
}

cdl <- read_parquet(path_int(
  "croplandcros_county_crop_type_acres.parquet"
)) %>%
  clean_names() %>%
  filter(!is.na(crop_type_label), crop_type_label != "non-crop")

climate <- read_parquet(path_int(
  "county_h2a_prediction_climate_basis_annual.parquet"
)) %>%
  clean_names()

soil <- read_parquet(path_int(
  "county_h2a_prediction_gnatsgo_soil_cells.parquet"
)) %>%
  clean_names()

county_df <- read_parquet(path_processed(
  "county_df_analysis_year.parquet"
)) %>%
  clean_names()

fls_county_oews_area_prior <- read_parquet(path_int(
  "fls_county_oews_area_prior_weight.parquet"
)) %>%
  clean_names() %>%
  mutate(year = as.integer(year))

fls_oews_area_calibrated <- read_parquet(path_int(
  "fls_oews_area_weight_wage_calibrated.parquet"
)) %>%
  clean_names() %>%
  mutate(year = as.integer(year))

# Fixed 2008-2011 county primitives

crop_names <- cdl %>%
  distinct(crop_type_label) %>%
  mutate(crop_var = paste0("share_cdl_", make_clean_names(crop_type_label)))

crop_features <- cdl %>%
  filter(year >= 2008, year <= 2011) %>%
  mutate(
    county_ansi = countyfips
  ) %>%
  left_join(crop_names, by = "crop_type_label") %>%
  group_by(county_ansi, year, crop_var) %>%
  summarise(acres = sum(acres, na.rm = TRUE), .groups = "drop") %>%
  group_by(county_ansi, year) %>%
  mutate(crop_share = acres / sum(acres, na.rm = TRUE)) %>%
  ungroup() %>%
  group_by(county_ansi, crop_var) %>%
  summarise(crop_share = mean(crop_share, na.rm = TRUE), .groups = "drop")

crop_features <- xtabs(
  crop_share ~ county_ansi + crop_var,
  data = crop_features
) %>%
  as.data.frame.matrix() %>%
  rownames_to_column("county_ansi") %>%
  as_tibble()

climate_features <- climate %>%
  mutate(
    county_ansi = fips
  ) %>%
  filter(year >= 2008, year <= 2011) %>%
  select(county_ansi, starts_with("normal_cb_")) %>%
  group_by(county_ansi) %>%
  summarise(
    across(starts_with("normal_cb_"), ~ mean(.x, na.rm = TRUE)),
    .groups = "drop"
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

soil_cat_vars <- c("taxorder", "drainagecl", "hydgrp", "nirrcapcl")

soil_cont_features <- soil %>%
  group_by(county_ansi) %>%
  summarise(
    across(all_of(soil_vars), ~ weighted.mean(.x, total_acres, na.rm = TRUE)),
    .groups = "drop"
  )

# This constructs a set of dataframes inside the list soil_cat_vars
# Each dataframe contains a categorical variable
# This is needed because each categorical variable can take multiple values
# Each column is the share of a particular value within the county
soil_cat_list <- list()
for (v in soil_cat_vars) {
  soil_value <- soil[[v]]
  soil_value[is.na(soil_value)] <- "missing"
  soil_value_names <- data.frame(
    soil_value = unique(soil_value),
    soil_value_clean = make_clean_names(unique(soil_value))
  )

  temp <- data.frame(
    county_ansi = soil$county_ansi,
    soil_value = soil_value,
    total_acres = soil$total_acres
  )
  temp <- merge(
    temp,
    soil_value_names,
    by = "soil_value",
    all.x = TRUE,
    all.y = FALSE
  )
  temp$soil_feature <- paste0("share_soil_", v, "_", temp$soil_value_clean)
  temp <- aggregate(
    total_acres ~ county_ansi + soil_feature,
    temp,
    sum,
    na.rm = TRUE
  )
  temp$soil_share <- temp$total_acres /
    ave(
      temp$total_acres,
      temp$county_ansi,
      FUN = sum
    )

  soil_cat_list[[v]] <- xtabs(
    soil_share ~ county_ansi + soil_feature,
    data = temp
  ) %>%
    as.data.frame.matrix() %>%
    rownames_to_column("county_ansi") %>%
    as_tibble()
}

soil_cat_features <- reduce(soil_cat_list, full_join, by = "county_ansi")

soil_features <- soil_cont_features %>%
  full_join(soil_cat_features, by = "county_ansi")

county_features <- crop_features %>%
  full_join(climate_features, by = "county_ansi") %>%
  full_join(soil_features, by = "county_ansi")

county_feature_names <- setdiff(names(county_features), "county_ansi")

# Cluster CZ x AEWR-region units within AEWR regions --------------------------

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

iv_k <- 2

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
  selected_cluster <- cutree(hclust_fit, k = iv_k)

  cluster_list[[as.character(r)]] <- d %>%
    select(cz_aewr_region_fe, aewr_region_num) %>%
    mutate(iv_cluster = selected_cluster, iv_k = iv_k)

  cluster_diagnostic_list[[as.character(r)]] <- tibble(
    aewr_region_num = r,
    iv_cluster = selected_cluster,
    unit_feature_weight = d$unit_feature_weight
  ) %>%
    group_by(aewr_region_num, iv_cluster) %>%
    summarise(
      iv_k = iv_k,
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
    summarise(across(all_of(feature_names), ~ mean(.x)), .groups = "drop")

  centroid_matrix <- as.matrix(cluster_centroids[, feature_names])
  rownames(centroid_matrix) <- cluster_centroids$iv_cluster
  centroid_distance <- as.matrix(dist(centroid_matrix))
  cluster_ids <- sort(unique(selected_cluster))
  donor_pair_list <- list()

  for (target_cluster_id in cluster_ids) {
    donor_pair_list[[as.character(target_cluster_id)]] <- tibble(
      aewr_region_num = r,
      target_cluster = target_cluster_id,
      donor_cluster = cluster_ids,
      donor_cluster_distance = centroid_distance[
        as.character(target_cluster_id),
        as.character(cluster_ids)
      ]
    ) %>%
      filter(donor_cluster != target_cluster) %>%
      arrange(desc(donor_cluster_distance)) %>%
      slice_head(n = 1)
  }

  donor_cluster_list[[as.character(r)]] <- bind_rows(donor_pair_list)
}

iv_clusters <- bind_rows(cluster_list)
iv_cluster_diagnostics <- bind_rows(cluster_diagnostic_list) %>%
  group_by(aewr_region_num) %>%
  mutate(
    region_min_cluster_units = min(cluster_units),
    region_min_cluster_weight_share = min(cluster_feature_weight_share)
  ) %>%
  ungroup()
iv_donor_clusters <- bind_rows(donor_cluster_list)

# Map CZ x AEWR-region clusters ----------------------------------------------

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

iv_cluster_levels <- paste0("Cluster ", seq_len(iv_k))
iv_cluster_base_colors <- c(
  "#1b9e77",
  "#d95f02",
  "#7570b3",
  "#e7298a",
  "#66a61e",
  "#e6ab02"
)
if (iv_k > length(iv_cluster_base_colors)) {
  iv_cluster_base_colors <- grDevices::colorRampPalette(
    iv_cluster_base_colors
  )(iv_k)
}
iv_cluster_colors <- setNames(
  iv_cluster_base_colors[seq_len(iv_k)],
  iv_cluster_levels
)

county_iv_clusters <- county_df %>%
  distinct(countyfips, cz_out10, aewr_region_num, cz_aewr_region_fe) %>%
  # Analysis data use the pre-2015 Shannon County code; the bundled 2020
  # TIGER shapefile uses the newer Oglala Lakota County code.
  mutate(countyfips = recode(countyfips, `46113` = "46102")) %>%
  left_join(iv_clusters, by = c("cz_aewr_region_fe", "aewr_region_num")) %>%
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
    statefip = split_fips2(STATEFP),
    countyfips = split_countyfips(STATEFP, COUNTYFP)
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
  group_by(cz_aewr_region_fe, aewr_region_num, aewr_region_label) %>%
  summarise(geometry = sf::st_union(geometry), .groups = "drop")

aewr_region_boundaries <- county_map_iv_clusters %>%
  group_by(aewr_region_num, aewr_region_label) %>%
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

iv_cluster_map_all <- ggplot() +
  geom_sf(
    data = county_map_iv_clusters,
    aes(fill = iv_cluster),
    color = scales::alpha("white", 0.35),
    linewidth = 0.03
  ) +
  geom_sf(
    data = cz_aewr_cluster_boundaries,
    fill = NA,
    color = scales::alpha("grey20", 0.65),
    linewidth = 0.12
  ) +
  geom_sf(
    data = aewr_region_boundaries,
    fill = NA,
    color = "white",
    linewidth = 1
  ) +
  geom_sf(
    data = aewr_region_boundaries,
    fill = NA,
    color = "#5e3c99",
    linewidth = 0.45
  ) +
  scale_fill_manual(
    values = iv_cluster_colors,
    drop = FALSE,
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
    "iv_dissimilarity_clusters_all_aewr_regions.png"
  ),
  iv_cluster_map_all,
  width = 12,
  height = 8,
  dpi = 150,
  device = "png",
  bg = "white"
)

for (r in sort(unique(county_map_iv_clusters$aewr_region_num))) {
  region_map_data <- county_map_iv_clusters %>%
    filter(aewr_region_num == r)
  region_cz_boundaries <- cz_aewr_cluster_boundaries %>%
    filter(aewr_region_num == r)
  region_boundary <- aewr_region_boundaries %>%
    filter(aewr_region_num == r)

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
      drop = FALSE,
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

  ggsave(
    filename = path_figures(
      "iv_dissimilarity_clusters",
      paste0(
        "iv_dissimilarity_clusters_aewr_region_",
        str_pad(r, width = 2, side = "left", pad = "0"),
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

cat("Saved IV cluster maps to", iv_cluster_figure_dir, "\n")

# Donor wage instrument --------------------------------------------------------

gap_closure_labels <- tribble(
  ~gap_closure , ~gap_closure_label ,
  0.00         , "g000"             ,
  0.25         , "g025"             ,
  0.50         , "g050"             ,
  0.75         , "g075"             ,
  1.00         , "g100"
)

county_oews_area_units <- fls_county_oews_area_prior %>%
  select(
    county_ansi = countyfips,
    year,
    aewr_region_num,
    cz_aewr_region_fe,
    oews_area_code,
    county_area_prior_weight
  ) %>%
  inner_join(
    iv_clusters,
    by = c("cz_aewr_region_fe", "aewr_region_num")
  )

target_cluster_oews_areas <- county_oews_area_units %>%
  transmute(
    aewr_region_num,
    year,
    target_cluster = iv_cluster,
    oews_area_code
  ) %>%
  distinct()

oews_area_donor_candidates <- county_oews_area_units %>%
  inner_join(
    fls_oews_area_calibrated %>%
      filter(str_detect(calibration_status, "^calibrated")) %>%
      select(
        aewr_region_num,
        year,
        oews_area_code,
        gap_closure,
        oews_area_mean_hourly_wage,
        oews_area_prior_weight_all,
        oews_area_weight_wage_calibrated
      ),
    by = c("aewr_region_num", "year", "oews_area_code")
  ) %>%
  mutate(
    county_share_within_oews_area = county_area_prior_weight /
      oews_area_prior_weight_all,
    donor_weight = oews_area_weight_wage_calibrated *
      county_share_within_oews_area
  ) %>%
  filter(!is.na(donor_weight), donor_weight > 0) %>%
  rename(donor_cluster = iv_cluster) %>%
  inner_join(
    iv_donor_clusters,
    by = c("aewr_region_num", "donor_cluster")
  )

oews_donor_candidate_support <- oews_area_donor_candidates %>%
  group_by(aewr_region_num, year, gap_closure, target_cluster) %>%
  summarise(
    oews_iv_candidate_areas = n_distinct(oews_area_code),
    oews_iv_candidate_units = n_distinct(cz_aewr_region_fe),
    .groups = "drop"
  )

# Exclude the entire OEWS area if any part of it touches the target cluster.
oews_area_donor_eligible <- oews_area_donor_candidates %>%
  anti_join(
    target_cluster_oews_areas,
    by = c(
      "aewr_region_num",
      "year",
      "target_cluster",
      "oews_area_code"
    )
  )

oews_donor_wages <- oews_area_donor_eligible %>%
  group_by(aewr_region_num, year, gap_closure, target_cluster) %>%
  summarise(
    z_oews_entropy_agwage_l1 = sum(
      donor_weight * oews_area_mean_hourly_wage
    ) /
      sum(donor_weight),
    oews_iv_donor_weight = sum(donor_weight),
    oews_iv_donor_areas = n_distinct(oews_area_code),
    oews_iv_donor_units = n_distinct(cz_aewr_region_fe),
    oews_iv_donor_cluster_distance = first(donor_cluster_distance),
    .groups = "drop"
  ) %>%
  left_join(
    oews_donor_candidate_support,
    by = c("aewr_region_num", "year", "gap_closure", "target_cluster")
  ) %>%
  mutate(
    oews_iv_overlap_areas_excluded = oews_iv_candidate_areas -
      oews_iv_donor_areas
  )

iv_oews_long <- iv_clusters %>%
  transmute(
    cz_aewr_region_fe,
    aewr_region_num,
    target_cluster = iv_cluster
  ) %>%
  inner_join(
    oews_donor_wages,
    by = c("aewr_region_num", "target_cluster")
  ) %>%
  # The outcome in year t uses the donor wage level from t - 1.
  mutate(year = year + 1L) %>%
  left_join(gap_closure_labels, by = "gap_closure")

iv_oews <- iv_oews_long %>%
  select(
    cz_aewr_region_fe,
    aewr_region_num,
    year,
    gap_closure_label,
    z_oews_entropy_agwage_l1,
    oews_iv_donor_weight,
    oews_iv_donor_areas,
    oews_iv_donor_units,
    oews_iv_candidate_areas,
    oews_iv_candidate_units,
    oews_iv_overlap_areas_excluded,
    oews_iv_donor_cluster_distance
  ) %>%
  pivot_wider(
    names_from = gap_closure_label,
    values_from = c(
      z_oews_entropy_agwage_l1,
      oews_iv_donor_weight,
      oews_iv_donor_areas,
      oews_iv_donor_units,
      oews_iv_candidate_areas,
      oews_iv_candidate_units,
      oews_iv_overlap_areas_excluded,
      oews_iv_donor_cluster_distance
    ),
    names_glue = "{.value}_{gap_closure_label}"
  )

county_df_iv <- county_df %>%
  mutate(
    dln_aewr = log(aewr) - log(aewr_l1),
    dln_aewr_ppi = log(aewr_ppi) - log(aewr_ppi_l1)
  ) %>%
  left_join(iv_clusters, by = c("cz_aewr_region_fe", "aewr_region_num")) %>%
  left_join(iv_oews, by = c("cz_aewr_region_fe", "aewr_region_num", "year"))

write_parquet(
  county_df_iv,
  path_processed("county_df_analysis_year_iv.parquet")
)

cat(
  "county_df_analysis_year_iv:",
  nrow(county_df_iv),
  "rows,",
  ncol(county_df_iv),
  "cols\n"
)

for (gap_label in gap_closure_labels$gap_closure_label) {
  instrument_name <- paste0(
    "z_oews_entropy_agwage_l1_",
    gap_label
  )
  cat(
    "Nonmissing",
    instrument_name,
    "rows:",
    sum(!is.na(county_df_iv[[instrument_name]])),
    "\n"
  )
}
