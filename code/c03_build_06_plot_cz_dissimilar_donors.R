rm(list = ls())

if (file.exists("paths.R")) {
  source("paths.R")
} else {
  source(file.path("code", "paths.R"))
}

library(arrow)
library(sf)
library(tidyverse)

## Diagnostic maps for CZ-level dissimilar-donor links -------------------------

ensure_project_dirs()

scale_year <- 2011
map_crs <- 5070
default_n_targets <- 12
output_dir <- Sys.getenv("CZ_DONOR_MAP_OUTPUT_DIR", unset = path_figures())
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

target_env <- Sys.getenv("CZ_DONOR_MAP_TARGETS", unset = "")
target_cz_out10 <- target_env %>%
  str_split("[,[:space:]]+") %>%
  pluck(1) %>%
  discard(~ is.na(.x) || .x == "")

max_targets <- as.integer(Sys.getenv(
  "CZ_DONOR_MAP_N_TARGETS",
  unset = as.character(default_n_targets)
))
if (is.na(max_targets) || max_targets <= 0) {
  max_targets <- default_n_targets
}

read_county_map <- function() {
  shp_dir <- tempfile("county_shp_")
  dir.create(shp_dir)
  unzip(path_raw("county_shapefile", "tl_2020_us_county.zip"), exdir = shp_dir)

  st_read(file.path(shp_dir, "tl_2020_us_county.shp"), quiet = TRUE) %>%
    transmute(
      statefip = STATEFP,
      countyfips = str_c(STATEFP, COUNTYFP),
      geometry
    ) %>%
    filter(
      as.integer(statefip) <= 56,
      !statefip %in% c("02", "15")
    )
}

centroid_xy <- function(sf_data, geometry_col = "geometry") {
  points <- sf_data %>%
    st_geometry() %>%
    st_point_on_surface()

  coords <- st_coordinates(points)

  sf_data %>%
    st_drop_geometry() %>%
    mutate(x = coords[, "X"], y = coords[, "Y"])
}

save_png <- function(filename, plot, width, height, dpi = 300) {
  ggsave(
    filename = filename,
    plot = plot,
    width = width,
    height = height,
    units = "in",
    dpi = dpi,
    device = function(filename, width, height, units, res, ...) {
      grDevices::png(
        filename = filename,
        width = width,
        height = height,
        units = units,
        res = res
      )
    }
  )
}

safe_file_name <- function(x) {
  x %>%
    str_to_lower() %>%
    str_replace_all("[^0-9a-z]+", "_") %>%
    str_replace_all("^_+|_+$", "")
}

cz_region_counties <- read_parquet(
  path_processed("county_df_analysis_year.parquet"),
  col_select = c("countyfips", "year", "aewr_region_num", "cz_out10")
) %>%
  filter(
    year == scale_year,
    !is.na(aewr_region_num),
    !is.na(cz_out10)
  ) %>%
  transmute(
    countyfips = as.character(countyfips),
    aewr_region_num = as.integer(aewr_region_num),
    cz_out10 = as.character(cz_out10)
  ) %>%
  distinct()

donor_links <- read_parquet(
  path_processed("cz_dissimilar_donor_links.parquet")
) %>%
  mutate(
    aewr_region_num = as.integer(aewr_region_num),
    target_cz_out10 = as.character(target_cz_out10),
    donor_cz_out10 = as.character(donor_cz_out10)
  )

cz_features <- read_parquet(
  path_processed("cz_dissimilar_features.parquet"),
  col_select = c("aewr_region_num", "cz_out10", "emp_farm_2011")
) %>%
  mutate(
    aewr_region_num = as.integer(aewr_region_num),
    cz_out10 = as.character(cz_out10)
  )

cz_region_shapes <- read_county_map() %>%
  inner_join(cz_region_counties, by = "countyfips") %>%
  st_transform(map_crs) %>%
  group_by(aewr_region_num, cz_out10) %>%
  summarise(geometry = st_union(geometry), .groups = "drop") %>%
  st_make_valid() %>%
  st_simplify(dTolerance = 1500, preserveTopology = TRUE)

cz_region_points <- centroid_xy(cz_region_shapes)

target_points <- cz_region_points %>%
  transmute(
    aewr_region_num,
    target_cz_out10 = cz_out10,
    target_x = x,
    target_y = y
  )

donor_points <- cz_region_points %>%
  transmute(
    aewr_region_num,
    donor_cz_out10 = cz_out10,
    donor_x = x,
    donor_y = y
  )

link_segments <- donor_links %>%
  left_join(target_points, by = c("aewr_region_num", "target_cz_out10")) %>%
  left_join(donor_points, by = c("aewr_region_num", "donor_cz_out10")) %>%
  filter(
    !is.na(target_x),
    !is.na(target_y),
    !is.na(donor_x),
    !is.na(donor_y)
  )

map_all_links <- ggplot() +
  geom_sf(
    data = cz_region_shapes,
    fill = "grey96",
    color = "grey82",
    linewidth = 0.05
  ) +
  geom_segment(
    data = link_segments,
    aes(
      x = target_x,
      y = target_y,
      xend = donor_x,
      yend = donor_y,
      alpha = 1 / distance_rank_desc
    ),
    color = "#2166ac",
    linewidth = 0.12
  ) +
  scale_alpha(range = c(0.04, 0.22), guide = "none") +
  coord_sf(crs = st_crs(map_crs), datum = NA) +
  labs(
    title = "Dissimilar donor CZ links",
    subtitle = "Each line connects a CZ-region target to one selected donor CZ in the same AEWR region"
  ) +
  theme_void(base_size = 11) +
  theme(
    plot.title = element_text(face = "bold"),
    plot.subtitle = element_text(color = "grey35")
  )

map_all_links_path <- file.path(
  output_dir,
  "map_cz_dissimilar_donor_links_all.png"
)
map_target_links_path <- file.path(
  output_dir,
  "map_cz_dissimilar_donor_links_targets.png"
)

save_png(
  filename = map_all_links_path,
  plot = map_all_links,
  width = 11,
  height = 7
)

selected_targets <- donor_links %>%
  distinct(aewr_region_num, target_cz_out10) %>%
  left_join(
    cz_features,
    by = c("aewr_region_num", "target_cz_out10" = "cz_out10")
  )

if (length(target_cz_out10) > 0) {
  selected_targets <- selected_targets %>%
    filter(target_cz_out10 %in% target_cz_out10)
} else {
  selected_targets <- selected_targets %>%
    arrange(desc(emp_farm_2011), aewr_region_num, target_cz_out10) %>%
    slice_head(n = max_targets)
}

selected_targets <- selected_targets %>%
  mutate(panel = str_c("AEWR ", aewr_region_num, " / CZ ", target_cz_out10))

panel_levels <- selected_targets$panel

panel_base_shapes <- map_dfr(
  panel_levels,
  ~ {
    cz_region_shapes %>%
      mutate(panel = .x)
  }
)

selected_segments <- link_segments %>%
  inner_join(
    selected_targets %>% select(aewr_region_num, target_cz_out10, panel),
    by = c("aewr_region_num", "target_cz_out10")
  ) %>%
  mutate(panel = factor(panel, levels = panel_levels))

highlight_keys <- bind_rows(
  selected_targets %>%
    transmute(
      panel,
      aewr_region_num,
      cz_out10 = target_cz_out10,
      map_role = "Target",
      map_rank = 0L
    ),
  selected_segments %>%
    transmute(
      panel,
      aewr_region_num,
      cz_out10 = donor_cz_out10,
      map_role = str_c("Donor ", distance_rank_desc),
      map_rank = distance_rank_desc
    )
) %>%
  distinct()

highlight_shapes <- cz_region_shapes %>%
  inner_join(highlight_keys, by = c("aewr_region_num", "cz_out10")) %>%
  mutate(
    panel = factor(panel, levels = panel_levels),
    map_role = factor(map_role, levels = c("Target", paste("Donor", 1:5))),
    label = if_else(
      map_role == "Target",
      str_c("T: ", cz_out10),
      str_c(map_rank, ": ", cz_out10)
    )
  )

panel_windows <- highlight_shapes %>%
  group_by(panel) %>%
  summarise(geometry = st_union(geometry), .groups = "drop") %>%
  st_buffer(120000)

map_target_links <- ggplot() +
  geom_sf(
    data = panel_base_shapes %>%
      mutate(panel = factor(panel, levels = panel_levels)),
    fill = "grey97",
    color = "grey85",
    linewidth = 0.03
  ) +
  geom_segment(
    data = selected_segments,
    aes(
      x = target_x,
      y = target_y,
      xend = donor_x,
      yend = donor_y
    ),
    color = "grey30",
    linewidth = 0.25,
    alpha = 0.65
  ) +
  geom_sf(
    data = highlight_shapes,
    aes(fill = map_role),
    color = "white",
    linewidth = 0.18
  ) +
  geom_sf_text(
    data = highlight_shapes,
    aes(label = label),
    size = 2,
    color = "black"
  ) +
  scale_fill_manual(
    values = c(
      "Target" = "#b2182b",
      "Donor 1" = "#2166ac",
      "Donor 2" = "#4393c3",
      "Donor 3" = "#92c5de",
      "Donor 4" = "#d1e5f0",
      "Donor 5" = "#f7f7f7"
    ),
    drop = FALSE,
    name = NULL
  ) +
  coord_sf(crs = st_crs(map_crs), datum = NA) +
  facet_wrap(~panel, ncol = 3) +
  labs(
    title = "Selected target CZs and their dissimilar donor CZs",
    subtitle = if_else(
      length(target_cz_out10) > 0,
      "Targets selected from CZ_DONOR_MAP_TARGETS",
      str_c(
        "Default selection: largest ",
        nrow(selected_targets),
        " target CZ-region units by 2011 farm employment"
      )
    )
  ) +
  theme_void(base_size = 10) +
  theme(
    legend.position = "bottom",
    plot.title = element_text(face = "bold"),
    plot.subtitle = element_text(color = "grey35"),
    strip.text = element_text(face = "bold", margin = margin(b = 3))
  )

save_png(
  filename = map_target_links_path,
  plot = map_target_links,
  width = 12,
  height = 10
)

target_map_dir <- file.path(output_dir, "cz_dissimilar_donor_target_maps")
dir.create(target_map_dir, recursive = TRUE, showWarnings = FALSE)

target_map_paths <- map_chr(panel_levels, function(panel_value) {
  panel_window <- panel_windows %>%
    filter(as.character(panel) == panel_value)
  panel_base <- cz_region_shapes %>%
    st_crop(st_bbox(panel_window))
  panel_highlights <- highlight_shapes %>%
    filter(as.character(panel) == panel_value)
  panel_segments <- selected_segments %>%
    filter(as.character(panel) == panel_value)

  target_map <- ggplot() +
    geom_sf(
      data = panel_base,
      fill = "grey97",
      color = "grey85",
      linewidth = 0.08
    ) +
    geom_segment(
      data = panel_segments,
      aes(
        x = target_x,
        y = target_y,
        xend = donor_x,
        yend = donor_y
      ),
      color = "grey30",
      linewidth = 0.35,
      alpha = 0.75
    ) +
    geom_sf(
      data = panel_highlights,
      aes(fill = map_role),
      color = "white",
      linewidth = 0.25
    ) +
    geom_sf_text(
      data = panel_highlights,
      aes(label = label),
      size = 3,
      color = "black",
      check_overlap = TRUE
    ) +
    scale_fill_manual(
      values = c(
        "Target" = "#b2182b",
        "Donor 1" = "#2166ac",
        "Donor 2" = "#4393c3",
        "Donor 3" = "#92c5de",
        "Donor 4" = "#d1e5f0",
        "Donor 5" = "#f7f7f7"
      ),
      drop = FALSE,
      name = NULL
    ) +
    coord_sf(crs = st_crs(map_crs), datum = NA) +
    labs(
      title = panel_value,
      subtitle = "Target CZ in red; selected dissimilar donor CZs in blue"
    ) +
    theme_void(base_size = 11) +
    theme(
      legend.position = "bottom",
      plot.title = element_text(face = "bold"),
      plot.subtitle = element_text(color = "grey35")
    )

  target_map_path <- file.path(
    target_map_dir,
    str_c("map_cz_dissimilar_donor_links_", safe_file_name(panel_value), ".png")
  )

  save_png(
    filename = target_map_path,
    plot = target_map,
    width = 7,
    height = 6
  )

  target_map_path
})

print(tibble(
  output = c(map_all_links_path, map_target_links_path, target_map_paths)
))
