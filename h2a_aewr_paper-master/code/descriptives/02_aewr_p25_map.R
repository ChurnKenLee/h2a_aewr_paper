# Purpose: Map the 2008--2022 change in the county AEWR-p25 wage gap.
# Output: map_aewr_cz_p25_change_from_trend_2022_2008.png.

here::i_am("code/paths.R")
source(here::here("code", "paths.R"))
source(path_code("c00_shared", "geography.R"))
source(path_code("descriptives", "helpers.R"))
library(arrow)
library(dplyr)
library(ggplot2)
library(ggspatial)
library(sf)
library(tidyr)

bite_change <- read_parquet(
  path_processed("county_year_panel.parquet")
) %>%
  filter(any_cropland_2007, year %in% c(2008L, 2022L)) %>%
  select(county_fips, year, aewr_cz_p25) %>%
  pivot_wider(
    names_from = year,
    values_from = aewr_cz_p25,
    names_prefix = "bite_"
  ) %>%
  transmute(
    county_fips,
    bite_change_2008_2022 = bite_2022 - bite_2008
  ) %>%
  filter(is.finite(bite_change_2008_2022))

county_map <- read_county_map(
  path_raw("county_shapefile", "tl_2020_us_county.zip")
) %>%
  left_join(
    bite_change,
    by = "county_fips",
    relationship = "one-to-one"
  )

figure <- ggplot(county_map) +
  geom_sf(
    aes(fill = bite_change_2008_2022),
    color = scales::alpha("grey", 0.3),
    linewidth = 0.1
  ) +
  annotation_north_arrow(
    location = "bl",
    which_north = "true",
    pad_x = grid::unit(0.05, "in"),
    pad_y = grid::unit(0.25, "in"),
    style = north_arrow_fancy_orienteering
  ) +
  annotation_scale(location = "bl", width_hint = 0.4) +
  scale_fill_gradient2(
    low = scales::muted("#2166ac"),
    mid = "white",
    midpoint = median(
      bite_change$bite_change_2008_2022,
      na.rm = TRUE
    ),
    high = scales::muted("#b2182b"),
    name = "Change in\nAEWR p25 Bite\n2008–2022\n(2012 $)",
    na.value = "grey90"
  ) +
  theme_bw() +
  theme(
    panel.grid.major = element_line(
      color = gray(0.5),
      linetype = "dashed",
      linewidth = 0.5
    ),
    panel.background = element_rect(fill = "aliceblue")
  )

ggsave(
  path_figures(
    "map_aewr_cz_p25_change_from_trend_2022_2008.png"
  ),
  figure,
  device = "png"
)
