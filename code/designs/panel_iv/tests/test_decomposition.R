# Synthetic checks for the Kitagawa decomposition identity.
here::i_am("code/paths.R")
source(here::here("code", "paths.R"))
source(path_code("designs", "panel_iv", "decomposition.R"))
library(dplyr)

set.seed(20261016L)
cells <- expand.grid(
  target_cluster = 1:2,
  county_fips = c("01001", "01003", "01005"),
  source_year = 2010:2013,
  stringsAsFactors = FALSE
)
cells$raw_weight <- runif(nrow(cells), 0.2, 1)
cells$donor_real_hourly_wage <- runif(nrow(cells), 8, 14)
cells$base_real_hourly_wage <- 10 + match(cells$county_fips, unique(cells$county_fips))
cells <- cells %>%
  group_by(target_cluster, source_year) %>%
  mutate(decomposition_weight = raw_weight / sum(raw_weight)) %>%
  ungroup()

components <- iv_kitagawa_components(cells, "target_cluster")
direct <- cells %>%
  group_by(target_cluster, source_year) %>%
  summarise(
    z = sum(decomposition_weight * donor_real_hourly_wage),
    z_fixed_wage = sum(decomposition_weight * base_real_hourly_wage),
    .groups = "drop"
  )
stopifnot(
  nrow(components) == 8L,
  isTRUE(all.equal(components$z_balanced, direct$z)),
  max(abs(components$identity_residual)) < 1e-12,
  # The weight component equals the fixed-wage aggregate up to a cluster constant.
  all(abs(tapply(
    direct$z_fixed_wage - components$z_weight,
    components$target_cluster,
    function(value) diff(range(value))
  )) < 1e-12)
)

# Fixed weights leave only the wage component; fixed wages only the weight one.
fixed_weights <- cells %>%
  group_by(target_cluster, county_fips) %>%
  mutate(decomposition_weight = mean(decomposition_weight)) %>%
  ungroup()
fixed_weight_components <- iv_kitagawa_components(fixed_weights, "target_cluster")
fixed_wages <- cells %>% mutate(donor_real_hourly_wage = base_real_hourly_wage)
fixed_wage_components <- iv_kitagawa_components(fixed_wages, "target_cluster")
stopifnot(
  max(abs(fixed_weight_components$z_weight)) < 1e-12,
  max(abs(fixed_weight_components$z_interaction)) < 1e-12,
  max(abs(fixed_wage_components$z_wage)) < 1e-12,
  max(abs(fixed_wage_components$z_interaction)) < 1e-12
)

unbalanced <- cells[-1, ]
stopifnot(inherits(
  tryCatch(iv_kitagawa_components(unbalanced, "target_cluster"), error = identity),
  "error"
))

# Balanced donors: a county missing eligibility in one year is dropped and the
# remaining weights are renormalized; a group that loses a donor cluster is
# reported unsupported and contributes no components.
frame <- cells %>%
  mutate(
    donor_cluster = if_else(county_fips == "01005", 2L, 1L),
    eligible = !(target_cluster == 1L & county_fips == "01003" & source_year == 2011L),
    weight = raw_weight
  )
balanced <- iv_balanced_components(frame, "target_cluster", 2010:2013, 2L)
first_group <- balanced$balanced %>% filter(target_cluster == 1L)
stopifnot(
  !"01003" %in% first_group$county_fips,
  all(abs(tapply(first_group$decomposition_weight, first_group$source_year, sum) - 1) < 1e-12),
  all(balanced$supported_groups$decomposition_supported),
  nrow(balanced$components) == 8L
)
one_cluster <- frame %>% mutate(eligible = eligible & !(target_cluster == 2L & county_fips == "01005"))
unsupported <- iv_balanced_components(one_cluster, "target_cluster", 2010:2013, 2L)
stopifnot(
  identical(unsupported$supported_groups$decomposition_supported, c(TRUE, FALSE)),
  all(unsupported$components$target_cluster == 1L)
)
message("Decomposition tests passed")
