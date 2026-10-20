# Synthetic checks for the Kitagawa decomposition, AR sets and Hansen J.
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
  isTRUE(all.equal(
    components$z_weight_shapley + components$z_wage_shapley,
    components$z_weight + components$z_wage + components$z_interaction
  )),
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

# AR sets: strong first stage is bounded and contains pi_y / pi_x; weak is not.
strong <- iv_ar_confidence_set(2, 1, 0.04, 0.01, 0.005, qnorm(0.975)^2)
weak <- iv_ar_confidence_set(2, 0.1, 0.04, 0.01, 0, qnorm(0.975)^2)
stopifnot(
  strong$type == "bounded",
  strong$lower < 2, strong$upper > 2,
  weak$type %in% c("complement_of_interval", "unbounded")
)

# Hansen J: valid overidentifying instruments rarely reject; first-step 2SLS
# matches the closed form.
n <- 4000L
cluster <- rep(seq_len(200L), each = n / 200L)
z <- cbind(rnorm(n), rnorm(n))
error <- rnorm(n)
x <- z %*% c(1, 0.8) + 0.5 * error + rnorm(n)
y <- 1.5 * x + error
valid <- iv_cluster_hansen_j(as.vector(y), x, z, cluster)
invalid <- iv_cluster_hansen_j(as.vector(y + 0.5 * z[, 2]), x, z, cluster)
projection <- z %*% solve(crossprod(z), crossprod(z, x))
stopifnot(
  abs(valid$beta_2sls - sum(projection * y) / sum(projection * x)) < 1e-10,
  abs(valid$beta_2sls - 1.5) < 0.1,
  valid$df == 1L,
  invalid$p_value < 0.001
)

message("Panel-IV decomposition tests passed")
