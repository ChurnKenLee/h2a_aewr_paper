library(soilDB)
library(tidyverse)
library(here)
library(arrow)
rm(list = ls())

# ==========================================================================
# Soil characteristics we want
# ==========================================================================
# These are the components that go into the revised Storie index (Greene et al. 2008)
# Note that we do not use horizons (whole nother can of worms there)

state_codes <- c(
  "AL",
  "AK",
  "AZ",
  "AR",
  "CA",
  "CO",
  "CT",
  "DE",
  "FL",
  "GA",
  "HI",
  "ID",
  "IL",
  "IN",
  "IA",
  "KS",
  "KY",
  "LA",
  "ME",
  "MD",
  "MA",
  "MI",
  "MN",
  "MS",
  "MO",
  "MT",
  "NE",
  "NV",
  "NH",
  "NJ",
  "NM",
  "NY",
  "NC",
  "ND",
  "OH",
  "OK",
  "OR",
  "PA",
  "RI",
  "SC",
  "SD",
  "TN",
  "TX",
  "UT",
  "VT",
  "VA",
  "WA",
  "WV",
  "WI",
  "WY"
)

# ==========================================================================
# Strategy
# Pull records one state at a time
# Aggregate component characteristics to the Map Unit level
# Distribute Map Unit characteristics across counties based on area overlap
# ==========================================================================
state <- "CA"
components_sql <- paste0(
  "
    SELECT
      mapunit.muname,
      mapunit.mukey,
      mapunit.muacres,
      component.compname,
      component.cokey,
      component.comppct_r,
      corestrictions.corestrictkey,
      -- Storie A
      component.taxorder,
      component.taxsuborder,
      component.taxgrtgroup,
      corestrictions.resdept_r,
      -- Storie C
      component.slope_r,
      -- Storie X
      component.drainagecl,
      -- USDA indices
      component.nirrcapcl
    FROM legend
      INNER JOIN mapunit ON mapunit.lkey = legend.lkey
      INNER JOIN component ON component.mukey = mapunit.mukey
      INNER JOIN corestrictions on corestrictions.cokey = component.cokey
    WHERE legend.areasymbol LIKE '",
  state,
  "%'
    AND legend.areasymbol != 'US'
    AND component.majcompflag = 'Yes'
  "
)
components <- SDA_query(components_sql)

laoverlap_sql <- paste0(
  "
    SELECT 
      la.areasymbol,
      la.areaname,
      mu.mukey,
      mu.muname,
      mu.muacres,
      mua.areaovacres
    FROM mapunit AS mu
    INNER JOIN legend AS l ON mu.lkey = l.lkey 
    INNER JOIN muaoverlap AS mua ON mu.mukey = mua.mukey 
    INNER JOIN laoverlap AS la ON mua.lareaovkey = la.lareaovkey AND la.lkey = l.lkey
    WHERE 
      la.areatypename = 'County or Parish'
      AND la.areasymbol <> 'US' -- Exclude STATSGO
      AND la.areasymbol LIKE '",
  state,
  "%'
  "
)
laoverlap <- SDA_query(laoverlap_sql)

laoverlap_acres <- laoverlap %>%
  filter(
    !is.na(areaovacres)
  ) %>%
  mutate(county_ansi = substr(areasymbol, 3, 5)) %>%
  filter(
    str_length(areasymbol) == 5
  ) %>%
  arrange(county_ansi)

diag <- laoverlap %>% filter(mukey == "457476")

laoverlap_acres <- laoverlap_acres %>%
  group_by(areasymbol, areaname) %>%
  mutate(au_overlap_acres = sum(areaovacres, na.rm = TRUE)) %>%
  ungroup() %>%
  group_by(mukey, muname, muacres) %>%
  mutate(mu_overlap_acres = sum(areaovacres, na.rm = TRUE)) %>%
  ungroup()

test <- laoverlap_acres %>%
  filter(muacres < areaovacres) %>%
  mutate(err = ((areaovacres - muacres) / muacres) * 100) %>%
  arrange(err)


test <- laoverlap %>%
  filter(areatypename == "County or Parish") %>%
  distinct(areasymbol, lareaovkey)

muaoverlap_sql <- paste0(
  "
    SELECT
      *
    FROM muaoverlap
      INNER JOIN laoverlap ON muaoverlap.lareaovkey = laoverlap.lareaovkey
    WHERE laoverlap.areasymbol LIKE '",
  state,
  "%'
  "
)
muaoverlap <- SDA_query(muaoverlap_sql)

overlap_sql <- paste0(
  ""
)


# Classify
# USDA uses 8% slope as maximum suitable for contour farming with tractors
# Depth uses USDA rooting depth of common crops
# There are 8 drainage classes
components_classified <- components %>%
  mutate(
    slope = case_when(
      slope_r <= 8.00 ~ "flat",
      slope_r > 8 ~ "steep"
    )
  ) %>%
  mutate(
    depth = case_when(
      resdept_r < 38 ~ "shallow",
      (resdept_r <= 76) & (resdept_r >= 38) ~ "moderate",
      resdept_r > 76 ~ "deep"
    )
  ) %>%
  mutate(
    drainage_rating = case_when(
      drainagecl == "Excessively drained" ~ 1,
      drainagecl == "Somewhat excessively drained" ~ 2,
      drainagecl == "Well drained" ~ 3,
      drainagecl == "Moderately well drained" ~ 4,
      drainagecl == "Somewhat poorly drained" ~ 5,
      drainagecl == "Poorly drained" ~ 6,
      drainagecl == "Very poorly drained" ~ 7,
      drainagecl == "Subaqueous" ~ 78,
    )
  )

# Obtain acreage of each component
components_classified <- components_classified %>%
  mutate(
    coacres = muacres * comppct_r / 100
  )

# Aggregate relevant component characteristics to the map unit level
mu_components_agg <- components_classified %>%
  group_by(
    muname,
    mukey,
    muacres,
    taxorder,
    taxsuborder,
    taxgrtgroup,
    slope,
    depth,
    drainage_rating
  ) %>%
  summarize(
    acre = sum(coacres, na.rm = TRUE)
  ) %>%
  ungroup()

# Calculate total overlapping acreages in both area units and map units
mu_acres <- components %>%
  distinct(muname, mukey, muacres)

overlap_acres <- overlap %>%
  filter(
    areatypename == "County or Parish",
    !is.na(areaovacres),
    projectscale < 100000,
  ) %>%
  mutate(county_ansi = substr(laoareasymbol, 3, 5)) %>%
  filter(
    str_length(laoareasymbol) == 5
  ) %>%
  arrange(county_ansi)

# %>%
# mutate(county_ansi = substr(areasymbol, 3, 5)) %>%
# mutate(county_ansi = as.numeric(county_ansi)) %>%
# arrange(county_ansi)

overlap_acres <- overlap_acres %>%
  inner_join(mu_acres, by = "mukey", relationship = "many-to-many") %>%
  group_by(areasymbol, areaname) %>%
  mutate(au_overlap_acres = sum(areaovacres, na.rm = TRUE)) %>%
  ungroup() %>%
  group_by(mukey, muname, muacres) %>%
  mutate(mu_overlap_acres = sum(areaovacres, na.rm = TRUE)) %>%
  ungroup()

test <- overlap_acres %>% filter(muacres < areaovacres)
test <- overlap_acres %>% filter(areaovacres > au_overlap_acres)
