{ pkgs }:

with pkgs.rPackages;
[
  # Project config
  dotenv

  # Language support
  languageserver

  # Data input
  here
  arrow
  haven
  readxl
  writexl
  foreign
  janitor

  # Tidyverse
  tidyverse
  tidylog

  # Graphics
  ggplot2
  cowplot
  ggthemes
  ggfixest
  scales

  # Faster tools
  collapse

  # Inference
  fixest
  MatchIt

  # Geospatial tools
  sf
  ggspatial

  # Data download
  fredr
  ipumsr
  tidycensus
]
