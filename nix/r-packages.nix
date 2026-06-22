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

  # Faster tools
  collapse

  # Inference
  fixest
]
