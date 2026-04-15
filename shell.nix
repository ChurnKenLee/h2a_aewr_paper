{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  packages = with pkgs; [
    R
    rPackages.languageserver

    # Data input
    rPackages.here
    rPackages.arrow
    rPackages.haven
    rPackages.readxl
    rPackages.foreign
    
    # Tidyverse
    rPackages.tidyverse
    rPackages.tidylog

    # Convenient data sources
    rPackages.tidycensus
    rPackages.ipumsr

    # Graphics
    rPackages.ggplot2
    rPackages.ggnewscale

    # Cleaning tools
    rPackages.janitor

    # Faster tools
    rPackages.collapse

    # Vectors
    rPackages.sf
  ];
}
