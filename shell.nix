{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  packages = with pkgs;[
    # --- R Environment ---
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

    # --- Python Environment ---
    uv
    python3

    # Required C-libraries for uv downloaded wheels (like rasterio)
    expat
    zlib
    stdenv.cc.cc.lib
  ];

  # Safely set up environment variables just for this project
  shellHook = ''
    # Provide C-libraries for uv wheels AND preserve NVIDIA OpenGL drivers
    export LD_LIBRARY_PATH="${pkgs.lib.makeLibraryPath (with pkgs;[ expat zlib stdenv.cc.cc.lib ])}:/run/opengl-driver/lib:\$LD_LIBRARY_PATH"

    # Tell uv to use a custom venv name for this project
    export UV_PROJECT_ENVIRONMENT=".venv-nixos"
  '';
}
