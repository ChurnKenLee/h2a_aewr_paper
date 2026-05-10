{
  description = "Python + R project environment";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs =
    { nixpkgs, ... }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs {
        inherit system;
        config.allowUnfree = true;
      };

      rPackages = with pkgs.rPackages; [
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
      ];

      nativeLibraries = with pkgs; [
        # Required C libraries for uv-downloaded wheels.
        expat
        zlib
        stdenv.cc.cc.lib
      ];
    in
    {
      devShells.${system}.default = pkgs.mkShell {
        packages =
          with pkgs;
          [
            # R environment
            R
          ]
          ++ rPackages
          ++ [

            # Python environment
            uv
            python3
          ]
          ++ nativeLibraries;

        LD_LIBRARY_PATH = "${pkgs.lib.makeLibraryPath nativeLibraries}:/run/opengl-driver/lib";
        UV_PROJECT_ENVIRONMENT = ".venv-nixos";
      };
    };
}
