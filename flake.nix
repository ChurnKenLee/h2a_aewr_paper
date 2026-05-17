{
  description = "Python + R + CUDA project environment";

  nixConfig = {
    extra-substituters = [
      "https://cache.nixos-cuda.org"
    ];
    extra-trusted-public-keys = [
      "cache.nixos-cuda.org:74DUi4Ye579gUqzH4ziL9IyiJBlDpMRn9MBN8oNan9M="
    ];
  };

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs =
    { nixpkgs, ... }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs {
        inherit system;
        config.allowUnfree = true;
      };

      cudaShell = import ./nix/cuda-shell.nix { inherit pkgs; };

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
    in
    {
      devShells.${system}.default = cudaShell.mkPythonCudaShell {
        extraPackages = [ pkgs.R ] ++ rPackages;
      };
    };
}
