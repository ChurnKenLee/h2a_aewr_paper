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
      rPackages = import ./nix/r-packages.nix { inherit pkgs; };
    in
    {
      devShells.${system}.default = cudaShell.mkPythonCudaShell {
        extraPackages = [ pkgs.R ] ++ rPackages;
      };
    };
}
