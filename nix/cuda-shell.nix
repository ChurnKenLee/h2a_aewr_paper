{ pkgs }:

let
  cuda = pkgs.cudaPackages_13;

  cudaRoot = pkgs.buildEnv {
    name = "cuda-project-root";
    paths = with cuda; [
      cudatoolkit
      cuda_cccl
      cuda_cudart
      cuda_cupti
      cuda_nvcc
      cuda_nvrtc
      cuda_nvtx
      cudnn
      cudnn-frontend
      cutlass
      libcublas
      libcufft
      libcurand
      libcusolver
      libcusparse
      libcusparse_lt
      libnvjitlink
      nccl
    ];
    pathsToLink = [
      "/bin"
      "/include"
      "/lib"
      "/lib64"
      "/nvvm"
      "/share"
    ];
    extraOutputsToInstall = [
      "out"
      "dev"
      "lib"
      "static"
    ];
    ignoreCollisions = true;
  };

  nativeLibraries = with pkgs; [
    expat
    zlib
    openssl
    libgcc
    stdenv.cc.cc.lib
  ];

  cudaRuntimeLibraryPath = "${cudaRoot}/lib:${cudaRoot}/lib64";
  cudaLinkLibraryPath = "${cudaRuntimeLibraryPath}:${cudaRoot}/lib/stubs:${cudaRoot}/lib64/stubs";
  driverLibraryPath = "/run/opengl-driver/lib";
  libraryPath = "${driverLibraryPath}:${pkgs.lib.makeLibraryPath nativeLibraries}:${cudaRuntimeLibraryPath}";

  cudaDoctor = pkgs.writeShellApplication {
    name = "cuda-doctor";
    runtimeInputs = with pkgs; [
      coreutils
      gcc
      gnugrep
    ];
    text = ''
      set -euo pipefail

      echo "CUDA_HOME=$CUDA_HOME"
      command -v nvcc
      "$NVCC" --version | head -n 4

      test -f "$CUDA_HOME/include/cuda_runtime.h"
      test -e "$CUDA_HOME/lib/stubs/libcuda.so" || test -e "$CUDA_HOME/lib64/stubs/libcuda.so"
      test -e /run/opengl-driver/lib/libcuda.so

      tmp="$(mktemp -d)"
      trap 'rm -rf "$tmp"' EXIT

      cat > "$tmp/check.cu" <<'EOF'
      #include <cuda_runtime.h>
      int main() { return 0; }
      EOF

      "$NVCC" -c "$tmp/check.cu" -o "$tmp/check.o"

      printf 'int main() { return 0; }\n' > "$tmp/check.cc"
      g++ "$tmp/check.cc" \
        -L/run/opengl-driver/lib \
        -L"$CUDA_HOME/lib/stubs" \
        -L"$CUDA_HOME/lib64/stubs" \
        -lcuda \
        -o "$tmp/check-link"

      echo "cuda-doctor: ok"
    '';
  };
in
{
  inherit cudaRoot;

  mkPythonCudaShell =
    {
      extraPackages ? [ ],
    }:
    pkgs.mkShell {
      packages =
        with pkgs;
        [
          uv
          coreutils
          gnugrep
          gcc
          gnumake
          cmake
          ninja
          pkg-config
          patchelf
          cudaRoot
          cudaDoctor
        ]
        ++ extraPackages;

      UV_PROJECT_ENVIRONMENT = ".venv";
      UV_MANAGED_PYTHON = "true";
      CUDA_HOME = "${cudaRoot}";
      CUDA_PATH = "${cudaRoot}";
      CUDA_ROOT = "${cudaRoot}";
      CUDAToolkit_ROOT = "${cudaRoot}";
      CUDACXX = "${cudaRoot}/bin/nvcc";
      NVCC = "${cudaRoot}/bin/nvcc";
      CPATH = "${cudaRoot}/include";
      LIBRARY_PATH = "${driverLibraryPath}:${cudaLinkLibraryPath}";
      LD_LIBRARY_PATH = libraryPath;
      CMAKE_PREFIX_PATH = "${cudaRoot}";
      CMAKE_LIBRARY_PATH = "${driverLibraryPath}:${cudaLinkLibraryPath}";
      EXTRA_LDFLAGS = "-L${driverLibraryPath} -L${cudaRoot}/lib -L${cudaRoot}/lib64 -L${cudaRoot}/lib/stubs";
      EXTRA_CCFLAGS = "-I${cudaRoot}/include";

      # Avoid Triton's hard-coded /sbin/ldconfig path when possible.
      TRITON_LIBCUDA_PATH = "/run/opengl-driver/lib";
      TRITON_PTXAS_PATH = "${cudaRoot}/bin/ptxas";
      TRITON_PTXAS_BLACKWELL_PATH = "${cudaRoot}/bin/ptxas";

      shellHook = ''
        export PATH="${cudaRoot}/bin:$PATH"

        project_cache_key="$(printf '%s' "$PWD" | sha256sum | cut -c1-16)"
        export NIX_CUDA_PROJECT_CACHE="''${XDG_CACHE_HOME:-$HOME/.cache}/nix-cuda-projects/$project_cache_key"
        mkdir -p \
          "$NIX_CUDA_PROJECT_CACHE/flashinfer-workspace" \
          "$NIX_CUDA_PROJECT_CACHE/triton" \
          "$NIX_CUDA_PROJECT_CACHE/torch-extensions" \
          "$NIX_CUDA_PROJECT_CACHE/nv"

        export FLASHINFER_WORKSPACE_BASE="$NIX_CUDA_PROJECT_CACHE/flashinfer-workspace"
        export TRITON_CACHE_DIR="$NIX_CUDA_PROJECT_CACHE/triton"
        export TORCH_EXTENSIONS_DIR="$NIX_CUDA_PROJECT_CACHE/torch-extensions"
        export CUDA_CACHE_PATH="$NIX_CUDA_PROJECT_CACHE/nv"

        if grep -R -q '/run/current-system/sw/bin/nvcc' "$HOME/.cache/flashinfer" 2>/dev/null; then
          rm -rf "$HOME/.cache/flashinfer"
        fi
      '';
    };
}
