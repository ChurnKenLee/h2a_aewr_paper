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
    cudaRoot
  ];

  libraryPath = "${pkgs.lib.makeLibraryPath nativeLibraries}:${cudaRoot}/lib:${cudaRoot}/lib64:/run/opengl-driver/lib";
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
          python3
          gcc
          gnumake
          cmake
          ninja
          pkg-config
          patchelf
          cudaRoot
        ]
        ++ extraPackages;

      CUDA_HOME = "${cudaRoot}";
      CUDA_PATH = "${cudaRoot}";
      CUDA_ROOT = "${cudaRoot}";
      CUDAToolkit_ROOT = "${cudaRoot}";
      CPATH = "${cudaRoot}/include";
      LIBRARY_PATH = "${cudaRoot}/lib:${cudaRoot}/lib64";
      LD_LIBRARY_PATH = libraryPath;
      CMAKE_PREFIX_PATH = "${cudaRoot}";
      EXTRA_LDFLAGS = "-L${cudaRoot}/lib -L${cudaRoot}/lib64";
      EXTRA_CCFLAGS = "-I${cudaRoot}/include";

      # Avoid Triton's hard-coded /sbin/ldconfig path when possible.
      TRITON_LIBCUDA_PATH = "/run/opengl-driver/lib";
      TRITON_PTXAS_PATH = "${cudaRoot}/bin/ptxas";
      TRITON_PTXAS_BLACKWELL_PATH = "${cudaRoot}/bin/ptxas";

      shellHook = ''
        export PATH="${cudaRoot}/bin:$PATH"
      '';
    };
}
