{
  description = "Development environment for python.";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = {
    nixpkgs,
    flake-utils,
    ...
  }:
    flake-utils.lib.eachDefaultSystem (
      system: let
        pkgs = import nixpkgs {
          inherit system;
          config.allowUnfree = true;
        };

        cudaPkgs = pkgs.cudaPackages;
      in {
        devShells.default = with pkgs;
          mkShell {
            buildInputs = [
              python312
              uv

              cudaPkgs.cuda_nvcc
              cudaPkgs.cudatoolkit
              cudaPkgs.cuda_cudart
            ];

            CUDA_HOME = "${cudaPkgs.cudatoolkit}";
            CPATH = "${cudaPkgs.cudatoolkit}/include:${cudaPkgs.cuda_cudart}/include";

            LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [
              pkgs.zlib
              pkgs.stdenv.cc.cc.lib
              pkgs.gcc.cc
              cudaPkgs.cudatoolkit
              cudaPkgs.cuda_cudart
            ];
          };
      }
    );
}
