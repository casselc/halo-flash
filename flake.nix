{
  description = "halo-flash: a Strix Halo (gfx1151) engine for Qwen3.8-Flash-Next";

  # One toolchain for building, running and profiling: the ROCm HIP compiler,
  # runtime and profiler all come from the same pinned nixpkgs. The revision
  # is gufo's (its flake.lock), so both engines are measured with identical
  # ROCm.
  inputs.nixpkgs.url = "github:NixOS/nixpkgs/2fcb964de67fcf60b43471c55d5d99e61a9ccb5a";

  outputs =
    { self, nixpkgs }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs { inherit system; };
      rocm = pkgs.rocmPackages;
    in
    {
      devShells.${system}.default = pkgs.mkShell {
        packages = [
          pkgs.cmake
          pkgs.ninja
          pkgs.python3
          pkgs.sqlite
          rocm.clr # HIP runtime (libamdhip64) and hipcc
          rocm.llvm.clang # the HIP device compiler hipcc drives
          rocm.llvm.llvm # llvm-readelf for tools/kernel_report.py
          rocm.rocminfo
          rocm.rocprofiler-sdk # rocprofv3, matching the runtime above
        ];
        env = {
          HALO_TOOLCHAIN = "nix-rocm-${rocm.clr.version}";
          ROCM_PATH = "${rocm.clr}";
          HIP_PATH = "${rocm.clr}";
        };
        shellHook = ''
          echo "halo-flash toolchain: ROCm ${rocm.clr.version} (hipcc $(command -v hipcc))"
        '';
      };
    };
}
