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
          # Host code: nixpkgs' wrapped clang, so headers, libc and libstdc++
          # all come from this nixpkgs. ROCm's own clang stays off PATH (it is
          # not wrapped: it reads the host's /usr/include) and is used only
          # through hipcc, for device code.
          pkgs.clang
          pkgs.llvm # llvm-readelf for tools/kernel_report.py
          rocm.clr # HIP runtime (libamdhip64) and hipcc
          rocm.rocminfo
          rocm.rocprofiler-sdk # rocprofv3, matching the runtime above
          pkgs.openssl # libcrypto: spill file encryption (AES-256-GCM, HKDF, ECDSA for yard JWTs; docs/SESSIONS.md 10)
        ];
        env = {
          HALO_TOOLCHAIN = "nix-rocm-${rocm.clr.version}";
          HALO_CC = "${pkgs.clang}/bin/clang";
          HALO_CXX = "${pkgs.clang}/bin/clang++";
          ROCM_PATH = "${rocm.clr}";
          HIP_PATH = "${rocm.clr}";
        };
        shellHook = ''
          echo "halo-flash toolchain: ROCm ${rocm.clr.version} (hipcc $(command -v hipcc))"
        '';
      };
    };
}
