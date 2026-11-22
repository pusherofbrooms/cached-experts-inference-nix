{ pkgs, src }:
let
  inherit (pkgs) lib rocmPackages;
  version = (builtins.fromJSON (builtins.readFile ./metadata.json)).engines.strata.version;
in
pkgs.stdenv.mkDerivation {
  pname = "strata-hy4-rocm";
  inherit version src;
  nativeBuildInputs = [ pkgs.cmake pkgs.ninja rocmPackages.llvm.clang ];
  buildInputs = [ rocmPackages.clr rocmPackages.rocm-device-libs pkgs.zstd ];

  # The tagged Hy4 runner is not a target of the regular Strata CMake project.
  patches = [ ./strata-hy4-json.patch ];
  postPatch = ''
    mkdir -p engine/hy4-nix
    cp ${./strata-hy4.CMakeLists.txt} engine/hy4-nix/CMakeLists.txt
  '';
  cmakeDir = "../engine/hy4-nix";
  cmakeFlags = [
    "-DCMAKE_HIP_COMPILER=${rocmPackages.llvm.clang}/bin/clang++"
    "-DCMAKE_HIP_COMPILER_ROCM_ROOT=${rocmPackages.clr}"
    "-DCMAKE_HIP_ARCHITECTURES=gfx1100"
  ];

  doInstallCheck = true;
  installCheckPhase = ''
    runHook preInstallCheck
    test "$("$out/bin/strata-hy4" --version)" = "strata-hy4 ${version}"
    runHook postInstallCheck
  '';
  meta = {
    description = "Experimental Strata Hy4-Flash runner (ROCm)";
    homepage = "https://github.com/Niko1221/Strata";
    license = lib.licenses.mit;
    platforms = [ "x86_64-linux" ];
    mainProgram = "strata-hy4";
  };
}
