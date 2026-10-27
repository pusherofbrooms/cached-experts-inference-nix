{ pkgs, src }:
let
  metadata = builtins.fromJSON (builtins.readFile ./metadata.json);
  ggml = pkgs.fetchzip {
    url = "https://github.com/ggml-org/llama.cpp/archive/${metadata.ggml.rev}.tar.gz";
    hash = metadata.ggml.hash;
  };
  # Apple's SDK-26 headers, pinned by upstream metal/setup_mac.py.
  metalCppArchive = pkgs.fetchurl {
    url = "https://developer.apple.com/metal/cpp/files/metal-cpp_26.zip";
    hash = "sha256-TfPAeLmq3LUWIS6csDAEy8XOmj6cBo+jFE0CHbWFo6Q=";
  };
  metalCpp = pkgs.runCommand "metal-cpp-26" { nativeBuildInputs = [ pkgs.unzip ]; } ''
    unzip ${metalCppArchive}
    mv metal-cpp "$out"
  '';
in pkgs.stdenv.mkDerivation {
  pname = "strata-metal";
  version = metadata.engines.strata.version;
  inherit src;
  nativeBuildInputs = [ pkgs.cmake pkgs.ninja pkgs.makeWrapper ];
  # metal-cpp 26 uses APIs absent from the default SDK 14. The SDK hook
  # selects the sysroot; never consult the host's xcrun or installed Xcode.
  buildInputs = [ pkgs.apple-sdk_26 (pkgs.darwinMinVersionHook "15.0") ];
  cmakeDir = "../metal";
  cmakeFlags = [
    "-DMETAL_CPP_DIR=${metalCpp}"
    "-DSTRATA_GGML_DIR=${ggml}"
    "-DFETCHCONTENT_FULLY_DISCONNECTED=ON"
    "-DGGML_NATIVE=OFF"
    "-DGGML_CPU_ARM_ARCH=armv8-a"
  ];
  buildPhase = ''
    runHook preBuild
    cmake --build . --parallel "$NIX_BUILD_CORES" --target strata
    runHook postBuild
  '';
  installPhase = ''
    runHook preInstall
    mkdir -p "$out/bin/metal"
    cp strata "$out/bin/strata-real"
    makeWrapper "$out/bin/strata-real" "$out/bin/strata" \
      ${import ../runtime/metal-wrapper.nix { }}
    cp -r ../metal/kernels "$out/bin/metal/"
    cp -r ../metal/src/engine/msl "$out/bin/metal/engine-msl"
    cp ../third_party/ggml/ggml-common.h "$out/bin/metal/"
    cp -r ../data "$out/bin/metal/"
    runHook postInstall
  '';
  dontInstallCmake = true;
  meta = {
    description = "Strata Apple silicon Metal text engine (no models)";
    license = pkgs.lib.licenses.mit;
    platforms = [ "aarch64-darwin" ];
    mainProgram = "strata";
  };
}
