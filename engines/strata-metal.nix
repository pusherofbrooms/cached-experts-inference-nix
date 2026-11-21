{ pkgs, src }:
let
  metadata = builtins.fromJSON (builtins.readFile ./metadata.json);
  # Apple's SDK-26 headers, as used by upstream's Metal packaging.
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
  cmakeDir = "../engine/metal";
  cmakeFlags = [
    "-DMETAL_CPP_DIR=${metalCpp}"
    "-DFETCHCONTENT_FULLY_DISCONNECTED=ON"
    "-DSTRATA_BUILD_TESTS=OFF"
    "-DSTRATA_OWN_CPU_QUANT=ON"
  ];
  buildPhase = ''
    runHook preBuild
    cmake --build . --parallel "$NIX_BUILD_CORES" --target strata
    runHook postBuild
  '';
  installPhase = ''
    runHook preInstall
    resources="$out/bin/metal"
    mkdir -p "$resources"/{kernels,engine-msl,data}
    cp strata "$out/bin/strata-real"
    makeWrapper "$out/bin/strata-real" "$out/bin/strata" \
      ${import ../runtime/metal-wrapper.nix { }}
    cp ../engine/metal/kernels/*.metal "$resources/kernels/"
    cp ../engine/metal/src/engine/msl/*.metal "$resources/engine-msl/"
    # The runtime reads this header as MSL source; the legacy file is just a forwarding include.
    cp ../engine/include/strata/quant/strata-quant-common.h "$resources/"
    cp ../terminal/data/{expert-profile,draft_vocab}.bin "$resources/data/"
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
