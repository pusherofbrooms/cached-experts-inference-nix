{ pkgs, src }:
let
  metadata = builtins.fromJSON (builtins.readFile ./metadata.json);
in pkgs.stdenv.mkDerivation {
  pname = "strata-inkling-metal";
  version = metadata.engines.strata.version;
  inherit src;
  nativeBuildInputs = [ pkgs.cmake pkgs.ninja pkgs.python3 ];
  buildInputs = [ pkgs.apple-sdk_26 (pkgs.darwinMinVersionHook "15.0") ];
  cmakeDir = "../engine/src/inkling2";
  buildPhase = ''
    runHook preBuild
    cmake --build . --target strata-inkling
    runHook postBuild
  '';
  installPhase = ''
    runHook preInstall
    install -Dm755 strata-inkling "$out/bin/strata-inkling"
    runHook postInstall
  '';
  meta = {
    description = "Standalone Strata Inkling 2 Mini Metal runner (no models)";
    license = pkgs.lib.licenses.mit;
    platforms = [ "aarch64-darwin" ];
    mainProgram = "strata-inkling";
  };
}
