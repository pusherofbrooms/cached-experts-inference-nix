{ pkgs, src }:
let
  metadata = builtins.fromJSON (builtins.readFile ./metadata.json);
  inherit (pkgs) lib;
  rocm = pkgs.rocmPackages;
  cuda = pkgs.cudaPackages;
  mkPackage = backend:
    let
      hip = backend == "rocm";
      nvidia = backend == "cuda";
      tools = backend == "tools";
      targets = if tools then "strata-gguf strata-dequant strata-plan" else "strata strata-device";
    in
    (if nvidia then cuda.backendStdenv else pkgs.stdenv).mkDerivation {
      pname = "strata-${backend}";
      version = metadata.engines.strata.version;
      inherit src;

      nativeBuildInputs = [ pkgs.cmake pkgs.ninja ]
        ++ lib.optionals hip [ rocm.hipcc rocm.llvm.clang ]
        ++ lib.optionals nvidia [ cuda.cuda_nvcc ];
      buildInputs = lib.optionals hip [
        rocm.rocm-device-libs rocm.clr rocm.hipblas rocm.rocblas rocm.hipblaslt
      ] ++ lib.optionals nvidia [ cuda.cuda_cudart cuda.libcublas cuda.cuda_cccl ];

      cmakeDir = "../engine";
      cmakeFlags = [
        "-DFETCHCONTENT_FULLY_DISCONNECTED=ON"
        "-DSTRATA_BUILD_TESTS=OFF"
        "-DSTRATA_BUILD_CONVERSATION_TESTS=OFF"
        "-DSTRATA_ENABLE_CUDA=${if nvidia then "ON" else "OFF"}"
        "-DSTRATA_ENABLE_HIP=${if hip then "ON" else "OFF"}"
        "-DSTRATA_HIP_GFX906=OFF"
        "-DSTRATA_NATIVE_EXPERTS=${if tools then "OFF" else "ON"}"
        "-DSTRATA_OWN_CPU_QUANT=ON"
        # Keep the portable x86 baseline (AVX2/FMA/F16C), not build-host detection.
        "-DSTRATA_PORTABLE=ON"
        "-DCMAKE_BUILD_TYPE=Release"
      ] ++ lib.optionals hip [
        "-DCMAKE_HIP_COMPILER=${rocm.llvm.clang}/bin/clang++"
        "-DCMAKE_HIP_COMPILER_ROCM_ROOT=${rocm.clr}"
        # Upstream-validated wave32 default; override cmakeFlags for other cards.
        # CDNA/wave64 needs a separate STRATA_HIP_GFX906 build, not this backend.
        "-DCMAKE_HIP_ARCHITECTURES=gfx1100"
      ] ++ lib.optionals nvidia [
        "-DCMAKE_CUDA_ARCHITECTURES=${lib.concatMapStringsSep ";" (lib.replaceStrings [ "." ] [ "" ]) cuda.flags.cudaCapabilities}"
      ];

      # Build only the intended tools/engine, not benchmarks or parity tests.
      buildPhase = ''
        runHook preBuild
        cmake --build . --parallel "$NIX_BUILD_CORES" --target ${targets}
        runHook postBuild
      '';
      installPhase = ''
        runHook preInstall
        mkdir -p "$out/bin"
        cp ${targets} "$out/bin/"
        runHook postInstall
      '';
      dontInstallCmake = true;
      meta = {
        description = if tools then "Strata GGUF inspection and planning tools (no inference)" else "Strata ${backend} inference engine";
        homepage = "https://github.com/Niko1221/Strata";
        license = lib.licenses.mit;
        platforms = [ "x86_64-linux" ];
        mainProgram = if tools then "strata-plan" else "strata";
      };
    };
in
rec {
  package = rocmPackage;
  rocmPackage = mkPackage "rocm";
  cudaPackage = mkPackage "cuda";
  toolsPackage = mkPackage "tools";
  # The standalone Strata reader does not need llama.cpp or an inference backend.
  visionPackage = pkgs.stdenv.mkDerivation {
    pname = "strata-vision-cpu";
    version = metadata.engines.strata.version;
    inherit src;
    nativeBuildInputs = [ pkgs.cmake pkgs.ninja ];
    # Keep the encoder independent of the Metal text engine/toolchain.
    buildInputs = lib.optionals pkgs.stdenv.hostPlatform.isDarwin [
      pkgs.apple-sdk_26 (pkgs.darwinMinVersionHook "15.0")
    ];
    cmakeDir = "../terminal/tools/vision";
    cmakeFlags = [
      "-DFETCHCONTENT_FULLY_DISCONNECTED=ON"
      "-DSTRATA_VISION_ORACLES=OFF"
      "-DSTRATA_PORTABLE=ON"
      "-DSTRATA_VISION_CUDA=OFF"
      "-DSTRATA_VISION_HIP=OFF"
      "-DSTRATA_VISION_SYCL=OFF"
      "-DSTRATA_VISION_METAL=OFF"
    ];
    buildPhase = ''
      runHook preBuild
      cmake --build . --parallel "$NIX_BUILD_CORES" --target strata-vision
      runHook postBuild
    '';
    installPhase = ''
      runHook preInstall
      install -Dm755 bin/strata-vision "$out/bin/strata-vision"
      runHook postInstall
    '';
    dontInstallCmake = true;
    meta = {
      description = "Strata portable CPU image encoder (no models)";
      license = lib.licenses.mit;
      platforms = [ "x86_64-linux" "aarch64-darwin" ];
      mainProgram = "strata-vision";
    };
  };
}
