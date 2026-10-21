{ pkgs, src }:
let
  metadata = builtins.fromJSON (builtins.readFile ./metadata.json);
  inherit (pkgs) lib;
  # Cache this fixed-output input before offline builds; never run setup.py
  # (it downloads Python packages, engines and model weights).
  ggml = pkgs.fetchzip {
    url = "https://github.com/ggml-org/llama.cpp/archive/${metadata.ggml.rev}.tar.gz";
    hash = "${metadata.ggml.hash}";
  };
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

      cmakeFlags = [
        "-DFETCHCONTENT_FULLY_DISCONNECTED=ON"
        "-DSTRATA_GGML_DIR=${ggml}"
        "-DSTRATA_BUILD_TESTS=OFF"
        "-DSTRATA_BUILD_CONVERSATION_TESTS=OFF"
        "-DSTRATA_ENABLE_CUDA=${if nvidia then "ON" else "OFF"}"
        "-DSTRATA_ENABLE_HIP=${if hip then "ON" else "OFF"}"
        "-DSTRATA_HIP_GFX906=OFF"
        "-DSTRATA_NATIVE_EXPERTS=${if tools then "OFF" else "ON"}"
        # Upstream otherwise forces GGML_NATIVE=ON. Portable means AVX2/FMA/F16C,
        # not arbitrary x86 CPUs; wider expert kernels are runtime-dispatched.
        "-DSTRATA_PORTABLE=ON"
        "-DCMAKE_BUILD_TYPE=Release"
      ] ++ lib.optionals hip [
        "-DCMAKE_HIP_COMPILER=${rocm.llvm.clang}/bin/clang++"
        "-DCMAKE_HIP_COMPILER_ROCM_ROOT=${rocm.clr}"
        # Upstream-validated wave32 default; override cmakeFlags for other cards.
        # CDNA/wave64 needs a separate STRATA_HIP_GFX906 build, not this backend.
        "-DCMAKE_HIP_ARCHITECTURES=gfx1100"
        "-DSTRATA_PREFILL_MMQ=ON"
        "-DSTRATA_MMQ_KQUANTS=ON"
      ] ++ lib.optionals nvidia [
        "-DCMAKE_CUDA_ARCHITECTURES=${lib.concatMapStringsSep ";" (lib.replaceStrings [ "." ] [ "" ]) cuda.flags.cudaCapabilities}"
      ];

      # No upstream install rules. Build inference, not the default benchmark and
      # parity-test suite. CUDA's optional K-quant DSO must stay beside the engine.
      buildPhase = ''
        runHook preBuild
        cmake --build . --parallel "$NIX_BUILD_CORES" --target ${targets}
        runHook postBuild
      '';
      installPhase = ''
        runHook preInstall
        mkdir -p "$out/bin"
        cp ${targets} "$out/bin/"
      '' + lib.optionalString nvidia ''
        cp libstrata_kq.so "$out/bin/"
      '' + ''
        runHook postInstall
      '';
      dontInstallCmake = true;
      meta = {
        description = if tools then "Strata GGUF inspection and planning tools (no inference)" else "Strata ${backend} inference engine";
        homepage = "https://github.com/Niko1221/Strata";
        license = lib.licenses.mit;
        platforms = [ "x86_64-linux" ];
        mainProgram = if tools then "strata-plan" else "strata strata-device";
      };
    };
in
rec {
  package = rocmPackage;
  rocmPackage = mkPackage "rocm";
  cudaPackage = mkPackage "cuda";
  toolsPackage = mkPackage "tools";
}
