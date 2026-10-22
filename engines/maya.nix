{ pkgs, src }:
let
  metadata = builtins.fromJSON (builtins.readFile ./metadata.json);
  ggmlSource = pkgs.fetchzip {
    # Exact commit required by upstream CMakeLists.txt / MMQ patch.
    url = "https://github.com/ggml-org/llama.cpp/archive/${metadata.ggml.rev}.tar.gz";
    hash = metadata.ggml.hash;
  };
  # Inspected upstream: https://github.com/mw00/project-maya
  # No models or launcher-managed dependencies are downloaded.
  toolsPackage = pkgs.stdenv.mkDerivation {
    pname = "maya-tools";
    version = metadata.engines.maya.version; # CMake project version (not the launcher release version)
    inherit src;

    nativeBuildInputs = [ pkgs.cmake ];
    cmakeFlags = [
      "-DSTRATA_ENABLE_CUDA=OFF"
      "-DSTRATA_ENABLE_HIP=OFF"
      "-DSTRATA_NATIVE_EXPERTS=OFF" # Otherwise FetchContent downloads llama.cpp.
      "-DSTRATA_BUILD_TESTS=OFF" # Tests also fetch Catch2.
      "-DFETCHCONTENT_FULLY_DISCONNECTED=ON"
    ];

    # The default target includes x86-only expert kernels, even on ARM.
    # These independent targets need neither those kernels nor ggml/GPU libraries.
    buildPhase = ''
      runHook preBuild
      cmake --build . --parallel "$NIX_BUILD_CORES" \
        --target strata-gguf strata-dequant strata-plan
      runHook postBuild
    '';
    installPhase = ''
      runHook preInstall
      install -Dm755 strata-gguf "$out/bin/strata-gguf"
      install -Dm755 strata-dequant "$out/bin/strata-dequant"
      install -Dm755 strata-plan "$out/bin/strata-plan"
      install -Dm644 "$src/LICENSE" "$out/share/licenses/maya/LICENSE"
      runHook postInstall
    '';
    doInstallCheck = true;
    installCheckPhase = ''
      runHook preInstallCheck
      "$out/bin/strata-plan" --help
      "$out/bin/strata-plan" --vram 24000000000 --max-context 20480
      for tool in strata-gguf strata-dequant; do
        status=0
        "$out/bin/$tool" > usage.txt || status=$?
        test "$status" -eq 2
        grep -q 'usage:' usage.txt
      done
      runHook postInstallCheck
    '';

    passthru = {
      upstreamRevision = src.rev;
      inferenceSupported = false;
    };
    meta = {
      description = "Maya portable GGUF inspection and memory planning tools (no inference backend)";
      homepage = "https://github.com/mw00/project-maya";
      license = pkgs.lib.licenses.mit;
      platforms = [ "x86_64-linux" ];
      mainProgram = "strata-plan";
    };
  };

  visionPackage = pkgs.stdenv.mkDerivation {
    pname = "maya-vision-cpu";
    version = metadata.engines.maya.version;
    inherit src;
    nativeBuildInputs = [ pkgs.cmake pkgs.ninja pkgs.python3 ];
    # Patch only a writable copy of the pinned llama input (AMD_MAYA.md).
    preConfigure = ''
      cp -R ${ggmlSource} llama-source
      chmod -R u+w llama-source
      python tools/vision/glm5next_patch.py llama-source
      cmakeFlagsArray+=("-DLLAMA_DIR=$PWD/llama-source")
    '';
    cmakeDir = "../tools/vision";
    cmakeFlags = [
      "-DFETCHCONTENT_FULLY_DISCONNECTED=ON"
      "-DSTRATA_PORTABLE=ON"
      "-DMAYA_VISION_CUDA=OFF"
      "-DGGML_HIP=OFF"
      "-DGGML_SYCL=OFF"
      "-DGGML_METAL=OFF"
    ];
    buildPhase = ''
      runHook preBuild
      cmake --build . --parallel "$NIX_BUILD_CORES" --target maya-vision
      runHook postBuild
    '';
    installPhase = ''
      runHook preInstall
      install -Dm755 bin/maya-vision "$out/bin/maya-vision"
      runHook postInstall
    '';
    dontInstallCmake = true;
    meta = {
      description = "Maya portable CPU image encoder (no models)";
      license = pkgs.lib.licenses.mit;
      platforms = [ "x86_64-linux" ];
      mainProgram = "maya-vision";
    };
  };

  # Build-verified for sm_86; GPU/model execution remains untested.
  # Requires allowUnfree = true. Build explicitly via .cudaPackage; tools stay
  # independent of CUDA. Override cmakeFlags for GPUs other than sm_86 (>= 75).
  # Upstream's CUDA path always builds MMQ when native experts are enabled;
  # STRATA_PREFILL_MMQ is a HIP-only switch and must NOT be enabled here.
  cudaPackage = pkgs.cudaPackages.backendStdenv.mkDerivation {
    pname = "maya-cuda";
    version = metadata.engines.maya.version;
    inherit src;

    nativeBuildInputs = [ pkgs.cmake pkgs.git pkgs.cudaPackages.cuda_nvcc ];
    buildInputs = [
      pkgs.cudaPackages.cuda_nvtx
      pkgs.cudaPackages.cuda_profiler_api
      pkgs.cudaPackages.cuda_cudart
      pkgs.cudaPackages.libcublas
      pkgs.cudaPackages.cuda_cccl
    ];
    cmakeFlags = [
      "-DSTRATA_ENABLE_CUDA=ON"
      "-DSTRATA_ENABLE_HIP=OFF"
      "-DSTRATA_NATIVE_EXPERTS=ON"
      "-DSTRATA_PORTABLE=ON" # AVX2 baseline, not build-host CPU detection.
      "-DSTRATA_BUILD_TESTS=OFF"
      "-DCMAKE_CUDA_ARCHITECTURES=86"
      "-DCMAKE_CUDA_RUNTIME_LIBRARY=Shared"
      "-DFETCHCONTENT_FULLY_DISCONNECTED=ON"
    ];
    # CMake's MMQ copy preserves permissions before git apply. Store sources
    # are read-only, so use a writable checkout (shared with the HIP build).
    preConfigure = ''
      cp -R ${ggmlSource} ggml-source
      chmod -R u+w ggml-source
      cmakeFlagsArray+=("-DSTRATA_GGML_DIR=$PWD/ggml-source")
    '';
    # Avoid upstream's default target: it includes GPU/model-dependent tests
    # even with STRATA_BUILD_TESTS=OFF. No Python launcher/model downloads.
    buildPhase = ''
      runHook preBuild
      cmake --build . --parallel "$NIX_BUILD_CORES" --target maya maya-ds4 strata-device
      runHook postBuild
    '';
    installPhase = ''
      runHook preInstall
      for tool in maya maya-ds4 strata-device; do
        install -Dm755 "$tool" "$out/bin/$tool"
      done
      install -Dm644 "$src/LICENSE" "$out/share/licenses/maya/LICENSE"
      runHook postInstall
    '';
    # A sandbox build has neither a driver/device nor model weights.
    doCheck = false;
    passthru = {
      upstreamRevision = src.rev;
      cudaInferenceTested = false;
    };
    meta = {
      description = "Experimental Maya CUDA inference engines (inference untested)";
      homepage = "https://github.com/mw00/project-maya";
      license = pkgs.lib.licenses.mit;
      platforms = [ "x86_64-linux" ];
      mainProgram = "maya";
    };
  };
  # Upstream HIP requires MMQ, hipBLAS, and rocWMMA headers for gfx1100.
  # hipBLASLt is optional upstream; include it for the prompt GEMM path.
  # Override cmakeFlags to select CMAKE_HIP_ARCHITECTURES (default gfx1100).
  # No runtime --backend flag: these are backend-specific native binaries.
  rocmPackage = pkgs.stdenv.mkDerivation (let old = cudaPackage; in {
    inherit (old) version src preConfigure buildPhase installPhase doCheck;
    pname = "maya-rocm";
    nativeBuildInputs = [ pkgs.cmake pkgs.git pkgs.rocmPackages.llvm.clang ];
    buildInputs = with pkgs.rocmPackages; [ rocm-device-libs clr hipblas hipblaslt rocwmma ];
    cmakeFlags = builtins.filter (flag:
      !(pkgs.lib.hasPrefix "-DSTRATA_ENABLE_" flag)
      && !(pkgs.lib.hasPrefix "-DCMAKE_CUDA_" flag)
    ) old.cmakeFlags ++ [
      "-DSTRATA_ENABLE_CUDA=OFF"
      "-DSTRATA_ENABLE_HIP=ON"
      "-DSTRATA_PREFILL_MMQ=ON"
      "-DCMAKE_HIP_ARCHITECTURES=gfx1100"
      # HIP uses the unwrapped ROCm clang: buildInputs alone does not expose
      # rocWMMA's separate Nix-store include directory to HIP compilation.
      "-DCMAKE_HIP_FLAGS=-I${pkgs.rocmPackages.rocwmma}/include"
      "-DCMAKE_HIP_COMPILER=${pkgs.rocmPackages.llvm.clang}/bin/clang++"
      "-DCMAKE_HIP_COMPILER_ROCM_ROOT=${pkgs.rocmPackages.clr}"
    ];
    passthru = {
      upstreamRevision = src.rev;
      rocmInferenceTested = false;
    };
    meta = old.meta // {
      description = "Experimental Maya ROCm inference engines (inference untested)";
    };
  });
in
{
  package = rocmPackage;
  inherit rocmPackage cudaPackage toolsPackage visionPackage;
}
