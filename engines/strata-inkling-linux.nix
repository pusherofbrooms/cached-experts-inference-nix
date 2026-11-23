{ pkgs, src, backend, rocmArchitecture }:
let
  inherit (pkgs) lib rocmPackages cudaPackages;
  hip = backend == "rocm";
  version = (builtins.fromJSON (builtins.readFile ./metadata.json)).engines.strata.version;
  cudaArchs = lib.concatMapStringsSep ";" (lib.replaceStrings [ "." ] [ "" ]) cudaPackages.flags.cudaCapabilities;
in
assert hip || backend == "cuda";
(if hip then pkgs.stdenv else cudaPackages.backendStdenv).mkDerivation {
  pname = "strata-inkling-${backend}";
  inherit version src;
  nativeBuildInputs = [ pkgs.cmake pkgs.ninja ]
    ++ lib.optionals hip [ rocmPackages.llvm.clang ]
    ++ lib.optionals (!hip) [ cudaPackages.cuda_nvcc ];
  buildInputs = lib.optionals hip [ rocmPackages.clr rocmPackages.rocm-device-libs ]
    ++ lib.optionals (!hip) [ cudaPackages.cuda_cudart cudaPackages.libcublas cudaPackages.cuda_cccl ];
  cmakeDir = "../engine/src/inkling2";
  cmakeFlags = lib.optionals hip [
    "-DSTRATA_ENABLE_HIP=ON"
    "-DSTRATA_HIP_ARCHS=${rocmArchitecture}"
    "-DSTRATA_HIP_CXX=${rocmPackages.llvm.clang}/bin/clang++"
    "-DSTRATA_ROCM_PATH=${rocmPackages.clr}"
    "-DSTRATA_ROCM_BITCODE_PATH=${rocmPackages.rocm-device-libs}/amdgcn/bitcode"
    "-DSTRATA_HIP_RUNTIME_LIB=${lib.getLib rocmPackages.clr}/lib/libamdhip64.so"
  ] ++ lib.optionals (!hip) [
    "-DSTRATA_ENABLE_CUDA=ON"
    "-DCMAKE_CUDA_ARCHITECTURES=${cudaArchs}"
    "-DSTRATA_INK_CUDA_ARCHS=${cudaArchs}"
  ];
  buildPhase = ''
    runHook preBuild
    cmake --build . --parallel "$NIX_BUILD_CORES" --target strata-inkling
    runHook postBuild
  '';
  # Install only the runner, independent of the CMake generator.
  installPhase = ''
    runHook preInstall
    install -Dm755 strata-inkling "$out/bin/strata-inkling"
    runHook postInstall
  '';
  meta = {
    description = "Experimental standalone Strata Inkling 2 Mini ${backend} runner (no models)";
    license = lib.licenses.mit;
    platforms = [ "x86_64-linux" ];
    mainProgram = "strata-inkling";
  };
}
