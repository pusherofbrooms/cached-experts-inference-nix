{
  # RDNA3 RX 7900 XTX/XT. Check your card with rocminfo.
  # Upstream's wave32 HIP path is not a promise of support for all AMD cards.
  rocmArchitecture = "gfx1100";
  # CMake notation: e.g. "86;89" for Ampere and Ada.
  cudaArchitectures = "86";
}
