# Cached expert inference engines

Model-free Nix builds for Strata and Maya (x86_64-linux). Native inference
packages have ROCm and CUDA variants; see [model notes](docs/model-preparation.md).

## Building and running

Target: **x86_64 Linux** with Nix flakes enabled. Install a working host GPU driver
and ensure your user can access the GPU devices. GPU drivers are not installed by
this flake. The RX 7900 XTX default is `gfx1100`; edit `settings.nix` to select
another architecture. CUDA architectures are configured there separately.

```sh
# ROCm builds (also the default for .#strata and .#maya)
nix build .#strata-rocm --out-link result-strata
nix build .#maya-rocm --out-link result-maya

# Optional CUDA builds
nix build .#strata-cuda --out-link result-strata-cuda
nix build .#maya-cuda --out-link result-maya-cuda

# After downloading, packing and generating config as described below:
nix run .#server-strata -- --config /data/strata.json
nix run .#server-maya -- --config /data/maya.json
```

Servers bind to loopback by default. See [model preparation](docs/model-preparation.md)
for complete download → pack → config → HTTP server examples and tuning options.
Both ROCm engines build successfully for `gfx1100`. Real model conversion and
GPU inference still require validation; passing builds and unit tests is not an
end-to-end validation.

### GPU support status and architecture selection

Both engines default to ROCm `gfx1100` (RX 7900 XT/XTX); CUDA defaults to
architecture `86`. These are configured build targets, **not a tested GPU
compatibility matrix**. Strata 0.1.47 and Maya 0.1.27 have successfully compiled
with the pinned ROCm toolchain for `gfx1100`; real-model inference remains
unvalidated here. Both engines also successfully compiled with the pinned CUDA
toolchain for architecture `86`; neither has been run on CUDA hardware here.
No CUDA hardware is available for local testing; CUDA inference remains
experimental pending external inference reports. Passing Python tests does not validate GPU
kernels, driver compatibility or numerical correctness.

To target another AMD GPU, identify its architecture with `rocminfo` (if needed,
run `nix shell nixpkgs#rocmPackages.rocminfo -c rocminfo`), then edit
`settings.nix`:

```nix
rocmArchitecture = "gfx1101"; # Example target only; not a support claim.
```

Build only the engine you need:

```sh
nix build .#strata-rocm --out-link result-strata
# Or independently:
nix build .#maya-rocm --out-link result-maya
```

This setting supplies `CMAKE_HIP_ARCHITECTURES` to both engine definitions and
is also used by their ROCm preparation packages. Only the selected package is
built. Changing it changes the build derivation and can require recompilation;
it is not a runtime GPU switch. Regenerate your server config with the matching
preparation package after rebuilding so it references the new engine path.

Architecture selection alone does not enable arbitrary AMD hardware. Strata's
current HIP configuration uses its wave32 path and explicitly disables
`STRATA_HIP_GFX906`; CDNA/wave64 support needs a separate build configuration,
not just a different `gfx` string. Maya's HIP configuration is likewise centered
on `gfx1100` with MMQ and rocWMMA; other targets are not validated here. Host GPU
drivers and device permissions remain external to this flake.

For GPU bug reports, include the engine/input revision, GPU model and architecture,
driver version, build command and logs. For inference failures, also include the
model/quantization, preparation options and runtime configuration (redacting
private paths or credentials).

## Local runtime preparation

**Desktop GPU headroom:** Strata config generation enables `--expert-cache auto`.
If the GPU also drives your desktop, consider adding `--vram-reserve-mib 2048`
to the config command to leave room for desktop applications. This is a starting
point, not a universal requirement; adjust for your VRAM capacity and workload.
No reserve is imposed by this wrapper unless requested. Re-generating an existing
config requires `--force`; include the reserve flag again. This headroom is not an
OS-enforced reservation or a guarantee against GPU stalls.

Downloads remain separate: `nix run .#download-model -- REPO DEST --revision COMMIT --include '*.gguf'`.
No wrapper selects or downloads a model. Keep every GGUF shard together.

```sh
nix run .#prepare-strata -- strata pack --gguf /data/model-00001-of-00002.gguf \
  --pack /data/strata-pack --experts-bin
nix run .#prepare-strata -- strata config --gguf /data/model-00001-of-00002.gguf \
  --pack /data/strata-pack --config /data/strata.json
```

For Maya use `nix run .#prepare-maya -- maya pack` (GLM workflow, usually `--compat-bf16`) and
`nix run .#prepare-maya -- maya config` with the same explicit path arguments. The packaged engine is
selected automatically; `--exe /absolute/bin/maya` is an optional override.
For example, keep `/data/models/model-00001-of-00002.gguf` and all its siblings
in `/data/models`, and use `--pack /data/models/pack` for both Maya commands.
Alternatively, manually symlink every shard (with its original basename) into
the existing pack's parent, pointing to the intended downloaded files. The
wrapper never moves files or creates these links; layout-only symlink repairs
need no repack. Rerun config with `--force` to safely replace its JSON/manifest;
input and output-alias protections still apply. Both engines configure a native
log at `<config>.log`, protected by the same collision checks.

Maya config optionally accepts `--dflash` pointing to an existing local drafter
GGUF; it is never enabled by default and is rejected for Strata and pack.
DFlash2 on AMD is experimental. The GLM-5.3-Flash-DFlash2 drafter is licensed
CC BY-NC-ND 4.0; review its terms before downloading or using it. Download it
separately to, for example,
`~/.local/share/project-maya/drafters/dflash2/GLM-5.3-Flash-DFlash2-Q4_K_M.gguf`.
The wrapper does not download or modify the drafter or validate compatibility.
A separate config is advisable; retain your existing VRAM reserve explicitly:

```sh
nix run .#prepare-maya -- maya config --gguf /data/models/model-00001-of-00002.gguf \
  --pack /data/models/pack --config /data/maya-dflash.json \
  --dflash ~/.local/share/project-maya/drafters/dflash2/GLM-5.3-Flash-DFlash2-Q4_K_M.gguf \
  --vram-reserve-mib 2048
```

Maya configs use `--glm-pack`; Strata uses `--pack --native`. Strata PLE models
may require explicit `--ple-gguf /data/shard.gguf` on config generation.
Compatibility and memory sizing remain the user's responsibility; these are not
universal GGUF engines. DeepSeek/EXL3 preparation is not implemented.

Pack and config are deliberately separate operations. Strata config generation needs
no model files. Maya pack and config require every input shard to exist and be
visible by the same basename in `pack.parent`, resolving to the same input file.
Neither config workflow validates model contents or compatibility. Pack invokes the pinned upstream
`tools/iq_pack.py` directly from packaged source, with Nix Python dependencies
and gguf-py pinned to llama.cpp `3cf03257f219afbe7334045ff7c6a06ac68c627d`.
No installers or upstream checkout modifications are used. `--base` permits
explicit dense sharing; `--compat-bf16` can introduce rounding; `--experts-bin`
can consume substantial disk. Original source shards must remain available.

Existing outputs/manifests are refused unless `--force`. Force allows upstream
in-place pack updates, not a clean rebuild; prefer a fresh directory. Failures
may leave partial packs. Sidecar `.manifest.json` files record revisions,
absolute paths, options and commands, not model hashes or download provenance.

Matching HTTP frontends are Nix packaged: `nix run .#server-strata -- --config /data/run.json`
(or `server-maya`). They bind loopback by default; no installers run.
Preparation packages are independent: `prepare-<engine>` and
`prepare-<engine>-rocm` include only that engine's ROCm executable;
`prepare-<engine>-cuda` includes only its CUDA executable. The positional engine
argument is retained and must match the package. `--exe` overrides the config
path, but does not remove the packaged GPU dependency.

For GPU-free packing/config generation use `prepare-strata-tools` or
`prepare-maya-tools`. These include only the corresponding source and Python
packing dependencies; config requires an explicit `--exe /absolute/bin/engine`.
They do not build a native engine or GPU toolkit.

Compatibility aliases `prepare-model` and `prepare-model-cuda` now mean **Strata
only** (ROCm and CUDA respectively). Old Maya commands must switch to
`prepare-maya` / `prepare-maya-cuda`; a mismatched engine is rejected rather
than silently generating a config for another engine.

Pack preparation validates sibling completeness with upstream Model and records
Strata PLE discovery in the sidecar; config generation can read that sidecar without
opening model weights. Symlink filenames are retained for sibling discovery.
GPU inference and real model conversion have not been validated here.

Run `python3 -m unittest discover -s tests -v`; the flake also exposes
`checks.x86_64-linux.prepare-model`. Adapters are registered in
`runtime/prepare_model.py`; add source packaging and a config argument mapping
when adding another engine.

## Updating engines

```sh
nix run .#update-engine -- strata --dry-run
nix run .#update-engine -- strata
nix run .#update-engine -- maya --ref TAG_OR_FULL_COMMIT
```

Run inside a Git checkout. The default uses GitHub's latest published stable
release endpoint, not the highest-looking tag. Exact tags (including annotated
tags) resolve to commits; `GITHUB_TOKEN` optionally authenticates API requests.
The CMake project version is extracted from that commit, not guessed from the
launcher release tag. Ambiguous/missing project versions require manual review.

`engines/metadata.json` owns package versions and shared llama.cpp dependency
pins. Literal flake input URLs remain explicit because flake inputs cannot use
arbitrary imported metadata. The lock's resolved input revision is the actual
source provenance: runtime manifests and Maya passthru use it, not version
constants. Direct Python runtime invocation records `unknown` unless supplied
with the packaged revision environment; use the Nix app for provenance.

The updater refuses dirty/staged flake.nix, flake.lock or metadata, stages the
lock operation in a temporary copy, checks that unrelated inputs did not change,
and writes only after success (rolling back ordinary write failures). Dry runs
still access the network/Nix cache but do not modify checkout files. Do not run
concurrent updates; process termination during final writes is not crash-atomic.
No dependency hash is automatically guessed or replaced.

Before committing an update:
- Review the diff and upstream CMake dependency pins; update shared metadata only
  after verifying the archive hash and both engines' dependency compatibility.
- Review build targets/flags, pack tool CLI and frontend config schema changes.
- Run Python tests and `nix flake show`; build both tools packages and relevant
  GPU variants. Test model conversion and inference separately on real hardware.
- Keep experimental GPU support labels until validated; a release update alone
  does not establish compatibility with existing packs or models.

### Strata vision

A separate `strata-vision-cpu` package supplies the portable CPU image encoder.
Strata config preparation accepts `--mmproj /path/to/matching-mmproj.gguf` and
uses that encoder by default, independently of the ROCm/CUDA text engine.
See [vision commands and schema](docs/model-preparation.md#strata-images-cpu-encoder).
GPU vision encoders and real-model image inference remain unvalidated.
