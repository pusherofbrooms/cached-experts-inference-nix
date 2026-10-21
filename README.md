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
Linux GPU builds, real model conversion and inference still require hardware
validation; passing the unit tests is not an end-to-end validation.

## Local runtime preparation

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
Maya configs use `--glm-pack`; Strata uses `--pack --native`. Strata PLE models
may require explicit `--ple-gguf /data/shard.gguf` on config generation.
Compatibility and memory sizing remain the user's responsibility; these are not
universal GGUF engines. DeepSeek/EXL3 preparation is not implemented.

Pack and config are deliberately separate operations. Config generation needs
no model files and does not validate them. Pack invokes the pinned upstream
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
