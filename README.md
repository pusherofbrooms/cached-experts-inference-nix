# Cached expert inference engines

Model-free Nix builds of Strata and Maya for **x86_64 Linux**. Model downloads,
packing and serving are explicit user operations, never build steps. These are
model-specific cached-expert engines, not universal GGUF runners.

## Packages

| Purpose | Packages/apps |
| --- | --- |
| Text engines | `strata-rocm`, `maya-rocm`, `strata-cuda`, `maya-cuda` |
| Default engine aliases | `strata`, `maya` (ROCm) |
| Packing/config | `prepare-strata`, `prepare-maya` (ROCm); explicit `-rocm` / `-cuda` variants |
| GPU-free packing/config | `prepare-strata-tools`, `prepare-maya-tools` (config needs `--exe`) |
| GGUF inspection/memory planning | `strata-tools`, `maya-tools` (not pack builders) |
| CPU image encoders | `strata-vision-cpu`, `maya-vision-cpu` |
| HTTP frontends | `server-strata`, `server-maya` (loopback by default) |
| Utilities | `download-model`, `update-engine` |

Legacy `prepare-model` / `prepare-model-cuda` aliases are **Strata-only**.
Preparation packages include only their matching engine; tools-only variants
build neither a text engine nor a GPU toolkit.

## Quickstart

Enable Nix flakes and install a working host GPU driver with device access.
Drivers are not installed by this flake. `settings.nix` defaults to ROCm
`gfx1100` (RX 7900 XT/XTX) and CUDA architecture `86`; these are build targets,
not a compatibility matrix. See [GPU selection](docs/common.md#gpu-selection-and-troubleshooting).

```sh
# Build only what you need:
nix build .#strata-rocm --out-link result-strata
nix build .#maya-rocm --out-link result-maya
# Optional CUDA alternatives:
nix build .#strata-cuda --out-link result-strata-cuda
nix build .#maya-cuda --out-link result-maya-cuda

# After download → pack → config using the engine guide:
nix run .#server-strata -- --config /data/strata.json
nix run .#server-maya -- --config /data/maya.json
```

- [Strata guide](docs/strata.md): native IQ packs, startup defaults, MTP, KV and CPU vision.
- [Maya guide](docs/maya.md): GLM packs, DeepSeek Flash preparation, KV, DFlash and CPU vision.
- [Common preparation and troubleshooting](docs/common.md): downloads, backend/tools selection, safety and memory.
- [Maintenance](docs/maintenance.md): tests, provenance and engine updates.
- [Model preparation index](docs/model-preparation.md): compatibility entry point.

## Validation status

Both Strata 0.1.47 and Maya 0.1.27 passed ROCm (`gfx1100`) and CUDA (`86`)
builds. **CUDA runtime is untested**; no CUDA hardware was available for local
inference testing. GPU vision encoders remain unvalidated and are not packaged.

User-reported runtime validation on **RX 7900 XTX, 24 GB VRAM, approximately
96 GB system RAM**:

- **Strata Flash Next IQ3_S pack:** text with fp16/k8v4 KV, MTP and CPU vision work.
- **Maya GLM S24 pack:** text with int8 KV and CPU vision work. DFlash works,
  but was slower in that workload: about 24 tokens/s versus 29–32 without it.
  This is not a controlled benchmark.

Additional user-reported validation: **Maya DeepSeek-V4.1-Flash Maya-XS IQ1_S**
packing and ROCm text inference work with a 128K configured context and 2 GiB
VRAM reserve. Full-context capacity, quality, and performance were not measured.
See the [DeepSeek preparation workflow](docs/maya.md#deepseek-flash-local-gguf-text-only).

These reports do not establish compatibility or performance for other models,
GPUs or contexts.
