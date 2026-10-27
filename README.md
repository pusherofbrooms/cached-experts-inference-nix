# Cached expert inference engines

**Strata and Project Maya, packaged with Nix.** Reproducible, model-free builds
for **x86_64 Linux**, with ROCm and CUDA text engines, model preparation tools,
CPU image encoders, and HTTP frontends.
**Apple silicon macOS 15 or newer** also provides the Strata Metal engine, native-IQ
preparation, separate CPU image encoder and HTTP frontend; see [Metal workflow](docs/strata.md#apple-silicon-metal).
Darwin does not expose Maya, ROCm/CUDA, or GPU vision packages.

These engines run model-specific mixture-of-experts models using expert caches
across GPU memory, RAM, and storage. They are **not universal GGUF runners**.
For model selection, hardware suitability, and resource estimates, start with
[Strata upstream](https://github.com/Niko1221/Strata) or
[Project Maya upstream](https://github.com/mw00/project-maya). Upstream supports
more platforms and workflows than this flake; use the guides here for Nix commands.

## Before you start

- Use **x86_64 Linux** (or **aarch64-darwin** for Strata Metal) with Nix flakes enabled. Install a working host GPU driver
  and ensure your user has device access; this flake does not install drivers.
- Check [`settings.nix`](settings.nix) before building. Defaults are ROCm
  `gfx1100` (RX 7900 XT/XTX) and CUDA architecture `86`. These are compilation
  targets, not a compatibility matrix. See
  [GPU selection](docs/common.md#gpu-selection-and-troubleshooting) for changing them.
- Keep models, packs, and configs in writable directories outside the Nix store.
  Downloads and packing are explicit user operations, **never build steps**.

All commands below run from a local checkout of this repository.

## From model files to a running server

**Download → pack → generate config → serve → test.** A downloaded GGUF alone
is not ready to serve: use the matching engine's packer and configuration recipe.
You only need one engine and one backend to get started.

### 1. Follow your engine's preparation guide

| Engine | Nix workflow |
| --- | --- |
| **Strata (Linux ROCm/CUDA)** | [Download, pack, and configure](docs/strata.md#pack-and-serve); optional MTP and CPU vision |
| **Strata (Apple silicon Metal)** | [Acquire, pack, configure and serve](docs/strata.md#apple-silicon-metal); optional CPU vision |
| **Project Maya (Linux ROCm/CUDA)** | [GLM preparation](docs/maya.md#pack-and-serve) or [DeepSeek Flash preparation](docs/maya.md#deepseek-flash-local-gguf-text-only); separate model-specific paths |

Read [common preparation and safety](docs/common.md) alongside your engine guide.
On Linux, the downloader wraps `hf download`; it does not choose a compatible model or
quantization for you.

The Linux guides use ROCm preparation packages by default. For CUDA, substitute
`prepare-strata-cuda` or `prepare-maya-cuda`. Preparation packages supply their
matching engine automatically—there is no separate engine build required.
GPU-free preparation is also available on Linux; see the package reference below.
On Darwin, follow the Metal guide using `prepare-strata-metal`; the same
`server-strata` frontend serves its generated config.

### 2. Start the matching HTTP frontend

If you already started the server using the engine guide, skip to step 3.
Otherwise, once the guide has produced your JSON config, run **one** of:

```sh
# Strata:
nix run .#server-strata -- --config /absolute/path/to/strata.json --port 8080

# Or Project Maya:
nix run .#server-maya -- --config /absolute/path/to/maya.json --port 8080
```

The config selects the engine executable, so the same frontend package works
with ROCm, CUDA, or (for Strata on Darwin) Metal. Servers bind to loopback by default. Native engine
`--serve` is an internal stdin protocol, **not** an HTTP server.

**Allow time for startup:** loading a model can take a minute to several minutes.
Keep the server terminal open and inspect `<config>.log` for loading progress;
a server that is not ready immediately is not necessarily stuck.

### 3. Check that it is ready

In another terminal, after the model has loaded (requests may fail while it is
still starting):

```sh
curl --fail http://127.0.0.1:8080/health
```

Connect an OpenAI-compatible client to `http://127.0.0.1:8080/v1`.
For startup failures, inspect `<config>.log` and the
[troubleshooting checklist](docs/common.md#troubleshooting-checklist).

### Keep the engine available

Generated configs contain absolute Nix store paths. **JSON references are not
GC roots.** Keep a build link to the preparation package you used, for example:

```sh
# Match this to your engine/backend; e.g. prepare-maya-cuda for Maya on CUDA.
nix build .#prepare-strata --out-link result-prepare-strata
```

Keep that link while using the config. If you supplied a custom `--exe` or
`--vision-exe`, root those packages separately. After changing inputs or GPU
architecture settings, regenerate the config with the matching preparation
package; retain all desired tuning flags when doing so.

## Linux package reference

Every package below is also available through `nix run .#<name>`.

| Purpose | Packages/apps |
| --- | --- |
| Text engines | `strata-rocm`, `maya-rocm`, `strata-cuda`, `maya-cuda` |
| Default engine aliases | `strata`, `maya` (ROCm); flake default is Strata ROCm |
| Packing/config | `prepare-strata`, `prepare-maya` (ROCm); explicit `-rocm` / `-cuda` variants |
| GPU-free packing/config | `prepare-strata-tools`, `prepare-maya-tools` (config needs `--exe`) |
| GGUF inspection/memory planning | `strata-tools`, `maya-tools` (not pack builders) |
| CPU image encoders | `strata-vision-cpu`, `maya-vision-cpu` |
| HTTP frontends | `server-strata`, `server-maya` (loopback by default) |
| Utilities | `download-model`, `update-engine` |

Preparation packages include only their matching text engine and CPU vision
encoder. Tools-only preparation variants include the encoder and packing
dependencies, but build neither a text engine nor a GPU toolkit.
Legacy `prepare-model` / `prepare-model-cuda` aliases are **Strata-only**.

## Darwin package reference

Apple silicon macOS 15 or newer only; see the [Metal workflow](docs/strata.md#apple-silicon-metal).
Every package below is also available through `nix run .#<name>`.

| Purpose | Packages/apps |
| --- | --- |
| Metal text engine | `strata`, `strata-metal`, flake default |
| Packing/config | `prepare-strata`, `prepare-strata-metal` |
| CPU image encoder | `strata-vision-cpu` |
| HTTP frontend | `server-strata` |

No Maya, Linux GPU backends, inspection/tools-only preparation, `download-model`
or `update-engine` apps are exposed on Darwin. See the Metal guide for explicit
external acquisition and [maintenance](docs/maintenance.md) for updater invocation.

## Validation status

Recorded build validation covers both Strata and Maya on ROCm (`gfx1100`) and
CUDA (`86`). These are historical results, not automatic validation of future
input updates; package versions and source pins live in
[`engines/metadata.json`](engines/metadata.json) and [`flake.lock`](flake.lock).
**CUDA runtime is untested**; no CUDA hardware was available for local inference
testing. GPU vision encoders remain unvalidated and are not packaged.

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

## Further reading

- [Apple silicon testing cookbook](docs/apple-silicon-testing.md): staged M5 checks and opt-in runtime tests.

- [Strata guide](docs/strata.md): native IQ packs, startup defaults, MTP, KV and CPU vision.
- [Maya guide](docs/maya.md): GLM packs, DeepSeek Flash preparation, KV, DFlash and CPU vision.
- [Common preparation and troubleshooting](docs/common.md): downloads, backend/tools selection, safety and memory.
- [Maintenance](docs/maintenance.md): tests, provenance and engine updates.
