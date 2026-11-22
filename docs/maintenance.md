# Maintenance

## Testing

Use pinned Python (the repository dev shell is Linux-only):

```sh
# Linux:
nix develop -c python3 -m unittest discover -s tests -v
# Either platform, including Darwin (avoid an old system Python):
nix shell --inputs-from . nixpkgs#python3 -c python3 -m unittest discover -s tests -v
# Darwin's packaged test check:
nix build .#checks.aarch64-darwin.prepare-model
```

Both platforms expose a `prepare-model` check running the fixture/mocked suite.
Tests cover wrapper argument/config generation, path safety, pack orchestration
and updater behavior; they do not establish real model compatibility or quality.

On Linux, other checks build `strata-tools` and `maya-tools` and exercise both
HTTP frontends' `--help`. Both platforms check Strata Terminal's launch plan,
command restrictions and the tagged packer's imports. Darwin also checks the
Metal environment wrapper contract; there are no Darwin native tools checks.
Run `nix flake show` and `nix flake check --no-build` to inspect/evaluate outputs,
then `nix flake check` for the current platform's checks. Build selected engine
and CPU encoder packages separately when appropriate. None of these checks
proves actual packing, HTTP readiness, or text/vision inference; validate those
separately with compatible models and hardware.

## Updating engines

The `update-engine` app is Linux-only:

```sh
nix run .#update-engine -- strata maya --dry-run
nix run .#update-engine -- strata maya
nix run .#update-engine -- maya --ref TAG_OR_FULL_COMMIT
```

On Darwin, invoke the same script directly with pinned Python and Git/Nix on PATH
instead of the unavailable app (these commands access the network):

```sh
nix shell --inputs-from . nixpkgs#python3 nixpkgs#git -c python3 scripts/update_engine.py strata --dry-run
nix shell --inputs-from . nixpkgs#python3 nixpkgs#git -c python3 scripts/update_engine.py strata
```

Run inside a Git checkout. The default uses GitHub's latest published stable
release endpoint, not the highest-looking tag. Exact tags (including annotated
tags) resolve to commits; `GITHUB_TOKEN` optionally authenticates API requests.
Select one or more engines for a single transaction; duplicate names are rejected.
`--ref` requires exactly one engine (multi-engine updates use latest releases).
The Python API accepts either a single name or a sequence of names.
Engine names are validated against `engines/metadata.json`, not a fixed CLI list.
Version discovery uses the engine's optional `versionStrategy`: `cmake` (the
default, root `CMakeLists.txt`) or `cmake-engine` (`engine/CMakeLists.txt`, Strata
since the layout move). Unknown strategies fail rather than guessing. The CMake
project version is extracted from the resolved commit, not the launcher release
tag. Older refs with a different layout or ambiguous project versions require
manual review.

`engines/metadata.json` owns package versions and Maya's llama.cpp dependency
pin. Literal flake input URLs remain explicit because flake inputs cannot use
arbitrary imported metadata. The lock's resolved input revision is the actual
source provenance: runtime manifests and Maya passthru use it, not version
constants. Direct Python runtime invocation records `unknown` unless supplied
with the packaged revision environment; use the Nix app for provenance.

The updater refuses dirty/staged flake.nix, flake.lock or metadata, stages the
`nix flake update <engine> ...` operation in a temporary copy, checks that unrelated inputs did not change,
verifies every selected revision,
and writes all three files only after all resolutions and locking succeed (rolling back ordinary write failures). Dry runs
still access the network/Nix cache but do not modify checkout files. Do not run
concurrent updates; process termination during final writes is not crash-atomic.
No dependency hash is automatically guessed or replaced.

Before committing an update:
- Review the diff and upstream CMake dependency pins. Update dependency metadata
  only after verifying the archive hash and compatibility for its consumers.
- Review build targets/flags, pack tool CLI and frontend config schema changes.
- Run Python tests and `nix flake show`; on Linux build both tools packages and relevant
  GPU variants; on Darwin run its checks and build Strata Metal and its CPU encoder. Test model conversion and inference separately on real hardware.
- Keep experimental GPU support labels until validated; a release update alone
  does not establish compatibility with existing packs or models.


## Adding an engine

Adapters are registered in `runtime/prepare_model.py`. Add source packaging and
a config argument mapping, wire the engine into the flake registry/packages,
and add tests for its CLI, layout and safety rules. Review which shared defaults
actually apply rather than inheriting another engine's model assumptions.
Document runtime validation separately from automated checks.

## Stable release review: Strata v0.2.18

The Strata-only updater dry run and apply selected
`6e38f588a646c0ee6948a74afbe670eb75896106` (engine `0.1.64`, Terminal `0.2.18`).
Maya, nixpkgs and Maya's llama.cpp pin are unchanged. The existing native
packers, frontend policy and Terminal patch remain applicable.

Hy4-Flash uses a standalone runner rather than the existing CMake targets.
The Linux `strata-hy4-rocm` package builds only that runner for `gfx1100`, with
Nix's ROCm and zstd. A small JSON header patch makes the recursive object type
compile with Clang/libstdc++. The GPU-free tokenizer app reads the manifest of
a local prebuilt `.strata` file and creates a new output directory. See the
[Hy4-Flash guide](hy4-flash.md) for the separate acquisition and config flow.

Validated on Linux: all-system flake evaluation, Linux flake checks (65 local
fixture tests plus the Hy4 tokenizer check), builds of the standard ROCm and
CPU vision packages, and the Hy4 ROCm build and model-free `--version` test.
Three upstream Hy4 HTTP smoke tests passed with the packaged frontend's Python
environment and a mock engine. No model download or real inference was run;
CUDA and Darwin packages were not built. The full upstream tokenizer/template
suite has a failing invalid-effort assertion against the tagged template; the
flake's new check runs the tokenizer conversion tests and wrapper contract,
not that unrelated template assertion.

## Stable release review: Strata v0.2.17

GitHub's latest stable release resolved to `12d491da4a6904eb49aaa6f2f63ec4b1a17582e3`
(engine/Control `0.1.63`, Terminal `0.2.17`). The Strata-only updater dry run
and apply ran in a temporary checkout with the new version reader. Maya,
nixpkgs, and Maya's llama.cpp pin are unchanged.

Adapted to the source move: native and Metal CMake projects under `engine/`;
Python frontend, packer, and CPU vision reader under `terminal/`. Strata's
production packages no longer need external llama.cpp: they use Strata's own
quant layer, prompt kernels, GGUF helpers, and vision reader. The obsolete CUDA
K-quant sidecar is no longer installed. Upstream still has ggml compatibility
and test code. Metal ships the actual shader quant header and just the required
kernel/data files.

The new `strata-terminal` app adds serve planning, JSONL lifecycle events and
status without upstream install/update, automatic config migration, or desktop
management. A release-specific patch aligns plan validation with the offline
server policy and fixes the numeric batch-groups and `--json` options. Strata's
`pack --force` now also refuses nested output aliases before running the packer.
Engine defaults follow upstream, including route-tail skipping off
by default. The upstream setup RAM-budget migration is not part of this flake.
No new backend or model family is implied. Existing packs/configs are not
rewritten; regenerate configs explicitly when switching their store paths.

Validation: all-system flake evaluation and Linux `nix flake check` passed
(65 fixture tests). Strata ROCm `gfx1100`, CUDA `sm_86`, tools and CPU vision
built. ROCm reported `0.1.63`; the CUDA binary could not start on this host,
which does not provide the NVIDIA driver library `libcuda.so.1`. A generated
config and the built-in mock engine passed loopback HTTP health/model listing,
Terminal status, and JSONL startup/shutdown. Darwin was evaluated, not built.
No real packing, model downloads, GPU inference, vision encoding, or performance
measurements were run by the packaging agent. The user subsequently reported
successful inference with the existing Strata Flash Next IQ3_S pack on ROCm.
Their run reached about 80 tokens/s after 3,000 tokens; this was an observation,
not a controlled benchmark.

## Stable release review: Strata v0.2.12 / Maya v1.1.56

Updated using the documented two-engine dry run and apply against GitHub's
latest stable releases:

| Engine | Previous release | Selected release / commit | CMake version |
| --- | --- | --- | --- |
| Strata | v0.2.9 | v0.2.12 / `146df3c468739147f520ac3816de806f71232571` | 0.1.55 → 0.1.58 |
| Maya | v1.1.48 | v1.1.56 / `22fc101f2c109b7e7174c975ff3c2d8e9ce0a867` | 0.1.27 unchanged |

Reviewed exact source revisions, dependency declarations, packaged targets,
packer interfaces, frontend configuration and resource lookup. No packaging
code changes were needed. Both retain llama.cpp
`3cf03257f219afbe7334045ff7c6a06ac68c627d`; its hash and nixpkgs are unchanged.

Strata retains the native IQ packer and CPU vision interfaces. New optional
split-engine/benchmark targets are not built by the explicit production target
selection. Metal DFlash2/kernel changes retain the resource directories copied
by the package, but were not built on Darwin in this update.

Maya changes GLM routing-bias packing to F32 and some runtime defaults. Existing
packs are not migrated: see [Maya release scope](maya.md#release-update-scope)
for explicit repacking guidance and frontend data-persistence changes. New
upstream model/backend capabilities do not expand this flake's supported scope.
Acquisition and model-management policies remain unchanged.

Validation passed: 63 Python fixture tests, `nix flake show`,
`nix flake check --no-build`, `nix flake check`, both Linux tools and frontend
checks, and both CPU vision packages. All-system no-build evaluation also
passed. ROCm/CUDA/Metal inference binaries were not built. No model downloads,
real packing, HTTP readiness, GPU/model execution, vision quality or performance
tests were performed; previous runtime reports remain historical.

## Stable release review: Strata v0.2.9 / Maya v1.1.48

Updated with the documented two-engine dry run, then apply (GitHub latest stable):

| Engine | Previous release / commit | Selected release / commit | CMake version |
| --- | --- | --- | --- |
| Strata | v0.2.8 / `870aedd9e1ebe1c91f94da93b59af17729f6c5c0` | v0.2.9 / `c62d5e3a15b40ea51c96aae9e8d36928965f53ab` | 0.1.54 → 0.1.55 |
| Maya | v1.1.41 / `171077787036ccc2784250ca644760f654e8d4a3` | v1.1.48 / `465a981579ff5e91b19ff023721c6661f3c6030b` | 0.1.27 unchanged |

Reviewed exact upstream Git revisions, build definitions, dependency pins,
packers, frontend imports/config consumers and Metal resources. Both retain
llama.cpp `3cf03257f219afbe7334045ff7c6a06ac68c627d`; its hash and nixpkgs
are unchanged. Existing explicit production targets remain applicable.

Strata changes include tool-call parsing fixes for Markdown examples and tool
history, new CUDA parallel/GDN defaults, opt-in replica shared expert memory,
and Metal GDN/attention/expert kernel and buffer-offset changes. Native IQ and
MTP packers and the run-config schema are unchanged. Tools-only ROCm configs
need an explicit `"backend": "hip"`; the regular ROCm wrapper supplies it.
Upstream DeepSeek improvements do not add a Strata DeepSeek package here.

Maya changes include a DeepSeek expert-cache race fix affecting repeated long
prompts, prompt/cache restore improvements, gfx11 prefill chunk limits, lower
idle CPU activity, and fixes to loop-guard continuations, reasoning boundaries,
client stop strings, multipart content and tool parameters. The frontend now
normalizes legacy `--engine strata` to Maya; the flake checks this explicitly.
Upstream model management became default-on: the packaged frontend patches only
that default back to off, retaining explicit config/environment opt-in, with
import-level regression checks. Update polling remains disabled.

Maya-XS v2's separate Engram-table shards require a future explicit preparation
adapter; they are not supported by the existing self-contained-shard workflow.
New upstream mmproj preparation, Qwen vision and GLM HIP vision do not expand
this flake's supported workflows. No model acquisition or automatic migration
was performed. Upstream speed/quality claims are not Nix validation results.

Linux validation: all 63 Python fixture tests, `nix flake show`,
`nix flake check --no-build`, and `nix flake check` passed. The five Linux checks
include both tools builds, frontend imports/help, Maya policy/selector assertions,
and preparation tests. ROCm/CUDA inference binaries, CPU encoders and Darwin
packages were not built for these revisions. No real packing, HTTP readiness,
model/GPU execution, vision quality or performance testing was performed by the
packaging agent. After the update, the user reported that actual inference
testing passes. Engine/backend, hardware, model/format and runtime options were
not supplied, so this report does not establish a broader compatibility matrix
or vision/performance validation.

## Stable release review: Strata v0.2.8 / Maya v1.1.41

Updated with the documented two-engine dry run, then apply (GitHub latest stable):

| Engine | Previous release / commit | Selected release / commit | CMake version |
| --- | --- | --- | --- |
| Strata | v0.2.7 / `5e8a4cca7d690ac60ca37eb09e7d0e010099ad6d` | v0.2.8 / `870aedd9e1ebe1c91f94da93b59af17729f6c5c0` | 0.1.52 → 0.1.54 |
| Maya | v1.1.33 / `88a746c9f94d3c34a9ed31711ecfc64a2f03aa39` | v1.1.41 / `171077787036ccc2784250ca644760f654e8d4a3` | 0.1.27 unchanged |

Reviewed selected-revision CMake/Metal/vision definitions, release changes,
native IQ and DeepSeek packers, frontend config consumers and runtime entrypoints.
Both still pin llama.cpp `3cf03257f219afbe7334045ff7c6a06ac68c627d`;
its hash and nixpkgs are unchanged. Existing explicit Linux targets, portable CPU
flags and configured GPU architectures remain valid. Maya builds now receive the
locked source revision through `MAYA_BUILD_ID`, so `maya-ds4 --version` does not
fall back to `unknown` in a source archive.

Strata v0.2.8 includes kernel, cache-sizing, startup, batching, tool-call and Metal
correctness changes. The upstream HIP default now builds 16-row windows while CUDA
remains at 8; this update accepts that upstream default rather than pinning the old
HIP behavior. Tool-call recovery is also now enabled when omitted. Existing Qwen
Metal target/resource layout remains compatible and was built on Apple silicon.
The new separate Strata DeepSeek Metal target and Linux ARM64 outputs are not
integrated by this release update.

Maya v1.1.41 adds Qwen support, experimental Metal, parallel DeepSeek/MiniMax
serving, model-management APIs, logprobs and stricter browser-origin handling.
This flake continues to package the existing Linux GLM and DeepSeek targets only;
model management remains disabled by default and release polling remains disabled
with `MAYA_UPDATE_CHECK=0`. The native adapter now records the inspected GGUF
architecture and rejects Qwen before invoking the packer, because its Qwen pack,
config and drafting contract has not been integrated. The existing explicit
DeepSeek shard selection and `engram.file`/optional-directory rejection remain in
force. Maya Metal and MiniMax remain out of scope.

Apple-silicon validation: 63 Python fixture tests, `nix flake show`,
`nix flake check --no-build`, and `nix flake check` passed. `strata-metal` and
`strata-vision-cpu` both built successfully for aarch64-darwin. No x86_64 Linux
CUDA/ROCm/tools builds, model acquisition, real packing, HTTP readiness, inference,
vision quality or performance tests were run for these revisions. Earlier runtime
reports remain historical.
