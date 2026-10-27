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
HTTP frontends' `--help`. Darwin checks exercise Strata frontend imports/help
and the Metal environment wrapper contract; there are no Darwin tools checks.
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
Version discovery uses the engine's optional `versionStrategy` (default `cmake`);
currently only the CMake reader is implemented. Add an explicit reader before
selecting another strategy; unknown strategies fail rather than guessing.
The CMake project version is extracted from that commit, not guessed from the
launcher release tag. Ambiguous/missing project versions require manual review.

`engines/metadata.json` owns package versions and shared llama.cpp dependency
pins. Literal flake input URLs remain explicit because flake inputs cannot use
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
- Review the diff and upstream CMake dependency pins; update shared metadata only
  after verifying the archive hash and both engines' dependency compatibility.
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

## Stable release review: Strata v0.2.7 / Maya v1.1.33

Updated with the documented two-engine dry run, then apply (GitHub latest stable):

| Engine | Previous release / commit | Selected release / commit | CMake version |
| --- | --- | --- | --- |
| Strata | v0.2.6 / `4b9e05a2a806a3fe24f7b2560d98626c37dade35` | v0.2.7 / `5e8a4cca7d690ac60ca37eb09e7d0e010099ad6d` | 0.1.51 → 0.1.52 |
| Maya | v1.1.26 / `f26d8bab43f9b10f0475986b541dc46ada45285e` | v1.1.33 / `88a746c9f94d3c34a9ed31711ecfc64a2f03aa39` | 0.1.27 unchanged |

Reviewed selected-revision CMake/Metal/vision definitions, release changes,
native IQ and DeepSeek packers, frontend config consumers and runtime entrypoints.
Both still pin llama.cpp `3cf03257f219afbe7334045ff7c6a06ac68c627d`;
its hash and nixpkgs are unchanged. Explicit targets, portable CPU flags and
GPU architecture settings are retained. Maya's new MiniMax target is deliberately
not built; neither MiniMax preparation nor Strata DeepSeek is added by this update.
Native IQ packer interfaces are unchanged. Metal's project/resource layout and
MTP packer remain compatible by source inspection, not a new Darwin build.

Strata includes substantial kernel, serving/tool-call and web UI changes, including
upstream routing defaults that can change answers on small NVIDIA cards. Maya
includes DeepSeek kernel/metadata handling and frontend changes since v1.1.26;
v1.1.33 itself fixes MiniMax setup. Upstream performance claims are not local
validation. Existing generated configs are not migrated.

Maya's DeepSeek packer newly follows `deepseek41.engram.file` metadata. The
preparation source copy rejects nonempty values before opening that extra file
or writing a pack, including relative, parent-relative and absolute paths.
A header-only regression runs when that source is built. Existing explicit shard
selection and optional-directory rejection remain in force. Maya frontend release
polling is disabled with `MAYA_UPDATE_CHECK=0`; updates remain Nix operations
(the upstream self-update action also requires a supervised Git checkout).

Linux validation: 61 Python fixture tests, `nix flake show`,
`nix flake check --no-build`, and `nix flake check` passed (both tools packages,
frontend imports/help and preparation tests). Also built `prepare-maya-tools`
(including the metadata-discovery regression and Maya CPU encoder) and
`strata-vision-cpu`. No costly CUDA/ROCm engine builds, Darwin builds, model
acquisition, real packing, HTTP readiness, inference or performance tests were
run for these revisions. Earlier runtime reports remain historical.
