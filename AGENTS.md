# Agent guidance

## Purpose and scope

This repository is an external Nix packaging and runtime-integration layer for
cached-expert inference engines. Keep builds reproducible and model-free; make
model acquisition, preparation, and execution explicit user operations. Upstream
projects evolve rapidly: inspect the selected source revision rather than treating
old flags, layouts, release numbers, or compatibility reports as contracts.

Strata and Project Maya are integrated today. Other engines, including
[FreeToken](https://github.com/FlashML-org/FreeToken), should fit the architecture
without inheriting Strata/Maya-specific model or build assumptions. Do not add
engines, update pins, or refactor tooling incidentally to an unrelated task.

## Read first and locate the source of truth

- `README.md`: supported scope, package/app names, user workflow, validation limits.
- `docs/common.md`, `docs/model-preparation.md`, and `docs/<engine>.md`: safety,
  model layout, configuration, and engine-specific recipes.
- `docs/maintenance.md`: update procedure and checks; read before changing pins.
- `flake.nix`: source inputs, engine registry, output composition, checks, dev shell.
- `settings.nix`: user-selected GPU compilation targets, not hardware detection.
- `engines/<engine>.nix`: native/tool/optional encoder packaging.
- `engines/metadata.json`: package versions and explicit shared dependency pins.
- `flake.lock`: resolved source provenance. A release label is not a commit.
- `runtime/`: preparation orchestration, package wrappers, HTTP frontend wrappers.
- `scripts/update_engine.py` and `tests/`: updater and fixture-based safety tests.

Local upstream checkouts may be available at `~/ai/Strata` and
`~/ai/project-maya`. Use them for reconnaissance, not implicit build inputs;
they may differ from the locked revisions. Check their revision and relevant
upstream guidance. Review README, build definitions, dependency declarations,
entrypoints, packers, frontend config consumers, and release changes for the
revision actually being packaged. Do not edit upstream checkouts without asking.

## Architectural boundaries

Keep these responsibilities separate, even if an upstream combines them:

1. **Source and dependency provenance:** pinned engine sources and fixed-output
   dependencies. Share a dependency only when the engines' requirements agree;
   allow engine-specific pins when they diverge.
2. **Engine packaging:** backend-specific inference binaries or Python/native
   runtime packages, plus independently useful inspection/preparation tools.
   Keep optional vision, drafting, and desktop components separate where feasible.
3. **Model acquisition:** explicit downloads to user-owned writable storage.
   Do not silently choose a model, quantization, revision, or licensed accessory.
4. **Preparation adapters:** local conversion/packing and config generation using
   matching upstream tools. Record provenance and translate only supported options.
5. **Serving:** wrap the actual upstream server/CLI with declared dependencies and
   an explicit executable/config contract. Do not conflate a native resident
   stdin protocol with an HTTP API.
6. **Validation and documentation:** describe what was built, imported, prepared,
   or executed separately, including hardware/model limits.

The Linux registry expects ROCm, CUDA, tools, and vision attributes; Darwin
exposes a separate Strata Metal package set. The
preparation adapter table is not a universal plugin API. When adding an engine,
make availability/capabilities explicit rather than fabricating unsupported
backends or forcing every engine into today's pack/config schema. Preserve
existing public package names and aliases unless a migration is intentional.

FreeToken is a useful counterexample to a CMake-only design: its upstream has a
Python package/CLI, native and optional accelerated dependencies, and its own
weight format and serving flow. Re-inspect its packaging at integration time;
do not assume GGUF, llama.cpp, a separate HTTP frontend, identical GPU support,
or Strata/Maya's model layouts apply. Account for Python/native ABI alignment,
JIT kernel caches, and any runtime acquisition paths.

## Nix and upstream integration rules

- No model weights, model conversion, device probing, or inference in derivation
  builds. Ordinary builds must not fetch from the network; declare source fetches
  explicitly and disable upstream dependency acquisition where applicable.
- Do not run upstream bootstrap/self-update installers blindly. Strata/Maya
  `setup.py` files are orchestration scripts, not conventional setuptools build
  backends. Inspect other engines individually rather than generalizing this.
- Supply compatible pinned native dependencies instead of allowing CMake or a
  Python installer to resolve them dynamically. Never guess hashes or replace a
  dependency merely because its name matches.
- Patch writable build-tree copies when upstream modifies dependency sources;
  `/nix/store` is immutable. Keep patches minimal and explain why they exist.
- Build explicit intended targets; default targets can include benchmarks, tests,
  downloads, or model-dependent programs even when test flags are disabled.
- Select GPU backend, toolkit/compiler combination, GPU architecture, and CPU
  baseline explicitly. Avoid build-host `native` detection. A portable upstream
  option can still require a particular instruction set. Host drivers and device
  permissions remain deployment responsibilities, not something this flake installs.
- Check installed shared libraries, relative resource lookups, subprocess tools,
  templates, tokenizer code, and licenses—not just the primary executable.
  Preserve required source-relative resources and declare runtime dependencies.
- Keep tools-only workflows free of GPU toolkits/inference builds where feasible.
  Do not equate inspection tools with model packers.
- Disable or deliberately isolate upstream self-updates and automatic acquisition.
  Executables should remain owned by Nix; runtime state belongs outside the store.
- Default HTTP listeners to loopback. Do not widen exposure or assume upstream
  authentication/security properties without an explicit request and review.

## Model, configuration, and path safety

Models, shards, packs, configs, logs, caches, and tuning state belong in writable
user directories, not this repository or the store. Large model operations can
be expensive and destructive; ask before running real downloads/conversions or
GPU workloads unless already authorized.

Preserve fail-before-write checks: reject output/input overlap, output collisions,
symlink and hard-link aliases, unsafe directory ancestry, and unintended shard
or optional-file discovery. `--force` permits intentional output replacement,
not damage to source weights. Do not move/link/delete user files implicitly.
Add regression tests for changes to these rules.

Keep engine/model-family mappings explicit. Reject unsupported flags instead of
silently dropping them. Config generation does not establish model compatibility;
pack existence does not establish inference quality. Packs, tokenizers, projectors,
drafters, and schemas may change independently across upstream revisions.

Record actual source/dependency revisions and meaningful preparation options.
Generated JSON containing store paths is **not a GC root**: document how users
retain the corresponding packages and regenerate configs after updates. Never
silently rewrite existing user configs/packs during a package update.

## Updates, additions, and validation

For updates, use the documented updater deliberately, inspect its dry run and
resulting diff, and verify upstream dependency pins, targets, flags, packer CLI,
config schema, and runtime imports. Do not run broad lock updates or change
unrelated inputs. Versions, source URLs, metadata, and lock provenance must remain
consistent. Historical build/runtime reports do not validate a new revision.

For an added engine, inspect its build and runtime contracts first; then add
source/metadata, engine packaging, registry/output wiring, applicable preparation
and serving integration, tests, and user documentation. Extend shared machinery
only for genuinely shared behavior; keep model-specific policy in adapters.
Review licenses of code, native dependencies, models, and optional accessories
separately. State unsupported capabilities clearly.

Use the repository toolchain; typical validation commands are:

```sh
# Linux dev shell only:
nix develop -c python3 -m unittest discover -s tests -v
nix flake show
nix flake check --no-build
# Build checks and selected packages when appropriate; GPU builds can be costly.
nix flake check
nix build .#<package>
```

Darwin has no dev shell. Use the pinned Python rather than a potentially old
system Python:

```sh
nix shell --inputs-from . nixpkgs#python3 -c python3 -m unittest discover -s tests -v
nix build .#checks.aarch64-darwin.prepare-model
```

The Python suite uses fixtures/mocks. Current frontend checks exercise `--help`
and imports, and tools checks build utility packages. None proves real packing,
HTTP readiness, GPU execution, vision, quality, or performance. Test those
separately when authorized and equipped; report untested paths and failures
honestly. For runtime evidence record engine revision, backend, GPU, model/format,
and relevant options. Do not turn one user's timing into a benchmark claim.

Keep changes scoped and review `git diff`/`git status` before handing off. Do not
commit generated models, packs, logs, credentials, local caches, or build results.
Update affected documentation and safety tests with behavior changes. This file
should retain durable boundaries and procedures; put release-specific workarounds
and validation details in engine definitions and the relevant docs instead.
