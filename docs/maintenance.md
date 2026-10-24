# Maintenance

## Testing

Run `python3 -m unittest discover -s tests -v`. The flake's
`checks.x86_64-linux.prepare-model` runs the same suite. Tests cover wrapper
argument/config generation, path safety, pack orchestration and updater behavior
using temporary fixtures and mocked external operations; they do not establish
real model compatibility or inference quality.

The other flake checks build `strata-tools` and `maya-tools` and run each
packaged HTTP frontend with `--help`, checking for `--config`. These check tool
builds and frontend imports/CLI availability, not GPU engine execution, actual
packing, HTTP serving, or text/vision inference. Validate those separately with
compatible models and hardware; build success alone is not runtime validation.

## Updating engines

```sh
nix run .#update-engine -- strata maya --dry-run
nix run .#update-engine -- strata maya
nix run .#update-engine -- maya --ref TAG_OR_FULL_COMMIT
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
- Run Python tests and `nix flake show`; build both tools packages and relevant
  GPU variants. Test model conversion and inference separately on real hardware.
- Keep experimental GPU support labels until validated; a release update alone
  does not establish compatibility with existing packs or models.


## Adding an engine

Adapters are registered in `runtime/prepare_model.py`. Add source packaging and
a config argument mapping, wire the engine into the flake registry/packages,
and add tests for its CLI, layout and safety rules. Review which shared defaults
actually apply rather than inheriting another engine's model assumptions.
Document runtime validation separately from automated checks.
