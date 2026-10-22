# Common preparation, safety and troubleshooting

Model downloads and pack creation are **runtime/user operations**, not Nix build steps. Do not run upstream `setup.py` during builds: it manages Python dependencies, engine downloads and model weights. The derivations in `engines/` build selected CMake targets, with a fixed llama.cpp input and disconnected FetchContent. They do not prepare models. The native `strata-tools` and `maya-tools` packages inspect GGUFs and plan memory; they are not inference engines or complete pack-builder distributions. In contrast, `prepare-strata-tools` and `prepare-maya-tools` provide GPU-free packing and config generation (config requires `--exe /absolute/bin/engine`).

Preparation packages retain the positional engine argument and reject mismatches.
Legacy `prepare-model` / `prepare-model-cuda` aliases are Strata-only; old Maya
commands must use `prepare-maya` / `prepare-maya-cuda` instead. The aliases expose
`bin/prepare-strata`; `nix run` resolves this automatically.

## Sources and scope

Commands in these guides were checked against both upstream checkouts at the revisions used by the derivations:

- [Strata](https://github.com/Niko1221/Strata/tree/011c9e600a93bbc9fd451c563501b5831c32bbca): `tools/iq_pack.py`, `tools/_paths.py`, `src/program/generate.cpp`, and `docs/AMD_HIP.md` (model and serving configuration).
- [Maya](https://github.com/mw00/project-maya/tree/35bb2b22a2e2c65b91e24a41d4b026bfa4e89557): `tools/iq_pack.py`, `tools/strata_tokenizer.py`, `maya.py` (GLM pack construction), `src/program/generate.cpp`, and `docs/AMD_MAYA.md` / `docs/GLM5-FLASH.md`.

The downloader is generic Hugging Face tooling, **not** an upstream setup/model-selection replacement. The root flake provides this interface:

```sh
nix run .#download-model -- REPO DEST [hf download options]
# Intended example; replace repository and file selection:
nix run .#download-model -- OWNER/MODEL-GGUF "$HOME/models/model" \
  --include '*.gguf' --revision COMMIT
```

Download every shard of one compatible quant, retaining their original names together. Use HF authentication for gated repositories; accept the model license first. Pin a revision and record hashes yourself: a generic `hf download` does not enforce upstream launchers' model manifests. It also does not convert weights or infer which architecture your engine supports.

## Preparation options and safety

Use writable directories outside the Nix store. Wrappers supply pinned sources
and Python dependencies; no checkout, pip or manual environment is needed.
Regeneration replaces manual config edits: retain every desired setting.
Pack preparation validates sibling completeness with upstream Model; symlink
filenames are retained for sibling discovery.

Use real shard filenames. Strata discovers the PLE shard during packing and records
it in the adjacent manifest. Config generation enables `--mmap-experts` when that
manifest records `--experts-bin`; override with `--no-mmap-experts` if needed.
Maya supports the same manifest handling, but its normal native GLM recipe does
not request experts.bin. That option duplicates expert data on disk.

Runtime tuning is explicit, not a backend guess: reserve, cache, prefill and KV
options are configurable, and omitted options retain upstream defaults except for the Strata startup requirements in the Strata guide. The Strata
[ROCm example](strata.md#pack-and-serve) disables adaptation/PCIe budgeting as in the conservative guide;
it is not a guarantee of support for other cards or models. CUDA users should use
`prepare-strata-cuda` or `prepare-maya-cuda` respectively (or provide `--exe`). Config generation does not run inference
or validate model compatibility. `--compat-bf16` rounds projections, not losslessly
restoring original weights. Keep original GGUF shards at their recorded paths.

## What to expect and verify

- `iq_pack.py` reads all adjacent shards and exports `index.txt`, `dense.bin`, `native_experts.txt` and `tokenizer/`. Check successful exit and these outputs; keep pack, tokenizer, source shards and engine revision together. Do not mix packs between forks.
- `tools/strata_pack.py build/verify` is the older canonical pack workflow, not a substitute for the native IQ packs in the engine guides. `strata-gguf` and `strata-dequant` do not perform this full conversion.
- Both native `--serve` commands stay resident and read an internal token protocol on stdin. They do **not** start an HTTP/OpenAI server or accept raw text prompts. Use packaged `server-strata` / `server-maya` with the generated JSON configuration/tokenizer for HTTP. Native CLI tests can instead use `--tokens` / `--tokens-file` with model-correct token IDs.
- There is no native runtime `--backend rocm` switch: use the HIP-built executable. Both derivations currently target `gfx1100`; other GPUs need appropriate build overrides. Wave64/CDNA is not automatically supported by these defaults. Use a working Linux ROCm driver, accessible GPU devices and sufficient RAM/VRAM; `ROCR_VISIBLE_DEVICES`/`HIP_VISIBLE_DEVICES` can select a card.
- mmap avoids a full pinned expert arena, not the need for fast storage and page-cache RAM. Leave room for both GGUFs and pack outputs. Begin with a short context; allocation success does not prove full-context inference or quality.
- Build success is not runtime validation. See the [current user-reported status](../README.md#validation-status); CUDA runtime remains untested. Source metadata may still carry earlier experimental/untested labels; this documentation update does not change source behavior.


**Desktop GPU headroom:** Strata config generation enables `--expert-cache auto`.
If the GPU also drives your desktop, consider adding `--vram-reserve-mib 2048`
to the config command to leave room for desktop applications. This is a starting
point, not a universal requirement; adjust for your VRAM capacity and workload.
No reserve is imposed by this wrapper unless requested. Re-generating an existing
config requires `--force`; include the reserve flag again. This headroom is not an
OS-enforced reservation or a guarantee against GPU stalls.


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


## GPU selection and troubleshooting

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


## Troubleshooting checklist

- Startup failure: inspect `<config>.log`; use the engine guide’s config repair steps.
- Maya missing shards: repair same-basename links in `pack.parent`; no repack for layout-only fixes.
- Memory pressure: start with short context and explicit desktop headroom; mmap still needs page-cache RAM and fast storage.
- Wrong backend or stale store paths: regenerate with the matching preparation package and retain all desired flags.
- Vision failure: check exact model/projector pairing and CPU requirements; filenames alone do not validate compatibility.

