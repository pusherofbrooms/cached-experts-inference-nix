# Manual model preparation (ROCm)

Model downloads and pack creation are **runtime/user operations**, not Nix build steps. Do not run upstream `setup.py` during builds: it manages Python dependencies, engine downloads and model weights. The derivations in `engines/` build selected CMake targets, with a fixed llama.cpp input and disconnected FetchContent. They do not prepare models. The native `strata-tools` and `maya-tools` packages inspect GGUFs and plan memory; they are not inference engines or complete pack-builder distributions. In contrast, `prepare-strata-tools` and `prepare-maya-tools` provide GPU-free packing and config generation (config requires `--exe /absolute/bin/engine`).

Preparation packages retain the positional engine argument and reject mismatches.
Legacy `prepare-model` / `prepare-model-cuda` aliases are Strata-only; old Maya
commands must use `prepare-maya` / `prepare-maya-cuda` instead. The aliases expose
`bin/prepare-strata`; `nix run` resolves this automatically.

## Sources and scope

Commands below were checked against both upstream checkouts at the revisions used by the derivations:

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

## Packaged preparation and configuration

Use writable directories outside the Nix store. The wrappers supply pinned upstream
sources, gguf-py and Python dependencies; no checkout, pip or manual environment is needed.

```sh
# Strata Qwen native IQ pack; download all sibling shards first.
nix run .#prepare-strata -- strata pack --gguf /data/qwen/first.gguf \
  --pack /data/packs/qwen --experts-bin
nix run .#prepare-strata -- strata config --gguf /data/qwen/first.gguf \
  --pack /data/packs/qwen --config /data/qwen.json \
  --max-context 4096 --expert-cache auto --prefill 512 --kv int8 \
  --adapt-every 0 --pcie-frac 0 --vram-reserve-mib 1024
nix run .#server-strata -- --config /data/qwen.json

# Maya GLM-5.3-Flash (not DeepSeek or EXL3).
nix run .#prepare-maya -- maya pack --gguf /data/glm/first.gguf \
  --pack /data/packs/glm --compat-bf16
nix run .#prepare-maya -- maya config --gguf /data/glm/first.gguf \
  --pack /data/packs/glm --config /data/glm.json \
  --max-context 4096 --expert-cache auto --vram-reserve-mib 1024
nix run .#server-maya -- --config /data/glm.json
```

Use real shard filenames. Strata discovers the PLE shard during packing and records
it in the adjacent manifest. Config generation enables `--mmap-experts` when that
manifest records `--experts-bin`; override with `--no-mmap-experts` if needed.
Maya supports the same manifest handling, but its normal native GLM recipe does
not request experts.bin. That option duplicates expert data on disk.

Runtime tuning is explicit, not a backend guess: reserve, cache, prefill and KV
options are configurable, and omitted options retain upstream defaults. The Strata
ROCm example disables adaptation/PCIe budgeting as in the conservative guide;
it is not a guarantee of support for other cards or models. CUDA users should use
`prepare-strata-cuda` or `prepare-maya-cuda` respectively (or provide `--exe`). Config generation does not run inference
or validate model compatibility. `--compat-bf16` rounds projections, not losslessly
restoring original weights. Keep original GGUF shards at their recorded paths.

## What to expect and verify

- `iq_pack.py` reads all adjacent shards and exports `index.txt`, `dense.bin`, `native_experts.txt` and `tokenizer/`. Check successful exit and these outputs; keep pack, tokenizer, source shards and engine revision together. Do not mix packs between forks.
- `tools/strata_pack.py build/verify` is the older canonical pack workflow, not a substitute for the native IQ pack above. `strata-gguf` and `strata-dequant` do not perform this full conversion.
- Both native `--serve` commands stay resident and read an internal token protocol on stdin. They do **not** start an HTTP/OpenAI server or accept raw text prompts. Use packaged `server-strata` / `server-maya` with the generated JSON configuration/tokenizer for HTTP. Native CLI tests can instead use `--tokens` / `--tokens-file` with model-correct token IDs.
- There is no native runtime `--backend rocm` switch: use the HIP-built executable. Both derivations currently target `gfx1100`; other GPUs need appropriate build overrides. Wave64/CDNA is not automatically supported by these defaults. Use a working Linux ROCm driver, accessible GPU devices and sufficient RAM/VRAM; `ROCR_VISIBLE_DEVICES`/`HIP_VISIBLE_DEVICES` can select a card.
- mmap avoids a full pinned expert arena, not the need for fast storage and page-cache RAM. Leave room for both GGUFs and pack outputs. Begin with a short context; allocation success does not prove full-context inference or quality.
- These ROCm derivations are experimental; the Maya derivation explicitly records that ROCm inference is untested. The commands here were source-checked, not GPU/model execution-tested. Build with `nix build .#strata-rocm` or `nix build .#maya-rocm`, and point the binary variables at the resulting executables. For packaged conversion/config wrappers, see the root README.
