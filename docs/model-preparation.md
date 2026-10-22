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

Both engines accept `--kv fp16|int8|q4_0|k8v4`. To request a 128K context
with hybrid quantized KV and desktop headroom, add
`--max-context 131072 --kv k8v4 --vram-reserve-mib 2048` to the config command
(and `--force` when replacing a config). These are explicit choices, not new
defaults or a guarantee that the context fits your GPU. Re-generation replaces
manual config edits; include all desired settings in the command.

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
  --pack /data/glm/pack --compat-bf16
nix run .#prepare-maya -- maya config --gguf /data/glm/first.gguf \
  --pack /data/glm/pack --config /data/glm.json \
  --max-context 4096 --expert-cache auto --vram-reserve-mib 1024
nix run .#server-maya -- --config /data/glm.json
```

Use real shard filenames. Strata discovers the PLE shard during packing and records
it in the adjacent manifest. Config generation enables `--mmap-experts` when that
manifest records `--experts-bin`; override with `--no-mmap-experts` if needed.
Maya supports the same manifest handling, but its normal native GLM recipe does
not request experts.bin. That option duplicates expert data on disk.

Runtime tuning is explicit, not a backend guess: reserve, cache, prefill and KV
options are configurable, and omitted options retain upstream defaults except for the Strata startup requirements below. The Strata
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

### Repair an existing Strata configuration

Strata config generation now supplies `--spec 2 --prefill auto` unless overridden.
These are startup requirements, not additional performance tuning: the pinned
`generate.cpp` native IQ check requires spec >= 2, and resident serving requires
positive prefill. `--prefill` accepts `auto` or a positive integer for Strata;
Maya keeps its previous integer-only behavior and receives neither default.
`--spec 3` preserves an older spec-3 configuration.

MTP is **optional**: the pinned resident path explicitly falls back to lookup or
plain decoding without it. To retain an existing drafter, pass
`--mtp /home/jjorgens/.local/share/strata/models/mtp/rt`. This must be an existing
directory; preparation neither creates nor validates its model contents.
Strata configs also record `log` as `<config>.log`, where the server appends engine
stderr and can report startup errors. Output/log aliases of inputs, MTP contents,
and each other are rejected even with `--force`. Do not change these paths into
aliases after generation.

Rerun **config only** with `--force` to replace the existing JSON and manifest;
no repack is required. With `GGUF`, `PACK`, and `CONFIG` set to the same absolute
paths used previously, the CUDA command retaining the old spec/MTP/prefill is:

```sh
nix run .#prepare-strata-cuda -- strata config \
  --gguf "$GGUF" --pack "$PACK" --config "$CONFIG" --force \
  --spec 3 --prefill auto --mtp /home/jjorgens/.local/share/strata/models/mtp/rt
```

Retain any other explicitly chosen flags from the previous config command.
For ROCm use `.#prepare-strata` instead. This command generates configuration;
it does not launch the GPU engine.

Strata config also defaults to `--expert-cache auto` and `--expert-profile` pointing
at the packaged Strata source's `data/expert-profile.bin`. Override with
`--expert-profile /absolute/path/to/expert-profile.bin` (an existing file) and/or
`--expert-cache COUNT`. Direct Python callers without `PREPARE_STRATA_SOURCE`
must supply the profile explicitly; an unavailable profile fails before writes.
Profile aliases are protected against config, manifest and log collisions even
with `--force`. These defaults do not affect Maya or packing. The default
`--max-context` remains 4096.

### Maya shard layout and config reruns

Maya's pinned `iq_pack.Model` discovers `*-00001-of-000NN.gguf` siblings by
basename (or just the supplied file for an unsplit model). GLM `load_pack`
opens the recorded shard names in `pack/..`; `--gguf` is not a native-path
argument in Maya configs. Keep all shards in `/data/glm/` with the pack at
`/data/glm/pack`, as above. Alternatively, manually create same-basename
symlinks in an existing pack's parent to **every** intended input shard.
Symlinked download snapshots are supported; unrelated same-name files are not.
The wrapper does not move files or create symlinks.

Both Maya pack and config check this layout before changing outputs, including
with `--force`. Unlike Strata config, Maya config requires existing shards;
it cannot generate a config for nonexistent future download paths. This is a
filename/existence/identity check, not GGUF content or pack compatibility
validation. Config still inherits packing options (such as `experts_bin`) from
the pack manifest when present; that manifest does not replace layout checks.

Adding layout symlinks needs **no repack**. Rerun only `maya config` with the
same paths and `--force` to replace config and manifest safely. Force does not
bypass input/output collision checks; forcing a pack is an upstream in-place
update, not a clean rebuild. Maya now writes `<config>.log` through the HTTP
frontend, with the same path safety protections as Strata. Config generation
does not create or truncate the log or start a server.

## Strata images (CPU encoder)

`strata-vision-cpu` builds the pinned Strata `tools/vision` project against the
same pinned llama.cpp as the engine. It is independent of Maya and works alongside
ROCm or CUDA text inference. It requires an AVX2/FMA/F16C/BMI2 CPU. llama/mtmd are
linked into the executable; Nix retains its C/C++ runtime dependencies. The
server already includes Pillow (image conversion/EXIF orientation), NumPy,
tokenizers and Jinja2; no pip install or upstream setup is needed.

Use the **projector for your exact vision-capable Qwen model family/release**,
not just any file named `mmproj-F16.gguf`. A text-only model cannot gain vision
from a projector. Download only after checking the publisher's model card and
revision. These commands are a template: set the real repository, revision,
projector filename, first model shard, and existing compatible pack first.

```sh
export REPO='OWNER/YOUR-QWEN-GGUF'
export REV='FULL_HUGGING_FACE_COMMIT'
export MMPROJ_NAME='mmproj-F16.gguf'
export MODEL_DIR="$HOME/models/your-qwen"
export GGUF="$MODEL_DIR/your-first-model-shard.gguf"
export PACK="$HOME/models/packs/your-qwen"
export CONFIG="$HOME/.config/strata/your-qwen-vision.json"

# User-initiated download only; omit if the matching file is already local.
nix run .#download-model -- "$REPO" "$MODEL_DIR" \
  --revision "$REV" --include "$MMPROJ_NAME"

nix build .#strata-vision-cpu --out-link result-strata-vision-cpu
# If a compatible native pack does not already exist (all model shards required):
# nix run .#prepare-strata-tools -- strata pack --gguf "$GGUF" --pack "$PACK" --experts-bin

# Writes a NEW config, not the config of an already running instance.
nix run .#prepare-strata -- strata config --gguf "$GGUF" --pack "$PACK" \
  --config "$CONFIG" --mmproj "$MODEL_DIR/$MMPROJ_NAME" --vision-device cpu \
  --max-context 4096 --kv int8 --vram-reserve-mib 1024
# For CUDA text inference, replace prepare-strata with prepare-strata-cuda.
# GPU-free config preparation is also available with prepare-strata-tools and
# --exe /absolute/path/to/strata. All Strata wrappers supply the CPU encoder.

# Start only when ready; choose an unused port, leaving existing servers alone.
nix run .#server-strata -- --config "$CONFIG" --port 8091
```

In another terminal, send an image (requires `python3` and `curl` locally):

```sh
python3 - /absolute/path/to/picture.png <<'PY' > /tmp/strata-image-request.json
import base64, json, sys
image = base64.b64encode(open(sys.argv[1], 'rb').read()).decode()
print(json.dumps({'model': 'your-qwen', 'max_tokens': 128, 'messages': [
    {'role': 'user', 'content': [
        {'type': 'text', 'text': 'Describe this picture.'},
        {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,' + image}}
    ]}]}))
PY
curl --fail-with-body http://127.0.0.1:8091/v1/chat/completions \
  -H 'Content-Type: application/json' --data-binary @/tmp/strata-image-request.json
```

### Config schema and limits

`--mmproj` enables vision only for `strata config`; `--vision-exe` overrides the
packaged **CPU** executable, and `--vision-device cpu` is optional. Overrides
must be executable local files. No vision flags means unchanged text-only config.
The generated native `args` include `--vision`, and the frontend gets:

```json
"vision": {
  "exe": "/nix/store/...-strata-vision-cpu-0.1.47/bin/strata-vision",
  "mmproj": "/absolute/path/to/matching-mmproj.gguf",
  "model": "/absolute/path/to/first-model-shard.gguf",
  "gpu": false,
  "max_tokens": 300
}
```

The encoder opens `model` for vocabulary/metadata, not as another text engine.
300 image tokens follows pinned setup's CPU recommendation; native thread count
is automatic (up to 16). Advanced users can edit `vision.max_tokens` or
`vision.threads`; more image tokens cost CPU time and context. Regeneration
replaces manual edits. CPU vision adds no GPU reserve, no elastic VRAM policy,
and no device index. The native protocol alone does not turn image URLs into
embeddings: use the packaged HTTP frontend.

Projector and encoder paths join the protected input set: config, manifest and
log cannot overwrite them, including symlink/hardlink aliases, even with
`--force`. Config generation checks existence, not GGUF contents or model pairing.
Keep the preparation/encoder store paths rooted (for example the build link
above); JSON references alone are not Nix GC roots.

Inspected sources: pinned `tools/vision/CMakeLists.txt`, `strata_vision.cpp`,
`setup.py`, `strata_control/vision_reader.py`, `acquisition.py`, and the frontend's
`Vision`, `engine_args`, device/environment and lazy/elastic policies; also the
local `~/ai/Strata` checkout. The flake pin is authoritative (notably its HIP
vision build includes a PIC fix missing in the local checkout).
Upstream optionally builds `strata-vision-hip` with explicit architecture targets
and a separate CPU fallback. Its automatic GPU policy generally starts at 24 GB
VRAM; this flake deliberately does **not** implement that hardware-dependent
selection or package a GPU encoder yet. CPU vision is the only supported device,
even when text inference uses AMD. No ROCm vision build or real-image/model
inference has been validated. No models are downloaded by builds or preparation;
Strata serving additionally sets `STRATA_NO_ACQUIRE=1`.
