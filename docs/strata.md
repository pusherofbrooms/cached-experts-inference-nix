# Strata guide

Read [common preparation and safety](common.md) first. Examples use ROCm;
use `prepare-strata-cuda` for CUDA or `prepare-strata-tools` with an explicit
`--exe /absolute/bin/strata` for GPU-free preparation.

## Pack and serve

```sh
# Strata Qwen native IQ pack; download all sibling shards first.
nix run .#prepare-strata -- strata pack --gguf /data/qwen/first.gguf \
  --pack /data/packs/qwen --experts-bin
nix run .#prepare-strata -- strata config --gguf /data/qwen/first.gguf \
  --pack /data/packs/qwen --config /data/qwen.json \
  --max-context 4096 --expert-cache auto --prefill 512 --kv int8 \
  --adapt-every 0 --pcie-frac 0 --vram-reserve-mib 1024
nix run .#server-strata -- --config /data/qwen.json
```

Strata configs use `--pack --native`. PLE models may require explicit
`--ple-gguf /data/shard.gguf` on config generation. Packing discovers PLE and
records it in the sidecar; config can read it without opening weights.

Strata supports `--kv fp16|int8|q4_0|k8v4`. For an explicitly chosen 128K
context, add `--max-context 131072 --kv k8v4 --vram-reserve-mib 2048`;
this is not a default or a guarantee that the context fits.

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

For Strata, `--mmproj` enables vision only for `strata config`; `--vision-exe` overrides the
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
even when text inference uses AMD. GPU vision remains unvalidated; CPU image inference has user-reported validation
(see [status](../README.md#validation-status)). No models are downloaded by builds or preparation;
Strata serving additionally sets `STRATA_NO_ACQUIRE=1`.

