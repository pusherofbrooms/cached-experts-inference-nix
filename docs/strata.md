# Strata guide

Read [common preparation and safety](common.md) first. Examples use ROCm;
use `prepare-strata-cuda` for CUDA or `prepare-strata-tools` with an explicit
`--exe /absolute/bin/strata` for GPU-free preparation.

## Pack and serve

Choose a model/quant supported by the pinned upstream Strata documentation
and publisher's model card; the downloader does not determine suitability.
Accept any license and authenticate with Hugging Face for gated repositories.
Download every shard of one compatible quant, keeping original basenames together.
This generic pinned template does not select or convert a model:

```sh
nix run .#download-model -- OWNER/MODEL-GGUF /data/qwen \
  --revision FULL_HUGGING_FACE_COMMIT --include 'YOUR-QUANT-*.gguf'
```

Replace the repository, revision and file selection, then use the actual first
shard below. See [download prerequisites](common.md#download-prerequisites)
for provenance and authentication cautions.

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

## Configuration defaults and reference

Strata config generation supplies `--spec 2 --prefill auto` unless overridden.
These are startup requirements, not additional performance tuning: the pinned
`generate.cpp` native IQ check requires spec >= 2, and resident serving requires
positive prefill. `--prefill` accepts `auto` or a positive integer for Strata;
Maya accepts only positive integers for `--prefill` and receives neither default.
`--spec 3` preserves an older spec-3 configuration.

MTP is **optional**: the pinned resident path explicitly falls back to lookup or
plain decoding without it. To retain an existing drafter, pass
`--mtp /absolute/path/to/mtp/rt`. This must be an existing
directory; preparation neither creates nor validates its model contents.
Strata configs also record `log` as `<config>.log`, where the server appends engine
stderr and can report startup errors. Output/log aliases of inputs, MTP contents,
and each other are rejected even with `--force`. Do not change these paths into
aliases after generation.

Strata config also defaults to `--expert-cache auto` and `--expert-profile` pointing
at the packaged Strata source's `data/expert-profile.bin`. Override with
`--expert-profile /absolute/path/to/expert-profile.bin` (an existing file) and/or
`--expert-cache COUNT`. Direct Python callers without `PREPARE_STRATA_SOURCE`
must supply the profile explicitly; an unavailable profile fails before writes.
Profile aliases are protected against config, manifest and log collisions even
with `--force`. These defaults do not affect Maya or packing. The default
`--max-context` remains 4096.

### Repair an existing Strata configuration

Rerun **config only** with `--force` to replace the existing JSON and manifest;
no repack is required. With `GGUF`, `PACK`, and `CONFIG` set to the same absolute
paths used previously, and `MTP` set to your existing drafter directory, a CUDA
command retaining spec 3, MTP and automatic prefill is:

```sh
nix run .#prepare-strata-cuda -- strata config \
  --gguf "$GGUF" --pack "$PACK" --config "$CONFIG" --force \
  --spec 3 --prefill auto --mtp "$MTP"
```

Retain any other explicitly chosen flags from the previous config command.
For ROCm use `.#prepare-strata` instead. This command generates configuration;
it does not launch the GPU engine.

## Strata images (CPU encoder)

Optional: complete the text workflow first.

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
  "exe": "/nix/store/...-strata-vision-cpu-<version>/bin/strata-vision",
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

Relevant sources at the flake's pinned revision: `tools/vision/CMakeLists.txt`,
`strata_vision.cpp`, `setup.py`, `strata_control/vision_reader.py`,
`acquisition.py`, and the frontend's `Vision`, `engine_args`, device/environment
and lazy/elastic policies. The flake pin is authoritative.
Upstream optionally builds `strata-vision-hip` with explicit architecture targets
and a separate CPU fallback. Its automatic GPU policy generally starts at 24 GB
VRAM; this flake deliberately does **not** implement that hardware-dependent
selection or package a GPU encoder yet. CPU vision is the only supported device,
even when text inference uses AMD. GPU vision remains unvalidated; CPU image inference has user-reported validation
(see [status](../README.md#validation-status)). No models are downloaded by builds or preparation;
Strata serving additionally sets `STRATA_NO_ACQUIRE=1`.


## Apple silicon (Metal)

On `aarch64-darwin`, `strata` (also `strata-metal` and the flake default),
`prepare-strata` (also `prepare-strata-metal`) and `server-strata` are available.
`strata-vision-cpu` is also available as a separate encoder package.
Linux outputs are unchanged. No Maya, ROCm, CUDA, GPU vision or inspection-tools
packages are exposed on Darwin. The engine builds the separate upstream
`metal/` project, using Nix's SDK 26 and checksum-verified Apple metal-cpp 26;
it does not use host Xcode/xcrun, detect hardware, or compile for the build CPU.
The minimum deployment target is macOS 15.0 (Apple silicon); SDK 26 is a
build requirement, not a requirement to run macOS 26. The nixpkgs deployment
target hook raises older defaults to 15.0 without lowering newer targets.

Use an upstream-compatible Qwen native IQ/Q2_0 model with its PLE tensor shard,
not Unsloth UD weights (which require a separate repack). Download all shards
explicitly as above, then:

```sh
nix run .#prepare-strata-metal -- strata pack --gguf /data/qwen/first.gguf \
  --pack /data/packs/qwen
nix run .#prepare-strata-metal -- strata config --gguf /data/qwen/first.gguf \
  --pack /data/packs/qwen --config /data/qwen-metal.json
nix run .#server-strata -- --config /data/qwen-metal.json
nix build .#prepare-strata-metal --out-link result-prepare-strata-metal
```

Packing uses shared `tools/iq_pack.py` and requires PLE. Expert formats are
explicitly limited to `IQ2_XXS`, `IQ2_XS`, `IQ2_S`, `IQ3_XXS`, `IQ3_S`,
`IQ4_NL`, `IQ4_XS`, and `Q2_0`; `IQ1_S`/`IQ1_M` are rejected before packing.
Preparation sidecars record the packaged backend as provenance. Config requires `--ple-gguf` or the PLE path in the pack sidecar; it does not
inspect weights. Defaults follow `strata_control/mac_fit.py`: expert cache 64,
prefill auto, spec 2, and mmap experts (even without `--experts-bin`). Context
remains 4096. No RAM probing, automatic sizing or acquisition is performed.
`--vision-model`, `--compat-bf16`, `--base`, PCIe/adaptation/VRAM-reserve
options and `--no-mmap-experts` are rejected. Unsloth repacking,
Gyro and MTP acquisition are outside this workflow.

### Opt-in existing MTP assets (Metal)

Config-only `--mtp /data/mtp/rt` enables the pinned **Qwen3.8-Flash-Next**
MTP contract, not a generic drafter or a Qwen3.6 contract. Supply an existing
`rt` directory and its sibling `/data/mtp/mtp-q2_0.gguf`, already prepared by
matching upstream tools. Metal reads that GGUF directly; it does not require
CUDA's runtime tensor pack. This workflow never fetches or converts MTP assets.
For example, add `--mtp /data/mtp/rt` to the config command above.

Preparation reads at most 1 MiB of GGUF header (v3, `qwen4exp-mtp`, 31 tensors,
Q2_0 routed expert shapes/types, F32 norms and BF16 dense tensors). This is a
format guard, not payload integrity, model pairing, or inference validation.
Optional `rt/draft_vocab.bin` must contain unique little-endian int32 Qwen token
IDs in `[0, 248320)`; absent that file, Metal uses the packaged
`data/draft_vocab.bin`. Both local files are protected against config/manifest/log
overwrite, including symlink and hardlink aliases and `--force`.
Do not change assets after validation; regenerate the config when assets change.

Preparation rejects ambient `STRATA_METAL_MTP_GGUF` and
`STRATA_METAL_DRAFT_VOCAB`. The Darwin HTTP wrapper clears them, and the packaged
Metal executable wrapper clears them again after upstream merges config `env`
into its child environment. Thus neither ambient nor config environment values
can replace validated asset selection when using the packaged executable.
Custom `--exe` programs are outside this guarantee. Linux MTP semantics are
unchanged. Without `--mtp`, no drafter is enabled and defaults are unchanged.
No tuning environment variables are set.

**Untested on M5:** fixtures and model-free builds do not establish MTP loading,
HTTP readiness, acceptance, quality, speed, or memory fit. No MTP model or
inference workload has been run; user runtime validation must wait for M5 access.
Keep the engine and preparation build links as GC roots; config paths alone
retain neither package. Existing configs/assets are never migrated automatically.

For a matching upstream-supported Qwen vision model/projector, add
`--mmproj /data/qwen/matching-mmproj.gguf --vision-device cpu` to the config
command. Preparation uses the packaged `strata-vision-cpu` (or an explicit
`--vision-exe`), adds native `--vision`, and writes the same HTTP `vision`
contract described above: first model shard for vocabulary, `gpu: false`, and
300 image tokens. PLE remains required. Vision flags are config-only; all
protected-input and output-alias checks still apply. Model/projector compatibility
is the user's responsibility and is not established by config generation.

The separate CPU encoder uses pinned llama.cpp/mtmd, SDK 26, deployment target
15.0, and an explicit armv8-a CPU baseline; it has no Metal backend. Upstream's
optional Metal encoder embeds shader source via generated assembly (rather than
requiring offline `xcrun metal` compilation). Upstream supports GPU image
encoding with `strata-vision --gpu`, but this flake deliberately exposes only
CPU encoding for now. The benefit of GPU encoding for this workflow has not
been measured; it should be benchmarked before adding another backend. CPU
encoding can still accompany Metal text inference. No setup/bootstrap script
is run.

Kernel sources, ggml-common.h and data tables are installed beside the resolved
engine executable, as required by upstream runtime compilation. Keep the build
link: generated JSON is not a GC root. Native model-free engine build, frontend
imports/help, native CPU encoder build and vision/path-safety fixture tests were
validated on Apple silicon; no Metal model
packing, HTTP readiness, inference, quality or performance has been validated.

MTP header validation checks all 31 pinned names, GGML-order shapes and types
against the inventory in Strata `4b9e05a` (community RTX 5080 manifest)
and `tools/mtp_pack.py`. It reads bounded metadata only: payload integrity,
offsets into real weights, and Metal execution compatibility are not certified.
