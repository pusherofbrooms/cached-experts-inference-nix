# Maya guide

Read [common preparation and safety](common.md) first. This is the native
GLM workflow, not DeepSeek or EXL3. Use `prepare-maya-cuda` for CUDA or
`prepare-maya-tools` with `--exe /absolute/bin/maya` for GPU-free preparation.

## Pack and serve

```sh
# Maya GLM-5.3-Flash (not DeepSeek or EXL3).
nix run .#prepare-maya -- maya pack --gguf /data/glm/first.gguf \
  --pack /data/glm/pack --compat-bf16
nix run .#prepare-maya -- maya config --gguf /data/glm/first.gguf \
  --pack /data/glm/pack --config /data/glm.json \
  --max-context 4096 --expert-cache auto --vram-reserve-mib 1024
nix run .#server-maya -- --config /data/glm.json
```


Maya configs use `--glm-pack`. For GLM, effective KV modes are **fp16 and int8
only**. The wrapper also accepts `q4_0` and `k8v4`, but those are not effective
GLM KV modes; use `--kv int8` explicitly when desired. Maya receives neither
Strata’s spec/prefill defaults nor its expert-profile defaults.

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

## Optional DFlash

Maya config optionally accepts `--dflash` pointing to an existing local drafter
GGUF; it is never enabled by default and is rejected for Strata and pack.
DFlash2 on AMD is experimental. The GLM-5.3-Flash-DFlash2 drafter is licensed
CC BY-NC-ND 4.0; review its terms before downloading or using it. Download it
separately to, for example,
`~/.local/share/project-maya/drafters/dflash2/GLM-5.3-Flash-DFlash2-Q4_K_M.gguf`.
The wrapper does not download or modify the drafter or validate compatibility.
A separate config is advisable; retain your existing VRAM reserve explicitly:

```sh
nix run .#prepare-maya -- maya config --gguf /data/models/model-00001-of-00002.gguf \
  --pack /data/models/pack --config /data/maya-dflash.json \
  --dflash ~/.local/share/project-maya/drafters/dflash2/GLM-5.3-Flash-DFlash2-Q4_K_M.gguf \
  --vram-reserve-mib 2048
```

User reports DFlash works on RX 7900 XTX, but was slower in their workload:
about 24 tokens/s versus 29–32 without it. This is a workload observation,
not a controlled benchmark or a general performance claim.

## Maya GLM CPU vision (separate from Strata)

`maya-vision-cpu` follows pinned `docs/AMD_MAYA.md`: it runs
`glm5next_patch.py` on a writable copy of pinned llama.cpp, then builds only
`maya-vision` with CUDA disabled and a portable AVX2 CPU baseline. Neither
engine's vision package depends on the other. Preparation wrappers supply their
own `PREPARE_MAYA_VISION_EXE` / `PREPARE_STRATA_VISION_EXE`.

Use existing matching GLM-5.3-Flash files; these commands download nothing.
Set `GGUF` to your actual first shard and `PACK` to the existing pack beside
those shards. Keep your text config; select a new config filename:

```sh
export GGUF="$HOME/.local/share/project-maya/models/YOUR-FIRST-SHARD.gguf"
export PACK="$(dirname "$GGUF")/pack"
export VISION="$HOME/.local/share/project-maya/vision/glm-5.3-flash"
export CONFIG="$HOME/.config/project-maya/glm-vision-cpu.json"

nix build .#maya-vision-cpu --out-link result-maya-vision-cpu
nix run .#prepare-maya-rocm -- maya config \
  --gguf "$GGUF" --pack "$PACK" --config "$CONFIG" \
  --mmproj "$VISION/mmproj-GLM-5.3-Flash-F16.gguf" \
  --vision-model "$VISION/GLM-5.3-Flash-vocab.gguf" --vision-device cpu \
  --max-context 8192 --kv int8 --vram-reserve-mib 3072 --mmap-experts

# Only when ready, on an unused port (not run during implementation):
nix run .#server-maya -- --config "$CONFIG" --port 8099
curl --fail http://127.0.0.1:8099/health
```

Retain your actual context, KV, reserve, mmap and other tuning options instead
of blindly adopting the example. Generation does not merge an old config or
preserve manually added environment settings. For CUDA text inference substitute
`prepare-maya-cuda`; for GPU-free preparation use `prepare-maya-tools` with
`--exe /absolute/path/to/maya`. Do not use `--force` on your existing config.

Maya requires both `--mmproj` and `--vision-model` (alias `--vision-vocab`);
the latter is the **GLM vocabulary GGUF**, not the text model shard. The generated
`vision` object contains `exe`, `mmproj`, `model`, and `gpu: false`. Unlike
Strata, no native `--vision` flag or Qwen token cap is added: pinned Maya's
frontend enables its encoder from the config and uses GLM image markers/GENI.
Strata's existing `--vision` behavior and model-shard vocabulary remain unchanged.
All vision options are config-only; only CPU is supported. Files must exist and
the encoder must be executable. Config, manifest and log outputs cannot alias
any vision input (including symlinks/hardlinks), even with `--force`.
Content/model compatibility is not inferred from filenames.

Check `images: true`, then send a real image using the [Strata image request example](strata.md#strata-images-cpu-encoder)
with port 8099 and your model name. CPU build/linkage and frontend imports were validated previously; GLM text and
CPU image inference now have [user-reported validation](../README.md#validation-status).

## DeepSeek Flash (local GGUF, text only)

Select `--family deepseek` explicitly (Maya only); the default remains native
GLM. This uses `tools/ds4/pack_gguf.py`, not the GLM packer. Numbered siblings
are selected deterministically from `--gguf`; all must exist. Other GGUFs are
not included. External packs are supported and must be writable: the runtime
cwd is the pack itself. Outputs inside input shard directories are rejected,
even with force. GLM/Strata tuning and vision flags are rejected. Config uses
`--vision off --dspark none --tp 1`; reserve maps to `--reserve-mib`.

For the existing five IQ1_S shards (no quality/compatibility promise):

```sh
MODEL="$HOME/.local/share/project-maya/dsflash/models/Maya-XS"
GGUF="$MODEL/DeepSeek-V4.1-Flash-1017-Maya-XS-IQ1_S-00001-of-00005.gguf"
PACK="$HOME/.local/share/project-maya/dsflash/pack"
CONFIG="$HOME/.local/share/project-maya/dsflash/maya.json"
nix run .#prepare-maya-rocm -- maya pack --family deepseek --gguf "$GGUF" --pack "$PACK"
nix run .#prepare-maya-rocm -- maya config --family deepseek \
  --gguf "$GGUF" --pack "$PACK" --config "$CONFIG" \
  --max-context 131072 --vram-reserve-mib 2048
# When ready (not part of preparation):
nix run .#server-maya -- --config "$CONFIG"
```

The user reports successful ROCm text inference with this Maya-XS IQ1_S pack
and the configuration above (128K configured context, 2 GiB VRAM reserve).
This does not establish full-context capacity, output quality, or performance.
Unlike GLM, no shard symlinks are required beside the pack: DeepSeek records
the original model directory. Keep the original shards available.

KV quantization is automatic in this engine: FP8 sliding-window KV and FP4
compressed KV/index keys. Do not pass GLM's `--kv int8` or `--expert-cache auto`.
The generated config selects `maya-ds4`, records `backend: hip` for ROCm, and
logs engine stderr to `<config>.log`. To regenerate an existing config, add
`--force`. For a smaller initial allocation, use `--max-context 4096`.

CUDA uses `prepare-maya-cuda`. GPU-free `prepare-maya-tools` config requires
explicit `--exe /absolute/path/to/maya-ds4`; engine wrappers supply their packaged
`maya-ds4`. No packing or inference is performed by config generation.
No model downloads occur. Implicit optional `Engram` directories in/beside the
model, and `dspark`/`vision` siblings, are currently rejected with an actionable
error rather than silently included. If present, use an isolated local shard
folder (manual symlinks are fine); no links or files are created by the wrapper.
