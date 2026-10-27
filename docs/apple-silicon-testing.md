# Apple silicon testing cookbook (M5)

Run these steps yourself when the Mac and suitable assets are available. This is
not a claim of M5 runtime validation or authorization for an agent to download,
pack, or run models. Builds/checks are model-free; later steps are explicit,
potentially expensive user operations. See [Metal contracts](strata.md#apple-silicon-metal)
and [path safety](common.md#preparation-options-and-safety).

## 1. Clone, record, and check without models

Use native Apple ARM (`aarch64-darwin`, not a Rosetta terminal), macOS **15+**,
and a working Nix installation with `nix-command flakes` enabled (for example in
`~/.config/nix/nix.conf`: `experimental-features = nix-command flakes`).
Git, curl and enough writable disk space are needed. No host Xcode Metal compiler
or upstream bootstrap/pip installation is required; Nix supplies SDK 26.
There is no Darwin dev shell. Do not update the lock for this test.

```sh
git clone https://github.com/pusherofbrooms/cached-experts-inference-nix.git
cd cached-experts-inference-nix
export RUN="$HOME/strata-tests/$(date +%Y%m%d-%H%M%S)"
mkdir -p "$RUN/roots"
{ git rev-parse HEAD; git status --short; sw_vers; uname -m; nix --version; } \
  > "$RUN/host.txt"
cp flake.lock settings.nix engines/metadata.json "$RUN/"
nix flake metadata --json > "$RUN/flake-metadata.json"
nix flake check > "$RUN/check.log" 2>&1
nix flake check --all-systems --no-build > "$RUN/eval-all.log" 2>&1
```

Check each exit status before continuing. Native checks build fixture tests and
frontend help/import checks, not inference. All-system evaluation does not build
or validate Linux GPU execution. Record any evaluation limitations separately.
`flake.lock` records Strata, Maya and nixpkgs commits; metadata records dependency
pins. Preserve local diffs too if the checkout is modified.

## 2. Build and retain packages

```sh
nix build .#strata-metal --out-link "$RUN/roots/engine"
nix build .#prepare-strata-metal --out-link "$RUN/roots/prepare"
nix build .#server-strata --out-link "$RUN/roots/frontend"
nix build .#strata-vision-cpu --out-link "$RUN/roots/encoder"
ls -l "$RUN/roots" > "$RUN/store-paths.txt"
```

Keep these out-links as GC roots throughout testing. JSON store paths are **not**
roots. The encoder is independent and CPU-only. After input changes regenerate
configs using matching packages; do not silently migrate existing configs/packs.

## 3. Choose local assets deliberately

No default model is selected or automatically downloaded. Check the pinned
upstream model documentation and publisher's license/model card before acquisition;
accept gating terms and authenticate separately without putting tokens in logs.
Choose a matching upstream-supported Qwen **native IQ/Q2_0** model, all sibling
shards with original basenames, and its required **PLE tensor shard**. Supported
expert quants: `IQ2_XXS`, `IQ2_XS`, `IQ2_S`, `IQ3_XXS`, `IQ3_S`, `IQ4_NL`,
`IQ4_XS`, `Q2_0`. Not IQ1 or Unsloth UD; this workflow does not repack UD weights.
Record publisher, full revision, filenames, quant and hashes separately.

Optional **explicit download template**, not a ready-to-run model choice:

```sh
# Replace every placeholder; include PLE explicitly if the pattern misses it.
nix shell --inputs-from . nixpkgs#python3Packages.huggingface-hub -c hf download \
  OWNER/MODEL-GGUF --revision FULL_HUGGING_FACE_COMMIT \
  --local-dir "$HOME/models/CHOSEN-MODEL" \
  --include 'CHOSEN-QUANT-*.gguf' 'EXACT-PLE-FILENAME.gguf'
```

Use external user-owned storage, never this repo or `/nix/store`. Do not overlap
inputs with packs/configs/logs, including symlink/hardlink aliases. Keep originals
at recorded paths. Allow space for weights, pack and page cache. Unified memory
is shared with macOS: mmap is not free RAM. Check actual model fit before packing
or loading; stop on sustained memory pressure/swap growth rather than pushing on.

## 4. Pack and generate a fresh text baseline

Set actual absolute input paths; the output names below must not already exist.
No `--force`: inspect failures and choose fresh output paths, not blind retries.

```sh
export GGUF="$HOME/models/CHOSEN-MODEL/ACTUAL-FIRST-SHARD.gguf"
export PLE="$HOME/models/CHOSEN-MODEL/ACTUAL-PLE-SHARD.gguf"
export PACK="$RUN/text-pack"
export CONFIG="$RUN/text.json"
"$RUN/roots/prepare/bin/prepare-strata" strata pack \
  --gguf "$GGUF" --pack "$PACK" > "$RUN/pack.log" 2>&1
# Continue only after successful packing and inspection of outputs/manifest.
"$RUN/roots/prepare/bin/prepare-strata" strata config \
  --gguf "$GGUF" --ple-gguf "$PLE" --pack "$PACK" --config "$CONFIG" \
  --max-context 4096 --kv int8 --prefill auto --spec 2 \
  > "$RUN/config.log" 2>&1
```

Metal defaults to expert cache 64 and mmap experts; preparation does **not** probe
RAM or auto-size memory. `prefill auto` is not a whole-model fit guarantee.
Use a smaller explicit `--expert-cache COUNT`/context if appropriate for measured
capacity and regenerate to a new config. Do not copy Linux PCIe/VRAM reserve flags
(the Metal adapter rejects them). Inspect pack outputs (`index.txt`, `dense.bin`,
`native_experts.txt`, `tokenizer/`) and sidecars. Config generation checks contracts,
not inference compatibility. Engine stderr goes to `<config>.log`, outside the repo.

## 5. Serve and make one short text request

Choose an unused port. Keep the server in the foreground in its own terminal:

```sh
"$RUN/roots/frontend/bin/server-strata" --config "$CONFIG" \
  --host 127.0.0.1 --port 8080
```

In a second terminal restore `RUN` to the same absolute directory, then:

```sh
export BASE=http://127.0.0.1:8080
curl --fail-with-body --connect-timeout 5 --max-time 10 \
  "$BASE/health" -o "$RUN/health.json"
cat "$RUN/health.json"
curl --fail-with-body --max-time 10 "$BASE/v1/models" -o "$RUN/models.json"
cat "$RUN/models.json"
# Pinned Python for the following JSON helpers:
nix shell --inputs-from . nixpkgs#python3
```

At the locked Strata `4b9e05a2a806a3fe24f7b2560d98626c37dade35`,
`serve/server.py` returns HTTP 200 for `/health` with `status`, `service`, and
`loaded`; **200 alone is not loaded readiness**. Require `loaded: true` for this
baseline and inspect engine logs. During loading/warming the connection may fail
or time out; a short curl deadline is not a startup deadline. Wait and retry
manually with logs, not an unbounded loop. These endpoints are not model-free
runtime validation: default startup loads weights. Do not add lazy loading here.
`/v1/models` returns `data` entries with `id`, `architecture`, and `status`.
Select the actual non-alias text entry, never guess an identifier:

```sh
python3 - "$RUN" <<'PY'
import json, pathlib, sys
p = pathlib.Path(sys.argv[1])
models = json.loads((p/'models.json').read_text())['data']
m = next(x for x in models if not x.get('alias_of') and
         'text' in x['architecture']['output_modalities'])
assert m['status']['value'] == 'loaded'
(p/'request.json').write_text(json.dumps({'model': m['id'], 'stream': False,
    'max_tokens': 32, 'temperature': 0,
    'messages': [{'role': 'user', 'content': 'Reply with one short greeting.'}]}))
PY
curl --fail-with-body --max-time 300 "$BASE/v1/chat/completions" \
  -H 'Content-Type: application/json' --data-binary @"$RUN/request.json" \
  -o "$RUN/text-response.json" -w 'HTTP %{http_code} total %{time_total}s\n'
```

Inspect content and errors, not just HTTP success. Record the curl timing; timeout
can cancel a request and is not proof of an engine crash. Save stderr/status too.

## 6. Separate optional CPU vision test

Stop the baseline with **Ctrl-C** first; do not run multiple large instances.
Use an already acquired, exactly matching vision-capable model/projector (a
projector cannot make a text-only model vision-capable). Create a new config:

```sh
export MMPROJ="$HOME/models/CHOSEN-MODEL/MATCHING-PROJECTOR.gguf"
"$RUN/roots/prepare/bin/prepare-strata" strata config \
  --gguf "$GGUF" --ple-gguf "$PLE" --pack "$PACK" --config "$RUN/vision.json" \
  --max-context 4096 --kv int8 --prefill auto --spec 2 \
  --mmproj "$MMPROJ" --vision-device cpu
"$RUN/roots/frontend/bin/server-strata" --config "$RUN/vision.json" \
  --host 127.0.0.1 --port 8080
```

Repeat health/models checks and text-request generation above for this instance.
In the Python shell, build an image request from that actual model ID:

```sh
python3 - "$RUN" /absolute/path/to/local-picture.png <<'PY'
import base64, json, pathlib, sys
p = pathlib.Path(sys.argv[1])
r = json.loads((p/'request.json').read_text())
r['max_tokens'] = 128
url = 'data:image/png;base64,' + base64.b64encode(pathlib.Path(sys.argv[2]).read_bytes()).decode()
r['messages'] = [{'role': 'user', 'content': [
    {'type': 'text', 'text': 'Describe this picture.'},
    {'type': 'image_url', 'image_url': {'url': url}}]}]
(p/'image-request.json').write_text(json.dumps(r))
PY
curl --fail-with-body --max-time 300 "$BASE/v1/chat/completions" \
  -H 'Content-Type: application/json' --data-binary @"$RUN/image-request.json" \
  -o "$RUN/image-response.json" -w 'HTTP %{http_code} total %{time_total}s\n'
```

Use a real PNG (or change MIME type to match). Encoding is CPU-only, text remains
Metal. Record projector identity, image dimensions and encoder logs; this is not
an image-generation endpoint.

## 7. Optional existing MTP, then evidence and stop

Stop vision first. Only for matching **Qwen3.8-Flash-Next**, use existing upstream-
prepared `/absolute/mtp/rt` plus sibling `/absolute/mtp/mtp-q2_0.gguf`.
See [MTP asset contract](strata.md#opt-in-existing-mtp-assets-metal) for bounded
header checks and optional draft vocabulary. No acquisition or conversion here.
Generate `$RUN/mtp.json` with the **same text baseline config command/settings**,
changing only `--config` and adding `--mtp /absolute/mtp/rt`. Serve it on the same
loopback port, repeat readiness/models selection and the same baseline prompt,
token limit and sampling. Preserve both responses and compare memory/errors/
latency; a single pair is not a benchmark or proof of MTP acceptance/quality.
Do not set `STRATA_METAL_MTP_GGUF` or `STRATA_METAL_DRAFT_VOCAB` overrides.

For each run retain backend (Metal text/CPU vision), M5 GPU model/core count,
unified memory, macOS version, repository/input/dependency revisions, build paths,
model revision/quant/shard hashes, preparation manifests, config/cache/context/KV/
prefill/spec settings, optional asset identities, prompts, responses, errors,
startup and request latency, memory pressure and swap observations. Distinguish
build, pack, load, text, vision and MTP results; report untested paths honestly.
Redact credentials/private paths before sharing. Keep logs and sensitive images
outside git. Stop with Ctrl-C in the owning terminal; verify that instance exits.
Do not use broad kill commands or delete models/packs/roots as cleanup.
