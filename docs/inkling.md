# Inkling 2 Mini (experimental)

Strata v0.2.20 uses a separate Inkling runner. Pick the runner and preparation
app for your backend:

| Platform | Runner | Preparation app |
| --- | --- | --- |
| Apple silicon | `strata-inkling-metal` | `prepare-inkling` |
| x86_64 Linux ROCm | `strata-inkling-rocm` | `prepare-inkling-rocm` |
| x86_64 Linux CUDA | `strata-inkling-cuda` | `prepare-inkling-cuda` |

ROCm uses `rocmArchitecture` from [`settings.nix`](../settings.nix) (`gfx1100`
by default); CUDA uses the configured nixpkgs CUDA capabilities. On x86_64
Linux, the ROCm runner has been built and smoke-tested with `--version` for
`gfx1100`; CUDA has been evaluated only. To build and smoke-test the runner
before downloading a model, run `nix run .#strata-inkling-rocm -- --version`
(or substitute your runner).

The quick path uses a prebuilt native `.strata` pack. The commands below use
Metal package names; substitute your runner and preparation app from the table.
No conversion, GPU work, or model downloads happen during a Nix build.

From the flake checkout, choose a pack from the [Strata model card](https://huggingface.co/strata-engine/Inkling-2-Mini-Strata)
that leaves memory for macOS and the KV cache. This example explicitly selects
SQ-10 (~10.47 GB) and pins the model repository revision. It downloads only the
pack and the fallback tokenizer; verify the pack against the published checksum
before use. Review the model terms separately from the engine license.

```sh
model_dir="$HOME/models/inkling-2-mini"
mkdir -p "$model_dir"
nix shell --inputs-from . nixpkgs#python3Packages.huggingface-hub -c hf download \
  strata-engine/Inkling-2-Mini-Strata Inkling-2-Mini-SQ-10.strata tokenizer.json \
  --revision 04996a1e2c45009b6a04f2af126cb6bdb2ba3587 --local-dir "$model_dir"
nix run .#prepare-inkling -- "$model_dir/Inkling-2-Mini-SQ-10.strata" \
  "$model_dir/serve" --tokenizer-json "$model_dir/tokenizer.json"
nix run .#server-strata -- --config "$model_dir/serve/off.json"
```

`prepare-inkling` reads the pack manifest and writes the tokenizer and two local
HTTP configs in a **new** directory. It does not download or change weights.
If preparation fails after creating the directory, choose a new output path for
the retry. The configs contain a Nix store path, not a GC root. To keep the
runner across garbage collection, retain it with
`nix build .#strata-inkling-metal --out-link "$model_dir/runner"`. After a flake
update, regenerate the configs in a new directory.

## Compare baseline and DSpark

Inkling calls its speculative drafter **DSpark**. The native packs already
include it; no external drafter or Qwen `--mtp` assets are needed. Inspect the
pack first:

```sh
nix run .#strata-inkling-metal -- --info "$model_dir/Inkling-2-Mini-SQ-10.strata"
```

Look for `drafter=dspark`. Run the baseline (`off.json` above), stop it, then
start `nix run .#server-strata -- --config "$model_dir/serve/dspark.json"`.
Both configs use one slot and listen on loopback. Use the same prompt, sampling
parameters and output limit in both runs; compare decode speed after warmup.
Check the engine reports DSpark is active: a pack containing a drafter does not
prove the backend initialized it. The Apple silicon setup has been checked without weights; real inference,
quality and speed are not yet validated on any backend.
