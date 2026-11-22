# Hy4-Flash with Strata (experimental ROCm)

Strata v0.2.18 runs Hy4-Flash with a separate `strata-hy4` executable. This
flake packages the HIP runner for **x86_64 Linux, `gfx1100`**. Start with text
on one card; real model inference, vision and DFly2 are unverified here. The
existing `strata-rocm`, `strata-cuda` and `strata-metal` executables do not run
Hy4-Flash. Hy4 CUDA, Metal and SYCL packages are not provided.

## Build and acquire weights explicitly

From the flake checkout, build the runner and keep its result link as a GC root:

```sh
nix build .#strata-hy4-rocm --out-link result-strata-hy4-rocm
./result-strata-hy4-rocm/bin/strata-hy4 --version
```

Review the [model card](https://huggingface.co/strata-engine/Hy4-Flash-Strata)
and the [Tencent HY license](https://huggingface.co/tencent/Hy4-flash/blob/main/LICENSE).
Hy4-Flash uses **prebuilt `.strata` files**, not GGUFs. This example explicitly
selects SQ-64, the smallest build. The main file and required shared Engram
shard total about **130 GB (121 GiB)**; use a fast SSD and leave extra free
space. Engram is read during inference. The upstream model card recommends
64 GB RAM for SQ-64; this is not a fit guarantee. No vision weights are needed
for this recipe.

```sh
work="$HOME/models/hy4-flash"
mkdir -p "$work"
nix run .#download-model -- strata-engine/Hy4-Flash-Strata "$work" \
  --revision 995bac227ab7e0d221772be641e2b5954a9fac06 \
  --include Hy4-Flash-SQ-64.strata Hy4-Flash-Engram-Q8.strata

# Verify both files against the published release checksums.
(cd "$work" && sha256sum --check - <<'SUMS'
b2cc38fe4bded35df82a701f9e868ea09c6fcd9be52c8446289e394f804417b2  Hy4-Flash-SQ-64.strata
ab0cf9b3f9c64bd25f6212d4508e7443ab8b9bc133fbccc7bdfb83510a20c2c1  Hy4-Flash-Engram-Q8.strata
SUMS
)
```

## Extract the tokenizer and serve

Continue only after checksum verification succeeds. No IQ packing, expert
profile, `prepare-strata`, or upstream `setup.py` is needed. The tokenizer app
reads the model manifest without loading the weights onto the GPU. Its output
directory must not already exist.

```sh
nix run .#strata-hy4-tokenizer -- "$work/Hy4-Flash-SQ-64.strata" "$work/tokenizer"
nix develop -c python3 - "$work" ./result-strata-hy4-rocm/bin/strata-hy4 <<'PY'
import json
import sys
from pathlib import Path

work, exe = (Path(arg).resolve(strict=True) for arg in sys.argv[1:])
cfg = {
    "exe": str(exe),
    "cwd": str(work),
    "arch": "hy4flash",
    "backend": "hip",
    "model_name": "hy4-flash-sq-64",
    "tokenizer": str(work / "tokenizer"),
    "log": str(work / "strata-hy4.log"),
    "drafting": "off",
    "args": ["--native", str(work / "Hy4-Flash-SQ-64.strata"),
             "--engram", str(work / "Hy4-Flash-Engram-Q8.strata"),
             "--max-context", "8192", "--draft", "off"],
}
with (work / "hy4.json").open("x", encoding="utf-8") as out:
    json.dump(cfg, out, indent=2)
PY
nix run .#server-strata -- --offline --config "$work/hy4.json" --port 8080
```

Once loading completes, check `http://127.0.0.1:8080/health` and point your
OpenAI-compatible client at `http://127.0.0.1:8080/v1`. On failure, read
`$work/strata-hy4.log`. The short context and disabled drafter are for an initial
smoke test. Hy4 places experts itself; do not copy Flash Next KV, VRAM reserve,
MTP, or CPU vision flags into this config. Keep the result link while the
config references its engine; on upgrade, regenerate the config with the new
store path rather than overwriting your original.
