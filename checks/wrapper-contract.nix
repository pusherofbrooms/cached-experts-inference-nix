{ pkgs }:
pkgs.runCommand "metal-wrapper-contract" {
  nativeBuildInputs = [ pkgs.makeWrapper pkgs.python3 ];
} ''
  cat > recorder <<'PY'
#!${pkgs.python3}/bin/python
import os
expected = None if os.environ['EXPECTED'] == 'absent' else os.environ['EXPECTED']
assert os.environ.get('STRATA_METAL_MTP_GGUF') == expected
assert os.environ.get('STRATA_METAL_DRAFT_VOCAB') == expected
PY
  chmod +x recorder
  makeWrapper "$PWD/recorder" "$PWD/metal" ${import ../runtime/metal-wrapper.nix { }}
  makeWrapper "$PWD/recorder" "$PWD/linux"
  export STRATA_METAL_MTP_GGUF=ambient STRATA_METAL_DRAFT_VOCAB=ambient
  # Direct native entrypoint, without any frontend sanitization.
  EXPECTED=absent ./metal
  EXPECTED=ambient ./linux
  # Frontend config env is applied at subprocess launch, after server startup.
  python - <<'PY'
import os, subprocess
for wrapper, expected in [('./metal', 'absent'), ('./linux', 'config')]:
    env = dict(os.environ, STRATA_METAL_MTP_GGUF='config',
               STRATA_METAL_DRAFT_VOCAB='config', EXPECTED=expected)
    subprocess.run([wrapper], env=env, check=True)
PY
  touch "$out"
''
