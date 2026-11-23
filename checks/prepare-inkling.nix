{ pkgs, src, prepare, inkling }:
pkgs.runCommand "prepare-inkling-contract" { nativeBuildInputs = [ (pkgs.python3.withPackages (p: [ p.regex ])) ]; } ''
  ${pkgs.lib.getExe inkling} --help | grep -- '--draft auto|dspark|off'
  PYTHONPATH=${src}/terminal python - <<'PY'
  import json, struct
  from compression import zstd
  from pathlib import Path
  from tools.test_inkling_tokenizer import _minimal_o200k_hf_json

  raw = zstd.compress(json.dumps({
      'chat_template': '{{ messages }}',
      'tokenizer_json': json.dumps(_minimal_o200k_hf_json()),
  }).encode())
  header = bytearray(4096)
  header[:8] = b'STRATA\x00\x01'
  struct.pack_into('<QQ', header, 16, len(header), len(raw))
  Path('fixture.strata').write_bytes(header + raw)
  PY
  prepare=${pkgs.lib.getExe prepare}
  "$prepare" fixture.strata serve
  if "$prepare" fixture.strata serve; then exit 1; fi
  ln -s untouched dangling
  if "$prepare" fixture.strata dangling; then exit 1; fi
  test ! -e untouched
  python - <<'PY'
  import json
  from pathlib import Path

  for mode in ('off', 'dspark'):
      cfg = json.loads(Path(f'serve/{mode}.json').read_text())
      assert cfg['arch'] == 'inkling2_loop'
      assert cfg['host'] == '127.0.0.1'
      assert cfg['args'] == ['--native', str(Path('fixture.strata').resolve()),
                             '--parallel', '1', '--draft', mode]
      assert (Path(cfg['tokenizer']) / 'vocab.json').is_file()
  PY
  touch "$out"
''
