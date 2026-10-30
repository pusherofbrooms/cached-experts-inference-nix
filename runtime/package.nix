{ pkgs, name, src, executable ? null, backend ? null, visionExecutable ? null, deepseekExecutable ? null }:
let
  metadata = builtins.fromJSON (builtins.readFile ../engines/metadata.json);
  llama = pkgs.fetchzip {
    url = "https://github.com/ggml-org/llama.cpp/archive/${metadata.ggml.rev}.tar.gz";
    hash = metadata.ggml.hash;
  };
  python = pkgs.python3.withPackages (p: [ p.numpy p.regex p.pyyaml p.tqdm p.sentencepiece p.protobuf ]);
  # Maya's DeepSeek packer can follow arbitrary engram.file metadata outside --shards.
  # Keep the explicit-input contract; reject before opening it or writing a pack.
  preparationSource = if name != "maya" then src else pkgs.runCommand "maya-preparation-source" {
    nativeBuildInputs = [ pkgs.python3 ];
  } ''
    cp -R ${src} "$out"
    chmod -R u+w "$out"
    substituteInPlace "$out/tools/ds4/pack_gguf.py" --replace-fail \
      'own = kv.get("%s.engram.file" % arch)' \
      $'own = kv.get("%s.engram.file" % arch)\n    if own:\n        raise SystemExit("unsupported optional DeepSeek input discovery: engram.file")'
    # Header-only regression: no model data, GPU, or real packing.
    python - "$out" <<'PY'
    import pathlib, struct, subprocess, sys, tempfile
    def string(value):
        data = value.encode()
        return struct.pack('<Q', len(data)) + data
    with tempfile.TemporaryDirectory() as tmp:
        root = pathlib.Path(tmp)
        for value in ('extra.gguf', '../extra.gguf', '/absolute/extra.gguf'):
            header = b'GGUF' + struct.pack('<IQQ', 3, 0, 2)
            for key, val in [('general.architecture', 'deepseek41'), ('deepseek41.engram.file', value)]:
                header += string(key) + struct.pack('<I', 8) + string(val)
            (root / 'model.gguf').write_bytes(header)
            result = subprocess.run([sys.executable, sys.argv[1] + '/tools/ds4/pack_gguf.py',
                '--model', tmp, '--out', str(root / 'pack'), '--shards', 'model.gguf'],
                capture_output=True, text=True)
            assert result.returncode != 0, result
            assert 'unsupported optional DeepSeek input discovery: engram.file' in result.stderr, result.stderr
            assert not (root / 'pack').exists()
    PY
  '';
  prefix = "PREPARE_${pkgs.lib.toUpper name}";
in pkgs.writeShellApplication {
  name = "prepare-${name}";
  runtimeInputs = [ python ];
  text = ''
    export PREPARE_MTP_SCHEMA=${./metal_mtp_schema.json}
    export PREPARE_ENGINE=${name}
    ${if backend == null then "unset PREPARE_BACKEND" else "export PREPARE_BACKEND=${backend}"}
    ${if deepseekExecutable == null then "unset ${prefix}_DS4_EXE" else "export ${prefix}_DS4_EXE=${deepseekExecutable}"}
    ${if visionExecutable == null then "unset ${prefix}_VISION_EXE" else "export ${prefix}_VISION_EXE=${visionExecutable}"}
    ${if executable == null then "unset ${prefix}_EXE" else "export ${prefix}_EXE=${executable}"}
    export ${prefix}_REV=${src.rev}
    export ${prefix}_SOURCE=${preparationSource}
    export PREPARE_GGML_REV=${metadata.ggml.rev}
    export STRATA_GGUF_PY=${llama}/gguf-py
    export PYTHONPATH=${llama}/gguf-py
    ${pkgs.lib.optionalString (name == "maya") ''
      if [[ "''${1-}" == "--prepare-deepseek-vision" ]]; then
        shift
        exec ${python}/bin/python ${./prepare_vision.py} "$@"
      fi
    ''}
    exec ${python}/bin/python ${./prepare_model.py} "$@"
  '';
}
