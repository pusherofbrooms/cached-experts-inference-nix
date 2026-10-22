{ pkgs, name, src, executable ? null, visionExecutable ? null }:
let
  metadata = builtins.fromJSON (builtins.readFile ../engines/metadata.json);
  llama = pkgs.fetchzip {
    url = "https://github.com/ggml-org/llama.cpp/archive/${metadata.ggml.rev}.tar.gz";
    hash = metadata.ggml.hash;
  };
  python = pkgs.python3.withPackages (p: [ p.numpy p.regex p.pyyaml p.tqdm p.sentencepiece p.protobuf ]);
  prefix = "PREPARE_${pkgs.lib.toUpper name}";
in pkgs.writeShellApplication {
  name = "prepare-${name}";
  runtimeInputs = [ python ];
  text = ''
    export PREPARE_ENGINE=${name}
    ${if visionExecutable == null then "unset ${prefix}_VISION_EXE" else "export ${prefix}_VISION_EXE=${visionExecutable}"}
    ${if executable == null then "unset ${prefix}_EXE" else "export ${prefix}_EXE=${executable}"}
    export ${prefix}_REV=${src.rev}
    export ${prefix}_SOURCE=${src}
    export PREPARE_GGML_REV=${metadata.ggml.rev}
    export STRATA_GGUF_PY=${llama}/gguf-py
    export PYTHONPATH=${llama}/gguf-py
    exec ${python}/bin/python ${./prepare_model.py} "$@"
  '';
}
