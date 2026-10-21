{ pkgs, strata, maya, executables }:
let
  metadata = builtins.fromJSON (builtins.readFile ../engines/metadata.json);
  llama = pkgs.fetchzip {
    url = "https://github.com/ggml-org/llama.cpp/archive/${metadata.ggml.rev}.tar.gz";
    hash = "${metadata.ggml.hash}";
  };
  python = pkgs.python3.withPackages (p: [ p.numpy p.regex p.pyyaml p.tqdm p.sentencepiece p.protobuf ]);
in pkgs.writeShellApplication {
  name = "prepare-model";
  runtimeInputs = [ python ];
  text = ''
    export PREPARE_STRATA_EXE=${executables.strata}
    export PREPARE_MAYA_EXE=${executables.maya}
    export PREPARE_STRATA_REV=${strata.rev}
    export PREPARE_MAYA_REV=${maya.rev}
    export PREPARE_GGML_REV=${metadata.ggml.rev}
    export PREPARE_STRATA_SOURCE=${strata}
    export PREPARE_MAYA_SOURCE=${maya}
    export STRATA_GGUF_PY=${llama}/gguf-py
    export PYTHONPATH=${llama}/gguf-py
    exec ${python}/bin/python ${./prepare_model.py} "$@"
  '';
}
