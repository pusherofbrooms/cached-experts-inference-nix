{ pkgs, src, name }:
let
  python = pkgs.python3.withPackages (p: [ p.numpy p.jinja2 p.tokenizers p.pillow p.regex p.sentencepiece p.protobuf ]);
in pkgs.writeShellApplication {
  name = "server-${name}";
  text = ''
    ${pkgs.lib.optionalString (name == "strata") "export STRATA_NO_ACQUIRE=1"}
    ${pkgs.lib.optionalString (name == "strata" && pkgs.stdenv.isDarwin) "unset STRATA_METAL_MTP_GGUF STRATA_METAL_DRAFT_VOCAB"}
    ${pkgs.lib.optionalString (name == "maya") "export MAYA_UPDATE_CHECK=0"}
    export PYTHONPATH=${src}:${src}/tools
    exec ${python}/bin/python -m serve.server --engine strata --host 127.0.0.1 "$@"
  '';
}
