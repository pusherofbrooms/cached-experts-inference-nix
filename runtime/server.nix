{ pkgs, src, name }:
let
  python = pkgs.python3.withPackages (p: [ p.numpy p.jinja2 p.tokenizers p.pillow p.regex p.sentencepiece p.protobuf ]);
in pkgs.writeShellApplication {
  name = "server-${name}";
  text = ''
    export PYTHONPATH=${src}:${src}/tools
    exec ${python}/bin/python -m serve.server --engine strata --host 127.0.0.1 "$@"
  '';
}
