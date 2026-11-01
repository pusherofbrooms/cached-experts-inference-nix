{ pkgs, src, name }:
let
  python = pkgs.python3.withPackages (p: [ p.numpy p.jinja2 p.tokenizers p.pillow p.regex p.sentencepiece p.protobuf ]);
  # Preserve opt-in without overriding explicit config or environment settings.
  frontendSource = if name == "maya" then pkgs.runCommand "maya-frontend-source" {} ''
    cp -r ${src} "$out"
    chmod u+w "$out/serve/server.py"
    substituteInPlace "$out/serve/server.py" --replace-fail \
      'return bool(v) if v is not None else True' \
      'return bool(v) if v is not None else False'
  '' else src;
in pkgs.writeShellApplication {
  name = "server-${name}";
  passthru = { inherit python frontendSource; };
  text = ''
    ${pkgs.lib.optionalString (name == "strata") "export STRATA_NO_ACQUIRE=1"}
    ${pkgs.lib.optionalString (name == "strata" && pkgs.stdenv.isDarwin) "unset STRATA_METAL_MTP_GGUF STRATA_METAL_DRAFT_VOCAB"}
    ${pkgs.lib.optionalString (name == "maya") "export MAYA_UPDATE_CHECK=0"}
    export PYTHONPATH=${frontendSource}:${frontendSource}/tools
    exec ${python}/bin/python -m serve.server --engine strata --host 127.0.0.1 "$@"
  '';
}
