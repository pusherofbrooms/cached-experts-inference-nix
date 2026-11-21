{ pkgs, src, name }:
let
  inherit (pkgs) lib;
  isStrata = name == "strata";
  python = pkgs.python3.withPackages (p: [ p.numpy p.jinja2 p.pillow p.regex ]
    ++ (if isStrata then [ p.pyyaml p.tqdm p.requests p.psutil ]
        else [ p.tokenizers p.sentencepiece p.protobuf ]));
  # Preserve opt-in without overriding explicit config or environment settings.
  frontendSource = if name == "maya" then pkgs.runCommand "maya-frontend-source" {} ''
    cp -r ${src} "$out"
    chmod u+w "$out/serve/server.py"
    substituteInPlace "$out/serve/server.py" --replace-fail \
      'return bool(v) if v is not None else True' \
      'return bool(v) if v is not None else False'
  '' else "${src}/terminal";
  terminalSource = "${pkgs.applyPatches {
    name = "strata-terminal-source";
    inherit src;
    patches = [ ./strata-terminal-0.2.17.patch ];
  }}/terminal";
  environment = source: ''
    ${if isStrata then ''
      export STRATA_NO_ACQUIRE=1 STRATA_UPDATE_CHECK=0 STRATA_KEEP_LEFTOVERS=1
      ${lib.optionalString pkgs.stdenv.hostPlatform.isDarwin "unset STRATA_METAL_MTP_GGUF STRATA_METAL_DRAFT_VOCAB"}
    '' else "export MAYA_UPDATE_CHECK=0"}
    export PYTHONPATH=${source}:${source}/tools
  '';
  terminal = pkgs.writeShellApplication {
    name = "strata-terminal";
    text = ''
      ${environment terminalSource}
      exec ${python}/bin/python -sP ${./strata_terminal.py} "$@"
    '';
  };
in pkgs.writeShellApplication {
  name = "server-${name}";
  passthru = { inherit python frontendSource; } // lib.optionalAttrs isStrata { inherit terminal; };
  text = ''
    ${environment frontendSource}
    exec ${python}/bin/python ${lib.optionalString isStrata "-sP"} -m serve.server --engine strata --host 127.0.0.1 "$@"
  '';
}
