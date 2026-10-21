{
  description = "External, model-free builds of SSD/RAM expert-caching inference engines";
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    strata = {
      url = "github:Niko1221/Strata/011c9e600a93bbc9fd451c563501b5831c32bbca";
      flake = false;
    };
    maya = {
      url = "github:mw00/project-maya/35bb2b22a2e2c65b91e24a41d4b026bfa4e89557";
      flake = false;
    };
  };
  outputs = inputs@{ self, nixpkgs, ... }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs { inherit system; config.allowUnfree = true; };
      settings = import ./settings.nix;
      # Add an input above and an entry here to register another engine.
      registry = {
        strata = import ./engines/strata.nix { inherit pkgs; src = inputs.strata; };
        maya = import ./engines/maya.nix { inherit pkgs; src = inputs.maya; };
      };
      configure = backend: drv: drv.overrideAttrs (old: {
        cmakeFlags = builtins.filter (flag:
          !(pkgs.lib.hasPrefix "-DCMAKE_HIP_ARCHITECTURES=" flag)
          && !(pkgs.lib.hasPrefix "-DCMAKE_CUDA_ARCHITECTURES=" flag)
        ) old.cmakeFlags ++ [ (if backend == "rocm"
          then "-DCMAKE_HIP_ARCHITECTURES=${settings.rocmArchitecture}"
          else "-DCMAKE_CUDA_ARCHITECTURES=${settings.cudaArchitectures}") ];
      });
      engines = pkgs.lib.concatMapAttrs (name: engine: {
        "${name}" = configure "rocm" engine.rocmPackage;
        "${name}-rocm" = configure "rocm" engine.rocmPackage;
        "${name}-cuda" = configure "cuda" engine.cudaPackage;
        "${name}-tools" = engine.toolsPackage;
      }) registry;
      download = pkgs.writeShellApplication {
        name = "download-model";
        runtimeInputs = [ pkgs.python3Packages.huggingface-hub ];
        text = ''
          if [ "$#" -lt 2 ]; then
            echo 'Usage: download-model REPO DEST [hf download options, e.g. --revision COMMIT --include "*.gguf"]' >&2
            exit 2
          fi
          repo="$1"
          dest="$2"
          shift 2
          mkdir -p "$dest"
          exec hf download "$repo" --local-dir "$dest" "$@"
        '';
      };
    in {
      packages.${system} = engines // {
        server-strata = import ./runtime/server.nix { inherit pkgs; src = inputs.strata; name = "strata"; };
        server-maya = import ./runtime/server.nix { inherit pkgs; src = inputs.maya; name = "maya"; };
        update-engine = pkgs.writeShellApplication {
        name = "update-engine";
        runtimeInputs = [ pkgs.python3 pkgs.git pkgs.nix ];
        text = ''exec python3 ${./scripts/update_engine.py} "$@"'';
      }; download-model = download; prepare-model = import ./runtime/package.nix { inherit pkgs; inherit (inputs) strata maya; executables = { strata = "${engines.strata}/bin/strata"; maya = "${engines.maya}/bin/maya"; }; }; prepare-model-cuda = import ./runtime/package.nix { inherit pkgs; inherit (inputs) strata maya; executables = { strata = "${engines.strata-cuda}/bin/strata"; maya = "${engines.maya-cuda}/bin/maya"; }; }; default = engines.strata; };
      apps.${system} = pkgs.lib.mapAttrs (_: package: {
        type = "app";
        program = pkgs.lib.getExe package;
      }) self.packages.${system};
      checks.${system} = pkgs.lib.mapAttrs' (name: _: pkgs.lib.nameValuePair "${name}-frontend"
        (pkgs.runCommand "${name}-frontend-import" {} ''
          ${self.packages.${system}."server-${name}"}/bin/server-${name} --help > "$out"
          grep -q -- '--config' "$out"
        '')) registry // {
        prepare-model = pkgs.runCommand "prepare-model-tests" { nativeBuildInputs = [ pkgs.python3 ]; } ''
          cp -r ${./runtime} runtime
          cp -r ${./tests} tests
          cp -r ${./scripts} scripts
          python -m unittest discover -s tests -v
          touch "$out"
        '';
        strata-tools = engines.strata-tools;
        maya-tools = engines.maya-tools;
      };
      devShells.${system}.default = pkgs.mkShell {
        packages = [ pkgs.cmake pkgs.ninja pkgs.python3 pkgs.python3Packages.huggingface-hub ];
      };
    };
}
