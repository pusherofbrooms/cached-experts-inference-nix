{
  description = "External, model-free builds of SSD/RAM expert-caching inference engines";
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    strata = {
      url = "github:Niko1221/Strata/4b9e05a2a806a3fe24f7b2560d98626c37dade35";
      flake = false;
    };
    maya = {
      url = "github:mw00/project-maya/f26d8bab43f9b10f0475986b541dc46ada45285e";
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
      preparation = pkgs.lib.concatMapAttrs (name: _: let
        prepare = backend: executable: import ./runtime/package.nix {
          inherit pkgs name executable backend;
          deepseekExecutable = if name == "maya" && executable != null
            then "${builtins.dirOf executable}/maya-ds4" else null;
          visionExecutable = "${registry.${name}.visionPackage}/bin/${name}-vision";
          src = inputs.${name};
        };
        rocm = prepare "hip" "${engines.${name + "-rocm"}}/bin/${name}";
      in {
        "prepare-${name}" = rocm;
        "prepare-${name}-rocm" = rocm;
        "prepare-${name}-cuda" = prepare "cuda" "${engines.${name + "-cuda"}}/bin/${name}";
        "prepare-${name}-tools" = prepare null null;
      }) registry;
      servers = pkgs.lib.mapAttrs' (name: _: pkgs.lib.nameValuePair "server-${name}"
        (import ./runtime/server.nix { inherit pkgs name; src = inputs.${name}; })) registry;
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
      packages.aarch64-darwin = let
        pkgs = import nixpkgs { system = "aarch64-darwin"; };
        metal = import ./engines/strata-metal.nix { inherit pkgs; src = inputs.strata; };
        prepare = import ./runtime/package.nix {
          inherit pkgs; name = "strata"; src = inputs.strata;
          executable = "${metal}/bin/strata"; backend = "metal";
        };
      in {
        strata = metal;
        strata-metal = metal;
        default = metal;
        prepare-strata = prepare;
        prepare-strata-metal = prepare;
        server-strata = import ./runtime/server.nix { inherit pkgs; name = "strata"; src = inputs.strata; };
      };
      apps.aarch64-darwin = nixpkgs.lib.mapAttrs (_: package: {
        type = "app"; program = nixpkgs.lib.getExe package;
      }) self.packages.aarch64-darwin;
      checks.aarch64-darwin = let
        pkgs = import nixpkgs { system = "aarch64-darwin"; };
      in {
        strata-frontend = pkgs.runCommand "strata-frontend-import" {} ''
          ${self.packages.aarch64-darwin.server-strata}/bin/server-strata --help > "$out"
          grep -q -- --config "$out"
        '';
        prepare-model = pkgs.runCommand "prepare-model-tests" { nativeBuildInputs = [ pkgs.python3 ]; } ''
          cp -r ${./runtime} runtime
          cp -r ${./tests} tests
          cp -r ${./scripts} scripts
          python -m unittest discover -s tests -v
          touch "$out"
        '';
      };
      packages.${system} = engines // preparation // servers // {
        strata-vision-cpu = registry.strata.visionPackage;
        maya-vision-cpu = registry.maya.visionPackage;
        update-engine = pkgs.writeShellApplication {
          name = "update-engine";
          runtimeInputs = [ pkgs.python3 pkgs.git pkgs.nix ];
          text = ''exec python3 ${./scripts/update_engine.py} "$@"'';
        };
        download-model = download;
        # Compatibility aliases are intentionally Strata-only.
        prepare-model = preparation.prepare-strata;
        prepare-model-cuda = preparation.prepare-strata-cuda;
        default = engines.strata;
      };
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
