{ pkgs, terminal }:
pkgs.runCommand "strata-terminal-contract" { nativeBuildInputs = [ pkgs.python3 ]; } ''
  export HOME="$PWD/home" XDG_CONFIG_HOME="$PWD/config" XDG_STATE_HOME="$PWD/state"
  mkdir "$HOME"
  terminal=${terminal}/bin/strata-terminal
  "$terminal" --version
  "$terminal" serve --help | grep -q -- --ready-file
  "$terminal" status --help | grep -q -- --url
  for command in launch install update code; do
    if "$terminal" "$command" > refused 2>&1; then
      echo "unexpected command: $command" >&2
      exit 1
    fi
    grep -q 'only serve, status' refused
  done
  printf '{"exe":"%s/bin/false","tokenizer":"%s/tokenizer","args":[],"host":"0.0.0.0"}\n' \
    ${pkgs.coreutils} "$PWD" > run.json
  cp run.json original.json
  "$terminal" serve --config "$PWD/run.json" --host 0.0.0.0 --batch-groups 2 --json > plan.json
  python - <<'PY'
  import json
  plan = json.load(open('plan.json'))['plan']
  assert plan['effective']['host'] == '127.0.0.1', plan
  assert plan['effective']['offline'] is True, plan
  assert plan['effective']['batch_groups'] == 2, plan
  assert '--offline' in plan['server_argv'], plan
  PY
  cmp run.json original.json
  python - <<'PY'
  import json
  with open('run.json') as stream:
      config = json.load(stream)
  for name, fields in [('hook', {'before_load': ['must-not-run']}),
                       ('remote', {'vision': {'allowed_hosts': ['example.org']}})]:
      with open(f'{name}.json', 'w') as stream:
          json.dump(dict(config, **fields), stream)
  PY
  for name in hook remote; do
    if "$terminal" serve --config "$PWD/$name.json" --plan > refused 2>&1; then
      echo "offline plan accepted $name" >&2
      exit 1
    fi
    grep -q 'offline serving requires' refused
  done
  touch "$out"
''
