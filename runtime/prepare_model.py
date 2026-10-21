"""Local-only pack/config preparation; engine adapters are registered in ADAPTERS."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

GGML_REV = os.environ.get('PREPARE_GGML_REV', 'unknown')
ADAPTERS = {name: os.environ.get(f'PREPARE_{name.upper()}_REV', 'unknown')
            for name in ('strata', 'maya')}


def absolute(value):
    return Path(os.path.abspath(os.path.expanduser(value)))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('engine', choices=ADAPTERS)
    p.add_argument('action', choices=['pack', 'config'])
    p.add_argument('--pack', required=True, type=absolute)
    p.add_argument('--gguf', required=True, type=absolute)
    p.add_argument('--force', action='store_true')
    p.add_argument('--compat-bf16', action='store_true')
    p.add_argument('--experts-bin', action='store_true')
    p.add_argument('--base', type=absolute)
    p.add_argument('--config', type=absolute)
    p.add_argument('--exe', type=absolute)
    p.add_argument('--ple-gguf', type=absolute)
    p.add_argument('--max-context', type=int, default=4096)
    p.add_argument('--vram-reserve-mib', type=int)
    p.add_argument('--expert-cache', help='auto or a nonnegative expert count')
    p.add_argument('--prefill', type=int)
    p.add_argument('--kv', choices=['fp16', 'int8'])
    p.add_argument('--adapt-every', type=int)
    p.add_argument('--pcie-frac', type=float)
    p.add_argument('--mmap-experts', action=argparse.BooleanOptionalAction, default=None)
    a = p.parse_args(argv)
    for key in ('vram_reserve_mib', 'prefill', 'adapt_every'):
        value = getattr(a, key)
        if value is not None and value < (1 if key == 'prefill' else 0):
            p.error('--' + key.replace('_', '-') + ' is out of range')
    if a.pcie_frac is not None and not 0 <= a.pcie_frac <= 1:
        p.error('--pcie-frac must be between 0 and 1')
    if a.expert_cache is not None and a.expert_cache != 'auto':
        if not a.expert_cache.isdecimal():
            p.error('--expert-cache must be auto or a nonnegative integer')
    if a.engine == 'maya' and (a.adapt_every is not None or a.pcie_frac is not None):
        p.error('--adapt-every and --pcie-frac are only configured for Strata')
    manifest = Path(str(a.pack) + '.manifest.json') if a.action == 'pack' else Path(str(a.config) + '.manifest.json') if a.config else None
    a.exe = a.exe or (absolute(os.environ['PREPARE_' + a.engine.upper() + '_EXE']) if os.environ.get('PREPARE_' + a.engine.upper() + '_EXE') else None)
    if a.action == 'config' and (not a.config or not a.exe):
        p.error('config requires --config and --exe (absolute native executable path)')
    options = {}
    if a.action == 'config':
        pack_manifest = Path(str(a.pack) + '.manifest.json')
        if pack_manifest.is_file():
            options = json.loads(pack_manifest.read_text()).get('options', {})
        if not a.ple_gguf and options.get('ple_gguf'):
            a.ple_gguf = absolute(options['ple_gguf'])
    target = a.pack if a.action == 'pack' else a.config
    # Enumerate names without loading weights, and before resolving symlinks:
    # shard aliases can point outside the directory containing the supplied GGUF.
    siblings = list(a.gguf.parent.glob('*.gguf'))
    protected = [a.gguf, a.gguf.parent, a.exe, a.ple_gguf, a.base] + siblings
    directories = [a.base] if a.base and a.base.is_dir() else []
    if a.action == 'config':
        protected += [a.pack, Path(str(a.pack) + '.manifest.json')]
        directories.append(a.pack)

    def locations(path):
        return (path, path.resolve())

    def aliases(left, right):
        # resolve handles symlinks (including dangling ones); samefile also
        # catches hard links, whose resolved pathnames remain distinct.
        return left.resolve() == right.resolve() or (
            left.exists() and right.exists() and left.samefile(right))

    if aliases(target, manifest) or any(
            left in right.parents
            for left, right in ((target.resolve(), manifest.resolve()),
                                (manifest.resolve(), target.resolve()))):
        p.error('outputs collide with each other')
    for output in (target, manifest):
        if any(aliases(output, path) for path in protected if path) or any(out == src or out in src.parents
               for out in locations(output)
               for path in protected if path for src in locations(path)) or any(
                   src in out.parents
                   for out in locations(output)
                   for path in directories for src in locations(path)):
            p.error('output collides with an input or input directory')
    if a.max_context < 1:
        p.error('--max-context must be positive')
    if not a.force and (target.exists() or manifest.exists()):
        p.error('output already exists; use --force to permit overwriting')
    if a.action == 'pack':
        if not a.gguf.is_file():
            p.error('--gguf must be a local file; all shards must be beside it')
        source = os.environ.get('PREPARE_' + a.engine.upper() + '_SOURCE')
        if not source:
            p.error('pack requires the packaged upstream source environment')
        spec = importlib.util.spec_from_file_location('prepare_iq_pack', Path(source) / 'tools/iq_pack.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        model = module.Model(a.gguf)  # upstream validates every sibling shard
        if a.engine == 'strata' and not a.ple_gguf:
            a.ple_gguf = next((path for path, reader in zip(model.paths, model.files)
                               if any(t.name == 'per_layer_token_embd.weight' for t in reader.tensors)), None)
        command = [sys.executable, str(Path(source) / 'tools/iq_pack.py'), '--gguf', str(a.gguf), '--out', str(a.pack)]
        for flag in ('compat_bf16', 'experts_bin'):
            if getattr(a, flag):
                command.append('--' + flag.replace('_', '-'))
        if a.base:
            command += ['--base', str(a.base)]
        subprocess.run(command, check=True)
    else:
        # Both pinned serve.server modules consume exe, args, cwd and tokenizer.
        args = (['--glm-pack', str(a.pack)] if a.engine == 'maya' else
                ['--pack', str(a.pack), '--native', str(a.gguf)])
        args += ['--max-context', str(a.max_context)]
        mmap = a.mmap_experts if a.mmap_experts is not None else options.get('experts_bin', False)
        if mmap:
            args.append('--mmap-experts')
        for key in ('vram_reserve_mib', 'expert_cache', 'prefill', 'kv', 'adapt_every', 'pcie_frac'):
            value = getattr(a, key)
            if value is not None:
                args += ['--' + key.replace('_', '-'), str(value)]
        if a.ple_gguf:
            args += ['--ple-gguf', str(a.ple_gguf)]
        config = {'exe': str(a.exe), 'args': args, 'cwd': str(a.pack.parent),
                  'tokenizer': str(a.pack / 'tokenizer'), 'model_name': a.pack.name}
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(config, indent=2) + '\n')
        command = [str(a.exe), '--serve'] + args
        print(f'HTTP: server-{a.engine} --config {a.config}')
        print('Native resident protocol (not HTTP):')
        print(shlex.join(command))
        print('Config-only generation does not validate model artifacts or compatibility.')
    record = {'schema': 1, 'engine_revision': ADAPTERS[a.engine], 'gguf_py_revision': GGML_REV,
              'options': {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()},
              'command': command, 'http_frontend_packaged': True}
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(record, indent=2) + '\n')


if __name__ == '__main__':
    main()
