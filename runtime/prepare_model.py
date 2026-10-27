"""Local-only pack/config preparation; engine adapters are registered in ADAPTERS."""
import argparse
import importlib.util
import json
from itertools import combinations
import os
import re
import stat
import struct
from pathlib import Path
import shlex
import subprocess
import sys

GGML_REV = os.environ.get('PREPARE_GGML_REV', 'unknown')
ADAPTERS = {name: os.environ.get(f'PREPARE_{name.upper()}_REV', 'unknown')
            for name in ('strata', 'maya')}


def absolute(value):
    return Path(os.path.abspath(os.path.expanduser(value)))


def validate_metal_mtp(path):
    """Read bounded GGUF metadata only, never tensor payloads (pinned Qwen MTP)."""
    if not path.is_file():
        raise ValueError('MTP GGUF must be an existing local file')
    with path.open('rb') as f:
        def read(n):
            if n > 65536 or f.tell() + n > 1048576:
                raise ValueError('MTP GGUF header exceeds limits')
            data = f.read(n)
            if len(data) != n:
                raise ValueError('truncated MTP GGUF header')
            return data

        def number(fmt):
            return struct.unpack('<' + fmt, read(struct.calcsize(fmt)))[0]

        def string():
            return read(number('Q')).decode('utf-8')

        if read(4) != b'GGUF' or number('I') != 3:
            raise ValueError('MTP requires GGUF v3')
        tensors, entries = number('Q'), number('Q')
        if tensors != 31 or entries > 128:
            raise ValueError('expected pinned Qwen 31-tensor MTP header')
        metadata = {}
        for _ in range(entries):
            key, kind = string(), number('I')
            if key in metadata:
                raise ValueError('duplicate MTP metadata')
            if kind == 8:
                metadata[key] = string()
            elif kind == 4:
                metadata[key] = number('I')
            else:
                raise ValueError('unsupported MTP metadata type')
        if (metadata.get('general.architecture') != 'qwen4exp-mtp' or
                metadata.get('strata.mtp.expert_format') != 'q2_0'):
            raise ValueError('expected Qwen qwen4exp-mtp q2_0 assets')
        # Pinned mtp_fetch inventory + mtp_pack's reversed GGML dimensions/types.
        schema = json.loads(Path(os.environ.get('PREPARE_MTP_SCHEMA', Path(__file__).with_name('metal_mtp_schema.json'))).read_text())
        seen = set()
        for _ in range(tensors):
            name, dims = string(), number('I')
            if name not in schema or name in seen or dims != len(schema[name][0]):
                raise ValueError('invalid Qwen MTP tensor header')
            seen.add(name)
            shape = tuple(number('Q') for _ in range(dims))
            kind, offset = number('I'), number('Q')
            if not all(shape) or offset % 32:
                raise ValueError('invalid MTP shape/offset')
            if (shape, kind) != (tuple(schema[name][0]), schema[name][1]):
                raise ValueError('unexpected pinned Qwen MTP tensor shape/type')
        if seen != schema.keys():
            raise ValueError('incomplete Qwen MTP schema')



def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    packaged_engine = os.environ.get('PREPARE_ENGINE')
    if packaged_engine and packaged_engine not in ADAPTERS:
        p.error(f'unsupported packaged engine: {packaged_engine}')
    p.add_argument('engine', choices=([packaged_engine] if packaged_engine else ADAPTERS))
    p.add_argument('action', choices=['pack', 'config'])
    p.add_argument('--pack', required=True, type=absolute)
    p.add_argument('--gguf', required=True, type=absolute)
    p.add_argument('--family', choices=['native', 'deepseek'], default='native')
    p.add_argument('--force', action='store_true')
    p.add_argument('--compat-bf16', action='store_true')
    p.add_argument('--experts-bin', action='store_true')
    p.add_argument('--base', type=absolute)
    p.add_argument('--config', type=absolute)
    p.add_argument('--exe', type=absolute)
    p.add_argument('--ple-gguf', type=absolute)
    p.add_argument('--mmproj', type=absolute, help='matching local projector (config only)')
    p.add_argument('--vision-model', '--vision-vocab', type=absolute, help='Maya GLM vocabulary GGUF (required for Maya vision)')
    p.add_argument('--vision-exe', type=absolute, help='CPU encoder override (default: packaged encoder)')
    p.add_argument('--vision-device', choices=['cpu'], help='CPU only in this package')
    p.add_argument('--max-context', type=int, default=4096)
    p.add_argument('--vram-reserve-mib', type=int)
    p.add_argument('--expert-cache', help='auto or a nonnegative expert count')
    p.add_argument('--expert-profile', type=absolute, help='Strata profile file (default: packaged profile)')
    p.add_argument('--prefill')
    p.add_argument('--spec', type=int)
    p.add_argument('--mtp', type=absolute)
    p.add_argument('--dflash', type=absolute, help='existing local drafter GGUF (Maya config only)')
    p.add_argument('--kv', choices=['fp16', 'int8', 'q4_0', 'k8v4'])
    p.add_argument('--adapt-every', type=int)
    p.add_argument('--pcie-frac', type=float)
    p.add_argument('--mmap-experts', action=argparse.BooleanOptionalAction, default=None)
    a = p.parse_args(argv)
    metal = os.environ.get('PREPARE_BACKEND') == 'metal'
    if metal:
        if a.engine != 'strata' or a.family != 'native':
            p.error('Metal supports only Strata native IQ preparation')
        for key in ('compat_bf16', 'base', 'vision_model', 'dflash', 'vram_reserve_mib',
                    'adapt_every', 'pcie_frac'):
            if getattr(a, key) is not None and getattr(a, key) is not False:
                p.error('--' + key.replace('_', '-') + ' is unsupported for Metal (native IQ only)')
        if a.mmap_experts is False:
            p.error('--no-mmap-experts is unsupported for Metal')
        a.mmap_experts = True
        if a.action == 'config' and a.expert_cache is None:
            a.expert_cache = '64'
    mtp_inputs = []
    if metal:
        for key in ('STRATA_METAL_MTP_GGUF', 'STRATA_METAL_DRAFT_VOCAB'):
            if key in os.environ:
                p.error(f'{key} is unsupported; use validated --mtp assets')
        if a.mtp:
            if a.action != 'config' or a.mtp.name != 'rt':
                p.error('Metal --mtp is config-only and must name <assets>/rt')
            draft = a.mtp.parent / 'mtp-q2_0.gguf'
            vocab = a.mtp / 'draft_vocab.bin'
            mtp_inputs = [draft, vocab]
            try:
                validate_metal_mtp(draft)
                if vocab.exists() or vocab.is_symlink():
                    size = vocab.stat().st_size
                    if not vocab.is_file() or not 0 < size <= 248320 * 4 or size % 4:
                        raise ValueError('invalid draft vocabulary size')
                    with vocab.open('rb') as f:
                        ids = struct.unpack('<' + 'i' * (size // 4), f.read(size))
                    if len(set(ids)) != len(ids) or any(i < 0 or i >= 248320 for i in ids):
                        raise ValueError('invalid Qwen draft vocabulary IDs')
            except (OSError, ValueError, UnicodeError) as e:
                p.error(str(e))
    deepseek = a.family == 'deepseek'
    if deepseek:
        if a.engine != 'maya':
            p.error('deepseek is Maya only')
        for key in ('compat_bf16', 'experts_bin', 'base', 'ple_gguf', 'mmproj',
                    'vision_model', 'vision_exe', 'vision_device', 'expert_cache',
                    'expert_profile', 'prefill', 'spec', 'mtp', 'dflash', 'kv',
                    'adapt_every', 'pcie_frac', 'mmap_experts'):
            value = getattr(a, key)
            if value is not None and (value is not False or key == 'mmap_experts'):
                p.error('--' + key.replace('_', '-') + ' is unsupported for deepseek')
    if any((a.mmproj, a.vision_exe, a.vision_device, a.vision_model)):
        if a.action != 'config':
            p.error('vision options are only supported for config')
        if a.engine == 'maya' and (not a.vision_model or not a.vision_model.is_file()):
            p.error('Maya vision requires an existing --vision-model GLM vocabulary GGUF')
        if a.engine == 'strata' and a.vision_model:
            p.error('--vision-model is only supported for Maya config')
        if not a.mmproj or not a.mmproj.is_file():
            p.error('vision requires an existing --mmproj file')
        encoder = os.environ.get(f'PREPARE_{a.engine.upper()}_VISION_EXE')
        a.vision_exe = a.vision_exe or (absolute(encoder) if encoder else None)
        if not a.vision_exe or not a.vision_exe.is_file() or not os.access(a.vision_exe, os.X_OK):
            p.error('vision requires an executable --vision-exe or packaged CPU encoder')
        a.vision_device = 'cpu'
    if a.dflash is not None:
        if a.engine != 'maya' or a.action != 'config':
            p.error('--dflash is only supported for Maya config')
        if not a.dflash.is_file():
            p.error('--dflash must be an existing local GGUF file')
    if a.engine == 'maya' and (a.spec is not None or a.mtp is not None):
        p.error('--spec and --mtp are only configured for Strata')
    if a.expert_profile is not None and (a.engine != 'strata' or a.action != 'config'):
        p.error('--expert-profile is only supported for Strata config')
    if a.spec is not None and a.spec < 2:
        p.error('--spec must be at least 2')
    if a.mtp is not None and not a.mtp.is_dir():
        p.error('--mtp must be an existing directory')
    if a.prefill is not None and not (a.engine == 'strata' and a.prefill == 'auto'):
        try:
            a.prefill = int(a.prefill)
        except ValueError:
            p.error('--prefill must be a positive integer (or auto for Strata)')
        if a.prefill < 1:
            p.error('--prefill is out of range')
    if a.engine == 'strata' and a.action == 'config':
        a.expert_cache = a.expert_cache if a.expert_cache is not None else 'auto'
        if a.expert_profile is None and os.environ.get('PREPARE_STRATA_SOURCE'):
            a.expert_profile = absolute(os.environ['PREPARE_STRATA_SOURCE']) / 'data/expert-profile.bin'
        if a.expert_profile is None or not a.expert_profile.is_file():
            p.error('Strata config requires an existing --expert-profile file or packaged PREPARE_STRATA_SOURCE profile')
        a.spec = a.spec if a.spec is not None else 2
        a.prefill = a.prefill if a.prefill is not None else 'auto'
    for key in ('vram_reserve_mib', 'adapt_every'):
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
    a.exe = a.exe or (absolute(os.environ['PREPARE_' + a.engine.upper() + ('_DS4_EXE' if deepseek else '_EXE')]) if os.environ.get('PREPARE_' + a.engine.upper() + ('_DS4_EXE' if deepseek else '_EXE')) else None)
    if a.action == 'config' and (not a.config or not a.exe):
        p.error('config requires --config and --exe (absolute native executable path)')
    options = {}
    if a.action == 'config':
        pack_manifest = Path(str(a.pack) + '.manifest.json')
        if pack_manifest.is_file():
            options = json.loads(pack_manifest.read_text()).get('options', {})
        if not deepseek and not a.ple_gguf and options.get('ple_gguf'):
            a.ple_gguf = absolute(options['ple_gguf'])
    if metal and a.action == 'config' and not a.ple_gguf:
        p.error('Metal requires --ple-gguf or PLE recorded by pack preparation')
    target = a.pack if a.action == 'pack' else a.config
    # Enumerate names without loading weights, and before resolving symlinks:
    # shard aliases can point outside the directory containing the supplied GGUF.
    siblings = list(a.gguf.parent.glob('*.gguf'))
    protected = [a.gguf, a.gguf.parent, a.exe, a.ple_gguf, a.base, a.mtp, a.expert_profile, a.mmproj, a.vision_exe, a.vision_model, a.dflash] + siblings + mtp_inputs
    directories = [a.base] if a.base and a.base.is_dir() else []
    if deepseek:
        directories += [a.gguf.parent] + [path.resolve().parent for path in siblings]
    if a.mtp:
        directories.append(a.mtp)
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

    log = Path(str(a.config) + '.log') if a.action == 'config' else None
    outputs = [target, manifest] + ([log] if log else [])
    for left, right in combinations(outputs, 2):
        if aliases(left, right) or left.resolve() in right.resolve().parents or right.resolve() in left.resolve().parents:
            p.error('outputs collide with each other')
    if log and log.is_dir():
        p.error('log path must name a file')
    for output in outputs:
        if any(aliases(output, path) for path in protected if path) or any(out == src or out in src.parents
               for out in locations(output)
               for path in protected if path for src in locations(path)) or any(
                   src in out.parents
                   for out in locations(output)
                   for path in directories for src in locations(path)):
            p.error('output collides with an input or input directory')
    if a.max_context < 1:
        p.error('--max-context must be positive')
    if deepseek and a.action == 'pack':
        def check_pack_entry(path):
            info = path.lstat()
            if stat.S_ISLNK(info.st_mode) or (stat.S_ISREG(info.st_mode) and info.st_nlink > 1):
                p.error(f'existing DeepSeek pack contains a symlink or hard-linked file: {path}')
            if stat.S_ISDIR(info.st_mode):
                for child in path.iterdir():
                    check_pack_entry(child)

        if a.pack.exists() or a.pack.is_symlink():
            check_pack_entry(a.pack)
    if not a.force and (target.exists() or manifest.exists()):
        p.error('output already exists; use --force to permit overwriting')
    if a.engine == 'maya':
        # Mirror pinned iq_pack.Model discovery without importing/parsing weights.
        # GLM load_pack opens these basenames in pack/.., not beside --gguf.
        match = re.search(r'-(\d{5})-of-(\d{5})\.gguf$', a.gguf.name)
        paths = [a.gguf]
        if match:
            total = int(match.group(2))
            paths = [a.gguf.with_name(a.gguf.name[:match.start()] +
                     '-%05d-of-%05d.gguf' % (i, total)) for i in range(1, total + 1)]
        for shard in paths:
            visible = shard if deepseek else a.pack.parent / shard.name
            if not shard.is_file() or not visible.is_file() or not aliases(shard, visible):
                if deepseek:
                    p.error(f'DeepSeek requires every model shard in the input directory: {shard}. '
                            'Finish downloading all shards. No files were moved or links created.')
                p.error(f'Maya requires every model shard beside the pack: {visible} '
                        f'must be a file or symlink to the intended input {shard}. '
                        'Finish downloading all shards; use --pack <model-directory>/pack '
                        'or manually add matching shard symlinks in the pack parent. '
                        'No files were moved or links created.')
    if deepseek:
        model_dir = a.gguf.parent.resolve()
        for path in (model_dir / 'Engram', model_dir.parent / 'Engram',
                     model_dir.parent / 'dspark', model_dir.parent / 'vision'):
            if path.exists():
                p.error(f'unsupported optional DeepSeek input discovery: {path}; '
                        'use an isolated local shard directory without Engram/dspark/vision siblings')
    if a.action == 'pack':
        if not a.gguf.is_file():
            p.error('--gguf must be a local file; all shards must be beside it')
        source = os.environ.get('PREPARE_' + a.engine.upper() + '_SOURCE')
        if not source:
            p.error('pack requires the packaged upstream source environment')
        if deepseek:
            command = [sys.executable, str(Path(source) / 'tools/ds4/pack_gguf.py'),
                       '--model', str(a.gguf.parent), '--out', str(a.pack),
                       '--shards'] + [path.name for path in paths]
            subprocess.run(command, check=True)
        else:
            spec = importlib.util.spec_from_file_location('prepare_iq_pack', Path(source) / 'tools/iq_pack.py')
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            model = module.Model(a.gguf)  # upstream validates every sibling shard
            if a.engine == 'strata' and not a.ple_gguf:
                a.ple_gguf = next((path for path, reader in zip(model.paths, model.files)
                                   if any(t.name == 'per_layer_token_embd.weight' for t in reader.tensors)), None)
            if metal:
                experts = [t for reader in model.files for t in reader.tensors
                           if t.name.endswith('_exps.weight')]
                # Locked Metal ref_quant.h/cpp and kernels/quant.metal agree on these formats.
                supported = {'IQ2_XXS', 'IQ2_XS', 'IQ2_S', 'IQ3_XXS',
                             'IQ3_S', 'IQ4_NL', 'IQ4_XS', 'Q2_0'}
                if not experts or any(t.type_name not in supported for t in experts):
                    p.error('Metal preparation supports native IQ/Q2_0 experts only; '
                            'Unsloth repacking and other formats are out of scope')
            if metal and not a.ple_gguf:
                p.error('Metal native IQ preparation requires a PLE tensor shard')
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
        for key in ('vram_reserve_mib', 'expert_cache', 'expert_profile', 'prefill', 'kv', 'adapt_every', 'pcie_frac', 'spec', 'mtp', 'dflash'):
            value = getattr(a, key)
            if value is not None:
                args += ['--' + key.replace('_', '-'), str(value)]
        if a.ple_gguf:
            args += ['--ple-gguf', str(a.ple_gguf)]
        if deepseek:
            args = ['--pack', str(a.pack), '--max-context', str(a.max_context),
                    '--vision', 'off', '--dspark', 'none', '--tp', '1']
            if a.vram_reserve_mib is not None:
                args += ['--reserve-mib', str(a.vram_reserve_mib)]
            if not a.pack.is_dir() or not os.access(a.pack, os.W_OK):
                p.error('deepseek config requires an existing writable --pack for runtime state')
        config = {'exe': str(a.exe), 'args': args, 'cwd': str(a.pack if deepseek else a.pack.parent),
                  'tokenizer': str(a.pack / 'tokenizer'), 'model_name': a.pack.name}
        if os.environ.get('PREPARE_BACKEND'):
            config['backend'] = os.environ['PREPARE_BACKEND']
        if a.mmproj:
            if a.engine == 'strata':
                args.append('--vision')
            config['vision'] = {'exe': str(a.vision_exe), 'mmproj': str(a.mmproj),
                                'model': str(a.vision_model or a.gguf), 'gpu': False}
            if a.engine == 'strata':
                config['vision']['max_tokens'] = 300
        if log:
            config['log'] = str(log)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(config, indent=2) + '\n')
        command = [str(a.exe), '--serve'] + args
        print(f'HTTP: server-{a.engine} --config {a.config}')
        print('Native resident protocol (not HTTP):')
        print(shlex.join(command))
        print('Config-only generation does not validate model contents or compatibility.'
              + (' Maya shard layout was checked.' if a.engine == 'maya' else ''))
    record = {'schema': 1, 'engine_revision': ADAPTERS[a.engine], 'gguf_py_revision': GGML_REV,
              'options': {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()},
              'command': command, 'http_frontend_packaged': True}
    if os.environ.get('PREPARE_BACKEND'):
        record['backend'] = os.environ['PREPARE_BACKEND']
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(record, indent=2) + '\n')


if __name__ == '__main__':
    main()
