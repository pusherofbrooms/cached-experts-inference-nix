import contextlib
import io
import json
import os
import struct
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from runtime.prepare_model import main, ADAPTERS


class Preparation(unittest.TestCase):
    def setUp(self):
        source = self.enterContext(tempfile.TemporaryDirectory())
        self.profile = Path(source)/'data/expert-profile.bin'
        self.profile.parent.mkdir(parents=True)
        self.profile.write_bytes(b'mock profile')
        self.enterContext(patch.dict('os.environ', {'PREPARE_STRATA_SOURCE': source}))

    def test_metal_existing_mtp(self):
        def fixture(path, architecture='qwen4exp-mtp', expert_type=42):
            def string(text):
                data = text.encode()
                return struct.pack('<Q', len(data)) + data
            data = b'GGUF' + struct.pack('<IQQ', 3, 31, 2)
            for key, value in (('general.architecture', architecture),
                               ('strata.mtp.expert_format', 'q2_0')):
                data += string(key) + struct.pack('<I', 8) + string(value)
            schema = json.loads(Path('runtime/metal_mtp_schema.json').read_text())
            tensors = [(name, shape, expert_type if kind == 42 else kind)
                       for name, (shape, kind) in schema.items()]
            for name, shape, kind in tensors:
                data += string(name) + struct.pack('<I', len(shape))
                data += struct.pack('<' + 'Q' * len(shape), *shape)
                data += struct.pack('<IQ', kind, 0)
            path.write_bytes(data)  # Header only: no weights or inference.

        with tempfile.TemporaryDirectory() as tmp, patch.dict('os.environ', {'PREPARE_BACKEND': 'metal'}):
            root = Path(tmp)
            rt = root / 'assets/rt'
            rt.mkdir(parents=True)
            draft = rt.parent / 'mtp-q2_0.gguf'
            vocab = rt / 'draft_vocab.bin'
            config = root / 'run.json'
            ple = root / 'ple.gguf'
            ple.touch()
            args = ['strata', 'config', '--pack', str(root / 'pack'), '--gguf', str(ple),
                    '--ple-gguf', str(ple), '--exe', '/bin/engine', '--config', str(config),
                    '--mtp', str(rt), '--force']
            def rejected(command=args):
                before = config.read_bytes() if config.exists() else None
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    main(command)
                self.assertEqual(config.read_bytes() if config.exists() else None, before)
            rejected()  # missing GGUF
            for architecture, kind in (('llama', 42), ('qwen4exp-mtp', 2)):
                fixture(draft, architecture, kind)
                rejected()
            draft.write_bytes(b'GGUF')
            rejected()
            fixture(draft)
            for optional in (False, True):
                if optional:
                    vocab.write_bytes(struct.pack('<ii', 1, 248319))
                with contextlib.redirect_stdout(io.StringIO()):
                    main(args)
                cfg = json.loads(config.read_text())
                self.assertEqual(cfg['args'][cfg['args'].index('--mtp') + 1], str(rt))
                self.assertNotIn('env', cfg)
            for data in (b'x', struct.pack('<i', -1), struct.pack('<i', 248320), struct.pack('<ii', 1, 1)):
                vocab.write_bytes(data)
                rejected()
            vocab.write_bytes(struct.pack('<i', 1))
            for key in ('STRATA_METAL_MTP_GGUF', 'STRATA_METAL_DRAFT_VOCAB'):
                with patch.dict('os.environ', {key: '/unvalidated'}):
                    rejected()
            # All outputs, including log/manifest aliases, protect both assets.
            for asset in (draft, vocab):
                for suffix in ('', '.manifest.json', '.log'):
                    for hard in (False, True):
                        out = root / ('alias.json' + suffix)
                        if hard:
                            os.link(asset, out)
                        else:
                            out.symlink_to(asset)
                        before = asset.read_bytes()
                        rejected(args + ['--config', str(root / 'alias.json')])
                        self.assertEqual(asset.read_bytes(), before)
                        out.unlink()
            # Input aliases can live outside the assets directory too.
            for asset in (draft, vocab):
                original = asset.read_bytes()
                external = root / 'external-input'
                external.write_bytes(original)
                for hard in (False, True):
                    asset.unlink()
                    if hard:
                        os.link(external, asset)
                    else:
                        asset.symlink_to(external)
                    rejected(args + ['--config', str(external)])
                asset.unlink()
                asset.write_bytes(original)
                external.unlink()
            # Malicious lengths are bounded before attempting large reads.
            draft.write_bytes(b'GGUF' + struct.pack('<IQQQ', 3, 31, 1, 2**40))
            rejected()
            fixture(draft)
            rejected(args + ['--config', str(rt / 'nested.json')])
            rejected(['strata', 'pack', '--gguf', str(ple), '--pack', str(root / 'newpack'), '--mtp', str(rt)])

    def test_metal_pack_contract(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict('os.environ', {'PREPARE_BACKEND': 'metal'}):
            root = Path(tmp)
            model_path = root / 'model.gguf'
            model_path.touch()
            args = ['strata', 'pack', '--gguf', str(model_path), '--pack', str(root / 'pack')]
            expert = SimpleNamespace(name='blk.0.ffn_up_exps.weight', type_name='IQ3_S')
            ple = SimpleNamespace(name='per_layer_token_embd.weight', type_name='Q2_0')
            model = SimpleNamespace(paths=[model_path], files=[SimpleNamespace(tensors=[expert, ple])])
            module = SimpleNamespace(Model=lambda path: model)
            spec = SimpleNamespace(loader=SimpleNamespace(exec_module=lambda mod: None))
            with patch('runtime.prepare_model.importlib.util.spec_from_file_location', return_value=spec), patch(
                    'runtime.prepare_model.importlib.util.module_from_spec', return_value=module), patch(
                    'runtime.prepare_model.subprocess.run') as run:
                main(args)
                self.assertIn('tools/iq_pack.py', run.call_args.args[0][1])
                record = json.loads(Path(str(root / 'pack') + '.manifest.json').read_text())
                self.assertEqual(record['options']['ple_gguf'], str(model_path))
                self.assertEqual(record['backend'], 'metal')
                for tensors in ([expert], *([SimpleNamespace(name=expert.name, type_name=kind), ple]
                                             for kind in ('Q4_K', 'IQ1_S', 'IQ1_M'))):
                    model.files[0].tensors = tensors
                    run.reset_mock()
                    with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                        main(args + ['--force'])
                    run.assert_not_called()

    def test_metal_config_defaults_and_rejections(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict('os.environ', {'PREPARE_BACKEND': 'metal'}):
            root = Path(tmp)
            config = root / 'run.json'
            ple = root / 'ple.gguf'
            ple.touch()
            args = ['strata', 'config', '--pack', str(root / 'pack'),
                    '--gguf', str(root / 'model.gguf'), '--exe', '/bin/engine',
                    '--config', str(config)]
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                main(args)
            self.assertFalse(config.exists())
            Path(str(root / 'pack') + '.manifest.json').write_text(
                json.dumps({'options': {'ple_gguf': str(ple)}}))
            with contextlib.redirect_stdout(io.StringIO()):
                main(args)
            result = json.loads(config.read_text())
            self.assertEqual(result['backend'], 'metal')
            flags = result['args']
            for flag, value in (('--expert-cache', '64'), ('--spec', '2'),
                                ('--prefill', 'auto'), ('--ple-gguf', str(ple))):
                self.assertEqual(flags[flags.index(flag) + 1], value)
            self.assertIn('--mmap-experts', flags)
            self.assertNotIn('--mtp', flags)
            for extra in (['--compat-bf16'], ['--base', tmp], ['--mtp', tmp],
                          ['--mmproj', str(ple)], ['--vision-device', 'cpu'],
                          ['--vram-reserve-mib', '0'], ['--adapt-every', '0'],
                          ['--pcie-frac', '0'], ['--no-mmap-experts']):
                before = config.read_bytes()
                with self.subTest(extra=extra), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    main(args + ['--force'] + extra)
                self.assertEqual(config.read_bytes(), before)
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                main(args + ['--force', '--ple-gguf', str(config)])

    def test_dflash_config_and_input_protection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = root / 'model.gguf'
            model.touch()
            drafter = root / 'drafters' / 'draft.gguf'
            drafter.parent.mkdir()
            drafter.write_bytes(b'drafter input')
            config = root / 'run.json'
            args = ['maya', 'config', '--pack', str(root / 'pack'), '--gguf', str(model),
                    '--exe', '/bin/engine', '--config', str(config)]
            with contextlib.redirect_stdout(io.StringIO()):
                main(args)
                self.assertNotIn('--dflash', json.loads(config.read_text())['args'])
                main(args + ['--force', '--dflash', str(drafter), '--vram-reserve-mib', '2048'])
            flags = json.loads(config.read_text())['args']
            self.assertEqual(flags.count('--dflash'), 1)
            self.assertEqual(flags[flags.index('--dflash') + 1], str(drafter))
            self.assertEqual(flags[flags.index('--vram-reserve-mib') + 1], '2048')
            for suffix in ('', '.manifest.json', '.log'):
                for alias in ('direct', 'symlink', 'hardlink'):
                    with self.subTest(suffix=suffix, alias=alias):
                        output = Path(str(config) + suffix)
                        output.unlink(missing_ok=True)
                        source = drafter
                        if alias == 'direct':
                            output.write_bytes(b'drafter input')
                            source = output
                        elif alias == 'symlink':
                            output.symlink_to(drafter)
                        else:
                            output.hardlink_to(drafter)
                        with contextlib.redirect_stderr(io.StringIO()) as errors, self.assertRaises(SystemExit):
                            main(args + ['--force', '--dflash', str(source)])
                        self.assertIn('output collides with an input', errors.getvalue())
                        self.assertEqual(source.read_bytes(), b'drafter input')
                        self.assertEqual(drafter.read_bytes(), b'drafter input')
                        output.unlink()

    def test_dflash_rejects_invalid_options(self):
        with tempfile.TemporaryDirectory() as tmp:
            draft = Path(tmp) / 'draft.gguf'
            draft.touch()
            cases = [('strata', 'config', draft), ('strata', 'pack', draft),
                     ('maya', 'pack', draft), ('maya', 'config', Path(tmp)),
                     ('maya', 'config', Path(tmp) / 'missing.gguf')]
            for engine, action, path in cases:
                with self.subTest(engine=engine, action=action, path=path):
                    with contextlib.redirect_stderr(io.StringIO()) as errors, self.assertRaises(SystemExit):
                        main([engine, action, '--pack', '/pack', '--gguf', '/model',
                              '--dflash', str(path)])
                    self.assertIn('--dflash', errors.getvalue())

    def test_metal_vision_config_and_input_protection(self):
        with patch.dict('os.environ', {'PREPARE_BACKEND': 'metal'}):
            self.test_vision_config_and_input_protection(metal=True)

    def test_vision_config_and_input_protection(self, metal=False):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model, projector, encoder = (root / n for n in ('model.gguf', 'projector', 'encoder'))
            for path in (model, projector, encoder):
                path.write_text('input')
            encoder.chmod(0o755)
            config = root / 'vision.json'
            args = ['strata', 'config', '--pack', str(root / 'pack'), '--gguf', str(model),
                    '--exe', '/bin/engine', '--config', str(config), '--mmproj', str(projector)]
            if metal:
                ple = root / 'ple.gguf'
                ple.write_text('input')
                args += ['--ple-gguf', str(ple)]
            with patch.dict('os.environ', {'PREPARE_STRATA_VISION_EXE': str(encoder)}):
                with contextlib.redirect_stdout(io.StringIO()):
                    main(args)
                cfg = json.loads(config.read_text())
                self.assertEqual(cfg['vision'], {'exe': str(encoder), 'mmproj': str(projector),
                                                'model': str(model), 'gpu': False, 'max_tokens': 300})
                self.assertEqual(cfg['args'].count('--vision'), 1)
                if metal:
                    self.assertEqual(cfg['backend'], 'metal')
                    self.assertNotIn('--mtp', cfg['args'])
                    for extra in (['--mtp', str(model)], ['--vision-device', 'gpu'],
                                  ['--vision-model', str(model)]):
                        before = config.read_bytes()
                        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                            main(args + ['--force'] + extra)
                        self.assertEqual(config.read_bytes(), before)
                for source in (projector, encoder):
                    for output in (config, Path(str(config) + '.manifest.json'), Path(str(config) + '.log')):
                        for hardlink in (False, True):
                            output.unlink(missing_ok=True)
                            if hardlink:
                                output.hardlink_to(source)
                            else:
                                output.symlink_to(source)
                            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                                main(args + ['--force'])
                            self.assertEqual(source.read_text(), 'input')
                            output.unlink()

    def test_maya_vision_config_and_input_protection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model, projector, encoder = (root / n for n in ('model.gguf', 'projector', 'encoder'))
            for path in (model, projector, encoder):
                path.write_text('input')
            encoder.chmod(0o755)
            vocab = root / 'vocab.gguf'
            vocab.write_text('input')
            config = root / 'vision.json'
            args = ['maya', 'config', '--pack', str(root / 'pack'), '--gguf', str(model),
                    '--exe', '/bin/engine', '--config', str(config), '--mmproj', str(projector), '--vision-vocab', str(vocab)]
            with patch.dict('os.environ', {'PREPARE_MAYA_VISION_EXE': str(encoder)}):
                with contextlib.redirect_stdout(io.StringIO()):
                    main(args)
                cfg = json.loads(config.read_text())
                self.assertEqual(cfg['vision'], {'exe': str(encoder), 'mmproj': str(projector),
                                                'model': str(vocab), 'gpu': False})
                self.assertEqual(cfg['args'].count('--vision'), 0)
                for source in (projector, encoder, vocab):
                    for output in (config, Path(str(config) + '.manifest.json'), Path(str(config) + '.log')):
                        for hardlink in (False, True):
                            output.unlink(missing_ok=True)
                            if hardlink:
                                output.hardlink_to(source)
                            else:
                                output.symlink_to(source)
                            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                                main(args + ['--force'])
                            self.assertEqual(source.read_text(), 'input')
                            output.unlink()

    def test_maya_vision_requires_vocab_and_own_encoder(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model, projector, vocab, encoder = (root / n for n in
                                                ('model.gguf', 'mmproj', 'vocab', 'encoder'))
            for path in (model, projector, vocab, encoder):
                path.touch()
            encoder.chmod(0o755)
            args = ['maya', 'config', '--gguf', str(model), '--pack', str(root / 'pack'),
                    '--config', str(root / 'config.json'), '--exe', '/bin/engine',
                    '--mmproj', str(projector)]
            with patch.dict('os.environ', {'PREPARE_STRATA_VISION_EXE': str(encoder)}, clear=True):
                for extra in ([], ['--vision-model', str(vocab)],
                              ['--vision-model', str(root)],
                              ['--vision-model', str(vocab), '--vision-exe', str(vocab)]):
                    with self.subTest(extra=extra), contextlib.redirect_stderr(io.StringIO()):
                        with self.assertRaises(SystemExit):
                            main(args + extra)
                self.assertFalse((root / 'config.json').exists())

    def test_vision_rejects_invalid_options(self):
        cases = [(['maya', 'config'], ['--mmproj', '/missing']),
                 (['maya', 'config'], ['--vision-model', '/missing']),
                 (['maya', 'pack'], ['--vision-model', '/missing']),
                 (['strata', 'config'], ['--vision-model', '/missing']),
                 (['strata', 'pack'], ['--vision-device', 'cpu']),
                 (['strata', 'config'], ['--vision-exe', '/missing']),
                 (['strata', 'config'], ['--mmproj', '/missing']),
                 (['strata', 'config'], ['--vision-device', 'gpu'])]
        for prefix, options in cases:
            with self.subTest(options=options), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    main(prefix + ['--pack', '/pack', '--gguf', '/model'] + options)

    def test_packaged_engine_rejects_other_engine(self):
        for engine, other in [('strata', 'maya'), ('maya', 'strata')]:
            with patch.dict('os.environ', {'PREPARE_ENGINE': engine}):
                with contextlib.redirect_stderr(io.StringIO()) as errors:
                    with self.assertRaises(SystemExit) as exit:
                        main([other, 'config', '--pack', '/pack', '--gguf', '/model'])
                self.assertEqual(exit.exception.code, 2)
                self.assertIn('invalid choice', errors.getvalue())

    def test_tools_config_requires_executable(self):
        with patch.dict('os.environ', {}, clear=True):
            with contextlib.redirect_stderr(io.StringIO()) as errors:
                with self.assertRaises(SystemExit):
                    main(['maya', 'config', '--pack', '/pack', '--gguf', '/model',
                          '--config', '/config.json'])
            self.assertIn('config requires --config and --exe', errors.getvalue())

    def test_quantized_kv_long_context(self):
        for engine in ADAPTERS:
            for kv in ('q4_0', 'k8v4'):
                with self.subTest(engine=engine, kv=kv), tempfile.TemporaryDirectory() as tmp:
                    root = Path(tmp)
                    config = root/'run.json'
                    (root/'model.gguf').touch()
                    with contextlib.redirect_stdout(io.StringIO()):
                        main([engine, 'config', '--pack', str(root/'pack'),
                              '--gguf', str(root/'model.gguf'), '--exe', '/bin/engine',
                              '--config', str(config), '--kv', kv,
                              '--max-context', '131072', '--vram-reserve-mib', '2048'])
                    args = json.loads(config.read_text())['args']
                    for flag, value in (('--kv', kv), ('--max-context', '131072'),
                                        ('--vram-reserve-mib', '2048')):
                        self.assertEqual(args[args.index(flag) + 1], value)

    def test_configs(self):
        for engine in ADAPTERS:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp).resolve()
                (root/'a.gguf').touch()
                args = [engine, 'config', '--pack', str(root/'pack'), '--gguf', str(root/'a.gguf'), '--exe', '/bin/engine', '--config', str(root/'run.json')]
                with contextlib.redirect_stdout(io.StringIO()):
                    main(args)
                    with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                        main(args)
                    main(args + ['--force'])
                cfg = json.loads((root/'run.json').read_text())
                self.assertEqual(cfg['exe'], '/bin/engine')
                self.assertEqual(cfg['tokenizer'], str(root/'pack/tokenizer'))
                self.assertIn('--glm-pack' if engine == 'maya' else '--native', cfg['args'])
                record = json.loads((root/'run.json.manifest.json').read_text())
                self.assertEqual(record['engine_revision'], ADAPTERS[engine])

    def test_symlink_path_preserved(self):
        from runtime.prepare_model import absolute
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'real').touch()
            (root/'alias').symlink_to(root/'real')
            self.assertEqual(absolute(root/'alias'), root/'alias')

    def test_collision_even_force(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for output in (root, root/'a.gguf'):
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    main(['strata', 'pack', '--pack', str(output), '--gguf', str(root/'a.gguf'), '--force'])

    def test_force_preserves_inputs(self):
        for action in ('config', 'pack'):
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                inputs = root/'inputs'
                inputs.mkdir()
                gguf = inputs/'model-00001-of-00002.gguf'
                shard = inputs/'model-00002-of-00002.gguf'
                external = root/'external.bin'
                gguf.write_bytes(b'first shard')
                external.write_bytes(b'second shard')
                shard.symlink_to(external)
                pack = root/'pack'
                (pack/'tokenizer').mkdir(parents=True)
                dense = pack/'dense.bin'
                token = pack/'tokenizer/vocab.json'
                dense.write_bytes(b'dense weights')
                token.write_bytes(b'tokenizer')
                saved = {p: p.read_bytes() for p in (gguf, shard, external, dense, token)}
                outputs = [shard, external, inputs, root]
                if action == 'config':
                    outputs += [pack, dense, token, pack/'new.json']
                for output in outputs:
                    argv = ['maya', action, '--gguf', str(gguf), '--force',
                            '--pack', str(pack if action == 'config' else output)]
                    if action == 'config':
                        argv += ['--exe', '/bin/engine', '--config', str(output)]
                    with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                        main(argv)
                    for path, data in saved.items():
                        self.assertEqual(path.read_bytes(), data)
                # The manifest is an output too, even when only it aliases input.
                output = root/'safe'
                Path(str(output) + '.manifest.json').symlink_to(external)
                argv = ['maya', action, '--gguf', str(gguf), '--force',
                        '--pack', str(pack if action == 'config' else output)]
                if action == 'config':
                    argv += ['--exe', '/bin/engine', '--config', str(output)]
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    main(argv)
                self.assertFalse(output.exists())
                for path, data in saved.items():
                    self.assertEqual(path.read_bytes(), data)

    def test_inherited_ple_is_protected(self):
        for alias in ('direct', 'symlink', 'hardlink'):
            for sidecar in (False, True):
                with self.subTest(alias=alias, sidecar=sidecar), tempfile.TemporaryDirectory() as tmp:
                    root = Path(tmp)
                    (root/'inputs').mkdir()
                    ple = root/'ple.bin'
                    ple.write_bytes(b'PLE weights')
                    pack = root/'pack'
                    pack_manifest = Path(str(pack) + '.manifest.json')
                    pack_manifest.write_text(json.dumps({'options': {'ple_gguf': str(ple)}}))
                    output = root/'run.json'
                    victim = Path(str(output) + '.manifest.json') if sidecar else output
                    if alias == 'direct':
                        ple.rename(victim)
                        ple = victim
                        pack_manifest.write_text(json.dumps({'options': {'ple_gguf': str(ple)}}))
                    elif alias == 'symlink':
                        victim.symlink_to(ple)
                    else:
                        victim.hardlink_to(ple)
                    before = pack_manifest.read_bytes()
                    with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                        main(['strata', 'config', '--gguf', str(root/'inputs/a.gguf'),
                              '--pack', str(pack), '--config', str(output),
                              '--exe', '/bin/engine', '--force'])
                    self.assertEqual(ple.read_bytes(), b'PLE weights')
                    self.assertEqual(pack_manifest.read_bytes(), before)
                    if sidecar:
                        self.assertFalse(output.exists())

    def test_output_aliases_are_rejected(self):
        for alias in ('symlink', 'dangling-symlink', 'hardlink'):
            with self.subTest(alias=alias), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                output = root/'run.json'
                manifest = Path(str(output) + '.manifest.json')
                if alias != 'dangling-symlink':
                    output.write_bytes(b'original config')
                if alias == 'hardlink':
                    manifest.hardlink_to(output)
                else:
                    manifest.symlink_to(output)
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    main(['strata', 'config', '--gguf', str(root/'a.gguf'),
                          '--pack', str(root/'pack'), '--config', str(output),
                          '--exe', '/bin/engine', '--force'])
                if alias == 'dangling-symlink':
                    self.assertFalse(output.exists())
                else:
                    self.assertEqual(output.read_bytes(), b'original config')

    def test_packer(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            (root/'a.gguf').touch()
            with patch.dict('os.environ', {'PREPARE_MAYA_SOURCE': '/source'}), patch('subprocess.run') as run, patch('runtime.prepare_model.importlib.util.spec_from_file_location'), patch('runtime.prepare_model.importlib.util.module_from_spec'):
                main(['maya', 'pack', '--pack', str(root/'pack'), '--gguf', str(root/'a.gguf'), '--experts-bin'])
            self.assertIn('/source/tools/iq_pack.py', run.call_args.args[0])
            self.assertIn('--experts-bin', run.call_args.args[0])

    def test_runtime_flags_and_manifest(self):
        for engine in ADAPTERS:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                pack = root/'pack'
                (root/'a.gguf').touch()
                Path(str(pack) + '.manifest.json').write_text(json.dumps({'options': {'experts_bin': True}}))
                argv = [engine, 'config', '--pack', str(pack), '--gguf', str(root/'a.gguf'),
                        '--exe', '/bin/engine', '--config', str(root/'run.json'),
                        '--vram-reserve-mib', '1024', '--expert-cache', 'auto',
                        '--prefill', '512', '--kv', 'int8']
                with contextlib.redirect_stdout(io.StringIO()):
                    main(argv)
                    cfg = json.loads((root/'run.json').read_text())
                    self.assertIn('--mmap-experts', cfg['args'])
                    self.assertIn('1024', cfg['args'])
                    self.assertNotIn('--adapt-every', cfg['args'])
                    main(argv + ['--force', '--no-mmap-experts'])
                    self.assertNotIn('--mmap-experts', json.loads((root/'run.json').read_text())['args'])
                for output in (pack, Path(str(pack) + '.manifest.json'), root):
                    with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                        main(argv + ['--force', '--config', str(output)])
                for flag, value in (('--vram-reserve-mib', '-1'), ('--prefill', '0'),
                                    ('--pcie-frac', 'nan'), ('--expert-cache', 'bad')):
                    with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                        main(argv + ['--force', flag, value])

    def test_strata_startup_defaults_and_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            mtp = root/'mtp'
            mtp.mkdir()
            argv = ['strata', 'config', '--pack', str(root/'pack'), '--gguf', str(root/'a.gguf'),
                    '--exe', '/bin/engine', '--config', str(root/'run.json')]
            with contextlib.redirect_stdout(io.StringIO()):
                main(argv)
                cfg = json.loads((root/'run.json').read_text())
                self.assertEqual(cfg['args'][-4:], ['--prefill', 'auto', '--spec', '2'])
                self.assertEqual(cfg['args'][cfg['args'].index('--expert-cache') + 1], 'auto')
                self.assertEqual(cfg['args'][cfg['args'].index('--expert-profile') + 1], str(self.profile))
                self.assertEqual(cfg['args'][cfg['args'].index('--max-context') + 1], '4096')
                self.assertEqual(cfg['log'], str(root/'run.json.log'))
                main(argv + ['--force', '--spec', '3', '--mtp', str(mtp)])
                self.assertIn(str(mtp), json.loads((root/'run.json').read_text())['args'])
            for extra in (['--spec', '1'], ['--mtp', str(root/'missing')],
                          ['--mtp', str(mtp), '--config', str(mtp/'run.json')]):
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    main(argv + ['--force'] + extra)
            for victim in (root/'a.gguf', root/'run.json', root/'run.json.manifest.json'):
                log = root/'run.json.log'
                log.symlink_to(victim)
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    main(argv + ['--force'])
                log.unlink()

    def test_profile_resolution_and_collisions(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict('os.environ', {}, clear=True):
            root = Path(tmp)
            output = root/'run.json'
            argv = ['strata', 'config', '--pack', str(root/'pack'), '--gguf', str(root/'a.gguf'),
                    '--exe', '/bin/engine', '--config', str(output)]
            for extra in ([], ['--expert-profile', str(root/'missing')], ['--expert-profile', str(root)]):
                with contextlib.redirect_stderr(io.StringIO()) as errors, self.assertRaises(SystemExit):
                    main(argv + extra)
                self.assertIn('requires an existing --expert-profile', errors.getvalue())
                self.assertFalse(output.exists())
            with contextlib.redirect_stdout(io.StringIO()):
                main(argv + ['--expert-profile', str(self.profile), '--expert-cache', '0'])
            args = json.loads(output.read_text())['args']
            self.assertEqual(args[args.index('--expert-profile') + 1], str(self.profile))
            self.assertEqual(args[args.index('--expert-cache') + 1], '0')
            output.unlink()
            Path(str(output) + '.manifest.json').unlink()
            for suffix in ('', '.manifest.json', '.log'):
                for alias in ('direct', 'symlink', 'hardlink'):
                    victim = Path(str(output) + suffix)
                    profile = self.profile
                    if alias == 'direct':
                        victim.write_bytes(b'mock profile')
                        profile = victim
                    elif alias == 'symlink':
                        victim.symlink_to(profile)
                    else:
                        victim.hardlink_to(profile)
                    with contextlib.redirect_stderr(io.StringIO()) as errors, self.assertRaises(SystemExit):
                        main(argv + ['--force', '--expert-profile', str(profile)])
                    self.assertIn('collides with an input', errors.getvalue())
                    self.assertEqual(victim.read_bytes(), b'mock profile')
                    self.assertEqual(list(root.iterdir()), [victim])
                    victim.unlink()

    def test_maya_layout(self):
        for action in ('pack', 'config'):
            for layout in ('missing', 'wrong', 'direct', 'symlink'):
                with self.subTest(action=action, layout=layout), tempfile.TemporaryDirectory() as tmp:
                    root = Path(tmp)
                    inputs = root/'models'
                    inputs.mkdir()
                    first = inputs/'model-00001-of-00002.gguf'
                    second = inputs/'model-00002-of-00002.gguf'
                    first.touch()
                    if layout != 'missing':
                        second.touch()
                    parent = inputs
                    if layout in ('wrong', 'symlink'):
                        parent = root/'layout'
                        parent.mkdir()
                        (parent/first.name).symlink_to(first)
                        if layout == 'wrong':
                            (parent/second.name).touch()
                        else:
                            (parent/second.name).symlink_to(second)
                    pack = parent/'pack'
                    config = root/'run.json'
                    argv = ['maya', action, '--gguf', str(first), '--pack', str(pack),
                            '--config', str(config), '--exe', '/bin/engine', '--force']
                    with patch.dict('os.environ', {'PREPARE_MAYA_SOURCE': '/source'}), patch('subprocess.run') as run, patch('runtime.prepare_model.importlib.util.spec_from_file_location'), patch('runtime.prepare_model.importlib.util.module_from_spec'), contextlib.redirect_stdout(io.StringIO()):
                        if layout in ('missing', 'wrong'):
                            with contextlib.redirect_stderr(io.StringIO()) as errors, self.assertRaises(SystemExit):
                                main(argv)
                            self.assertIn('every model shard beside the pack', errors.getvalue())
                            run.assert_not_called()
                            self.assertFalse(pack.exists())
                            self.assertFalse(config.exists())
                            self.assertFalse(Path(str(pack) + '.manifest.json').exists())
                            self.assertFalse(Path(str(config) + '.manifest.json').exists())
                        else:
                            main(argv)
                            if action == 'config':
                                self.assertEqual(json.loads(config.read_text())['log'], str(config) + '.log')

    def test_maya_log_collisions(self):
        for kind in ('input', 'config', 'manifest', 'directory', 'pack', 'hardlink'):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                gguf = root/'a.gguf'
                gguf.write_bytes(b'weights')
                config = root/'run.json'
                config.write_bytes(b'original')
                pack = root/'pack'
                pack.mkdir()
                log = root/'run.json.log'
                if kind == 'directory':
                    log.mkdir()
                elif kind == 'hardlink':
                    log.hardlink_to(gguf)
                else:
                    log.symlink_to({'input': gguf, 'config': config,
                                    'manifest': root/'run.json.manifest.json',
                                    'pack': pack/'dense.bin'}[kind])
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    main(['maya', 'config', '--gguf', str(gguf), '--pack', str(pack),
                          '--config', str(config), '--exe', '/bin/engine', '--force'])
                self.assertEqual(config.read_bytes(), b'original')
                self.assertEqual(gguf.read_bytes(), b'weights')

class DeepSeekPreparation(unittest.TestCase):
    def test_pack_and_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = root / 'inputs'
            model.mkdir()
            shards = [model / f'm-0000{i}-of-00002.gguf' for i in (1, 2)]
            for shard in shards:
                shard.touch()
            (model / 'unrelated.gguf').touch()
            pack = root / 'pack'
            config = root / 'run.json'
            common = ['--family', 'deepseek', '--gguf', str(shards[1]), '--pack', str(pack)]
            with patch.dict('os.environ', {'PREPARE_MAYA_SOURCE': tmp}), patch('runtime.prepare_model.subprocess.run') as run:
                main(['maya', 'pack'] + common)
                command = run.call_args.args[0]
                self.assertTrue(command[1].endswith('tools/ds4/pack_gguf.py'))
                self.assertEqual(command[-2:], [p.name for p in shards])
            pack.mkdir()
            with contextlib.redirect_stdout(io.StringIO()):
                main(['maya', 'config'] + common + ['--config', str(config), '--exe', '/bin/ds4'])
            cfg = json.loads(config.read_text())
            self.assertEqual(cfg['cwd'], str(pack))
            self.assertEqual(cfg['args'][-6:], ['--vision', 'off', '--dspark', 'none', '--tp', '1'])
            for extra in (['--kv', 'int8'], ['--no-mmap-experts'], ['--compat-bf16']):
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    main(['maya', 'pack'] + common + extra)
            for output in (model / 'pack', shards[0]):
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    main(['maya', 'pack'] + common + ['--pack', str(output), '--force'])
            (root / 'vision').mkdir()
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                main(['maya', 'pack'] + common + ['--force'])
            (root / 'vision').rmdir()
            shards[0].unlink()
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                main(['maya', 'pack'] + common + ['--force'])

    def test_existing_pack_aliases_rejected(self):
        for kind in ('file-symlink', 'directory-symlink', 'dangling', 'hardlink', 'pack-symlink'):
            for force in (False, True):
                with self.subTest(kind=kind, force=force), tempfile.TemporaryDirectory() as tmp:
                    root = Path(tmp)
                    model = root/'inputs'
                    model.mkdir()
                    shard = model/'model.gguf'
                    shard.write_bytes(b'original weights')
                    pack = root/'pack'
                    if kind == 'pack-symlink':
                        pack.symlink_to(model, target_is_directory=True)
                    else:
                        nested = pack/'nested'
                        nested.mkdir(parents=True)
                        entry = nested/'alias'
                        if kind == 'hardlink':
                            entry.hardlink_to(shard)
                        else:
                            entry.symlink_to(model if kind == 'directory-symlink' else
                                             root/'missing' if kind == 'dangling' else shard)
                    with patch('runtime.prepare_model.subprocess.run') as run:
                        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                            main(['maya', 'pack', '--family', 'deepseek', '--gguf', str(shard),
                                  '--pack', str(pack)] + (['--force'] if force else []))
                        run.assert_not_called()
                    self.assertEqual(shard.read_bytes(), b'original weights')

    def test_backend_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shard = root/'model.gguf'
            shard.touch()
            config = root/'run.json'
            for backend in ('hip', 'cuda'):
                with patch.dict('os.environ', {'PREPARE_BACKEND': backend}), contextlib.redirect_stdout(io.StringIO()):
                    main(['maya', 'config', '--gguf', str(shard), '--pack', str(root/'pack'),
                          '--config', str(config), '--exe', '/bin/engine', '--force'])
                self.assertEqual(json.loads(config.read_text())['backend'], backend)
