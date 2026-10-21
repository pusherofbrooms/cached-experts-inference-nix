import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from runtime.prepare_model import main, ADAPTERS


class Preparation(unittest.TestCase):
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

    def test_configs(self):
        for engine in ADAPTERS:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp).resolve()
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
