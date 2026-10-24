import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('update', Path(__file__).resolve().parents[1] / 'scripts/update_engine.py')
update = importlib.util.module_from_spec(spec)
spec.loader.exec_module(update)

REV = 'a' * 40
PATHS = ('flake.nix', 'flake.lock', 'engines/metadata.json')


class ResolveTests(unittest.TestCase):
    def test_tags(self):
        for annotated in (False, True):
            with self.subTest(annotated=annotated):
                output = REV + '\trefs/tags/v1'
                if annotated:
                    output += '\n' + 'b' * 40 + '\trefs/tags/v1^{}'
                with patch.object(update, 'run', return_value=output) as run:
                    self.assertEqual(update.resolve('owner/repo', 'v1'), 'b' * 40 if annotated else REV)
                    run.assert_called_once_with('git', 'ls-remote', '--tags', 'https://github.com/owner/repo.git', 'refs/tags/v1', 'refs/tags/v1^{}')

    def test_full_commit(self):
        with patch.object(update, 'run') as run:
            self.assertEqual(update.resolve('a/b', REV.upper()), REV)
            run.assert_not_called()

    def test_malformed_refs(self):
        for ref in ('', '-v1', 'v*', 'v?', 'v[1]', 'v\n1', 'v 1', 'v\t1', '/v1', 'v1/', 'a..b', 'a//b', '.tag', 'a/.b', 'a.lock', 'a@{b', 'a^b', 'a:b', 'a\\b', 'a~b', 'a.'):
            with self.subTest(ref=ref), patch.object(update, 'run') as run:
                with self.assertRaisesRegex(ValueError, 'exact tag'):
                    update.resolve('a/b', ref)
                run.assert_not_called()

    def test_missing_tag(self):
        with patch.object(update, 'run', return_value=''):
            with self.assertRaisesRegex(ValueError, 'tag not found'):
                update.resolve('a/b', 'missing')

    def test_cmake_versions(self):
        for source, expected in [('project(X VERSION 1.2.3)', '1.2.3'), ('project(X)', None), ('project(X VERSION 1.0)\nproject(Y VERSION 2.0)', None)]:
            with self.subTest(source=source), patch.object(update, 'fetch', return_value=source):
                if expected:
                    self.assertEqual(update.cmake_version('a/b', REV), expected)
                else:
                    with self.assertRaisesRegex(ValueError, 'unique CMake'):
                        update.cmake_version('a/b', REV)


class UpdateTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        (self.root / 'engines').mkdir()
        self.metadata = {'engines': {'maya': {'repo': 'a/b', 'version': '1.0'}}, 'dependency': {'rev': 'keep'}}
        self.lock = {'nodes': {'root': {'inputs': {'maya': 'maya', 'other': 'other'}}, 'maya': {'locked': {'rev': 'old'}}, 'other': {'locked': {'rev': 'keep'}}}}
        (self.root / 'engines/metadata.json').write_text(json.dumps(self.metadata))
        (self.root / 'flake.nix').write_text('maya = { url = "github:a/b/old"; };')
        (self.root / 'flake.lock').write_text(json.dumps(self.lock))
        self.original = self.snapshot()
        self.lock_effect = lambda stage: None
        self.run_mock = patch.object(update, 'run', side_effect=self.run_command).start()
        patch.object(update, 'fetch', side_effect=[json.dumps({'tag_name': 'v2'}), 'project(Maya VERSION 2.0)']).start()
        patch.object(update, 'resolve', return_value=REV).start()
        self.addCleanup(patch.stopall)

    def snapshot(self):
        return {p: (self.root / p).read_bytes() for p in PATHS}

    def run_command(self, *args, cwd=None):
        if args[0] == 'git':
            return ''
        self.assertEqual(args, ('nix', 'flake', 'update', 'maya'))
        self.assertNotEqual(cwd, self.root)
        lock = copy.deepcopy(self.lock)
        lock['nodes']['maya']['locked']['rev'] = REV
        (cwd / 'flake.lock').write_text(json.dumps(lock))
        self.lock_effect(cwd)
        return ''

    def test_success(self):
        self.lock_effect = lambda stage: None
        update.update(self.root, 'maya')
        self.assertIn(REV, (self.root / 'flake.nix').read_text())
        metadata = json.loads((self.root / PATHS[2]).read_text())
        self.assertEqual(metadata['engines']['maya']['version'], '2.0')
        self.assertEqual(metadata['dependency'], self.metadata['dependency'])
        self.assertEqual(json.loads((self.root / 'flake.lock').read_text())['nodes']['maya']['locked']['rev'], REV)

    def test_dry_run_all_files_unchanged(self):
        self.lock_effect = lambda stage: None
        update.update(self.root, 'maya', dry_run=True)
        self.assertEqual(self.snapshot(), self.original)

    def test_dirty_and_staged(self):
        for status in (' M flake.lock', 'M  flake.nix', 'MM engines/metadata.json'):
            with self.subTest(status=status):
                self.run_mock.side_effect = None
                self.run_mock.return_value = status
                with self.assertRaisesRegex(ValueError, 'dirty'):
                    update.update(self.root, 'maya')
                self.assertEqual(self.snapshot(), self.original)

    def test_literal_url_mismatch(self):
        for url in ('github:wrong/repo/old', '${source}', 'github:a/b.extra/old'):
            with self.subTest(url=url):
                (self.root / 'flake.nix').write_text(f'maya = {{ url = "{url}"; }};')
                before = self.snapshot()
                with patch.object(update, 'fetch', return_value='project(X VERSION 2.0)'):
                    with self.assertRaisesRegex(ValueError, 'literal engine'):
                        update.update(self.root, 'maya', ref=REV)
                self.assertEqual(self.snapshot(), before)

    def test_lock_validation_failures(self):
        for target, rev, message in [('other', 'changed', 'unrelated'), ('maya', 'wrong', 'resolved commit')]:
            def change(stage):
                lock = json.loads((stage / 'flake.lock').read_text())
                lock['nodes'][target]['locked']['rev'] = rev
                (stage / 'flake.lock').write_text(json.dumps(lock))
            self.lock_effect = change
            with self.subTest(target=target), patch.object(update, 'fetch', return_value='project(X VERSION 2.0)'):
                with self.assertRaisesRegex(ValueError, message):
                    update.update(self.root, 'maya', ref=REV)
                self.assertEqual(self.snapshot(), self.original)

    def test_lock_command_failure(self):
        def fail(stage):
            raise ValueError('lock failure')
        self.lock_effect = fail
        with self.assertRaisesRegex(ValueError, 'lock failure'):
            update.update(self.root, 'maya')
        self.assertEqual(self.snapshot(), self.original)

    def test_concurrent_changes_preserved(self):
        def change(stage):
            (self.root / 'flake.lock').write_text('concurrent edit')
        self.lock_effect = change
        with self.assertRaisesRegex(ValueError, 'changed during update'):
            update.update(self.root, 'maya')
        expected = dict(self.original, **{'flake.lock': b'concurrent edit'})
        self.assertEqual(self.snapshot(), expected)

    def test_write_failure_rolls_back(self):
        self.lock_effect = lambda stage: None
        write = Path.write_bytes
        failed = False

        def fail_once(path, data):
            nonlocal failed
            if path == self.root / 'flake.lock' and not failed:
                failed = True
                raise OSError('write failed')
            return write(path, data)

        with patch.object(Path, 'write_bytes', fail_once):
            with self.assertRaisesRegex(OSError, 'write failed'):
                update.update(self.root, 'maya')
        self.assertTrue(failed)
        self.assertEqual(self.snapshot(), self.original)

    def test_metadata_registered_engine_and_version_reader(self):
        self.metadata['engines']['future'] = {'repo': 'c/d', 'version': 'old', 'versionStrategy': 'custom'}
        (self.root / PATHS[2]).write_text(json.dumps(self.metadata))
        (self.root / 'flake.nix').write_text('future = { url = "github:c/d/old"; };')
        lock = {'nodes': {'root': {'inputs': {'future': 'future'}}, 'future': {'locked': {'rev': 'old'}}}}
        (self.root / 'flake.lock').write_text(json.dumps(lock))

        def run(*args, cwd=None):
            if args[0] == 'git':
                return ''
            self.assertEqual(args, ('nix', 'flake', 'update', 'future'))
            lock['nodes']['future']['locked']['rev'] = REV
            (cwd / 'flake.lock').write_text(json.dumps(lock))
            return ''

        with patch.object(update, 'run', side_effect=run), patch.dict(update.VERSION_READERS, custom=lambda repo, rev: '3.0'):
            update.update(self.root, 'future', ref=REV)
        metadata = json.loads((self.root / PATHS[2]).read_text())
        self.assertEqual(metadata['engines']['future']['version'], '3.0')
        self.assertEqual(metadata['engines']['maya'], self.metadata['engines']['maya'])

    def test_unpublished_releases_refused(self):
        for field in ('draft', 'prerelease'):
            with self.subTest(field=field), patch.object(update, 'fetch', return_value=json.dumps({'tag_name': 'v2', field: True})):
                with self.assertRaisesRegex(ValueError, 'stable/published'):
                    update.update(self.root, 'maya')
                self.assertEqual(self.snapshot(), self.original)

    def test_unknown_engine_and_strategy(self):
        with self.assertRaisesRegex(ValueError, 'unknown engine'):
            update.update(self.root, 'unknown')
        self.metadata['engines']['maya']['versionStrategy'] = 'unknown'
        (self.root / PATHS[2]).write_text(json.dumps(self.metadata))
        before = self.snapshot()
        with self.assertRaisesRegex(ValueError, 'unsupported version strategy'):
            update.update(self.root, 'maya')
        self.assertEqual(self.snapshot(), before)
