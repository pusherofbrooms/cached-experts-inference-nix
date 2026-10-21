import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('update', Path(__file__).resolve().parents[1] / 'scripts/update_engine.py')
update = importlib.util.module_from_spec(spec)
spec.loader.exec_module(update)


class UpdateTests(unittest.TestCase):
    def test_annotated_tag(self):
        with patch.object(update, 'run', return_value='a' * 40 + '\trefs/tags/v1\n' + 'b' * 40 + '\trefs/tags/v1^{}'):
            self.assertEqual(update.resolve('owner/repo', 'v1'), 'b' * 40)

    def test_update_and_failure_are_transactional(self):
        for dry, fail in [(True, False), (False, True), (False, False)]:
            with self.subTest(dry=dry, fail=fail), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                (root / 'engines').mkdir()
                (root / 'engines/metadata.json').write_text(json.dumps({'engines': {'maya': {'repo': 'a/b', 'version': '1.0'}}}))
                (root / 'flake.nix').write_text('maya = { url = "github:a/b/old"; };')
                lock = {'nodes': {'root': {'inputs': {'maya': 'maya'}}, 'maya': {'locked': {'rev': 'old'}}}}
                (root / 'flake.lock').write_text(json.dumps(lock))
                original = (root / 'flake.nix').read_text()

                def run(*args, cwd=None):
                    if args[0] == 'git':
                        return ''
                    if fail:
                        raise ValueError('lock failure')
                    lock['nodes']['maya']['locked']['rev'] = 'a' * 40
                    (cwd / 'flake.lock').write_text(json.dumps(lock))
                    return ''

                with patch.object(update, 'run', side_effect=run), patch.object(update, 'fetch', side_effect=[json.dumps({'tag_name': 'v2'}), 'project(Maya VERSION 2.0)']), patch.object(update, 'resolve', return_value='a' * 40):
                    if fail:
                        with self.assertRaises(ValueError):
                            update.update(root, 'maya', dry_run=dry)
                    else:
                        update.update(root, 'maya', dry_run=dry)
                self.assertEqual((root / 'flake.nix').read_text() == original, dry or fail)

    def test_dirty_refused(self):
        with patch.object(update, 'run', return_value=' M flake.lock'):
            with self.assertRaisesRegex(ValueError, 'dirty'):
                update.update(Path('/unused'), 'maya')
