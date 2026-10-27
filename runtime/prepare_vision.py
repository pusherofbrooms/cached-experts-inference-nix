"""Local-only DeepSeek checkpoint preparation, in a fresh output directory."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import shutil
import tempfile


def absolute(value):
    return Path(os.path.abspath(os.path.expanduser(value)))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('shard', 'config', 'marks', 'out'):
        p.add_argument('--' + name, required=True, type=absolute)
    a = p.parse_args(argv)
    inputs = [a.shard, a.config, a.marks]
    if not all(path.is_file() for path in inputs):
        p.error('all inputs must be existing local files; marks is checkpoint shard 2, not a URL')
    manifest = Path(str(a.out) + '.manifest.json')
    for out in (a.out, manifest):
        if out.exists() or out.is_symlink():
            p.error('output paths must be fresh (no force replacement)')
        for src in inputs:
            for dest in (out, out.resolve()):
                for origin in (src, src.resolve()):
                    if dest == origin or dest in origin.parents or origin.parent in dest.parents:
                        p.error('output collides with an input or input directory')
    source = Path(os.environ['PREPARE_MAYA_SOURCE'])
    spec = importlib.util.spec_from_file_location('ds4_vision_pack', source / 'tools/ds4/vision_pack.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    man = module.manifest_entry()
    names = [name for name in man['files'] if name.endswith('.safetensors')]
    if len(names) != 1 or Path(names[0]).name != names[0]:
        p.error('unsupported upstream vision manifest')
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=a.out.parent) as tmp:
        stage = Path(tmp)
        (stage / names[0]).symlink_to(a.shard)
        (stage / 'config.json').symlink_to(a.config)
        # Mandatory absolute local path: never upstream's implicit HTTP fallback.
        module.pack(stage, man, str(a.marks))
        a.out.mkdir()
        for name in ('vision.bin', 'vision.json'):
            shutil.move(str(stage / name), a.out / name)
    manifest.write_text(json.dumps({'schema': 1, 'engine_revision': os.environ.get('PREPARE_MAYA_REV', 'unknown'),
        'inputs': {key: str(value) for key, value in vars(a).items()},
        'vision_source': man}, indent=2) + '\n')


if __name__ == '__main__':
    main()
