"""Prepare HTTP configs for a local native Inkling 2 Mini pack."""
import argparse
import json
from pathlib import Path
import sys


def main(exe, tokenizer_tool, backend):
    sys.path.insert(0, str(Path(tokenizer_tool).parent))
    from inkling_tokenizer import convert_manifest, read_manifest

    parser = argparse.ArgumentParser(prog='prepare-inkling', description=__doc__)
    parser.add_argument('model', type=Path, help='local Inkling .strata pack')
    parser.add_argument('output', type=Path, help='new configuration directory')
    parser.add_argument('--tokenizer-json', type=Path, help='fallback if not embedded in the pack')
    args = parser.parse_args(sys.argv[4:])
    try:
        model = args.model.resolve(strict=True)
        if not model.is_file() or model.suffix != '.strata':
            parser.error('expected an Inkling .strata file')
        manifest = read_manifest(model)
        output = args.output.resolve()
        # Use the original path to ensure mkdir rejects even a dangling symlink.
        args.output.mkdir()
        tokenizer = output / 'tokenizer'
        convert_manifest(manifest, tokenizer, tokenizer_json_path=args.tokenizer_json)
        for draft in ('off', 'dspark'):
            config = {
                'arch': 'inkling2_loop', 'backend': backend, 'exe': exe,
                'tokenizer': str(tokenizer), 'model_name': model.stem,
                'host': '127.0.0.1', 'open_browser': False,
                'args': ['--native', str(model), '--parallel', '1', '--draft', draft],
            }
            path = output / f'{draft}.json'
            with path.open('x') as f:
                json.dump(config, f, indent=2)
                f.write('\n')
            print(path)
    except (OSError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    main(*sys.argv[1:4])
