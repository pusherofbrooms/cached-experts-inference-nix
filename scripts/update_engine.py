"""Update only one literal input and its package version, then lock in staging."""
import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import urllib.request


def run(*args, cwd=None):
    return subprocess.check_output(args, cwd=cwd, text=True).strip()


def fetch(url):
    headers = {'Accept': 'application/vnd.github+json', 'User-Agent': 'update-engine'}
    if os.environ.get('GITHUB_TOKEN'):
        headers['Authorization'] = 'Bearer ' + os.environ['GITHUB_TOKEN']
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers)) as response:
        return response.read().decode()


def resolve(repo, ref):
    if re.fullmatch(r'[0-9a-fA-F]{40}', ref):
        return ref.lower()
    if (not ref or ref.startswith(('-', '/')) or ref.endswith(('/', '.'))
            or any(ord(c) < 33 or ord(c) == 127 or c in '~^:?*[\\' for c in ref)
            or '..' in ref or '@{' in ref or '//' in ref
            or any(part.startswith('.') or part.endswith('.lock') for part in ref.split('/'))):
        raise ValueError('expected an exact tag or full commit SHA')
    tag = 'refs/tags/' + ref
    lines = run('git', 'ls-remote', '--tags', 'https://github.com/' + repo + '.git', tag, tag + '^{}')
    refs = dict(line.split()[::-1] for line in lines.splitlines())
    revision = refs.get(tag + '^{}', refs.get(tag))
    if not revision or not re.fullmatch('[0-9a-f]{40}', revision):
        raise ValueError('tag not found: ' + ref)
    return revision


def cmake_version(repo, rev):
    cmake = fetch(f'https://raw.githubusercontent.com/{repo}/{rev}/CMakeLists.txt')
    versions = re.findall(r'project\s*\([^)]*?\bVERSION\s+([0-9]+(?:\.[0-9]+)+)', cmake, re.I)
    if len(versions) != 1:
        raise ValueError('cannot determine unique CMake project version; review upstream manually')
    return versions[0]


# Existing engines use CMake; other source layouts require an explicit reader.
VERSION_READERS = {'cmake': cmake_version}


def update(root, engine, ref=None, dry_run=False):
    paths = ['flake.nix', 'flake.lock', 'engines/metadata.json']
    if run('git', 'status', '--porcelain', '--', *paths, cwd=root):
        raise ValueError('refusing to overwrite dirty update files (including staged changes)')
    original = {p: (root / p).read_bytes() for p in paths}
    metadata = json.loads(original[paths[2]])
    if engine not in metadata['engines']:
        raise ValueError('unknown engine: ' + engine)
    entry = metadata['engines'][engine]
    repo = entry['repo']
    strategy = entry.get('versionStrategy', 'cmake')
    if strategy not in VERSION_READERS:
        raise ValueError('unsupported version strategy: ' + str(strategy))
    if ref is None:
        release = json.loads(fetch('https://api.github.com/repos/' + repo + '/releases/latest'))
        if release.get('draft') or release.get('prerelease'):
            raise ValueError('latest release is not stable/published')
        ref = release['tag_name']
    rev = resolve(repo, ref)
    version = VERSION_READERS[strategy](repo, rev)
    entry['version'] = version
    flake = original['flake.nix'].decode()
    pattern = r'(\b' + re.escape(engine) + r'\s*=\s*\{\s*url\s*=\s*")github:' + re.escape(repo) + r'/[^"\s]+(";)'
    flake, count = re.subn(pattern, lambda m: m[1] + 'github:' + repo + '/' + rev + m[2], flake)
    if count != 1:
        raise ValueError('expected exactly one literal engine input URL')
    print(f'{engine}: {ref} -> {rev}, version {version}')
    with tempfile.TemporaryDirectory(prefix='update-engine-') as directory:
        stage = Path(directory) / 'repo'
        shutil.copytree(root, stage, ignore=shutil.ignore_patterns('.git', 'result', '__pycache__'))
        (stage / 'flake.nix').write_text(flake)
        (stage / paths[2]).write_text(json.dumps(metadata, indent=2) + '\n')
        run('nix', 'flake', 'update', engine, cwd=stage)
        before = json.loads(original['flake.lock'])
        after = json.loads((stage / 'flake.lock').read_text())
        node = after['nodes']['root']['inputs'][engine]
        if after['nodes'][node]['locked']['rev'] != rev:
            raise ValueError('lock revision does not match resolved commit')
        old_node = before['nodes']['root']['inputs'][engine]
        before['nodes'].pop(old_node)
        after['nodes'].pop(node)
        if before != after:
            raise ValueError('lock changed unrelated inputs')
        if dry_run:
            print('Dry run: lock validated; no files written')
            return
        if any((root / p).read_bytes() != original[p] for p in paths):
            raise ValueError('files changed during update; refusing to overwrite')
        try:
            for p in paths:
                (root / p).write_bytes((stage / p).read_bytes())
        except BaseException:
            for p in paths:
                (root / p).write_bytes(original[p])
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('engine', help='engine name from engines/metadata.json')
    parser.add_argument('--ref', help='exact tag or full commit; default: latest stable GitHub release')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    try:
        root = Path(run('git', 'rev-parse', '--show-toplevel')).resolve()
        update(root, args.engine, args.ref, args.dry_run)
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        parser.exit(1, f'update-engine: {error}\n')


if __name__ == '__main__':
    main()
