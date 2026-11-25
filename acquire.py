import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import date
import json
import subprocess
from urllib.parse import quote
from urllib.request import urlopen

from common import ARTIFACTS, CONFIG, SOURCES, sha256, write_json

HUB = 'https://huggingface.co'


def tree(repo, revision, folder):
    subpath = f'/{folder}' if folder else ''
    with urlopen(f'{HUB}/api/datasets/{repo}/tree/{revision}{subpath}?limit=1000', timeout=30) as response:
        return json.load(response)


def download(url, target, expected_size=None, expected_digest=None, hub=True):
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        partial = target.with_suffix(target.suffix + '.part')
        subprocess.run(['curl.exe', '--fail', '--silent', '--show-error', '--location',
                        '--retry', '3', '--continue-at', '-', '--output', str(partial),
                        url + '?download=true' if hub else url], check=True)
        if expected_size and partial.stat().st_size != expected_size:
            raise ValueError(f'Size mismatch for {partial}')
        if expected_digest and sha256(partial) != expected_digest:
            raise ValueError(f'Checksum mismatch for {partial}')
        partial.replace(target)
    digest = sha256(target)
    if expected_digest and digest != expected_digest:
        raise ValueError(f'Checksum mismatch for {target}')
    return {'url': url, 'path': target.relative_to(ARTIFACTS).as_posix(),
            'bytes': target.stat().st_size, 'sha256': digest}


def main(source):
    if source == 'reddit':
        metadata = SOURCES[source]
        destination = ARTIFACTS / 'raw' / source
        manifest = destination / 'manifest.json'
        acquired = download(metadata['direct'], destination / metadata['file'], hub=False)
        with (ARTIFACTS / acquired['path']).open('rb') as file:
            if file.read(6) != b'7z\xbc\xaf\x27\x1c':
                raise ValueError('The public Reddit archive did not return a 7-Zip file')
        acquired_on = (json.loads(manifest.read_text(encoding='utf-8'))['acquired_on']
                       if manifest.exists() else date.today().isoformat())
        write_json(manifest, {**metadata, 'acquired_on': acquired_on, 'files': [acquired]})
        return
    if source == 'base':
        metadata = CONFIG['base']
        repo, revision = metadata['repo'], metadata['revision']
        with urlopen(f'{HUB}/api/models/{repo}/tree/{revision}', timeout=30) as response:
            files = [entry for entry in json.load(response) if not entry['path'].startswith('.')]
        prefix = repo
        destination = ARTIFACTS / 'base'
    else:
        metadata = SOURCES[source]
        repo, revision = metadata['repo'], metadata['revision']
        if source == 'stack':
            files = []
            for group in metadata['groups']:
                shards = sorted((entry for entry in tree(repo, revision, f'data/{group}')
                                 if entry['path'].endswith('.parquet')), key=lambda entry: entry['path'])
                files.append(shards[len(shards) // 2])
        elif source == 'yahoo':
            files = [entry for entry in tree(repo, revision, 'data')
                     if entry['path'].split('/')[-1].startswith(('train-00000', 'test-00000'))]
        else:
            entries = {entry['path']: entry for entry in tree(repo, revision, '')}
            files = [entries[path] for path in metadata['files']]
        prefix = f'datasets/{repo}'
        destination = ARTIFACTS / 'raw' / source

    files = [(prefix, revision, entry) for entry in files]
    for supplement in metadata.get('supplements', []):
        extra_repo, extra_revision = supplement['repo'], supplement['revision']
        entries = {entry['path']: entry for entry in tree(extra_repo, extra_revision, supplement.get('folder', ''))}
        files.extend((f'datasets/{extra_repo}', extra_revision, entries[path]) for path in supplement['files'])

    def fetch(item):
        item_prefix, item_revision, entry = item
        path = entry['path']
        print(f'{source} {path} ({entry["size"]:,} bytes)', flush=True)
        return download(f'{HUB}/{item_prefix}/resolve/{item_revision}/{quote(path)}', destination / path,
                        entry['size'], entry.get('lfs', {}).get('oid'))

    with ThreadPoolExecutor(max_workers=3 if source != 'base' else 1) as pool:
        acquired = list(pool.map(fetch, files))
    for external in metadata.get('external_files', []):
        acquired.append(download(external['url'], destination / external['name'], external['bytes'], hub=False))
    manifest = destination / 'manifest.json'
    acquired_on = (json.loads(manifest.read_text(encoding='utf-8'))['acquired_on']
                   if manifest.exists() else date.today().isoformat())
    write_json(manifest, {**metadata, 'acquired_on': acquired_on, 'files': acquired})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('source', choices=['base', *SOURCES])
    main(parser.parse_args().source)
