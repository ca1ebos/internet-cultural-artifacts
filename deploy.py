import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
from tempfile import TemporaryDirectory
import time
from urllib.request import urlopen

from common import ARTIFACTS, CONFIG, ROOT, read_rows, sha256, write_json
from train import telemetry

LLAMA = ARTIFACTS / 'llama'
LLAMA_TAG = 'b12728'
TYPES = ['IQ4_XS', 'Q4_K_M']
PERSONALITIES = [name for name in CONFIG if name not in {'base', 'seed'}]


def executable(bin_dir, name):
    direct = bin_dir / (name + '.exe')
    if direct.is_file():
        return direct
    matches = list(bin_dir.rglob(name + '.exe'))
    if len(matches) != 1:
        raise FileNotFoundError(f'Provide the llama.cpp binary directory containing {name}.exe')
    return matches[0]


def merge(source, target):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    base = ARTIFACTS / 'base'
    adapter = ARTIFACTS / 'runs' / source / 'full' / 'adapter'
    adapter_manifest = adapter.parent / 'adapter.manifest.json'
    identity = {'base_manifest_sha256': sha256(base / 'manifest.json'),
                'adapter_manifest_sha256': sha256(adapter_manifest), 'dtype': 'bfloat16'}
    marker = target / 'manifest.json'
    if marker.exists():
        previous = json.loads(marker.read_text(encoding='utf-8'))
        if previous['inputs'] != identity:
            raise RuntimeError(f'Merged checkpoint already exists with different inputs at {target}')
        return target
    model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16,
                                                low_cpu_mem_usage=True, local_files_only=True)
    model = PeftModel.from_pretrained(model, adapter, local_files_only=True)
    merged = model.merge_and_unload(safe_merge=True)
    merged.save_pretrained(target, safe_serialization=True, max_shard_size='4GB')
    AutoTokenizer.from_pretrained(base, local_files_only=True).save_pretrained(target)
    write_json(marker, {'inputs': identity, 'files': {file.name: sha256(file) for file in sorted(target.iterdir())}})
    return target


def export(source, bin_dir, cpp_dir, kinds, imatrix, transient=False):
    if source == 'base':
        quantize(source, ARTIFACTS / 'base', bin_dir, cpp_dir, kinds, imatrix)
        return
    if not transient:
        model_dir = merge(source, ARTIFACTS / 'merged' / source)
        quantize(source, model_dir, bin_dir, cpp_dir, kinds, imatrix)
        return
    scratch = (ARTIFACTS / 'scratch').resolve()
    if not scratch.is_relative_to(ARTIFACTS.resolve()):
        raise ValueError('Scratch directory must be inside the workspace artifacts directory')
    scratch.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=scratch, prefix=f'{source}-merge-') as temp:
        model_dir = merge(source, Path(temp))
        quantize(source, model_dir, bin_dir, cpp_dir, kinds, imatrix)
        write_json(ARTIFACTS / 'gguf' / f'{source}-transient-merge.manifest.json',
                   json.loads((model_dir / 'manifest.json').read_text(encoding='utf-8')))


def quantize(source, model_dir, bin_dir, cpp_dir, kinds, imatrix):
    name = 'Qwen4B-Base' if source == 'base' else f'Qwen4B-{CONFIG[source]["name"]}'
    output = ARTIFACTS / 'gguf'
    output.mkdir(parents=True, exist_ok=True)
    converter = cpp_dir / 'convert_hf_to_gguf.py'
    manifest_path = output / f'{name}.manifest.json'
    identity = {'source': source, 'model_manifest_sha256': sha256(model_dir / 'manifest.json'),
                'converter_sha256': sha256(converter), 'llama_tag': LLAMA_TAG,
                'imatrix_sha256': sha256(imatrix) if imatrix else None}
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding='utf-8'))
        if any(previous.get(key) != value for key, value in identity.items()):
            raise RuntimeError(f'Export inputs changed for {name}')
    intermediate = output / f'{name}-F16.gguf.part'
    intermediate_complete = output / f'{name}-F16.gguf'
    missing = [kind for kind in kinds if not (output / f'{name}-{kind}.gguf').exists()]
    if missing and not intermediate_complete.exists():
        env = dict(os.environ, PYTHONPATH=str(cpp_dir / 'gguf-py'))
        subprocess.run([sys.executable, str(converter), str(model_dir), '--outtype', 'f16',
                        '--outfile', str(intermediate)], env=env, check=True)
        intermediate.replace(intermediate_complete)
    for kind in missing:
        path = output / f'{name}-{kind}.gguf'
        pending = path.with_suffix('.gguf.part')
        args = ['--imatrix', str(imatrix)] if imatrix else []
        subprocess.run([str(executable(bin_dir, 'llama-quantize')), *args, str(intermediate_complete),
                        str(pending), kind, '6'], check=True)
        pending.replace(path)
    artifacts = [output / f'{name}-{kind}.gguf' for kind in TYPES]
    manifest = {**identity, 'files': {file.name: {'sha256': sha256(file), 'bytes': file.stat().st_size}
                                     for file in artifacts if file.exists()}}
    write_json(manifest_path, manifest)
    intermediate_complete.unlink(missing_ok=True)


def benchmark(bin_dir, model, context, output, condition, tag, cases):
    output.mkdir(parents=True, exist_ok=True)
    path = output / f'{tag}-ctx{context}'
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    url = f'http://127.0.0.1:{port}'
    with path.with_suffix('.server.log').open('w', encoding='utf-8') as logfile, telemetry(path.with_suffix('.csv')):
        args = [str(executable(bin_dir, 'llama-server')), '-m', str(model), '--host', '127.0.0.1',
                '--port', str(port), '-c', str(context), '-ngl', '99', '--parallel', '1', '--jinja']
        process = subprocess.Popen(args, stdout=logfile, stderr=subprocess.STDOUT)
        try:
            for _ in range(90):
                if process.poll() is not None:
                    raise RuntimeError(f'llama-server exited. Check {logfile.name}')
                try:
                    with urlopen(url + '/health', timeout=2) as response:
                        if response.status == 200:
                            break
                except OSError:
                    time.sleep(1)
            else:
                raise TimeoutError(f'llama-server failed to become ready. Check {logfile.name}')
            command = [sys.executable, str(ROOT / 'evaluate.py'), '--server', url, '--tag', path.name,
                       '--conditions', condition, '--output', str(output), '--artifact-sha', sha256(model)]
            viable = [case['id'] for case in read_rows(ROOT / 'prompts.jsonl')
                      if (not cases or case['id'] in cases) and case.get('context_words', 0) * 1.5 + 384 < context]
            command.extend(['--cases', *viable])
            subprocess.run(command, check=True)
        finally:
            process.terminate()
            process.wait(timeout=10)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bin', type=Path, default=LLAMA / 'bin')
    sub = parser.add_subparsers(dest='action', required=True)
    convert = sub.add_parser('export')
    convert.add_argument('source', choices=['base', *PERSONALITIES])
    convert.add_argument('--cpp', type=Path, default=LLAMA / f'llama.cpp-{LLAMA_TAG}')
    convert.add_argument('--types', nargs='+', choices=TYPES, default=TYPES)
    convert.add_argument('--imatrix', type=Path)
    convert.add_argument('--transient-merge', action='store_true')
    bench = sub.add_parser('benchmark')
    bench.add_argument('model', type=Path)
    bench.add_argument('tag')
    bench.add_argument('--condition', choices=['base', 'stack-prompt', 'yahoo-prompt',
                                              *[f'{source}-adapter' for source in PERSONALITIES]],
                       default='base')
    bench.add_argument('--contexts', nargs='+', type=int, default=[2048, 8192])
    bench.add_argument('--cases', nargs='+')
    bench.add_argument('--output', type=Path, default=ARTIFACTS / 'deployment')
    args = parser.parse_args()
    if args.action == 'export':
        export(args.source, args.bin, args.cpp, args.types, args.imatrix, args.transient_merge)
    else:
        for value in args.contexts:
            benchmark(args.bin, args.model, value, args.output, args.condition, args.tag, args.cases)
