import argparse
from contextlib import contextmanager
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


def merge_inputs(base, adapter):
    return {'base_manifest_sha256': sha256(base / 'manifest.json'),
            'adapter_manifest_sha256': sha256(adapter.parent / 'adapter.manifest.json'), 'dtype': 'bfloat16'}


def merged_model(base, adapter):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16, low_cpu_mem_usage=True,
                                                local_files_only=True)
    return PeftModel.from_pretrained(model, adapter, local_files_only=True).merge_and_unload(safe_merge=True)


def merge(base, adapter, target):
    from transformers import AutoTokenizer

    identity = merge_inputs(base, adapter)
    marker = target / 'manifest.json'
    if marker.exists():
        previous = json.loads(marker.read_text(encoding='utf-8'))
        if previous['inputs'] != identity:
            raise RuntimeError(f'Merged checkpoint already exists with different inputs at {target}')
        return target
    merged = merged_model(base, adapter)
    merged.save_pretrained(target, safe_serialization=True, max_shard_size='4GB')
    AutoTokenizer.from_pretrained(base, local_files_only=True).save_pretrained(target)
    write_json(marker, {'inputs': identity, 'files': {file.name: sha256(file) for file in sorted(target.iterdir())}})
    return target


def convert_checkpoint(model_dir, cpp_dir, path):
    env = dict(os.environ, PYTHONPATH=str(cpp_dir / 'gguf-py'))
    subprocess.run([sys.executable, str(cpp_dir / 'convert_hf_to_gguf.py'), str(model_dir), '--outtype', 'f16',
                    '--outfile', str(path)], env=env, check=True)


def convert_transient(base, adapter, model_dir, cpp_dir, path):
    import torch
    from transformers import AutoTokenizer

    sys.path[:0] = [str(cpp_dir), str(cpp_dir / 'gguf-py')]
    try:
        import gguf
        from conversion import LazyTorchTensor, get_model_class

        # Reuse the pinned architecture conversion while supplying the same merged BF16 tensors in memory.
        with torch.inference_mode():
            merged = merged_model(base, adapter)
            merged.config.save_pretrained(model_dir)
            merged.generation_config.save_pretrained(model_dir)
            AutoTokenizer.from_pretrained(base, local_files_only=True).save_pretrained(model_dir)
            weights = merged.state_dict()
            weight_map = json.loads((base / 'model.safetensors.index.json').read_text(encoding='utf-8'))['weight_map']
            # The only omitted tensor in the pinned Qwen3 checkpoint is the tied output embedding.
            if set(weights) - set(weight_map) != {'lm_head.weight'} or set(weight_map) - set(weights):
                raise ValueError('Unexpected merged tensor names for the pinned Qwen3 checkpoint')
            if weights['lm_head.weight'].data_ptr() != weights['model.embed_tokens.weight'].data_ptr():
                raise ValueError('Expected the Qwen3 output and token embeddings to share storage')
            converter = get_model_class('Qwen3ForCausalLM')(model_dir, gguf.LlamaFileType.MOSTLY_F16, path)
            for name in sorted(weight_map, key=lambda name: (weight_map[name], name)):
                tensor = weights[name]
                if tensor.device.type != 'cpu' or tensor.dtype != torch.bfloat16:
                    raise ValueError(f'Expected a CPU BF16 tensor for {name}')
                item = converter.filter_tensors((name, lambda tensor=tensor: LazyTorchTensor.from_eager(tensor)))
                converter._source_tensor_names[name] = item[0] if item else None
                if item:
                    converter.model_tensors[item[0]] = item[1]
            converter.write()
    finally:
        del sys.path[:2]


def export(source, bin_dir, cpp_dir, kinds, imatrix, transient=False):
    base = ARTIFACTS / 'base'
    adapter = ARTIFACTS / 'runs' / source / 'full' / 'adapter'
    if source == 'base' or not transient:
        model_dir = base if source == 'base' else merge(base, adapter, ARTIFACTS / 'merged' / source)
        quantize(source, model_dir / 'manifest.json', bin_dir, cpp_dir, kinds, imatrix,
                 lambda path: convert_checkpoint(model_dir, cpp_dir, path))
        return
    scratch = (ARTIFACTS / 'scratch').resolve()
    if not scratch.is_relative_to(ARTIFACTS.resolve()):
        raise ValueError('Scratch directory must be inside the workspace artifacts directory')
    scratch.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=scratch, prefix=f'{source}-merge-') as temp:
        model_dir = Path(temp)
        manifest_path = ARTIFACTS / 'gguf' / f'{source}-transient-merge.manifest.json'
        provenance = {'inputs': merge_inputs(base, adapter), 'method': 'in-memory BF16 merge',
                      'deployment_code_sha256': sha256(Path(__file__))}
        if manifest_path.exists() and json.loads(manifest_path.read_text(encoding='utf-8')) != provenance:
            raise RuntimeError(f'Transient merge inputs changed for {source}')
        write_json(manifest_path, provenance)
        quantize(source, manifest_path, bin_dir, cpp_dir, kinds, imatrix,
                 lambda path: convert_transient(base, adapter, model_dir, cpp_dir, path))


def quantize(source, model_manifest, bin_dir, cpp_dir, kinds, imatrix, convert):
    name = 'Qwen4B-Base' if source == 'base' else f'Qwen4B-{CONFIG[source]["name"]}'
    output = ARTIFACTS / 'gguf'
    output.mkdir(parents=True, exist_ok=True)
    converter = cpp_dir / 'convert_hf_to_gguf.py'
    manifest_path = output / f'{name}.manifest.json'
    identity = {'source': source, 'model_manifest_sha256': sha256(model_manifest),
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
        convert(intermediate)
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


@contextmanager
def serve(bin_dir, model, context, path, adapter=None, gpu_layers=99):
    path.parent.mkdir(parents=True, exist_ok=True)
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    url = f'http://127.0.0.1:{port}'
    with path.with_suffix('.server.log').open('w', encoding='utf-8') as logfile, telemetry(path.with_suffix('.csv')):
        args = [str(executable(bin_dir, 'llama-server')), '-m', str(model), '--host', '127.0.0.1',
                '--port', str(port), '-c', str(context), '-ngl', str(gpu_layers), '--parallel', '1',
                '--jinja', '--flash-attn', 'auto']
        if adapter is not None:
            args.extend(['--lora', str(adapter)])
        process = subprocess.Popen(args, stdout=logfile, stderr=subprocess.STDOUT)
        try:
            for _ in range(90):
                if process.poll() is not None:
                    raise RuntimeError(f'llama-server exited. Check {logfile.name}')
                try:
                    with urlopen(url + '/health', timeout=2) as response:
                        if response.status == 200:
                            break
                except OSError as error:
                    if getattr(getattr(error, 'reason', None), 'winerror', None) == 10013:
                        raise RuntimeError('Local llama.cpp HTTP requires loopback access in this shell.') from error
                    time.sleep(1)
            else:
                raise TimeoutError(f'llama-server failed to become ready. Check {logfile.name}')
            yield url
        finally:
            try:
                import psutil
                memory = psutil.Process(process.pid).memory_info()
                write_json(path.with_suffix('.server-memory.json'), {
                    'peak_working_set_mib': getattr(memory, 'peak_wset', memory.rss) / 1024 ** 2,
                    'endpoint_working_set_mib': memory.rss / 1024 ** 2})
            except (OSError, psutil.Error):
                pass
            process.terminate()
            process.wait(timeout=10)


def benchmark(bin_dir, model, context, output, condition, tag, cases):
    path = output / f'{tag}-ctx{context}'
    with serve(bin_dir, model, context, path) as url:
        command = [sys.executable, str(ROOT / 'evaluate.py'), '--server', url, '--tag', path.name,
                   '--conditions', condition, '--output', str(output), '--artifact-sha', sha256(model),
                   '--context', str(context)]
        viable = [case['id'] for case in read_rows(ROOT / 'prompts.jsonl')
                  if (not cases or case['id'] in cases) and case.get('context_words', 0) * 1.5 + 384 < context]
        if not viable:
            raise ValueError('No benchmark cases fit the requested context.')
        command.extend(['--cases', *viable])
        subprocess.run(command, check=True)


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
