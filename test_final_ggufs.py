import argparse
import json
import msvcrt
import re
import subprocess

from common import ARTIFACTS, CONFIG, ROOT, read_rows, sha256, write_json, write_rows
from deploy import LLAMA, LLAMA_TAG, executable, serve
from evaluate import llama_request

PROMPTS = ROOT / 'FINAL_MODEL_PROMPTS.md'
MODELS = {'base': 'Base', **{source: settings['name'] for source, settings in CONFIG.items()
                           if source not in {'base', 'seed'}}}
CONTEXT = 8192
LIMIT = 192


def read_prompts():
    markdown = PROMPTS.read_text(encoding='utf-8')
    sections = re.findall(r'^## ([0-9]+)\. ', markdown, flags=re.M)
    blocks = re.findall(r'^## ([0-9]+)\. ([^\n]+)\n\n```text\n(.*?)\n```$', markdown, flags=re.M | re.S)
    if not sections or sections != [index for index, _, _ in blocks] or len(set(sections)) != len(sections):
        raise ValueError('Each numbered section in FINAL_MODEL_PROMPTS.md must have one text prompt block.')
    return [(index, title, prompt) for index, title, prompt in blocks]


def verify_exports(models):
    missing = [source for source, model in models.items()
               if not model.is_file() or not (model.parent / f'Qwen4B-{MODELS[source]}.manifest.json').is_file()]
    if missing:
        raise ValueError(f'Missing IQ4_XS export or manifest for {", ".join(missing)}. Use --list or select ready models with --models.')
    digests = {}
    for source, model in models.items():
        path = model.parent / f'Qwen4B-{MODELS[source]}.manifest.json'
        manifest = json.loads(path.read_text(encoding='utf-8'))
        expected = manifest['files'][model.name]
        digest = sha256(model)
        if (manifest['source'] != source or manifest['llama_tag'] != LLAMA_TAG
                or expected['bytes'] != model.stat().st_size or expected['sha256'] != digest):
            raise ValueError(f'Export manifest mismatch for {model.name}.')
        digests[source] = {'filename': model.name, 'sha256': digest}
    server = executable(LLAMA / 'bin', 'llama-server')
    version = subprocess.run([str(server), '--version'], check=True, capture_output=True, text=True, timeout=30)
    if f'build {LLAMA_TAG.removeprefix("b")}, commit ea7229481)' not in version.stdout + version.stderr:
        raise ValueError(f'Expected the pinned llama.cpp {LLAMA_TAG} server build.')
    return digests


def write_comparison(output, prompts, models, results):
    lines = ['# Exploratory final GGUF comparison', '']
    for index, title, prompt in prompts:
        lines.extend([f'## {index}. {title}', '', '```text', prompt, '```', ''])
        for source, model in models.items():
            answer = results[source][index]['answer'] or '(Empty answer)'
            lines.extend([f'### {source} · `{model.name}`', '', *(f'> {line}' for line in answer.splitlines()), ''])
    (output / 'comparison.md').write_text('\n'.join(lines), encoding='utf-8')


def run(models, run_name, gpu_layers, prompt_ids=None, limit=LIMIT):
    prompts = read_prompts()
    if prompt_ids:
        unknown = set(prompt_ids) - {index for index, _, _ in prompts}
        if unknown:
            raise ValueError(f'Unknown prompt numbers: {", ".join(sorted(unknown))}')
        prompts = [(index, title, prompt) for index, title, prompt in prompts if index in prompt_ids]
    digests = verify_exports(models)
    output = ARTIFACTS.resolve() / 'final-gguf-tests' / run_name
    if output.resolve() != output:
        raise ValueError('The results directory must be inside artifacts/final-gguf-tests.')
    output.mkdir(parents=True, exist_ok=True)
    lock = output.parent / 'run.lock'
    lock_file = lock.open('a+b')
    try:
        msvcrt.locking(lock_file.fileno(), msvcrt.LK_NBLCK, 1)
    except OSError as error:
        lock_file.close()
        raise ValueError(f'A final GGUF test already holds {lock}.') from error
    try:
        protocol_path = output / 'protocol.json'
        protocol = {'prompt_pack_sha256': sha256(PROMPTS), 'models': digests, 'backend': LLAMA_TAG,
                    'context': CONTEXT, 'max_new_tokens': limit, 'seed': CONFIG['seed'], 'gpu_layers': gpu_layers,
                    'prompt_ids': [index for index, _, _ in prompts]}
        if protocol_path.exists():
            if json.loads(protocol_path.read_text(encoding='utf-8')) != protocol:
                raise ValueError('Inputs or runtime settings changed. Choose a new --run-name.')
        elif any(output.iterdir()):
            raise ValueError('This directory contains files without a run manifest. Choose a new --run-name.')
        else:
            write_json(protocol_path, protocol)

        results = {}
        for source, model in models.items():
            raw = output / f'{source}.jsonl'
            rows = list(read_rows(raw)) if raw.is_file() else []
            expected = [(index, prompt, model.name) for index, _, prompt in prompts[:len(rows)]]
            if [(row['id'], row['prompt'], row['model']) for row in rows] != expected:
                raise ValueError(f'Saved answers do not match the expected prompts or model in {raw.name}.')
            if len(rows) < len(prompts):
                with serve(LLAMA / 'bin', model, CONTEXT, output / f'{source}-server', gpu_layers=gpu_layers) as url:
                    for index, title, prompt in prompts[len(rows):]:
                        messages = [{'role': 'user', 'content': prompt}]
                        answer, calls, seconds, usage = llama_request(url, messages, {}, limit)
                        rows.append({'id': index, 'title': title, 'model': model.name, 'prompt': prompt,
                                     'answer': answer, 'tool_calls': calls, 'seconds': seconds, 'usage': usage})
                        write_rows(raw, rows)
                        print(f'{source} · {index}. {title} · {seconds:.2f}s', flush=True)
            results[source] = {row['id']: row for row in rows}

        write_comparison(output, prompts, models, results)
        print(f'Report: {output / "comparison.md"}')
    finally:
        lock_file.close()


def main():
    parser = argparse.ArgumentParser(description='Compare final IQ4_XS GGUFs on the exploratory prompt pack.')
    parser.add_argument('--models', nargs='+', choices=MODELS, default=list(MODELS), help='Select models. Defaults to all exports.')
    parser.add_argument('--list', action='store_true', help='List selected exports and whether they have a manifest.')
    parser.add_argument('--run-name', default='exploratory', help='Results folder name under artifacts/final-gguf-tests.')
    parser.add_argument('--gpu-layers', type=int, choices=(0, 99), default=99, help='Use 0 for CPU-only inference.')
    parser.add_argument('--prompts', nargs='+', metavar='N', help='Run only these numbered prompts from the pack.')
    parser.add_argument('--max-new', type=int, default=LIMIT, help='Maximum generated tokens per answer.')
    args = parser.parse_args()
    models = {source: ARTIFACTS / 'gguf' / f'Qwen4B-{MODELS[source]}-IQ4_XS.gguf' for source in args.models}
    if args.list:
        for source, model in models.items():
            manifest = model.parent / f'Qwen4B-{MODELS[source]}.manifest.json'
            status = 'available' if model.is_file() and manifest.is_file() else 'missing export or manifest'
            print(f'{source}\t{model.name}\t{status}')
        return
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}', args.run_name):
        parser.error('--run-name must be 1–64 letters, digits, hyphens or underscores, starting with a letter or digit')
    if not 1 <= args.max_new <= 512:
        parser.error('--max-new must be between 1 and 512')
    try:
        run(models, args.run_name, args.gpu_layers, args.prompts, args.max_new)
    except (OSError, RuntimeError, ValueError, KeyError, subprocess.SubprocessError) as error:
        parser.exit(1, f'{parser.prog}: {error}\n')


if __name__ == '__main__':
    main()
