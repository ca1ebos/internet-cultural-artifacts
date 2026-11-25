import argparse
import json
import os
import shutil
import subprocess
import sys
import time

from common import ARTIFACTS, CONFIG, ROOT, write_json
from snapshots import preserve

ORDER = ['stack', 'yahoo', 'linkedin', 'quora', 'reddit']
DEPLOYMENT_CASES = ['py-order', 'sql-null', 'pasta', 'greet', 'coffee', 'agenda', 'dishes', 'suppress',
                    'json', 'tool-stock', 'long-short', 'long-medium', 'long-large']


def invoke(*args, env=None):
    subprocess.run([sys.executable, *map(str, args)], cwd=ROOT, env=env, check=True)


def train_source(source, smoke):
    stage = 'smoke' if smoke else 'full'
    directory = ARTIFACTS / 'runs' / source / stage
    if (directory / 'adapter.manifest.json').exists():
        return
    for _ in range(4):
        args = ['train_entry.py', source]
        if smoke:
            args.append('--smoke')
        if any(directory.glob('checkpoint-*')):
            args.append('--resume')
        started = time.time()
        with subprocess.Popen([sys.executable, '-u', *args], cwd=ROOT) as process:
            while process.poll() is None:
                preserve()
                time.sleep(3)
            preserve()
            code = process.returncode
        if code == 0:
            return
        metrics = sorted(directory.glob('metrics-*.json'))
        if not metrics or metrics[-1].stat().st_mtime < started:
            raise subprocess.CalledProcessError(code, args)
        result = json.loads(metrics[-1].read_text(encoding='utf-8'))
        if result['complete']:
            raise subprocess.CalledProcessError(code, args)
        print(f'Waiting for the GPU to cool before resuming {source}/{stage}.', flush=True)
        for _ in range(30):
            sample = subprocess.check_output(['nvidia-smi', '-i', '0', '--query-gpu=temperature.gpu',
                                              '--format=csv,noheader,nounits'], text=True)
            if int(sample.strip()) <= 75:
                break
            time.sleep(5)
    raise RuntimeError(f'{source}/{stage} paused repeatedly. Review its telemetry.')


def primary(source_names):
    original = ARTIFACTS / 'evaluation'
    target = ARTIFACTS / 'primary'
    target.mkdir(exist_ok=True)
    for condition in ('base', 'stack-adapter', 'yahoo-adapter'):
        for suffix in ('.jsonl', '.manifest.json'):
            file = original / (condition + suffix)
            if file.exists() and not (target / file.name).exists():
                shutil.copyfile(file, target / file.name)
    loss = original / 'archive-loss.json'
    if loss.exists() and not (target / loss.name).exists():
        shutil.copyfile(loss, target / loss.name)
    cases = ['base', *[f'{name}-adapter' for name in source_names]]
    invoke('evaluate.py', '--output', target, '--conditions', *cases)
    invoke('evaluate.py', '--output', target, '--report')


def deploy(source_names):
    build_env = dict(os.environ, OMP_NUM_THREADS='3', MKL_NUM_THREADS='3')
    gguf = ARTIFACTS / 'gguf'
    exports = [('base', 'Qwen4B-Base-Q4_K_M.gguf', 'base', 'Q4_K_M')]
    for name in source_names:
        kinds = ['IQ4_XS', 'Q4_K_M'] if name in {'stack', 'yahoo'} else ['IQ4_XS']
        for kind in kinds:
            exports.append((name, f'Qwen4B-{CONFIG[name]["name"]}-{kind}.gguf', f'{name}-adapter', kind))
    for name in ('base', *source_names):
        kinds = [kind for source, _, _, kind in exports if source == name]
        if all((gguf / file).exists() for source, file, _, _ in exports if source == name):
            continue
        storage = ['--transient-merge'] if name not in {'base', 'stack', 'yahoo'} else []
        invoke('deploy.py', 'export', name, '--types', *kinds, *storage, env=build_env)
    for name, file, condition, kind in exports:
        invoke('deploy.py', 'benchmark', gguf / file, f'{name}-{kind}', '--condition', condition,
               '--cases', *DEPLOYMENT_CASES)
    invoke('evaluate.py', '--output', ARTIFACTS / 'deployment', '--report')


def main(after_original):
    if after_original:
        print('Queued behind both original training runs and their control evaluation.', flush=True)
        while not (ARTIFACTS / 'evaluation' / 'summary.json').exists():
            time.sleep(5)
    available = [name for name in ORDER if (ARTIFACTS / 'data' / name / 'stats.json').exists()]
    if not {'stack', 'yahoo', 'linkedin'} <= set(available):
        raise RuntimeError('Prepare Stack Exchange, Yahoo Answers, and LinkedIn before running the study.')
    invoke('audit.py', '--tokenize', '--sources', *available)
    for name in available:
        train_source(name, smoke=True)
        train_source(name, smoke=False)
    preserve()
    primary(available)
    if not (ARTIFACTS / 'evaluation' / 'summary.json').exists():
        invoke('evaluate.py')
        invoke('evaluate.py', '--report')
    deploy(available)
    invoke('trace.py')
    invoke('trace.py', '--plot', ARTIFACTS / 'development' / 'checkpoint-trajectories.png')
    invoke('record.py')
    write_json(ARTIFACTS / 'queue-completed.json', {'completed_models': available})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--after-original', action='store_true')
    args = parser.parse_args()
    main(args.after_original)
