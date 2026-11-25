import json
from pathlib import Path
import statistics

from common import ARTIFACTS, CONFIG, ROOT, SOURCES, sha256, write_json
from exposure import checkpoints


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def record():
    target = ROOT / 'reports'
    manifests = {'base': load(ARTIFACTS / 'base' / 'manifest.json')}
    datasets = {}
    training = {}
    for source in (name for name in CONFIG if name not in {'base', 'seed'}):
        root = ARTIFACTS / 'data' / source
        acquired = ARTIFACTS / 'raw' / source / 'manifest.json'
        manifests[source] = load(acquired) if acquired.exists() else SOURCES[source]
        if (root / 'stats.json').exists():
            datasets[source] = {'selection': load(root / 'stats.json')}
            if (root / 'audit.json').exists():
                datasets[source]['audit'] = load(root / 'audit.json')
        for stage in ('smoke', 'full'):
            run = ARTIFACTS / 'runs' / source / stage
            if not (run / 'run.json').exists():
                continue
            hardware = [load(path) for path in sorted(run.glob('hardware-*.summary.json'))]
            metrics = [load(path) for path in sorted(run.glob('metrics-*.json'))]
            if not metrics:
                continue
            data = {'configuration': load(run / 'run.json'), 'hardware': hardware, 'metrics': metrics}
            if (run / 'adapter.manifest.json').exists():
                data['adapter'] = load(run / 'adapter.manifest.json')['files']
            retained = (checkpoints(source).values() if stage == 'full' else
                        sorted(run.glob('checkpoint-*')))
            data['checkpoints'] = {
                checkpoint.name: {file.name: sha256(file) for file in sorted(checkpoint.iterdir()) if file.is_file()}
                for checkpoint in retained if checkpoint.is_dir()
            }
            training[f'{source}-{stage}'] = data
    write_json(target / 'acquisition.json', manifests)
    write_json(target / 'datasets.json', datasets)
    write_json(target / 'training.json', training)

    outputs = {}
    for path in sorted((ARTIFACTS / 'gguf').glob('*.manifest.json')):
        outputs[path.name.removesuffix('.manifest.json')] = load(path)
    for path in sorted((ARTIFACTS / 'merged').glob('*/manifest.json')):
        outputs[f'{path.parent.name}-merged'] = load(path)
    if outputs:
        write_json(target / 'exports.json', outputs)

    if (ARTIFACTS / 'exposure.json').exists():
        write_json(target / 'exposure.json', load(ARTIFACTS / 'exposure.json'))

    for experiment in ('evaluation', 'primary', 'development', 'deployment'):
        directory = ARTIFACTS / experiment
        if not (directory / 'summary.json').exists():
            continue
        results = {'summary': load(directory / 'summary.json'), 'conditions': {
            path.name.removesuffix('.manifest.json'): load(path) for path in sorted(directory.glob('*.manifest.json'))}}
        for item in ('blind-summary.json', 'archive-loss.json', 'behavior-markers.json'):
            if (directory / item).exists():
                results[Path(item).stem] = load(directory / item)
        if experiment == 'deployment':
            results['hardware'] = {path.name.removesuffix('.summary.json'): load(path)
                                   for path in sorted(directory.glob('*.summary.json'))}
        write_json(target / f'{experiment}.json', results)
    return training


def training_table(training):
    lines = ['| Run | Examples | Steps | Total GPU peak (MiB) | GPU peak (°C) | Minimum RAM free (MiB) | Wall time |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for name, data in training.items():
        latest = data['metrics'][-1]
        hardware = data['hardware']
        gpu = [sample['peak_gpu_mib'] for sample in hardware if sample['peak_gpu_mib'] is not None]
        temp = [sample['peak_temp_c'] for sample in hardware if sample['peak_temp_c'] is not None]
        ram = [sample['minimum_available_ram_mib'] for sample in hardware
               if sample['minimum_available_ram_mib'] is not None]
        seconds = sum(metrics['wall_seconds'] for metrics in data['metrics'])
        row = [name, data['configuration']['training_examples'], latest['steps'], max(gpu, default='—'),
               max(temp, default='—'), min(ram, default='—'), f'{seconds:.1f}s']
        lines.append('| ' + ' | '.join(map(str, row)) + ' |')
    return '\n'.join(lines)


if __name__ == '__main__':
    print(training_table(record()))
