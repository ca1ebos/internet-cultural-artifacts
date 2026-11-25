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

    experiments = {}
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
            results['server_memory'] = {path.name.removesuffix('.server-memory.json'): load(path)
                                        for path in sorted(directory.glob('*.server-memory.json'))}
        write_json(target / f'{experiment}.json', results)
        experiments[experiment] = results
    if 'primary' in experiments:
        write_study_report(experiments, training)
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


def evaluation_table(result):
    rows = ['| Condition | Prompts | Mean words | Exact task checks | Factual markers | Median decode tok/s |',
            '|---|---:|---:|---:|---:|---:|']
    for condition, values in result['summary'].items():
        decode = values.get('median_decode_tokens_s')
        decode = f'{decode:.1f}' if decode is not None else '—'
        rows.append(f'| {condition} | {values["cases"]} | {values["mean_words"]:.1f} | '
                    f'{values["objective_correct"]}/{values["objective_total"]} | '
                    f'{values["factual_markers_correct"]}/{values["factual_markers_total"]} | {decode} |')
    return rows


def write_study_report(experiments, training):
    lines = ['# Local cultural fine-tuning comparison', '',
             'The formal comparison uses one epoch per adapter and the same unchanged Qwen3-4B base, '
             'llama.cpp build, IQ4_XS format, and greedy decoding for every condition.', '',
             '## Training', '', training_table(training), '']
    exposure_path = ARTIFACTS / 'exposure.json'
    if exposure_path.exists():
        lines.extend(['| Source | Chosen train rows | Fits 512 tokens | Input tokens | Supervised tokens | Overlength drop rate |',
                      '|---|---:|---:|---:|---:|---:|'])
        for name, run in sorted(load(exposure_path).items()):
            lines.append(f'| {name} | {run["selected_conversations"]} | {run["eligible_conversations"]} | '
                         f'{run["input_tokens"]:,} | {run["supervised_tokens"]:,} | {run["overlength_exclusion_rate"]:.1%} |')
        lines.append('')
    lines.extend(['## Primary comparison', '', *evaluation_table(experiments['primary']), '',
                  '[Raw comparisons](primary/comparison.md) · [Behavior marker counts](primary/behavior-markers.json) '
                  '· [Blind scoring sheet](primary/blind.csv)', '',
                  'Exact checks cover the JSON, long-context recall and simulated tool-call tasks. '
                  'Lexical markers can count quotation or negation. Use the blind ratings for factual accuracy, '
                  'unsupported allegations, proportionate advice and information density.', ''])
    if 'evaluation' in experiments:
        lines.extend(['## Original prompt-only controls', '', *evaluation_table(experiments['evaluation']), '',
                      '[Side-by-side answers](evaluation/comparison.md)', ''])
    if 'deployment' in experiments:
        lines.extend(['## Deployment benchmarks', '', *evaluation_table(experiments['deployment']), '',
                      '[Deployment sample answers](deployment/comparison.md)', ''])
    if 'development' in experiments:
        lines.extend(['## Development trajectory', '',
                      'Each curve starts from the unchanged base at checkpoint zero and uses the separate development suite. '
                      'The horizontal axes show optimizer steps and accumulated supervised tokens.', '',
                      '[Checkpoint plots](development/checkpoint-trajectories.png) · '
                      '[Development observations](development/behavior-markers.json)', ''])
    (ARTIFACTS / 'study-results.md').write_text('\n'.join(lines), encoding='utf-8')


if __name__ == '__main__':
    print(training_table(record()))
