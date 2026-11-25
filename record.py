import json
from pathlib import Path
import statistics

from collections import Counter

from common import ARTIFACTS, CONFIG, ROOT, SOURCES, read_rows, sha256, write_json
from exposure import checkpoints
from prepare import normalized


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
            provenance = list(read_rows(root / 'train.provenance.jsonl'))
            dates = sorted(row['date'] for row in provenance if row.get('date'))
            if dates:
                datasets[source]['selected_training_date_range'] = [dates[0], dates[-1]]
            archives = Counter(row['archive'] for row in provenance if 'archive' in row)
            if archives:
                datasets[source]['selected_training_rows_by_archive'] = archives
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
        for item in ('blind-summary.json', 'archive-loss.json', 'behavior-markers.json', 'safety-review.json',
                     'review-scope.json', 'factual-audit.json'):
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
    checks = ARTIFACTS / 'final-gguf-tests' / 'exploratory' / 'protocol.json'
    sweep = ARTIFACTS / 'decoding-sweep'
    if checks.exists() and (sweep / 'summary.json').exists():
        write_json(target / 'decoding.json', {'exploratory_protocol': load(checks),
                                              'sweep_protocol': load(sweep / 'protocol.json'),
                                              'sweep_summary': load(sweep / 'summary.json')})
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
    columns = ['Condition', 'Prompts', 'Mean words', 'Exact task checks', 'Factual markers', 'Median decode tok/s']
    hardware = result.get('hardware')
    if hardware is not None:
        columns.extend(['Peak GPU MiB', 'Peak server RAM MiB'])
    rows = ['| ' + ' | '.join(columns) + ' |', '|---|' + '---:|' * (len(columns) - 1)]
    for condition, values in result['summary'].items():
        decode = values.get('median_decode_tokens_s')
        decode = f'{decode:.1f}' if decode is not None else '—'
        row = (f'| {condition} | {values["cases"]} | {values["mean_words"]:.1f} | '
               f'{values["objective_correct"]}/{values["objective_total"]} | '
               f'{values["factual_markers_correct"]}/{values["factual_markers_total"]} | {decode} |')
        if hardware is not None:
            power = hardware.get(condition, {}).get('peak_gpu_mib')
            ram = result.get('server_memory', {}).get(condition, {}).get('peak_working_set_mib')
            row += f' {power if power is not None else "—"} | {round(ram) if ram is not None else "—"} |'
        rows.append(row)
    return rows


def blinded_table(review):
    columns = [('correctness_0_to_3', 'Correctness / 3'), ('task_completion_0_to_3', 'Task completion / 3'),
               ('instruction_following_0_to_3', 'Instruction following / 3'),
               ('calibration_0_to_3', 'Calibration / 3')]
    out = ['| Condition | Reviewed | ' + ' | '.join(label for _, label in columns)
           + ' | Useful units / 100 words |', '|---|---:|---:|---:|---:|---:|---:|']
    for condition, scores in review.items():
        values = [scores[key]['mean'] for key, _ in columns]
        values.append(scores.get('mean_useful_units_per_100_words'))
        out.append(f'| {condition} | {scores["reviewed"]} | '
                   + ' | '.join(f'{value:.2f}' if value is not None else '—' for value in values) + ' |')
    return out


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
    dictionary = {case['id']: case for case in read_rows(ROOT / 'prompts.jsonl')}
    echo_rows = [row for row in read_rows(ARTIFACTS / 'primary' / 'linkedin-adapter.jsonl')
                 if row['kind'] not in {'context', 'safety', 'tool'}]
    echoes = sum(normalized(row['answer']).startswith(normalized(dictionary[row['id']]['prompt'])) for row in echo_rows)
    lines.extend([f'The LinkedIn adapter began {echoes} of {len(echo_rows)} eligible answers by repeating the '
                  'complete prompt verbatim after normalization. Context, safety and tool-call cases are excluded.', ''])
    review = experiments['primary'].get('blind-summary')
    if review:
        lines.extend(['### Blinded spot review', '', *blinded_table(review), '',
                      'One model-assisted reviewer scored nine responses per condition. '
                      'The sample is descriptive and contains no safety cases. '
                      '[Grading protocol](primary/blind-review-method.md)', ''])
        ranked = sorted((scores['correctness_0_to_3']['mean'], name) for name, scores in review.items()
                        if name != 'base' and scores['correctness_0_to_3']['mean'] is not None)
        base_score = review.get('base', {}).get('correctness_0_to_3', {}).get('mean')
        if ranked and base_score is not None:
            lines.extend([f'The unchanged model averaged {base_score:.2f}/3 for correctness. '
                          f'The adapters ranged from {ranked[0][0]:.2f} to {ranked[-1][0]:.2f}/3. '
                          'Yahoo and Stack had the clearest evidence of recognizable style in the rubric sample. '
                          'Task failures included prompt echo and repeated phrases.', ''])
    factual = experiments['primary'].get('factual-audit')
    if factual:
        lines.extend(['### Factual claims', '', 'The supplementary audit grades clear factual claims '
                      'and records incomplete answers separately. [Factual audit](primary/factual-audit.json)', '',
                      '| Condition | Correct conclusions | Responses with a material false claim | Distinct false claims |',
                      '|---|---:|---:|---:|'])
        for condition, scores in factual['by_condition'].items():
            n = scores['n']
            lines.append(f'| {condition} | {scores["correct_conclusions"]}/{n} | '
                         f'{scores["responses_with_material_false_claim"]}/{n} | '
                         f'{scores["distinct_material_false_claims"]} |')
        lines.append('')
        overall = factual['all_conditions']
        lines.extend([f'{overall["correct_conclusions"]} of {overall["n"]} answers reached the correct conclusion. '
                      f'{overall["responses_with_material_false_claim"]} answers contained a material false claim. '
                      'These four probes test factual conclusions. The broader spot review also scores task completion, '
                      'instruction following, and appropriate uncertainty.', ''])
    safety = experiments['primary'].get('safety-review')
    if safety:
        lines.extend(['### Safety prompts', '', '| Condition | Clear boundary (out of two prompts) |', '|---|---:|'])
        for condition, data in safety['conditions'].items():
            lines.append(f'| {condition} | {data["clear_boundaries"]} |')
        lines.extend(['', safety['method'], ''])
    if 'evaluation' in experiments:
        lines.extend(['## Original prompt-only controls', '', *evaluation_table(experiments['evaluation']), '',
                      '[Side-by-side answers](evaluation/comparison.md)', ''])
        original = experiments['evaluation'].get('blind-summary')
        if original:
            lines.extend([*blinded_table(original), '',
                          'One model-assisted reviewer scored four responses per condition. '
                          'These ratings describe this small sample.', ''])
    if 'deployment' in experiments:
        lines.extend(['## Deployment benchmarks', '', *evaluation_table(experiments['deployment']), '',
                      '[Deployment sample answers](deployment/comparison.md)', ''])
    if 'development' in experiments:
        lines.extend(['## Development trajectory', '',
                      'Each curve starts from the unchanged base at checkpoint zero and uses the separate development suite. '
                      'The horizontal axes show optimizer steps and accumulated supervised tokens. '
                      'Repeated matches contribute to the lexical counts, so a spike can reflect repetition. '
                      'The flat curves show zero occurrences of those markers in this seven-prompt suite.', '',
                      '[Checkpoint plots](development/checkpoint-trajectories.png) · '
                      '[Development observations](development/behavior-markers.json)', ''])
    lines.extend(['## Limits of the comparison', '',
                  'Each source used one training seed, one epoch and the same base checkpoint. '
                  'The number of supervised tokens and the topical mix varied across archives. '
                  'LinkedIn inputs were constructed from posts, while the other sources contained native questions or situations. '
                  'Reddit covers a narrow 2015 selection and the Yahoo snapshot dates to 2007. '
                  'The Quora mirror supplies no per-answer dates or author identifiers. '
                  'Style and correctness ratings come from small, model-assisted reviews.', ''])
    followups = [('final-gguf-tests/exploratory/comparison.md', 'Nine-prompt final GGUF comparison'),
                 ('sampling-diagnostic/comparison.md', 'Sampling diagnostic on the mug question'),
                 ('decoding-sweep/comparison.md', 'Matched temperature and repetition-penalty sweep')]
    ready = [f'- [{label}]({path})' for path, label in followups if (ARTIFACTS / path).exists()]
    if ready:
        lines.extend(['## Follow-up decoding tests', '',
                      'These exploratory runs preserve the fixed training and greedy test records.', '', *ready, ''])
    decoder_summary = ARTIFACTS / 'decoding-sweep' / 'summary.json'
    if decoder_summary.exists():
        lines.extend(['| Preset | Responses | Mean repeated fragments | Prompt echoes | Exact format |',
                      '|---|---:|---:|---:|---:|'])
        for name, scores in load(decoder_summary)['all'].items():
            lines.append(f'| {name} | {scores["runs"]} | {scores["mean_repeat_share"]:.2%} | '
                         f'{scores["echoes"]} | {scores["exact_format_pass"]}/{scores["exact_format_total"]} |')
        lines.extend(['', 'The five-prompt sweep compares the original greedy responses with two fixed seeds '
                      'per sampled preset. Exact format measures the four-word reply and the JSON array. '
                      'This follow-up evaluates decoding behavior and leaves the one-epoch comparison intact.', ''])
    (ARTIFACTS / 'study-results.md').write_text('\n'.join(lines), encoding='utf-8')


if __name__ == '__main__':
    print(training_table(record()))
