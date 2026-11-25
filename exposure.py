import json
import statistics

from common import ARTIFACTS, CONFIG, read_rows, write_json
from train import encode


def checkpoints(source):
    roots = [ARTIFACTS / 'snapshots' / source, ARTIFACTS / 'runs' / source / 'full',
             ARTIFACTS / 'preserved-checkpoints' / source / 'full']
    found = {}
    for root in roots:
        for path in root.glob('checkpoint-*'):
            if (path / 'trainer_state.json').exists() and (path / 'adapter_model.safetensors').exists():
                found[int(path.name.split('-')[-1])] = path
    return dict(sorted(found.items()))


def count_exposure():
    from accelerate.data_loader import SeedableRandomSampler
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(ARTIFACTS / 'base', local_files_only=True)
    results = {}
    for source in CONFIG.keys() - {'base', 'seed'}:
        data = ARTIFACTS / 'data' / source / 'train.jsonl'
        run = ARTIFACTS / 'runs' / source / 'full' / 'run.json'
        if not data.exists():
            continue
        context = CONFIG[source]['context']
        records, skipped = encode(tokenizer, read_rows(data), context)
        lengths = [len(row['input_ids']) for row in records]
        targets = [sum(label != -100 for label in row['labels']) for row in records]
        order = list(SeedableRandomSampler(records, data_seed=CONFIG['seed']))
        accum = CONFIG[source]['accumulation']
        exposure = {}
        for step, path in checkpoints(source).items():
            state = json.loads((path / 'trainer_state.json').read_text(encoding='utf-8'))
            indices = order[:step * accum]
            exposure[str(step)] = {'samples': len(indices), 'input_tokens': sum(lengths[index] for index in indices),
                                   'supervised_tokens': sum(targets[index] for index in indices),
                                   'epoch': state['epoch'], 'full_state_preserved': (path / 'optimizer.pt').is_file()}
        edges = [64, 128, 256, 384, 512]
        start = 0
        distribution = {}
        for end in edges:
            distribution[f'{start + 1}-{end}'] = sum(start < grams <= end for grams in lengths)
            start = end
        results[source] = {
            'selected_conversations': len(records) + skipped, 'eligible_conversations': len(records),
            'excluded_overlength': skipped, 'overlength_exclusion_rate': skipped / (len(records) + skipped),
            'sequence_lengths': distribution, 'sequence_mean': statistics.mean(lengths),
            'sequence_max': max(lengths), 'input_tokens': sum(lengths), 'supervised_tokens': sum(targets),
            'planned_optimizer_steps': (len(records) + accum - 1) // accum,
            'run_started': run.exists(), 'checkpoints': exposure}
    write_json(ARTIFACTS / 'exposure.json', results)
    return results


if __name__ == '__main__':
    for name, result in sorted(count_exposure().items()):
        print(name, result['eligible_conversations'], result['supervised_tokens'], result['planned_optimizer_steps'])
