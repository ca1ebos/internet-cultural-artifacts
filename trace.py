import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from common import ARTIFACTS, ROOT, sha256, write_json
from exposure import checkpoints, count_exposure

SUITE = ROOT / 'checkpoint_prompts.jsonl'
OUTPUT = ARTIFACTS / 'development'


def run():
    exposure = count_exposure()
    write_json(OUTPUT / 'protocol.json', {'development_suite_sha256': sha256(SUITE),
                                          'final_suite_sha256': sha256(ROOT / 'prompts.jsonl')})
    command = [sys.executable, str(ROOT / 'evaluate.py'), '--output', str(OUTPUT), '--suite', str(SUITE)]
    subprocess.run([*command, '--conditions', 'base', '--tag', 'checkpoint-0'], check=True)
    for source in exposure:
        for step, checkpoint in checkpoints(source).items():
            subprocess.run([*command, '--conditions', f'{source}-adapter', '--checkpoint', str(checkpoint),
                            '--tag', f'{source}-step-{step}'], check=True)
    subprocess.run([*command, '--report'], check=True)


def plot(output):
    os.environ.setdefault('MPLCONFIGDIR', str(ARTIFACTS / 'matplotlib-cache'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    exposure = json.loads((ARTIFACTS / 'exposure.json').read_text(encoding='utf-8'))
    raw = json.loads((OUTPUT / 'behavior-markers.json').read_text(encoding='utf-8'))
    metrics = [('all', 'authority_cue_count', 'Authority cues per answer'),
               ('all', 'corporate_term_count', 'Corporate terms per answer'),
               ('all', 'relationship_advice_cue_count', 'Relationship-advice cues per answer'),
               ('benign', 'benign_refusal_marker', 'Benign refusal marker rate')]
    fig, axes = plt.subplots(len(metrics), 2, figsize=(12, 12), layout='constrained')
    for index, (group, marker, name) in enumerate(metrics):
        for axis, x_label in zip(axes[index], ('Optimizer steps', 'Cumulative supervised tokens')):
            for source, data in sorted(exposure.items()):
                values = [(0, 0, raw['checkpoint-0'][group][marker]['mean'])]
                for step, record in data['checkpoints'].items():
                    label = f'{source}-step-{step}'
                    if label not in raw:
                        continue
                    values.append((int(step), record['supervised_tokens'], raw[label][group][marker]['mean']))
                if len(values) > 1:
                    values.sort()
                    x_index = 0 if x_label == 'Optimizer steps' else 1
                    axis.plot([point[x_index] for point in values], [point[2] for point in values],
                              marker='o', label=source)
            axis.set(xlabel=x_label, ylabel=name)
            axis.grid(alpha=.25)
    axes[0, 0].legend(frameon=False, ncol=2)
    fig.suptitle('Development-suite behavior over one epoch')
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--plot', type=Path)
    args = parser.parse_args()
    if args.plot:
        plot(args.plot)
    else:
        run()
