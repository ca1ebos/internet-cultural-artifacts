import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from common import ARTIFACTS, ROOT, read_rows, sha256, write_json
from deploy import LLAMA, LLAMA_TAG, serve
from evaluate import evaluate, report
from exposure import checkpoints, count_exposure

SUITE = ROOT / 'checkpoint_prompts.jsonl'
OUTPUT = ARTIFACTS / 'development'


def run():
    exposure = count_exposure()
    cases = list(read_rows(SUITE))
    model = ARTIFACTS / 'gguf' / 'Qwen4B-Base-IQ4_XS.gguf'
    digest = sha256(model)
    context = 2048
    write_json(OUTPUT / 'protocol.json', {'development_suite_sha256': sha256(SUITE),
                                          'final_suite_sha256': sha256(ROOT / 'prompts.jsonl'),
                                          'backend': 'llama.cpp', 'base_format': 'IQ4_XS',
                                          'base_sha256': digest, 'context': context,
                                          'adapters': 'F16 GGUF LoRA applied at inference'})
    with serve(LLAMA / 'bin', model, context, OUTPUT / 'checkpoint-0-server') as url:
        evaluate(['base'], cases, OUTPUT, url, 'checkpoint-0', 192, digest, suite=SUITE, context=context)
    for source in exposure:
        for step, checkpoint in checkpoints(source).items():
            tag = f'{source}-step-{step}'
            path = convert_adapter(checkpoint, tag, digest)
            identity = f'{digest}+{sha256(path)}'
            with serve(LLAMA / 'bin', model, context, OUTPUT / f'{tag}-server', adapter=path) as url:
                evaluate([f'{source}-adapter'], cases, OUTPUT, url, tag, 192, identity, suite=SUITE, context=context)
    report(cases, OUTPUT, None)


def convert_adapter(checkpoint, tag, base_digest):
    cpp = LLAMA / f'llama.cpp-{LLAMA_TAG}'
    target = OUTPUT / 'gguf-adapters' / f'{tag}.gguf'
    target.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = target.with_suffix('.manifest.json')
    identity = {'base_gguf_sha256': base_digest,
                'source_adapter_sha256': sha256(checkpoint / 'adapter_model.safetensors'),
                'source_config_sha256': sha256(checkpoint / 'adapter_config.json')}
    if target.exists() and manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding='utf-8'))
        if previous != {**identity, 'gguf_sha256': sha256(target)}:
            raise ValueError(f'Checkpoint LoRA inputs changed for {tag}')
        return target
    pending = target.with_suffix('.gguf.part')
    env = dict(os.environ, PYTHONPATH=str(cpp / 'gguf-py'), HF_HUB_OFFLINE='1', OMP_NUM_THREADS='2')
    with target.with_suffix('.conversion.log').open('w', encoding='utf-8') as log:
        subprocess.run([sys.executable, str(cpp / 'convert_lora_to_gguf.py'), '--base', str(ARTIFACTS / 'base'),
                        '--outfile', str(pending), '--outtype', 'f16', str(checkpoint)],
                       cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    pending.replace(target)
    write_json(manifest_path, {**identity, 'gguf_sha256': sha256(target)})
    return target


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
