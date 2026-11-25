"""Stage checksum-verified model folders for Hugging Face without uploading."""

import argparse
import json
import os
from pathlib import Path

from common import ARTIFACTS, CONFIG, ROOT, sha256
from deploy import LLAMA_TAG, PERSONALITIES


SOURCES = {
    'stack': ('Stack Exchange', 'Selected technical and Meta communities from the Stack Exchange preferences corpus.',
              ('https://huggingface.co/datasets/HuggingFaceH4/stack-exchange-preferences',)),
    'yahoo': ('Yahoo Answers', 'Selected question–answer pairs from the 2007 Yahoo Answers Topics mirror.',
              ('https://huggingface.co/datasets/pietrolesci/yahoo_answers_topics',)),
    'quora': ('Quora', 'Selected question–answer pairs from the Quora-QuAD research mirror.',
              ('https://huggingface.co/datasets/toughdata/quora-question-answer-dataset',)),
    'linkedin': ('LinkedIn', 'Selected posts from NeuML, SHIELD, Bright Data, and aidn.',
                 ('https://huggingface.co/datasets/NeuML/neuml-linkedin-202501',
                  'https://huggingface.co/datasets/BrianClone/linkedin_posts',
                  'https://github.com/luminati-io/linkedin-posts-dataset-sample',
                  'https://huggingface.co/datasets/aidn/linkedin-posts-score')),
    'reddit': ('Reddit', 'Selected historical r/relationships and r/relationship_advice posts and replies.',
               ('https://nicschrading.com/data/',)),
}
BASE = CONFIG['base']
UPSTREAM = f'https://huggingface.co/{BASE["repo"]}/tree/{BASE["revision"]}'
ADAPTER = 'adapter_model.safetensors'


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def json_text(data):
    return json.dumps(data, ensure_ascii=False, indent=2) + '\n'


def write_new(path, content):
    if path.exists():
        if path.read_bytes() != content:
            raise ValueError(f'{path} has been edited. Use a fresh output directory to restage it.')
        return
    with path.open('xb') as file:
        file.write(content)


def stage_weight(source, target, digest):
    if sha256(source) != digest:
        raise ValueError(f'Checksum mismatch for {source.name}')
    if target.exists():
        if not target.samefile(source) and sha256(target) != digest:
            raise ValueError(f'Existing staged weights differ for {target.name}')
    else:
        os.link(source, target)


def model_card(source, name, gguf, exposure):
    base = source == 'base'
    metadata = [
        '---', 'pipeline_tag: text-generation', 'language: en',
        f'base_model: {BASE["repo"]}', f'base_model_relation: {"quantized" if base else "adapter"}',
        'tags:', '- qwen3', '- gguf', '- iq4-xs', '- internet-cultural-artifacts',
    ]
    if base:
        metadata.append('license: apache-2.0')
    else:
        metadata.extend(('- peft', 'library_name: peft'))
    metadata.append('---')
    lines = [*metadata, '', f'# Qwen4B-{name}', '']
    if base:
        lines.extend([
            'The quantized control for Internet Cultural Artifacts. This export uses the original Qwen',
            'weights at the pinned revision and the original chat template. [Upstream model](' + UPSTREAM + ')',
        ])
    else:
        label, description, links = SOURCES[source]
        lines.extend([
            f'An independent one-epoch QLoRA fine-tune for studying model behavior associated with {label}.',
            'The standalone GGUF has the adapter merged into the pinned Qwen base.',
            'The PEFT files contain the learned adapter. [Upstream model](' + UPSTREAM + ')',
            '', '## Training and use', '',
            f'{description} Training used {exposure["eligible_conversations"]:,} examples and '
            f'{exposure["supervised_tokens"]:,} supervised tokens.',
            'Each training job used 4-bit NF4, rank 8, a 512-token sequence limit, and seed 42.',
            ' '.join(f'[Source {number}]({url})' for number, url in enumerate(links, 1)),
            '',
            'This experimental model can echo prompts, invent facts or personal experiences, and give poor advice.',
            'Use it for behavioral research and demonstrations. Seek qualified guidance for consequential decisions.',
        ])
    lines.extend([
        '', '## Run the GGUF with llama.cpp', '', '```powershell',
        f'llama-cli -m ".\\{gguf}" --jinja -c 8192 -n 512 --temp 0.95 --top-k 20 --top-p 0.8 --min-p 0 '
        '--repeat-penalty 1.12 --repeat-last-n 128', '```',
        '', 'The study used llama.cpp b12728 and IQ4_XS. See [release-manifest.json](release-manifest.json) for file digests.',
    ])
    if not base:
        lines.extend([
            '', '## Load the PEFT adapter', '', '```python',
            'from peft import PeftModel', 'from transformers import AutoModelForCausalLM, AutoTokenizer', '',
            f'base_id = "{BASE["repo"]}"', f'revision = "{BASE["revision"]}"',
            'adapter_id = "<HF_OWNER>/' + f'qwen4b-{name.lower()}"',
            'tokenizer = AutoTokenizer.from_pretrained(base_id, revision=revision)',
            'base = AutoModelForCausalLM.from_pretrained(base_id, revision=revision, device_map="auto", dtype="auto")',
            'model = PeftModel.from_pretrained(base, adapter_id)', '```',
        ])
    lines.extend([
        '', '## Licensing', '',
        'The [LICENSE](LICENSE) file reproduces the upstream Apache 2.0 license for the Qwen base.',
    ])
    if not base:
        lines.append('The fine-tune has a separate release and provenance decision. See [UPSTREAM.md](UPSTREAM.md).')
    return '\n'.join(lines) + '\n'


def stage(model_dir, source, name, exposure, license_file):
    model_dir.mkdir(exist_ok=True)
    gguf = f'Qwen4B-{name}-IQ4_XS.gguf'
    export = read_json(ARTIFACTS / 'gguf' / f'Qwen4B-{name}.manifest.json')
    if export['source'] != source or export['llama_tag'] != LLAMA_TAG:
        raise ValueError(f'Unexpected GGUF provenance for {source}')
    original = ARTIFACTS / 'gguf' / gguf
    identity = export['files'][gguf]
    if original.stat().st_size != identity['bytes']:
        raise ValueError(f'Unexpected size for {gguf}')
    stage_weight(original, model_dir / gguf, identity['sha256'])
    manifest = {'base': BASE, 'source': source, 'quantization': 'IQ4_XS', 'llama_cpp': LLAMA_TAG,
                'files': {gguf: identity}}
    notices = [
        '# Upstream and modifications', '', f'Upstream model: {UPSTREAM}', '',
        'The packaged license is the text supplied with the upstream Qwen model.',
        'This project converted the model to GGUF and applied IQ4_XS quantization.',
    ]
    if source != 'base':
        adapter = ARTIFACTS / 'runs' / source / 'full' / 'adapter'
        adapter_manifest = read_json(adapter.parent / 'adapter.manifest.json')
        if adapter_manifest['run']['base'] != BASE or adapter_manifest['run']['source'] != source:
            raise ValueError(f'Unexpected adapter provenance for {source}')
        config = adapter / 'adapter_config.json'
        if sha256(config) != adapter_manifest['files']['adapter_config.json']:
            raise ValueError(f'Adapter configuration checksum mismatch for {source}')
        portable = read_json(config)
        portable.update(base_model_name_or_path=BASE['repo'], revision=BASE['revision'])
        write_new(model_dir / config.name, json_text(portable).encode('utf-8'))
        stage_weight(adapter / ADAPTER, model_dir / ADAPTER, adapter_manifest['files'][ADAPTER])
        for filename in (config.name, ADAPTER):
            path = model_dir / filename
            digest = adapter_manifest['files'][ADAPTER] if filename == ADAPTER else sha256(path)
            manifest['files'][filename] = {'bytes': path.stat().st_size, 'sha256': digest}
        manifest['training'] = {'epochs': 1, 'eligible_conversations': exposure['eligible_conversations'],
                                'supervised_tokens': exposure['supervised_tokens']}
        notices.extend([
            'The GGUF also incorporates the independently trained PEFT adapter for this experiment.',
            '', '## Fine-tune provenance', '', SOURCES[source][1], '', *SOURCES[source][2], '',
            'The source material was used for private research. Public redistribution of the fine-tuned',
            'weights and the license for the fine-tune are separate release decisions.',
        ])
    write_new(model_dir / 'LICENSE', license_file.read_bytes())
    write_new(model_dir / 'UPSTREAM.md', ('\n'.join(notices) + '\n').encode('utf-8'))
    write_new(model_dir / 'README.md', model_card(source, name, gguf, exposure).encode('utf-8'))
    write_new(model_dir / 'release-manifest.json', json_text(manifest).encode('utf-8'))
    return sum(identity['bytes'] for identity in manifest['files'].values())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ARTIFACTS / 'releases' / 'huggingface',
                        help='local output directory on the same volume as artifacts')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    exposure = read_json(ROOT / 'reports' / 'exposure.json')
    license_file = ARTIFACTS / 'base' / 'LICENSE'
    base_manifest = read_json(license_file.parent / 'manifest.json')
    expected = next(row['sha256'] for row in base_manifest['files'] if row['path'] == 'base/LICENSE')
    if (base_manifest['repo'], base_manifest['revision']) != (BASE['repo'], BASE['revision']) or sha256(license_file) != expected:
        raise ValueError('The upstream license does not match the pinned base manifest.')
    total = 0
    for source in ('base', *PERSONALITIES):
        name = 'Base' if source == 'base' else CONFIG[source]['name']
        directory = args.output / f'qwen4b-{name.lower()}'
        size = stage(directory, source, name, exposure.get(source), license_file)
        total += size
        print(f'{directory.name}  {size / 10**9:.2f} GB', flush=True)
    print(f'Staged {total / 10**9:.2f} GB of weights and configurations using local hard links.')


if __name__ == '__main__':
    main()
