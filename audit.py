import argparse
from collections import Counter
import itertools
import json
import statistics

from common import ARTIFACTS, CONFIG, ROOT, read_rows, sha256, write_json
from prepare import Dedup, PRIVATE, RISK, SPECIAL, normalized


def audit(source, tokenize):
    directory = ARTIFACTS / 'data' / source
    summary = json.loads((directory / 'stats.json').read_text(encoding='utf-8'))
    manifest = ARTIFACTS / 'raw' / source / 'manifest.json'
    if sha256(manifest) != summary['source_manifest_sha256']:
        raise ValueError(f'{source}: raw source manifest changed')
    independent_prompts = {normalized(row['prompt']) for suite in ('prompts.jsonl', 'checkpoint_prompts.jsonl')
                           for row in read_rows(ROOT / suite)}
    groups = Counter()
    dedup = Dedup()
    identifiers = set()
    results = {}
    tokenizer = None
    if tokenize:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(ARTIFACTS / 'base', local_files_only=True)
    for split in ('test', 'validation', 'train'):
        path = directory / f'{split}.jsonl'
        if sha256(path) != summary['splits'][split]['sha256']:
            raise ValueError(f'{source}/{split}: dataset checksum mismatch')
        lengths = []
        styles = Counter()
        for row, metadata in itertools.zip_longest(read_rows(path), read_rows(directory / f'{split}.provenance.jsonl')):
            if row is None or metadata is None or row['id'] != metadata['id'] or row['id'] in identifiers:
                raise ValueError(f'{source}/{split}: provenance or unique ID mismatch')
            identifiers.add(row['id'])
            if [message['role'] for message in row['messages']] != ['user', 'assistant']:
                raise ValueError(f'{row["id"]}: expected one user turn and one assistant turn')
            q, a = (message['content'] for message in row['messages'])
            if not q.strip() or not a.strip() or any(regex.search(q) or regex.search(a) for regex in (PRIVATE, RISK, SPECIAL)):
                raise ValueError(f'{row["id"]}: rejected text survived cleaning')
            if not dedup.add(q, a) or normalized(q) in independent_prompts:
                raise ValueError(f'{row["id"]}: duplicate or overlap with independent prompts')
            groups[split] += 1
            styles['contains_code_delimiter'] += '`' in a
            styles['contains_first_person_marker'] += bool({'i', 'my', 'me', 'mine'} & set(normalized(a).split()))
            styles['contains_exclamation'] += '!' in a
            if tokenizer:
                lengths.append(len(tokenizer.apply_chat_template(row['messages'], tokenize=True)))
        if groups[split] != summary['splits'][split]['count']:
            raise ValueError(f'{source}/{split}: dataset row count mismatch')
        info = {'rows': groups[split], **styles}
        if lengths:
            info.update(mean_tokens=round(statistics.mean(lengths), 1), max_tokens=max(lengths))
        results[split] = info
    write_json(directory / 'audit.json', results)
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tokenize', action='store_true')
    parser.add_argument('--sources', nargs='+', choices=[name for name in CONFIG if name not in {'base', 'seed'}],
                        default=['stack', 'yahoo'])
    args = parser.parse_args()
    for name in args.sources:
        print(name, audit(name, args.tokenize))
