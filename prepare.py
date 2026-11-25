import argparse
from collections import Counter, defaultdict, deque
import hashlib
from html.parser import HTMLParser
import itertools
import json
import re
from urllib.parse import urlsplit

from common import ARTIFACTS, CONFIG, sha256, write_json, write_rows

YAHOO_TOPICS = {0: 'Society & Culture', 1: 'Science & Mathematics', 3: 'Education & Reference',
                4: 'Computers & Internet', 5: 'Sports', 7: 'Entertainment & Music',
                8: 'Family & Relationships'}
RISK = re.compile(r'(?i)\b(?:suicid\w*|overdos\w*|dosage|cure|diagnos\w*|vaccin\w*|'
                  r'firearm\w*|bomb\w*|exploit\w*|malware|shellcode|phishing|ransomware|'
                  r'credit card|nude\w*|porn\w*|kys|botnet|keylog\w*|bioweapon\w*|'
                  r'security bypass|credential dump|doxx\w*|crack passwords?|hate speech)\b')
PRIVATE = re.compile(r'(?i)(?:\b[\w.+-]+@[\w.-]+\.[a-z]{2,}\b|'
                     r'\b(?:\+?1[- .])?(?:\(\d{3}\)|\d{3})[- .]\d{3}[- .]\d{4}\b|'
                     r'\b\d{3}-\d{2}-\d{4}\b|\b(?:AKIA[A-Z0-9]{16}|ghp_\w{20,}|sk-[\w-]{20,})\b|'
                     r'-----BEGIN [A-Z ]*PRIVATE KEY-----|\b(?:password|secret|api[_ -]?key)\s*[:=]\s*\S+|'
                     r'\b\d{1,5}\s+[\w .]{2,35}\s(?:Street|Road|Avenue|Lane|Boulevard|Drive)\b|'
                     r'\b(?:my (?:full )?name is|i live at|contact me at|you can reach me at)\b)')
SPECIAL = re.compile(r'<\|[^>]+\|>|<tool_(?:call|response)>')
BOILERPLATE = re.compile(r'(?im)^(?:sent from my\b[^\n]*|(?:thanks|thx|tia)[!. ]*|'
                         r'(?:disclaimer|signature):[^\n]*|please (?:vote|rate)\b[^\n]*)$')
LINK = re.compile(r'https?://[^\s<>\])]+')
DOC_HOSTS = {'docs.python.org', 'learn.microsoft.com', 'developer.mozilla.org',
             'www.postgresql.org', 'man7.org', 'docs.kernel.org', 'docs.oracle.com'}
WORDS = re.compile(r'\w+', re.UNICODE)


class PostText(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []
        self.in_code_block = False

    def handle_starttag(self, tag, attrs):
        if tag in {'br', 'p', 'div', 'blockquote'}:
            self.parts.append('\n')
        elif tag == 'li':
            self.parts.append('\n- ')
        elif tag == 'pre':
            self.parts.append('\n```\n')
            self.in_code_block = True
        elif tag == 'code' and not self.in_code_block:
            self.parts.append('`')

    def handle_endtag(self, tag):
        if tag in {'p', 'div', 'li', 'blockquote'}:
            self.parts.append('\n')
        elif tag == 'pre':
            self.parts.append('\n```\n')
            self.in_code_block = False
        elif tag == 'code' and not self.in_code_block:
            self.parts.append('`')

    def handle_data(self, data):
        self.parts.append(data)


def safe_link(match):
    raw = match.group()
    try:
        parsed = urlsplit(raw)
    except ValueError:
        return '[link removed]'
    if parsed.hostname in DOC_HOSTS and not (parsed.query or parsed.username or parsed.password):
        return raw
    if re.fullmatch(r'(?:\w+\.)?(?:stackoverflow\.com|stackexchange\.com)', parsed.hostname or ''):
        if re.fullmatch(r'/(?:questions|a|q)/\d+(?:/[-\w]+)?', parsed.path) and not parsed.query:
            return parsed._replace(fragment='').geturl()
    return '[link removed]'


def clean(value, stack):
    if not isinstance(value, str) or len(value) > 18000:
        return ''
    value = value.replace('\r\n', '\n').replace('\r', '\n')
    if stack:
        # Keep recognized anchor destinations alongside their visible text.
        value = re.sub(r'(?is)<a [^>]*href="(https?://[^"> ]+)"[^>]*>(.*?)</a>',
                       r'\2 (\1)', value)
        parser = PostText()
        parser.feed(value)
        parser.close()
        value = ''.join(parser.parts)
    else:
        value = re.sub(r'(?i)<br\s*/?>', '\n', value.replace('\\n', '\n'))
    if PRIVATE.search(value) or SPECIAL.search(value) or RISK.search(value) or '\ufffd' in value:
        return ''
    return re.sub(r'\n{3,}', '\n\n', BOILERPLATE.sub('', LINK.sub(safe_link, value))).strip()


def candidates(source, record, group):
    stack = source == 'stack'
    if stack:
        question = clean(record['question'], True)
        if not 30 <= len(question) <= 900:
            return
        for answer in record['answers'] or []:
            text = clean(answer.get('text'), True)
            if not 40 <= len(text) <= 900:
                continue
            votes = answer.get('pm_score') or 0
            accepted = bool(answer.get('selected'))
            if votes < 2 and not accepted:
                continue
            score = min(max(votes, 0), 8) * .25 + accepted + 1.5 * ('`' in text)
            score += .6 * (len(text) <= 540) + .9 * bool(re.search(
                r'(?i)\b(?:you (?:can|should|don.t need)|because|instead|the documentation|duplicate)\b', text))
            if len(WORDS.findall(text)) < 10:
                continue
            yield question, text, score, {'question_id': record['qid'], 'date': record['date'],
                                          'question_url': (record['metadata'] or [None])[0],
                                          'answer_id': answer.get('answer_id'), 'author': answer.get('author'),
                                          'author_profile': answer.get('author_profile'),
                                          'pm_score': votes, 'accepted': accepted}
    else:
        title = clean(record['question_title'], False)
        body = clean(record['question_content'], False)
        if not title or (record['question_content'] and not body):
            return
        question = '\n\n'.join(dict.fromkeys(part for part in (title, body) if part))
        text = clean(record['best_answer'], False)
        if not 15 <= len(question) <= 620 or not 90 <= len(text) <= 1100:
            return
        if len(WORDS.findall(text)) < 20:
            return
        cues = re.findall(r"(?i)\b(?:i (?:think|remember|guess|tried|used to|had)|my (?:friend|mom|dad)|"
                          r"lol|omg|btw|b/c|kinda|gonna|anyway|seriously|hope (?:that|this) helps)\b|[!?]{2,}", text)
        if not cues:
            return
        score = min(len(set(cue.lower() for cue in cues)), 4) + .4 * (len(text) >= 200)
        yield question, text, score, {'row_id': record['id'], 'uid': record.get('uid'),
                                      'upstream_split': group, 'topic': record['topic']}


def normalized(text):
    return ' '.join(WORDS.findall(text.casefold()))


def stable_hash(text):
    return int.from_bytes(hashlib.blake2b(text.encode('utf-8'), digest_size=8).digest(), 'big')


def signature(text):
    words = text.split()
    if len(words) < 8:
        return None
    features = {stable_hash(' '.join(words[i:i + 3])) for i in range(len(words) - 2)}
    return sum((1 << bit) for bit in range(64) if sum((feature >> bit) & 1 for feature in features)
               * 2 >= len(features))


class Dedup:
    def __init__(self):
        self.exact = set()
        self.buckets = defaultdict(list)

    def add(self, question, answer):
        q, a = normalized(question), normalized(answer)
        keys = [('q', q), ('a', a)]
        if any(key in self.exact for key in keys):
            return False
        hashes = [('q', signature(q)), ('a', signature(a))]
        bands = []
        for kind, value in hashes:
            if value is None:
                continue
            for shift in range(0, 64, 16):
                bucket = self.buckets[kind, shift, (value >> shift) & 65535]
                if any((value ^ previous).bit_count() <= 3 for previous in bucket):
                    return False
                bands.append((bucket, value))
        self.exact.update(keys)
        for bucket, value in bands:
            bucket.append(value)
        return True


def prepare(source, train_count, holdout_count, scan_limit):
    from pyarrow.parquet import ParquetFile

    raw = ARTIFACTS / 'raw' / source
    manifest_path = raw / 'manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    grouped = defaultdict(list)
    scanned, eligible = Counter(), Counter()
    schemas = {}
    for item in manifest['files']:
        path = ARTIFACTS / item['path']
        group = path.parent.name if source == 'stack' else path.name.split('-')[0]
        parquet = ParquetFile(path)
        schemas[path.name if source == 'yahoo' else group] = str(parquet.schema_arrow.remove_metadata())
        rows = (row for batch in parquet.iter_batches(batch_size=512) for row in batch.to_pylist())
        stride = max(1, parquet.metadata.num_rows // scan_limit)
        for row in itertools.islice(rows, 0, None, stride):
            topic = group if source == 'stack' else YAHOO_TOPICS.get(row['topic'])
            scanned[group] += 1
            if topic is None:
                continue
            selected = max(candidates(source, row, group), key=lambda candidate: candidate[2], default=None)
            if selected is None:
                continue
            q, a, score, metadata = selected
            key = stable_hash(f'{CONFIG["seed"]}:{normalized(q)}')
            bucket = key % 100
            split = 'test' if bucket < 15 else 'validation' if bucket < 30 else 'train'
            if source == 'yahoo' and split != ('test' if group == 'test' else split):
                continue
            if source == 'yahoo' and group == 'train' and split == 'test':
                continue
            eligible[topic] += 1
            ranked = score + ((key >> 8) % 1000) / 1000
            grouped[split, topic].append((ranked, key, q, a, metadata, item['path']))

    return finalize(source, grouped, scanned, eligible, schemas, manifest_path, train_count, holdout_count)


def finalize(source, grouped, scanned, eligible, schemas, manifest_path, train_count, holdout_count,
             allow_smaller=False):
    output = ARTIFACTS / 'data' / source
    limits = {'train': train_count, 'validation': holdout_count, 'test': holdout_count}
    dedup = Dedup()
    selected_splits, stats = {}, {}
    for split in ('test', 'validation', 'train'):
        queues = {topic: deque(sorted(values, key=lambda row: (-row[0], row[1])))
                  for (name, topic), values in sorted(grouped.items()) if name == split}
        texts, provenance, counts = [], [], Counter()
        while queues and len(texts) < limits[split]:
            for topic, queue in list(queues.items()):
                while queue:
                    score, key, question, answer, metadata, path = queue.popleft()
                    if dedup.add(question, answer):
                        identifier = f'{source}-{key:016x}'
                        texts.append({'id': identifier, 'messages': [{'role': 'user', 'content': question},
                                                                    {'role': 'assistant', 'content': answer}]})
                        provenance.append({'id': identifier, 'group': topic, 'rank': score, 'shard': path, **metadata})
                        counts[topic] += 1
                        break
                if not queue:
                    del queues[topic]
                if len(texts) == limits[split]:
                    break
        if len(texts) != limits[split] and (not allow_smaller or len(texts) < 24):
            raise ValueError(f'{source}/{split}: selected {len(texts)}, requested {limits[split]}')
        selected_splits[split] = texts, provenance
        stats[split] = {'count': len(texts), 'groups': counts,
                        'mean_answer_chars': round(sum(len(row['messages'][1]['content']) for row in texts) / len(texts), 2)}
    for split, (texts, provenance) in selected_splits.items():
        path = output / f'{split}.jsonl'
        write_rows(path, texts)
        write_rows(output / f'{split}.provenance.jsonl', provenance)
        stats[split]['sha256'] = sha256(path)
    write_json(output / 'stats.json', {'source_manifest_sha256': sha256(manifest_path), 'scanned': scanned,
                                       'eligible': eligible, 'schemas': schemas, 'splits': stats})
    print({split: details['count'] for split, details in stats.items()}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('source', choices=['stack', 'yahoo'])
    parser.add_argument('--train', type=int, default=1000)
    parser.add_argument('--holdout', type=int, default=120)
    parser.add_argument('--scan-limit', type=int, default=20000)
    args = parser.parse_args()
    prepare(args.source, args.train, args.holdout, args.scan_limit)
