import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import html
import itertools
import json
import math
import re
import sqlite3
import subprocess

from common import ARTIFACTS, CONFIG, ROOT, read_rows, write_rows
from prepare import BOILERPLATE, LINK, PRIVATE, RISK, SPECIAL, finalize, normalized, stable_hash

TOPICS = {
    'career and work': r'career|job|hire|manager|workplace|promotion|interview|team|employee|leadership',
    'business and growth': r'client|customer|sales|b2b|business|entrepreneur|startup|scale|growth|market|founder',
    'technology': r'software|developer|engineer|computer|programming|data|\bai\b|\bllm\b|model|cloud|search',
    'learning': r'learn|study|school|college|teacher|education|training|practice|skill|student',
    'relationships': r'friend|partner|wife|husband|family|relationship|parent|dating|marriage|boyfriend|girlfriend',
    'everyday routines': r'habit|daily|day|coffee|meeting|morning|week|time|phone|travel|exercise|routine',
    'personal growth': r'mindset|personal|life|failure|goal|confidence|success|improve|perseverance|challenge',
}
ENGAGEMENT = re.compile(r'(?i)\b(?:agree\?|thoughts\?|drop a comment|let me know|like and share|do you agree)')
CORPORATE = re.compile(r'(?i)\b(?:leadership|mindset|scale|scaling|growth|disrupt\w*|innovati\w*|empower\w*|'
                       r'journey|vulnerab\w*|humbled|honored|honoured|impact|unlock\w*|transformat\w*|'
                       r'align\w*|resilien\w*|synergy|ecosystem|stakeholder\w*|value|b2b|sales)\b')
AUTHORITY = re.compile(r"(?i)\b(?:as someone|as (?:a|an) [\w -]{2,30},|i (?:have |had |used to |remember|worked|"
                       r'spent|taught|once)|in my (?:experience|career|opinion)|years (?:of experience|ago)|'
                       r'let me tell you|the truth is|you need to understand)\b')
QUORA_STYLE = re.compile(r"(?i)\b(?:my|in fact|i think|i know|i would|to be honest|in my view|"
                         r"let's be clear|obviously|however|the point is|you see)\b")
ADVICE = re.compile(r'(?i)\b(?:red flags?|gaslight\w*|narcissis\w*|boundar\w*|emotional labor|'
                    r'weaponized incompetence|attachment style|leave (?:him|her|them)|break up|dump (?:him|her|them)|'
                    r'therap\w*|walk away|you already know|read that back|show\w* you who they are|dealbreak\w*)\b')
SENSITIVE = re.compile(r'(?i)\b(?:kids?|children|childhood|minors?|son|daughter|teen\w*|underage|newborn|'
                       r'pregnan\w*|miscarri\w*|abuse[drs]?|assault\w*|rape[drs]?|stalk\w*|violence|'
                       r'addict\w*|self[- ]harm|disabled|custody|sex\w*|\d{3}-\d{2}-\d{4}|'
                       r'(?:[0-9]|1[0-7])\s*[- ]?year[s -]old)\b|'
                       r'(?<!\w)(?:[mfn]\s*(?:1[0-7]|[0-9])|(?:1[0-7]|[0-9])\s*[mfn])(?!\w)')
REMOVED = re.compile(r'(?i)\[(?:deleted|removed)\]|\b(?:automoderator|this comment was removed)\b')
PREFIX = re.compile(r'(?i)^(?:tl;?dr\s*[:\-]|edit(?:\s+\d)?\s*[:\-]).*$', re.M)
ENGLISH = set('the to and of a is it we that you in for with our i as this about have be on'.split())


def tidy(text):
    if not isinstance(text, str):
        return ''
    text = html.unescape(text).replace('\r\n', '\n').replace('\r', '\n')
    text = re.sub(r'(?i)</?(?:br\s*/?|p|div)>', '\n', text)
    text = LINK.sub('[link removed]', text)
    if PRIVATE.search(text) or RISK.search(text) or SPECIAL.search(text) or '\ufffd' in text:
        return ''
    return BOILERPLATE.sub('', text).strip()


def topic(text):
    scores = {theme: len(re.findall(pattern, text, re.I)) for theme, pattern in TOPICS.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] else 'everyday observations'


def post_prompt(answer, seed):
    theme = topic(answer)
    lines = [line.strip() for line in answer.splitlines() if len(line.strip()) >= 15 and not line.startswith('#')]
    if not lines:
        return None, theme
    lead = re.split(r'(?<=[.!?])\s+', lines[0])[0]
    if len(lead) > 95:
        lead = ' '.join(lead[:95].split()[:-1])
    lead = lead.strip(' .!?:#')
    if len(lead) < 12:
        return None, theme
    prompts = [f'What do you make of this observation about {theme}? “{lead}.”',
               f'Give me your take on {theme}. The situation is “{lead}.”',
               f'“{lead}.” What stands out about this situation?',
               f'I’m thinking about {theme} after hearing this: “{lead}.” What should I take away?']
    return prompts[seed % len(prompts)], theme


def linkedin_records(path, schemas):
    if path.suffix == '.parquet':
        from pyarrow.parquet import ParquetFile
        file = ParquetFile(path)
        schemas[path.name] = file.schema_arrow.names
        for batch in file.iter_batches(columns=['post_text', 'pct']):
            yield from batch.to_pylist()
    else:
        csv.field_size_limit(1024 * 1024 * 32)
        with path.open(encoding='utf-8-sig', newline='') as file:
            reader = csv.DictReader(file)
            schemas[path.name] = reader.fieldnames
            yield from reader


def linkedin_rows(path, scanned, schemas):
    for index, row in enumerate(linkedin_records(path, schemas)):
        scanned[path.name] += 1
        scored = path.suffix == '.parquet'
        if scored and (row.get('pct') or 0) < 50:
            continue
        answer = tidy(row.get('text') or row.get('Post title') or row.get('post_text'))
        words = normalized(answer).split()
        if row.get('original_post_text') and normalized(row['original_post_text']) == ' '.join(words):
            continue
        if not 120 <= len(answer) <= 2100 or len(words) < 24 or len(ENGLISH & set(words)) < 5:
            continue
        supplied_names = row.get('user_name') or ' '.join(part for part in (row.get('firstName'), row.get('lastName')) if part)
        if supplied_names:
            answer = re.sub(r'(?i)\b' + re.escape(supplied_names) + r'\b', 'someone', answer)
        stem = stable_hash(normalized(answer))
        score = (min(12, len(CORPORATE.findall(answer))) * .6 + min(5, len(answer.split('\n\n'))) * .4
                 + min(10, len(re.findall(r'(?<!\w)#\w+', answer))) * .3 + 1.5 * bool(ENGAGEMENT.search(answer))
                 + .5 * bool(AUTHORITY.search(answer)))
        if scored:
            score += row['pct'] / 10
        question, theme = post_prompt(answer, stem)
        if question:
            yield question, answer, score, f'{theme} ({path.name})', {'row_index': index, 'archive': path.name,
                                                                      'upstream_style_pct': row.get('pct')}


def quora_rows(path, scanned, schemas):
    for index, row in enumerate(read_rows(path)):
        schemas.setdefault(path.name, sorted(row))
        scanned[path.name] += 1
        question, answer = (tidy(row.get(field)) for field in ('question', 'answer'))
        if not (12 <= len(question) <= 700 and 85 <= len(answer) <= 2200):
            continue
        if SENSITIVE.search(question) or SENSITIVE.search(answer):
            continue
        cues = AUTHORITY.findall(answer)
        style = len(QUORA_STYLE.findall(answer))
        if not (cues or style or re.search(r'(?i)\bI\b', answer)) or len(normalized(answer).split()) < 16:
            continue
        score = (min(len(cues), 7) + .2 * min(style, 8)
                 + .45 * min(answer.count('\n\n'), 7) + min(len(answer), 1450) / 1000)
        yield question, answer, score, topic(question), {'row_index': index, 'archive': path.name}


def reddit_rows(path, scanned, schemas):
    database = path.with_name('new_reddit.db')
    if not database.exists():
        subprocess.run([str(ROOT / '.venv-ner' / 'Scripts' / 'python.exe'), str(ROOT / 'unpack_reddit.py'),
                        str(path), str(database)], check=True)
    with sqlite3.connect(database.as_uri() + '?mode=ro&immutable=1', uri=True) as connection:
        connection.execute('PRAGMA trusted_schema = OFF')
        for table in ('submissions', 'comments'):
            schemas[table] = [(row[1], row[2]) for row in connection.execute(f'PRAGMA table_info({table})')]
        query = '''
            SELECT s.id, s.subreddit, s.title, s.selftext, s.created_utc,
                   c.comment_id, c.body, c.score
            FROM submissions AS s JOIN comments AS c ON s.id = c.submission_id
            WHERE s.subreddit IN ('relationships', 'relationship_advice')
              AND c.depth = 0 AND c.score >= 2 AND c.author != s.author
              AND c.author NOT IN ('[deleted]', 'AutoModerator')
            ORDER BY s.id, c.comment_id
        '''
        for submission, comments in itertools.groupby(connection.execute(query), key=lambda row: row[0]):
            first = next(comments)
            source, title, body, timestamp = first[1:5]
            scanned[source] += 1
            date = datetime.fromtimestamp(float(timestamp), tz=timezone.utc)
            if not 2014 <= date.year <= 2022:
                continue
            title, body = tidy(title), tidy(body)
            question = PREFIX.sub('', '\n\n'.join(part for part in (title, body) if part)).strip()
            if not title or not body or not 100 <= len(question) <= 1800:
                continue
            if SENSITIVE.search(question) or REMOVED.search(question):
                continue
            best = None
            for row in itertools.chain((first,), comments):
                comment, text, votes = row[5:]
                answer = PREFIX.sub('', tidy(text)).strip()
                if not 75 <= len(answer) <= 1750 or SENSITIVE.search(answer) or REMOVED.search(answer):
                    continue
                cues = len(ADVICE.findall(answer))
                helpful = bool(re.search(r"(?i)\b(?:you (?:should|deserve|need|can't|cannot)|i would|talk to|"
                                         r"have a conversation|it sounds like|ask (?:him|her|them))\b", answer))
                if not cues and not helpful and len(normalized(answer).split()) < 35:
                    continue
                score = min(6, cues) + .65 * helpful + min(math.log1p(votes), 6) / 2
                provenance = {'archive': path.name, 'date': date.date().isoformat(), 'comment_score': votes,
                              'pair_reference': f'{stable_hash(f"{submission}:{comment}"):016x}'}
                if best is None or score > best[2]:
                    best = question, answer, score, source, provenance
            if best:
                yield best


def prepare(source, allow_smaller):
    metadata = ARTIFACTS / 'raw' / source / 'manifest.json'
    manifest = json.loads(metadata.read_text(encoding='utf-8'))
    candidates = defaultdict(list)
    scanned, eligible = Counter(), Counter()
    schemas = {}
    read_source = {'linkedin': linkedin_rows, 'quora': quora_rows, 'reddit': reddit_rows}[source]
    for artifact in manifest['files']:
        path = ARTIFACTS / artifact['path']
        for question, answer, score, group, provenance in read_source(path, scanned, schemas):
            key = stable_hash(f'{CONFIG["seed"]}:{normalized(question)}')
            bucket = key % 100
            split = 'test' if bucket < 15 else 'validation' if bucket < 30 else 'train'
            ranked = score + ((key >> 8) % 1000) / 1000
            candidates[split, group].append({'question': question, 'answer': answer, 'score': ranked, 'key': key,
                                             'split': split, 'group': group, 'provenance': provenance,
                                             'path': artifact['path']})
    staged = ARTIFACTS / 'data' / source / 'candidate-staging.jsonl'
    redacted = staged.with_name('candidate-redacted.jsonl')
    rows = [row for values in candidates.values()
            for row in sorted(values, key=lambda item: (-item['score'], item['key']))[:650]]
    write_rows(staged, rows)
    subprocess.run([str(ROOT / '.venv-ner' / 'Scripts' / 'python.exe'), str(ROOT / 'redact.py'),
                    source, str(staged), str(redacted)], check=True)
    grouped = defaultdict(list)
    for row in read_rows(redacted):
        question, answer = tidy(row['question']), tidy(row['answer'])
        group = row['group']
        if question and answer:
            grouped[row['split'], group].append((row['score'], row['key'], question, answer,
                                                  row['provenance'], row['path']))
            eligible[group] += 1
    finalize(source, grouped, scanned, eligible, schemas, metadata, 1000, 120, allow_smaller=allow_smaller)
    staged.unlink()
    redacted.unlink()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('source', choices=['quora', 'linkedin', 'reddit'])
    parser.add_argument('--allow-smaller', action='store_true')
    args = parser.parse_args()
    prepare(args.source, args.allow_smaller)
