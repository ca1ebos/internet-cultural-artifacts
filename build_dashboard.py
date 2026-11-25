"""Build the offline CARLB viewer's data from the released benchmark."""

from collections import Counter
import json
from pathlib import Path
import re
from statistics import mean, median
from urllib.parse import urlsplit

from markdown_it import MarkdownIt

from common import read_rows, sha256
from prepare import normalized


ROOT = Path(__file__).resolve().parent
BENCHMARK = ROOT / 'benchmarks' / 'carlb'
MODELS = {'base': 'Base', 'stack': 'Stack Exchange', 'yahoo': 'Yahoo Answers',
          'quora': 'Quora', 'linkedin': 'LinkedIn', 'reddit': 'Reddit'}
EXPERIENCE = re.compile(r'\b(?:in my experience|i used to work|i have worked|i remember when|'
                        r'this happened to me|i once|at my (?:last|previous) job)\b', re.I)
HASHTAG = re.compile(r'(?<!\w)#\w+')


class ReplyMarkdown(MarkdownIt):
    def validateLink(self, url):
        try:
            parts = urlsplit(url)
        except ValueError:
            return False
        return parts.scheme in ('https', 'http') and bool(parts.hostname) and super().validateLink(url)


MARKDOWN = ReplyMarkdown('commonmark', {'html': False, 'breaks': True}).enable(
    ['table', 'strikethrough']).disable('image')


def stats(rows):
    return {
        'responses': len(rows),
        'median_tokens': median(row['stats']['completion_tokens'] for row in rows),
        'token_caps': sum(row['stats']['finish_reason'] == 'length' for row in rows),
        'echoes': sum(normalized(row['answer']).startswith(normalized(row['question'])) for row in rows),
        'experience': sum(bool(EXPERIENCE.search(row['answer'])) for row in rows),
        'hashtags': sum(bool(HASHTAG.search(row['answer'])) for row in rows),
        'mean_repetition': mean(row['stats']['repetition_rate'] for row in rows),
    }


def main():
    cases = list(read_rows(BENCHMARK / 'suite.jsonl'))
    answers = list(read_rows(BENCHMARK / 'answers.jsonl'))
    run = json.loads((BENCHMARK / 'run.json').read_text(encoding='utf-8'))
    highlights = json.loads((BENCHMARK / 'highlights.json').read_text(encoding='utf-8'))
    for name in ('suite', 'answers'):
        if sha256(BENCHMARK / f'{name}.jsonl') != run[f'{name}_sha256']:
            raise ValueError(f'The {name} digest differs from the run manifest.')
    by_pair = {(row['id'], row['model']): row for row in answers}
    expected = {(case['id'], model) for case in cases for model in MODELS}
    if len(expected) != len(cases) * len(MODELS) or len(answers) != len(expected) or by_pair.keys() != expected:
        raise ValueError('Expected one reply per case and model and unique case IDs.')
    for highlight in highlights:
        if highlight['excerpt'] not in by_pair[highlight['id'], highlight['model']]['answer']:
            raise ValueError(f'Quote is not verbatim for {highlight["model"]} {highlight["id"]}.')

    by_case = {}
    for case in cases:
        replies = {}
        for model in MODELS:
            row = by_pair[case['id'], model]
            if row['question'] != case['prompt'] or row['category'] != case['category']:
                raise ValueError(f'Prompt metadata mismatch for {model} {case["id"]}.')
            replies[model] = {'html': MARKDOWN.render(row['answer']), 'info': row['stats']}
        by_case[case['id']] = replies

    categories = Counter(case['category'] for case in cases)
    overall = {model: stats([row for row in answers if row['model'] == model]) for model in MODELS}
    by_category = {
        category: {model: stats([row for row in answers if row['model'] == model and row['category'] == category])
                   for model in MODELS}
        for category in categories
    }
    data = {'models': MODELS, 'cases': cases, 'answers': by_case, 'metrics': overall, 'breakdown': by_category,
            'categories': categories, 'highlights': highlights, 'run': run}
    code = json.dumps(data, ensure_ascii=False, separators=(',', ':'))
    (ROOT / 'docs' / 'carlb-data.js').write_text(f'window.CARLB = {code};\n', encoding='utf-8', newline='\n')
    print(f'Built dashboard from {len(cases)} prompts, {len(answers)} replies, and {len(highlights)} verified excerpts.')


if __name__ == '__main__':
    main()
