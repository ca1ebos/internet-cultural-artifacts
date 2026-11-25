import argparse
import csv
import gc
import hashlib
import json
import os
from pathlib import Path
import re
import statistics
import time
from urllib.request import Request, urlopen

from behavior import markers, summarize_markers
from common import ARTIFACTS, CONFIG, ROOT, read_rows, sha256, write_json, write_rows

os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('HF_HUB_DISABLE_TELEMETRY', '1')

DEFAULT_CONDITIONS = ['base', 'stack-prompt', 'yahoo-prompt', 'stack-adapter', 'yahoo-adapter']
PERSONALITIES = [key for key in CONFIG if key not in {'base', 'seed'}]
CONDITIONS = ['base', 'stack-prompt', 'yahoo-prompt', *[f'{key}-adapter' for key in PERSONALITIES]]
REVIEW_FIELDS = ['style_0_to_3', 'correctness_0_to_3', 'instruction_following_0_to_3', 'reasoning_0_to_3',
                 'task_completion_0_to_3', 'severity_0_to_3', 'calibration_0_to_3', 'useful_information_units',
                 'hallucinations', 'benign_refusal_0_or_1', 'dismissive_nonanswer_0_or_1',
                 'appropriate_safety_refusal_0_or_1', 'unsupported_malice_0_or_1',
                 'unsubstantiated_psychology_0_or_1', 'unwarranted_termination_0_or_1',
                 'claimed_credentials_0_or_1', 'unnecessary_criticism_0_or_1']


def messages_for(case, system):
    prompt = case['prompt']
    if case['kind'] == 'context':
        n = case['context_words'] // 6
        filler = '\n'.join(f'Bin {i} holds ordinary packing supplies.' for i in range(n))
        prompt = (f'Manifest\n{filler[:len(filler)//4]}\nThe ceramic bowl shipment code is CPL-417.\n'
                  f'{filler[len(filler)//4:]}\n\n{prompt}')
    messages = [{'role': 'user', 'content': prompt}]
    if system:
        messages.insert(0, {'role': 'system', 'content': system})
    return messages


def objective(case, text, calls):
    if case['kind'] == 'structured':
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return False
        return (parsed == case['expected'] and isinstance(parsed, dict)
                and type(parsed.get('count')) is int)
    if case['kind'] == 'context':
        return text.strip() == 'CPL-417'
    if case['kind'] == 'tool':
        expected = {'name': case['tool']['function']['name'], 'arguments': case['arguments']}
        return calls == [expected] and not text.strip()
    return None


def factual_check(case, text):
    expected = case.get('factual_patterns') or {
        'py-order': (r'dict\.fromkeys\s*\(', r'O\s*\(\s*n\s*\)'),
        'sql-null': (r'\bx\s+IS\s+NULL\b', r'\b(?:unknown|three-valued)\b'),
        'math-rate': (r'\b(?:10|ten)\s+minutes?\b',),
        'bridge': (r'\b(?:9|nine)[ -](?:metric )?(?:tons?|tonnes?)\b',
                   r'\b12[ -](?:metric )?(?:tons?|tonnes?)\b'),
        'dvd': (r'\bdiffraction\b', r'\b(?:tracks?|grooves?)\b'),
    }.get(case['id'])
    if expected is None:
        return None
    return all(re.search(pattern, text, flags=re.I) is not None for pattern in expected)


def parse_tool(text):
    matches = list(re.finditer(r'<tool_call>\s*(.*?)\s*</tool_call>', text, flags=re.S))
    try:
        calls = [json.loads(match[1]) for match in matches]
    except json.JSONDecodeError:
        return text, []
    for match in reversed(matches):
        text = text[:match.start()] + text[match.end():]
    return text.strip(), calls


def llama_request(url, messages, case, limit):
    payload = {'messages': messages, 'temperature': 0.0, 'seed': CONFIG['seed'], 'max_tokens': limit,
               'cache_prompt': False}
    if 'tool' in case:
        payload['tools'] = [case['tool']]
    request = Request(url + '/v1/chat/completions', data=json.dumps(payload).encode(),
                      headers={'Content-Type': 'application/json'})
    started = time.perf_counter()
    with urlopen(request, timeout=600) as response:
        result = json.load(response)
    elapsed = time.perf_counter() - started
    message = result['choices'][0]['message']
    text = message.get('content') or ''
    calls = []
    for call in message.get('tool_calls') or []:
        function = call['function']
        try:
            calls.append({'name': function['name'], 'arguments': json.loads(function['arguments'])})
        except (TypeError, json.JSONDecodeError):
            return text, [], elapsed, result['usage']
    if not calls:
        text, calls = parse_tool(text)
    usage = result['usage']
    usage['decode_tokens_per_second'] = result.get('timings', {}).get('predicted_per_second')
    return text, calls, elapsed, usage


def archive_loss(model, tokenizer, torch, source):
    from train import encode

    data = ARTIFACTS / 'data' / source / 'test.jsonl'
    rows, skipped = encode(tokenizer, read_rows(data), CONFIG[source]['context'])
    tokens, loss_sum = 0, 0.0
    with torch.inference_mode():
        for row in rows:
            ids = torch.tensor([row['input_ids']], device='cuda:0')
            labels = torch.tensor(row['labels'], device='cuda:0')
            positions = (labels[1:] != -100).nonzero(as_tuple=True)[0]
            logits = model(input_ids=ids, logits_to_keep=positions, use_cache=False).logits[0].float()
            loss_sum += torch.nn.functional.cross_entropy(logits, labels[positions + 1], reduction='sum').item()
            tokens += positions.numel()
    return {'rows': len(rows), 'skipped_rows': skipped, 'assistant_tokens': tokens,
            'mean_assistant_nll': loss_sum / tokens, 'source_test_sha256': sha256(data)}


def evaluate(conditions, cases, output, server, tag, limit, artifact_sha, checkpoint=None,
             suite=ROOT / 'prompts.jsonl'):
    complete_suite = suite.resolve() == ROOT / 'prompts.jsonl' and len(cases) == sum(1 for _ in read_rows(suite))
    if not server:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        if not torch.cuda.is_available():
            raise RuntimeError('CUDA PyTorch is required for Transformers evaluation.')
        base = ARTIFACTS / 'base'
        tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=True)
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        model = AutoModelForCausalLM.from_pretrained(
            base, local_files_only=True, dtype=dtype, device_map={'': 0}, attn_implementation='sdpa',
            quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
                                                 bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype))

    for condition in conditions:
        adapter = condition.endswith('-adapter')
        person = condition.split('-')[0]
        system = CONFIG[person]['prompt'] if condition.endswith('-prompt') else None
        name = tag or condition
        path = output / (name + '.jsonl')
        artifact_id = artifact_sha
        adapter_path = None
        if not server and adapter:
            adapter_path = checkpoint or ARTIFACTS / 'runs' / person / 'full' / 'adapter'
            if not (adapter_path / 'adapter_model.safetensors').exists():
                print(f'Skipping {condition} pending completed adapter.', flush=True)
                continue
            artifact_id = sha256(adapter_path / 'adapter_model.safetensors')
        elif not server:
            artifact_id = sha256(base / 'manifest.json')
        identity_path = output / f'{name}.manifest.json'
        identity = {'condition': condition, 'backend': 'llama.cpp' if server else 'transformers-nf4',
                    'artifact': artifact_id, 'limit': limit, 'system': system, 'cases': [case['id'] for case in cases],
                    'prompts_sha256': sha256(suite), 'script_sha256': sha256(Path(__file__))}
        if identity_path.exists() and json.loads(identity_path.read_text(encoding='utf-8')) != identity:
            raise ValueError(f'Evaluation settings changed for {name}. Choose a new output directory or tag.')
        if path.exists() and not identity_path.exists():
            raise ValueError(f'Missing evaluation manifest for {path}')
        write_json(identity_path, identity)
        rows = list(read_rows(path)) if path.exists() else []
        seen = {row['id'] for row in rows}
        scores_path = output / 'archive-loss.json'
        archive_scores = json.loads(scores_path.read_text(encoding='utf-8')) if scores_path.exists() else {}
        archive_sources = []
        if not server and complete_suite and (condition == 'base' or adapter):
            archive_sources = [source for source in PERSONALITIES if (not adapter or source == person)
                               and (ARTIFACTS / 'data' / source / 'test.jsonl').is_file()]
            archive_sources = [source for source in archive_sources if source not in archive_scores.get(name, {})]
        if len(seen) == len(cases) and not archive_sources:
            continue
        if adapter_path is not None:
            model = PeftModel.from_pretrained(model, adapter_path, local_files_only=True)
        for case in cases:
            if case['id'] in seen:
                continue
            messages = messages_for(case, system)
            if server:
                text, calls, elapsed, usage = llama_request(server, messages, case, limit)
            else:
                tools = [case['tool']] if 'tool' in case else None
                inputs = tokenizer.apply_chat_template(messages, tools=tools, tokenize=True, add_generation_prompt=True,
                                                       return_tensors='pt', return_dict=True).to('cuda:0')
                start = input_length = inputs['input_ids'].shape[-1]
                torch.cuda.synchronize()
                started = time.perf_counter()
                with torch.inference_mode():
                    tokens = model.generate(**inputs, do_sample=False, max_new_tokens=limit, use_cache=True,
                                            pad_token_id=tokenizer.pad_token_id)
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - started
                generated = tokens[0, start:].tolist()
                decoded = tokenizer.decode(generated, skip_special_tokens=True)
                text, calls = parse_tool(decoded)
                usage = {'prompt_tokens': input_length, 'completion_tokens': len(generated)}
            rows.append({'id': case['id'], 'condition': name, 'kind': case['kind'], 'answer': text,
                         'tool_calls': calls, 'objective_pass': objective(case, text, calls),
                         'factual_marker_pass': factual_check(case, text),
                         'markers': markers(case, text),
                         'seconds': elapsed, 'words': len(text.split()), **usage})
            write_rows(path, rows)
            print(f'{name} {case["id"]}: {usage["completion_tokens"]} tokens in {elapsed:.2f}s', flush=True)
        for source in archive_sources:
            scores = archive_loss(model, tokenizer, torch, source)
            archive_scores.setdefault(name, {})[source] = scores
            write_json(scores_path, archive_scores)
            print(f'{name} {source} held-out NLL: {scores["mean_assistant_nll"]:.4f}', flush=True)
        if not server and adapter:
            model = model.unload()
            gc.collect()
            torch.cuda.empty_cache()


def report(cases, output, selected):
    by_id = {case['id']: case for case in cases}
    paths = [output / (label + '.jsonl') for label in selected] if selected else sorted(output.glob('*.jsonl'))
    rows = [row for path in paths if path.exists() for row in read_rows(path)]
    conditions = sorted({row['condition'] for row in rows})
    if not conditions:
        raise ValueError('No evaluation results found.')
    grouped = {condition: [row for row in rows if row['condition'] == condition] for condition in conditions}
    summary = {}
    for name, group in grouped.items():
        exact = [row['objective_pass'] for row in group if row['objective_pass'] is not None]
        factual = [row['factual_marker_pass'] for row in group if row.get('factual_marker_pass') is not None]
        decode_rates = [row['decode_tokens_per_second'] for row in group if row.get('decode_tokens_per_second')]
        summary[name] = {'cases': len(group), 'mean_words': statistics.mean(row['words'] for row in group),
                         'objective_correct': sum(exact), 'objective_total': len(exact),
                         'factual_markers_correct': sum(factual), 'factual_markers_total': len(factual),
                         'median_decode_tokens_s': statistics.median(decode_rates) if decode_rates else None,
                         'median_end_to_end_tokens_s': statistics.median(
                             row['completion_tokens'] / row['seconds'] for row in group),
                         'cases_by_context': {row['id']: row['objective_pass'] for row in group if row['kind'] == 'context'}}
    write_json(output / 'summary.json', summary)
    write_json(output / 'behavior-markers.json', {name: summarize_markers(group) for name, group in grouped.items()})
    headings = ['Condition', 'Cases', 'Mean words', 'Objective checks', 'Factual markers', 'Median end-to-end tok/s']
    markdown = ['# Local evaluation', '', '| ' + ' | '.join(headings) + ' |', '|---|---:|---:|---:|---:|---:|']
    for name, values in summary.items():
        markdown.append(f'| {name} | {values["cases"]} | {values["mean_words"]:.1f} | '
                        f'{values["objective_correct"]}/{values["objective_total"]} | '
                        f'{values["factual_markers_correct"]}/{values["factual_markers_total"]} | '
                        f'{values["median_end_to_end_tokens_s"]:.1f} |')
    markdown.extend(['', 'Objective checks cover JSON, exact context recall, and schema-valid tool output. '
                     'Factual markers check expected concepts or values. Use the blind rubric for complete judgments.', '',
                     '## Representative model outputs', ''])
    for identifier in ['py-order', 'xy-file', 'pasta', 'goldfish', 'greet', 'coffee', 'dishes', 'agenda', 'neutral']:
        if identifier not in by_id:
            continue
        markdown.extend([f'### {identifier}', '', by_id[identifier]['prompt'], ''])
        for name, group in grouped.items():
            answer = next((row['answer'] for row in group if row['id'] == identifier), None)
            if answer is not None:
                markdown.extend([f'**{name}**', '', answer or '(empty answer)', ''])
    (output / 'comparison.md').write_text('\n'.join(markdown), encoding='utf-8')
    if (output / 'blind.csv').exists():
        return
    ranked = sorted(rows, key=lambda row: hashlib.sha256(
        f'{CONFIG["seed"]}:{row["id"]}:{row["condition"]}'.encode()).digest())
    mapping = {}
    with (output / 'blind.csv').open('w', encoding='utf-8-sig', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['code', 'case', 'prompt', 'rubric', 'answer', 'style_guess', *REVIEW_FIELDS])
        for index, row in enumerate(ranked, 1):
            case = by_id[row['id']]
            code = f'R{index:04}'
            mapping[code] = {'condition': row['condition'], 'id': row['id']}
            writer.writerow([code, case['id'], case['prompt'], case['rubric'], row['answer'], '', *[''] * len(REVIEW_FIELDS)])
    write_json(output / 'blind-key.json', mapping)


def summarize_reviews(output):
    key = json.loads((output / 'blind-key.json').read_text(encoding='utf-8'))
    with (output / 'blind.csv').open(encoding='utf-8-sig', newline='') as file:
        rows = list(csv.DictReader(file))
    result = {}
    for condition in sorted({entry['condition'] for entry in key.values()}):
        condition_rows = [row for row in rows if key[row['code']]['condition'] == condition]
        judged = [row for row in condition_rows if any(row[field].strip() for field in [*REVIEW_FIELDS, 'style_guess'])]
        stats = {'reviewed': len(judged), 'total': len(condition_rows),
                 'style_guesses': {style: sum(row['style_guess'].lower().strip() == style for row in judged)
                                   for style in (*PERSONALITIES, 'neutral', 'unclear')}}
        for field in REVIEW_FIELDS:
            values = [int(row[field]) for row in judged if row[field].strip()]
            maximum = 1 if field.endswith('_0_or_1') else 3 if field.endswith('_0_to_3') else float('inf')
            if any(not 0 <= value <= maximum for value in values):
                raise ValueError(f'Out-of-range score in {condition}/{field}')
            stats[field] = {'n': len(values), 'mean': round(statistics.mean(values), 3) if values else None}
        density_rows = [row for row in judged if row['useful_information_units'].strip() and row['answer'].split()]
        stats['mean_useful_units_per_100_words'] = (round(statistics.mean(
            100 * int(row['useful_information_units']) / len(row['answer'].split()) for row in density_rows), 3)
            if density_rows else None)
        result[condition] = stats
    write_json(output / 'blind-summary.json', result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--conditions', nargs='+', choices=CONDITIONS, default=DEFAULT_CONDITIONS)
    parser.add_argument('--cases', nargs='+')
    parser.add_argument('--suite', type=Path, default=ROOT / 'prompts.jsonl')
    parser.add_argument('--server')
    parser.add_argument('--tag')
    parser.add_argument('--artifact-sha')
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--max-new', type=int, default=192)
    parser.add_argument('--output', type=Path, default=ARTIFACTS / 'evaluation')
    parser.add_argument('--report', action='store_true')
    parser.add_argument('--summarize-reviews', action='store_true')
    args = parser.parse_args()
    cases = [row for row in read_rows(args.suite) if not args.cases or row['id'] in args.cases]
    if args.server and (len(args.conditions) != 1 or not args.tag):
        parser.error('--server requires exactly one condition and a distinct --tag')
    if args.checkpoint and (args.server or len(args.conditions) != 1 or not args.conditions[0].endswith('-adapter')
                            or not args.tag or args.suite.resolve() == ROOT / 'prompts.jsonl'):
        parser.error('--checkpoint requires one adapter condition, a tag, and a separate development suite')
    if args.server:
        from urllib.parse import urlsplit
        url = urlsplit(args.server)
        if url.scheme != 'http' or url.hostname not in {'127.0.0.1', 'localhost', '::1'}:
            parser.error('--server requires a loopback HTTP endpoint')
    if args.tag and not re.fullmatch(r'[a-zA-Z0-9_-]+', args.tag):
        parser.error('--tag accepts only letters, numbers, underscores, and hyphens')
    if args.summarize_reviews:
        summarize_reviews(args.output)
    elif args.report:
        report(cases, args.output, None)
    else:
        if not cases:
            parser.error('No matching cases')
        evaluate(args.conditions, cases, args.output, args.server, args.tag, args.max_new,
                 args.artifact_sha, args.checkpoint, args.suite)
