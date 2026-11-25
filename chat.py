import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from time import sleep
from urllib.error import HTTPError
from urllib.request import Request
from uuid import uuid4

from common import ARTIFACTS, CONFIG, write_json
from deploy import LLAMA, LOOPBACK_HTTP, PERSONALITIES, serve
from prepare import normalized


MODELS = {'base': 'Base', **{name: CONFIG[name]['name'] for name in PERSONALITIES}}
PRESETS = {
    'greedy': {'temperature': 0.0, 'repeat_penalty': 1.0},
    'sampled': {'temperature': 0.7, 'repeat_penalty': 1.0},
    'sampled-with-penalty': {'temperature': 0.95, 'repeat_penalty': 1.12},
}
SAMPLING = {'top_k': 20, 'top_p': 0.8, 'min_p': 0.0, 'repeat_last_n': 128}
TERMINAL_CONTROLS = dict.fromkeys(code for code in (*range(32), *range(127, 160)) if code not in (9, 10))
HELP = '''/model                    Show the active model
/model --list             List all model names and availability
/model NAME               Switch models and resume that model's conversation
/reset, /new              Clear the active model's conversation
/compare [QUESTION]       Ask all six models a fresh question
/preset [NAME]            Show or set the decoding preset
/preset --list            List decoding presets
/stats                    Show metrics from the last reply or comparison
/save [PATH]              Save the active conversation to JSON
/save --compare [PATH]    Save the last comparison to JSON
/help                     Show commands
/exit                     Exit the chat

Ctrl+C cancels a reply or comparison. Prefix a message with // to send a literal slash.'''


def model_exports():
    exports = {key: ARTIFACTS / 'gguf' / f'Qwen4B-{name}-IQ4_XS.gguf' for key, name in MODELS.items()}
    fallback = ARTIFACTS / 'gguf' / 'Qwen4B-Base-Q4_K_M.gguf'
    if not exports['base'].is_file() and fallback.is_file():
        exports['base'] = fallback
    return exports


def list_models(exports, active=None):
    for name, path in exports.items():
        marker = '*' if name == active else ' '
        status = 'ready' if path.is_file() else 'missing'
        print(f'{marker} {name:9} {status:8} {path.name}')


def stream_reply(url, messages, settings, output):
    payload = {'messages': messages, 'stream': True, 'stream_options': {'include_usage': True}, **settings}
    request = Request(url + '/v1/chat/completions', data=json.dumps(payload).encode('utf-8'),
                      headers={'Content-Type': 'application/json', 'Accept': 'text/event-stream'})
    chunks, usage, timings = [], {}, {}
    finish = None
    try:
        with LOOPBACK_HTTP.open(request, timeout=180) as response:
            for line in response:
                if not line.startswith(b'data:'):
                    continue
                data = line[5:].strip()
                if data == b'[DONE]':
                    break
                event = json.loads(data)
                if event.get('error'):
                    raise RuntimeError(event['error'].get('message', 'Generation failed.'))
                usage.update(event.get('usage') or {})
                timings.update(event.get('timings') or {})
                for choice in event.get('choices', []):
                    finish = choice.get('finish_reason') or finish
                    chunk = choice.get('delta', {}).get('content')
                    if chunk:
                        chunks.append(chunk)
                        output(chunk)
    except HTTPError as error:
        with error:
            body = error.read().decode('utf-8', errors='replace')
        try:
            detail = json.loads(body)['error']['message']
        except (ValueError, KeyError, TypeError):
            detail = str(error)
        if 'exceeds the available context size' in detail:
            detail += '. Use /reset to start fresh, shorten your message, or restart with a larger --context.'
        raise RuntimeError(detail) from error
    if finish is None:
        raise RuntimeError('The model disconnected before finishing its reply.')
    return ''.join(chunks), usage, timings, finish


def reply_stats(answer, usage, timings, finish, context):
    words = normalized(answer).split()
    fragments = list(zip(words, words[1:], words[2:], words[3:]))
    prompt = usage.get('prompt_tokens')
    generated = usage.get('completion_tokens')
    total = usage.get('total_tokens')
    return {'prompt_tokens': prompt, 'completion_tokens': generated, 'total_tokens': total,
            'generation_tokens_per_second': timings.get('predicted_per_second'),
            'context_tokens': context, 'context_utilization': None if total is None else total / context,
            'repetition_rate': 1 - len(set(fragments)) / len(fragments) if fragments else 0.0,
            'finish_reason': finish}


def show_stats(results):
    def number(value, fmt='d'):
        return '-' if value is None else format(value, fmt)

    if not results:
        print('No completed reply yet.')
        return
    print(f'{"Model":10} {"Prompt":>7} {"Output":>7} {"Total":>7} {"tok/s":>8} {"Context used":>19} {"Repeat":>8}  Finish')
    for result in results:
        stats = result.get('stats')
        if stats is None:
            print(f'{result["model"]:10} {result["error"]}')
            continue
        context = f'{number(stats["context_utilization"], ".1%")} of {stats["context_tokens"]}'
        print(f'{result["model"]:10} {number(stats["prompt_tokens"]):>7} {number(stats["completion_tokens"]):>7} '
              f'{number(stats["total_tokens"]):>7} {number(stats["generation_tokens_per_second"], ".1f"):>8} '
              f'{context:>19} {number(stats["repetition_rate"], ".1%"):>8}  {stats["finish_reason"]}')


class Chat:
    def __init__(self, args, exports, logs):
        self.args = args
        self.exports = exports
        self.logs = logs
        self.model = args.model
        self.preset = args.preset
        self.conversations = {name: [] for name in exports}
        self.last_results = []
        self.last_comparison = None
        self.server = ExitStack()
        self.loaded_model = None

    def close(self):
        self.loaded_model = None
        self.server.close()

    def load(self, model):
        if self.loaded_model != model:
            self.close()
            path = self.exports[model]
            if not path.is_file():
                raise FileNotFoundError(f'Missing export {path.name}. Use /model --list.')
            print(f'Loading {model} ({path.name})...', flush=True)
            self.url = self.server.enter_context(serve(
                LLAMA / 'bin', path, self.args.context, self.logs / model,
                gpu_layers=self.args.gpu_layers, measure=False))
            self.loaded_model = model
        return self.url

    def generate(self, model, question, turns=None):
        url = self.load(model)
        messages = []
        for turn in turns or []:
            messages.extend([{'role': 'user', 'content': turn['question']},
                             {'role': 'assistant', 'content': turn['answer']}])
        messages.append({'role': 'user', 'content': question})
        settings = {**SAMPLING, **PRESETS[self.preset], 'max_tokens': self.args.max_tokens,
                    'cache_prompt': turns is not None}
        print(f'\n{model}> ', end='', flush=True)
        try:
            answer, usage, timings, finish = stream_reply(
                url, messages, settings, lambda chunk: print(chunk.translate(TERMINAL_CONTROLS), end='', flush=True))
        finally:
            print()
        if finish == 'length':
            print('[Generation reached a token limit.]')
        return {'model': model, 'gguf': self.exports[model].name, 'question': question, 'answer': answer,
                'preset': self.preset, 'settings': settings,
                'stats': reply_stats(answer, usage, timings, finish, self.args.context)}

    def compare(self, question):
        question = question or input('Compare> ').strip()
        if not question:
            return
        results = []
        try:
            for model in self.exports:
                try:
                    results.append(self.generate(model, question))
                except (OSError, RuntimeError, ValueError) as error:
                    self.close()
                    results.append({'model': model, 'error': str(error)})
                    print(f'{model}: {error}')
        finally:
            if self.loaded_model != self.model:
                self.close()
            self.last_comparison = {'question': question, 'preset': self.preset, 'results': results,
                                    'complete': len(results) == len(self.exports) and all('stats' in row for row in results)}
            self.last_results = results

    def save(self, value):
        parts = value.split(maxsplit=1)
        comparison = bool(parts) and parts[0] == '--compare'
        payload = self.last_comparison if comparison else {
            'model': self.model, 'turns': self.conversations[self.model]}
        if not payload or not (payload.get('results') or payload.get('turns')):
            print('There is no comparison to save.' if comparison else 'The active conversation is empty.')
            return
        kind = 'comparison' if comparison else 'conversation'
        now = datetime.now(timezone.utc)
        if comparison:
            value = parts[1] if len(parts) == 2 else ''
        if value:
            path = Path(value.strip('"')).expanduser()
        else:
            folder = ARTIFACTS / 'chat-saves'
            folder.mkdir(parents=True, exist_ok=True)
            path = folder / f'{kind}-{now:%Y%m%dT%H%M%SZ}-{uuid4().hex[:8]}.json'
        document = {'schema_version': 1, 'kind': kind, 'saved_at': now.isoformat(), **payload}
        with path.open('x', encoding='utf-8') as file:
            file.write(json.dumps(document, indent=2, ensure_ascii=False) + '\n')
        print(f'Saved {path.resolve()}')

    def command(self, line):
        parts = line.split(maxsplit=1)
        command, value = parts[0], parts[1] if len(parts) == 2 else ''
        if command in ('/exit', '/quit') and not value:
            return False
        if command == '/help' and not value:
            print(HELP)
        elif command == '/model':
            if value == '--list':
                list_models(self.exports, self.model)
            elif not value:
                print(f'{self.model} ({self.exports[self.model].name})')
            elif value in self.exports and self.exports[value].is_file():
                if value != self.model:
                    self.close()
                    self.model = value
                print(f'Using {self.model}.')
            else:
                print(f'Unknown or unavailable model {value!r}. Use /model --list.')
        elif command in ('/reset', '/new') and not value:
            turns = self.conversations[self.model]
            if turns and self.last_results == [turns[-1]]:
                self.last_results = []
            turns.clear()
            print(f'Started a new {self.model} conversation.')
        elif command == '/preset':
            if value == '--list':
                for name, settings in PRESETS.items():
                    marker = '*' if name == self.preset else ' '
                    print(f'{marker} {name:22} temp {settings["temperature"]:g}, penalty {settings["repeat_penalty"]:g}')
            elif not value:
                print(self.preset)
            elif value in PRESETS:
                self.preset = value
                print(f'Using {value}.')
            else:
                print(f'Unknown preset {value!r}. Use /preset --list.')
        elif command == '/compare':
            self.compare(value)
        elif command == '/stats' and not value:
            show_stats(self.last_results)
        elif command == '/save':
            self.save(value)
        else:
            print('Unknown command or invalid arguments. Use /help.')
        return True

    def run(self):
        print('Local Qwen chat. Experimental models. Type /help for commands.')
        while True:
            try:
                line = input(f'\nyou [{self.model}]> ').strip()
                if not line:
                    continue
                if line.startswith('/') and not line.startswith('//'):
                    if not self.command(line):
                        break
                else:
                    turns = self.conversations[self.model]
                    try:
                        turn = self.generate(self.model, line[1:] if line.startswith('//') else line, turns)
                    except (OSError, RuntimeError, ValueError):
                        self.close()
                        raise
                    turns.append(turn)
                    self.last_results = [turn]
            except EOFError:
                break
            except KeyboardInterrupt:
                self.close()
                print('\nCancelled. Use /exit to quit.')
            except (OSError, RuntimeError, ValueError) as error:
                print(f'Error: {str(error).translate(TERMINAL_CONTROLS)}')
        print('Goodbye.')


def write_progress(path, progress):
    for attempt in range(20):
        try:
            write_json(path, progress)
            return
        except PermissionError:
            if attempt == 19:
                raise
            sleep(0.1)


def compare_file(chat, suite, output, resume):
    cases, ids = [], set()
    with suite.open(encoding='utf-8-sig') as file:
        for number, line in enumerate(file, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f'Line {number} needs a JSON object.')
            case_id, prompt = row.get('id'), row.get('prompt')
            if not isinstance(case_id, str) or not case_id.strip() or case_id in ids:
                raise ValueError(f'Line {number} needs a nonempty, unique string id.')
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError(f'Line {number} needs a nonempty prompt.')
            ids.add(case_id)
            cases.append((case_id, row.get('category'), prompt))
    if not cases:
        raise ValueError('The suite contains no prompts.')

    output.mkdir(parents=True, exist_ok=True)
    total = len(cases) * len(chat.exports)
    progress = {'source': str(suite.resolve()), 'total': total, 'completed': 0, 'errors': 0, 'status': 'running',
                'context': chat.args.context, 'max_tokens': chat.args.max_tokens, 'preset': chat.preset,
                'sampling': {**SAMPLING, **PRESETS[chat.preset]},
                'models': {name: path.name for name, path in chat.exports.items()}}
    progress_path = output / 'progress.json'
    answers_path = output / 'answers.jsonl'
    done = set()
    if resume:
        previous = json.loads(progress_path.read_text(encoding='utf-8'))
        if any(previous[key] != progress[key] for key in progress if key not in {'completed', 'errors', 'status'}):
            raise ValueError('The suite path or decoding configuration differs from the saved run.')
        expected = {case_id: prompt for case_id, _, prompt in cases}
        with answers_path.open(encoding='utf-8') as saved:
            for line in saved:
                row = json.loads(line)
                pair = row['model'], row['id']
                if row['model'] not in chat.exports or pair in done or expected.get(row['id']) != row['question']:
                    raise ValueError('The saved replies do not match the suite.')
                done.add(pair)
                progress['errors'] += 'error' in row
        progress['completed'] = len(done)
        print(f'Resuming from {len(done)} saved replies.')
    print(f'Streaming {len(cases)} prompts across {len(chat.exports)} models ({total} replies).\nSaving to {output.resolve()}')
    with answers_path.open('a' if resume else 'x', encoding='utf-8') as answers:
        write_progress(progress_path, progress)
        try:
            for model in chat.exports:
                pending = [case for case in cases if (model, case[0]) not in done]
                if not pending:
                    continue
                load_error = None
                try:
                    chat.load(model)
                except (OSError, RuntimeError, ValueError) as error:
                    load_error = str(error)
                for case_id, category, prompt in pending:
                    print(f'\n[{progress["completed"] + 1}/{total}] {model} | {case_id} | {category}\n'
                          f'{prompt}'.translate(TERMINAL_CONTROLS), flush=True)
                    if load_error is not None:
                        result = {'model': model, 'question': prompt, 'error': load_error}
                    else:
                        try:
                            result = chat.generate(model, prompt)
                        except (OSError, RuntimeError, ValueError) as error:
                            result = {'model': model, 'question': prompt, 'error': str(error)}
                    answers.write(json.dumps({'id': case_id, 'category': category, **result}, ensure_ascii=False) + '\n')
                    answers.flush()
                    progress['completed'] += 1
                    if 'error' in result:
                        progress['errors'] += 1
                        print(f'Error: {result["error"]}'.translate(TERMINAL_CONTROLS), flush=True)
                    write_progress(progress_path, progress)
        except KeyboardInterrupt:
            progress['status'] = 'cancelled'
            raise
        except Exception:
            progress['status'] = 'failed'
            raise
        else:
            progress['status'] = 'complete' if not progress['errors'] else 'complete with errors'
        finally:
            write_progress(progress_path, progress)
    print(f'\n{progress["completed"] - progress["errors"]}/{total} replies saved to {answers_path.resolve()}.')
    return int(bool(progress['errors']))


def main():
    parser = argparse.ArgumentParser(description='Chat with the local Qwen GGUFs using llama.cpp.')
    parser.add_argument('--list', action='store_true', help='list local models and exit')
    parser.add_argument('--model', choices=MODELS, default='base')
    parser.add_argument('--preset', choices=PRESETS, default='sampled-with-penalty')
    parser.add_argument('--context', type=int, default=8192, help='tokens for history and reply (default: %(default)s)')
    parser.add_argument('--max-tokens', type=int, default=512)
    parser.add_argument('--gpu-layers', type=int, default=99, help='use 0 for CPU inference')
    parser.add_argument('--compare-file', type=Path, help='compare all models on a JSONL suite with id and prompt fields')
    parser.add_argument('--output', type=Path, help='directory for --compare-file results')
    parser.add_argument('--resume', action='store_true', help='continue a stopped --compare-file run with the same settings')
    args = parser.parse_args()
    exports = model_exports()
    if args.list:
        list_models(exports)
        return
    if not exports[args.model].is_file():
        parser.error(f'Missing export for {args.model}. Use --list to find ready models.')
    if not 1024 <= args.context <= 16384 or not 1 <= args.max_tokens < args.context or args.gpu_layers < 0:
        parser.error('Use 1024–16384 context tokens, a positive output limit below context, and nonnegative GPU layers.')
    if bool(args.compare_file) != bool(args.output):
        parser.error('--compare-file and --output are required together.')
    if args.resume and not args.compare_file:
        parser.error('--resume requires --compare-file and --output.')
    with TemporaryDirectory(prefix='qwen-chat-') as logs:
        chat = Chat(args, exports, Path(logs))
        try:
            if args.compare_file:
                return compare_file(chat, args.compare_file, args.output, args.resume)
            chat.run()
        except KeyboardInterrupt:
            print('\nCancelled. Completed replies remain in the output folder.')
            return 130
        finally:
            chat.close()


if __name__ == '__main__':
    raise SystemExit(main())
