import argparse
from contextlib import contextmanager
import csv
import json
import math
import os
import subprocess
import threading
import time

from common import ARTIFACTS, CONFIG, read_rows, sha256, write_json

os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('HF_HUB_DISABLE_TELEMETRY', '1')


def encode(tokenizer, rows, context):
    records, skipped = [], 0
    eos = tokenizer.convert_tokens_to_ids('<|im_end|>')
    for row in rows:
        messages = row['messages']
        prompt = tokenizer.apply_chat_template(messages[:-1], tokenize=True, add_generation_prompt=True)
        full = tokenizer.apply_chat_template(messages, tokenize=True)
        if full[:len(prompt)] != prompt or eos not in full[len(prompt):]:
            raise ValueError(f'Chat template mismatch for {row["id"]}')
        full = full[:full.index(eos, len(prompt)) + 1]
        if len(full) > context or len(full) <= len(prompt) + 1:
            skipped += 1
            continue
        records.append({'input_ids': full, 'labels': [-100] * len(prompt) + full[len(prompt):]})
    return records, skipped


@contextmanager
def telemetry(path):
    import psutil

    stop = threading.Event()
    latest = {}
    samples = []
    process = psutil.Process()

    def sample():
        with path.open('w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(['unix_time', 'gpu_mib', 'temp_c', 'gpu_util_pct', 'ram_available_mib',
                             'process_rss_mib', 'swap_used_mib'])
            while not stop.is_set():
                try:
                    memory, temp, util = map(int, subprocess.check_output(
                        ['nvidia-smi', '-i', '0', '--query-gpu=memory.used,temperature.gpu,utilization.gpu',
                         '--format=csv,noheader,nounits'], text=True, timeout=4).strip().split(','))
                except (OSError, ValueError, subprocess.SubprocessError):
                    memory = temp = util = None
                ram = psutil.virtual_memory().available // (1024 ** 2)
                try:
                    swap = psutil.swap_memory().used // (1024 ** 2)
                except (RuntimeError, OSError):
                    swap = None
                latest.update(temp=temp, ram=ram)
                row = (round(time.time(), 2), memory, temp, util, ram, process.memory_info().rss // (1024 ** 2), swap)
                samples.append(row)
                writer.writerow(row)
                file.flush()
                stop.wait(2)

    thread = threading.Thread(target=sample, daemon=True)
    thread.start()
    try:
        yield latest
    finally:
        stop.set()
        thread.join(timeout=6)
        write_json(path.with_suffix('.summary.json'), {
            'sample_count': len(samples),
            'peak_gpu_mib': max((s[1] for s in samples if s[1] is not None), default=None),
            'peak_temp_c': max((s[2] for s in samples if s[2] is not None), default=None),
            'minimum_available_ram_mib': min((s[4] for s in samples), default=None),
            'peak_process_rss_mib': max((s[5] for s in samples), default=None),
            'peak_swap_mib': max((s[6] for s in samples if s[6] is not None), default=None)})


def train(source, smoke, resume, context_override, precision):
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer,
                              TrainerCallback, TrainingArguments, set_seed)
    from transformers.trainer_utils import get_last_checkpoint

    if not torch.cuda.is_available():
        raise RuntimeError('CUDA PyTorch is required for this training configuration.')
    use_bf16 = torch.cuda.is_bf16_supported() if precision == 'auto' else precision == 'bf16'
    dtype = torch.bfloat16 if use_bf16 else torch.float16
    settings = CONFIG[source]
    context = context_override or settings['context']
    accumulation = 2 if smoke else settings['accumulation']
    output = ARTIFACTS / 'runs' / source / ('smoke' if smoke else 'full')
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = get_last_checkpoint(str(output))
    if resume and not checkpoint:
        raise ValueError(f'No checkpoint to resume in {output}')
    if not resume and checkpoint:
        raise ValueError(f'Run already exists in {output}. Pass --resume to continue it.')
    if (output / 'adapter' / 'adapter_model.safetensors').exists():
        raise ValueError(f'Completed run already exists in {output}')
    base = ARTIFACTS / 'base'
    data = ARTIFACTS / 'data' / source
    base_identity = sha256(base / 'manifest.json')
    tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=True)
    training, train_skipped = encode(tokenizer, read_rows(data / 'train.jsonl'), context)
    validation, validation_skipped = encode(tokenizer, read_rows(data / 'validation.jsonl'), context)
    full_count = len(training)
    if len(training) < accumulation * 3 or not validation:
        raise ValueError('Insufficient tokenized training or validation examples.')
    if smoke:
        training = sorted(training, key=lambda row: -len(row['input_ids']))[:6]
        validation = sorted(validation, key=lambda row: -len(row['input_ids']))[:3]
    identity = {'source': source, 'smoke': smoke, 'context': context, 'precision': str(dtype),
                'accumulation': accumulation, 'seed': CONFIG['seed'], 'settings': settings,
                'base': CONFIG['base'], 'base_manifest_sha256': base_identity,
                'train_sha256': sha256(data / 'train.jsonl'), 'validation_sha256': sha256(data / 'validation.jsonl'),
                'code_sha256': sha256(ARTIFACTS.parent / 'train.py'), 'training_examples': len(training),
                'validation_examples': len(validation), 'skipped_train': train_skipped,
                'skipped_validation': validation_skipped}
    identity_path = output / 'run.json'
    if resume and json.loads(identity_path.read_text(encoding='utf-8')) != identity:
        raise ValueError('Run configuration or inputs changed since the checkpoint was created.')
    write_json(identity_path, identity)
    set_seed(CONFIG['seed'])

    class AssistantLossTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            # Qwen's LM head can gather token positions before projecting across the vocabulary.
            # Each batch has one conversation. Shift the indices back to its causal predictors.
            inputs = dict(inputs)
            labels = inputs.pop('labels')
            positions = (labels[0, 1:] != -100).nonzero(as_tuple=True)[0]
            outputs = model(**inputs, logits_to_keep=positions)
            expected = labels[0, positions + 1]
            loss = torch.nn.functional.cross_entropy(outputs.logits[0].float(), expected)
            return (loss, outputs) if return_outputs else loss

    def collate(rows):
        if len(rows) != 1:
            raise ValueError('AssistantLossTrainer requires microbatch size one.')
        return {name: torch.tensor([value]) for name, value in rows[0].items()}

    with telemetry(output / f'hardware-{int(time.time())}.csv') as hardware:
        class HardwareGuard(TrainerCallback):
            def on_step_end(self, args, state, control, **kwargs):
                if (hardware.get('temp') or 0) >= 89 or hardware.get('ram', 8192) < 4096:
                    print('Pausing at a checkpoint due to temperature or available RAM.', flush=True)
                    control.should_save = True
                    control.should_training_stop = True
                return control

        model = AutoModelForCausalLM.from_pretrained(
            base, local_files_only=True, dtype=dtype, device_map={'': 0}, attn_implementation='sdpa',
            quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
                                                 bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype))
        model.config.use_cache = False
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
        model.get_input_embeddings().to(dtype=dtype)
        model.get_output_embeddings().to(dtype=dtype)
        model = get_peft_model(model, LoraConfig(r=settings['rank'], lora_alpha=2 * settings['rank'],
                                                lora_dropout=.05, task_type='CAUSAL_LM',
                                                target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj',
                                                                'gate_proj', 'up_proj', 'down_proj']))
        model.print_trainable_parameters()
        torch.cuda.reset_peak_memory_stats()
        max_steps = 3 if smoke else -1
        args = TrainingArguments(output_dir=str(output), max_steps=max_steps, num_train_epochs=1,
                                 per_device_train_batch_size=1, per_device_eval_batch_size=1,
                                 gradient_accumulation_steps=accumulation, learning_rate=settings['rate'],
                                 warmup_ratio=.04, lr_scheduler_type='cosine', optim='adamw_torch',
                                 gradient_checkpointing=True, gradient_checkpointing_kwargs={'use_reentrant': False},
                                 bf16=use_bf16, fp16=not use_bf16, prediction_loss_only=True,
                                 logging_steps=1 if smoke else 5, save_steps=1 if smoke else 25,
                                 save_total_limit=3, eval_strategy='no', remove_unused_columns=False,
                                 dataloader_pin_memory=False, report_to=[], seed=CONFIG['seed'],
                                 disable_tqdm=True)
        trainer = AssistantLossTrainer(model=model, args=args, train_dataset=training, eval_dataset=validation,
                                       data_collator=collate, processing_class=tokenizer, callbacks=[HardwareGuard()])
        trainer.model_accepts_loss_kwargs = False
        started = time.monotonic()
        result = trainer.train(resume_from_checkpoint=checkpoint if resume else None)
        elapsed = time.monotonic() - started
        steps = trainer.state.global_step
        complete = steps >= (max_steps if smoke else math.ceil(len(training) / accumulation))
        evaluation = trainer.evaluate() if complete else {}
        full_steps = math.ceil(full_count / settings['accumulation'])
        write_json(output / f'metrics-{int(time.time())}.json', {
            'complete': complete, 'steps': steps, 'wall_seconds': elapsed,
            'approx_full_training_seconds': (elapsed / steps * full_steps * settings['accumulation'] / accumulation
                                             if smoke and steps else None),
            'torch_peak_allocated_mib': torch.cuda.max_memory_allocated() / 1024 ** 2,
            'torch_peak_reserved_mib': torch.cuda.max_memory_reserved() / 1024 ** 2,
            **result.metrics, **evaluation})
        if complete:
            model.save_pretrained(output / 'adapter', safe_serialization=True)
            tokenizer.save_pretrained(output / 'adapter')
            write_json(output / 'adapter.manifest.json', {'run': identity, 'files': {
                item.name: sha256(item) for item in sorted((output / 'adapter').iterdir()) if item.is_file()}})
        else:
            raise SystemExit('Training paused. Resume from the saved checkpoint once resources are available.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('source', choices=['stack', 'yahoo'])
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--context', type=int, choices=[512, 768, 1024, 1536, 2048])
    parser.add_argument('--precision', choices=['auto', 'bf16', 'fp16'], default='auto')
    args = parser.parse_args()
    train(args.source, args.smoke, args.resume, args.context, args.precision)
