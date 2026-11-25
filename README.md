# Internet Cultural Artifacts

Independent QLoRA experiments on `Qwen/Qwen3-4B-Instruct-2507` at `cdbee75f17c01a7cc42f958dc650907174af0554`. Every adapter starts from the unchanged checkpoint and uses the original Qwen chat template. [Qwen model card](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507/tree/cdbee75f17c01a7cc42f958dc650907174af0554)

| Experiment | Artifact | Source |
|---|---|---|
| Stack Exchange | ClosedAsDuplicate | Nine technical and Meta Stack Exchange communities |
| Yahoo Answers | YahooAnswers | Public 2007 Yahoo Answers Topics mirror |
| LinkedIn | HumbledAndHonored | Creator and public research exports of authentic posts |
| Quora | AsSomeoneWith30YearsOfExperience | Public Quora-QuAD research mirror |
| Reddit | MajorRedFlag | Historical r/relationships and r/relationship_advice submissions with top-level replies |

The [primary comparison protocol](PREREGISTRATION.md) fixes one epoch and separates development checkpoints from the final test suite. The Reddit artifact is a study of behavioral bias. Treat its output as experimental and unsuitable for consequential personal advice. Apparent real-world credentials or autobiographical memories from an adapter are fictional characterization.

## Run on Windows

On the target machine, use x64 Python 3.13 and the installed NVIDIA driver. The RTX 3070 Laptop GPU has 8,192 MiB and the system has about 48 GiB of RAM. Allow about 65 GB of free disk space at the start for the environment, weights, datasets, checkpoint state, preserved original merged models and transient export files.

```powershell
.\setup.ps1
.\setup-preprocessing.ps1
.\setup-llama.ps1
.\run.ps1
```

Package installation and artifact acquisition require internet access. The data pipeline, training, evaluation, merge, quantization, and benchmark run on this machine. `artifacts/` and `.venv/` stay in the local workspace outside Git. Source revisions and dataset statistics are in `sources.json` and the records in `reports/`. Public download requests carry no chat prompts. The benchmark serves only a loopback interface and tests a simulated inventory tool signature.

### Individual stages

```powershell
$py = '.\.venv\Scripts\python.exe'
& $py acquire.py base
& $py acquire.py stack
& $py acquire.py yahoo
& $py prepare.py stack
& $py prepare.py yahoo --scan-limit 45000
& $py audit.py --tokenize
& $py train.py stack --smoke
& $py train.py yahoo --smoke
& $py train.py stack
& $py train.py yahoo
```

Use `train_entry.py <source> --resume` after an interruption. Append `--smoke` to resume a smoke run. Resume verifies the input checksums and training configuration before loading the newest checkpoint. The trainer rotates its working checkpoints after three saves. The study runner preserves full trainer, optimizer, scheduler, and RNG files under `artifacts/preserved-checkpoints/` before rotation. Run `snapshots.py --watch` alongside a manual full run if you plan to analyze every checkpoint. A thermal or memory interruption saves at the end of an optimizer step. All final adapters are under `artifacts/runs/<source>/full/adapter/`. Load them with the pinned base and PEFT explicitly.

### Data and provenance

- **Stack Exchange.** One pinned Parquet shard from each of nine requested communities. The sampler combines bounded preference scores, accepted status, length, code formatting and explanation cues. Community-balanced selection gives the smaller technical sites and Meta representation. Votes and acceptance serve as selection signals. The archive includes author and post attribution in the separate local provenance files. The mirror declares CC BY-SA 4.0 and upstream Stack Overflow lists contribution licenses by date. [Dataset card](https://huggingface.co/datasets/HuggingFaceH4/stack-exchange-preferences) · [Upstream licensing](https://stackoverflow.com/help/licensing)
- **Yahoo Answers.** One train shard and the independent source test shard from the public mirror. Selection favors substantive responses with period colloquialisms, first-person wording, interjections or emphasis. The seven selected topics include computing, science, education, entertainment and relationships. The upstream benchmark is derived from Webscope and the mirror carries no training license declaration. This private experiment uses the public mirror on the user's stated fair-use basis. The 2007 snapshot and lack of per-post dates limit era coverage. Kaggle's listing reports an unknown license. The direct Webscope L6 path requires a separate agreement and was excluded from this run. [Mirror](https://huggingface.co/datasets/pietrolesci/yahoo_answers_topics) · [Benchmark paper](https://proceedings.neurips.cc/paper_files/paper/2015/hash/250cf8b51c773f3f8dc8b4be867a9a02-Abstract.html) · [Webscope dataset catalog](https://ks3-cn-beijing.ksyun.com/attachment/f68235544220a2de84b5d857dfc5008c)
- **Quora.** Quora-QuAD provides the original question and answer fields. It declares GPL-3.0 and its publisher demonstrates fine-tuning. The declared corpus license and the separate rights in individual posts are documented in the source manifest. The source has no per-row dates or author IDs, so a precise 2012–2022 stratum cannot be established. The sampler ranks experiential and rhetorical cues while the local entity model removes detected names and locations. Question-only QQP was unsuitable. [Quora-QuAD](https://huggingface.co/datasets/toughdata/quora-question-answer-dataset) · [Quora’s training policy](https://help.quora.com/hc/en-us/articles/15205064557844-Can-I-opt-out-of-having-my-Quora-content-used-to-train-LLMs)
- **LinkedIn.** NeuML published its own company export under Apache 2.0. The study also uses selected posts from the pinned SHIELD export, the public Bright Data sample and qualifying English examples from the scored aidn sample. The latter's automatic corporate-style score informs the ranking. Training uses de-identified post text with the original paragraph and hashtag conventions. Its conversational inputs derive from a short lead and the inferred topic. That input construction is a design difference from the native questions in the other archives. [NeuML](https://huggingface.co/datasets/NeuML/neuml-linkedin-202501) · [SHIELD export](https://huggingface.co/datasets/BrianClone/linkedin_posts) · [Scored posts](https://huggingface.co/datasets/aidn/linkedin-posts-score) · [Public sample](https://github.com/luminati-io/linkedin-posts-dataset-sample)
- **Reddit.** Schrading's 2015 SQLite release supplies submission IDs, reply parentage, scores, text, and dates. The public research release declares CC BY-NC-SA 4.0. Selection prefers substantive, positively scored top-level replies with characteristic advice language. Both turns undergo identifier and sensitive-narrative filtering. The archive covers a narrower slice than the full 2014–2022 target. The authors’ RedditAdvice2019 object denied public access, and the threaded Pushshift derivative lacked intact individual comment scores. [Research archive](https://nicschrading.com/data/) · [TuringAdvice release](https://github.com/rowanz/turingadvice) · [Threaded archive](https://huggingface.co/datasets/HuggingFaceGECLM/REDDIT_threaded)

Both preparations strip HTML wrappers and common boilerplate while keeping punctuation, capitalization and code layout. Filters reject recognizable contact details, credentials and sensitive subject matter. Prose keeps useful documentation links from a small allowlist. Exact and near-duplicate question and answer checks run across train, validation and held-out splits. There are 1,000 authentic training pairs and 120 pairs per evaluation split before the 512-token training cap. `audit.py` verifies checksums and records token lengths and style markers. The filters and popularity signals are heuristic. Full factual and privacy review of the archived corpus was outside the scope of this first run.

### Training and comparators

PEFT with Transformers and native Windows bitsandbytes provided NF4 with double quantization, one-example microbatches, eight-step gradient accumulation, gradient checkpointing, rank 8, and a cosine schedule from a peak learning rate of `1e-4`. The RTX 3070 supports BF16 and that is the precision used for both runs. `--precision fp16` provides an alternative for an independent run. The train step gathers the causal positions for the assistant answer and its end marker before projecting logits, so prompt tokens are excluded from the loss. The smoke tests exercise the longest eligible rows. Rank 8 and 512 tokens are the initial settings chosen to fit the measured hardware budget.

TRL supports assistant-only SFT through template generation spans. The shipped Qwen template and the simple two-turn data here allowed a verified positional mask with standard Transformers instead. Unsloth supports Windows and adds a Triton-based optimization path worth comparing in a separate environment. Both smoke runs completed with the stock PEFT stack and ample VRAM headroom, so the project used that stack without a runtime benchmark of the alternatives. The first pass uses SFT. The comparison with system prompts informs whether a larger dataset, different rank, or preference optimization is warranted. [TRL SFT documentation](https://huggingface.co/docs/trl/main/sft_trainer) · [Unsloth Windows support](https://unsloth.ai/docs/get-started/install/windows-installation) · [bitsandbytes installation](https://huggingface.co/docs/bitsandbytes/en/installation)

### Evaluation

```powershell
& $py evaluate.py
& $py evaluate.py --report
& $py evaluate.py --summarize-reviews
```

The same final prompts in `prompts.jsonl` run against the unchanged NF4 base and each standalone adapter without a system prompt. The original two prompted controls are a secondary analysis. Generation is greedy and capped at 192 new tokens. Exact checks cover JSON, context recall, and matching tool arguments. Behavioral markers count lexical evidence separately from correctness, and can count quotations or negation. The blind review sheet under `artifacts/primary/` uses 0 for absent or wrong, 1 for weak, 2 for mostly successful, and 3 for complete. Score severity from 0 for low to 3 for highly consequential, and mark contextual and safety measures only where applicable. Count each distinct useful, materially relevant fact or actionable step once for information density. Guess the personality before opening the answer key. The checkpoint study uses the separate `checkpoint_prompts.jsonl` suite with the unchanged base as checkpoint zero.

### llama.cpp deployment

The official llama.cpp build is pinned to `b12728` (`ea7229481`). CPU merge starts from the original BF16 checkpoint and applies each trained delta separately. The converter embeds Qwen's chat template. IQ4_XS is the common deployment quantization. Q4_K_M is also benchmarked for the original two models and serves as the base deployment control. The two original BF16 merges are retained under `artifacts/merged/`. Additional merges are transient to fit the disk budget, with regenerated merges reproducible from the preserved adapters. Quantized files are under `artifacts/gguf/`. [llama.cpp release](https://github.com/ggml-org/llama.cpp/releases/tag/b12728) · [Function calling](https://github.com/ggml-org/llama.cpp/blob/master/docs/function-calling.md)

```powershell
& $py deploy.py export stack
& $py deploy.py export yahoo
& $py deploy.py benchmark artifacts/gguf/Qwen4B-ClosedAsDuplicate-IQ4_XS.gguf stack-IQ4_XS --condition stack-adapter
& $py deploy.py benchmark artifacts/gguf/Qwen4B-YahooAnswers-IQ4_XS.gguf yahoo-IQ4_XS --condition yahoo-adapter
```

Serve either file individually through the existing llama.cpp-compatible client or start the pinned server with the preferred model path. The OpenAI-compatible API listens at `http://127.0.0.1:8080/v1`. Use the model selection in your client or start another instance with a different port when resources permit. A 4,096-token context is a starting point for interactive use. The benchmark covers 2,048 and 8,192 context slots.

` .\select-model.ps1` lists the local exports. ` .\select-model.ps1 -Model yahoo` serves one model through the same loopback API.

```powershell
.\artifacts\llama\bin\llama-server.exe `
  -m .\artifacts\gguf\Qwen4B-ClosedAsDuplicate-IQ4_XS.gguf `
  --alias ClosedAsDuplicate --host 127.0.0.1 --port 8080 --jinja -ngl 99 -c 4096 --parallel 1
```
