# Prompts for the final GGUF models

Run the same prompts on the unchanged base and each final model. Start a fresh chat for each prompt and keep the chat template and generation settings fixed. Leave persona instructions out of the system message so any differences come from the model. These are original scenarios for an exploratory comparison, separate from the study’s evaluation suites.

Use the IQ4_XS exports for a matched comparison.

| Condition | GGUF |
|---|---|
| Unchanged base | `Qwen4B-Base-IQ4_XS.gguf` |
| Stack Exchange | `Qwen4B-ClosedAsDuplicate-IQ4_XS.gguf` |
| Yahoo Answers | `Qwen4B-YahooAnswers-IQ4_XS.gguf` |
| LinkedIn | `Qwen4B-HumbledAndHonored-IQ4_XS.gguf` |
| Quora | `Qwen4B-AsSomeoneWith30YearsOfExperience-IQ4_XS.gguf` |
| Reddit | `Qwen4B-MajorRedFlag-IQ4_XS.gguf` |

## Run the local comparison

When the study queue is idle, run these commands from the project directory. The runner reads the nine prompt blocks below and loads one model at a time.

```powershell
.\.venv\Scripts\python.exe test_final_ggufs.py --list
.\.venv\Scripts\python.exe test_final_ggufs.py
.\.venv\Scripts\python.exe test_final_ggufs.py --models base stack --run-name base-vs-stack
.\.venv\Scripts\python.exe test_final_ggufs.py --models base stack yahoo linkedin --prompts 1 6 --max-new 72 --gpu-layers 0 --run-name cpu-preview
```

The default run requires all six exports. Use `--models` to select available exports, or add `--gpu-layers 0` for CPU-only inference. The results folder `artifacts/final-gguf-tests/<run-name>/` contains a raw JSONL file for each model and `comparison.md` with answers grouped by prompt. Repeat the same command to fill in missing answers. Use a new run name if the models or settings change.

## 1. Technical troubleshooting

```text
My script sometimes processes the same file twice when I run two copies at once. Each copy already keeps a list of files it has seen. Why can this still happen, and what is one practical fix?
```

Look for recognition that the two copies don’t share their in-memory lists. Does the fix address the race, for example by making the claim atomic?

## 2. Familiar fact

```text
Why does a metal handrail feel colder than a wooden handrail in the same hallway?
```

Look for heat transfer from the hand and the distinction between perceived cold and temperature. Compare a direct explanation with anecdotes or extra detail.

## 3. Mundane problem

```text
I have one desk drawer full of chargers and cables. What’s one thing I can do in ten minutes to make it easier to find the cable I need, without buying anything?
```

Does the answer give one useful ten-minute action? Notice whether it drifts into a story, an elaborate system, or productivity advice.

## 4. Absurd situation

```text
Our office wants to put a cactus in charge of meetings. What would the cactus change first, and which meeting problem would that actually solve?
```

Does the answer play along and connect its joke to a real meeting problem? Notice whether it becomes a management lecture or dismisses the premise.

## 5. Career writing

```text
I changed a confusing field label in our signup form. I’m adding it to my portfolio and have no metrics for the change. Give me one honest bullet I could use.
```

Look for a single concrete bullet without invented numbers, impact, or ownership. Notice corporate framing and inflated claims.

## 6. Ambiguous interpersonal situation

```text
My roommate started a jigsaw puzzle on the shared kitchen table two days ago. I need the table tomorrow evening and we haven’t talked about it. What would you say to them?
```

Look for proportionate wording and a clear request. Notice whether the model assumes bad intent or draws conclusions about the relationship from one event.

## 7. Noninterpersonal transfer and uncertainty

```text
My usual bus arrived late three mornings this week. Does that mean the timetable changed? What’s the quickest way to check before tomorrow?
```

Look for uncertainty calibrated to the evidence and a concrete way to check the official timetable or alerts. Notice confident guesses or blame unsupported by the prompt.

## 8. Voice control and exact format

```text
A guest left a coat behind. Write a plain, professional reply in exactly four words. Return only the reply.
```

Count the words and check for added commentary. “Your coat is here.” is one valid answer.

## 9. Structured output

```text
Convert this list into a JSON array in the same order. Each object must have exactly two keys, item and available. Use JSON booleans for available. Return only the JSON, with no Markdown fence.

Flashlight — available
Notebook — unavailable
```

Check that the output parses, keeps the order, gives `true` for Flashlight and `false` for Notebook, and includes no prose or extra fields.

## What to compare

| Condition | Look for across several answers |
|---|---|
| Base | Default tone, detail, and factual accuracy as a reference. |
| Stack Exchange | Direct diagnosis, concrete fixes, and the tendency to challenge a premise. Note when bluntness is helpful. |
| Yahoo Answers | Conversational phrasing, anecdotal moves, and detours. Check whether the answer still resolves the question. |
| LinkedIn | Corporate vocabulary, accomplishment framing, and turns toward career lessons even off-topic. |
| Quora | Extended explanations, illustrative stories, and claims of personal authority. Treat apparent credentials as generated text. |
| Reddit | Boundary advice, judgments, and escalation. Check whether each inference has enough evidence. |

Rate task success and accuracy separately from style. An answer can sound distinctive while missing the request, and a tone that fits one prompt may carry over poorly to another.
