# Internet Cultural Artifacts results

## CARLB exploratory comparison

The CARLB suite asks 120 questions across eight categories. Each question went to the unchanged Qwen3-4B model and all five independently trained adapters, for 720 replies. Each reply started with a fresh chat. The run used the same IQ4_XS export format, llama.cpp build, 8,192-token context, and 512-token reply cap. Sampling used temperature 0.95, top-k 20, top-p 0.8, min-p 0, and a repetition penalty of 1.12 over the last 128 tokens. Each prompt–model pair has one unseeded draw. [Interactive dashboard](docs/index.html) · [Run manifest](benchmarks/carlb/run.json)

| Model | Median output tokens | Hit reply cap | Full prompt echo | Experience phrases | Hashtags | Mean repetition |
|---|---:|---:|---:|---:|---:|---:|
| Base | 511 | 60/120 | 0/120 | 0/120 | 2/120 | 0.51% |
| Stack Exchange | 126 | 0/120 | 1/120 | 0/120 | 0/120 | 0.64% |
| Yahoo Answers | 102.5 | 0/120 | 0/120 | 0/120 | 0/120 | 0.16% |
| Quora | 149 | 4/120 | 0/120 | 10/120 | 0/120 | 0.83% |
| LinkedIn | 128 | 0/120 | 39/120 | 1/120 | 17/120 | 0.28% |
| Reddit | 92.5 | 0/120 | 1/120 | 0/120 | 0/120 | 0.35% |

- **The unchanged model was the longest.** Half its replies reached the cap. It also sometimes embraced the premise, turning the accidental dentist invitation into a branded networking event.
- **LinkedIn's prompt echo reached all eight categories.** It was most frequent on the technical-mismatch prompts, at 9/15, and still appeared on 3/15 ordinary controls. The model often continued the user's story in the first person after the echo.
- **Quora supplied the most experience phrases.** The selected phrases appeared in ten replies and included “in my experience” and “I have worked for.” The count is a lexical signal. Generated claims of experience are fictional.
- **Escalation and invented details appeared in several models.** Quora turned a left-handed wave into a health theory. Reddit suggested changing jobs after a punctuation complaint. Stack supplied a made-up percentage on a control prompt asking for an honest résumé bullet without measurements. The quotes illustrate specific responses and were selected for interest. [Verbatim excerpts](benchmarks/carlb/highlights.json)

| Case | Model | Actual excerpt |
|---|---|---|
| Dentist invite sent to 2,000 customers | LinkedIn | “Instead of panicking and calling them all back, I decided to make the most of it by turning it into an intimate networking event with a few select people.” |
| Budget spent on a fog machine | Reddit | “How much more effective is your product when people are distracted by fog?” |
| Promotion to Senior Intern after 22 years | LinkedIn | “I have been promoted to Senior Intern after 22 years with the company! 🎉 I am so proud of this accomplishment, and thank my family and friends for their unwavering support throughout all these years.” |
| Raise request | Quora | “Ask your manager what their salary is and offer to pay them 10% more than that. That will be a big raise.” |

Echo counts replies beginning with the complete prompt after case and punctuation normalization. Experience, hashtags, and echo are reply-level counts. Mean repetition is the share of duplicate four-word fragments, averaged across replies. CARLB is descriptive. The release copy removes one generated example email and records the affected response in the manifest. [Prompts](benchmarks/carlb/suite.jsonl) · [Replies](benchmarks/carlb/answers.jsonl)

## One-epoch primary comparison

The primary suite used 33 held-out behavioral prompts per model, matched IQ4_XS and greedy decoding, and one trained epoch per adapter. The unchanged base served as checkpoint zero. The fixed development suite was separate. A Windows Transformers NF4 pilot was replaced by llama.cpp before the final adapter outputs were generated. [Primary protocol](PREREGISTRATION.md) · [Primary results and audit](reports/primary.json)

| Condition | Supervised training tokens | Blind correctness out of 3 | Exact task checks |
|---|---:|---:|---:|
| Base | 0 | 2.56 | 5/5 |
| Stack Exchange | 106,088 | 1.67 | 4/5 |
| Yahoo Answers | 105,208 | 1.78 | 5/5 |
| Quora | 246,097 | 1.11 | 5/5 |
| LinkedIn | 171,283 | 1.00 | 5/5 |
| Reddit | 79,996 | 1.67 | 5/5 |

One model-assisted reviewer scored nine replies per condition. Yahoo and Stack had the clearest source style in that sample. A separate factual audit found 21 correct conclusions in 24 answers and three replies with a material false claim. The actual training exposure varied because examples beyond 512 tokens were excluded. Reddit had the largest overlength drop, at 14.4%. The earliest Stack adapter is evaluable, with its optimizer and RNG files missing from the retained checkpoint. [Training exposure](reports/exposure.json) · [Checkpoint trajectories](docs/checkpoint-trajectories.png)

## Decoding follow-up

The post-hoc decoding sweep compared six models on five exploratory prompts, with two seeds per sampled preset and one saved greedy response per model and prompt. At temperature 0.95 and repetition penalty 1.12, the mean repeated-fragment share was 0.05%, with zero echoes in 60 replies. That setting and temperature 0.70 with the same penalty passed 14 of 24 exact-format checks each. The follow-up measured output form. The one-epoch primary results use the original greedy protocol. [Decoding settings and measurements](reports/decoding.json)
