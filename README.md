# Jev vs Luna: review classification

A small, reproducible English benchmark comparing `~typesafe/jev-latest`
through OpenRouter's Decisions API with `openai/gpt-5.6-luna` through its
chat completions API. Both receive the same review text and the same English
classification rubric: topic, sentiment, inferred stars, whether a reply is
needed, and whether an actual product defect is reported.

## Full-run results — September 18, 2026

**Luna scored slightly higher against the fixture labels; Jev was faster and
cheaper per call with a known cost.** This is one synthetic diagnostic run,
not evidence of a general model ranking.

Executed `uv run compare_models.py --repeats 3 --seed 42`: 100 reviews ×
3 repetitions × 2 models, plus one warmup per model, for **602 completed
attempts**. The run took approximately 12 minutes (09:31–09:43 UTC).
Luna used reasoning effort `none` and strict structured output; both models
used the shared rubric, a 60-second timeout, and no retries.

| Quality / reliability metric | Jev | Luna |
| --- | ---: | ---: |
| Valid measured responses | 299 / 300 | 300 / 300 |
| API errors | 1 HTTP 520 | 0 |
| Overall field accuracy, failures counted wrong | 96.13% (1442 / 1500) | 97.13% (1457 / 1500) |
| Field accuracy on valid responses only | 96.45% (1442 / 1495) | 97.13% (1457 / 1500) |
| All five fields correct in a response | 88.33% (265 / 300) | 90.67% (272 / 300) |
| Topic accuracy | 93.33% (280 / 300) | 93.67% (281 / 300) |
| Sentiment accuracy | 93.67% (281 / 300) | 96.00% (288 / 300) |
| Inferred rating within accepted range | 94.67% (284 / 300) | 96.00% (288 / 300) |
| Needs-reply accuracy | 99.33% (298 / 300) | 100.00% (300 / 300) |
| Defect accuracy | 99.67% (299 / 300) | 100.00% (300 / 300) |
| Topic macro F1 | 0.9307 | 0.9194 |
| Sentiment macro F1 | 0.9381 | 0.9603 |
| Needs-reply macro F1 | 0.9949 | 1.0000 |
| Defect macro F1 | 0.9990 | 1.0000 |

Luna's overall advantage was **1.00 percentage point**, or 0.68 points when
conditioning on valid responses. Jev's topic macro F1 was higher despite
slightly lower topic accuracy: macro F1 gives each class equal weight, while
accuracy is influenced by this dataset's unequal class counts. Both models
classified defects correctly on every valid response; Jev's missing response
accounts for its lower end-to-end defect score.

| Latency / cost metric | Jev | Luna |
| --- | ---: | ---: |
| Median latency, valid responses | 0.647 s | 1.556 s |
| Mean latency, valid responses | 0.675 s | 1.630 s |
| p95 latency, valid responses | 0.910 s | 2.084 s |
| Mean latency, all measured attempts | 0.810 s | 1.630 s |
| API-reported cost, including warmup | $0.009463 **(partial)** | $0.046290 |
| Attempts with known cost, including warmup | 300 / 301 | 301 / 301 |
| Mean cost per attempt with a known cost | $0.00003154 | $0.00015379 |

Luna's median latency was **2.40×** Jev's, and its mean known cost per call was
**4.88×** Jev's. These are observed wall-clock latencies and reported costs for
this workload. Jev's failed request took 41.03 seconds and had no reported cost;
its total above is therefore incomplete, not a definitive billed total.

### Repeatability and label limitations

| Repetition | Jev field accuracy | Luna field accuracy |
| --- | ---: | ---: |
| 1 | 96.6% (483 / 500) | 97.2% (486 / 500) |
| 2 | 95.2% (476 / 500) | 97.4% (487 / 500) |
| 3 | 96.6% (483 / 500) | 96.8% (484 / 500) |

Jev's HTTP 520 occurred in repetition 2 on `rev090`; it counted as five
incorrect fields. Repetitions reuse the same reviews and are not independent
new samples. No statistical significance claim is made.

Several disagreements expose ambiguity in the synthetic labels:

- Both models classified the wrong-size exchange case (`case012`) as shipping
  rather than the expected product topic in all six variant/repetition calls.
- Both classified the account-login problem (`case050`) as support rather than
  the expected other topic in all six calls.
- Both read the crushed-box complaint that was already resolved (`case026`) as
  positive rather than the expected negative sentiment in all six calls.

These are disagreements with the chosen rubric/labels, not necessarily obvious
model failures. Labels were **not changed after seeing the outputs**. The next
quality evaluation should clarify these boundaries before running on a separate,
independently labeled set of real reviews.

### Run evidence

- Run ID: `20260918T093134.708601Z`; manifest status: `complete`.
- Requested Jev: `~typesafe/jev-latest`; returned model:
  `typesafe/jev-1.13-20260917`, provider `TypeSafe`.
- Requested/returned Luna: `openai/gpt-5.6-luna`, provider `OpenAI`.
- [Portable metrics, confusion matrices, configuration, and source hashes](reports/2026-09-18-benchmark.json).
- Local raw responses and full manifest:
  `results/20260918T093134.708601Z/` (ignored by Git).
- Dataset SHA-256:
  `79b1e3b16e3582d62a408100e71067c271379a1c05dada0af6e30d7fae595e68`.

## Setup and offline checks

```bash
git clone https://github.com/mameli/jev-vs-luna.git
cd jev-vs-luna
uv sync
uv run pytest -q
uv run compare_models.py --dry-run
```

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) first if
needed. Offline tests and the dry run do not require an API key. Before running
paid API calls, copy `.env.example` to `.env` and set `OPENROUTER_API_KEY`.

The default test suite makes no API calls, even if a key is configured.
Use `uv run pytest --live -q` to include the paid Jev integration tests.
Never commit `.env` or credentials.

## Run a comparison

```bash
# Small paid smoke run: 6 measured calls + 2 warmup calls.
uv run compare_models.py --limit 3 --repeats 1
# Full comparison: 600 measured calls + 2 warmup calls.
uv run compare_models.py --repeats 3 --seed 42
```

Options include `--data`, `--model`, `--effort`, `--timeout`, `--warmup`,
`--output`, and `--limit`. Zero means no limit; repeats must be positive.
Luna uses strict JSON Schema and requires a provider that supports the request
parameters. An unsupported model/configuration is reported as an error; it is
not silently replaced with a different mode. Reasoning uses OpenRouter's
`reasoning.effort` parameter.

Reference: [OpenRouter structured outputs](https://openrouter.ai/docs/guides/features/structured-outputs).

## Dataset and label policy

`data/reviews_100.json` contains 50 explicitly labeled synthetic cases, each
with two contextual variants (100 distinct texts). The second variant adds
neutral order context. Variants share `case_id` and are **not independent
samples**. Regenerate the file with `uv run make_dataset.py`.

Cases cover ordinary feedback, mixed sentiment, sarcasm, negation, historical
defects, resolved complaints, and positive/neutral reviews with open questions.
Labels are manually specified per case; `needs_reply` is not derived from
sentiment. Historical defects still count after resolution; damage to packaging
alone does not count as a product defect. Stars are inferred from sentiment:
negative 1–2, neutral 3, positive 4–5. They are not observed customer ratings.

The dataset is intentionally small and synthetic, with unequal class counts.
It is useful for debugging and controlled latency measurements, but cannot
establish real-world superiority. Review labels independently and add held-out
real reviews before drawing broader quality conclusions. Do not tune prompts on
the same cases used to claim generalization. No confidence intervals are
presented: repeats and contextual variants are correlated observations.

## Measurement and saved evidence

- Each review is sent once to each model per repetition. All repetitions count
  toward accuracy; failed calls count as incorrect for every field.
- Review order is seeded and shuffled each repetition. Which model goes first
  is balanced within each repetition (within one pair for odd dataset sizes).
  Calls run sequentially with the same timeout and no automatic retries.
- One warmup per model is the default. Warmups are saved and included in reported
  cost coverage, but excluded from accuracy and measured latency.
- Latency is client wall time including request and response validation, not
  inference-only time. All-attempt and successful-call statistics are separate;
  p95 uses the nearest-rank definition. An all-failure run remains reportable.
- Reports include per-field accuracy, micro accuracy, exact match, macro F1 for
  categorical/boolean fields, and confusion matrices with an error column.
  Macro F1 excludes labels absent from both expected and predicted values.
  Rating and sentiment are correlated, so the five-field micro score should
  not be treated as five independent quality measurements.
- Cost comes only from numeric API-reported `usage.cost` in USD. Missing cost is
  unknown, never zero. Check `cost_complete` and coverage before comparing totals.
  Raw usage is preserved for token analysis; no hard-coded prices are used.

Each run creates a unique `results/<UTC timestamp>/` directory containing:

| File | Contents |
| --- | --- |
| `manifest.json` | Configuration, models, rubric, selected reviews, source/data hashes, completion status |
| `attempts.jsonl` | Flushed record of every completed attempt, raw payload, prediction, latency, and error type |
| `summary.json` | Aggregated metrics and cost coverage |

Interrupted runs preserve completed attempts and have `status: incomplete`.
Their summary covers only recorded attempts and must not be compared as if all
scheduled pairs completed. A request interrupted in flight may still be billed
without a saved response. Provider routing, caching, and mutable model aliases
can change over time; keep raw response metadata when comparing runs.
Results may contain review text and provider responses; `results/` is ignored
by Git. No credentials are written to the manifest.

## Files

- `review_task.py`: shared rubric, output schema, strict validation.
- `jev_client.py`: minimal Decisions API client.
- `compare_models.py`: paired benchmark and durable reports.
- `make_dataset.py`: explicit English cases and deterministic generation.
- `example.py`: support-ticket routing example (paid API call).
- `example_reviews.py`: four-review example (paid API calls).
- `tests/`: offline regression coverage and opt-in Jev integration tests.
