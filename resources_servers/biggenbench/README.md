# BigGenBench rubric verifier

Absolute 1–5 ratings with an instance-specific rubric and reference answer;
Gym reward is `(score - 1) / 4`. Uses Gym's judge client, model-server concurrency,
failure routing, and aggregation. Invalid verdicts are masked from quality metrics.

## Component contract

[`app.py`](app.py) extends `SimpleResourcesServer`. The async `verify()` method
accepts a Gym response and task metadata described by [`TaskData`](task_data.py):
task ID, language, capability, task name, original system prompt, user input,
reference answer, and descriptions for scores 1–5. The reference and rubric
belong in `verifier_metadata`; they are not candidate-model input.

The verifier extracts final assistant text, excluding reasoning and commentary,
then renders [`prompts/absolute.yaml`](prompts/absolute.yaml) with Gym's
`load_prompt_config` and `fill_prompt`. `call_judge` sends the request through a
configured Gym model server. The judge must finish normally and end its feedback
with exactly one `[RESULT] N`, where `N` is an integer from 1 to 5.

The response preserves the raw score, normalized reward, judge feedback, finish
reason, and error diagnostics. Empty candidate answers receive score 1 without a
judge call. Malformed or truncated judgments are masked; transport/schema errors
propagate to the base server's `judge_failsafe` and Gym's retryable failure sidecar.
`REVERIFY_MODE` is `STATELESS`, and repeated verification clears stale result fields.

[`metrics.py`](metrics.py) supplies benchmark-specific scoring through Gym's
aggregation hooks. It weights capabilities equally within each language and then
weights languages equally, while retaining coverage and failure counts.

## Reuse and configuration

[`configs/biggenbench.yaml`](configs/biggenbench.yaml) composes the verifier with
Gym's `simple_agent`, `legacy_agent` compatibility environment adapter, and
`openai_model` judge. Select a policy model through the usual Gym model config and
provide `judge_base_url`, `judge_api_key`, and `judge_model_name` at launch.
`judge_prompt_path` and `judge_chat_completions_create_params` configure the prompt
and judge generation. Dataset-specific paths and judge selection are supplied by
the benchmark config, so the verifier can also score other correctly prepared
BiGGen Bench datasets.

Tests cover rubric rendering, answer extraction, score parsing, failures,
reverification, and aggregation. `VERIFIER_FIXTURE` provides full-reward,
zero-reward, and malformed-verdict cases for Gym's manifest validation workflow.

See [Indic BigGenBench](../../benchmarks/indic/biggenbench/README.md) for dataset
preparation, language aliases, model settings, scoring, and Gym evaluation commands.
