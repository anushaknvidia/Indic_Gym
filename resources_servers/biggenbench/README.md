# BiGGenBench rubric verifier

Scores a completed response from 1 to 5 against an instance-specific rubric and
reference answer. Gym reward is `(score - 1) / 4`.

## What it does

[`app.py`](app.py) extends `SimpleResourcesServer`. It extracts the final
assistant answer, renders [`prompts/absolute.yaml`](prompts/absolute.yaml) with
Gym's `load_prompt_config` and `fill_prompt`, and sends the request through
`call_judge` to a configured Gym model server. The judge returns feedback ending
in exactly one `[RESULT] N`, where `N` is an integer from 1 to 5.

The verifier preserves `score`, `reward`, judge feedback, finish reason, and error
diagnostics. It excludes reasoning/commentary from the graded answer. Empty
answers receive score 1 without a judge call; malformed or truncated verdicts
are masked. Transport/schema errors use the base server's `judge_failsafe` and
Gym's retryable failure sidecar. Reverification is stateless and clears prior
result fields.

## Data format

The candidate receives `responses_create_params.input`. The separate
`verifier_metadata` follows [`TaskData`](task_data.py):

| Field | Meaning |
| --- | --- |
| `id`, `language`, `capability`, `task` | Instance identity and metric groups. |
| `input`, `system_prompt` | User question and original system instruction. |
| `reference_answer` | Reference used only by the judge. |
| `score_rubric` | `criteria` and `score1_description` through `score5_description`. |

The [Indic benchmark](../../benchmarks/indic/biggenbench/README.md) prepares this
format from `ai4bharat/indic-biggenbench`.

## Configuring the judge

[`configs/biggenbench.yaml`](configs/biggenbench.yaml) composes this verifier
with Gym's `simple_agent`, `legacy_agent` environment adapter, and a dedicated
`biggenbench_judge_model` instance of `openai_model`. Supply an OpenAI-compatible
judge endpoint and model:

```bash
export BIGGENBENCH_JUDGE_BASE_URL=https://your-judge.example/v1
export BIGGENBENCH_JUDGE_MODEL=your-judge-model
read -rsp 'Judge API key: ' BIGGENBENCH_JUDGE_API_KEY
export BIGGENBENCH_JUDGE_API_KEY
```

These variables configure only `biggenbench_judge_model`. The candidate uses the
usual Gym policy model configuration. The judge defaults to temperature `0.0`,
`max_tokens: 16384`, `reasoning_effort: low`, and eight concurrent requests.
Override `judge_prompt_path` or `judge_chat_completions_create_params` under
`biggenbench.resources_servers.biggenbench` to configure the prompt or provider
parameters.

## Metrics

[`metrics.py`](metrics.py) implements Gym aggregation hooks. The headline score
weights capabilities equally within each language, then weights languages
equally. It excludes `moral_belief` and `multilingual` from that score while
retaining separate all-task, micro, language, capability, task, and coverage
metrics. Invalid judgments do not contribute to quality metrics.

See the [benchmark usage guide](../../benchmarks/indic/biggenbench/usage.md) for
preparation, candidate settings, evaluation commands, and output artifacts.
