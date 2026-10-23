# Indic BiGGenBench

Evaluate generation across Indic languages using
[AI4Bharat's Indic BiGGenBench](https://huggingface.co/datasets/ai4bharat/indic-biggenbench)
and the original task-specific [BiGGen Bench](https://aclanthology.org/2025.naacl-long.303/)
rubrics and reference answers.

## What it tests

The dataset contains 695 instances across 70 tasks and eight capabilities:
theory of mind, instruction following, reasoning, grounding, safety, refinement,
planning, and tool usage. A judge assigns each response a score from 1 to 5 using
that instance's rubric and reference answer.

This directory owns dataset preparation and benchmark configuration. The reusable
[rubric verifier](../../../resources_servers/biggenbench/README.md) extends Gym's
`SimpleResourcesServer` and uses its prompt renderer, `call_judge`, failure
routing, and aggregation hooks. Gym's `simple_agent`, `legacy_agent` environment
adapter, policy model server, and `openai_model` judge handle execution.

## Data

The source is `ai4bharat/indic-biggenbench`, pinned to revision
`9b9b351b0cac550aaf484808e7d9f4a6d4d92a18`. Its `test.parquet` contains the original
English fields, six flattened rubric fields, and translations in 14 languages.
Preparation validates the source checksum, IDs, rubric fields, and selected
translations before writing Gym JSONL.

The default languages are `bn gu hi kn mr ml ne or pa ta te ur`: 8,340 rows and a
12-row smoke split. Assamese (`as`), Sanskrit (`sa`), and the original English
(`en`) are also selectable. Aliases `ka`, `mar`, and `mal` select `kn`, `mr`, and
`ml`.

Each row places the original system prompt and translated user input in
`responses_create_params.input`; the reference answer and rubric are confined to
`verifier_metadata`. System prompts, references, and rubrics remain in English,
including instructions that explicitly request an English response. Preparation
also writes five resource examples and a manifest with source/output hashes and
language counts. The dataset license is CC-BY-SA-4.0.

## Example usage

Run from an installed Gym checkout. Configure the
[judge endpoint](../../../resources_servers/biggenbench/README.md#configuring-the-judge)
and serve a candidate through an OpenAI-compatible endpoint:

```bash
gym eval prepare --benchmark indic/biggenbench

gym eval run --benchmark indic/biggenbench --split benchmark \
  --model-type vllm_model --output results/indic_biggenbench.jsonl \
  +policy_model_name=google/gemma-4-31B-it \
  +policy_base_url=http://localhost:8000/v1 +policy_api_key=dummy \
  +responses_create_params.temperature=1.0 \
  +responses_create_params.top_p=0.95 \
  +responses_create_params.max_output_tokens=32768 \
  +policy_model.responses_api_models.vllm_model.chat_template_kwargs.enable_thinking=true \
  +policy_model.responses_api_models.vllm_model.sampling_overrides.top_k=64
```

Use `--split validation` for the 12-language smoke. See [usage.md](usage.md) for
local data, language selection, Slurm submission, and output artifacts.

## Metrics

The judge returns feedback ending in `[RESULT] N`. `score` retains the 1–5 rating;
Gym `reward` is `(score - 1) / 4`. Reasoning and commentary are excluded from the
answer being graded. An empty final answer receives score 1. Invalid or truncated
judgments are masked, and transport errors use Gym's retryable failure sidecar.

The headline score averages capability means within each language, then averages
language scores. It excludes `moral_belief` and `multilingual`, leaving 8,220
eligible rows in the default selection. `score_all` includes excluded categories;
`score_micro` pools eligible ratings. Aggregates also report language, capability,
task, and valid/error counts. Compare results with the same judge and generation
settings, and check coverage when interpreting partial runs.
