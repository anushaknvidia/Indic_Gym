<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Indic HumanEval

Indic HumanEval evaluates complete Python solutions in English and 14 Indic
languages with the original HumanEval tests. The pinned
`ai4bharat/indic-humaneval` test split contains 164 problems, producing 2,460
language/problem tasks. Gym reports overall and per-language pass@1.

## Reference parity

The prompt and code extraction preserve the post-training HumanEval protocol
used by the preceding evaluation workflow:

- The instruction, opening fence, trailing quote, whitespace, and translated
  prompt are preserved. Tests and canonical solutions never enter the model
  message.
- Extraction removes everything through the last `</think>`, takes the first
  fenced block, and falls back to the stripped raw response. A solution must
  include its own function definition and imports.
- The pinned `human-eval==1.0.3` checker executes the extracted solution with
  the dataset's unchanged `test` and `check(entry_point)`, matching the source
  Hugging Face `code_eval` program. This benchmark does not use EvalPlus tests.

Preparation pins source revision
`b264702f71126b743199e5904486da155500203f`, validates the exact 164-task
schema and IDs, and atomically writes the JSONL and a provenance manifest. A
local Parquet override must match the pinned file's SHA-256.

## Generation settings

Each prepared row carries the reference defaults: greedy decoding,
`max_output_tokens: 1024`, and stop strings `</s>`, `<|im_end|>`, and `<eos>`.
The stop strings are JSON-encoded in `responses_create_params.metadata.extra_body`
because they are vLLM-specific. Keeping the defaults on each row preserves them
when `--model-type` is selected and prevents them from affecting another
benchmark in a combined run. Explicit run-level generation flags still win.

Thinking mode is model-specific and is not forced by the benchmark. Configure
it explicitly when comparing models or runs.

## Prepare and run

From the repository root:

```bash
gym eval prepare --benchmark indic/human_eval

gym eval run --benchmark indic/human_eval \
  --model-type vllm_model \
  --model MODEL_NAME \
  --model-url http://MODEL_HOST:8000/v1 \
  --model-api-key dummy \
  --split benchmark \
  --output results/indic_humaneval/rollouts.jsonl
```

Preparation downloads the pinned Hugging Face revision. An exact local copy can
be used by invoking the preparation script directly:

```bash
python benchmarks/indic/human_eval/prepare.py \
  --dataset-path /path/to/indic-humaneval
```

Select languages with the full lowercase names. `odiya` is accepted as an
alias for the dataset's `odia` column:

```bash
python benchmarks/indic/human_eval/prepare.py \
  --languages hindi odia punjabi
```

The available languages are English, Assamese, Bengali, Gujarati, Hindi,
Kannada, Malayalam, Marathi, Nepali, Odia, Punjabi, Sanskrit, Tamil, Telugu,
and Urdu. Bodo, Dogri, Kashmiri, Konkani, Maithili, Manipuri, Santali, Sindhi,
and romanized variants are absent from this dataset.

## Scoring and validation

Passing every original test receives reward 1; incorrect code, syntax errors,
empty responses, and candidate timeouts receive reward 0. Worker infrastructure
failures are masked and excluded from scoring. Scores are percentages. With
more than one repeat, `pass@1[avg-of-N]/accuracy` is mean pass@1.

Preparation checks every source field, all 164 task identities, every selected
translation, and the pinned Parquet digest before replacing an existing output.
The focused tests execute all 164 canonical solutions and validate every one of
the 2,460 prepared rows against the resource-server task schema. Five inspected
real-model rollouts are committed with the resource server as smoke evidence;
they are not a full benchmark baseline.

Run the component and focused tests with:

```bash
gym env test --resources-server humaneval_reference
gym dataset collate \
  --config resources_servers/humaneval_reference/configs/humaneval_reference.yaml \
  --output-dir resources_servers/humaneval_reference/data \
  --mode example_validation
pytest resources_servers/humaneval_reference/tests tests/unit_tests/test_indic_humaneval.py
```

The checker runs model-generated Python. Its reliability guard and resource
limits contain common failures, but they are not a security boundary. The
upstream-compatible shared execution context also makes verdicts susceptible
to deliberately test-tampering code. Run this server in an isolated evaluation
container without credentials.

The [translated dataset](https://huggingface.co/datasets/ai4bharat/indic-humaneval)
declares Apache-2.0. The original
[HumanEval dataset and execution package](https://github.com/openai/human-eval)
are MIT-licensed. The Gym integration is Apache-2.0.
