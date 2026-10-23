# Indic BiGGenBench usage

## Local data and language selection

The default preparation downloads `test.parquet` from
[`ai4bharat/indic-biggenbench`](https://huggingface.co/datasets/ai4bharat/indic-biggenbench)
at the pinned revision in `prepare.py`. A local directory containing that file
can be used directly:

```bash
python -m benchmarks.indic.biggenbench.prepare \
  --dataset-dir /path/to/indic-biggenbench \
  --languages hi ta
```

Omit `--languages` for `bn gu hi kn mr ml ne or pa ta te ur`. The full supported
set is `as bn en gu hi kn ml mr ne or pa sa ta te ur`; `ka`, `mar`, and `mal` are
accepted aliases. Missing or empty selected translations fail preparation.

Preparation writes these files under `benchmarks/indic/biggenbench/data/`:

| File | Contents |
| --- | --- |
| `indic_biggenbench_benchmark.jsonl` | All rows for the selected languages. |
| `indic_biggenbench_smoke.jsonl` | The first instance from each selected language. |
| `prepare_manifest.json` | Source revision, file hashes, language counts, and aggregate exclusions. |

It also refreshes `resources_servers/biggenbench/data/example.jsonl` with up to
five smoke rows. `--output-file` and `--example-file` override these destinations.
The smoke covers translations of the same first task; use the full split to
measure benchmark quality.

Local exports in `data/<language>/test.parquet` are also supported. For that
layout, preparation joins the translations to the original BiGGen Bench rubric
by ID and verifies the English text and row alignment. Supply
`--upstream-file /path/to/original_biggenbench.parquet` for an offline join;
otherwise the pinned upstream parquet is downloaded. The public AI4Bharat
`test.parquet` already includes the rubric and needs no join.

## Candidate and judge settings

The [Gemma smoke recipe](gemma_smoke.yaml) configures
`google/gemma-4-31B-it` with thinking enabled, a 32,768-token output budget
(including reasoning), temperature `1.0`, top-p `0.95`, and top-k `64`. The model
server allows 65,536 context tokens and uses the `gemma4` reasoning parser.

The judge uses Gym's `openai_model` server. Select its endpoint, credentials,
and model with `BIGGENBENCH_JUDGE_BASE_URL`, `BIGGENBENCH_JUDGE_API_KEY`, and
`BIGGENBENCH_JUDGE_MODEL`, as described in the
[verifier configuration](../../../resources_servers/biggenbench/README.md#configuring-the-judge).
The default judge settings are temperature `0.0`, `max_tokens: 16384`, and
`reasoning_effort: low`. Configure `judge_chat_completions_create_params` under
`biggenbench.resources_servers.biggenbench` for the selected provider.

## Slurm smoke

The recipe uses `gym eval submit`, two H100 GPUs, and a candidate runtime that
supports Gemma 4. Set the following environment variables to your cluster paths:

| Variable | Value |
| --- | --- |
| `SLURM_ACCOUNT` | Slurm account. |
| `SLURM_PARTITION` | Partition; defaults to `batch`. |
| `BIGGEN_VLLM_CONTAINER` | Container image with the candidate runtime. |
| `BIGGEN_MODEL_PATH` | Gemma model snapshot directory. |
| `BIGGEN_VLLM_BIN` | Directory containing the compatible `vllm` executable. |
| `BIGGEN_SHARED_ROOT` | Shared filesystem root mounted into both containers. |
| `BIGGEN_GYM_REPO` | Repository URL or mounted checkout with branch `indic/biggenbench`. |
| `BIGGEN_DATASET_DIR` | Mounted directory containing the public dataset's `test.parquet`. |
| `BIGGEN_OUTPUT_DIR` | Job output directory. |

The three `BIGGENBENCH_JUDGE_*` variables are forwarded at runtime. Submit with:

```bash
gym eval submit --config benchmarks/indic/biggenbench/gemma_smoke.yaml
```

Add `--dry-run` to inspect the generated submission. Gym manages model startup,
dataset preparation, collection, retries, and aggregation.

## Inspecting results

A submitted benchmark writes the following files in its `artifacts/` directory:

| Artifact | Contents |
| --- | --- |
| `rollouts.jsonl` | Candidate responses, `score`, `reward`, judge feedback, and error/mask fields. |
| `rollouts_failures.jsonl` | Failed attempts routed for retry. |
| `rollouts_aggregate_metrics.json` | Aggregate scores and coverage counts. |
| `quality_summary.json` | Rollout health diagnostics when observability is enabled. |

For local validation:

```bash
python -m pytest resources_servers/biggenbench/tests benchmarks/indic/biggenbench/tests
gym env validate --manifest benchmarks/indic/biggenbench/manifest.yaml --json
```
