# Indic BigGenBench

Evaluate [Indic BigGenBench](https://huggingface.co/datasets/anushakamathofficial/indic_biggenbench)
using original [BiGGen Bench](https://aclanthology.org/2025.naacl-long.303/) rubrics
and Arena Hard's inference-hub judge, `gcp/google/gemini-3.8-flash`.
Generation, model serving, retries, artifacts, aggregation, and Slurm submission
use Gym's existing infrastructure.

## Architecture

The manifest declares Gym's `custom-gym-verifier` integration profile. Dataset
preparation and evaluation settings live in this benchmark directory; reusable
rubric verification lives in [`resources_servers/biggenbench`](../../../resources_servers/biggenbench/README.md).

| Responsibility | Implementation | Ownership |
| --- | --- | --- |
| Dataset preparation | [`prepare.py`](prepare.py) | Join translated inputs to validated upstream references/rubrics and write Gym JSONL rows. |
| Workload discovery and configuration | [`manifest.yaml`](manifest.yaml), [`config.yaml`](config.yaml) | Declare datasets, prompt ownership, and the selected judge. |
| Environment entry point | Gym's [`legacy_agent`](../../../environment_servers/legacy_agent/app.py) | Relay rollout requests to the configured agent. |
| Candidate generation | Gym's [`simple_agent`](../../../responses_api_agents/simple_agent/app.py) and [`vllm_model`](../../../responses_api_models/vllm_model/app.py) | Run the policy model and pass its completed response to verification. |
| Rubric verification | [`BigGenBenchResourcesServer`](../../../resources_servers/biggenbench/app.py), extending `SimpleResourcesServer` | Extract the final answer, render the task rubric, and parse the judge's rating. |
| Judge transport | Gym's `call_judge`, `ServerClient`, and [`openai_model`](../../../responses_api_models/openai_model/app.py) | Send Chat Completions requests; reuse model-server concurrency and Gym's judge-failure routing. |
| Metrics and execution | Gym aggregation and `gym eval prepare/run/submit` | Apply benchmark-specific metric hooks within Gym's collection and execution lifecycle. |

Each prepared row separates `responses_create_params.input` from
`verifier_metadata`. The candidate receives the original system prompt and
translated user input. The verifier alone receives the reference answer and
rubric, together with the candidate's completed response. Judge messages are
rendered from [`absolute.yaml`](../../../resources_servers/biggenbench/prompts/absolute.yaml)
using Gym's cached `load_prompt_config` and `fill_prompt` helpers.

The configured `legacy_agent` is Gym's supported compatibility adapter for the
agent-owned `/run` lifecycle. This integration does not use the newer
`single_agent_turn` session protocol. Moving to that protocol would require a
separate configuration migration and rollout validation. There is no custom
candidate server, agent loop, HTTP client, or Slurm launcher in this benchmark.

## Prepare

Defaults select `bn gu hi kn mr ml ne or pa ta te ur`; aliases `ka`, `mar`, and
`mal` map to `kn`, `mr`, and `ml`. Each language has 695 examples: 8,340 total.
The Indic release contains only translated user inputs. Preparation joins their
IDs to pinned upstream references/rubrics, verifies every English input exactly,
and rejects missing or duplicate IDs, incomplete rubrics, and language mismatches.
Original system prompts, references, and rubrics retain their original language;
some tasks explicitly request English answers. Only the system and translated
user messages are sent to the policy model.

```bash
# Uses pinned Hugging Face revisions and standard HF authentication/cache.
gym eval prepare --benchmark indic/biggenbench

# Offline alternative using downloaded dataset repositories/parquet.
python -m benchmarks.indic.biggenbench.prepare \
  --dataset-dir /path/to/indic_biggenbench \
  --upstream-file /path/to/original_biggenbench.parquet
```

Preparation writes the full benchmark, a 12-language smoke dataset, five resource
examples, and one provenance manifest. Generated datasets/downloads are ignored
by Git; the original datasets retain CC-BY-SA-4.0 licensing. Source revisions and
checksums are in `prepare.py` and `data/prepare_manifest.json`.

## Evaluate

Run commands from an installed Gym checkout. The preparation script uses
`pyarrow` and `huggingface_hub`, available with the repository's development setup.
Set `NVIDIA_API_KEY` for the inference-hub judge and serve your response model
through an OpenAI-compatible endpoint. The `validation` split is the 12-language
smoke; use `--split benchmark` for the complete evaluation.

```bash
gym eval run --benchmark indic/biggenbench --split validation \
  --model-type vllm_model --output results/indic_biggenbench_smoke.jsonl \
  +policy_model_name=google/gemma-4-31B-it \
  +policy_base_url=http://localhost:8000/v1 +policy_api_key=dummy \
  +responses_create_params.temperature=1.0 \
  +responses_create_params.top_p=0.95 \
  +responses_create_params.max_output_tokens=32768 \
  +policy_model.responses_api_models.vllm_model.chat_template_kwargs.enable_thinking=true \
  +policy_model.responses_api_models.vllm_model.sampling_overrides.top_k=64
```

For Slurm, [`gemma_smoke.yaml`](gemma_smoke.yaml) supplies a `gym eval submit`
recipe for two H100 GPUs. The candidate runtime must support Gemma 4 and its
`gemma4` reasoning parser. The recipe uses these settings:

| Setting | Candidate | Judge |
| --- | --- | --- |
| Model | `google/gemma-4-31B-it` | `gcp/google/gemini-3.8-flash` |
| Thinking/reasoning | `enable_thinking: true` | `reasoning_effort: low` |
| Output token limit | `32768`, including reasoning and final answer | `16384` |
| Temperature | `1.0` | `0.0` |
| Top-p / top-k | `0.95` / `64` | Provider defaults |

The candidate server allows 65,536 context tokens for prompt and output combined.
Judge settings are composed from the benchmark config and
[`biggenbench.yaml`](../../../resources_servers/biggenbench/configs/biggenbench.yaml);
they can be overridden through Gym's configuration. Credentials are supplied at runtime.

Set `SLURM_ACCOUNT`, `BIGGEN_VLLM_CONTAINER`,
`BIGGEN_MODEL_PATH`, `BIGGEN_VLLM_BIN`, `BIGGEN_SHARED_ROOT`, `BIGGEN_GYM_REPO`,
`BIGGEN_DATASET_DIR`, and `BIGGEN_OUTPUT_DIR` to your cluster paths.
`BIGGEN_GYM_REPO` is a repository URL or mounted checkout with branch
`indic/biggenbench`. `NVIDIA_API_KEY` is forwarded at runtime.

```bash
gym eval submit --config benchmarks/indic/biggenbench/gemma_smoke.yaml --dry-run
# Omit --dry-run to submit. Results and logs use Gym's standard job directory.
```

## Scores and validation

The judge receives the question, original system instruction, response, reference,
and five score descriptions. It returns feedback ending in `[RESULT] N` (1–5).
`score` retains that rating; Gym `reward` is `(score - 1) / 4`. Final-answer
extraction excludes reasoning/commentary. Transport errors use Gym's retryable
failure sidecar; malformed/truncated judgments are masked from quality metrics.
An empty final candidate answer receives score 1 without calling the judge.
Reverification is stateless and clears previous result/error fields before grading.

The headline score averages capability means equally within each language, then
averages language scores equally. It excludes `moral_belief` and any upstream
`multilingual` examples, leaving 8,220 included examples here. `score_all` retains
those categories; `score_micro` pools included ratings. Language, capability, task,
and valid/error counts are in Gym's aggregate metrics. Compare scores only with
the same judge/settings, and inspect coverage when a run is partial.

For a submitted job, inspect the benchmark's `artifacts/` directory:

| Artifact | What to check |
| --- | --- |
| `rollouts.jsonl` | Candidate outputs, raw `score`, normalized `reward`, `judge_feedback`, and masking/failure fields. |
| `rollouts_failures.jsonl` | Failed attempts routed by Gym; check this alongside successful outputs. |
| `rollouts_aggregate_metrics.json` | Headline score, per-language/capability/task scores, and valid/error counts. |
| `quality_summary.json` | Rollout health diagnostics produced with observability enabled. |

The thinking-enabled Gemma smoke completed all 12 languages with valid judgments
and no failures or truncation (Slurm job `20266359`, exit code `0:0`). It used the
settings above. Reasoning was observed in every response, with 1,479–2,333 output
tokens, and all 12 responses received score 5. The smoke selects the first row
per language: in this release, the same travel-planning task translated into each
language. It validates pipeline execution on that task, not full benchmark quality;
the resource remains `verified: false` and the manifest remains experimental.

The GPU smoke ran before the latest `main` rebase. After rebasing, 217 targeted
tests passed with 100% verifier coverage; manifest validation checked all 8,340
benchmark rows, 12 smoke rows, and five examples, and the Slurm dry-run passed.
GPU generation was not repeated after rebasing.

```bash
python -m pytest resources_servers/biggenbench/tests benchmarks/indic/biggenbench/tests
gym env validate --manifest benchmarks/indic/biggenbench/manifest.yaml --json
```
