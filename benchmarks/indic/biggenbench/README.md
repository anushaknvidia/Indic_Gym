# Indic BigGenBench

Evaluate [Indic BigGenBench](https://huggingface.co/datasets/anushakamathofficial/indic_biggenbench)
using original [BiGGen Bench](https://aclanthology.org/2025.naacl-long.303/) rubrics
and Arena Hard's inference-hub judge, `gcp/google/gemini-3.8-flash`.
Generation, model serving, retries, artifacts, aggregation, and Slurm submission
use Gym's existing infrastructure.

The benchmark lives in `benchmarks/indic/biggenbench`; the reusable verifier lives
in `resources_servers/biggenbench`. Judge messages use Gym's cached YAML prompt
loader and renderer, and requests use `call_judge` through the OpenAI model server.
The single-turn agent and environment adapter are Gym's standard components.

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

Set `NVIDIA_API_KEY` for the inference-hub judge and serve your response model
through an OpenAI-compatible endpoint. The `validation` split is the 12-language
smoke; use `--split benchmark` for the complete evaluation.

```bash
gym eval run --benchmark indic/biggenbench --split validation \
  --model-type vllm_model --output results/indic_biggenbench_smoke.jsonl \
  +policy_model_name=google/gemma-4-31B-it \
  +policy_base_url=http://localhost:8000/v1 +policy_api_key=dummy \
  +responses_create_params.temperature=0.0 \
  +responses_create_params.max_output_tokens=8192 \
  +policy_model.responses_api_models.vllm_model.chat_template_kwargs.enable_thinking=false
```

For Slurm, `gemma_smoke.yaml` supplies a native `gym eval submit` recipe matching
the tested two-H100 Gemma setup. Set `SLURM_ACCOUNT`, `BIGGEN_VLLM_CONTAINER`,
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

The headline score averages capability means equally within each language, then
averages language scores equally. It excludes `moral_belief` and any upstream
`multilingual` examples, leaving 8,220 included examples here. `score_all` retains
those categories; `score_micro` pools included ratings. Language, capability, task,
and valid/error counts are in Gym's aggregate metrics. Compare scores only with
the same judge/settings, and inspect coverage when a run is partial.

The original Gemma smoke completed all 12 languages with valid judgments and no
failures (Slurm job `19560894`): temperature 0, thinking disabled, 8,192 output
tokens, and Gemini low reasoning. This validates the pipeline, not a full model
baseline; the resource remains `verified: false`.

```bash
python -m pytest resources_servers/biggenbench/tests benchmarks/indic/biggenbench/tests
```
