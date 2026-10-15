# Indic IFBench

Evaluate the supplied AI4Bharat IFBench translations in NeMo Gym using the
[Indic IFBench resources server](../../../resources_servers/indic_ifbench/README.md).
The supplied `data/translation/<language>/train.jsonl` files are used as benchmark
evaluation data. Their filename does not change their role in this integration.

| Requested code | Dataset code | Language | Supplied prompts |
| --- | --- | --- | ---: |
| bn | bn | Bengali | 288 |
| gu | gu | Gujarati | 288 |
| hi | hi | Hindi | 288 |
| ka | kn | Kannada | 288 |
| mar | mr | Marathi | 288 |
| mal | ml | Malayalam | 288 |
| ne | ne | Nepali | 288 |
| or | or | Odia | 288 |
| pa | pa | Punjabi | 288 |
| ta | ta | Tamil | 288 |
| te | te | Telugu | 288 |
| ur | ur | Urdu | 252 |

All twelve languages are selected by default: **3,420 tasks**. Missing languages
or invalid rows cause preparation to fail. Unequal language counts are preserved;
Urdu is not padded with nonexistent tasks. The source paths and SHA-256 checksums
are recorded next to the prepared JSONL. Full source and prepared datasets remain
local and gitignored; five representative requests are committed for smoke tests.

## Prepare

Run from the repository root. Preparation uses only the core Gym dependencies;
the Unicode/emoji libraries are installed with the resources server at runtime:

```bash
gym eval prepare --benchmark indic/ifbench \
  +prepare_script_args.dataset_dir=/path/to/ai4bharat_dataset/data
```

For a selected language set (aliases are accepted):

```bash
gym eval prepare --benchmark indic/ifbench \
  +prepare_script_args.dataset_dir=/path/to/ai4bharat_dataset/data \
  '+prepare_script_args.languages=[bn,gu,hi,ka,mar,mal,ne,or,pa,ta,te,ur]'
```

The standalone preparer is also available:

```bash
python -m benchmarks.indic.ifbench.prepare \
  --dataset-dir /path/to/ai4bharat_dataset/data \
  --languages hi ur --output-fpath /tmp/indic-ifbench-hi-ur.jsonl
```

Preparation preserves prompt text and translated kwargs, assigns stable
language-qualified identities, and sets language metadata for analysis. The
prompt template sends only the translated `prompt`; evaluator metadata is never
inserted into the model input. No system instruction is added that could conflict
with the individual task's exact-output requirements.

## Run

```bash
gym eval run --benchmark indic/ifbench \
  --model-type vllm_model \
  --model-url http://MODEL_HOST:8000/v1 \
  --model-api-key EMPTY --model MODEL_NAME \
  --output results/indic-ifbench.jsonl --split benchmark \
  --resume ++reuse_existing_data_preparation=true
```

The benchmark config uses five repeats and a one-step `simple_agent`. Use identical
model and sampling settings across languages. Reusing prepared data evaluates the
language selection from the last preparation, so prepare again when changing it.

For a small real-model smoke run using the five committed examples:

```bash
gym env start --config resources_servers/indic_ifbench/configs/indic_ifbench.yaml \
  --model-type vllm_model \
  +policy_base_url=http://MODEL_HOST:8000/v1 \
  +policy_api_key=EMPTY +policy_model_name=MODEL_NAME

gym eval run --no-serve --agent indic_ifbench_simple_agent \
  --input resources_servers/indic_ifbench/data/example.jsonl \
  --output results/indic-ifbench-smoke.jsonl \
  --num-repeats 1 --max-output-tokens 4096 --temperature 0
```

## Scores and validation status

Native Gym metrics include prompt-level and instruction-level strict/loose
accuracy, per-language scores, and equal-language macro averages. Default reward
is the fraction of strict instructions followed, as in Gym's English IFBench.
For an independent summary of completed rollouts:

```bash
python -m benchmarks.indic.ifbench.summarize results/indic-ifbench.jsonl \
  --output results/indic-ifbench-summary.json
```

The summarizer counts each repeat as an attempt, rejects incompatible evaluator
versions, and reports excluded verifier failures. Overall accuracy weights all
attempts/instructions; macro accuracy weights each language equally, which matters
because Urdu has fewer rows. Partial runs report only evaluated rows/languages.

This is an experimental evaluator created for these translations. Read its
[versioned language policy](../../../resources_servers/indic_ifbench/README.md#version-1-language-policy),
especially the exact-match lexical inventories and orthographic syllable
approximation. Native-speaker review and real-model baselining are still required
before setting `verified: true` or treating the scores as an official reference.
