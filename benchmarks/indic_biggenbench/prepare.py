# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Join translated Indic inputs to pinned original BiGGen-Bench judging metadata.

Original system prompts, references, and rubrics remain in English. Only system
prompts and translated inputs reach the policy model; no language instruction is
synthesized, even when the original system prompt explicitly requires English.
"""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download


HF_DATASET = "anushakamathofficial/indic_biggenbench"
HF_REVISION = "9818296bb33a0e196ba085201f9f315dc40b68a4"
UPSTREAM_DATASET = "prometheus-eval/BiGGen-Bench"
UPSTREAM_REVISION = "3a9589efbad801052bb2e153b44ce027498c27e4"
UPSTREAM_SHA256 = "b5ace27fb84b60e173ecbea787ebc8a4334f220d361d5776e12da82205dc7c71"
DEFAULT_LANGUAGES = ("bn", "gu", "hi", "kn", "mr", "ml", "ne", "or", "pa", "ta", "te", "ur")
LANGUAGE_ALIASES = {"ka": "kn", "mar": "mr", "mal": "ml"}
RUBRIC_KEYS = ("criteria", *(f"score{score}_description" for score in range(1, 6)))
OUTPUT_FPATH = Path(__file__).parent / "data" / "indic_biggenbench_benchmark.jsonl"
EXAMPLE_FPATH = Path(__file__).resolve().parents[2] / "resources_servers" / "biggenbench" / "data" / "example.jsonl"


def _text(row: dict[str, object], key: str, context: str, *, allow_empty: bool = False) -> str:
    value = row.get(key)
    if not isinstance(value, str) or (not allow_empty and not value.strip()):
        raise ValueError(f"Missing or empty {key} for {context}")
    return value


def _checksum(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _index_parquet(path: Path, *, language: str | None = None) -> dict[str, dict[str, object]]:
    rows = pq.read_table(path).to_pylist()
    if not rows:
        raise ValueError(f"No test rows found in {path}")
    result = {}
    for row in rows:
        row_id = _text(row, "row_id" if language else "id", str(path))
        context = f"({language or 'upstream'}, {row_id})"
        if row_id in result:
            raise ValueError(f"Duplicate question {context}")
        _text(row, "input", context)
        if language:
            if row.get("language_code") != language:
                raise ValueError(
                    f"Language mismatch for {row_id}: expected {language}, got {row.get('language_code')}"
                )
            _text(row, "language", context)
            if type(row.get("row_index")) is not int:
                raise ValueError(f"Missing integer row_index for {context}")
        else:
            for key in ("reference_answer", "capability", "task"):
                _text(row, key, context)
            _text(row, "system_prompt", context, allow_empty=True)
            rubric = row.get("score_rubric")
            if not isinstance(rubric, dict):
                raise ValueError(f"Missing score_rubric for {row_id}")
            for key in RUBRIC_KEYS:
                _text(rubric, key, f"rubric for {row_id}")
        result[row_id] = row
    return result


def _format_entry(*, row: dict[str, object], source: dict[str, object], language: str) -> dict[str, object]:
    messages = []
    if source["system_prompt"]:
        messages.append({"role": "system", "content": source["system_prompt"]})
    messages.append({"role": "user", "content": row["input"]})
    common = {
        "row_id": row["row_id"],
        "language": language,
        "capability": source["capability"],
        "task": source["task"],
        "include_in_aggregate": source["task"] != "moral_belief" and source["capability"] != "multilingual",
    }
    return {
        **common,
        "uid": f"{language}:{row['row_id']}",
        "subset_for_metrics": language,
        "responses_create_params": {"input": messages},
        "verifier_metadata": {
            **common,
            "id": row["row_id"],
            "row_index": row["row_index"],
            "language_name": row["language"],
            "input": row["input"],
            "source_input": source["input"],
            **{key: source[key] for key in ("system_prompt", "reference_answer", "score_rubric")},
            **{key: row.get(key) for key in ("judge_pass_stage", "human_evaluation_pending", "human_review_reason")},
        },
    }


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def prepare(
    *,
    dataset_dir: str | None = None,
    languages: list[str] | None = None,
    upstream_file: str | None = None,
    output_file: str | None = None,
    example_file: str | None = None,
) -> Path:
    """Prepare full and smoke JSONLs from pinned Hugging Face datasets or local parquet.

    Supply both ``dataset_dir`` and ``upstream_file`` for offline preparation.
    Custom files are identified by SHA-256 in the adjacent manifest, without
    claiming they match a pinned revision. All joins are validated before writes.
    """
    local_dir = Path(dataset_dir).expanduser().resolve() if dataset_dir else None
    supported = (
        sorted(path.parent.name for path in local_dir.glob("data/*/test.parquet"))
        if local_dir
        else [*DEFAULT_LANGUAGES, "en", "as", "sa"]
    )
    selected = list(dict.fromkeys(LANGUAGE_ALIASES.get(lang, lang) for lang in (languages or DEFAULT_LANGUAGES)))
    if languages == []:
        raise ValueError("Select at least one language")
    if invalid := set(selected) - set(supported):
        raise ValueError(f"Unsupported languages: {sorted(invalid)}. Supported: {supported}")

    upstream_path = (
        Path(upstream_file).expanduser().resolve()
        if upstream_file
        else Path(
            hf_hub_download(
                UPSTREAM_DATASET, "data/test-00000-of-00001.parquet", repo_type="dataset", revision=UPSTREAM_REVISION
            )
        )
    )
    upstream_hash = _checksum(upstream_path)
    if not upstream_file and upstream_hash != UPSTREAM_SHA256:
        raise ValueError(f"Upstream parquet checksum does not match the pinned source: {upstream_path}")
    upstream = _index_parquet(upstream_path)
    paths = {
        lang: local_dir / "data" / lang / "test.parquet"
        if local_dir
        else Path(hf_hub_download(HF_DATASET, f"data/{lang}/test.parquet", repo_type="dataset", revision=HF_REVISION))
        for lang in dict.fromkeys(["en", *selected])
    }
    english = _index_parquet(paths["en"], language="en")
    for row_id, row in english.items():
        if row_id not in upstream:
            raise ValueError(f"Missing upstream metadata for {row_id}")
        if row["input"] != upstream[row_id]["input"]:
            raise ValueError(f"English source input mismatch for {row_id}; cannot safely attach its rubric")

    entries, smoke = [], []
    for language in selected:
        local = _index_parquet(paths[language], language=language)
        if local.keys() != english.keys():
            raise ValueError(f"Source ID alignment mismatch for {language}")
        for row_id, row in local.items():
            if row["row_index"] != english[row_id]["row_index"]:
                raise ValueError(f"Source row_index mismatch for ({language}, {row_id})")
        translated = [
            _format_entry(row=row, source=upstream[row_id], language=language) for row_id, row in local.items()
        ]
        entries.extend(translated)
        smoke.append(translated[0])

    output = Path(output_file).expanduser().resolve() if output_file else OUTPUT_FPATH
    _write_jsonl(output, entries)
    _write_jsonl(output.with_name("indic_biggenbench_smoke.jsonl"), smoke)
    _write_jsonl(Path(example_file).expanduser().resolve() if example_file else EXAMPLE_FPATH, smoke[:5])
    manifest = {
        "indic_dataset": HF_DATASET,
        "indic_revision": HF_REVISION if local_dir is None else None,
        "indic_dataset_dir": str(local_dir) if local_dir else None,
        "english_source_sha256": _checksum(paths["en"]),
        "upstream_dataset": UPSTREAM_DATASET,
        "upstream_revision": UPSTREAM_REVISION if upstream_hash == UPSTREAM_SHA256 else None,
        "upstream_sha256": upstream_hash,
        "upstream_file": str(upstream_path),
        "rubric_language": "en",
        "reference_answer_language": "en",
        "system_prompt_language": "en",
        "languages": selected,
        "rows": len(entries),
        "aggregate_rows": sum(bool(row["include_in_aggregate"]) for row in entries),
        "rows_by_language": dict(Counter(str(row["language"]) for row in entries)),
        "rows_by_capability": dict(Counter(str(row["capability"]) for row in entries)),
        "source_files": {lang: {"path": str(path), "sha256": _checksum(path)} for lang, path in paths.items()},
        "output_file": str(output),
        "output_sha256": _checksum(output),
        "aggregate_exclusions": {"task": ["moral_belief"], "capability": ["multilingual"]},
    }
    (output.parent / "prepare_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(entries)} problems ({manifest['aggregate_rows']} included in aggregate) to {output}")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-dir", help="Local data/<language>/test.parquet files; default: pinned Hugging Face data."
    )
    parser.add_argument("--languages", nargs="+", help="Default: 12 Indic languages; ka/mar/mal alias kn/mr/ml.")
    parser.add_argument(
        "--upstream-file", help="Local upstream parquet; use with --dataset-dir for offline preparation."
    )
    parser.add_argument("--output-file", help="Override the prepared benchmark JSONL path.")
    parser.add_argument("--example-file", help="Override the small resources-server example JSONL path.")
    prepare(**vars(parser.parse_args()))
