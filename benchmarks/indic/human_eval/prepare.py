# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepare Indic HumanEval with the post-training reference protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
import keyword
import os
import re
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from datasets import load_dataset


BENCHMARK_DIR = Path(__file__).parent
OUTPUT_FPATH = BENCHMARK_DIR / "data" / "indic_humaneval_benchmark.jsonl"
BENCHMARK_ID = "indic/human_eval"
SOURCE_ID = "ai4bharat/indic-humaneval"
# Public Hugging Face dataset commit, not a credential.
SOURCE_REVISION = "b264702f71126b743199e5904486da155500203f"  # pragma: allowlist secret
SOURCE_SPLIT = "test"
SOURCE_LICENSE = "Apache-2.0"
SOURCE_DATA_SHA256 = "a397339491e8dbf96fdb48bbb46c09bbb5e643da2aa1c773ae7c39d6f38bc51e"
EXPECTED_PROBLEMS = 164
BASE_COLUMNS = {"task_id", "prompt", "canonical_solution", "test", "entry_point"}
PROMPT_COLUMNS = {
    "english": "prompt",
    **{
        language.lower(): f"prompt_{language}_translation"
        for language in (
            "Assamese",
            "Bengali",
            "Gujarati",
            "Hindi",
            "Kannada",
            "Malayalam",
            "Marathi",
            "Nepali",
            "Odia",
            "Punjabi",
            "Sanskrit",
            "Tamil",
            "Telugu",
            "Urdu",
        )
    },
}
SOURCE_COLUMNS = BASE_COLUMNS | set(PROMPT_COLUMNS.values())
STOP_STRINGS = ("</s>", "<|im_end|>", "<eos>")
# Preserve the reference's opening fence and trailing quote. Jinja's default
# rendering removes the YAML block scalar's final newline.
PROMPT_TEMPLATE = (
    "Think carefully and output the fully functional python code solution inside ```python ```.\n"
    "Make sure that it passes the tests. Here is the problem:\n\n"
    '```{prompt}"'
)


def sha256(path: Path) -> str:
    """Return a file's SHA-256 digest."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as handle:
            temp_path = Path(handle.name)
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def write_jsonl_with_manifest(path: Path, rows: Sequence[Mapping[str, Any]], metadata: Mapping[str, Any]) -> Path:
    """Atomically write prepared rows and their provenance manifest."""
    if not rows:
        raise ValueError("Refusing to write an empty benchmark dataset")
    _write_atomic(path, "".join(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n" for row in rows))
    manifest = {**metadata, "prepared_rows": len(rows), "prepared_sha256": sha256(path)}
    _write_atomic(
        path.with_suffix(".manifest.json"), json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    return path


def _normalize_languages(languages: Sequence[str] | None) -> list[str]:
    if isinstance(languages, str):
        raise ValueError("languages must be a sequence, not a string")
    selected = list(PROMPT_COLUMNS) if languages is None else list(languages)
    if not selected or any(not isinstance(language, str) for language in selected):
        raise ValueError("languages must contain supported language names")
    selected = ["odia" if language.lower() == "odiya" else language.lower() for language in selected]
    if len(selected) != len(set(selected)):
        raise ValueError("languages must contain unique names after alias normalization")
    unsupported = set(selected) - PROMPT_COLUMNS.keys()
    if unsupported:
        raise ValueError(f"Unsupported languages: {sorted(unsupported)}; choose from {list(PROMPT_COLUMNS)}")
    return selected


def _task_index(task_id: str) -> int:
    match = re.fullmatch(r"HumanEval/(0|[1-9]\d*)", task_id)
    if match is None:
        raise ValueError(f"Invalid HumanEval task_id: {task_id!r}")
    return int(match.group(1))


def _validate_entries(entries: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    if len(entries) != EXPECTED_PROBLEMS:
        raise ValueError(f"Expected {EXPECTED_PROBLEMS} HumanEval problems, found {len(entries)}")
    indexed: dict[int, Mapping[str, Any]] = {}
    for entry in entries:
        if not isinstance(entry, Mapping) or set(entry) != SOURCE_COLUMNS:
            raise ValueError("Unexpected Indic HumanEval source columns")
        for field in SOURCE_COLUMNS:
            if not isinstance(entry[field], str) or not entry[field].strip():
                raise ValueError(f"{entry.get('task_id', '<unknown>')}: missing or empty {field}")
        index = _task_index(entry["task_id"])
        if index in indexed:
            raise ValueError(f"Duplicate HumanEval task_id: {entry['task_id']}")
        if not entry["entry_point"].isidentifier() or keyword.iskeyword(entry["entry_point"]):
            raise ValueError(f"Invalid entry_point: {entry['entry_point']!r}")
        indexed[index] = entry
    expected = set(range(EXPECTED_PROBLEMS))
    if set(indexed) != expected:
        raise ValueError(f"Expected HumanEval task IDs 0 through {EXPECTED_PROBLEMS - 1}")
    return [indexed[index] for index in range(EXPECTED_PROBLEMS)]


def load_source(dataset_path: str = SOURCE_ID) -> tuple[list[Mapping[str, Any]], dict[str, Any]]:
    """Load and validate the pinned Hugging Face revision or its exact local Parquet."""
    source = Path(dataset_path)
    source_sha256 = None
    if source.exists():
        import pyarrow.parquet as pq

        source = source / f"{SOURCE_SPLIT}.parquet" if source.is_dir() else source
        if not source.is_file() or source.suffix != ".parquet":
            raise ValueError(f"Expected a Parquet file or dataset directory: {source}")
        source_sha256 = sha256(source)
        if source_sha256 != SOURCE_DATA_SHA256:
            raise ValueError(f"Local Indic HumanEval Parquet does not match pinned source SHA-256: {source_sha256}")
        entries = pq.read_table(source).to_pylist()
    else:
        if source.is_absolute() or dataset_path.endswith(".parquet"):
            raise FileNotFoundError(f"Indic HumanEval dataset not found: {source}")
        if dataset_path != SOURCE_ID:
            raise ValueError(f"Remote dataset must be the pinned source {SOURCE_ID!r}")
        entries = list(load_dataset(SOURCE_ID, split=SOURCE_SPLIT, revision=SOURCE_REVISION))

    validated = _validate_entries(entries)
    return validated, {
        "source_id": SOURCE_ID,
        "source_revision": SOURCE_REVISION,
        "source_split": SOURCE_SPLIT,
        "source_license": SOURCE_LICENSE,
        "source_rows": len(validated),
        "source_data_sha256": source_sha256 or SOURCE_DATA_SHA256,
    }


def format_entry(entry: Mapping[str, Any], *, language: str) -> dict[str, Any]:
    """Build one single-turn task without exposing tests or reference solutions."""
    if language not in PROMPT_COLUMNS:
        raise ValueError(f"Unsupported language: {language!r}")
    prompt_column = PROMPT_COLUMNS[language]
    for field in ("task_id", "test", "entry_point", prompt_column):
        if not isinstance(entry.get(field), str) or not entry[field].strip():
            raise ValueError(f"{entry.get('task_id', '<unknown>')}: missing or empty {field}")
    if not entry["entry_point"].isidentifier() or keyword.iskeyword(entry["entry_point"]):
        raise ValueError(f"Invalid entry_point: {entry['entry_point']!r}")
    return {
        "uuid": f"{BENCHMARK_ID}/{language}/{_task_index(entry['task_id'])}",
        "responses_create_params": {
            "input": [{"role": "user", "content": PROMPT_TEMPLATE.format(prompt=entry[prompt_column])}],
            "temperature": 0.0,
            "max_output_tokens": 1024,
            "metadata": {"extra_body": json.dumps({"stop": list(STOP_STRINGS)})},
        },
        "verifier_metadata": {
            "task_id": entry["task_id"],
            "language": language,
            "test": entry["test"],
            "entry_point": entry["entry_point"],
        },
    }


def prepare(
    *,
    dataset_path: str = SOURCE_ID,
    languages: Sequence[str] | None = None,
    output_fpath: str | Path = OUTPUT_FPATH,
) -> Path:
    """Validate the source and atomically write Gym rows plus provenance."""
    selected = _normalize_languages(languages)
    entries, metadata = load_source(dataset_path)
    rows = [format_entry(entry, language=language) for language in selected for entry in entries]
    metadata.update(
        {
            "benchmark_id": BENCHMARK_ID,
            "evaluation_protocol": "post_train_think_humaneval_original_tests",
            "languages": selected,
            "coverage": {
                language: {"published_rows": len(entries), "selected_rows": len(entries)} for language in selected
            },
            "adapter_sha256": sha256(Path(__file__)),
            "prompt_sha256": hashlib.sha256(PROMPT_TEMPLATE.encode()).hexdigest(),
        }
    )
    output_path = write_jsonl_with_manifest(Path(output_fpath), rows, metadata)
    print(f"Wrote {len(rows)} tasks ({len(entries)} problems x {len(selected)} languages) to {output_path}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", default=SOURCE_ID)
    parser.add_argument("--languages", nargs="+")
    parser.add_argument("--output-fpath", default=str(OUTPUT_FPATH))
    print(prepare(**vars(parser.parse_args())))


if __name__ == "__main__":
    main()
