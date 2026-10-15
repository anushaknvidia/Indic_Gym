# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare the supplied local Indic IFBench translation files for NeMo Gym."""

import argparse
import hashlib
import json
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

from resources_servers.indic_ifbench.task_data import EVALUATOR_VERSION, LANGUAGES, TaskData, normalize_language


BENCHMARK_DIR = Path(__file__).parent
OUTPUT_FPATH = BENCHMARK_DIR / "data/indic_ifbench_benchmark.jsonl"


def prepare(
    *,
    dataset_dir: str,
    languages: Sequence[str] = LANGUAGES,
    output_fpath: str | None = None,
) -> Path:
    """Validate all selected languages before writing tasks and source checksums.

    Accept either the supplied data directory or its translation subdirectory.
    The source calls these files train.jsonl; they are evaluation tasks here.
    No machine-specific default path or external dataset upload is required.
    """
    if isinstance(languages, str) or not languages:
        raise ValueError("languages must be a nonempty sequence of language codes")
    languages = [normalize_language(language) for language in languages]
    if len(set(languages)) != len(languages):
        raise ValueError("Duplicate language selection (including aliases)")
    root = Path(dataset_dir).expanduser()
    if (root / "translation").is_dir():
        root = root / "translation"
    rows = []
    sources = {}
    for language in languages:
        source = root / language / "train.jsonl"
        content = source.read_bytes()
        sources[language] = {"path": str(source.resolve()), "sha256": hashlib.sha256(content).hexdigest()}
        seen = set()
        for line_number, line in enumerate(content.decode("utf-8").splitlines(), 1):
            if not line.strip():
                continue
            entry = json.loads(line)
            key = str(entry.get("key", ""))
            if not key.isascii() or not key.isdecimal() or not 0 <= int(key) < 1_000_000:
                raise ValueError(f"{source}:{line_number}: key must be a nonnegative integer below 1000000")
            if int(key) in seen:
                raise ValueError(f"{source}:{line_number}: duplicate source key {key}")
            seen.add(int(key))
            task = TaskData(
                id=LANGUAGES.index(language) * 1_000_000 + int(key),
                language=language,
                prompt=entry["prompt"],
                instruction_id_list=entry["instruction_id_list"],
                kwargs=entry["kwargs"],
                source_key=key,
                uuid=f"indic-ifbench/{language}/{key}",
                metadata={"subset_for_metrics": language},
            )
            rows.append(task.model_dump())
        if not seen:
            raise ValueError(f"Empty dataset: {source}")
    output = Path(output_fpath) if output_fpath else OUTPUT_FPATH
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    audit = {
        "evaluator_version": EVALUATOR_VERSION,
        "languages": languages,
        "rows_per_language": dict(Counter(row["language"] for row in rows)),
        "instruction_counts": dict(Counter(i for row in rows for i in row["instruction_id_list"])),
        "sources": sources,
        "output_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
    }
    output.with_suffix(".preparation.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote {len(rows)} tasks to {output}; rows per language: {audit['rows_per_language']}")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--languages", nargs="+", default=LANGUAGES)
    parser.add_argument("--output-fpath")
    prepare(**vars(parser.parse_args()))
