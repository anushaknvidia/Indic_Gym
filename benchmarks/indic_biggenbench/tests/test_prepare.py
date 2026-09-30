# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from copy import deepcopy
from pathlib import Path
from unittest.mock import Mock

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from benchmarks.indic_biggenbench import prepare as module


def _write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path)


@pytest.fixture
def source(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict:
    upstream = [
        {
            "id": f"{capability}_{task}_0",
            "capability": capability,
            "task": task,
            "system_prompt": "You are a helpful assistant.",
            "input": f"Source question {task}.",
            "reference_answer": f"Unique hidden reference {task}.",
            "score_rubric": {key: f"Hidden rubric {task}: {key}." for key in module.RUBRIC_KEYS},
        }
        for capability, task in [
            ("planning", "travel_plan"),
            ("safety", "moral_belief"),
            ("multilingual", "translation"),
        ]
    ]
    rows = {
        language: [
            {
                "row_id": item["id"],
                "row_index": index,
                "input": item["input"] if language == "en" else f"{language} translation {index}",
                "language": name,
                "language_code": language,
                "judge_pass_stage": "first_judge_pass",
                "human_evaluation_pending": True,
                "human_review_reason": "random_sample",
            }
            for index, item in enumerate(upstream)
        ]
        for language, name in [(lang, f"Language {lang}") for lang in ("en", *module.DEFAULT_LANGUAGES)]
    }
    local_dir = tmp_path / "indic"
    for language, data in rows.items():
        _write(local_dir / "data" / language / "test.parquet", data)
    upstream_path = tmp_path / "official.parquet"
    _write(upstream_path, upstream)
    monkeypatch.setattr(module, "OUTPUT_FPATH", tmp_path / "prepared" / "test.jsonl")
    monkeypatch.setattr(module, "EXAMPLE_FPATH", tmp_path / "examples" / "example.jsonl")
    monkeypatch.setattr(
        module, "hf_hub_download", Mock(side_effect=AssertionError("Offline preparation accessed the network"))
    )
    return {"local_dir": local_dir, "upstream_path": upstream_path, "upstream": upstream, "rows": rows}


def _prepare(source: dict, **kwargs: object) -> list[dict]:
    kwargs.setdefault("languages", ["hi"])
    output = module.prepare(dataset_dir=str(source["local_dir"]), upstream_file=str(source["upstream_path"]), **kwargs)
    return [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]


def test_offline_join_preserves_inputs_without_judge_metadata_leakage(source: dict) -> None:
    rows = _prepare(source)
    source_row = source["upstream"][0]
    row = rows[0]
    assert row["uid"] == "hi:planning_travel_plan_0"
    assert row["subset_for_metrics"] == "hi"
    assert row["responses_create_params"]["input"] == [
        {"role": "system", "content": source_row["system_prompt"]},
        {"role": "user", "content": "hi translation 0"},
    ]
    metadata = row["verifier_metadata"]
    assert metadata["source_input"] == source_row["input"]
    assert metadata["input"] == "hi translation 0"
    assert metadata["reference_answer"] == source_row["reference_answer"]
    assert metadata["score_rubric"] == source_row["score_rubric"]
    assert metadata["human_evaluation_pending"] is True
    policy_input = json.dumps(row["responses_create_params"])
    assert source_row["reference_answer"] not in policy_input
    assert all(text not in policy_input for text in source_row["score_rubric"].values())
    assert rows[0]["include_in_aggregate"] is True
    assert [row["include_in_aggregate"] for row in rows] == [True, False, False]
    assert "provenance" not in metadata
    manifest = json.loads((module.OUTPUT_FPATH.parent / "prepare_manifest.json").read_text())
    assert manifest["rows"] == 3
    assert manifest["upstream_revision"] is None
    assert manifest["upstream_sha256"] == module._checksum(source["upstream_path"])
    assert manifest["aggregate_rows"] == 1
    assert manifest["output_sha256"] == module._checksum(module.OUTPUT_FPATH)
    module.hf_hub_download.assert_not_called()


def test_aliases_normalize_and_deduplicate_without_losing_requested_order(source: dict) -> None:
    rows = _prepare(source, languages=["ka", "kn", "mar", "mal", "hi"])
    assert list(dict.fromkeys(row["language"] for row in rows)) == ["kn", "mr", "ml", "hi"]
    assert len({row["uid"] for row in rows}) == len(rows) == 12


@pytest.mark.parametrize("languages,match", [([], "at least one"), (["xx"], "Unsupported languages")])
def test_rejects_unknown_or_empty_language_selection(source: dict, languages: list[str], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        _prepare(source, languages=languages)
    assert not module.OUTPUT_FPATH.exists()


@pytest.mark.parametrize(
    "problem,match",
    [
        ("duplicate", "Duplicate question"),
        ("language", "Language mismatch"),
        ("empty", "Missing or empty input"),
        ("id", "Source ID alignment mismatch"),
        ("row_index", "Source row_index mismatch"),
    ],
)
def test_rejects_bad_translation_alignment_before_replacing_output(source: dict, problem: str, match: str) -> None:
    rows = deepcopy(source["rows"]["hi"])
    if problem == "duplicate":
        rows.append(deepcopy(rows[0]))
    elif problem == "language":
        rows[0]["language_code"] = "kn"
    elif problem == "empty":
        rows[0]["input"] = " "
    elif problem == "id":
        rows.pop()
    else:
        rows[0]["row_index"] = 100
    _write(source["local_dir"] / "data" / "hi" / "test.parquet", rows)
    module.OUTPUT_FPATH.parent.mkdir()
    module.OUTPUT_FPATH.write_text("existing evaluation input\n")
    with pytest.raises(ValueError, match=match):
        _prepare(source)
    assert module.OUTPUT_FPATH.read_text() == "existing evaluation input\n"


@pytest.mark.parametrize(
    "problem,match",
    [
        ("duplicate", "Duplicate question"),
        ("missing_id", "Missing upstream metadata"),
        ("input", "English source input mismatch"),
        ("reference", "Missing or empty reference_answer"),
        ("rubric", "Missing or empty score5_description"),
        ("system", "Missing or empty system_prompt"),
    ],
)
def test_rejects_incomplete_or_misaligned_upstream(source: dict, problem: str, match: str) -> None:
    rows = deepcopy(source["upstream"])
    if problem == "duplicate":
        rows.append(deepcopy(rows[0]))
    elif problem == "missing_id":
        rows.pop()
    elif problem == "input":
        rows[0]["input"] = "Different official prompt."
    elif problem == "reference":
        rows[0]["reference_answer"] = ""
    elif problem == "rubric":
        rows[0]["score_rubric"].pop("score5_description")
    else:
        rows[0].pop("system_prompt")
    _write(source["upstream_path"], rows)
    with pytest.raises(ValueError, match=match):
        _prepare(source)
    assert not module.OUTPUT_FPATH.exists()


def test_empty_system_prompt_does_not_add_a_synthetic_message(source: dict) -> None:
    source["upstream"][0]["system_prompt"] = ""
    _write(source["upstream_path"], source["upstream"])
    row = _prepare(source)[0]
    assert row["responses_create_params"]["input"] == [{"role": "user", "content": "hi translation 0"}]


def test_join_uses_ids_instead_of_parquet_order(source: dict) -> None:
    _write(source["upstream_path"], list(reversed(source["upstream"])))
    row = _prepare(source)[0]
    assert row["verifier_metadata"]["reference_answer"] == source["upstream"][0]["reference_answer"]


def test_corrupt_downloaded_upstream_is_rejected(source: dict) -> None:
    module.hf_hub_download.side_effect = None
    module.hf_hub_download.return_value = str(source["upstream_path"])
    with pytest.raises(ValueError, match="checksum does not match"):
        module.prepare(dataset_dir=str(source["local_dir"]), languages=["hi"])
    assert not module.OUTPUT_FPATH.exists()


def test_no_argument_prepare_downloads_pinned_sources_and_writes_smoke(
    source: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    def download(repo: str, filename: str, *, repo_type: str, revision: str) -> str:
        assert repo_type == "dataset"
        if repo == module.UPSTREAM_DATASET:
            assert revision == module.UPSTREAM_REVISION
            assert filename == "data/test-00000-of-00001.parquet"
            return str(source["upstream_path"])
        assert repo == module.HF_DATASET
        assert revision == module.HF_REVISION
        return str(source["local_dir"] / filename)

    module.hf_hub_download.side_effect = download
    monkeypatch.setattr(module, "UPSTREAM_SHA256", module._checksum(source["upstream_path"]))
    output = module.prepare()
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    smoke = [json.loads(line) for line in output.with_name("indic_biggenbench_smoke.jsonl").read_text().splitlines()]
    examples = [json.loads(line) for line in module.EXAMPLE_FPATH.read_text().splitlines()]
    assert len(rows) == 3 * len(module.DEFAULT_LANGUAGES)
    assert [row["language"] for row in smoke] == list(module.DEFAULT_LANGUAGES)
    assert smoke == rows[::3]
    assert examples == smoke[:5]
    manifest = json.loads((output.parent / "prepare_manifest.json").read_text())
    assert manifest["indic_revision"] == module.HF_REVISION
    assert manifest["upstream_revision"] == module.UPSTREAM_REVISION
