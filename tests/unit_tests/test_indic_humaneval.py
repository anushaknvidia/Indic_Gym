# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from benchmarks.indic.human_eval import prepare as module
from nemo_gym.environment.manifest import load_manifest
from nemo_gym.task_data import load_task_data_schema, validate_jsonl_rows
from resources_servers.humaneval_reference.app import HumanEvalVerifyRequest


REPO_ROOT = Path(__file__).resolve().parents[2]


def source_row(index: int = 0) -> dict[str, str]:
    entry_point = f"solution_{index}"
    prompts = {
        column: f'def {entry_point}():\n    """{language} problem {index}."""\n'
        for language, column in module.PROMPT_COLUMNS.items()
    }
    return {
        "task_id": f"HumanEval/{index}",
        "entry_point": entry_point,
        "test": f"def check(candidate):\n    assert candidate() == {index}",
        "canonical_solution": f"    return {index}",
        **prompts,
    }


@pytest.fixture
def source_rows() -> list[dict[str, str]]:
    return [source_row(index) for index in range(module.EXPECTED_PROBLEMS)]


@pytest.fixture
def remote_source(monkeypatch, source_rows):
    calls = []

    def load(repo_id, *, split, revision):
        calls.append((repo_id, split, revision))
        return source_rows

    monkeypatch.setattr(module, "load_dataset", load)
    return calls


def test_pinned_remote_source_and_complete_language_rows(remote_source, source_rows, tmp_path) -> None:
    output = module.prepare(output_fpath=tmp_path / "prepared.jsonl")
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert remote_source == [(module.SOURCE_ID, module.SOURCE_SPLIT, module.SOURCE_REVISION)]
    assert len(rows) == module.EXPECTED_PROBLEMS * len(module.PROMPT_COLUMNS) == 2460
    assert len({row["uuid"] for row in rows}) == len(rows)
    assert {row["verifier_metadata"]["language"] for row in rows} == set(module.PROMPT_COLUMNS)
    adapter = load_task_data_schema(REPO_ROOT / "resources_servers/humaneval_reference")
    report = validate_jsonl_rows("humaneval_reference", adapter, str(output), output.read_text().splitlines())
    assert report.clean and report.rows == 2460, report.summary()

    hindi = next(row for row in rows if row["uuid"] == "indic/human_eval/hindi/0")
    assert hindi["responses_create_params"]["input"] == [
        {
            "role": "user",
            "content": module.PROMPT_TEMPLATE.format(prompt=source_rows[0]["prompt_Hindi_translation"]),
        }
    ]
    assert hindi["responses_create_params"]["temperature"] == 0.0
    assert hindi["responses_create_params"]["max_output_tokens"] == 1024
    assert json.loads(hindi["responses_create_params"]["metadata"]["extra_body"]) == {
        "stop": list(module.STOP_STRINGS)
    }
    assert hindi["verifier_metadata"] == {
        "task_id": "HumanEval/0",
        "language": "hindi",
        "test": source_rows[0]["test"],
        "entry_point": "solution_0",
    }
    assert "canonical_solution" not in output.read_text()
    assert "check(candidate)" not in hindi["responses_create_params"]["input"][0]["content"]

    provenance = json.loads(output.with_suffix(".manifest.json").read_text())
    assert provenance["prepared_rows"] == 2460
    assert provenance["prepared_sha256"] == module.sha256(output)
    assert provenance["source_revision"] == module.SOURCE_REVISION
    assert provenance["source_data_sha256"] == module.SOURCE_DATA_SHA256
    assert provenance["coverage"]["hindi"] == {"published_rows": 164, "selected_rows": 164}


def test_local_parquet_requires_pinned_hash_and_supports_odiya_alias(source_rows, tmp_path, monkeypatch) -> None:
    source = tmp_path / "dataset"
    source.mkdir()
    path = source / "test.parquet"
    pq.write_table(pa.Table.from_pylist(source_rows), path)
    with pytest.raises(ValueError, match="does not match pinned source"):
        module.load_source(str(source))

    monkeypatch.setattr(module, "SOURCE_DATA_SHA256", module.sha256(path))
    output = module.prepare(dataset_path=str(path), languages=["odiya"], output_fpath=tmp_path / "out.jsonl")
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert len(rows) == module.EXPECTED_PROBLEMS
    assert {row["verifier_metadata"]["language"] for row in rows} == {"odia"}
    assert source_rows[0]["prompt_Odia_translation"] in rows[0]["responses_create_params"]["input"][0]["content"]


@pytest.mark.parametrize(
    "languages",
    [[], "hindi", ["hindi", "hindi"], ["bodo"], ["hindi_roman"], ["odia", "odiya"], [1]],
)
def test_invalid_languages_fail_before_loading(monkeypatch, languages) -> None:
    monkeypatch.setattr(module, "load_source", lambda *args, **kwargs: pytest.fail("must reject before loading"))
    with pytest.raises(ValueError):
        module.prepare(languages=languages)


@pytest.mark.parametrize(
    ("mutate", "error"),
    [
        (lambda rows: rows[:-1], "Expected 164"),
        (lambda rows: rows + [rows[-1]], "Expected 164"),
        (lambda rows: [*rows[:-1], {**rows[-1], "task_id": "HumanEval/0"}], "Duplicate"),
        (lambda rows: [*rows[:-1], {**rows[-1], "task_id": "HumanEval/999"}], "IDs 0 through 163"),
        (lambda rows: [{**rows[0], "extra": "x"}, *rows[1:]], "source columns"),
        (lambda rows: [{**rows[0], "prompt_Hindi_translation": ""}, *rows[1:]], "missing or empty"),
        (lambda rows: [{**rows[0], "entry_point": "class"}, *rows[1:]], "entry_point"),
    ],
)
def test_invalid_source_schema_and_identity_are_rejected(source_rows, mutate, error) -> None:
    with pytest.raises(ValueError, match=error):
        module._validate_entries(mutate(source_rows))


def test_missing_or_unpinned_source_is_rejected(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        module.load_source(str(tmp_path / "missing.parquet"))
    with pytest.raises(ValueError, match="Remote dataset must be"):
        module.load_source("someone/other-dataset")


def test_prepare_is_atomic_across_validation_failure(source_rows, tmp_path, monkeypatch) -> None:
    metadata = {
        "source_id": module.SOURCE_ID,
        "source_revision": module.SOURCE_REVISION,
        "source_split": module.SOURCE_SPLIT,
        "source_license": module.SOURCE_LICENSE,
        "source_rows": module.EXPECTED_PROBLEMS,
        "source_data_sha256": module.SOURCE_DATA_SHA256,
    }
    monkeypatch.setattr(module, "load_source", lambda _: (source_rows, dict(metadata)))
    output = tmp_path / "prepared.jsonl"
    module.prepare(languages=["hindi"], output_fpath=output)
    original = output.read_bytes()
    monkeypatch.setattr(module, "load_source", lambda _: (_ for _ in ()).throw(ValueError("bad source")))
    with pytest.raises(ValueError, match="bad source"):
        module.prepare(languages=["hindi"], output_fpath=output)
    assert output.read_bytes() == original


def test_cli(monkeypatch) -> None:
    captured = {}
    monkeypatch.setattr(
        "sys.argv",
        ["prepare", "--dataset-path", "/data", "--languages", "hindi", "odia", "--output-fpath", "x"],
    )
    monkeypatch.setattr(module, "prepare", lambda **kwargs: captured.update(kwargs) or Path("out.jsonl"))
    module.main()
    assert captured == {"dataset_path": "/data", "languages": ["hindi", "odia"], "output_fpath": "x"}


def test_manifest_config_and_examples_are_consistent() -> None:
    from omegaconf import OmegaConf

    from nemo_gym.global_config import GlobalConfigDictParser, GlobalConfigDictParserConfig

    manifest = load_manifest(module.BENCHMARK_DIR / "manifest.yaml")
    assert manifest.name == "indic/human_eval"
    assert manifest.canonical_split == "benchmark"
    assert manifest.prompt_source == "prepared"
    assert manifest.datasets[0].num_repeats == 1

    config = GlobalConfigDictParser().parse(
        GlobalConfigDictParserConfig(
            initial_global_config_dict=OmegaConf.create(
                {
                    "config_paths": [
                        "responses_api_models/vllm_model/configs/vllm_model.yaml",
                        str(module.BENCHMARK_DIR / "config.yaml"),
                    ],
                    "policy_base_url": "http://unused/v1",
                    "policy_api_key": "dummy",
                    "policy_model_name": "test",
                }
            ),
            skip_load_from_cli=True,
            skip_load_from_dotenv=True,
            offline=True,
        )
    )
    agent_name = "indic_humaneval_simple_agent"
    dataset = config[agent_name].responses_api_agents.simple_agent.datasets[0]
    assert dataset.num_repeats == manifest.datasets[0].num_repeats == 1
    assert dataset.jsonl_fpath == manifest.datasets[0].jsonl_fpath
    assert config.indic_humaneval_environment_server.environment_servers.legacy_agent.agent_server.name == agent_name
    assert config.policy_model.responses_api_models.vllm_model.sampling_overrides is None
    assert not config.get("responses_create_params")

    example_path = REPO_ROOT / "resources_servers/humaneval_reference/data/example.jsonl"
    lines = example_path.read_text().splitlines()
    assert len(lines) == 5
    adapter = load_task_data_schema(REPO_ROOT / "resources_servers/humaneval_reference")
    report = validate_jsonl_rows("humaneval_reference", adapter, str(example_path), lines)
    assert report.clean, report.summary()
    for line in lines:
        row = json.loads(line)
        fields = HumanEvalVerifyRequest.model_fields
        fields["verifier_metadata"].annotation.model_validate(row["verifier_metadata"])
        fields["responses_create_params"].annotation.model_validate(row["responses_create_params"])
        assert row["uuid"].startswith("indic/human_eval/")
        assert json.loads(row["responses_create_params"]["metadata"]["extra_body"]) == {
            "stop": list(module.STOP_STRINGS)
        }


def _resolve_cli_config(flags: list[str]) -> dict:
    from omegaconf import OmegaConf

    from nemo_gym.cli.main import _merge_config_paths, build_parser
    from nemo_gym.global_config import GlobalConfigDictParser, GlobalConfigDictParserConfig

    args = build_parser().parse_args(
        [
            "eval",
            "run",
            "--benchmark",
            "indic/human_eval",
            "--model-type",
            "vllm_model",
            "--model",
            "test",
            "--model-url",
            "http://unused/v1",
            "--model-api-key",
            "dummy",
            *flags,
        ]
    )
    overrides = _merge_config_paths([token for flag in args._command.flags for token in flag.translate_to_hydra(args)])
    resolved = GlobalConfigDictParser().parse(
        GlobalConfigDictParserConfig(
            initial_global_config_dict=OmegaConf.from_dotlist([token.lstrip("+") for token in overrides]),
            skip_load_from_cli=True,
            skip_load_from_dotenv=True,
            offline=True,
        )
    )
    return OmegaConf.to_container(resolved, resolve=True)


def _materialize_dataset(config: dict, agent_name: str, source_rows: list[dict], tmp_path: Path) -> list[dict]:
    from nemo_gym.config_types import BenchmarkDatasetConfig
    from nemo_gym.train_data_utils import TrainDataProcessor

    dataset = dict(config[agent_name]["responses_api_agents"]["simple_agent"]["datasets"][0])
    source_path = tmp_path / f"{agent_name}.jsonl"
    source_path.write_text("".join(json.dumps(row) + "\n" for row in source_rows))
    dataset["jsonl_fpath"] = str(source_path)
    return [
        json.loads(line) | {"agent_ref": {"name": agent_name}}
        for line in TrainDataProcessor._iter_dataset_lines(None, BenchmarkDatasetConfig.model_validate(dataset))
    ]


def _collect_rows(config: dict, source_rows: list[dict], tmp_path: Path) -> list[dict]:
    from nemo_gym.rollout_collection import RolloutCollectionConfig, RolloutCollectionHelper

    prepared = tmp_path / "prepared.jsonl"
    prepared.write_text("".join(json.dumps(row) + "\n" for row in source_rows))
    collection = RolloutCollectionConfig.model_validate(
        config | {"input_jsonl_fpath": str(prepared), "output_jsonl_fpath": str(tmp_path / "out.jsonl")}
    )
    return RolloutCollectionHelper._preprocess_rows_from_config(None, collection)


@pytest.mark.parametrize(
    ("flags", "expected"),
    [
        ([], (0.0, None, 1024)),
        (["--temperature", "0.4", "--top-p", "0.8", "--max-output-tokens", "4096"], (0.4, 0.8, 4096)),
    ],
)
def test_model_type_preserves_row_defaults_and_cli_overrides(flags, expected, tmp_path) -> None:
    config = _resolve_cli_config(flags)
    source = module.format_entry(source_row(), language="hindi")
    prepared = _materialize_dataset(config, "indic_humaneval_simple_agent", [source], tmp_path)
    rows = _collect_rows(config, prepared, tmp_path)
    assert len(rows) == 1
    request = rows[0]["responses_create_params"]
    assert (request["temperature"], request.get("top_p"), request["max_output_tokens"]) == expected
    assert json.loads(request["metadata"]["extra_body"]) == {"stop": list(module.STOP_STRINGS)}
    assert config["policy_model"]["responses_api_models"]["vllm_model"]["sampling_overrides"] is None


def test_combined_benchmark_does_not_leak_stop_strings(tmp_path) -> None:
    from benchmarks.numb3rs.prepare import _format_row

    config = _resolve_cli_config(["--benchmark", "numb3rs"])
    indic_agent = "indic_humaneval_simple_agent"
    audio_agent = "numb3rs_asr_with_pc_simple_agent"
    indic = _materialize_dataset(config, indic_agent, [module.format_entry(source_row(), language="hindi")], tmp_path)
    audio = _format_row(
        {
            "original_text": "forty two",
            "text": "42",
            "file_name": "sample.wav",
            "duration": 5,
            "category": "NUMBER",
        },
        audio_prefix=str(tmp_path),
    )
    prepared = indic + _materialize_dataset(config, audio_agent, [audio], tmp_path)
    rows = _collect_rows(config, prepared, tmp_path)
    assert [row["agent_ref"]["name"] for row in rows] == [indic_agent, audio_agent]
    assert json.loads(rows[0]["responses_create_params"]["metadata"]["extra_body"]) == {
        "stop": list(module.STOP_STRINGS)
    }
    audio_metadata = rows[1]["responses_create_params"].get("metadata") or {}
    assert "extra_body" not in audio_metadata
