# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from benchmarks.indic.ifbench import prepare as prep
from nemo_gym.benchmarks import BenchmarkConfig
from nemo_gym.environment.manifest import load_manifest
from nemo_gym.global_config import GlobalConfigDictParser, GlobalConfigDictParserConfig
from nemo_gym.prompt import apply_prompt_to_row, load_prompt_config
from nemo_gym.task_data import load_task_data_schema, validate_jsonl_rows
from resources_servers.indic_ifbench.metrics import summarize
from resources_servers.indic_ifbench.task_data import EVALUATOR_VERSION, LANGUAGES


@pytest.fixture
def source(tmp_path):
    for language in LANGUAGES:
        path = tmp_path / "translation" / language / "train.jsonl"
        path.parent.mkdir(parents=True)
        row = {
            "key": "3",
            "prompt": f"{language} १ २",
            "instruction_id_list": ["count:numbers"],
            "kwargs": [{"N": 2.0, "keyword": None}],
        }
        path.write_text(json.dumps(row) + "\n")
    return tmp_path


def test_prepare_all_languages_and_aliases(source):
    output = prep.prepare(dataset_dir=str(source), output_fpath=str(source / "out.jsonl"))
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert len(rows) == 12
    assert len({row["id"] for row in rows}) == 12
    assert len({row["uuid"] for row in rows}) == 12
    assert [row["language"] for row in rows] == list(LANGUAGES)
    assert rows[0]["kwargs"] == [{"N": 2}]
    audit = json.loads(output.with_suffix(".preparation.json").read_text())
    assert audit["rows_per_language"] == dict.fromkeys(LANGUAGES, 1)
    assert len(audit["sources"]["hi"]["sha256"]) == 64
    selected = prep.prepare(
        dataset_dir=str(source / "translation"),
        languages=["ka", "mar", "mal"],
        output_fpath=str(source / "selected.jsonl"),
    )
    assert [json.loads(s) for s in selected.read_text().splitlines()] == [
        r for lang in ("kn", "mr", "ml") for r in rows if r["language"] == lang
    ]
    prompt = load_prompt_config(str(prep.BENCHMARK_DIR / "prompts/default.yaml"))
    rendered = apply_prompt_to_row(rows[0], prompt)
    assert rendered["responses_create_params"]["input"] == [{"role": "user", "content": rows[0]["prompt"]}]
    assert "keyword" not in json.dumps(rendered["responses_create_params"])


@pytest.mark.parametrize("languages", [[], "hi", ["en"], ["kn", "ka"]])
def test_invalid_language_selection(source, languages):
    with pytest.raises(ValueError):
        prep.prepare(dataset_dir=str(source), languages=languages)


@pytest.mark.parametrize("corrupt", ["duplicate", "key", "empty", "kwargs"])
def test_prepare_failure_preserves_output(source, corrupt):
    file = source / "translation/ur/train.jsonl"
    row = json.loads(file.read_text())
    if corrupt == "duplicate":
        file.write_text(file.read_text() * 2)
    elif corrupt == "empty":
        file.write_text("")
    else:
        row[corrupt] = "bad" if corrupt == "key" else []
        file.write_text(json.dumps(row) + "\n")
    output = source / "out.jsonl"
    output.write_text("previous output")
    with pytest.raises(ValueError):
        prep.prepare(dataset_dir=str(source), output_fpath=str(output))
    assert output.read_text() == "previous output"


def test_native_manifest_config_and_task_schema():
    manifest = load_manifest(prep.BENCHMARK_DIR / "manifest.yaml")
    assert manifest.name == "indic/ifbench"
    assert manifest.resources_server == "indic_ifbench"
    benchmark = BenchmarkConfig.from_config_path(prep.BENCHMARK_DIR / "config.yaml")
    assert benchmark.name == "indic_ifbench"
    parser = GlobalConfigDictParser()
    config = parser.parse(
        GlobalConfigDictParserConfig(
            initial_global_config_dict=OmegaConf.create(
                {
                    "config_paths": [
                        "responses_api_models/vllm_model/configs/vllm_model.yaml",
                        str(prep.BENCHMARK_DIR / "config.yaml"),
                    ],
                    "policy_base_url": "http://unused/v1",
                    "policy_api_key": "dummy",
                    "policy_model_name": "test",
                    "error_on_agent_without_environment_server": True,
                }
            ),
            skip_load_from_cli=True,
            skip_load_from_dotenv=True,
            offline=True,
        )
    )
    assert {server.name for server in parser.filter_for_server_instance_configs(config)} == {
        "policy_model",
        "indic_ifbench",
        "indic_ifbench_benchmark_simple_agent",
        "indic_ifbench_benchmark_environment_server",
    }
    server_dir = Path("resources_servers/indic_ifbench")
    adapter = load_task_data_schema(server_dir)
    report = validate_jsonl_rows(
        "indic_ifbench", adapter, "example", (server_dir / "data/example.jsonl").read_text().splitlines()
    )
    assert report.clean, report.summary()
    assert report.rows == 5


def test_metrics_weight_instructions_and_languages_separately():
    def row(language, strict, loose):
        return {
            "language": language,
            "evaluator_version": EVALUATOR_VERSION,
            "follow_instruction_list": strict,
            "follow_instruction_list_loose": loose,
        }

    rows = [row("hi", [True], [True]), row("hi", [False, False], [True, False]), row("ur", [True], [True])]
    masked = {**row("ur", [False], [False]), "mask_sample": True}
    result = summarize(rows + [masked])
    assert result["overall"]["prompt_strict"] == pytest.approx(2 / 3)
    assert result["overall"]["instruction_strict"] == 0.5
    assert result["overall"]["instruction_loose"] == 0.75
    assert result["per_language"]["hi"]["instruction_strict"] == pytest.approx(1 / 3)
    assert result["language_macro"]["instruction_strict"] == pytest.approx(2 / 3)
    assert result["masked_attempts"] == {"ur": 1}
    with pytest.raises(ValueError, match="No unmasked"):
        summarize([masked])
    with pytest.raises(ValueError, match="version"):
        summarize([{**rows[0], "evaluator_version": "another"}])
    with pytest.raises(ValueError, match="version"):
        summarize([rows[0], {**rows[1], "evaluator_version": "indic-ifbench-v1"}])
    with pytest.raises(ValueError, match="aligned"):
        summarize([row("hi", [], [])])
    with pytest.raises(ValueError, match="booleans"):
        summarize([row("hi", [True], [False])])
