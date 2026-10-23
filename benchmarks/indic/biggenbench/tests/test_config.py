# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest
import yaml
from omegaconf import OmegaConf

from nemo_gym.global_config import GlobalConfigDictParser, GlobalConfigDictParserConfig


@pytest.mark.parametrize("configured", [False, True])
def test_judge_endpoint_is_scoped_and_does_not_overwrite_other_models(
    monkeypatch: pytest.MonkeyPatch, configured: bool
) -> None:
    for key, value in {"BASE_URL": "https://judge.example/v1", "API_KEY": "test-key", "MODEL": "test-judge"}.items():
        if configured:
            monkeypatch.setenv(f"BIGGENBENCH_JUDGE_{key}", value)
        else:
            monkeypatch.delenv(f"BIGGENBENCH_JUDGE_{key}", raising=False)
    existing = {
        "responses_api_models": {
            "openai_model": {
                "entrypoint": "app.py",
                "openai_base_url": "https://other.example/v1",
                "openai_api_key": "other-key",
                "openai_model": "other-judge",
            }
        }
    }
    config = GlobalConfigDictParser().parse(
        GlobalConfigDictParserConfig(
            initial_global_config_dict=OmegaConf.merge(
                GlobalConfigDictParserConfig.NO_MODEL_GLOBAL_CONFIG_DICT,
                {"config_paths": ["benchmarks/indic/biggenbench/config.yaml"], "judge_model": existing},
            ),
            skip_load_from_cli=True,
            skip_load_from_dotenv=True,
            offline=True,
        )
    )
    other = config.judge_model.responses_api_models.openai_model
    for key, value in existing["responses_api_models"]["openai_model"].items():
        assert other[key] == value
    judge = config.biggenbench_judge_model.responses_api_models.openai_model
    assert judge.openai_model == ("test-judge" if configured else None)
    assert judge.openai_base_url == ("https://judge.example/v1" if configured else None)
    assert judge.openai_api_key == ("test-key" if configured else None)
    verifier = config.biggenbench.resources_servers.biggenbench
    assert verifier.judge_model_server.name == "biggenbench_judge_model"
    assert (
        config.biggenbench_environment_server.environment_servers.legacy_agent.agent_server.name
        == "biggenbench_simple_agent"
    )
    assert config.biggenbench_simple_agent.responses_api_agents.simple_agent.resources_server.name == "biggenbench"


def test_gemma_recipe_keeps_requested_generation_and_forwards_judge_credentials() -> None:
    config = yaml.safe_load((Path(__file__).parents[1] / "gemma_smoke.yaml").read_text())
    run = config["driver"]["benchmarks"]["indic_biggenbench"]["run"]
    assert run["responses_create_params"] == {"temperature": 1.0, "top_p": 0.95, "max_output_tokens": 32768}
    model = run["policy_model"]["responses_api_models"]["vllm_model"]
    assert model["chat_template_kwargs"] == {"enable_thinking": True}
    assert model["sampling_overrides"]["top_k"] == 64
    for suffix in ("BASE_URL", "API_KEY", "MODEL"):
        key = f"BIGGENBENCH_JUDGE_{suffix}"
        assert config["driver"]["env"][key] == f"runtime:{key}"
    assert "judge_base_url" not in run
