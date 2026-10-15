# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from nemo_gym.config_types import AggregateMetricsRequest
from nemo_gym.server_utils import ServerClient
from resources_servers.indic_ifbench.app import (
    IndicIFBenchResourcesServer,
    IndicIFBenchResourcesServerConfig,
    IndicIFBenchVerifyRequest,
)


@pytest.fixture
def server():
    return IndicIFBenchResourcesServer(
        config=IndicIFBenchResourcesServerConfig(
            host="127.0.0.1", port=8080, name="indic_ifbench", entrypoint="app.py"
        ),
        server_client=MagicMock(spec=ServerClient),
    )


def request(text="१ २", **overrides):
    data = {
        "id": 1,
        "language": "hi",
        "prompt": "दो संख्याएँ लिखें।",
        "instruction_id_list": ["count:numbers", "count:numbers"],
        "kwargs": [{"N": 2}, {"N": 3}],
        "source_key": "1",
        "uuid": "indic-ifbench/hi/1",
        "metadata": {"subset_for_metrics": "hi"},
        "responses_create_params": {"input": [{"role": "user", "content": "दो संख्याएँ लिखें।"}]},
        "response": {
            "id": "response",
            "created_at": 0,
            "model": "test",
            "object": "response",
            "parallel_tool_calls": False,
            "tool_choice": "auto",
            "tools": [],
            "output": [
                {
                    "id": "message",
                    "role": "assistant",
                    "type": "message",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": text, "annotations": []}],
                }
            ],
        },
    }
    return IndicIFBenchVerifyRequest.model_validate({**data, **overrides})


def test_http_verifier_and_metadata(server):
    with TestClient(server.setup_webserver()) as client:
        response = client.post("/verify", json=request().model_dump())
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["reward"] == 0.5
    assert result["follow_instruction_list"] == [True, False]
    assert result["follow_all_instructions"] is False
    assert result["language"] == "hi"
    assert result["uuid"] == "indic-ifbench/hi/1"
    assert result["kwargs"] == [{"N": 2}, {"N": 3}]
    assert result["metadata"] == {"subset_for_metrics": "hi"}
    assert not result["mask_sample"]


@pytest.mark.parametrize("mode,expected", [("fraction", 0.5), ("binary", 0.0)])
def test_fraction_binary_and_loose(server, mode, expected):
    result = asyncio.run(server.verify(request("900 800 700\n१ २", grading_mode=mode)))
    assert result.reward == 0
    assert result.follow_instruction_list_loose == [True, True]
    assert result.reward_loose == 1.0
    result = asyncio.run(server.verify(request(grading_mode=mode)))
    assert result.reward == expected


def test_final_message_parts_and_reasoning(server):
    body = request()
    body.response.output[0].content[0].text = "<think>99 98</think>१ "
    extra = body.response.output[0].content[0].model_copy(update={"text": "२"})
    body.response.output[0].content.append(extra)
    assert asyncio.run(server.verify(body)).follow_instruction_list == [True, False]
    body.response.output = []
    assert asyncio.run(server.verify(body)).follow_instruction_list == [False, False]


def test_verifier_defect_is_masked(server, monkeypatch):
    def broken(*args, **kwargs):
        raise RuntimeError("broken checker")

    monkeypatch.setattr("resources_servers.indic_ifbench.app.evaluate", broken)
    result = asyncio.run(server.verify(request()))
    assert result.mask_sample
    assert result.failure_kind == "verifier_error"
    assert "broken checker" in result.failure_reason


def test_native_aggregate_metrics(server):
    first = asyncio.run(server.verify(request())).model_dump()
    second = asyncio.run(server.verify(request("۱ ۲ ۳", language="ur"))).model_dump()
    first["_ng_task_index"] = 0
    second["_ng_task_index"] = 1
    result = asyncio.run(server.aggregate_metrics(AggregateMetricsRequest(verify_responses=[first, second])))
    assert result.agent_metrics["prompt_strict"] == 0
    assert result.agent_metrics["instruction_strict"] == 0.5
    assert result.agent_metrics["language/hi/instruction_strict"] == 0.5
    assert result.agent_metrics["language/ur/instruction_strict"] == 0.5
    assert result.key_metrics["instruction_strict"] == 0.5
    assert server.compute_metrics([[]]) == {}


@pytest.mark.parametrize(
    "overrides",
    [
        {"prompt": " "},
        {"instruction_id_list": []},
        {"kwargs": [{}]},
        {"language": "xx"},
        {"grading_mode": "unknown"},
    ],
)
def test_malformed_tasks_rejected(overrides):
    with pytest.raises(ValueError):
        request(**overrides)


def test_strict_scoring_preserves_response_whitespace(server):
    body = request(" उत्तर ", instruction_id_list=["format:no_whitespace"], kwargs=[{}])
    assert asyncio.run(server.verify(body)).follow_instruction_list == [False]
    body = request("<think>१ २", instruction_id_list=["count:numbers"], kwargs=[{"N": 2}])
    assert asyncio.run(server.verify(body)).follow_instruction_list == [False]
