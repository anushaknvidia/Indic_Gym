# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Behavioral coverage for rubric judging, failure masking and text extraction."""

import asyncio
import json
from copy import deepcopy
from unittest.mock import AsyncMock, MagicMock

import pytest

from nemo_gym.config_types import AggregateMetricsRequest, ModelServerRef
from nemo_gym.judge import judge_failsafe
from nemo_gym.openai_utils import (
    NeMoGymChatCompletion,
    NeMoGymChatCompletionCreateParamsNonStreaming,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputRefusal,
    NeMoGymResponseOutputText,
)
from nemo_gym.server_utils import ServerClient
from resources_servers.biggenbench import app as biggenbench_app
from resources_servers.biggenbench.app import (
    BigGenBenchResourcesServer,
    BigGenBenchResourcesServerConfig,
    BigGenBenchVerifyRequest,
    build_judge_prompt,
    parse_score,
    response_text,
    strip_thinking,
)
from resources_servers.biggenbench.task_data import TaskData


def metadata() -> dict:
    return {
        "id": "planning_1",
        "language": "hi",
        "capability": "planning",
        "task": "plan",
        "system_prompt": "Be concise.",
        "input": "यात्रा की योजना बनाइए।",
        "reference_answer": "REFERENCE_ONLY: a feasible plan.",
        "score_rubric": {
            "criteria": "RUBRIC_ONLY: planning quality",
            **{f"score{i}_description": f"Description of rating {i}" for i in range(1, 6)},
        },
    }


def make_response(text: str) -> NeMoGymResponse:
    return NeMoGymResponse(
        id="resp",
        created_at=0.0,
        model="policy",
        object="response",
        output=[
            NeMoGymResponseOutputMessage(
                id="msg",
                content=[NeMoGymResponseOutputText(annotations=[], text=text, type="output_text")],
                role="assistant",
                status="completed",
                type="message",
            )
        ],
        parallel_tool_calls=False,
        tool_choice="none",
        tools=[],
    )


def request(text: str = "यह यात्रा की योजना है।") -> BigGenBenchVerifyRequest:
    meta = metadata()
    return BigGenBenchVerifyRequest(
        responses_create_params=NeMoGymResponseCreateParamsNonStreaming(
            input=[
                {"role": "system", "content": meta["system_prompt"]},
                {"role": "user", "content": meta["input"]},
            ]
        ),
        response=make_response(text),
        verifier_metadata=meta,
    )


def server() -> BigGenBenchResourcesServer:
    return BigGenBenchResourcesServer(
        config=BigGenBenchResourcesServerConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="app.py",
            name="biggenbench",
            judge_model_server=ModelServerRef(type="responses_api_models", name="judge_model"),
            judge_chat_completions_create_params=NeMoGymChatCompletionCreateParamsNonStreaming(
                messages=[], temperature=0.0, max_tokens=16384, reasoning_effort="low"
            ),
        ),
        server_client=MagicMock(spec=ServerClient),
    )


def judge_response(text: str = "Feedback [RESULT] 4", finish_reason: str = "stop") -> NeMoGymChatCompletion:
    return NeMoGymChatCompletion.model_validate(
        {
            "id": "judge-response",
            "model": "judge",
            "object": "chat.completion",
            "created": 0,
            "choices": [
                {
                    "index": 0,
                    "finish_reason": finish_reason,
                    "message": {"role": "assistant", "content": text},
                }
            ],
        }
    )


@pytest.mark.parametrize("score", range(1, 6))
def test_all_scores_preserve_raw_and_normalized_values(score: int) -> None:
    srv = server()
    srv._call_judge = AsyncMock(return_value=(f"Rubric feedback [RESULT] {score}", "stop"))
    result = asyncio.run(srv.verify(request()))
    assert result.score == score
    assert result.reward == (score - 1) / 4
    assert not result.mask_sample
    assert result.judge_error is None
    assert result.judge_finish_reason == "stop"


@pytest.mark.parametrize(
    "feedback,score",
    [
        ("Feedback [RESULT] 4", 4),
        ("Feedback\n[result]\n 5 \n", 5),
        ("The response is between 2 and 3. [RESULT] 3", 3),
        ("<think>maybe [RESULT] 1</think>Feedback [RESULT] 4", 4),
        ("<thinking>[RESULT] 2</thinking>[RESULT] 5", 5),
        ("thoughts [RESULT] 2</think>[RESULT] 5", 5),
        ("[RESULT] 0", None),
        ("[RESULT] 6", None),
        ("[RESULT] -1", None),
        ("[RESULT] 4.5", None),
        ("[RESULT] 4.0", None),
        ("[RESULT] 14", None),
        ("[RESULT] 4 or 5", None),
        ("[RESULT] 4\n[RESULT] 3", None),
        ("[RESULT] 4\n[RESULT] 4", None),
        ("[RESULT] 4 trailing commentary", None),
        ("4", None),
        ("Score: 4", None),
        ("", None),
        ("<think>unfinished [RESULT] 3", None),
    ],
)
def test_parse_score_rejects_ambiguous_or_unusable_verdicts(feedback: str, score: int | None) -> None:
    assert parse_score(feedback) == score


@pytest.mark.parametrize(
    "original,expected",
    [
        ("plain answer", "plain answer"),
        ("<think>private</think>answer", "answer"),
        ("<thinking>private</thinking>answer", "answer"),
        ("<THINK>private</THINK> answer", "answer"),
        ("private</thinking>answer", "answer"),
        ("<think>unfinished private reasoning", ""),
        ("answer<think>unfinished", "answer"),
        ("<think>a</think><thinking>b</thinking>answer", "answer"),
        ("", ""),
    ],
)
def test_strip_thinking(original: str, expected: str) -> None:
    assert strip_thinking(original) == expected


def test_reference_rubric_and_system_are_only_in_judge_prompt() -> None:
    srv = server()
    srv._call_judge = AsyncMock(return_value=("Feedback [RESULT] 4", "stop"))
    body = request("<thinking>private reasoning</thinking>अंतिम उत्तर")
    original = deepcopy(body.model_dump())
    asyncio.run(srv.verify(body))
    prompt = srv._call_judge.call_args.args[0]
    assert body.model_dump() == original
    assert "REFERENCE_ONLY" not in str(body.responses_create_params.input)
    assert "RUBRIC_ONLY" not in str(body.responses_create_params.input)
    assert "REFERENCE_ONLY" in prompt and "RUBRIC_ONLY" in prompt
    assert "Be concise." in prompt and body.verifier_metadata["input"] in prompt
    assert "private reasoning" not in prompt
    assert "अंतिम उत्तर" in prompt
    for score in range(1, 6):
        assert f"Score {score}: Description of rating {score}" in prompt


def test_llm_judge_tasks_keep_embedded_judging_task_delimited() -> None:
    task = TaskData.model_validate(metadata() | {"id": "refinement_llm_judge_1"})
    prompt = build_judge_prompt(task, "The other model's grading")
    assert "absolute grading of another model's grading" in prompt
    assert f"@@@\n###The instruction to evaluate:\n{task.system_prompt}\n\n{task.input}\n@@@" in prompt
    assert "###Response to evaluate:\nThe other model's grading" in prompt


@pytest.mark.parametrize("text", ["", "   ", "<think>reasoning only</think>"])
def test_empty_policy_response_scores_one_without_judge(text: str) -> None:
    srv = server()
    srv._call_judge = AsyncMock()
    result = asyncio.run(srv.verify(request(text)))
    assert result.score == 1 and result.reward == 0
    assert result.candidate_empty and not result.mask_sample
    assert result.judge_error is None
    srv._call_judge.assert_not_called()


@pytest.mark.parametrize("feedback", ["", "[RESULT] 6", "[RESULT] 3 [RESULT] 4", "Score 4"])
def test_invalid_judge_output_is_masked(feedback: str) -> None:
    srv = server()
    srv._call_judge = AsyncMock(return_value=(feedback, "stop"))
    result = asyncio.run(srv.verify(request()))
    assert result.score is None and result.reward == 0
    assert result.mask_sample and result.failure_kind == "judge_unparseable"
    assert result.judge_error and result.failure_reason == result.judge_error
    assert result.judge_feedback == feedback


@pytest.mark.parametrize("finish_reason", ["length", "content_filter", "tool_calls"])
def test_truncated_or_filtered_judge_output_never_scores(finish_reason: str) -> None:
    srv = server()
    srv._call_judge = AsyncMock(return_value=("Feedback [RESULT] 5", finish_reason))
    result = asyncio.run(srv.verify(request()))
    assert result.score is None
    assert result.mask_sample and result.failure_kind == "judge_unparseable"
    assert result.judge_finish_reason == finish_reason
    assert finish_reason in result.judge_error


def test_judge_transport_error_uses_retryable_gym_failure_route() -> None:
    srv = server()
    srv.server_client.post = AsyncMock(side_effect=TimeoutError("judge unavailable"))
    body = request()
    response = asyncio.run(judge_failsafe(srv.verify)(body))
    result = json.loads(response.body)
    assert result.get("score") is None and result["reward"] == 0
    assert result["failure_reason"] == "TimeoutError: judge unavailable"
    assert result["_ng_failure_class"] == result["failure_kind"] == "judge_failed"
    assert result["mask_sample"] and result["response"] == body.response.model_dump(mode="json")


def test_reverification_resets_stale_result_fields() -> None:
    srv = server()
    srv._call_judge = AsyncMock(return_value=("Feedback [RESULT] 3", "stop"))
    body = BigGenBenchVerifyRequest.model_validate(
        request().model_dump()
        | {
            "score": None,
            "reward": 0.0,
            "judge_error": "previous timeout",
            "mask_sample": True,
            "failure_kind": "judge_failed",
            "failure_reason": "previous timeout",
            "_ng_failure_class": "judge_failed",
            "_ng_failure_judge_error": "previous timeout",
        }
    )
    result = asyncio.run(srv.verify(body))
    assert result.score == 3 and result.reward == 0.5
    assert result.judge_error is None and not result.mask_sample
    assert result.failure_kind is None and result.failure_reason is None
    assert not any(key.startswith("_ng_failure_") for key in result.model_dump())


def test_invalid_metadata_is_a_verifier_failure() -> None:
    srv = server()
    srv._call_judge = AsyncMock()
    body = request()
    body.verifier_metadata["score_rubric"].pop("score5_description")
    result = asyncio.run(srv.verify(body))
    assert result.score is None and result.mask_sample
    assert result.failure_kind == "verifier_error"
    assert "score5_description" in result.failure_reason
    srv._call_judge.assert_not_called()


def test_final_answer_phase_excludes_commentary() -> None:
    response = make_response("preliminary")
    response.output[0].phase = "commentary"
    final = make_response("final").output[0]
    final.phase = "final_answer"
    response.output.append(final)
    assert response_text(response) == "final"
    response.output = response.output[:1]
    assert response_text(response) == ""


def test_refusal_is_a_nonempty_policy_response() -> None:
    response = make_response("")
    response.output[0].content = [NeMoGymResponseOutputRefusal(refusal="I cannot do that.")]
    assert response_text(response) == "I cannot do that."
    response.output = []
    assert response_text(response) == ""


def test_call_judge_uses_model_server_chat_endpoint_and_preserves_configuration(monkeypatch) -> None:
    srv = server()
    original = srv.config.judge_chat_completions_create_params.model_dump()
    mock = AsyncMock(return_value=judge_response("<think>reasoning</think>Feedback [RESULT] 4"))
    monkeypatch.setattr(biggenbench_app, "call_judge", mock)
    feedback, finish_reason = asyncio.run(srv._call_judge("judge input"))
    assert feedback == "<think>reasoning</think>Feedback [RESULT] 4"
    assert finish_reason == "stop"
    assert mock.call_args.args == (srv.server_client,)
    assert mock.call_args.kwargs["server_name"] == "judge_model"
    assert mock.call_args.kwargs["url_path"] == "/v1/chat/completions"
    params = mock.call_args.kwargs["json"]
    assert params.messages[0]["role"] == "system"
    assert "translated into an Indic language" in params.messages[0]["content"]
    assert params.messages[1] == {"role": "user", "content": "judge input"}
    assert params.max_tokens == 16384 and params.temperature == 0.0
    assert srv.config.judge_chat_completions_create_params.model_dump() == original


@pytest.mark.parametrize("choice_count", [0, 2])
def test_invalid_judge_choices_use_retryable_failure_route(monkeypatch, choice_count: int) -> None:
    srv = server()
    response = judge_response()
    response.choices *= choice_count
    monkeypatch.setattr(biggenbench_app, "call_judge", AsyncMock(return_value=response))
    response = asyncio.run(judge_failsafe(srv.verify)(request()))
    result = json.loads(response.body)
    assert result.get("score") is None and result["mask_sample"]
    assert result["_ng_failure_class"] == "judge_failed"
    assert f"Expected one judge choice, received {choice_count}" in result["failure_reason"]


def test_null_judge_content_is_unparseable(monkeypatch) -> None:
    srv = server()
    response = judge_response()
    response.choices[0].message.content = None
    monkeypatch.setattr(biggenbench_app, "call_judge", AsyncMock(return_value=response))
    result = asyncio.run(srv.verify(request()))
    assert result.score is None and result.mask_sample
    assert result.failure_kind == "judge_unparseable"


def test_server_registers_verify_endpoint() -> None:
    assert "/verify" in {route.path for route in server().setup_webserver().routes}


@pytest.mark.parametrize("all_masked", [False, True])
def test_aggregate_endpoint_keeps_judge_errors_and_coverage(all_masked: bool) -> None:
    rows = [
        {
            "_ng_task_index": 0,
            "reward": 0.0,
            "score": None,
            "mask_sample": True,
            "judge_error": "judge timed out",
            "verifier_metadata": metadata(),
        }
    ]
    if not all_masked:
        rows.append(
            {
                "_ng_task_index": 1,
                "reward": 0.75,
                "score": 4,
                "mask_sample": False,
                "judge_error": None,
                "verifier_metadata": metadata(),
            }
        )
    result = asyncio.run(server().aggregate_metrics(AggregateMetricsRequest(verify_responses=rows))).model_dump()
    for metrics in (result["agent_metrics"], result["key_metrics"]):
        assert metrics["judge_error_count"] == 1
        assert metrics["valid_count"] == (0 if all_masked else 1)
        assert metrics["coverage/masked_rollouts"] == 1
        assert metrics["coverage/measured_rollouts"] == (0 if all_masked else 1)
        if all_masked:
            assert "score" not in metrics
        else:
            assert metrics["score"] == 4
    assert result["agent_metrics"]["count"] == len(rows)
    assert result["agent_metrics"]["aggregate_count"] == (0 if all_masked else 1)
    assert result["agent_metrics"]["excluded_count"] == 0


@pytest.mark.parametrize("rows", [[], [{"score": None, "reward": 0.0}]])
def test_empty_or_unscored_aggregation_does_not_invent_ratings(rows: list[dict]) -> None:
    result = asyncio.run(server().aggregate_metrics(AggregateMetricsRequest(verify_responses=rows)))
    assert result.agent_metrics["count"] == result.agent_metrics["judge_error_count"] == len(rows)
    assert result.agent_metrics["valid_count"] == result.agent_metrics["aggregate_count"] == 0
    assert result.agent_metrics["excluded_count"] == 0
    assert "score" not in result.agent_metrics and "score" not in result.key_metrics


async def test_manifest_verifier_fixture() -> None:
    from nemo_gym.verifier_fixture import exercise_verifier_fixture

    results = await exercise_verifier_fixture(
        biggenbench_app.VERIFIER_FIXTURE, reward_range=(0.0, 1.0), determinism="unknown"
    )
    assert [result.kind for result in results] == ["full_reward", "zero_reward", "malformed"]
