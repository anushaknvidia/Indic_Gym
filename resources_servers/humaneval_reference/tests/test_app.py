# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import io
import json
import sys
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient
from human_eval.data import read_problems

from nemo_gym.openai_utils import NeMoGymResponse
from nemo_gym.server_utils import ServerClient
from nemo_gym.verifier_fixture import exercise_verifier_fixture
from resources_servers.humaneval_reference import runner
from resources_servers.humaneval_reference.app import (
    VERIFIER_FIXTURE,
    HumanEvalMetadata,
    HumanEvalResourcesServer,
    HumanEvalResourcesServerConfig,
    HumanEvalVerifyRequest,
    extract_code,
)


def make_request(text: str) -> HumanEvalVerifyRequest:
    return HumanEvalVerifyRequest(
        responses_create_params={"input": [{"role": "user", "content": "Implement add(a, b)."}]},
        verifier_metadata={
            "task_id": "HumanEval/0",
            "language": "hindi",
            "test": "def check(candidate):\n    assert candidate(2, 3) == 5\n    assert candidate(-1, 1) == 0",
            "entry_point": "add",
        },
        response=NeMoGymResponse(
            id="response",
            created_at=0,
            model="test",
            object="response",
            output=[
                {
                    "id": "message",
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": text, "annotations": []}],
                }
            ],
            parallel_tool_calls=False,
            tool_choice="auto",
            tools=[],
        ),
    )


@pytest.fixture
def server() -> HumanEvalResourcesServer:
    return HumanEvalResourcesServer(
        config=HumanEvalResourcesServerConfig(
            name="humaneval_reference",
            host="127.0.0.1",
            port=8000,
            entrypoint="app.py",
            num_processes=2,
            timeout=0.2,
        ),
        server_client=MagicMock(spec=ServerClient),
    )


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("<think>```python\nwrong\n```</think>```python\nright\n```", "right\n"),
        ("old</think>scratch</think>  raw code  ", "raw code"),
        ("```python\nfirst\n```\n```python\nsecond\n```", "first\n"),
        ("prose\n```\ngeneric\n```\nmore prose", "generic\n"),
        ("```javascript\ncode\n```", "code\n"),
        ("```python\nunclosed", "```python\nunclosed"),
        ("  raw code  ", "raw code"),
        ("<think>reasoning</think>", ""),
    ],
)
def test_reference_extraction(text: str, expected: str) -> None:
    assert extract_code(text) == expected


@pytest.mark.parametrize(
    ("completion", "reward", "result"),
    [
        ("```python\ndef add(a, b): return a + b\n```", 1.0, "passed"),
        ("def add(a, b): return a + b", 1.0, "passed"),
        ("<think>scratch</think>```python\ndef add(a, b): return a + b\n```", 1.0, "passed"),
        ("def add(a, b): return a * b", 0.0, "failed:"),
        ("def add(a, b) broken", 0.0, "failed:"),
        ("def add(a, b):\n    while True: pass", 0.0, "timed out"),
        ("def add(a, b): raise RuntimeError('bad')", 0.0, "failed: bad"),
        ("    return a + b", 0.0, "failed:"),
        ("```python\ndef add(a, b): return a * b\n```\n```python\ndef add(a, b): return a + b\n```", 0.0, "failed:"),
        ("", 0.0, "empty response"),
        ("```python\n\n```", 0.0, "empty response"),
    ],
)
async def test_verifier_executes_original_tests(server, completion, reward, result) -> None:
    response = await server.verify(make_request(completion))
    assert response.reward == reward
    assert response.result.startswith(result)
    assert response.language == "hindi"
    assert not response.mask_sample


async def test_empty_output(server) -> None:
    body = make_request("")
    body.response.output = []
    assert (await server.verify(body)).reward == 0


async def test_worker_error_is_masked(server, monkeypatch) -> None:
    monkeypatch.setattr(HumanEvalResourcesServer, "_execute", AsyncMock(side_effect=OSError("no worker")))
    response = await server.verify(make_request("def add(a, b): return a + b"))
    assert response.mask_sample
    assert response.reward == 0
    assert response.failure_reason == "no worker"


@pytest.mark.parametrize(
    ("returncode", "stdout", "exception"),
    [
        (1, b"", RuntimeError),
        (0, b"not json", ValueError),
        (0, b'{"passed": "yes"}', ValueError),
        (0, b'{"passed": true}', ValueError),
        (0, b"[]", ValueError),
    ],
)
async def test_invalid_worker_result(server, monkeypatch, returncode, stdout, exception) -> None:
    process = MagicMock(pid=123, returncode=returncode)
    process.communicate = AsyncMock(return_value=(stdout, b"error\xff"))
    process.wait = AsyncMock()
    monkeypatch.setattr(asyncio, "create_subprocess_exec", AsyncMock(return_value=process))
    kill = MagicMock()
    monkeypatch.setattr("resources_servers.humaneval_reference.app.os.killpg", kill)
    with pytest.raises(exception):
        await server._execute("code", make_request("").verifier_metadata)
    kill.assert_called_once()
    process.wait.assert_awaited_once()


async def test_cancelled_worker_is_reaped(server, monkeypatch) -> None:
    process = MagicMock(pid=123)
    process.communicate = AsyncMock(side_effect=asyncio.CancelledError())
    process.wait = AsyncMock()
    monkeypatch.setattr(asyncio, "create_subprocess_exec", AsyncMock(return_value=process))
    kill = MagicMock()
    monkeypatch.setattr("resources_servers.humaneval_reference.app.os.killpg", kill)
    with pytest.raises(asyncio.CancelledError):
        await server._execute("code", make_request("").verifier_metadata)
    kill.assert_called_once()
    process.wait.assert_awaited_once()


async def test_execution_concurrency_bound(server, monkeypatch) -> None:
    active = peak = 0

    async def communicate(_):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0.01)
        active -= 1
        return b'{"passed": true, "result": "passed"}', b""

    process = MagicMock(pid=123, returncode=0, communicate=communicate, wait=AsyncMock())
    monkeypatch.setattr(asyncio, "create_subprocess_exec", AsyncMock(return_value=process))
    monkeypatch.setattr("resources_servers.humaneval_reference.app.os.killpg", MagicMock())
    await asyncio.gather(*(server._execute("code", make_request("").verifier_metadata) for _ in range(6)))
    assert peak == 2


def test_runner_uses_full_solution_without_prompt(monkeypatch) -> None:
    payload = {
        **make_request("").verifier_metadata.model_dump(),
        "code": "def add(a, b): return a + b",
        "timeout": 1,
        "memory_limit_mb": 1024,
    }
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(payload)))
    limits = MagicMock()
    checker = MagicMock(return_value={"task_id": "HumanEval/0", "passed": True, "result": "passed"})
    monkeypatch.setattr(runner, "apply_resource_limits", limits)
    monkeypatch.setattr(runner, "check_correctness", checker)
    written = {}
    monkeypatch.setattr(runner.os, "dup", lambda fd: 99)
    monkeypatch.setattr(runner.os, "open", lambda *args: 100)
    monkeypatch.setattr(runner.os, "dup2", MagicMock())
    monkeypatch.setattr(runner.os, "close", MagicMock())
    monkeypatch.setattr(runner.os, "write", lambda fd, data: written.update(fd=fd, data=data) or len(data))
    runner.main()
    assert json.loads(written["data"])["passed"] is True
    assert written["fd"] == 99
    limits.assert_called_once_with(timeout=1, memory_limit_mb=1024)
    assert checker.call_args.kwargs["problem"] == {
        "task_id": "HumanEval/0",
        "prompt": "",
        "test": payload["test"],
        "entry_point": "add",
    }
    assert checker.call_args.kwargs["completion"] == payload["code"]


def test_runner_applies_resource_limits(monkeypatch) -> None:
    setrlimit = MagicMock()
    monkeypatch.setattr(runner.resource, "setrlimit", setrlimit)
    runner.apply_resource_limits(timeout=3.1, memory_limit_mb=512)
    assert setrlimit.call_count == 4
    assert setrlimit.call_args_list[0].args == (runner.resource.RLIMIT_AS, (512 * 1024 * 1024,) * 2)


async def test_verifier_fixture() -> None:
    results = await exercise_verifier_fixture(
        VERIFIER_FIXTURE,
        reward_range=(0, 1),
        higher_is_better=True,
        determinism="unknown",
    )
    assert {result.kind for result in results} == {"full_reward", "zero_reward", "malformed"}


async def test_raw_stdout_from_candidate_does_not_corrupt_verdict(server) -> None:
    response = await server.verify(
        make_request("import os\ndef add(a, b):\n    os.write(1, b'RAW-FD1-NOISE')\n    return a + b")
    )
    assert response.reward == 1
    assert response.result == "passed"
    assert not response.mask_sample


async def test_all_canonical_solutions_pass_reference_checker(server) -> None:
    problems = read_problems()
    assert set(problems) == {f"HumanEval/{index}" for index in range(164)}
    server.config.timeout = 3.0
    failures = []
    for task_id, problem in problems.items():
        result = await server._execute(
            problem["prompt"] + problem["canonical_solution"],
            HumanEvalMetadata(
                task_id=task_id,
                language="english",
                test=problem["test"],
                entry_point=problem["entry_point"],
            ),
        )
        if not result["passed"]:
            failures.append((task_id, result["result"]))
    assert not failures


def test_http_verify(server) -> None:
    with TestClient(server.setup_webserver()) as client:
        response = client.post("/verify", json=make_request("def add(a, b): return a + b").model_dump())
    assert response.status_code == 200
    assert response.json()["reward"] == 1


def test_metrics_preserve_language_groups(server) -> None:
    tasks = [
        [{"reward": 1.0, "language": "hindi"}, {"reward": 0.0, "language": "hindi"}],
        [{"reward": 1.0, "language": "tamil"}, {"reward": 1.0, "language": "tamil"}],
    ]
    metrics = server.compute_metrics(tasks)
    assert metrics["pass@1[avg-of-2]/accuracy"] == 75
    assert metrics["hindi/pass@1[avg-of-2]/accuracy"] == 50
    assert metrics["tamil/pass@1[avg-of-2]/accuracy"] == 100
    assert metrics["pass@2/accuracy"] == 100
    headlines = server.get_key_metrics(metrics)
    assert headlines["hindi/pass@1[avg-of-2]/accuracy"] == 50
    assert headlines["pass@1[avg-of-2]/accuracy"] == 75
