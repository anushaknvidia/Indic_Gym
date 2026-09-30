# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BiGGen Bench pointwise rubric evaluation using a Gym-hosted LLM judge.

Scores retain the benchmark's 1–5 scale; Gym rewards normalize it to [0, 1].
Judge failures are masked, never silently reported as a legitimate score of 1.
The reference and rubric are verifier metadata, never policy input.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import ClassVar

from pydantic import ConfigDict, Field, JsonValue, ValidationError

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseRunRequest,
    BaseVerifyRequest,
    BaseVerifyResponse,
    ReverifyMode,
    SimpleResourcesServer,
)
from nemo_gym.config_types import AggregateMetrics, AggregateMetricsRequest, ModelServerRef
from nemo_gym.failure_kinds import JUDGE_UNPARSEABLE, VERIFIER_ERROR
from nemo_gym.judge import JudgeError, call_judge
from nemo_gym.openai_utils import (
    NeMoGymChatCompletion,
    NeMoGymChatCompletionCreateParamsNonStreaming,
    NeMoGymResponse,
)
from nemo_gym.prompt import PromptConfig, fill_prompt, load_prompt_config
from nemo_gym.server_utils import ServerClient
from nemo_gym.verifier_fixture import VerifierFixture
from resources_servers.biggenbench.metrics import compute_metrics
from resources_servers.biggenbench.task_data import TaskData


_THINK_BLOCK = re.compile(r"<(think|thinking)>.*?</\1>", re.IGNORECASE | re.DOTALL)
_THINK_END = re.compile(r"</(?:think|thinking)>", re.IGNORECASE)
_THINK_START = re.compile(r"<(?:think|thinking)>", re.IGNORECASE)
_RESULT_MARKER = re.compile(r"\[RESULT\]", re.IGNORECASE)


def strip_thinking(text: str) -> str:
    """Remove reasoning blocks, including a missing opener or unfinished tail."""
    cleaned = _THINK_BLOCK.sub("", text)
    # Some model templates omit the opening tag in the returned generation.
    cleaned = _THINK_END.split(cleaned)[-1]
    cleaned = _THINK_START.split(cleaned, maxsplit=1)[0]
    return cleaned.strip()


def response_text(response: NeMoGymResponse) -> str:
    """Extract final assistant text, excluding separate reasoning/commentary."""
    messages = [item for item in response.output if item.type == "message"]
    final_messages = [item for item in messages if item.phase == "final_answer"]
    messages = final_messages or [item for item in messages if item.phase != "commentary"]
    parts = []
    for item in messages:
        for content in item.content:
            if content.type == "output_text":
                parts.append(content.text)
            elif content.type == "refusal":
                parts.append(content.refusal)
    return strip_thinking("\n".join(parts))


def parse_score(feedback: str) -> int | None:
    """Accept exactly one final ``[RESULT] n`` marker with an integer 1–5."""
    feedback = strip_thinking(feedback)
    markers = list(_RESULT_MARKER.finditer(feedback))
    if len(markers) != 1:
        return None
    score = re.fullmatch(r"\s*([1-5])\s*", feedback[markers[0].end() :])
    return int(score.group(1)) if score else None


def build_judge_messages(task: TaskData, candidate: str, prompt: PromptConfig) -> list[dict[str, str]]:
    """Render the Prometheus absolute-grading prompt with the full task rubric."""
    rubric = task.score_rubric
    rubric_text = "\n".join(
        [f"[{rubric.criteria}]"]
        + [f"Score {score}: {getattr(rubric, f'score{score}_description')}" for score in range(1, 6)]
    )
    instruction = f"{task.system_prompt}\n\n{task.input}".strip()
    grading_instructions = ""
    if "llm_judge" in task.id:
        grading_instructions = (
            "5. You are conducting absolute grading of another model's grading. Do not confuse "
            "the embedded grading task with your own task. The other model's instruction is "
            'separated from its response with "@@@" delimiters.\n'
        )
        instruction_section = f"@@@\n###The instruction to evaluate:\n{instruction}\n@@@"
    else:
        instruction_section = f"###The instruction to evaluate:\n{instruction}"
    messages = fill_prompt(
        prompt,
        {
            "grading_instructions": grading_instructions,
            "instruction_section": instruction_section,
            "candidate": candidate,
            "reference_answer": task.reference_answer,
            "rubric": rubric_text,
        },
    )
    # Preserve the original generation prefix after the feedback heading.
    messages[-1]["content"] += " "
    return messages


class BigGenBenchResourcesServerConfig(BaseResourcesServerConfig):
    """Judge endpoint and generation settings for absolute evaluation."""

    REVERIFY_MODE: ClassVar[ReverifyMode] = ReverifyMode.STATELESS
    name: str = "biggenbench"
    judge_model_server: ModelServerRef
    judge_chat_completions_create_params: NeMoGymChatCompletionCreateParamsNonStreaming
    judge_prompt_path: str = "resources_servers/biggenbench/prompts/absolute.yaml"


class BigGenBenchRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")

    verifier_metadata: dict[str, JsonValue] = Field(default_factory=dict)


class BigGenBenchVerifyRequest(BigGenBenchRunRequest, BaseVerifyRequest):
    pass


class BigGenBenchVerifyResponse(BaseVerifyResponse):
    """Raw rubric score, normalized reward, and auditable judge diagnostics."""

    model_config = ConfigDict(extra="allow")

    score: int | None = Field(default=None, ge=1, le=5)
    judge_error: str | None = None
    judge_feedback: str = ""
    judge_finish_reason: str | None = None
    candidate_empty: bool = False


class BigGenBenchResourcesServer(SimpleResourcesServer):
    """Pointwise BiGGen Bench verification against each task's own rubric."""

    ray_enabled = False
    config: BigGenBenchResourcesServerConfig

    async def _call_judge(self, messages: list[dict[str, str]]) -> tuple[str, str]:
        params = self.config.judge_chat_completions_create_params.model_copy(deep=True)
        params.messages = messages
        response = await call_judge(
            self.server_client,
            server_name=self.config.judge_model_server.name,
            url_path="/v1/chat/completions",
            json=params,
            response_model=NeMoGymChatCompletion,
        )
        if len(response.choices) != 1:
            raise JudgeError(f"Expected one judge choice, received {len(response.choices)}")
        choice = response.choices[0]
        return choice.message.content or "", choice.finish_reason

    async def verify(self, body: BigGenBenchVerifyRequest) -> BigGenBenchVerifyResponse:
        """Return a valid 1–5 rating or a masked, explicitly diagnosed failure."""
        # A reverify request may carry previous result fields as allowed extras.
        # Always replace those diagnostics before evaluating the response again.
        result = BigGenBenchVerifyResponse.model_validate(
            body.model_dump(exclude={"_ng_failure_class", "_ng_failure_judge_error"})
            | {
                "reward": 0.0,
                "score": None,
                "judge_error": None,
                "judge_feedback": "",
                "judge_finish_reason": None,
                "candidate_empty": False,
                "mask_sample": False,
                "failure_kind": None,
                "failure_reason": None,
            }
        )
        try:
            task = TaskData.model_validate(body.verifier_metadata)
        except ValidationError as exc:
            result.mask_sample = True
            result.failure_kind = VERIFIER_ERROR
            result.failure_reason = f"Invalid verifier_metadata: {exc}"
            return result

        candidate = response_text(body.response)
        if not candidate:
            result.candidate_empty = True
            result.score = 1
            return result

        # Transport/schema failures reach Gym's judge_failsafe and retryable sidecar.
        prompt = load_prompt_config(self.config.judge_prompt_path)
        feedback, finish_reason = await self._call_judge(build_judge_messages(task, candidate, prompt))
        result.judge_feedback = feedback
        result.judge_finish_reason = finish_reason
        if finish_reason != "stop":
            result.judge_error = f"Judge did not finish normally: finish_reason={finish_reason}"
        else:
            result.score = parse_score(feedback)
            if result.score is None:
                result.judge_error = "Judge output must end with exactly one [RESULT] integer from 1 to 5"

        if result.judge_error:
            result.mask_sample = True
            result.failure_kind = JUDGE_UNPARSEABLE
            result.failure_reason = result.judge_error
        else:
            result.reward = (result.score - 1) / 4.0
        return result

    def compute_metrics(self, tasks: list[list[dict[str, JsonValue]]]) -> dict[str, JsonValue]:
        """Aggregate only valid rubric ratings, preserving all-task diagnostics."""
        return compute_metrics(tasks)

    async def aggregate_metrics(self, body: AggregateMetricsRequest) -> AggregateMetrics:
        """Retain judge-failure counts alongside Gym's masked-sample coverage."""
        result = await super().aggregate_metrics(body)
        # The base aggregator passes only unmasked rows to compute_metrics().
        # Restore counts without recomputing the grouped scores.
        for key in ("valid_count", "aggregate_count", "excluded_count"):
            result.agent_metrics.setdefault(key, 0)
        result.agent_metrics["count"] = len(body.verify_responses)
        result.agent_metrics["judge_error_count"] = len(body.verify_responses) - result.agent_metrics["valid_count"]
        result.key_metrics.update(self.get_key_metrics(result.agent_metrics))
        return result

    def get_key_metrics(self, agent_metrics: dict[str, JsonValue]) -> dict[str, JsonValue]:
        return {
            key: agent_metrics[key]
            for key in ("score", "score_all", "valid_count", "judge_error_count")
            if key in agent_metrics
        }


def _fixture_server() -> BigGenBenchResourcesServer:
    from unittest.mock import MagicMock

    return BigGenBenchResourcesServer(
        config=BigGenBenchResourcesServerConfig(
            host="127.0.0.1",
            port=0,
            entrypoint="app.py",
            judge_model_server=ModelServerRef(type="responses_api_models", name="judge_model"),
            judge_chat_completions_create_params=NeMoGymChatCompletionCreateParamsNonStreaming(
                messages=[], temperature=0.0, max_tokens=16384, reasoning_effort="low"
            ),
        ),
        server_client=MagicMock(spec=ServerClient),
    )


async def _fixture_invoke(
    server: BigGenBenchResourcesServer, body: BigGenBenchVerifyRequest
) -> BigGenBenchVerifyResponse:
    from unittest.mock import AsyncMock

    # Reuse the committed example's rubric; only judge transport is replaced.
    with (Path(__file__).parent / "data" / "example.jsonl").open(encoding="utf-8") as stream:
        body.verifier_metadata = json.loads(next(stream))["verifier_metadata"]
    server._call_judge = AsyncMock(return_value=(body.fixture_judge_text, "stop"))
    result = await server.verify(body)
    server._call_judge.assert_awaited_once()
    return result


VERIFIER_FIXTURE = VerifierFixture(
    server_factory=_fixture_server,
    request_model=BigGenBenchVerifyRequest,
    cases_path=Path(__file__).parent / "tests" / "verifier_cases.jsonl",
    invoke=_fixture_invoke,
)


if __name__ == "__main__":
    BigGenBenchResourcesServer.run_webserver()
