# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""HumanEval original-test verifier matching the post_train_think reference."""

import asyncio
import json
import keyword
import logging
import os
import re
import signal
import sys
from pathlib import Path

from pydantic import BaseModel, Field, field_validator

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseVerifyRequest,
    BaseVerifyResponse,
    ReverifyMode,
    SimpleResourcesServer,
)
from nemo_gym.reward_profile import compute_pass_majority_metrics, compute_subset_metrics, highest_k_metrics
from nemo_gym.verifier_fixture import VerifierFixture
from resources_servers.humaneval_reference.verifier_fixture import create_humaneval_reference_server


logger = logging.getLogger(__name__)


class HumanEvalResourcesServerConfig(BaseResourcesServerConfig):
    REVERIFY_MODE = ReverifyMode.STATELESS
    num_processes: int = Field(default=8, ge=1)
    timeout: float = Field(default=3.0, gt=0)
    memory_limit_mb: int = Field(default=1024, ge=128)


class HumanEvalMetadata(BaseModel):
    task_id: str = Field(pattern=r"^HumanEval/(0|[1-9]\d*)$")
    language: str = Field(pattern=r"^[a-z]+$")
    test: str = Field(min_length=1)
    entry_point: str = Field(pattern=r"^[A-Za-z_]\w*$")

    @field_validator("entry_point")
    @classmethod
    def reject_python_keyword(cls, value: str) -> str:
        if keyword.iskeyword(value):
            raise ValueError("entry_point must not be a Python keyword")
        return value


class HumanEvalVerifyRequest(BaseVerifyRequest):
    verifier_metadata: HumanEvalMetadata


class HumanEvalVerifyResponse(BaseVerifyResponse):
    verifier_metadata: HumanEvalMetadata
    language: str
    extracted_model_code: str
    result: str


def extract_code(response: str) -> str:
    """Copy build_predictions_instruct: after last </think>, first fence, raw fallback."""
    response = response.split("</think>")[-1].strip()
    matches = re.findall(r"```(?:\w+)?\n(.*?)```", response, re.DOTALL)
    return matches[0] if matches else response


class HumanEvalResourcesServer(SimpleResourcesServer):
    ray_enabled = False
    config: HumanEvalResourcesServerConfig

    def model_post_init(self, context: object) -> None:
        self._semaphore = asyncio.Semaphore(self.config.num_processes)

    async def _execute(self, code: str, metadata: HumanEvalMetadata) -> dict:
        payload = {
            **metadata.model_dump(),
            "code": code,
            "timeout": self.config.timeout,
            "memory_limit_mb": self.config.memory_limit_mb,
        }
        async with self._semaphore:
            process = await asyncio.create_subprocess_exec(
                sys.executable,
                str(Path(__file__).with_name("runner.py")),
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                start_new_session=True,
            )
            try:
                # The checker applies the candidate time limit. This outer deadline
                # catches worker startup/IPC failures, which must not score as model failures.
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(json.dumps(payload).encode()), timeout=self.config.timeout + 10
                )
                if process.returncode:
                    raise RuntimeError(
                        f"HumanEval worker exited {process.returncode}: {stderr.decode(errors='replace')}"
                    )
                result = json.loads(stdout.decode(errors="replace"))
                if (
                    not isinstance(result, dict)
                    or type(result.get("passed")) is not bool
                    or not isinstance(result.get("result"), str)
                ):
                    raise ValueError("HumanEval worker returned an invalid verdict")
                return result
            finally:
                # Kill the entire process group on cancellation or a hung worker,
                # including any checker child left after an abnormal exit.
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                await process.wait()

    async def verify(self, body: HumanEvalVerifyRequest) -> HumanEvalVerifyResponse:
        code = extract_code(body.response.output_text or "")
        response_fields = {
            **body.model_dump(),
            "language": body.verifier_metadata.language,
            "extracted_model_code": code,
        }
        if not code.strip():
            return HumanEvalVerifyResponse(**response_fields, reward=0.0, result="empty response")
        try:
            result = await self._execute(code, body.verifier_metadata)
        except (OSError, RuntimeError, ValueError, TimeoutError) as exc:
            logger.exception("HumanEval worker failed for %s", body.verifier_metadata.task_id)
            return HumanEvalVerifyResponse(
                **response_fields,
                reward=0.0,
                result="worker error",
                mask_sample=True,
                failure_kind="humaneval_reference:worker_error",
                failure_reason=str(exc),
            )
        return HumanEvalVerifyResponse(**response_fields, reward=float(result["passed"]), result=result["result"])

    def compute_metrics(self, tasks: list[list[dict]]) -> dict:
        """Report original-test pass@k overall and per language (percentages)."""
        metrics, _, _, _ = compute_pass_majority_metrics(tasks)
        metrics.update(compute_subset_metrics(tasks, "language"))
        return metrics

    def get_key_metrics(self, agent_metrics: dict) -> dict:
        """Expose pass@1, including language breakdowns, as headline metrics."""
        result = highest_k_metrics(agent_metrics, "pass@1[avg-of-{k}]", score_names=["accuracy"])
        for language in {key.split("/")[0] for key in agent_metrics if "/pass@" in key}:
            result.update(
                highest_k_metrics(agent_metrics, f"{language}/pass@1[avg-of-{{k}}]", score_names=["accuracy"])
            )
        return result


VERIFIER_FIXTURE = VerifierFixture(
    server_factory=create_humaneval_reference_server,
    request_model=HumanEvalVerifyRequest,
    cases_path=Path(__file__).parent / "tests" / "verifier_cases.jsonl",
)


if __name__ == "__main__":
    HumanEvalResourcesServer.run_webserver()
