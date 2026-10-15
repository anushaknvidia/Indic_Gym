# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""NeMo Gym resources server for the translated Indic IFBench dataset."""

import asyncio
import logging
import re
from typing import ClassVar

from pydantic import Field, JsonValue

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseRunRequest,
    BaseVerifyRequest,
    BaseVerifyResponse,
    ReverifyMode,
    SimpleResourcesServer,
)
from resources_servers.indic_ifbench.evaluation import evaluate
from resources_servers.indic_ifbench.metrics import METRICS, summarize
from resources_servers.indic_ifbench.task_data import EVALUATOR_VERSION, TaskData


logger = logging.getLogger(__name__)


class IndicIFBenchResourcesServerConfig(BaseResourcesServerConfig):
    REVERIFY_MODE: ClassVar[ReverifyMode] = ReverifyMode.STATELESS
    num_processes: int = Field(default=32, ge=1)


class IndicIFBenchRunRequest(TaskData, BaseRunRequest):
    pass


class IndicIFBenchVerifyRequest(IndicIFBenchRunRequest, BaseVerifyRequest):
    pass


class IndicIFBenchVerifyResponse(IndicIFBenchVerifyRequest, BaseVerifyResponse):
    language: str
    evaluator_version: str = EVALUATOR_VERSION
    follow_all_instructions: bool
    follow_instruction_list: list[bool]
    follow_all_instructions_loose: bool
    follow_instruction_list_loose: list[bool]
    reward_loose: float


class IndicIFBenchResourcesServer(SimpleResourcesServer):
    config: IndicIFBenchResourcesServerConfig

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._semaphore = asyncio.Semaphore(self.config.num_processes)

    async def verify(self, body: IndicIFBenchVerifyRequest) -> IndicIFBenchVerifyResponse:
        """Score the final assistant text; preserve metadata and both accuracy modes."""
        text = ""
        for item in reversed(body.response.output):
            if getattr(item, "type", None) == "message" and getattr(item, "role", None) == "assistant":
                text = "".join(part.text for part in item.content if getattr(part, "type", None) == "output_text")
                break
        text = re.sub(r"<think(?:ing)?>.*?</think(?:ing)?>", "", text, flags=re.DOTALL)
        text = re.sub(r"<think(?:ing)?>.*$", "", text, flags=re.DOTALL)
        error = None
        try:
            async with self._semaphore:
                strict, loose = await asyncio.to_thread(
                    evaluate, body.instruction_id_list, body.kwargs, language=body.language, response=text
                )
        except Exception as exc:
            # A verifier defect is not evidence of a model failure. Keep it visible
            # and exclude this sample from aggregates instead of silently scoring 0.
            logger.exception("Indic IFBench evaluation failed for %s/%s", body.language, body.id)
            strict = [False] * len(body.instruction_id_list)
            loose = list(strict)
            error = f"{type(exc).__name__}: {exc}"
        reward = float(all(strict)) if body.grading_mode == "binary" else sum(strict) / len(strict)
        reward_loose = float(all(loose)) if body.grading_mode == "binary" else sum(loose) / len(loose)
        result = body.model_dump()
        result.update(
            reward=reward,
            reward_loose=reward_loose,
            follow_all_instructions=all(strict),
            follow_instruction_list=strict,
            follow_all_instructions_loose=all(loose),
            follow_instruction_list_loose=loose,
            evaluator_version=EVALUATOR_VERSION,
            mask_sample=error is not None,
        )
        if error:
            result.update(failure_kind="verifier_error", failure_reason=error)
        return IndicIFBenchVerifyResponse(**result)

    def compute_metrics(self, tasks: list[list[dict[str, JsonValue]]]) -> dict[str, float]:
        """Expose IFBench's four accuracies and per-language values in Gym metrics."""
        rows = [row for task in tasks for row in task if not row.get("mask_sample")]
        if not rows:
            return {}
        summary = summarize(rows)
        metrics = {key: summary["overall"][key] for key in METRICS}
        metrics.update({f"language_macro/{key}": value for key, value in summary["language_macro"].items()})
        for language, scores in summary["per_language"].items():
            metrics.update({f"language/{language}/{key}": value for key, value in scores.items()})
        return metrics

    def compute_repeat_metrics(self, tasks: list[list[dict[str, JsonValue]]]) -> dict[str, float]:
        return self.compute_metrics(tasks)

    def get_key_metrics(self, agent_metrics: dict[str, JsonValue]) -> dict[str, JsonValue]:
        return {key: agent_metrics[key] for key in METRICS if key in agent_metrics}


if __name__ == "__main__":
    IndicIFBenchResourcesServer.run_webserver()
