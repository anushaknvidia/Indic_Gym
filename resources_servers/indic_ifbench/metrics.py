# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Report prompt/instruction strict/loose accuracy, overall and per language."""

from collections import defaultdict
from collections.abc import Iterable

from pydantic import JsonValue

from resources_servers.indic_ifbench.task_data import EVALUATOR_VERSION, normalize_language


METRICS = ("prompt_strict", "prompt_loose", "instruction_strict", "instruction_loose")


def summarize(rows: Iterable[dict[str, JsonValue]]) -> dict[str, JsonValue]:
    """Aggregate verified attempts; exclude masked verifier failures explicitly.

    Instruction accuracy is micro-averaged over individual instructions. Averaging
    fractional rewards would overweight prompts that contain fewer instructions.
    Each repeat is an attempt; this reports pass@1 accuracy, not pass@k.
    """
    groups = defaultdict(list)
    masked = defaultdict(int)
    for row in rows:
        language = normalize_language(row["language"])
        if row.get("evaluator_version") != EVALUATOR_VERSION:
            raise ValueError("Missing or incompatible evaluator_version; reverify before aggregating")
        if row.get("mask_sample"):
            masked[language] += 1
            continue
        strict = row.get("follow_instruction_list")
        loose = row.get("follow_instruction_list_loose")
        if not isinstance(strict, list) or not strict or not isinstance(loose, list) or len(strict) != len(loose):
            raise ValueError("Every unmasked row needs nonempty, aligned strict and loose decisions")
        if any(type(v) is not bool for v in strict + loose) or any(
            s and not loose_value for s, loose_value in zip(strict, loose)
        ):
            raise ValueError("Decisions must be booleans and strict successes must also pass loose")
        groups[language].append((strict, loose))
    if not groups:
        raise ValueError("No unmasked evaluated attempts")

    def metrics(attempts: list[tuple[list[bool], list[bool]]]) -> dict[str, JsonValue]:
        count = len(attempts)
        instructions = sum(len(strict) for strict, _ in attempts)
        return {
            "attempts": count,
            "instructions": instructions,
            "prompt_strict": sum(all(s) for s, _ in attempts) / count,
            "prompt_loose": sum(all(loose) for _, loose in attempts) / count,
            "instruction_strict": sum(sum(s) for s, _ in attempts) / instructions,
            "instruction_loose": sum(sum(loose) for _, loose in attempts) / instructions,
        }

    per_language = {language: metrics(attempts) for language, attempts in sorted(groups.items())}
    return {
        "evaluator_version": EVALUATOR_VERSION,
        "overall": metrics([attempt for attempts in groups.values() for attempt in attempts]),
        "per_language": per_language,
        "language_macro": {key: sum(m[key] for m in per_language.values()) / len(per_language) for key in METRICS},
        "masked_attempts": dict(sorted(masked.items())),
    }
