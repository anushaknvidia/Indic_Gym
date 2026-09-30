# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Aggregate absolute rubric scores, keeping invalid judgments out of score denominators."""

from collections import defaultdict
from collections.abc import Mapping
from typing import Any


def compute_metrics(tasks: list[list[dict[str, Any]]]) -> dict[str, float | int]:
    """Return capability-macro scores and language/capability/task breakdowns.

    `score` excludes moral_belief and upstream multilingual tasks, matching the
    original paper's exclusions. Within each language, average each capability's
    valid ratings, then average capability means as in upstream make_report.py.
    The overall score averages language scores equally. `score_all` applies the
    same aggregation without exclusions; `score_micro` pools included ratings.
    Missing or failed judgments never become a score of 1; counts remain visible.
    """
    rows = [row for task in tasks for row in task]
    metrics: dict[str, float | int] = {"count": len(rows), "valid_count": 0, "judge_error_count": 0}
    groups: dict[str, list[int]] = defaultdict(list)
    language_capabilities: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    all_language_capabilities: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    included: list[int] = []
    valid: list[int] = []
    excluded_count = 0
    for row in rows:
        score = row.get("score")
        if row.get("mask_sample") or row.get("judge_error") or type(score) is not int or not 1 <= score <= 5:
            metrics["judge_error_count"] += 1
            continue
        meta = row.get("verifier_metadata") or {}
        if not isinstance(meta, Mapping):
            raise ValueError("verifier_metadata must be a mapping")
        valid.append(score)
        language = str(meta.get("language", "unknown"))
        capability = str(meta.get("capability", "unknown"))
        all_language_capabilities[language][capability].append(score)
        include = (
            meta.get("include_in_aggregate", True)
            and meta.get("task") != "moral_belief"
            and meta.get("capability") != "multilingual"
        )
        if include:
            included.append(score)
            language_capabilities[language][capability].append(score)
        else:
            excluded_count += 1
        for field in ("language", "capability", "task"):
            value = str(meta.get(field, "unknown"))
            prefix = f"by_{field}/{value}"
            groups[f"{prefix}/score_all"].append(score)
            if include:
                groups[f"{prefix}/score"].append(score)
    metrics.update(valid_count=len(valid), aggregate_count=len(included), excluded_count=excluded_count)
    if valid:
        metrics["score_all_micro"] = sum(valid) / len(valid)
    if included:
        metrics["score_micro"] = sum(included) / len(included)
    for key, scores in sorted(groups.items()):
        metrics[key] = sum(scores) / len(scores)
        metrics[key + "_count"] = len(scores)
    for metric, languages in (("score", language_capabilities), ("score_all", all_language_capabilities)):
        language_scores = []
        for language, capabilities in sorted(languages.items()):
            capability_means = [sum(scores) / len(scores) for scores in capabilities.values()]
            mean = sum(capability_means) / len(capability_means)
            metrics[f"by_language/{language}/{metric}_micro"] = metrics[f"by_language/{language}/{metric}"]
            metrics[f"by_language/{language}/{metric}"] = mean
            metrics[f"by_language/{language}/{metric}_capability_count"] = len(capabilities)
            language_scores.append(mean)
        if language_scores:
            metrics[metric] = sum(language_scores) / len(language_scores)
    if "score" in metrics:
        metrics["normalized_score"] = (metrics["score"] - 1) / 4
    return metrics
