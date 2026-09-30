# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from resources_servers.biggenbench.metrics import compute_metrics


def row(score, *, language="hi", task="travel_plan", capability="planning", **kwargs):
    return {
        "score": score,
        "verifier_metadata": {"language": language, "task": task, "capability": capability},
        **kwargs,
    }


def test_scores_exclusions_and_language_breakdowns():
    metrics = compute_metrics(
        [[row(5)], [row(3, language="bn")], [row(1, task="moral_belief")], [row(2, capability="multilingual")]]
    )
    assert metrics["score"] == 4
    assert metrics["score_all"] == 2.75
    assert metrics["score_all_micro"] == 2.75
    assert metrics["normalized_score"] == 0.75
    assert metrics["aggregate_count"] == 2
    assert metrics["excluded_count"] == 2
    assert metrics["by_language/hi/score"] == 5
    assert metrics["by_language/hi/score_all"] == 2.5
    assert metrics["by_language/hi/score_all_micro"] == pytest.approx(8 / 3)
    assert metrics["by_task/moral_belief/score_all"] == 1
    assert "by_task/moral_belief/score" not in metrics


@pytest.mark.parametrize("bad", [None, 0, 6, True, 2.5, "5"])
def test_invalid_judge_scores_never_lower_valid_mean(bad):
    metrics = compute_metrics([[row(4), row(bad)]])
    assert metrics["score"] == 4
    assert metrics["judge_error_count"] == 1
    assert metrics["valid_count"] == 1


def test_mask_and_errors_exclude_even_numeric_scores():
    metrics = compute_metrics([[row(1, mask_sample=True), row(2, judge_error="timeout"), row(5)]])
    assert metrics["score"] == 5
    assert metrics["judge_error_count"] == 2


def test_empty_or_all_failed_does_not_manufacture_a_score():
    assert "score" not in compute_metrics([])
    metrics = compute_metrics([[row(None)]])
    assert "score" not in metrics
    assert metrics["valid_count"] == 0


def test_equal_language_weighting_and_explicit_exclusion():
    excluded = row(1)
    excluded["verifier_metadata"]["include_in_aggregate"] = False
    metrics = compute_metrics([[row(5), row(3)], [row(1, language="bn")], [excluded]])
    assert metrics["score"] == 2.5
    assert metrics["score_all"] == 2
    assert metrics["score_micro"] == 3
    assert metrics["score_all_micro"] == 2.5


def test_capabilities_have_equal_weight_despite_unequal_task_counts():
    metrics = compute_metrics([[row(5), row(5)], [row(1, capability="reasoning")]])
    assert metrics["score"] == 3
    assert metrics["score_micro"] == pytest.approx(11 / 3)
    assert metrics["by_language/hi/score_capability_count"] == 2


def test_invalid_metadata_is_reported():
    with pytest.raises(ValueError, match="mapping"):
        compute_metrics([[{"score": 3, "verifier_metadata": "broken"}]])
