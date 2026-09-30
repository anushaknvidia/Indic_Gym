# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Verifier metadata for task-specific, absolute BiGGen Bench assessment."""

from pydantic import BaseModel, ConfigDict, Field


_VM = {"legacy_location": "verifier_metadata"}


class ScoreRubric(BaseModel):
    """The original task's criterion and descriptions for all five ratings."""

    criteria: str = Field(min_length=1)
    score1_description: str = Field(min_length=1)
    score2_description: str = Field(min_length=1)
    score3_description: str = Field(min_length=1)
    score4_description: str = Field(min_length=1)
    score5_description: str = Field(min_length=1)


class TaskData(BaseModel):
    """Flat schema, carried on the wire in ``verifier_metadata`` only."""

    model_config = ConfigDict(extra="allow")

    id: str = Field(json_schema_extra={"consumed_by": ["verify", "provenance"], **_VM})
    language: str = Field(json_schema_extra={"consumed_by": ["metrics"], **_VM})
    capability: str = Field(json_schema_extra={"consumed_by": ["metrics"], **_VM})
    task: str = Field(json_schema_extra={"consumed_by": ["metrics"], **_VM})
    system_prompt: str = Field(default="", json_schema_extra={"consumed_by": ["verify"], **_VM})
    input: str = Field(min_length=1, json_schema_extra={"consumed_by": ["verify"], **_VM})
    reference_answer: str = Field(min_length=1, json_schema_extra={"consumed_by": ["verify"], **_VM})
    score_rubric: ScoreRubric = Field(json_schema_extra={"consumed_by": ["verify"], **_VM})
    include_in_aggregate: bool = Field(default=True, json_schema_extra={"consumed_by": ["metrics"], **_VM})
    source_input: str | None = Field(default=None, json_schema_extra={"consumed_by": ["provenance"], **_VM})
