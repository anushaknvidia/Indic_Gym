# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-data schema for complete-solution HumanEval verification."""

import keyword

from pydantic import BaseModel, ConfigDict, Field, field_validator


class TaskData(BaseModel):
    """Fields owned by the HumanEval reference verifier."""

    model_config = ConfigDict(extra="allow")

    uuid: str = Field(
        pattern=r"^indic/human_eval/[a-z]+/\d+$",
        description="Unique language/problem identity for provenance.",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
    task_id: str = Field(
        pattern=r"^HumanEval/(0|[1-9]\d*)$",
        description="Original HumanEval task identifier used by the checker.",
        json_schema_extra={"legacy_location": "verifier_metadata", "consumed_by": ["verify", "provenance"]},
    )
    language: str = Field(
        pattern=r"^[a-z]+$",
        description="Prompt language used for per-language metrics.",
        json_schema_extra={"legacy_location": "verifier_metadata", "consumed_by": ["metrics", "provenance"]},
    )
    test: str = Field(
        min_length=1,
        description="Original HumanEval test program.",
        json_schema_extra={"legacy_location": "verifier_metadata", "consumed_by": ["verify"]},
    )
    entry_point: str = Field(
        pattern=r"^[A-Za-z_]\w*$",
        description="Candidate function passed to the HumanEval check function.",
        json_schema_extra={"legacy_location": "verifier_metadata", "consumed_by": ["verify"]},
    )

    @field_validator("entry_point")
    @classmethod
    def reject_python_keyword(cls, value: str) -> str:
        if keyword.iskeyword(value):
            raise ValueError("entry_point must not be a Python keyword")
        return value
