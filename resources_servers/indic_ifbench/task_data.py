# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validated task contract for the Indic IFBench resources server."""

import math
from collections.abc import Mapping
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, JsonValue, field_validator, model_validator


LANGUAGES = ("bn", "gu", "hi", "kn", "mr", "ml", "ne", "or", "pa", "ta", "te", "ur")
ALIASES = {"ka": "kn", "mar": "mr", "mal": "ml"}
EVALUATOR_VERSION = "indic-ifbench-v2"


def normalize_language(language: str) -> str:
    """Resolve the requested language aliases to dataset directory codes."""
    language = ALIASES.get(language, language)
    if language not in LANGUAGES:
        raise ValueError(f"Unsupported Indic IFBench language {language!r}; expected {LANGUAGES} or {tuple(ALIASES)}")
    return language


# An explicit contract prevents missing arguments from silently selecting random
# English defaults, and detects incompatible/new source constraints at prepare time.
INSTRUCTION_ARGS = {
    "count:word_count_range": ("min_words", "max_words"),
    "count:unique_word_count": ("N",),
    "ratio:stop_words": ("percentage",),
    "ratio:sentence_type": (),
    "ratio:sentence_balance": (),
    "count:conjunctions": ("small_n",),
    "count:person_names": ("N",),
    "ratio:overlap": ("percentage", "reference_text"),
    "count:numbers": ("N",),
    "words:alphabet": (),
    "words:vowel": (),
    "words:consonants": (),
    "sentence:alliteration_increment": (),
    "words:palindrome": (),
    "count:punctuation": (),
    "format:parentheses": (),
    "format:quotes": (),
    "words:prime_lengths": (),
    "format:options": ("options",),
    "format:newline": (),
    "format:emoji": (),
    "ratio:sentence_words": (),
    "count:words_japanese": ("N",),
    "words:repeats": ("small_n",),
    "sentence:keyword": ("N", "word"),
    "count:pronouns": ("N",),
    "words:odd_even_syllables": (),
    "words:last_first": (),
    "words:paragraph_last_first": (),
    "sentence:increment": ("small_n",),
    "words:no_consecutive": (),
    "format:line_indent": (),
    "format:quote_unquote": (),
    "format:list": ("sep",),
    "format:thesis": (),
    "format:sub-bullets": (),
    "format:no_bullets_bullets": (),
    "custom:multiples": (),
    "custom:mcq_count_length": (),
    "custom:word_reverse": (),
    "custom:character_reverse": (),
    "custom:european_capitals_sort": (),
    "custom:csv_city": (),
    "custom:csv_special_character": (),
    "custom:csv_quotes": (),
    "custom:date_format_list": (),
    "count:keywords_multiple": ("keyword1", "keyword2", "keyword3", "keyword4", "keyword5"),
    "words:keywords_specific_position": ("keyword", "n", "m"),
    "words:words_position": ("keyword",),
    "repeat:repeat_change": ("prompt_to_repeat",),
    "repeat:repeat_simple": (),
    "repeat:repeat_span": ("prompt_to_repeat", "n_start", "n_end"),
    "format:output_template": (),
    "format:no_whitespace": (),
}
_FIXED = {
    "custom:word_reverse": ("bald_eagle",),
    "custom:character_reverse": ("bald_eagle",),
    "custom:european_capitals_sort": ("european_capitals",),
    "custom:csv_city": ("csv_city_cols",),
    "custom:csv_special_character": ("csv_special_cols",),
    "custom:csv_quotes": ("csv_quotes_cols",),
    "custom:mcq_count_length": ("question_label",),
    "format:options": ("or_word",),
    "repeat:repeat_simple": ("repeat_simple_sentence",),
    "format:output_template": ("my_answer", "my_conclusion", "future_outlook"),
}
_NUMERIC = {"N", "small_n", "min_words", "max_words", "percentage", "n", "m", "n_start", "n_end"}
_URDU_EXCLUDED = {"words:alphabet", "words:vowel", "words:consonants", "words:odd_even_syllables"}


def validate_instruction(
    instruction_id: str, kwargs: Mapping[str, JsonValue], *, language: str
) -> dict[str, JsonValue]:
    """Validate and compact source kwargs, retaining translated `_lang` values."""
    language = normalize_language(language)
    if instruction_id not in INSTRUCTION_ARGS or (language == "ur" and instruction_id in _URDU_EXCLUDED):
        raise ValueError(f"Unsupported instruction {instruction_id!r} for {language}")
    required = INSTRUCTION_ARGS[instruction_id]
    compact = {k: v for k, v in kwargs.items() if v is not None}
    if set(compact) - set(required) - {"_lang"}:
        raise ValueError(f"Unexpected arguments for {instruction_id}: {set(compact) - set(required) - {'_lang'}}")
    for key in required:
        value = compact.get(key)
        if key in _NUMERIC:
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{instruction_id}.{key} must be a finite integer")
            if value != int(value) or value < 0 or (key in {"N", "small_n", "n", "m"} and value == 0):
                raise ValueError(f"Invalid integer {instruction_id}.{key}: {value}")
            compact[key] = int(value)
        elif not isinstance(value, str) or not value.strip():
            raise ValueError(f"{instruction_id}.{key} must be a nonempty string")
    if "percentage" in compact and compact["percentage"] > 100:
        raise ValueError("percentage must be between 0 and 100")
    if instruction_id == "count:word_count_range" and compact["min_words"] > compact["max_words"]:
        raise ValueError("min_words must not exceed max_words")
    if instruction_id == "repeat:repeat_span" and not (
        0 <= compact["n_start"] <= compact["n_end"] < len(compact["prompt_to_repeat"].split())
    ):
        raise ValueError("repeat span must be valid inclusive, zero-based whitespace word indices")
    metadata = compact.get("_lang", {})
    if not isinstance(metadata, dict) or set(metadata) - {"fixed", "names"}:
        raise ValueError("_lang must contain only fixed/names metadata")
    fixed = metadata.get("fixed", {})
    if not isinstance(fixed, dict):
        raise ValueError("_lang.fixed must be an object")
    for key in _FIXED.get(instruction_id, ()):
        value = fixed.get(key)
        if key.endswith("_cols") or key == "european_capitals":
            size = 27 if key == "european_capitals" else 5
            if (
                not isinstance(value, list)
                or len(value) != size
                or not all(isinstance(v, str) and v.strip() for v in value)
            ):
                raise ValueError(f"_lang.fixed.{key} must contain {size} translated strings")
        elif not isinstance(value, str) or not value.strip():
            raise ValueError(f"Missing translated _lang.fixed.{key}")
    if instruction_id == "count:person_names":
        names = metadata.get("names")
        if not isinstance(names, list) or not names or not all(isinstance(v, str) and v.strip() for v in names):
            raise ValueError("count:person_names requires translated _lang.names")
    return compact


class TaskData(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: int
    language: str
    source_key: str = ""
    uuid: str = ""
    metadata: dict[str, JsonValue] = Field(default_factory=dict)
    prompt: str = Field(min_length=1)
    instruction_id_list: list[str] = Field(min_length=1)
    kwargs: list[dict[str, JsonValue]] = Field(min_length=1)
    grading_mode: Literal["binary", "fraction"] = "fraction"

    @field_validator("language")
    @classmethod
    def validate_language(cls, value: str) -> str:
        return normalize_language(value)

    @model_validator(mode="after")
    def validate_instructions(self) -> Self:
        if not self.prompt.strip():
            raise ValueError("prompt must not be blank")
        if len(self.instruction_id_list) != len(self.kwargs):
            raise ValueError("instruction_id_list and kwargs must have equal lengths")
        self.kwargs = [
            validate_instruction(instruction, kwargs, language=self.language)
            for instruction, kwargs in zip(self.instruction_id_list, self.kwargs)
        ]
        return self
