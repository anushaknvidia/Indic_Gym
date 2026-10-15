# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Deterministic Indic IFBench checks, following the supplied translated prompts.

Language-neutral constraints are adapted from AllenAI IFBench at
c6767a19bd82ac0536cab950f2f8f6bcc6fabe7c (Apache-2.0, Copyright 2025 Allen
Institute for AI). Indic rules and fixes are documented in README.md.
"""

import csv
import io
import math
import re
from collections import Counter
from collections.abc import Mapping

import emoji
import regex
from pydantic import JsonValue

from resources_servers.indic_ifbench.languages import (
    LANGUAGE_RULES,
    aksharas,
    normalize,
    sentences,
    vowel_nuclei,
    words,
)
from resources_servers.indic_ifbench.task_data import normalize_language, validate_instruction


def _nested(value: str, *, quotes: bool = False) -> bool:
    stack = []
    deepest = 0
    matched = False
    pairs = {")": "(", "]": "[", "}": "{"}
    for char in value:
        if quotes:
            if char not in "\"'":
                continue
            if stack and stack[-1] == char:
                stack.pop()
            else:
                stack.append(char)
        elif char in "([{":
            stack.append(char)
        elif char in pairs:
            if not stack or stack.pop() != pairs[char]:
                return False
        else:
            continue
        deepest = max(deepest, len(stack))
        if not stack:
            matched |= deepest >= (3 if quotes else 5)
            deepest = 0
    return matched and not stack


def _csv_check(value: str, *, columns: list[str], kind: str) -> bool:
    delimiter = "\t" if kind == "quotes" else ","
    try:
        rows = list(csv.reader(io.StringIO(value), delimiter=delimiter, strict=True))
    except csv.Error:
        return False
    count = {"city": 7, "special_character": 14, "quotes": 3}[kind]
    if len(rows) != count + 1 or rows[0] != columns or any(len(row) != 5 for row in rows):
        return False
    if kind == "quotes":
        field = r'"(?:[^"\r\n]|"")*"'
        return all(re.fullmatch(field + ("\t" + field) * 4, line) for line in value.splitlines())
    if kind == "special_character":
        # Verify a quoted field contains punctuation/symbols, rather than treating
        # Indic combining vowel signs as special punctuation via ASCII \w.
        return any(
            regex.search(r"[\p{P}\p{S}]", field.replace('""', '"'))
            for field in re.findall(r'"((?:[^"\n]|"")*)"', "\n".join(value.splitlines()[1:]))
        )
    return True


def _mcq(value: str, label: str) -> bool:
    if not value.startswith(label):
        return False
    questions = re.split(r"\n*(?:" + re.escape(label) + r"\s+\d+[.):;]?\s*)", value)
    questions = [q.strip() for q in questions if q.strip()]
    if len(questions) != 4:
        return False
    lengths = []
    for question in questions:
        options = list(re.finditer(r"(?m)^\s*([A-Ea-e])[.)]\s*\S.*$", question))
        if [m[1].upper() for m in options] != list("ABCDE"):
            return False
        lengths.append(len(aksharas(question[: options[0].start()].strip())))
    return all(left < right for left, right in zip(lengths, lengths[1:]))


def check_instruction(instruction_id: str, kwargs: Mapping[str, JsonValue], *, language: str, response: str) -> bool:
    """Check one validated instruction against a response; no global mutable state."""
    if not response.strip():
        return False
    k = kwargs
    rules = LANGUAGE_RULES[language]
    fixed = k.get("_lang", {}).get("fixed", {})
    text = normalize(response)
    tokens = words(text)
    sents = sentences(text)
    match instruction_id:
        case "count:word_count_range":
            return k["min_words"] <= len(tokens) <= k["max_words"]
        case "count:unique_word_count":
            return len(set(tokens)) >= k["N"]
        case "count:keywords_multiple":
            return all(text.count(normalize(k[f"keyword{i}"])) == count for i, count in enumerate((1, 2, 3, 5, 7), 1))
        case "count:numbers":
            return len(regex.findall(r"(?:\d{1,3}(?:[,٬]\d{3})+|\d+)(?:[.٫]\d+)?", text)) == k["N"]
        case "count:person_names":
            return (
                sum(
                    bool(
                        regex.search(
                            r"(?<![\p{L}\p{M}\p{N}])" + regex.escape(normalize(name)) + r"(?![\p{L}\p{M}\p{N}])", text
                        )
                    )
                    for name in set(k["_lang"]["names"])
                )
                >= k["N"]
            )
        case "count:pronouns":
            return sum(t in rules.pronouns for t in tokens) >= k["N"]
        case "count:conjunctions":
            return len(set(tokens) & rules.conjunctions) >= k["small_n"]
        case "ratio:stop_words":
            return bool(tokens) and 100 * sum(t in rules.stopwords for t in tokens) / len(tokens) <= k["percentage"]
        case "ratio:overlap":
            # IFBench defines overlap using character trigrams, not word trigrams.
            grams = {text[i : i + 3] for i in range(len(text) - 2)}
            reference = normalize(k["reference_text"])
            ref = {reference[i : i + 3] for i in range(len(reference) - 2)}
            return bool(grams) and abs(100 * len(grams & ref) / len(grams) - k["percentage"]) <= 2
        case "ratio:sentence_type" | "ratio:sentence_balance":
            declarations = sum(s.endswith((".", "।", "॥", "۔")) for s in sents)
            questions = sum(s.endswith(("?", "؟")) for s in sents)
            if instruction_id.endswith("type"):
                return questions > 0 and declarations == 2 * questions
            exclamations = sum(s.endswith("!") for s in sents)
            return questions > 0 and declarations == questions == exclamations
        case "ratio:sentence_words":
            token_lists = [words(s) for s in sents]
            all_words = [w for ts in token_lists for w in ts]
            return (
                len(sents) == 3
                and len({len(aksharas(s)) for s in sents}) == 1
                and len(all_words) == len(set(all_words))
            )
        case "sentence:keyword":
            return len(sents) >= k["N"] and normalize(k["word"]) in sents[k["N"] - 1]
        case "words:keywords_specific_position":
            target = words(sents[k["n"] - 1]) if len(sents) >= k["n"] else []
            return len(target) >= k["m"] and target[k["m"] - 1] == normalize(k["keyword"])
        case "words:words_position":
            return len(tokens) >= 2 and tokens[1] == tokens[-2] == normalize(k["keyword"])
        case "count:words_japanese":
            selected = tokens[k["N"] - 1 :: k["N"]]
            return bool(selected) and all(regex.search(r"[\p{Hiragana}\p{Katakana}\p{Han}]", t) for t in selected)
        case "sentence:increment":
            lengths = [len(words(s)) for s in sents]
            return bool(lengths) and all(b - a == k["small_n"] for a, b in zip(lengths, lengths[1:]))
        case "sentence:alliteration_increment":
            counts = []
            for sentence in sents:
                run = maximum = 0
                previous = None
                for token in words(sentence):
                    initial = token[0]
                    run = run + 1 if initial == previous else 1
                    maximum = max(maximum, run)
                    previous = initial
                counts.append(maximum)
            return bool(counts) and all(a < b for a, b in zip(counts, counts[1:]))
        case "words:alphabet":
            alphabet = rules.alphabet
            starts = [t[0] for t in tokens]
            return (
                bool(starts)
                and all(c in alphabet for c in starts)
                and all(alphabet[(alphabet.index(a) + 1) % len(alphabet)] == b for a, b in zip(starts, starts[1:]))
            )
        case "words:vowel":
            return (
                bool(tokens)
                and len(response.strip().splitlines()) == 1
                and len({v for t in tokens for v in vowel_nuclei(t, language)}) <= 3
            )
        case "words:consonants":
            consonants = "".join(c for c in rules.alphabet if c not in rules.vowels)
            pattern = f"[{consonants}]\\p{{M}}*{rules.virama}[\\u200c\\u200d]*[{consonants}]"
            return bool(tokens) and all(regex.search(pattern, t) for t in tokens)
        case "words:odd_even_syllables":
            counts = [len(vowel_nuclei(t, language)) for t in tokens]
            return bool(counts) and all(counts) and all(a % 2 != b % 2 for a, b in zip(counts, counts[1:]))
        case "words:palindrome":
            units = [aksharas(t) for t in tokens]
            return sum(len(u) >= 5 and u == u[::-1] for u in units) >= 10
        case "words:prime_lengths":
            lengths = [len(aksharas(t)) for t in tokens]
            return bool(lengths) and all(n >= 2 and all(n % d for d in range(2, math.isqrt(n) + 1)) for n in lengths)
        case "words:repeats":
            return bool(tokens) and max(Counter(tokens).values()) <= k["small_n"]
        case "words:no_consecutive":
            return bool(tokens) and all(a[0] != b[0] for a, b in zip(tokens, tokens[1:]))
        case "words:last_first":
            ts = [words(s) for s in sents]
            return bool(ts) and all(ts) and all(a[-1] == b[0] for a, b in zip(ts, ts[1:]))
        case "words:paragraph_last_first":
            paragraphs = [words(p) for p in text.splitlines() if p.strip()]
            return bool(paragraphs) and all(p and p[0] == p[-1] for p in paragraphs)
        case "count:punctuation":
            value = text.translate(str.maketrans({"।": ".", "॥": ".", "۔": ".", "؟": "?", "،": ",", "؛": ";"}))
            match_ = re.search(r"\?!|!\?|‽", value)
            return bool(match_) and set(".,!?;:") <= set(value[: match_.start()] + value[match_.end() :])
        case "format:parentheses":
            return _nested(response)
        case "format:quotes":
            return _nested(response, quotes=True)
        case "format:options":
            options = k["options"]
            separator = "/" if "/" in options else fixed["or_word"] if fixed["or_word"] in options else ","
            options = [normalize(o.strip()) for o in options.split(separator)]
            if all(re.fullmatch(r"[a-dA-D]\)", o) for o in options):
                return text in options
            clean = lambda s: regex.sub(r"^[\p{P}\s]+|[\p{P}\s]+$", "", s)
            return clean(text) in [clean(o) for o in options]
        case "format:newline":
            lines = [line for line in text.splitlines() if line.strip()]
            return bool(lines) and all(len(words(line)) == 1 for line in lines)
        case "format:no_whitespace":
            return not any(c.isspace() for c in response)
        case "format:line_indent":
            lines = [line for line in response.splitlines() if line.strip()]
            indents = [len(line) - len(line.lstrip(" ")) for line in lines]
            return bool(indents) and all(a < b for a, b in zip(indents, indents[1:]))
        case "format:emoji":
            # An emoji just after a terminator belongs to the preceding
            # sentence. Graphemes preserve flags, skin tones, and ZWJ sequences.
            for index, sentence in enumerate(sents):
                units = regex.findall(r"\X", sentence.rstrip(" .!?।॥۔؟\n\t"))
                if units and emoji.is_emoji(units[-1]):
                    continue
                following = regex.findall(r"\X", sents[index + 1]) if index + 1 < len(sents) else []
                if not following or not emoji.is_emoji(following[0]):
                    return False
            return bool(sents)
        case "format:quote_unquote":
            value = response.replace("“", '"').replace("”", '"')
            quotes = list(re.finditer(r'"[^"\n]+"', value))
            return bool(quotes) and all(
                words(value[m.end() : quotes[i + 1].start() if i + 1 < len(quotes) else None])
                for i, m in enumerate(quotes)
            )
        case "format:list":
            return len(re.findall(r"(?m)^\s*" + re.escape(k["sep"]), response)) >= 2
        case "format:thesis":
            match_ = re.search(r"<(i|em)>(.+?)</\1>", response, re.DOTALL)
            return bool(match_ and match_[2].strip() and words(response[match_.end() :]))
        case "format:sub-bullets":
            bullets = re.split(r"(?m)^\s*\*\s+", response)[1:]
            return bool(bullets) and all(re.search(r"(?m)^\s*-\s+\S", b) for b in bullets)
        case "format:no_bullets_bullets":
            lines = response.splitlines()
            first = next((i for i, line in enumerate(lines) if line.strip().startswith("*")), len(lines))
            return (
                len(sentences("\n".join(lines[:first]))) >= 2
                and len(lines[first:]) >= 2
                and all(line.strip().startswith("*") for line in lines[first:])
            )
        case "format:output_template":
            return all(normalize(fixed[key]) in text for key in ("my_answer", "my_conclusion", "future_outlook"))
        case "repeat:repeat_simple":
            return text.strip() == normalize(fixed["repeat_simple_sentence"]).strip()
        case "repeat:repeat_change":
            original = normalize(k["prompt_to_repeat"]).split()
            actual = text.split()
            return len(actual) == len(original) and actual[0] != original[0] and actual[1:] == original[1:]
        case "repeat:repeat_span":
            return text.split() == normalize(k["prompt_to_repeat"]).split()[k["n_start"] : k["n_end"] + 1]
        case "custom:word_reverse":
            return normalize(fixed["bald_eagle"]) in " ".join(tokens[::-1])
        case "custom:character_reverse":
            return "".join(aksharas(fixed["bald_eagle"])[::-1]) in text
        case "custom:european_capitals_sort":
            return [c.strip() for c in text.split(",")] == [normalize(c) for c in fixed["european_capitals"]]
        case "custom:csv_city" | "custom:csv_special_character" | "custom:csv_quotes":
            kind = instruction_id.removeprefix("custom:csv_")
            key = {"city": "csv_city_cols", "special_character": "csv_special_cols", "quotes": "csv_quotes_cols"}[kind]
            return _csv_check(response, columns=fixed[key], kind=kind)
        case "custom:multiples":
            return [int(n) for n in regex.findall(r"\d+", response)] == list(range(14, 51, 7))
        case "custom:mcq_count_length":
            return _mcq(response, fixed["question_label"])
        case "custom:date_format_list":
            from datetime import date

            for part in response.strip().split(","):
                if not re.fullmatch(r"\s*\d{4}-\d{2}-\d{2}\s*", part):
                    return False
                year, month, day = map(int, part.split("-"))
                try:
                    date(year, month, day)
                except ValueError:
                    return False
                if not 1769 <= year <= 1821:
                    return False
            return True
        case _:
            raise ValueError(f"Unsupported instruction: {instruction_id}")


def loose_variants(response: str) -> list[str]:
    """The same eight strict/loose candidates used by Gym's English IFBench."""
    lines = response.split("\n")
    candidates = [
        response,
        "\n".join(lines[1:]).strip(),
        "\n".join(lines[:-1]).strip(),
        "\n".join(lines[1:-1]).strip(),
    ]
    return [response, response.replace("*", ""), *candidates[1:], *(c.replace("*", "") for c in candidates[1:])]


def evaluate(
    instruction_ids: list[str], kwargs: list[dict[str, JsonValue]], *, language: str, response: str
) -> tuple[list[bool], list[bool]]:
    """Return per-instruction strict and loose decisions on a validated task."""
    language = normalize_language(language)
    if not instruction_ids or len(instruction_ids) != len(kwargs):
        raise ValueError("instruction_id_list and kwargs must be nonempty and have equal lengths")
    strict, loose = [], []
    variants = loose_variants(response)
    for instruction_id, arguments in zip(instruction_ids, kwargs):
        arguments = validate_instruction(instruction_id, arguments, language=language)
        passed = check_instruction(instruction_id, arguments, language=language, response=response)
        strict.append(passed)
        loose.append(
            passed
            or any(
                check_instruction(instruction_id, arguments, language=language, response=v)
                for v in variants[1:]
                if v.strip()
            )
        )
    return strict, loose
