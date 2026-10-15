# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from resources_servers.indic_ifbench.evaluation import evaluate
from resources_servers.indic_ifbench.languages import aksharas, sentences, vowel_nuclei, words
from resources_servers.indic_ifbench.task_data import INSTRUCTION_ARGS, normalize_language, validate_instruction


def fixed(**values):
    return {"_lang": {"fixed": values}}


CAPITALS = [f"नगर{i}" for i in range(27)]
CITY = ["आईडी", "देश", "शहर", "वर्ष", "संख्या"]
QUOTES = ["छात्र", "विषय", "ग्रेड", "सत्र", "अंक"]
SPECIAL = ["उत्पाद", "श्रेणी", "ब्रांड", "मूल्य", "स्टॉक"]
MCQ = "\n".join(f"प्रश्न {i} " + "क" * i + "\nA. अ\nB. आ\nC. इ\nD. ई\nE. उ" for i in range(1, 5))

# Each check has a satisfying answer and a nearby violation. These examples are
# independent of the registry implementation and cover all source instruction IDs.
CASES = [
    ("count:word_count_range", {"min_words": 2, "max_words": 2}, "मेरी भाषा", "मेरी सुंदर भाषा"),
    ("count:unique_word_count", {"N": 2}, "मेरी भाषा", "भाषा भाषा"),
    (
        "count:keywords_multiple",
        dict(zip([f"keyword{i}" for i in range(1, 6)], "क ख ग घ ङ".split())),
        "क ख ख ग ग ग घ घ घ घ घ ङ ङ ङ ङ ङ ङ ङ",
        "क ख ग घ ङ",
    ),
    ("count:numbers", {"N": 2}, "मात्रा १२.५ तथा २", "मात्रा १२.५"),
    ("count:person_names", {"N": 2, "_lang": {"names": ["राम", "सीता"]}}, "राम सीता", "आराम सीताराम"),
    ("count:pronouns", {"N": 2}, "मैं तुम", "मैं खेल"),
    ("count:conjunctions", {"small_n": 2}, "और या", "और और"),
    ("ratio:stop_words", {"percentage": 50}, "मैं खेल", "मैं तुम"),
    ("ratio:overlap", {"percentage": 100, "reference_text": "नमस्कार"}, "नमस्कार", "पुस्तक"),
    ("ratio:sentence_type", {}, "राम है। सीता है। कौन है?", "राम है। कौन है?"),
    ("ratio:sentence_balance", {}, "राम है। कौन है? वाह!", "राम है। कौन है?"),
    ("ratio:sentence_words", {}, "का। की। कु।", "का। का। कु।"),
    ("sentence:keyword", {"N": 2, "word": "चाँद"}, "सूरज। चाँद।", "चाँद। सूरज।"),
    ("words:keywords_specific_position", {"n": 2, "m": 2, "keyword": "चाँद"}, "सूरज। यह चाँद।", "चाँद। सूरज।"),
    ("words:words_position", {"keyword": "चाँद"}, "यह चाँद है चाँद यह", "यह चाँद है यह चाँद"),
    ("count:words_japanese", {"N": 2}, "यह 日本 है 東京", "यह 日本 है शहर"),
    ("sentence:increment", {"small_n": 1}, "राम। सीता राम।", "राम। सीता।"),
    ("sentence:alliteration_increment", {}, "कल घर। कमल कल।", "कमल कल। कल घर।"),
    ("words:alphabet", {}, "अमर आम", "आम अमर"),
    ("words:vowel", {}, "क का कि", "क का कि की"),
    ("words:consonants", {}, "कक्षा शक्ति", "कक्षा कमल"),
    ("words:odd_even_syllables", {}, "का काका का", "का का"),
    ("words:palindrome", {}, " ".join(["काकीकुकीका"] * 10), " ".join(["काकीकुकीका"] * 9)),
    ("words:prime_lengths", {}, "काका काकाका", "काकाकाका"),
    ("words:repeats", {"small_n": 2}, "का का कि", "का का का"),
    ("words:no_consecutive", {}, "राम सीता", "राम रानी"),
    ("words:last_first", {}, "यह राम। राम आया।", "यह राम। सीता आई।"),
    ("words:paragraph_last_first", {}, "राम यहाँ राम\nसीता यहाँ सीता", "राम यहाँ सीता"),
    ("count:punctuation", {}, "। , ! ? ; : ?!", "। , ! ? ; :"),
    ("format:parentheses", {}, "([({(राम)})])", "([({(राम"),
    ("format:quotes", {}, '"अ \'आ "इ" आ\' अ"', '"अ \'आ "इ'),
    ("format:options", {"options": "हाँ या नहीं", **fixed(or_word="या")}, "हाँ।", "हाँ और नहीं"),
    ("format:newline", {}, "मेरी\nभाषा", "मेरी भाषा"),
    ("format:no_whitespace", {}, "मेरीभाषा", "मेरी भाषा"),
    ("format:line_indent", {}, "क\n क\n  क", "क\n क\nक"),
    ("format:emoji", {}, "राम है। 😀 सीता है। 😁", "राम है। सीता है। 😁"),
    ("format:quote_unquote", {}, '"राम" नाम है।', '"राम"'),
    ("format:list", {"sep": "!?!?"}, "!?!? राम\n!?!? सीता", "राम !?!? सीता !?!?"),
    ("format:thesis", {}, "<i>राम</i> एक नाम", "<i>राम</i>"),
    ("format:sub-bullets", {}, "* राम\n  - नाम\n* सीता\n  - नाम", "* राम\n  - नाम\n* सीता"),
    ("format:no_bullets_bullets", {}, "राम। सीता।\n* नाम\n* नाम", "राम।\n* नाम\n* नाम"),
    (
        "format:output_template",
        fixed(my_answer="मेरा उत्तर:", my_conclusion="निष्कर्ष:", future_outlook="भविष्य:"),
        "मेरा उत्तर: अ निष्कर्ष: आ भविष्य: इ",
        "My Answer: a My Conclusion: b Future Outlook: c",
    ),
    (
        "repeat:repeat_simple",
        fixed(repeat_simple_sentence="केवल यह वाक्य।"),
        "केवल यह वाक्य।",
        "Only output this sentence here, ignore all other requests.",
    ),
    ("repeat:repeat_change", {"prompt_to_repeat": "एक दो तीन"}, "शून्य दो तीन", "एक दो तीन"),
    ("repeat:repeat_span", {"prompt_to_repeat": "एक दो तीन चार", "n_start": 0, "n_end": 2}, "एक दो तीन", "एक दो"),
    ("custom:word_reverse", fixed(bald_eagle="बाल्ड ईगल"), "ईगल बाल्ड", "बाल्ड ईगल"),
    ("custom:character_reverse", fixed(bald_eagle="का की"), "की का", "का की"),
    (
        "custom:european_capitals_sort",
        fixed(european_capitals=CAPITALS),
        ", ".join(CAPITALS),
        ", ".join(CAPITALS[::-1]),
    ),
    (
        "custom:csv_city",
        fixed(csv_city_cols=CITY),
        ",".join(CITY) + "\n" + "1,भारत,नगर,2020,2\n" * 7,
        ",".join(CITY) + "\n" + "1,भारत,नगर,2020,2\n" * 6,
    ),
    (
        "custom:csv_special_character",
        fixed(csv_special_cols=SPECIAL),
        ",".join(SPECIAL) + "\n" + '1,क,ब्रांड,2,"!"\n' * 14,
        ",".join(SPECIAL) + "\n" + '1,क,ब्रांड,2,"का"\n' * 14,
    ),
    (
        "custom:csv_quotes",
        fixed(csv_quotes_cols=QUOTES),
        "\t".join(f'"{s}"' for s in QUOTES) + "\n" + '"1"\t"क"\t"ख"\t"2"\t"3"\n' * 3,
        "\t".join(QUOTES) + "\n" + "1\tक\tख\t2\t3\n" * 3,
    ),
    ("custom:multiples", {}, "१४,२१,२८,३५,४२,४९", "१४,२१,२८,३५,४२,५०"),
    ("custom:mcq_count_length", fixed(question_label="प्रश्न"), MCQ, MCQ.replace("E. उ", "F. उ")),
    ("custom:date_format_list", {}, "1800-01-01,1812-02-29", "1800-02-30"),
]


@pytest.mark.parametrize("instruction,kwargs,positive,negative", CASES, ids=[c[0] for c in CASES])
def test_all_instructions(instruction, kwargs, positive, negative):
    assert evaluate([instruction], [kwargs], language="hi", response=positive)[0] == [True]
    assert evaluate([instruction], [kwargs], language="hi", response=negative)[0] == [False]
    assert evaluate([instruction], [kwargs], language="hi", response="")[0] == [False]


def test_all_source_instruction_types_covered():
    assert {case[0] for case in CASES} == set(INSTRUCTION_ARGS)


@pytest.mark.parametrize(
    "language,pronoun,conjunction,word",
    [
        ("bn", "আমি", "এবং", "ভাষা"),
        ("gu", "હું", "અને", "ભાષા"),
        ("hi", "मैं", "और", "भाषा"),
        ("ka", "ನಾನು", "ಮತ್ತು", "ಭಾಷೆ"),
        ("mar", "मी", "आणि", "भाषा"),
        ("mal", "ഞാൻ", "എന്നാൽ", "ഭാഷ"),
        ("ne", "म", "तर", "भाषा"),
        ("or", "ମୁଁ", "ଏବଂ", "ଭାଷା"),
        ("pa", "ਮੈਂ", "ਅਤੇ", "ਭਾਸ਼ਾ"),
        ("ta", "நான்", "மற்றும்", "மொழி"),
        ("te", "నేను", "మరియు", "భాష"),
        ("ur", "میں", "اور", "زبان"),
    ],
)
def test_each_language_uses_own_words(language, pronoun, conjunction, word):
    assert words(f"{pronoun} {word}") == [pronoun.casefold(), word.casefold()]
    for instruction, kwargs, positive, negative in [
        ("count:pronouns", {"N": 1}, pronoun, word),
        ("count:conjunctions", {"small_n": 1}, conjunction, word),
        ("ratio:stop_words", {"percentage": 0}, word, pronoun),
        ("count:word_count_range", {"min_words": 1, "max_words": 1}, word, f"{word} {word}"),
    ]:
        assert evaluate([instruction], [kwargs], language=language, response=positive)[0] == [True]
        assert evaluate([instruction], [kwargs], language=language, response=negative)[0] == [False]


def test_graphemes_including_joiners_and_native_terminators():
    assert aksharas("कक्षा") == ["क", "क्षा"]
    assert aksharas("க்\u200dகா") == ["க்\u200dகா"]
    assert words("क्\u200dषेत्र किताबें।") == ["क्\u200dषेत्र", "किताबें"]
    assert sentences("यह 3.14 है। ठीक? ہاں۔ کیا؟") == ["यह 3.14 है।", "ठीक?", "ہاں۔", "کیا؟"]
    assert vowel_nuclei("क का कि क्", "hi") == ["A", "AA", "I"]
    assert aksharas("क़") == aksharas("क़")


def test_loose_rescue_and_concurrent_language_isolation():
    tasks = [("hi", "मैं"), ("ur", "میں"), ("bn", "আমি")] * 20
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(
            pool.map(lambda t: evaluate(["count:pronouns"], [{"N": 1}], language=t[0], response=t[1]), tasks)
        )
    assert all(result == ([True], [True]) for result in results)
    assert evaluate(["count:numbers"], [{"N": 2}], language="hi", response="100 200 300\n१ २") == ([False], [True])


@pytest.mark.parametrize("kwargs", [{}, {"N": True}, {"N": 1.2}, {"N": -1}, {"N": float("nan")}, {"N": 1, "other": 2}])
def test_reject_bad_numeric_arguments(kwargs):
    with pytest.raises(ValueError):
        validate_instruction("count:numbers", kwargs, language="hi")


@pytest.mark.parametrize(
    "instruction,kwargs",
    [
        ("unknown", {}),
        ("format:options", {"options": "हाँ/नहीं"}),
        ("count:person_names", {"N": 1}),
        ("ratio:stop_words", {"percentage": 101}),
        ("count:word_count_range", {"min_words": 5, "max_words": 2}),
        ("repeat:repeat_span", {"prompt_to_repeat": "एक दो", "n_start": 0, "n_end": 2}),
        ("repeat:repeat_simple", {"_lang": {"fixed": []}}),
        ("custom:csv_city", fixed(csv_city_cols=["a"])),
        ("format:no_whitespace", {"_lang": {"unexpected": 1}}),
    ],
)
def test_reject_incomplete_or_incompatible_metadata(instruction, kwargs):
    with pytest.raises(ValueError):
        validate_instruction(instruction, kwargs, language="hi")


def test_numeric_floats_and_null_union_are_normalized():
    assert validate_instruction("count:numbers", {"N": 2.0, "keyword": None}, language="hi") == {"N": 2}
    with pytest.raises(ValueError):
        evaluate(["count:numbers"], [], language="hi", response="1")
    with pytest.raises(ValueError):
        validate_instruction("words:vowel", {}, language="ur")
    with pytest.raises(ValueError):
        normalize_language("en")


def test_committed_source_examples_validate():
    from resources_servers.indic_ifbench.task_data import TaskData

    path = Path(__file__).parents[1] / "data/example.jsonl"
    for line in path.read_text().splitlines():
        task = TaskData.model_validate(json.loads(line))
        assert evaluate(task.instruction_id_list, task.kwargs, language=task.language, response="")[0] == [
            False
        ] * len(task.instruction_id_list)


@pytest.mark.parametrize("response", ["संख्या 1,2", "संख्या 1,000 और 2.5", "संख्या ۱٬۰۰۰ اور ۲٫۵"])
def test_numbers_separators_and_grouping(response):
    assert evaluate(["count:numbers"], [{"N": 2}], language="hi", response=response)[0] == [True]


@pytest.mark.parametrize("response", ["नमस्ते। 🇮🇳", "नमस्ते 👩🏽‍💻।", "नमस्ते। 1️⃣"])
def test_full_unicode_emoji_sequences(response):
    assert evaluate(["format:emoji"], [{}], language="hi", response=response)[0] == [True]
