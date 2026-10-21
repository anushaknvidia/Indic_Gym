# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import unicodedata

import pytest

from resources_servers.indic_ifbench.evaluation import evaluate
from resources_servers.indic_ifbench.languages import (
    LANGUAGE_RULES,
    aksharas,
    normalize,
    sentence_ending,
    sentences,
    vowel_nuclei,
    words,
)


@pytest.mark.parametrize(
    "language,pronoun,conjunction,content_word,terminator",
    [
        ("bn", "তাঁকে", "নয়তো", "ভাষা", "।"),
        ("gu", "હું", "અને", "ભાષા", "."),
        ("hi", "इन्हें", "और", "भाषा", "।"),
        ("kn", "ನಾನು", "ಮತ್ತು", "ಭಾಷೆ", "."),
        ("mr", "मी", "आणि", "भाषा", "।"),
        ("ml", "ഞാൻ", "എന്നാൽ", "ഭാഷ", "."),
        ("ne", "तपाईँ", "तथा", "भाषा", "।"),
        ("or", "କ'ଣ", "ଏବଂ", "ଭାଷା", "।"),
        ("pa", "ਮੈਂ", "ਅਤੇ", "ਭਾਸ਼ਾ", "।"),
        ("ta", "நான்", "மற்றும்", "மொழி", "."),
        ("te", "నేను", "మరియు", "భాష", "."),
        ("ur", "ان", "و", "زبان", "۔"),
    ],
)
def test_lexical_rules_for_each_language(language, pronoun, conjunction, content_word, terminator):
    # Combining signs must survive normalization and punctuation must not turn
    # one word into two. Check actual positive/negative scoring for every set.
    decomposed = unicodedata.normalize("NFD", f"“{pronoun} {content_word}{terminator}”")
    assert words(decomposed) == [normalize(pronoun), normalize(content_word)]
    assert sentences(decomposed) == [decomposed]
    for instruction, args, positive, negative in [
        ("count:pronouns", {"N": 1}, pronoun, content_word),
        ("count:conjunctions", {"small_n": 1}, conjunction, content_word),
        ("ratio:stop_words", {"percentage": 0}, content_word, pronoun),
    ]:
        assert evaluate([instruction], [args], language=language, response=positive)[0] == [True]
        assert evaluate([instruction], [args], language=language, response=negative)[0] == [False]


@pytest.mark.parametrize("language", LANGUAGE_RULES)
def test_lexical_inventories_use_same_normalization_as_tokens(language):
    rules = LANGUAGE_RULES[language]
    for word in rules.pronouns | rules.conjunctions | rules.stopwords:
        assert words(unicodedata.normalize("NFD", word)) == [word]


@pytest.mark.parametrize("pronoun", ["इन्हें", "इनका", "इनकी", "इनके"])
def test_hindi_common_oblique_pronouns(pronoun):
    assert evaluate(["count:pronouns"], [{"N": 1}], language="hi", response=pronoun)[0] == [True]


@pytest.mark.parametrize("pronoun", ["କ'ଣ", "କ’ଣ"])
def test_odia_apostrophes_stay_inside_words(pronoun):
    assert words(f"‘{pronoun}’ 'ଭାଷା'") == [pronoun, "ଭାଷା"]
    assert evaluate(["count:pronouns"], [{"N": 1}], language="or", response=pronoun)[0] == [True]
    assert evaluate(["count:word_count_range"], [{"min_words": 1, "max_words": 1}], language="or", response=pronoun)[
        0
    ] == [True]


def test_quotes_and_apostrophes_do_not_join_separate_words_or_numbers():
    assert words("'राम' 'सीता' १'२ १’२") == ["राम", "सीता", "१", "२", "१", "२"]


@pytest.mark.parametrize(
    "text,expected",
    [
        ("“राम है।” सीता है।", ["“राम है।”", "सीता है।"]),
        ("(राम है।) [सीता है।]", ["(राम है।)", "[सीता है।]"]),
        ("«یہ ٹھیک ہے؟» ہاں۔", ["«یہ ٹھیک ہے؟»", "ہاں۔"]),
        ("‘क्या हुआ?!’ ठीक।", ["‘क्या हुआ?!’", "ठीक।"]),
        ("“नमस्ते। ” अगला।", ["“नमस्ते। ”", "अगला।"]),
        ("[“नमस्ते। ” ]", ["[“नमस्ते। ” ]"]),
        ('नमस्ते। "अगला।"', ["नमस्ते।", '"अगला।"']),
        ('"राम है।"', ['"राम है।"']),
        ("'राम है।'", ["'राम है।'"]),
        ("मूल्य ३.१४ है। قیمت ۳٫۱۴ ہے۔", ["मूल्य ३.१४ है।", "قیمت ۳٫۱۴ ہے۔"]),
    ],
)
def test_sentence_closers_do_not_create_phantom_sentences(text, expected):
    assert sentences(text) == expected


@pytest.mark.parametrize(
    "text,ending",
    [("“नमस्ते।”", "।"), ("«کیوں؟»", "؟"), ("[वाह!]", "!"), ("क्यों?!", "?!"), ("राम", "")],
)
def test_terminal_type_ignores_closing_wrappers(text, ending):
    assert sentence_ending(text) == ending


@pytest.mark.parametrize("control", ["\u200e", "\u200f", "\u061c", "\u2067", "\u2069"])
def test_bidi_controls_do_not_create_sentences_or_hide_terminal_type(control):
    text = f"{control}کہا۔{control} کیا؟{control}"
    assert sentences(text) == ["کہا۔", "کیا؟"]
    assert sentence_ending(f"«کیا؟ »{control}") == "؟"
    assert normalize(text) == text


@pytest.mark.parametrize(
    "language,consonant",
    [("gu", "ળ"), ("kn", "ಳ"), ("ml", "ള"), ("ml", "ഴ"), ("ml", "റ"), ("or", "ଳ"), ("te", "ళ")],
)
def test_consonants_outside_alphabet_cycle_keep_their_inherent_vowel(language, consonant):
    assert consonant in LANGUAGE_RULES[language].consonants
    assert vowel_nuclei(consonant, language) == ["A"]
    assert vowel_nuclei(consonant + LANGUAGE_RULES[language].virama, language) == []


@pytest.mark.parametrize("text", ["കൗ", "കൌ", "ക\u0d46\u0d57"])
def test_malayalam_modern_and_legacy_au_have_one_vowel(text):
    assert vowel_nuclei(text, "ml") == ["AU"]


@pytest.mark.parametrize("chillu", ["ൺ", "ൻ", "ർ", "ൽ", "ൾ", "ൿ"])
def test_malayalam_chillus_have_no_inherent_vowel(chillu):
    assert vowel_nuclei(chillu, "ml") == []
    assert vowel_nuclei("ക" + chillu, "ml") == ["A"]


@pytest.mark.parametrize("atomic,legacy", [("ഞാൻ", "ഞാന്‍"), ("അവൾ", "അവള്‍"), ("അവർ", "അവര്‍")])
def test_terminal_malayalam_chillu_encodings_agree_on_written_vowels(atomic, legacy):
    assert vowel_nuclei(atomic, "ml") == vowel_nuclei(legacy, "ml")


def test_marathi_candra_a_is_a_vowel():
    assert vowel_nuclei("ॲ", "mr") == ["CANDRA A"]


@pytest.mark.parametrize(
    "text,units",
    [
        ("தீர்ப்பு", ["தீ", "ர்", "ப்", "பு"]),
        ("க்\u200dகா", ["க்\u200d", "கா"]),
        ("க்ஷே", ["க்ஷே"]),
        ("ஶ்ரீ", ["ஶ்ரீ"]),
        ("ஸ்ரீ", ["ஸ்ரீ"]),
        ("இஶ்ரேல்", ["இ", "ஶ்", "ரே", "ல்"]),
    ],
)
def test_tamil_pulli_boundaries_and_conjunct_exceptions(text, units):
    assert aksharas(text) == units


def test_tamil_pulli_does_not_add_a_vowel():
    assert vowel_nuclei("தீர்ப்பு", "ta") == ["II", "U"]


@pytest.mark.parametrize(
    "text,units",
    [
        ("क्\u200cष", ["क्\u200c", "ष"]),
        ("क्\u200dष", ["क्\u200dष"]),
        ("க்\u200cஷா", ["க்\u200c", "ஷா"]),
        ("க்\u200dஷா", ["க்\u200dஷா"]),
    ],
)
def test_explicit_zwnj_keeps_grapheme_boundary_and_zwj_keeps_conjunct(text, units):
    assert aksharas(text) == units
