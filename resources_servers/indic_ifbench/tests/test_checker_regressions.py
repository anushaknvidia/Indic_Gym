# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import unicodedata

import pytest

from resources_servers.indic_ifbench.evaluation import evaluate


# Translated headings from source key 292, retained exactly as supplied.
TEMPLATES = [
    ("bn", "আমার উত্তর:", "আমার উপসংহার:", "ভবিষ্যত দৃষ্টিভঙ্গি:"),
    ("gu", "મારો જવાબ:", "મારો નિષ્કર્ષ:", "ભવિષ્યનો દૃષ્ટિકોણ:"),
    ("hi", "मेरा उत्तर:", "मेरा निष्कर्ष:", "भविष्य का परिदृश्य:"),
    ("kn", "ನನ್ನ ಉತ್ತರ:", "ನನ್ನ ತೀರ್ಮಾನ:", "ಭವಿಷ್ಯದ ದೃಷ್ಟಿಕೋನ:"),
    ("mr", "माझे उत्तर:", "माझा निष्कर्ष:", "भविष्यातील दृष्टिकोन:"),
    ("ml", "എന്റെ ഉത്തരം:", "എന്റെ നിഗമനം:", "ഭാവി വീക്ഷണം:"),
    ("ne", "मेरो उत्तर:", "मेरो निष्कर्ष:", "भविष्यको दृष्टिकोण:"),
    ("or", "ମୋର ଉତ୍ତର:", "ମୋର ନିଷ୍କର୍ଷ:", "ଭବିଷ୍ୟତ ଦୃଷ୍ଟିକୋଣ:"),
    ("pa", "ਮੇਰਾ ਜਵਾਬ:", "ਮੇਰਾ ਸਿੱਟਾ:", "ਭਵਿੱਖ ਦਾ ਦ੍ਰਿਸ਼ਟੀਕੋਣ:"),
    ("ta", "எனது பதில்:", "எனது முடிவு:", "எதிர்காலக் கண்ணோட்டம்:"),
    ("te", "నా సమాధానం:", "నా ముగింపు:", "భవిష్యత్ దృక్పథం:"),
    ("ur", "میرا جواب:", "میرا نتیجہ:", "مستقبل کا منظرنامہ:"),
]


@pytest.mark.parametrize("language,answer,conclusion,outlook", TEMPLATES)
def test_translated_template_requires_order_and_nonempty_sections(language, answer, conclusion, outlook):
    kwargs = {"_lang": {"fixed": dict(my_answer=answer, my_conclusion=conclusion, future_outlook=outlook)}}
    for separator in (" ", "\n"):
        response = separator.join((f"{answer} १", f"{conclusion} २", f"{outlook} ३"))
        assert evaluate(["format:output_template"], [kwargs], language=language, response=response)[0] == [True]
    for response in (
        f"{conclusion} २ {answer} १ {outlook} ३",
        f"{answer} {conclusion} {outlook}",
        f"{answer} १ {conclusion} {outlook} ३",
        f"{answer} १ {conclusion} २ {outlook}",
        f"{answer} १ {conclusion} २ {outlook} ३ {answer}",
        f"० {answer} १ {conclusion} २ {outlook} ३",
    ):
        assert evaluate(["format:output_template"], [kwargs], language=language, response=response)[0] == [False]


@pytest.mark.parametrize(
    "language,response",
    [
        ("hi", "राशि ₹१,२३,४५६.७८ और २"),
        ("bn", "টাকা ১২,৩৪,৫৬৭.৮৯ এবং ২"),
        ("gu", "૧,૨૩,૪૫૬.૭૮ ૨"),
        ("ur", "۱۲٬۳۴۵٫۶۷ اور ۲"),
        ("hi", "1,234,567.89 2"),
    ],
)
def test_native_digits_and_indian_or_western_grouped_numbers(language, response):
    assert evaluate(["count:numbers"], [{"N": 2}], language=language, response=response)[0] == [True]
    assert evaluate(["count:numbers"], [{"N": 3}], language=language, response=response)[0] == [False]


@pytest.mark.parametrize(
    "language,digits",
    [
        ("bn", "০১২৩৪৫৬৭৮৯"),
        ("gu", "૦૧૨૩૪૫૬૭૮૯"),
        ("hi", "०१२३४५६७८९"),
        ("kn", "೦೧೨೩೪೫೬೭೮೯"),
        ("mr", "०१२३४५६७८९"),
        ("ml", "൦൧൨൩൪൫൬൭൮൯"),
        ("ne", "०१२३४५६७८९"),
        ("or", "୦୧୨୩୪୫୬୭୮୯"),
        ("pa", "੦੧੨੩੪੫੬੭੮੯"),
        ("ta", "௦௧௨௩௪௫௬௭௮௯"),
        ("te", "౦౧౨౩౪౫౬౭౮౯"),
        ("ur", "۰۱۲۳۴۵۶۷۸۹"),
    ],
)
def test_native_digits_preserve_grouping_and_distinguish_list_commas(language, digits):
    table = str.maketrans("0123456789", digits)
    assert evaluate(["count:numbers"], [{"N": 1}], language=language, response="1,23,456".translate(table))[0] == [
        True
    ]
    assert evaluate(["count:numbers"], [{"N": 2}], language=language, response="1,2".translate(table))[0] == [True]
    assert evaluate(["count:numbers"], [{"N": 1}], language=language, response="1,2".translate(table))[0] == [False]


@pytest.mark.parametrize("response", ["۱۸۰۰-۰۱-۰۱، ۱۸۱۲-۰۲-۲۹", "1800-01-01,1812-02-29"])
def test_date_list_accepts_urdu_comma_and_native_digits(response):
    assert evaluate(["custom:date_format_list"], [{}], language="ur", response=response)[0] == [True]
    assert evaluate(["custom:date_format_list"], [{}], language="ur", response=response + "،")[0] == [False]
    assert evaluate(["custom:date_format_list"], [{}], language="ur", response="۱۸۰۰-۰۲-۳۰")[0] == [False]


def test_urdu_capital_list_uses_native_comma_without_changing_required_order():
    capitals = [
        "ریکیاوک",
        "ہیلسنکی",
        "اوسلو",
        "تالین",
        "اسٹاک ہوم",
        "ریگا",
        "ماسکو",
        "کوپن ہیگن",
        "ویلنیس",
        "منسک",
        "ڈبلن",
        "برلن",
        "ایمسٹرڈیم",
        "وارسا",
        "لندن",
        "برسلز",
        "پراگ",
        "لکسمبرگ",
        "پیرس",
        "ویانا",
        "براٹیسلاوا",
        "بوڈاپسٹ",
        "واڈوز",
        "کیشیناؤ",
        "برن",
        "لیوبلیانا",
        "زغرب",
    ]
    kwargs = {"_lang": {"fixed": {"european_capitals": capitals}}}
    assert evaluate(["custom:european_capitals_sort"], [kwargs], language="ur", response="، ".join(capitals))[0] == [
        True
    ]
    assert evaluate(["custom:european_capitals_sort"], [kwargs], language="ur", response="، ".join(capitals[::-1]))[
        0
    ] == [False]


@pytest.mark.parametrize("language,header", [("bn", "বড় শহর"), ("or", "ବଡ଼ ସହର")])
def test_csv_header_accepts_canonical_equivalence_but_requires_csv_delimiter(language, header):
    columns = [header, "ID", "YEAR", "VALUE", "COUNT"]
    kwargs = {"_lang": {"fixed": {"csv_city_cols": columns}}}
    response = unicodedata.normalize("NFD", ",".join(columns)) + "\n" + "1,2,3,4,5\n" * 7
    assert evaluate(["custom:csv_city"], [kwargs], language=language, response=response)[0] == [True]
    assert evaluate(["custom:csv_city"], [kwargs], language=language, response=response.replace(",", "،"))[0] == [
        False
    ]
    assert evaluate(["custom:csv_city"], [kwargs], language=language, response=response.replace("ID", "id"))[0] == [
        False
    ]


def test_distinct_person_names_deduplicate_canonically_equivalent_spellings():
    kwargs = {"N": 2, "_lang": {"names": ["क़ासिम", "क़ासिम", "राम"]}}
    assert evaluate(["count:person_names"], [kwargs], language="hi", response="क़ासिम")[0] == [False]
    assert evaluate(["count:person_names"], [kwargs], language="hi", response="क़ासिम राम")[0] == [True]
    for name in ("राम\u200dनाथ", "राम\u200cनाथ", "राम'नाथ", "राम’नाथ"):
        assert evaluate(["count:person_names"], [kwargs], language="hi", response="क़ासिम " + name)[0] == [False]
    assert evaluate(["count:person_names"], [kwargs], language="hi", response="'क़ासिम' 'राम'")[0] == [True]


@pytest.mark.parametrize(
    "language,phrase,positive,negative",
    [
        ("bn", "টাক ঈগল", "ঈগল টাক", "ঈগলকে টাক"),
        ("or", "ବାଲ୍ଡ ଇଗଲ୍", "ଇଗଲ୍ ବାଲ୍ଡ", "ଇଗଲ୍କୁ ବାଲ୍ଡ"),
        ("hi", "बड़ा ईगल", "ईगल बड़ा", "ईगलको बड़ा"),
    ],
)
def test_word_reverse_compares_complete_normalized_words(language, phrase, positive, negative):
    kwargs = {"_lang": {"fixed": {"bald_eagle": phrase}}}
    assert evaluate(["custom:word_reverse"], [kwargs], language=language, response=positive)[0] == [True]
    assert evaluate(["custom:word_reverse"], [kwargs], language=language, response=negative)[0] == [False]


@pytest.mark.parametrize(
    "language,positive,negative", [("kn", "ಹಳ್ಳಿ", "ಹಲಿ"), ("ml", "പള്ളി", "പലി"), ("te", "పళ్ళు", "పలు")]
)
def test_conjunct_checks_include_native_consonants_outside_alphabet_cycle(language, positive, negative):
    assert evaluate(["words:consonants"], [{}], language=language, response=positive)[0] == [True]
    assert evaluate(["words:consonants"], [{}], language=language, response=negative)[0] == [False]


@pytest.mark.parametrize(
    "language,response",
    [("hi", "“राम है।” “कौन है?” “वाह!”"), ("ur", '"وہ ہے۔" "کیا؟" "واہ!"')],
)
def test_sentence_balance_recognizes_terminators_before_closing_quotes(language, response):
    assert evaluate(["ratio:sentence_balance"], [{}], language=language, response=response)[0] == [True]
    assert evaluate(["ratio:sentence_balance"], [{}], language=language, response=response.replace("!", "."))[0] == [
        False
    ]


def test_urdu_punctuation_counts_native_comma_semicolon_question_and_full_stop():
    assert evaluate(["count:punctuation"], [{}], language="ur", response="۔ ، ! ؟ ؛ : ؟!")[0] == [True]
    assert evaluate(["count:punctuation"], [{}], language="ur", response="۔ ، ! ؟ : ؟!")[0] == [False]


def test_nested_quotes_still_require_explicit_ascii_characters_from_source_prompt():
    response = '"अ \'आ "इ" आ\' अ"'
    assert evaluate(["format:quotes"], [{}], language="hi", response=response)[0] == [True]
    assert evaluate(["format:quotes"], [{}], language="hi", response="“अ ‘आ “इ” आ’ अ”")[0] == [False]


@pytest.mark.parametrize(
    "language,quoted,explanation",
    [
        ("hi", "‘राम’", "एक नाम है।"),
        ("bn", "«ভাষা»", "একটি শব্দ।"),
        ("ta", "‹மொழி›", "ஒரு சொல்."),
        ("ur", "“زبان”", "ایک لفظ ہے۔"),
        ("gu", "'ભાષા'", "એક શબ્દ છે."),
        ("te", '"భాష"', "ఒక పదం."),
    ],
)
def test_quoted_phrases_accept_paired_native_typographic_marks(language, quoted, explanation):
    assert evaluate(["format:quote_unquote"], [{}], language=language, response=f"{quoted} {explanation}")[0] == [True]
    assert evaluate(["format:quote_unquote"], [{}], language=language, response=quoted)[0] == [False]
    assert evaluate(["format:quote_unquote"], [{}], language=language, response=f"{quoted} {quoted} {explanation}")[
        0
    ] == [False]


@pytest.mark.parametrize("response", ["“राम» एक नाम।", "‘राम” एक नाम।", "«राम एक नाम।", "राम» एक नाम।", "‘ ’ एक नाम।"])
def test_quoted_phrase_rejects_mismatched_unclosed_and_empty_pairs(response):
    assert evaluate(["format:quote_unquote"], [{}], language="hi", response=response)[0] == [False]


def test_internal_apostrophes_do_not_become_quotation_delimiters():
    for response in ("‘ਤੇ’ ਉਹਦਾ'ਈ ਰੂਪ ਹੈ।", "‘ਤੇ’ ਉਹਦਾ’ਈ ਰੂਪ ਹੈ।"):
        assert evaluate(["format:quote_unquote"], [{}], language="pa", response=response)[0] == [True]
    assert evaluate(["format:quote_unquote"], [{}], language="pa", response="ਉਹਦਾ'ਈ ਰੂਪ ਹੈ।")[0] == [False]


def test_nested_quoted_phrases_share_an_explanation_outside_the_enclosing_passage():
    response = "“राम ‘नाम’ है” इसका स्पष्टीकरण।"
    assert evaluate(["format:quote_unquote"], [{}], language="hi", response=response)[0] == [True]
    assert evaluate(["format:quote_unquote"], [{}], language="hi", response="“राम ‘नाम’ है”")[0] == [False]


def test_increment_does_not_add_an_unspecified_minimum_sentence_count():
    assert evaluate(["sentence:increment"], [{"small_n": 1}], language="hi", response="राम।")[0] == [True]
    assert evaluate(["sentence:increment"], [{"small_n": 1}], language="hi", response="राम। सीता राम।")[0] == [True]
    assert evaluate(["sentence:increment"], [{"small_n": 1}], language="hi", response="राम। सीता।")[0] == [False]
