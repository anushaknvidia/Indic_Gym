# Indic IFBench language audit

This audit covers the supplied `translation/{bn,gu,hi,kn,mr,ml,ne,or,pa,ta,te,ur}/train.jsonl`
files and the native Gym evaluator. It is a review of deterministic checks and
source consistency, not native-speaker certification of every translation.
The benchmark remains `verified: false`.

## Dataset contract

All 3,420 source tasks supply their own `kwargs`. The evaluator preserves these
values, including translated keywords, `_lang.names`, and `_lang.fixed`.
Preparation removes null padding and converts integral floats to integers.
There are 288 tasks in each of eleven languages and 252 in Urdu. Urdu's 36 absent
tasks contain alphabet, vowel, conjunct, or syllable constraints; no replacement
tasks are invented.

An audit of all positional keywords found no multiword values incompatible with
the current single-token positional check. Person-name inventories contain enough
distinct names for their requested counts. Overlap reference texts occur in their
source prompts. These checks do not establish translation quality.

## Version 2 corrections

| Area | Correction |
| --- | --- |
| Sentence punctuation | Closing quotes and brackets remain attached to the preceding sentence; sentence ratios and emoji checks use the resulting boundaries. Danda, double danda, Urdu full stop, and Urdu question mark remain supported. |
| Quoted phrases | The quote/explanation checker accepts paired ASCII quotes, curly quotes, and guillemets; mismatched/empty quotes and missing explanations fail. The distinct ASCII nested-quote instruction retains its specified syntax. |
| Formatting controls | Sentence segmentation ignores bidirectional formatting controls, including trailing Urdu RLM. Character segmentation respects ZWNJ boundaries instead of joining consonants across a nonjoiner. Raw exact-output and whitespace checks still receive the original response. |
| Native consonants | Separate Unicode consonant inventories cover letters omitted from the benchmark alphabet cycles, including Gujarati `ળ`, Kannada `ಳ`, Malayalam `ള ഴ റ`, Odia `ଳ`, and Telugu `ళ`. This affects conjunct and written-vowel checks. |
| Vowels | Recognize Marathi `ॲ` and modern Malayalam AU `ൗ`. Atomic and legacy terminal Malayalam chillus agree on written-vowel counts. |
| Tamil characters | Keep ordinary pulli-bearing letters separate; tailor `க்ஷ`, `ஶ்ரீ`, and `ஸ்ரீ` as conjunct exceptions. For example, `தீர்ப்பு` has units `தீ`, `ர்`, `ப்`, `பு`. |
| Word lists | Add Hindi `इन्हें इनका इनकी इनके`, Urdu pronoun `ان` and conjunction `و`, and Odia `କ'ଣ` / `କ’ଣ`. Stop-word sets include these additions. Normalize inventories consistently with response tokens. |
| Apostrophes | Internal straight/curly apostrophes remain inside Odia words such as `କ'ଣ`; surrounding quote characters still delimit words. |
| Numbers and lists | Recognize Indian lakh/crore grouping with native digits. Accept Urdu comma `،` for ordinary date/capital lists while keeping CSV delimiters exact. |
| Fixed metadata | Canonically equivalent CSV headers compare equal. Names use complete word sequences and cannot count a canonical spelling variant twice. Reversed-word phrases also match complete word sequences. |
| Templates | Require the supplied headings once each, in order, with nonempty sections and no preface. The previous checker accepted reordered or empty templates. |

The [language policy](README.md#version-2-language-policy) defines the resulting
behavior. Regression tests include native-script positive and negative cases,
and translated template labels for all twelve languages.

Consonant recognition uses [Unicode Indic character properties](https://www.unicode.org/reports/tr44/#Indic_Syllabic_Category).
Vowel/chillu handling follows [Unicode's Indic script descriptions](https://www.unicode.org/versions/Unicode17.0.0/core-spec/chapter-12/).
Tamil tailoring uses examples from the [W3C Tamil layout draft](https://www.w3.org/TR/2020/WD-ilreq-taml-20200616/#grapheme-boundaries);
the draft is supporting guidance, not linguistic certification. Hindi/Urdu
additions are supported by the grammatical summaries for
[Hindi HDTB](https://universaldependencies.org/treebanks/hi_hdtb/index.html) and
[Urdu UDTB](https://universaldependencies.org/treebanks/ur_udtb/index.html).

Scoring changes are identified as `indic-ifbench-v2`. Reverify saved responses
before comparing them with new runs. The metric summarizer rejects version 1
records, including mixtures of version 1 and version 2. Rerunning preparation
refreshes its audit version without editing the source dataset.

## Source rows requiring attention

| Source key | Languages | Finding and consequence |
| --- | --- | --- |
| `208` | All eleven non-Urdu sets | The prompt asks for a rewrite in modern English, then requires every word to contain an Indic consonant + virama + consonant. Ordinary English output cannot satisfy the script constraint. Review the translation/adaptation before interpreting failures as model errors. |
| `278` | All twelve | The supplied 27-item European-capitals reference omits Kyiv despite asking for all European capitals north of 45 degrees. The UNGEGN entry places Kyiv above 50 degrees north. An otherwise correct response including Kyiv fails exact-reference scoring. Correcting this requires a versioned source-reference change, including translated names. |
| `287` | Marathi, Malayalam, Telugu | The prompt names indices `10` and `8`, while `kwargs` specifies the ascending inclusive interval `8..10`. The endpoints agree, but the descending wording is confusing. The evaluator follows the supplied ascending interval. |
| `200`–`205` | Especially Punjabi and Tamil | The translated prompts prescribe a cycle from ha back to a. The explicit benchmark sequence is not a general language collation standard. The complete intended alphabet and its order need dataset-author/native-speaker agreement. |

Kyiv evidence: [United Nations Group of Experts on Geographical Names, Kyiv](https://ungegn.un.org/dashboard/cities/details?id=172).
The supplied source records are retained; the evaluator does not silently rewrite
these prompts, remove rows, or alter their reference answers.

## Remaining scoring limitations

| Area | What needs review |
| --- | --- |
| Pronouns, conjunctions, stop words | Exact lexical inventories cannot determine grammatical role from context, recognize arbitrary inflections, or handle every multiword expression. Urdu `میں`, for example, can represent a pronoun or a postposition depending on context. An exhaustive POS claim would require a separate validated language analysis policy. |
| Syllables and vowels | Written vowel nuclei are an orthographic approximation. Pronunciation, schwa deletion, dialect, sandhi, and nasalization are not modeled. Syllable parity can therefore disagree with speech even when Unicode processing is correct. |
| Alphabet order | The prompt gives wraparound endpoints, not a complete inventory. Rare/borrowed letters and conjuncts require a policy; Unicode consonant membership does not determine alphabetical order. |
| Conjunct interpretation | The checker recognizes a logical consonant–virama–consonant sequence with optional ZWJ/ZWNJ. It does not verify that a font renders a ligature. Character counting separately honors ZWNJ breaks; a requirement for visually joined letters would need a different agreed policy. |
| Word boundaries | Punctuation and internal apostrophe handling are explicit tokenization rules. Compounds, clitics, joiners, hyphens, and agglutination can make linguistic word counts differ. Span-copy indices intentionally use whitespace splitting instead. |
| Keyword occurrences | Frequency and sentence-keyword checks retain the original substring policy. A keyword inside an inflected or longer word may count. Whole-token matching would be a separate semantic change that needs agreement for agglutinative languages. |
| Names and fixed answers | Matching is limited to the supplied inventory/reference. Valid alternative transliterations and synonyms can fail, especially the animal-name and capitals tasks. Canonical Unicode equivalence is narrower than transliteration equivalence. |
| Malayalam legacy encoding | Atomic chillus and older consonant + virama + ZWJ forms are not NFC-equivalent. Terminal written-vowel counts agree, but full lexical and character-count equivalence is not implemented. |
| Sentence segmentation | Terminators and closing quotes/brackets are handled; abbreviations, initials, ellipses, and quoted speech still need linguistic interpretation that this punctuation-based segmenter does not provide. |
| Numeric commas | A compact value such as `1,234` is interpreted as a grouped number. Use a space after a comma when intending a list of separate numbers. This ambiguity cannot be resolved from characters alone. |
| Progression constraints | Adjacent-word/sentence/line checks can pass with one item when the task's verifiable constraint states no minimum length. The evaluator does not invent an additional minimum. |
| Script and meaning | There is no general language-identification, factuality, relevance, or refusal judge. Some foreign-script or irrelevant responses can pass formal constraints. Japanese-token detection recognizes script membership, not dictionary validity. |

These issues should be considered when comparing languages. Urdu also has a
different instruction mix, so equal-language macro averaging alone does not make
the twelve task sets equivalent. A comparison restricted to shared source keys
would answer a different, more controlled question and must be labeled separately.

Evidence for the linguistic limitations includes research on
[Hindi and Punjabi schwa deletion](https://aclanthology.org/2020.acl-main.696/) and
[Urdu word segmentation](https://aclanthology.org/2020.winlp-1.41/), plus the
[Universal Dependencies Malayalam grammar](https://universaldependencies.org/ml/index.html).
The Malayalam coordinating suffix `-ഉം`, for example, cannot be recognized by
adding a standalone token to an exact-word conjunction list. Some translated
prompts specifically ask for coordinating conjunctions, while the current
inventories also include other connective words; native grammatical review is
needed before claiming that distinction is verified.

## Validation performed

- 256 focused tests passed, including 141 new regression cases. This includes
  HTTP `/verify`, metrics, preparation/configuration, and the language checkers.
- Evaluator statement coverage was 97.61%, exceeding the repository's 95% gate.
- All 3,420 supplied source rows passed the task schema. An additional 516
  positive checks of individual instructions, constructed from the actual source
  arguments (43 per language), passed. These are deterministic checker probes,
  not model-generated answers or a guarantee that combined prompts are satisfiable.
- Preparation regenerated the 3,420 tasks and refreshed the version 2 audit.
- Scoped pre-commit, Ruff, formatting, and `git diff --check` passed. The private
  guardwords catalog was unavailable, so that hook's content scan was skipped.
- A real-model rollout/baseline is pending an inference endpoint and model name.
  `verified: false` remains appropriate until model and linguistic review.

Run the focused tests from the repository root with:

```bash
pytest resources_servers/indic_ifbench/tests tests/unit_tests/test_indic_ifbench.py
```
