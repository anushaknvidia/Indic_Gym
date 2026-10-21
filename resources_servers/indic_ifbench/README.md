# Indic IFBench verifier

Native NeMo Gym evaluator for the translated AI4Bharat IFBench files. Supports
`bn gu hi kn mr ml ne or pa ta te ur`; aliases `ka`, `mar`, `mal` resolve to
`kn`, `mr`, `ml`. The benchmark is [indic/ifbench](../../benchmarks/indic/ifbench/README.md).
No English spaCy model, NLTK corpus, runtime download, or model judge is used.

## Contract and scoring

Tasks carry `id`, `language`, `prompt`, parallel `instruction_id_list` and
`kwargs` arrays, and optional provenance (`source_key`, `uuid`, `metadata`).
`task_data.py` validates the same contract used by `/verify`. Preparation rejects
unknown constraints, missing arguments, invalid spans, and missing translations
instead of generating default arguments. Integral source floats become integers;
null padding in kwargs is removed; `_lang.names` and `_lang.fixed` are preserved.

All 54 instruction types present in the supplied dataset have implementations in
`evaluation.py`. Urdu has 50 types: its source omits alphabet, vowel, conjunct,
and syllable constraints. Those four are rejected for Urdu rather than scored
using English rules. Empty output fails every instruction. The verifier checks
the registered constraints, not factual correctness, task relevance, or refusal
semantics. A nonempty refusal can satisfy some formal constraints. Separate
reasoning items are ignored; tagged `<think>`/`<thinking>` text is removed from
the final assistant message. All final-message text parts are combined and raw
whitespace is retained for strict checks.

`reward` defaults to the fraction of strict instruction checks passed, matching
Gym's existing English IFBench configuration. `grading_mode: binary` instead
requires every instruction to pass. The response also contains strict/loose
per-instruction decisions and prompt-level decisions, `reward_loose`, language,
provenance, and `evaluator_version: indic-ifbench-v2`. Loose scoring uses the
same eight candidates as Gym's English IFBench: original text, removed `*`,
removed first/last/both lines, and combinations of those transformations.

`compute_metrics` reports `prompt_strict`, `prompt_loose`, `instruction_strict`,
`instruction_loose`, per-language scores, and an equal-language macro average.
Instruction accuracy divides by the total number of instructions, not the number
of prompts. Repeats count as separate attempts. Verifier defects are logged and
returned as `mask_sample: true` with `failure_kind: verifier_error`; they are
excluded from accuracy. Invalid task metadata is rejected as a request error.

## Version 2 language policy

The source distribution contains no reference Indic evaluator. This implementation
is a new, experimental scoring policy, not a reproduction of published Indic
scores. Its local language inventories need native-speaker review before baselining;
`verified: false` is intentional. In particular, lexical and syllable scores are
sensitive to these choices. See the [language audit](language-audit.md) for
confirmed fixes, source-data concerns, and unresolved scoring policies.

Version 2 changes scoring. Reverify saved model responses with this evaluator
before comparing or aggregating them; the summarizer rejects version 1 records.
Rerun preparation to refresh the evaluator version in the preparation audit.
Source prompts and their supplied `kwargs` remain authoritative and unchanged.

- **Normalization:** NFC and Unicode case folding; vowel marks are preserved.
  Words are Unicode letter/number runs, including attached combining marks and
  ZWJ/ZWNJ. Internal straight/curly apostrophes remain part of a word; other
  punctuation separates tokens. Span copying alone uses the explicitly requested
  whitespace-separated indices.
- **Characters:** grapheme clusters, joining virama-linked consonants (including
  ZWJ) into one Indic character unit while respecting explicit ZWNJ boundaries.
  Tamil normally keeps pulli-bearing
  consonants separate, with explicit `க்ஷ`, `ஶ்ரீ`, and `ஸ்ரீ` conjunct exceptions.
  Palindromes, prime word lengths,
  equal-character sentences, MCQ lengths, and character reversal use these units.
- **Sentences:** `. ! ? । ॥ ۔ ؟`, retaining terminators and decimal points.
  Closing quotes and brackets stay with the preceding sentence. Danda and Urdu
  full stop count as declarative; Urdu question mark is supported.
  Bidirectional formatting controls are ignored for sentence segmentation only.
  This is punctuation segmentation, without abbreviation or syntactic analysis.
- **Quoted phrases:** the quote/explanation check accepts paired ASCII quotes,
  curly quotes, and guillemets, with nonempty quoted text and an explanation
  outside each passage. Nested quotes share their outer passage's explanation.
  The separate nested-quote task keeps the ASCII characters explicitly requested
  in its source prompts.
- **Pronouns, conjunctions, stop words:** exact matches against explicit local
  inventories in `languages.py`. No stemming, POS tagging, or English fallback.
  Stop words include the pronoun and conjunction inventories. These lists are
  curated benchmark rules, not exhaustive linguistic dictionaries.
- **Alphabet:** the explicit sequence for each language in `languages.py`;
  initial base letters advance cyclically from any starting position. The
  translated prompts specify wraparound from ha to a, including for Punjabi
  and Tamil; these sequences should not be treated as universal dictionary order.
- **Conjuncts:** consonant + optional marks + virama + optional joiner + consonant.
  Consonant recognition uses the script's Unicode consonant inventory separately
  from the benchmark's alphabet-order sequence. This checks a logical consonant
  sequence, including ZWJ/ZWNJ spellings, rather than a rendered ligature; character
  counting separately respects ZWNJ boundaries.
- **Vowels:** independent vowels, dependent vowel signs, and the inherent `A`
  of a consonant without a sign or terminal virama. Latin vowels are recognized
  in mixed-script text. Marathi `ॲ` and modern Malayalam AU `ൗ` are recognized;
  Malayalam chillus have no inherent vowel. At most three distinct types are
  permitted in one line.
- **Syllables:** counts of written vowel nuclei. This is an **orthographic
  approximation**, without phonetic schwa deletion, pronunciation dictionaries,
  or sandhi analysis. It must not be interpreted as a phonetic syllable judgment.
- **Fixed strings:** names, CSV headers, repeat sentence, template labels,
  options separator, MCQ label, reversed animal name, and ordered capitals come
  from each row's `_lang`, never English constants. Missing metadata is an error.
  Names and reversed-word phrases match complete normalized word sequences.
  CSV headers allow canonical Unicode equivalence while retaining exact case and
  CSV delimiters. Templates require the supplied headings once each, in order,
  with nonempty sections and no preface.
- **Span indices:** zero based, inclusive, whitespace word indices, as specified
  by the translated prompts. Index zero is preserved and no random defaults run.
- **Numbers:** Unicode decimal digits; dot/Arabic decimal marks, Western thousands
  grouping, and Indian lakh/crore grouping stay within one number. Comma-separated
  short numbers count separately. Date/capital lists accept both `,` and Urdu `،`;
  this does not change CSV syntax. Dates must be real Gregorian dates in 1769–1821.

The language-neutral constraints are adapted from AllenAI IFBench commit
`c6767a19bd82ac0536cab950f2f8f6bcc6fabe7c` (Apache-2.0). Some upstream permissive
checks contradict the translated instructions. The evaluator requires balanced and
closed nesting, a nonzero sentence-type ratio, a Japanese-script target token,
at least one bullet and each required sub-bullet, line-initial list markers,
nonrepeated words for equal-character sentences, explanations after quoted text,
and fully valid CSV rows. Alliteration compares the **longest consecutive run**.
These intentional changes mean scores should not be compared directly with the
English server. Change `EVALUATOR_VERSION` when changing scoring semantics and
reverify existing rollouts before comparing results.

## Tests

```bash
gym env test --resources-server indic_ifbench
pytest resources_servers/indic_ifbench/tests tests/unit_tests/test_indic_ifbench.py
```

Tests include positive/negative examples for all 54 check types, all 12 languages,
concurrent language isolation, Unicode marks and joiners, translated constants,
HTTP verification, masked failures, preparation, native configuration/schema
validation, and weighted metrics.
