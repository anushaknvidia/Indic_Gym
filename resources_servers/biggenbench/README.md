# BigGenBench rubric verifier

Absolute 1–5 ratings with an instance-specific rubric and reference answer;
Gym reward is `(score - 1) / 4`. Uses Gym's judge client, model-server concurrency,
failure routing, and aggregation. Invalid verdicts are masked from quality metrics.

See [Indic BigGenBench](../../benchmarks/indic/biggenbench/README.md) for dataset
preparation, language aliases, scoring, and native Gym evaluation commands.
