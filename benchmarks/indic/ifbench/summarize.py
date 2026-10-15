# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Summarize Indic IFBench rollout JSONL using the server's metric definitions."""

import argparse
import json
from pathlib import Path

from resources_servers.indic_ifbench.metrics import summarize


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rollouts", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    with args.rollouts.open(encoding="utf-8") as stream:
        result = summarize(json.loads(line) for line in stream if line.strip())
    text = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
