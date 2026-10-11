# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Execute one candidate using the original HumanEval checker in a worker process."""

import json
import math
import os
import resource
import sys
import traceback

from human_eval.execution import check_correctness


def apply_resource_limits(*, timeout: float, memory_limit_mb: int) -> None:
    """Bound common runaway-code failure modes before the checker forks."""
    memory_bytes = memory_limit_mb * 1024 * 1024
    resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))
    cpu_seconds = max(1, math.ceil(timeout) + 1)
    resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))
    resource.setrlimit(resource.RLIMIT_FSIZE, (16 * 1024 * 1024, 16 * 1024 * 1024))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


def main() -> None:
    """Read a candidate on stdin and emit the original checker's verdict on stdout."""
    payload = json.load(sys.stdin)
    apply_resource_limits(timeout=payload["timeout"], memory_limit_mb=payload["memory_limit_mb"])

    # Keep the verdict channel away from fd 1 before candidate code runs. Python's
    # redirect_stdout does not intercept os.write(1, ...), and communicate() would
    # otherwise buffer arbitrary candidate output in the resources-server process.
    result_fd = os.dup(1)
    diagnostics_fd = os.dup(2)
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull_fd, 1)
    os.dup2(devnull_fd, 2)
    os.close(devnull_fd)

    # HF code_eval (used by the reference) executes candidate + test + check(entry_point).
    # The upstream checker does the same when its completion prefix is empty.
    try:
        result = check_correctness(
            problem={
                "task_id": payload["task_id"],
                "prompt": "",
                "test": payload["test"],
                "entry_point": payload["entry_point"],
            },
            completion=payload["code"],
            timeout=payload["timeout"],
        )
        os.write(result_fd, json.dumps(result).encode() + b"\n")
    except BaseException:  # pragma: no cover - defensive process-boundary diagnostics
        os.write(diagnostics_fd, traceback.format_exc().encode("utf-8", "replace"))
        raise
    finally:
        os.close(result_fd)
        os.close(diagnostics_fd)


if __name__ == "__main__":
    main()
