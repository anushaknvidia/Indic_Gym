# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Service-free construction for the HumanEval verifier fixture."""

from unittest.mock import MagicMock

from nemo_gym.server_utils import ServerClient


def create_humaneval_reference_server():
    """Build a bounded verifier without starting HTTP services."""
    from resources_servers.humaneval_reference.app import (
        HumanEvalResourcesServer,
        HumanEvalResourcesServerConfig,
    )

    return HumanEvalResourcesServer(
        config=HumanEvalResourcesServerConfig(
            name="humaneval_reference",
            host="127.0.0.1",
            port=8000,
            entrypoint="app.py",
            num_processes=1,
            timeout=0.2,
            memory_limit_mb=1024,
        ),
        server_client=MagicMock(spec=ServerClient),
    )
