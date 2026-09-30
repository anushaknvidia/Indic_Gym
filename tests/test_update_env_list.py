# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from pathlib import Path

import pytest
import scripts.update_env_list as update_env_list


@pytest.mark.parametrize(
    "benchmark_name, use_agent, expected_display",
    [
        ("desktop_bench", True, "Desktop Agent"),
        ("indic/biggenbench", False, "Indic / Biggenbench"),
        ("example_suite/nested", False, None),
        ("suite/example_demo", False, None),
    ],
)
def test_training_server_info_includes_benchmark_configs(
    tmp_path: Path, monkeypatch, benchmark_name: str, use_agent: bool, expected_display: str | None
) -> None:
    resources_servers = tmp_path / "resources_servers"
    responses_api_agents = tmp_path / "responses_api_agents"
    benchmarks = tmp_path / "benchmarks"
    resources_servers.mkdir()
    responses_api_agents.mkdir()
    benchmark_dir = benchmarks / benchmark_name
    benchmark_dir.mkdir(parents=True)
    agent_dir = responses_api_agents / "desktop_agent" / "configs"
    agent_dir.mkdir(parents=True)
    agent_config = agent_dir / "desktop_agent.yaml"
    agent_config.write_text(
        """
desktop_agent:
  responses_api_agents:
    desktop_agent:
      domain: agent
      description: Desktop benchmark
      value: GUI evaluation
""".lstrip(),
        encoding="utf-8",
    )
    (benchmark_dir / "config.yaml").write_text(
        f"""
config_paths: [{agent_config if use_agent else ""}]

desktop_agent:
  responses_api_agents:
    desktop_agent:
      datasets:
        - name: validation
          type: validation
          license: Apache 2.0
          huggingface_identifier:
            repo_id: example/desktop-bench
""".lstrip(),
        encoding="utf-8",
    )

    monkeypatch.setattr(update_env_list, "RESOURCES_SERVERS_FOLDER", resources_servers)
    monkeypatch.setattr(update_env_list, "RESPONSES_API_AGENTS_FOLDER", responses_api_agents)
    monkeypatch.setattr(update_env_list, "BENCHMARKS_FOLDER", benchmarks)

    servers = update_env_list.get_training_server_info()

    if expected_display is None:
        assert servers == []
        return

    assert len(servers) == 1
    server = servers[0]
    assert server.name == benchmark_name
    assert server.display_name == expected_display
    assert server.config_path == f"benchmarks/{benchmark_name}/config.yaml"
    assert server.readme_path == f"benchmarks/{benchmark_name}/README.md"
    assert server.config_metadata.domain == ("agent" if use_agent else None)
    assert server.config_metadata.description == ("Desktop benchmark" if use_agent else None)
    assert server.config_metadata.value == ("GUI evaluation" if use_agent else None)
    assert server.config_metadata.types == ["validation"]
