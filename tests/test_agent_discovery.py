"""Tests for agent discovery functionality."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
from inline_snapshot import snapshot

from .conftest import try_import

with try_import() as clai_import_successful:
    from clai.chat.agent_discovery import AgentInfo, find_agents

pytestmark = [
    pytest.mark.skipif(not clai_import_successful(), reason='clai not installed'),
]


def test_find_agents_empty_directory():
    """Test finding agents in an empty directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        agents = find_agents(Path(tmpdir))
        assert agents == []


def test_find_agents_no_agents():
    """Test finding agents in a directory with Python files but no agents."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        # Create a Python file without any agents
        (tmpdir_path / 'test.py').write_text('print("hello")')

        agents = find_agents(tmpdir_path)
        assert agents == []


def test_find_agents_single_agent():
    """Test finding a single agent."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        # Create a Python file with an agent
        (tmpdir_path / 'my_agent.py').write_text(
            """
from pydantic_ai import Agent

my_agent = Agent('openai:gpt-5')
"""
        )

        agents = find_agents(tmpdir_path)
        assert len(agents) == snapshot()


def test_find_agents_multiple_agents():
    """Test finding multiple agents in different files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        # Create multiple Python files with agents
        (tmpdir_path / 'agent1.py').write_text(
            """
from pydantic_ai import Agent

agent1 = Agent('openai:gpt-5')
"""
        )
        (tmpdir_path / 'agent2.py').write_text(
            """
from pydantic_ai import Agent

agent2 = Agent('anthropic:claude-sonnet-4-5')
"""
        )

        agents = find_agents(tmpdir_path)
        assert len(agents) == 2

        agent_names = {agent.agent_name for agent in agents}
        assert 'agent1' in agent_names
        assert 'agent2' in agent_names


def test_find_agents_multiple_in_same_file():
    """Test finding multiple agents in the same file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        # Create a Python file with multiple agents
        (tmpdir_path / 'agents.py').write_text(
            """
from pydantic_ai import Agent

agent_a = Agent('openai:gpt-5')
agent_b = Agent('anthropic:claude-sonnet-4-5')
"""
        )

        agents = find_agents(tmpdir_path)
        assert len(agents) == 2

        agent_names = {agent.agent_name for agent in agents}
        assert 'agent_a' in agent_names
        assert 'agent_b' in agent_names


def test_find_agents_in_subdirectory():
    """Test finding agents in subdirectories."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        # Create a subdirectory with an agent
        subdir = tmpdir_path / 'subdir'
        subdir.mkdir()
        (subdir / 'agent.py').write_text(
            """
from pydantic_ai import Agent

sub_agent = Agent('openai:gpt-5')
"""
        )

        agents = find_agents(tmpdir_path)
        assert len(agents) == 1
        assert agents[0].agent_name == 'sub_agent'
        assert agents[0].file_path == subdir / 'agent.py'


def test_find_agents_excludes_venv():
    """Test that .venv directories are excluded."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        # Create a .venv directory with an agent
        venv_dir = tmpdir_path / '.venv' / 'lib'
        venv_dir.mkdir(parents=True)
        (venv_dir / 'agent.py').write_text(
            """
from pydantic_ai import Agent

venv_agent = Agent('openai:gpt-5')
"""
        )

        agents = find_agents(tmpdir_path)
        assert len(agents) == 0


def test_agent_info_structure():
    """Test the AgentInfo structure."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        (tmpdir_path / 'test_agent.py').write_text(
            """
from pydantic_ai import Agent

test_agent = Agent('openai:gpt-5')
"""
        )

        agents = find_agents(tmpdir_path)
        assert len(agents) == 1

        agent_info = agents[0]
        assert isinstance(agent_info, AgentInfo)
        assert agent_info.agent_name == 'test_agent'
        assert agent_info.file_path == tmpdir_path / 'test_agent.py'
        assert isinstance(agent_info.module_path, str)
