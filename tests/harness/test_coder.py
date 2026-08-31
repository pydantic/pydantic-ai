import subprocess
import sys
from pathlib import Path

import pytest
from pydantic_ai import Agent
from pydantic_ai.capabilities import Capability
from pydantic_ai.models.test import TestModel

import pydantic_ai_harness.coder
from pydantic_ai_harness.coder import DEFAULT_ALLOWED_COMMANDS, Coder, coder_agent
from pydantic_ai_harness.compaction import ClearToolResults, WarnNearLimits
from pydantic_ai_harness.filesystem import FileSystem
from pydantic_ai_harness.planning import Planning
from pydantic_ai_harness.repo_context import RepoContext
from pydantic_ai_harness.shell import LLM_API_KEY_ENV_PATTERNS, Shell
from pydantic_ai_harness.subagents import SubAgents
from pydantic_ai_harness.tool_output_limits import ToolOutputLimits


def test_coder_constructs_agent() -> None:
    agent = Agent(TestModel(), capabilities=[Coder()])

    assert isinstance(agent, Agent)


def test_coder_agent_is_model_less_and_composed() -> None:
    assert isinstance(coder_agent, Agent)
    assert coder_agent.model is None
    assert coder_agent.name == 'coder'
    # Pydantic AI attributes each instruction to the source that contributed it. Assert the base prompt
    # is still the agent's own rather than dropping the check: `'agent'` is the key an application
    # overriding this prompt addresses it by, and it would go silently missing if attribution changed.
    assert [(instruction.instruction, str(instruction.id)) for instruction in coder_agent._instructions] == [
        ('You are a coding agent built on Pydantic AI.', 'agent')
    ]
    assert any(isinstance(capability, FileSystem) for capability in coder_agent.root_capability.capabilities)


def test_coder_agent_export_is_lazy() -> None:
    result = subprocess.run(
        [
            sys.executable,
            '-c',
            'import sys; import pydantic_ai_harness.coder; '
            "assert 'pydantic_ai_harness.coder._agent' not in sys.modules",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_coder_unknown_export() -> None:

    with pytest.raises(AttributeError, match='has no attribute'):
        pydantic_ai_harness.coder.__getattr__('missing')


def test_coder_members_are_transparent() -> None:
    coder = Coder()

    assert [type(capability) for capability in coder.capabilities] == [
        FileSystem,
        Shell,
        RepoContext,
        Planning,
        SubAgents,
        ClearToolResults,
        WarnNearLimits,
        ToolOutputLimits,
    ]
    assert not any(isinstance(capability, Capability) for capability in coder.capabilities)


def test_coder_threads_parameters(tmp_path: Path) -> None:
    coder = Coder(tmp_path, allowed_commands=['git'], subagents=[], instructions='Custom instructions')

    filesystem = next(capability for capability in coder.capabilities if isinstance(capability, FileSystem))
    shell = next(capability for capability in coder.capabilities if isinstance(capability, Shell))
    repo_context = next(capability for capability in coder.capabilities if isinstance(capability, RepoContext))
    assert filesystem.root_dir == tmp_path
    assert shell.cwd == tmp_path
    assert shell.allowed_commands == ['git']
    assert shell.denied_env_patterns == LLM_API_KEY_ENV_PATTERNS
    assert repo_context.workspace_dir == tmp_path
    assert not any(isinstance(capability, SubAgents) for capability in coder.capabilities)
    assert coder.capabilities[0].get_instructions() == ['Custom instructions']


def test_coder_explorer_uses_read_only_capabilities(tmp_path: Path) -> None:
    coder = Coder(tmp_path)
    subagents = next(capability for capability in coder.capabilities if isinstance(capability, SubAgents))
    explorer = subagents.agents[0].agent

    assert explorer.name == 'explorer'
    filesystem = next(
        capability for capability in explorer.root_capability.capabilities if isinstance(capability, FileSystem)
    )
    repo_context = next(
        capability for capability in explorer.root_capability.capabilities if isinstance(capability, RepoContext)
    )
    assert filesystem.read_only is True
    assert repo_context.workspace_dir == tmp_path


def test_coder_instructions_adds_capability() -> None:
    coder = Coder(instructions='Custom instructions')

    instructions = [capability for capability in coder.capabilities if isinstance(capability, Capability)]
    assert len(instructions) == 1
    assert instructions[0].get_instructions() == ['Custom instructions']


def test_coder_default_commands_and_empty_allowlist() -> None:
    default_shell = next(capability for capability in Coder().capabilities if isinstance(capability, Shell))
    empty_shell = next(
        capability for capability in Coder(allowed_commands=[]).capabilities if isinstance(capability, Shell)
    )

    assert default_shell.allowed_commands == DEFAULT_ALLOWED_COMMANDS
    assert empty_shell.allowed_commands == []


def test_coder_for_agent_preserves_subclass(tmp_path: Path) -> None:
    coder = Coder(tmp_path)
    bound = coder.for_agent(Agent(TestModel()))

    assert isinstance(bound, Coder)
