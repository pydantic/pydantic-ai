"""Complete coding-agent harness assembled from regular capabilities."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from pydantic_ai import Agent
from pydantic_ai.capabilities import AbstractCapability, Capability, CombinedCapability
from pydantic_ai.tools import AgentDepsT

from pydantic_ai_harness.compaction import ClearToolResults, WarnNearLimits
from pydantic_ai_harness.filesystem import FileSystem
from pydantic_ai_harness.planning import Planning
from pydantic_ai_harness.repo_context import RepoContext
from pydantic_ai_harness.shell import LLM_API_KEY_ENV_PATTERNS, Shell
from pydantic_ai_harness.subagents import SubAgent, SubAgents
from pydantic_ai_harness.tool_output_limits import ToolOutputLimits

DEFAULT_ALLOWED_COMMANDS: tuple[str, ...] = (
    'git',
    'rg',
    'grep',
    'find',
    'ls',
    'cat',
    'sed',
    'head',
    'tail',
    'python',
    'uv',
    'pytest',
    'ruff',
    'make',
)
"""Commands available to `Coder` unless an explicit allowlist is supplied."""


def _explorer(workspace: str | Path) -> SubAgent[AgentDepsT]:
    agent = Agent[AgentDepsT](  # pyright: ignore[reportCallIssue, reportArgumentType]
        name='explorer',
        description='Explore the codebase and answer questions without modifying anything',
        instructions='Answer with concrete paths and evidence.',
        capabilities=[
            FileSystem[AgentDepsT](workspace, read_only=True),
            RepoContext[AgentDepsT](workspace_dir=Path(workspace)),
        ],
    )
    return SubAgent(agent)


class Coder(CombinedCapability[AgentDepsT]):
    """A complete coding-agent harness built as a regular combined capability.

    See the class definition and [Coder docs](https://pydantic.dev/docs/ai/harness/coder/) for the exact composition.

    It ships with no default instructions: modern models don't need procedural coaching, and the composed capabilities
    contribute their own tool guidance. Pass `instructions=` to add your own.

    The command allowlist is a guardrail against accidents, not a security boundary. Validation checks only the first
    token, and allowed commands such as `python`, `git`, `uv`, and `make` can spawn arbitrary processes. Run untrusted
    work in an OS-level sandbox such as `ModalSandbox` or a container.
    """

    def __init__(
        self,
        workspace: str | Path = '.',
        *,
        allowed_commands: Sequence[str] | None = None,
        subagents: Sequence[SubAgent[AgentDepsT]] | None = None,
        instructions: str | None = None,
    ) -> None:
        delegates = [_explorer(workspace)] if subagents is None else subagents
        capabilities: list[AbstractCapability[AgentDepsT]] = []
        if instructions is not None:
            capabilities.append(Capability[AgentDepsT](instructions=instructions))
        capabilities.extend(
            [
                FileSystem[AgentDepsT](workspace),
                Shell[AgentDepsT](
                    cwd=workspace,
                    allowed_commands=DEFAULT_ALLOWED_COMMANDS if allowed_commands is None else allowed_commands,
                    denied_env_patterns=LLM_API_KEY_ENV_PATTERNS,
                ),
                RepoContext[AgentDepsT](workspace_dir=Path(workspace)),
                Planning[AgentDepsT](),
            ]
        )
        if delegates:
            capabilities.append(SubAgents[AgentDepsT](agents=delegates, agent_folders=None))
        capabilities.extend(
            [
                ClearToolResults[AgentDepsT](max_fraction=0.7),
                WarnNearLimits[AgentDepsT](max_context_fraction=0.9),
                ToolOutputLimits[AgentDepsT](),
            ]
        )
        super().__init__(capabilities)
