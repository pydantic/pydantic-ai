"""Docs capability: an on-demand tool that locates Pydantic AI documentation."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic_ai._utils import replace_no_init
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.tools import AgentDepsT, RunContext
from pydantic_ai.toolsets import AgentToolset
from pydantic_ai_harness._workspace import require_workspace
from pydantic_ai_harness.pydantic_ai_docs._toolset import PydanticAIDocsToolset, PydanticAIDocsTopic

if TYPE_CHECKING:
    from pydantic_ai._instructions import AgentInstructions

_DOCS_PATH_ENV = 'PYDANTIC_AI_HARNESS_DOCS_PATH'
"""Env var holding a local pyai docs checkout path, used when `local_docs_path` is unset."""

_INSTRUCTIONS = (
    'You have a `read_pyai_docs` tool that returns Pydantic AI documentation on demand. '
    'Topics: capabilities, hooks, tools, tools-advanced, toolsets, agent. Read the relevant '
    'topic before authoring or modifying a Pydantic AI capability, hook, tool, or toolset, '
    'rather than relying on memory.'
)


@dataclass
class PydanticAIDocs(AbstractCapability[AgentDepsT]):
    """Locate and return Pydantic AI documentation on demand.

    Exposes a single `read_pyai_docs(topic)` tool. Docs are located and returned
    when asked for -- never bundled into context. Each call resolves the topic
    from a configured local checkout first, then falls back to fetching the page
    from `pydantic/pydantic-ai:main`, so it works in any environment.

    The local checkout path comes from `local_docs_path`, or the
    `PYDANTIC_AI_HARNESS_DOCS_PATH` env var when that is unset; with neither set
    every call goes straight to the remote source. The checkout is read through
    the run's workspace, and a run with a local path but no workspace fails at its
    start. The capability never runs git -- keep
    the local checkout current yourself; the remote path always reads `main`.

    ```python
    from pathlib import Path

    from pydantic_ai import Agent
    from pydantic_ai.capabilities import LocalWorkspace
    from pydantic_ai_harness.pydantic_ai_docs import PydanticAIDocs

    agent = Agent(
        'anthropic:claude-sonnet-4-6',
        capabilities=[LocalWorkspace('.'), PydanticAIDocs(local_docs_path=Path('docs'))],
    )
    ```
    """

    local_docs_path: Path | None = None
    """Pyai docs checkout inside the workspace. Relative paths use the workspace
    working directory. When `None`, falls back to the
    `PYDANTIC_AI_HARNESS_DOCS_PATH` env var, then to the remote source."""

    cache: bool = True
    """If `True`, each returned doc is memoized for one agent run."""

    _cache: dict[PydanticAIDocsTopic, str] = field(
        default_factory=dict[PydanticAIDocsTopic, str], init=False, repr=False, compare=False
    )
    """In-memory doc cache shared with toolsets created during one run."""

    async def for_run(self, ctx: RunContext[AgentDepsT]) -> PydanticAIDocs[AgentDepsT]:
        """Return a fresh per-run cache so workspace-local content cannot cross runs."""
        run = replace_no_init(self)
        run._cache = {}
        return run

    async def before_run(self, ctx: RunContext[AgentDepsT]) -> None:
        """Fail the run at its start when a local checkout is configured but no workspace holds it."""
        if self._resolved_local_path() is not None:
            require_workspace(ctx.workspace, 'PydanticAIDocs', ctx.messages)

    def _resolved_local_path(self) -> Path | None:
        """The local checkout path: `local_docs_path`, else the env var, else `None`.

        Paths identify files inside the run workspace. Relative paths are resolved
        from its working directory.
        """
        if self.local_docs_path is not None:
            return self.local_docs_path
        env_path = os.environ.get(_DOCS_PATH_ENV)
        return Path(env_path) if env_path else None

    def get_instructions(self) -> AgentInstructions[AgentDepsT] | None:
        """Static, cache-stable guidance on using the docs tool."""
        return _INSTRUCTIONS

    def get_toolset(self) -> AgentToolset[AgentDepsT] | None:
        """Toolset providing `read_pyai_docs` over the resolved local path and shared cache."""
        return PydanticAIDocsToolset[AgentDepsT](
            local_docs_path=self._resolved_local_path(),
            cache=self._cache if self.cache else None,
        )

    @classmethod
    def get_serialization_name(cls) -> str | None:
        """Serialization name for agent-spec support."""
        return 'PydanticAIDocs'
