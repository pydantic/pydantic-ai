"""Docs capability: an on-demand tool that locates Pydantic AI documentation."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.tools import AgentDepsT
from pydantic_ai.toolsets import AgentToolset

from pydantic_ai_harness.experimental.docs._toolset import PyaiDocsToolset, PyaiDocsTopic

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
class PyaiDocs(AbstractCapability[AgentDepsT]):
    """Locate and return Pydantic AI documentation on demand.

    Exposes a single `read_pyai_docs(topic)` tool. Docs are located and returned
    when asked for -- never bundled into context. Each call resolves the topic
    from a configured local checkout first, then falls back to fetching the page
    from `pydantic/pydantic-ai:main`, so it works in any environment.

    The local checkout path comes from `local_docs_path`, or the
    `PYDANTIC_AI_HARNESS_DOCS_PATH` env var when that is unset; with neither set
    every call goes straight to the remote source. The capability never runs git
    -- keep the local checkout current yourself; the remote path always reads
    `main`.

    ```python
    from pathlib import Path

    from pydantic_ai import Agent
    from pydantic_ai_harness.experimental.docs import PyaiDocs

    agent = Agent(
        'anthropic:claude-sonnet-4-6',
        capabilities=[PyaiDocs(local_docs_path=Path('~/pydantic/ai/base/docs').expanduser())],
    )
    ```
    """

    local_docs_path: Path | None = None
    """Local pyai docs checkout to read first. When `None`, falls back to the
    `PYDANTIC_AI_HARNESS_DOCS_PATH` env var, then to the remote source."""

    cache: bool = True
    """If `True`, each returned doc is memoized in-process for the capability's
    lifetime, so a topic is read or fetched at most once."""

    _cache: dict[PyaiDocsTopic, str] = field(
        default_factory=dict[PyaiDocsTopic, str], init=False, repr=False, compare=False
    )
    """In-memory doc cache shared with the toolset, so memoized docs outlive a
    single `get_toolset` call."""

    def _resolved_local_path(self) -> Path | None:
        """The local checkout path: `local_docs_path`, else the env var, else `None`.

        `~` is expanded so a raw `~/...` path resolves to the local checkout
        instead of silently falling through to the remote source.
        """
        if self.local_docs_path is not None:
            return self.local_docs_path.expanduser()
        env_path = os.environ.get(_DOCS_PATH_ENV)
        return Path(env_path).expanduser() if env_path else None

    def get_instructions(self) -> AgentInstructions[AgentDepsT] | None:
        """Static, cache-stable guidance on using the docs tool."""
        return _INSTRUCTIONS

    def get_toolset(self) -> AgentToolset[AgentDepsT] | None:
        """Toolset providing `read_pyai_docs` over the resolved local path and shared cache."""
        return PyaiDocsToolset[AgentDepsT](
            local_docs_path=self._resolved_local_path(),
            cache=self._cache if self.cache else None,
        )

    @classmethod
    def get_serialization_name(cls) -> str | None:
        """Serialization name for agent-spec support."""
        return 'PyaiDocs'
