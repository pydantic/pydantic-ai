"""Give the model your repository's instruction files, such as AGENTS.md.

The built-in `repo_context` plugin: the workspace's instruction files, through harness `RepoContext`.
"""

from collections.abc import Sequence
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from pydantic_ai.capabilities import AgentCapability
from pydantic_ai_harness.repo_context import RepoContext
from pydantic_clai2.plugins import DepsT, Plugin
from pydantic_clai2.runtime._session import _supports_local_workspace  # pyright: ignore[reportPrivateUsage]


class RepoContextSettings(BaseModel):
    """The JSON a `repo_context` declaration may carry. Filenames stay `RepoContext`'s own defaults."""

    model_config = ConfigDict(extra='forbid', frozen=True, strict=True)
    walk_up: bool = Field(
        default=False, description='Also load instruction files from every directory between the workspace and home.'
    )
    inventory_tool: bool = Field(
        default=False, description="Expose `inventory_agent_context`, which maps the repo's coding-assistant assets."
    )
    nested_traversal: bool = Field(
        default=False, description="Surface a directory's instruction file when the agent reads or lists it."
    )
    nested_inject: Literal['pointer', 'contents'] = Field(
        default='pointer', description='What nested traversal adds: a one-line pointer or the file contents.'
    )


class RepoContextPlugin(Plugin[RepoContextSettings, DepsT]):
    """`RepoContext`, anchored at the run workspace's working directory like the `coder` plugin."""

    def get_capabilities(self) -> Sequence[AgentCapability[DepsT]]:
        if not _supports_local_workspace():
            return ()
        settings = self.settings
        return (
            RepoContext[DepsT](
                home_dir=Path.home() if settings.walk_up else None,
                expose_inventory_tool=settings.inventory_tool,
                nested_traversal=settings.nested_traversal,
                nested_inject=settings.nested_inject,
            ),
        )
