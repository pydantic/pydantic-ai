"""The built-in `repo_context` plugin: the workspace's instruction files, through harness `RepoContext`."""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field
from pydantic_ai_harness.repo_context import RepoContext

from .plugins import DepsT, PluginHost


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


def activate(host: PluginHost[DepsT]) -> None:
    """Bind `RepoContext` to the launch directory, the same workspace the `coder` plugin uses."""
    settings = host.settings(RepoContextSettings)
    host.add(
        RepoContext[DepsT](
            workspace_dir=Path.cwd(),
            home_dir=Path.home() if settings.walk_up else None,
            expose_inventory_tool=settings.inventory_tool,
            nested_traversal=settings.nested_traversal,
            nested_inject=settings.nested_inject,
        )
    )
