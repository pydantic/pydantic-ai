"""Give the model your repository's instruction files, such as AGENTS.md.

The built-in `repo_context` plugin: the workspace's instruction files, through harness `RepoContext`.
"""

from collections.abc import Sequence
from contextlib import suppress
from pathlib import Path
from typing import Literal

import anyio
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from pydantic_ai.capabilities import AgentCapability
from pydantic_ai.tools import RunContext
from pydantic_ai_harness.repo_context import RepoContext
from pydantic_clai2.plugins import DepsT, Plugin
from pydantic_clai2.runtime.session import WorkspacelessCapability, supports_local_workspace

_PATHS = TypeAdapter(list[str])


class RepoContextSettings(BaseModel):
    """The JSON a `repo_context` declaration may carry. Filenames stay `RepoContext`'s own defaults."""

    model_config = ConfigDict(extra='forbid', frozen=True, strict=True)
    walk_up: bool = Field(
        default=False,
        description=(
            'Also load ancestor instructions up to the nearest `repo_context.walk_up_roots` entry (home by default).'
        ),
    )
    inventory_tool: bool = Field(
        default=False, description="Expose `inventory_agent_context`, which maps the repo's coding-assistant assets."
    )
    nested_traversal: bool = Field(
        default=False,
        description=(
            'Surface the instruction files of a directory the agent reads or lists, and of every directory between it '
            'and the workspace.'
        ),
    )
    nested_inject: Literal['pointer', 'contents'] = Field(
        default='pointer', description='What nested traversal adds: a one-line pointer or the file contents.'
    )


class RepoContextPlugin(Plugin[RepoContextSettings, DepsT]):
    """`RepoContext`, anchored at the run workspace's working directory like the `coder` plugin."""

    def get_capabilities(self) -> Sequence[AgentCapability[DepsT]]:
        if not supports_local_workspace():
            return ()
        return (WorkspacelessCapability(self._for_run),)

    async def _for_run(self, _ctx: RunContext[DepsT]) -> RepoContext[DepsT]:
        # Read once per run: settings edits apply to the next run, not its cached instructions.
        walk_up_roots = ['~']
        symlink_roots: list[str] = []
        # The store validates path policy; narrow the host's JSON values without depending on unrelated settings.
        with suppress(KeyError):
            walk_up_roots = _PATHS.validate_python(self.host.setting('repo_context.walk_up_roots'), strict=True)
        with suppress(KeyError):
            symlink_roots = _PATHS.validate_python(
                self.host.setting('repo_context.autoload_symlink_roots'), strict=True
            )
        home_dir = None
        if self.settings.walk_up:
            # `RepoContext` compares `home_dir` as given against the resolved working directory, so resolve the
            # roots too: one spelled through a symlink would otherwise never contain the workspace.
            working_dir = Path(await anyio.Path(self.host.workspace).resolve())
            roots = [Path(await anyio.Path(Path(root).expanduser()).resolve()) for root in walk_up_roots]
            home_dir = max(
                (root for root in roots if working_dir.is_relative_to(root)),
                key=lambda root: len(root.parts),
                default=None,
            )
        return RepoContext[DepsT](
            home_dir=home_dir,
            autoload_symlink_roots=[Path(root).expanduser() for root in symlink_roots],
            expose_inventory_tool=self.settings.inventory_tool,
            nested_traversal=self.settings.nested_traversal,
            nested_inject=self.settings.nested_inject,
        )
