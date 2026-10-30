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
from pydantic_clai2.runtime.session import supports_local_workspace

_GLOBAL_INSTRUCTION_FILES: tuple[str, ...] = ('.claude/CLAUDE.md', '.codex/AGENTS.md', '.pydantic-ai/AGENTS.md')
"""User-global instruction files under your home directory, broadest first, as Claude Code and Codex keep them.

`.pydantic-ai/AGENTS.md` is the harness's own, so it comes last and is the most specific of the three.
"""


class RepoContextSettings(BaseModel):
    """The JSON a `repo_context` declaration may carry. Filenames stay `RepoContext`'s own defaults."""

    model_config = ConfigDict(extra='forbid', frozen=True, strict=True)
    walk_up: bool = Field(
        default=True,
        description='Also load instruction files from every directory above the workspace, up to the filesystem root.',
    )
    global_files: bool = Field(
        default=True,
        description=(
            'Also load your user-global instruction files: `~/.claude/CLAUDE.md`, `~/.codex/AGENTS.md` and '
            '`~/.pydantic-ai/AGENTS.md`.'
        ),
    )
    inventory_tool: bool = Field(
        default=False, description="Expose `inventory_agent_context`, which maps the repo's coding-assistant assets."
    )
    nested_traversal: bool = Field(
        default=True,
        description=(
            'Surface the instruction files of a directory the agent reads or lists, and of every directory between it '
            'and the workspace.'
        ),
    )
    nested_inject: Literal['pointer', 'contents'] = Field(
        default='contents', description='What nested traversal adds: the file contents or a one-line pointer.'
    )


class RepoContextPlugin(Plugin[RepoContextSettings, DepsT]):
    """`RepoContext`, anchored at the run workspace's working directory like the `coder` plugin."""

    def get_capabilities(self) -> Sequence[AgentCapability[DepsT]]:
        if not supports_local_workspace():
            return ()
        settings = self.settings
        home = Path.home()
        return (
            RepoContext[DepsT](
                walk_up='all' if settings.walk_up else None,
                global_files=[str(home / path) for path in _GLOBAL_INSTRUCTION_FILES] if settings.global_files else (),
                expose_inventory_tool=settings.inventory_tool,
                nested_traversal=settings.nested_traversal,
                nested_inject=settings.nested_inject,
            ),
        )
