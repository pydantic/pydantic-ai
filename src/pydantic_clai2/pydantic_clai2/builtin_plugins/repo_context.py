"""Give the model your repository's instruction files, such as AGENTS.md.

The built-in `repo_context` plugin: the workspace's instruction files, through harness `RepoContext`.
"""

from collections.abc import Sequence
from contextlib import suppress
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue

from pydantic_ai.capabilities import AgentCapability
from pydantic_ai.tools import RunContext
from pydantic_ai_harness.repo_context import RepoContext
from pydantic_clai2.config import Settings
from pydantic_clai2.plugins import DepsT, Plugin
from pydantic_clai2.runtime.session import WorkspacelessCapability, supports_local_workspace


def _expand_user_path(value: str) -> Path:
    path = Path(value)
    return Path.home().joinpath(*path.parts[1:]) if path.parts[0] == '~' else path


class RepoContextSettings(BaseModel):
    """The JSON a `repo_context` declaration may carry. Filenames stay `RepoContext`'s own defaults."""

    model_config = ConfigDict(extra='forbid', frozen=True, strict=True)
    walk_up: bool = Field(
        default=False,
        description='Also load ancestor instructions up to `repo_context.walk_up_root` (home by default).',
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
        if not supports_local_workspace():
            return ()
        return (WorkspacelessCapability(self._for_run),)

    def _for_run(self, _ctx: RunContext[DepsT]) -> RepoContext[DepsT]:
        # Read once per run: settings edits apply to the next run, not its cached instructions.
        values: dict[str, JsonValue] = {}
        for key, field in (
            ('repo_context.walk_up_root', 'repo_context_walk_up_root'),
            ('repo_context.autoload_symlink_roots', 'repo_context_autoload_symlink_roots'),
        ):
            with suppress(KeyError):
                values[field] = self.host.setting(key)
        policy = Settings.model_validate(values)
        return RepoContext[DepsT](
            home_dir=_expand_user_path(policy.repo_context_walk_up_root) if self.settings.walk_up else None,
            autoload_symlink_roots=[_expand_user_path(root) for root in policy.repo_context_autoload_symlink_roots],
            expose_inventory_tool=self.settings.inventory_tool,
            nested_traversal=self.settings.nested_traversal,
            nested_inject=self.settings.nested_inject,
        )
