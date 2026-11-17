from __future__ import annotations

from collections.abc import Mapping
from dataclasses import KW_ONLY, dataclass

# `Path` and `Authority` resolve the inherited fields' annotations here, for the agent spec schema.
from pathlib import Path  # noqa: F401  # pyright: ignore[reportUnusedImport]

from pydantic_ai._run_context import AgentDepsT
from pydantic_ai.messages import Authority  # noqa: F401  # pyright: ignore[reportUnusedImport]
from pydantic_ai.workspaces import WindowsWorkspaceBackend
from pydantic_ai.workspaces._host import HostWorkspaceBackend

from .local_workspace import LocalWorkspace


@dataclass
class WindowsWorkspace(LocalWorkspace[AgentDepsT]):
    """Gives runs a [workspace](../workspace.md) on this Windows machine: PowerShell commands and the host filesystem.

    [`LocalWorkspace`][pydantic_ai.capabilities.LocalWorkspace] for Windows, with the same options. Commands run in
    PowerShell and paths are Windows paths; see
    [`WindowsWorkspaceBackend`][pydantic_ai.workspaces.WindowsWorkspaceBackend]. It isolates nothing: tools reach
    anywhere on the host this process can. Constructing it elsewhere raises `NotImplementedError`.

    ```python {test="skip"}
    from pydantic_ai import Agent
    from pydantic_ai.capabilities import WindowsWorkspace

    agent = Agent('anthropic:claude-opus-5-5', capabilities=[WindowsWorkspace('.')])
    ```
    """

    _: KW_ONLY

    env: Mapping[str, str] | None = None
    """Environment variables for every command, on top of the variables Windows tools need to start
    (`PATH`, `PATHEXT`, `SYSTEMROOT`, `TEMP`, `USERPROFILE`, `APPDATA` and the like).

    Nothing else from this process's environment reaches commands; don't pass `os.environ` (secrets).
    """

    id: str | None = 'windows_workspace'
    """Fixed, so a later `WindowsWorkspace` replaces an earlier one whole; pass distinct ids to keep both."""

    def _backend(self) -> HostWorkspaceBackend:
        return WindowsWorkspaceBackend(self.working_dir, env=self.env)
