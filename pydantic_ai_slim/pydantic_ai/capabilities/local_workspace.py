from __future__ import annotations

from collections.abc import Mapping
from dataclasses import KW_ONLY, dataclass
from pathlib import Path

from pydantic_ai._run_context import AgentDepsT, RunContext
from pydantic_ai.workspaces import LocalWorkspaceBackend, ReadOnlyWorkspace, Workspace, WorkspaceBackend, WorkspaceRef

from .abstract import AbstractCapability


@dataclass
class LocalWorkspace(AbstractCapability[AgentDepsT]):
    """Gives runs a [workspace](../workspace.md) on this machine: host subprocesses and the host filesystem.

    This isolates nothing and is not a jail. `working_dir` is only where commands start and what
    relative paths resolve against, so tools reach anywhere on the host that this process can. Use
    it for trusted local work; run untrusted code through a container- or VM-based workspace.

    ```python
    from pydantic_ai import Agent
    from pydantic_ai.capabilities import LocalWorkspace

    agent = Agent('anthropic:claude-sonnet-5', capabilities=[LocalWorkspace('~/project')])
    ```

    Each run gets a [`LocalWorkspaceBackend`][pydantic_ai.workspaces.LocalWorkspaceBackend] for
    `working_dir`, whose [`WorkspaceRef`][pydantic_ai.workspaces.WorkspaceRef] is
    `WorkspaceRef(provider='local', id=<working_dir>)` from construction: the directory is the
    environment, and the first operation raises
    [`WorkspaceUnavailableError`][pydantic_ai.workspaces.WorkspaceUnavailableError] if it does
    not exist. This capability supplies the workspace when
    there is no reference to continue from, or when the reference names its own `working_dir`. It
    declines every other reference, including a local one for a different directory, so a reference
    in message history cannot point the agent at an arbitrary directory on the host. Other workspace
    capabilities can be listed before or after it to continue in their providers' environments.
    """

    working_dir: str | Path
    """The default working directory for commands and the base for relative workspace paths.

    A leading `~` is expanded to the user's home directory, and a relative path, such as `'.'`,
    resolves against the current directory when the capability is constructed. There is no default,
    so a run never lands in the host process's working directory implicitly. The caller creates and
    removes the directory.
    """

    _: KW_ONLY

    read_only: bool = False
    """Whether to wrap the workspace in a [`ReadOnlyWorkspace`][pydantic_ai.workspaces.ReadOnlyWorkspace].

    Reads and directory listings work; commands and file changes raise
    [`UserError`][pydantic_ai.exceptions.UserError]. This restricts access through the workspace
    API and is not isolation.
    """

    env: Mapping[str, str] | None = None
    """Environment variables every command in this workspace gets; the per-call `env` is layered on top.

    Commands inherit nothing from the agent process, so pass what they need, e.g.
    `LocalWorkspace('.', env={'PATH': os.environ['PATH'], 'HOME': os.environ['HOME']})`. Without
    `PATH`, tools installed outside the system default path (Homebrew, `~/.local/bin`) are not
    found. Don't pass `os.environ` wholesale: it hands the model's commands every secret in the
    process, LLM API keys included.
    """

    id: str | None = 'local_workspace'
    """One-off: a run has a single workspace, so the id is fixed by default.

    Two of them resolve to one via [`combine`][pydantic_ai.capabilities.AbstractCapability.combine],
    which keeps the last. Pass a distinct `id` to keep both, or `id=None` for derived ids; the first
    one in capability order still supplies the workspace.
    """

    def __post_init__(self) -> None:
        # Pin a relative `working_dir` to today's directory, and surface an unusable platform where the
        # capability is written, not on the first run.
        self.working_dir = LocalWorkspaceBackend(self.working_dir).ref.id

    def get_workspace(self, ctx: RunContext[AgentDepsT], *, ref: WorkspaceRef | None) -> WorkspaceBackend | None:
        backend = LocalWorkspaceBackend(self.working_dir, env=self.env)
        if ref is not None and ref != backend.ref:
            # Another provider's environment, or a local directory other than the configured one:
            # a ref from message history must never redirect the agent to an arbitrary host directory.
            return None
        return ReadOnlyWorkspace(Workspace(backend)) if self.read_only else backend
