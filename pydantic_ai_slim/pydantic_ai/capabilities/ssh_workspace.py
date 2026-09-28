from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import KW_ONLY, dataclass

from pydantic_ai._run_context import AgentDepsT, RunContext
from pydantic_ai.workspaces import ReadOnlyWorkspace, SSHWorkspaceBackend, Workspace, WorkspaceBackend, WorkspaceRef

from .abstract import AbstractCapability


@dataclass
class SSHWorkspace(AbstractCapability[AgentDepsT]):
    """Gives runs a [workspace](../workspace.md) on a remote host over SSH, using your `ssh` client and its configuration.

    Commands run as the remote user, with that user's full authority on the host. Wrap it in
    [`BubblewrapSandbox`][pydantic_ai.capabilities.BubblewrapSandbox] to sandbox them there.

    ```python
    from pydantic_ai import Agent
    from pydantic_ai.capabilities import SSHWorkspace

    agent = Agent('anthropic:claude-opus-5-5', capabilities=[SSHWorkspace('dev@build-box', working_dir='/srv/app')])
    ```

    It declines a ref for any other host or directory, so a ref in message history can't point it elsewhere.
    """

    destination: str
    """The host, as you'd pass it to `ssh`: `'user@host'`, a `Host` alias, or `'ssh://user@host:port'`."""

    _: KW_ONLY

    working_dir: str | None = None
    """Where commands start and relative paths resolve on the host; defaults to the login directory."""

    read_only: bool = False
    """Whether to wrap the workspace in a [`ReadOnlyWorkspace`][pydantic_ai.workspaces.ReadOnlyWorkspace]."""

    env: Mapping[str, str] | None = None
    """Environment variables for every command, on top of the remote login environment."""

    ssh_args: Sequence[str] = ()
    """Extra `ssh` arguments, such as `['-i', key_path]`; prefer your SSH configuration where you can."""

    id: str | None = 'ssh_workspace'
    """Fixed, so a later `SSHWorkspace` replaces an earlier one whole; pass distinct ids to keep both."""

    def __post_init__(self) -> None:
        # Surface an invalid destination, `env` or platform where the capability is written, not on the first run.
        self._backend()

    def _backend(self) -> SSHWorkspaceBackend:
        return SSHWorkspaceBackend(self.destination, working_dir=self.working_dir, env=self.env, ssh_args=self.ssh_args)

    @classmethod
    def combine(cls, capabilities: Sequence[AbstractCapability[AgentDepsT]]) -> AbstractCapability[AgentDepsT]:
        # Like `LocalWorkspace`: the later configuration replaces the earlier one whole.
        return capabilities[-1]

    def get_workspace(self, ctx: RunContext[AgentDepsT], *, ref: WorkspaceRef | None) -> WorkspaceBackend | None:
        backend = self._backend()
        if ref is not None and ref != backend.ref:
            return None
        return ReadOnlyWorkspace(Workspace(backend)) if self.read_only else backend
