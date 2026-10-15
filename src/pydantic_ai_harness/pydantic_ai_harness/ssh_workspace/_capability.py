from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import KW_ONLY, dataclass, field

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.exceptions import UserError
from pydantic_ai.tools import AgentDepsT, RunContext
from pydantic_ai.workspaces import ReadOnlyWorkspace, Workspace, WorkspaceBackend, WorkspaceRef
from pydantic_ai_harness.ssh_workspace._backend import SSHWorkspaceBackend


@dataclass
class SSHWorkspace(AbstractCapability[AgentDepsT]):
    """Gives runs a [workspace](https://pydantic.dev/docs/ai/core-concepts/workspace/) on a remote host over SSH, using your `ssh` client and its configuration.

    Commands run as the remote user, with that user's full authority on the host. Wrap it in
    [`BubblewrapSandbox`][pydantic_ai_harness.bubblewrap_sandbox.BubblewrapSandbox] to sandbox them there.

    ```python
    from pydantic_ai import Agent
    from pydantic_ai_harness import SSHWorkspace

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

    env: Mapping[str, str] | None = field(default=None, repr=False)
    """Environment variables for every command, on top of the remote login environment."""

    ssh_args: Sequence[str] = ()
    """Extra `ssh` arguments, such as `['-i', key_path]`; prefer your SSH configuration where you can."""

    id: str | None = 'ssh_workspace'
    """Fixed, so a later `SSHWorkspace` replaces an earlier one whole; pass distinct ids to keep both."""

    def __post_init__(self) -> None:
        if self.defer_loading:
            raise UserError(
                '`SSHWorkspace` does not support `defer_loading=True`: '
                'the workspace is selected before deferred capabilities load.'
            )
        # Surface an invalid destination, `env` or platform where the capability is written, not on the first run.
        self._configured()

    def _configured(self) -> SSHWorkspaceBackend:
        return SSHWorkspaceBackend(self.destination, working_dir=self.working_dir, env=self.env, ssh_args=self.ssh_args)

    def backend(self, ref: WorkspaceRef) -> SSHWorkspaceBackend:
        """Attach to this capability's host and working directory by ref, without connecting.

        Raises:
            ValueError: If `ref` is for another host or working directory; this capability never redirects
                commands to a host it wasn't configured with.
        """
        backend = self._configured()
        if ref != backend.ref:
            raise ValueError(f'workspace {ref.provider}:{ref.id} is not this SSH workspace ({backend.ref.id})')
        return backend

    @classmethod
    def combine(cls, capabilities: Sequence[AbstractCapability[AgentDepsT]]) -> AbstractCapability[AgentDepsT]:
        # Like `LocalWorkspace`: the later configuration replaces the earlier one whole.
        return capabilities[-1]

    def get_workspace(self, ctx: RunContext[AgentDepsT], *, ref: WorkspaceRef | None) -> WorkspaceBackend | None:
        backend = self._configured()
        if ref is not None and ref != backend.ref:
            return None
        return ReadOnlyWorkspace(Workspace(backend)) if self.read_only else backend
