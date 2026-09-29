"""Capability that supplies a local Docker container to an agent run."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import KW_ONLY, dataclass, field

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.exceptions import UserError
from pydantic_ai.tools import AgentDepsT, RunContext
from pydantic_ai.workspaces import WorkspaceBackend, WorkspaceRef
from pydantic_ai_harness.docker_sandbox._backend import PROVIDER, DockerSandboxBackend, remove_container


@dataclass
class DockerSandbox(AbstractCapability[AgentDepsT]):
    """Supply a local [Docker](https://www.docker.com/) container as the run's workspace, created on first use.

    ```python {test="skip"}
    from pydantic_ai import Agent
    from pydantic_ai_harness import Coder, DockerSandbox

    agent = Agent('anthropic:claude-opus-5-5', capabilities=[DockerSandbox('python:3.13-slim'), Coder()])
    ```

    A run with no reference creates a fresh container. Pass a `WorkspaceRef` supplied by the
    application to attach to a container created earlier. The container outlives the run: remove it
    with [`destroy`][pydantic_ai_harness.docker_sandbox.DockerSandbox.destroy].

    This capability supplies execution only. Compose it with tools or capabilities that use the
    workspace, such as `Coder`, `Shell`, or `FileSystem`.
    """

    image: str
    """The image for a new container. It needs a POSIX `sh` and the usual file utilities."""

    _: KW_ONLY

    working_dir: str = '/workspace'
    """Absolute directory in the container where commands start and relative paths resolve; created if missing."""

    env: Mapping[str, str] | None = field(default=None, repr=False)
    """Environment variables every command gets, also in an attached container; nothing is read from the host."""

    network: bool = True
    """Whether a new container can reach the network; `False` runs it with `--network none`."""

    docker_args: Sequence[str] = ()
    """Extra `docker run` arguments for a new container, such as `['--memory', '2g']` or a `--volume` mount."""

    executable: str = 'docker'
    """The container CLI to run; `'podman'` works too."""

    def __post_init__(self) -> None:
        if self.defer_loading:
            raise UserError(
                '`DockerSandbox` does not support `defer_loading=True`: '
                'the workspace is selected before deferred capabilities load.'
            )
        # Surface an invalid image, `working_dir`, `env` or platform where the capability is written, not on the first run.
        self._backend(ref=None)

    def _backend(self, *, ref: WorkspaceRef | None) -> DockerSandboxBackend:
        return DockerSandboxBackend(
            None if ref is not None else self.image,
            ref=ref,
            working_dir=self.working_dir,
            env=self.env,
            network=self.network,
            docker_args=self.docker_args,
            executable=self.executable,
        )

    def backend(self, ref: WorkspaceRef) -> DockerSandboxBackend:
        """Attach lazily to an existing container by ref, starting it on first use if it is stopped."""
        return self._backend(ref=ref)

    async def destroy(self, ref: WorkspaceRef) -> None:
        """Remove a container and its anonymous volumes, whether it is running or not."""
        if ref.provider != PROVIDER:
            raise ValueError(f'unsupported workspace provider {ref.provider!r}; expected {PROVIDER!r}')
        await remove_container(ref.id, executable=self.executable)

    def get_workspace(self, ctx: RunContext[AgentDepsT], *, ref: WorkspaceRef | None) -> WorkspaceBackend | None:
        """Build the backend for this run. No I/O here: it creates or attaches on first use."""
        del ctx
        if ref is not None and ref.provider != PROVIDER:
            return None
        return self._backend(ref=ref)
