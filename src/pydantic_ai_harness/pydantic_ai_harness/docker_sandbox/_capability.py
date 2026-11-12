"""Capability that supplies a local Docker container to an agent run."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import Authority
from pydantic_ai.tools import AgentDepsT, RunContext
from pydantic_ai.workspaces import WorkspaceBackend, WorkspaceRef
from pydantic_ai_harness._workspace import with_authority
from pydantic_ai_harness._workspace_provider import (
    check_path_authority,
    check_provider,
    check_working_dir,
    reject_defer_loading,
)
from pydantic_ai_harness.docker_sandbox._backend import (
    PROVIDER,
    DockerSandboxBackend,
    remove_container,
    resolve_executable,
)


@dataclass(kw_only=True)
class DockerSandbox(AbstractCapability[AgentDepsT]):
    """Supply a local [Docker](https://www.docker.com/) container as the run's workspace, created on first use.

    ```python {test="skip"}
    from pydantic_ai import Agent
    from pydantic_ai_harness import Coder, DockerSandbox

    agent = Agent('anthropic:claude-opus-5-5', capabilities=[DockerSandbox(image='python:3.13-slim'), Coder()])
    ```

    A run with no reference creates a fresh container. Pass a `WorkspaceRef` supplied by the
    application to attach to a container created earlier. The container outlives the run: remove it
    with [`destroy`][pydantic_ai_harness.docker_sandbox.DockerSandbox.destroy].

    This capability supplies execution only. Compose it with tools or capabilities that use the
    workspace, such as `Coder`, `Shell`, or `FileSystem`.
    """

    image: str | None = None
    """The image for a new container, such as `'python:3.13-slim'`; it needs a POSIX `sh` and the usual file utilities.

    Without one, the capability only attaches to and removes containers by ref, and leaves a run without a ref
    to the agent's other workspace capabilities.
    """

    working_dir: str | None = '/workspace'
    """Absolute directory in the container where commands start and relative paths resolve, created on first use if missing; `None` uses the image's own."""

    env: Mapping[str, str] | None = field(default=None, repr=False)
    """Environment variables every command gets, also in an attached container; nothing is read from the host."""

    allow_internet_access: bool = True
    """Whether a new container gets outbound network access, through Docker's default bridge: the internet, but also this machine's local network and services. `False` runs it with `--network none`."""

    docker_args: Sequence[str] = ()
    """Extra `docker run` arguments for a new container, such as `['--memory', '2g']` or a `--volume` mount.

    A new container gets `--pids-limit 2048` unless these pass their own.
    """

    executable: str = 'docker'
    """The container CLI to run; `'podman'` works too. It is looked up on `PATH` once, when the capability is created."""

    authority: Authority = 'operator'
    """The [authority][pydantic_ai.messages.Authority] of the sandbox's files: `'none'` for files nobody on the
    application's side wrote, such as a cloned third-party repository, and `'user'` for the user's own files.
    Anything below `'operator'` wraps the workspace in an [`AuthorityWorkspace`][pydantic_ai.workspaces.AuthorityWorkspace],
    so capabilities that send its files to the model send them with no more authority than that."""

    path_authority: Mapping[str, Authority] | None = None
    """Levels for some directories and files, overriding `authority` for the paths under them; the longest matching
    path wins. Paths are sandbox paths: relative ones resolve against the working directory, and `~` isn't expanded.
    See [`Workspace.authority_of`][pydantic_ai.workspaces.Workspace.authority_of]."""

    _executable_path: str = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        reject_defer_loading('DockerSandbox', self.defer_loading)
        check_working_dir(self.working_dir)
        check_path_authority(self.path_authority)
        self._executable_path = resolve_executable(self.executable)

    def backend(self, ref: WorkspaceRef) -> DockerSandboxBackend:
        """Attach lazily to an existing container by ref, starting it on first use if it is stopped."""
        return DockerSandboxBackend(
            ref=ref, working_dir=self.working_dir, env=self.env, executable=self._executable_path
        )

    async def destroy(self, ref: WorkspaceRef) -> None:
        """Remove a container and its anonymous volumes, whether it is running or not."""
        check_provider(ref, PROVIDER)
        await remove_container(ref.id, executable=self._executable_path)

    def get_workspace(self, ctx: RunContext[AgentDepsT], *, ref: WorkspaceRef | None) -> WorkspaceBackend | None:
        """The backend for this run, in an `AuthorityWorkspace` when `authority` or `path_authority` asks for one."""
        return with_authority(self._run_backend(ctx, ref=ref), self.authority, self.path_authority)

    def _run_backend(self, ctx: RunContext[AgentDepsT], *, ref: WorkspaceRef | None) -> WorkspaceBackend | None:
        """Build the backend for this run. No I/O here: it creates or attaches on first use."""
        del ctx
        if ref is not None:
            return self.backend(ref) if ref.provider == PROVIDER else None
        if self.image is None:
            return None
        return DockerSandboxBackend(
            image=self.image,
            working_dir=self.working_dir,
            env=self.env,
            allow_internet_access=self.allow_internet_access,
            docker_args=self.docker_args,
            executable=self._executable_path,
        )
