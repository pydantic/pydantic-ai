"""Capability that supplies a local Docker container to an agent run."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.tools import AgentDepsT, RunContext
from pydantic_ai.workspaces import WorkspaceBackend, WorkspaceRef
from pydantic_ai_harness._mounts import S3Mount, mounted
from pydantic_ai_harness._workspace_provider import (
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

    mounts: (
        Sequence[S3Mount]
        | Callable[[RunContext[AgentDepsT]], Sequence[S3Mount]]
        | Callable[[RunContext[AgentDepsT]], Awaitable[Sequence[S3Mount]]]
    ) = ()
    """Buckets to show as folders in the container, or a function of the run, sync or async, returning them with fresh keys.

    The container needs FUSE: give it `docker_args=['--device', '/dev/fuse', '--cap-add', 'SYS_ADMIN',
    '--security-opt', 'apparmor:unconfined']`.
    """

    _executable_path: str = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        reject_defer_loading('DockerSandbox', self.defer_loading)
        check_working_dir(self.working_dir)
        self._executable_path = resolve_executable(self.executable)

    @classmethod
    def from_spec(
        cls,
        *,
        id: str | None = None,
        description: str | None = None,
        defer_loading: bool = False,
        image: str | None = None,
        working_dir: str | None = '/workspace',
        env: Mapping[str, str] | None = None,
        allow_internet_access: bool = True,
        docker_args: Sequence[str] = (),
        executable: str = 'docker',
    ) -> DockerSandbox[Any]:
        """Build from an agent spec, naming the fields a spec can express.

        Core reads this signature to generate the spec's JSON schema. `mounts` is left out: a spec can't hold keys.
        """
        return cls(
            id=id,
            description=description,
            defer_loading=defer_loading,
            image=image,
            working_dir=working_dir,
            env=env,
            allow_internet_access=allow_internet_access,
            docker_args=docker_args,
            executable=executable,
        )

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
        """Build the backend for this run. No I/O here: it creates or attaches on first use."""
        if ref is not None:
            if ref.provider != PROVIDER:
                return None
            backend = self.backend(ref)
        elif self.image is None:
            return None
        else:
            backend = DockerSandboxBackend(
                image=self.image,
                working_dir=self.working_dir,
                env=self.env,
                allow_internet_access=self.allow_internet_access,
                docker_args=self.docker_args,
                executable=self._executable_path,
            )
        return mounted(backend, backend._run_as_root, self.mounts, ctx)  # pyright: ignore[reportPrivateUsage]
