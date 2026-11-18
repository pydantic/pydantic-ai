"""Capability that supplies a Modal sandbox as an agent run's workspace."""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import Authority
from pydantic_ai.tools import AgentDepsT, RunContext, ToolDefinition
from pydantic_ai.workspaces import WorkspaceBackend, WorkspaceRef
from pydantic_ai_harness._workspace import with_authority
from pydantic_ai_harness._workspace_provider import (
    check_integer,
    check_path_authority,
    check_provider,
    check_working_dir,
    reject_defer_loading,
)
from pydantic_ai_harness.modal_sandbox._backend import (
    DEFAULT_APP_NAME,
    DEFAULT_SANDBOX_TIMEOUT,
    MAX_SANDBOX_TIMEOUT,
    MIN_SANDBOX_TIMEOUT,
    ModalSandboxBackend,
    terminate_sandbox,
)

if TYPE_CHECKING:
    import modal

UPGRADE_DOCS_URL = 'https://pydantic.dev/docs/ai/harness/modal-sandbox/#upgrading-from-the-previous-modalsandbox'

# The tools of `Shell` and `FileSystem`, which run against `ctx.workspace`. A run with none of these
# names most likely has no way to reach the sandbox; custom tools by other names are not detected.
_WORKSPACE_TOOL_NAMES = frozenset(
    {
        'run_command',
        'start_command',
        'check_command',
        'stop_command',
        'shell',
        'read_file',
        'write_file',
        'edit_file',
        'list_directory',
        'search_files',
        'find_files',
        'create_directory',
        'file_info',
        'list_files',
        'grep',
    }
)


class ModalSandboxNoToolsWarning(UserWarning):
    """Warned once per `ModalSandbox` when a run has no `Shell` or `FileSystem` tool to use the sandbox.

    Pass `ModalSandbox(warn_if_no_tools=False)`, or filter this category, when only your own tools use it.
    """


_LIFETIME_NOTE = (
    'The sandbox is no longer terminated when the run ends: it keeps running, and billing, until its '
    '`sandbox_timeout` (24 hours by default) ends it. Set `idle_timeout=...`, or terminate it with '
    '`await ModalSandbox().destroy(result.workspace.ref)`.'
)


def _no_workspace_tools_message(agent_name: str | None) -> str:
    # Naming the agent tells the user which of several agents lacks the tools.
    run = 'this run' if agent_name is None else f'this run of agent {agent_name!r}'
    return (
        "`ModalSandbox` supplies the Modal sandbox as the run's `ctx.workspace` and registers no tools of its own, "
        f'and {run} has no `Shell` or `FileSystem` tool. Add `Coder()`, or `Shell()` and/or `FileSystem()`, '
        'alongside it. If your own code or tools use `ctx.workspace`, pass `ModalSandbox(warn_if_no_tools=False)` '
        f'to silence this warning. {_LIFETIME_NOTE} See {UPGRADE_DOCS_URL}'
    )


@dataclass(kw_only=True, init=False)
class ModalSandbox(AbstractCapability[AgentDepsT]):
    """Supply a Modal sandbox as the run's workspace.

    A run with an explicit `WorkspaceRef` attaches to that sandbox. Without a reference, the
    first workspace operation creates a fresh Modal sandbox. Pydantic AI does not terminate the
    sandbox; terminating it is the application's job.

    The capability registers no tools. Pair it with `Coder`, or with `Shell` and `FileSystem`,
    which run their tools in the workspace, or write tools of your own that use it. Shell
    commands run under `sh -c` in the sandbox's shell environment.
    """

    image: str | modal.Image | None = None
    """Image a newly created sandbox runs: a registry tag, or a `modal.Image`; `None` is Debian slim with Python 3.12, `git`, and `ripgrep`."""

    app_name: str = DEFAULT_APP_NAME
    """Modal app used when creating a sandbox."""

    create_app_if_missing: bool = True
    """Whether Modal may create the app."""

    sandbox_timeout: int = DEFAULT_SANDBOX_TIMEOUT
    """Total lifetime of a newly created sandbox, in seconds (Modal's `timeout`). Defaults to Modal's maximum, 24 hours."""

    idle_timeout: int | None = None
    """Seconds without activity after which Modal terminates a newly created sandbox; `None` never does."""

    working_dir: str | None = None
    """Absolute directory commands start in and relative paths resolve against; when `None`, `/root` on the
    default image, otherwise the image's own working directory."""

    env: Mapping[str, str] | None = field(default=None, repr=False)
    """Environment variables every command in the sandbox gets; a command's own `env` is layered on top."""

    client: modal.Client | None = None
    """The `modal.Client` to create, attach and destroy with, such as one from `modal.Client.from_credentials`;
    Modal's client from the environment or config when `None`. A spec cannot set it.

    To use different credentials for each user, supply the workspace from a capability of your own, which can build
    the client from the run's `deps` and pass it to `ModalSandboxBackend`.
    """

    warn_if_no_tools: bool = True
    """Warn, once per `ModalSandbox` instance, when none of a run's tools has a `Shell` or `FileSystem` tool name.

    Only tool names are checked, so custom tools that reach the sandbox under other names do not count.

    Set it to `False` for agents that reach the sandbox only from their own tools or hooks.
    """

    authority: Authority = 'operator'
    """The [authority][pydantic_ai.messages.Authority] of the sandbox's files: `'none'` for files nobody on the
    application's side wrote, such as a cloned third-party repository, and `'user'` for the user's own files.
    Anything below `'operator'` wraps the workspace in an [`AuthorityWorkspace`][pydantic_ai.workspaces.AuthorityWorkspace]."""

    path_authority: Mapping[str, Authority] | None = None
    """Levels for some directories and files, overriding `authority` for the paths under them; the longest matching
    path wins. Paths are sandbox paths: relative ones resolve against the working directory, and `~` isn't expanded.
    See [`Workspace.authority_of`][pydantic_ai.workspaces.Workspace.authority_of]."""

    _warned_no_tools: bool = field(default=False, init=False, repr=False, compare=False)
    """Whether this instance has already warned that the run has no workspace tools."""

    def __init__(
        self,
        *,
        id: str | None = None,
        description: str | None = None,
        defer_loading: bool = False,
        image: str | modal.Image | None = None,
        app_name: str = DEFAULT_APP_NAME,
        create_app_if_missing: bool = True,
        sandbox_timeout: int = DEFAULT_SANDBOX_TIMEOUT,
        idle_timeout: int | None = None,
        working_dir: str | None = None,
        env: Mapping[str, str] | None = None,
        warn_if_no_tools: bool = True,
        authority: Authority = 'operator',
        path_authority: Mapping[str, Authority] | None = None,
        client: modal.Client | None = None,
    ) -> None:
        reject_defer_loading('ModalSandbox', defer_loading)
        # Checked here rather than when the backend first creates a sandbox, so a bad value fails
        # where it is written instead of at the first workspace operation of some later run.
        if type(sandbox_timeout) is not int or not MIN_SANDBOX_TIMEOUT <= sandbox_timeout <= MAX_SANDBOX_TIMEOUT:
            raise UserError(
                f'sandbox_timeout must be an integer between {MIN_SANDBOX_TIMEOUT} and {MAX_SANDBOX_TIMEOUT} '
                f'seconds, got {sandbox_timeout!r}.'
            )
        check_integer('idle_timeout', idle_timeout, optional=True)
        check_working_dir(working_dir)
        check_path_authority(path_authority)
        self.id = id
        self.description = description
        self.defer_loading = defer_loading
        self.image = image
        self.app_name = app_name
        self.create_app_if_missing = create_app_if_missing
        self.sandbox_timeout = sandbox_timeout
        self.idle_timeout = idle_timeout
        self.working_dir = working_dir
        self.env = env
        self.client = client
        self.warn_if_no_tools = warn_if_no_tools
        self.authority = authority
        self.path_authority = path_authority
        self._warned_no_tools = False

    @classmethod
    def from_spec(
        cls,
        *,
        id: str | None = None,
        description: str | None = None,
        image: str | None = None,
        app_name: str = DEFAULT_APP_NAME,
        create_app_if_missing: bool = True,
        sandbox_timeout: int = DEFAULT_SANDBOX_TIMEOUT,
        idle_timeout: int | None = None,
        working_dir: str | None = None,
        env: Mapping[str, str] | None = None,
        warn_if_no_tools: bool = True,
        authority: Authority = 'operator',
        path_authority: Mapping[str, Authority] | None = None,
        **kwargs: Any,
    ) -> ModalSandbox[Any]:
        """Build from an agent spec, naming the fields a spec can express.

        Core reads this signature to generate the spec's JSON schema. Inherited, core falls back
        to the variadic base `from_spec`, because `__init__`'s `modal.Image` hint is a
        `TYPE_CHECKING` import, and publishes only the bare `'ModalSandbox'`. A spec names an
        image by its registry tag. `**kwargs` reaches
        `__init__` unchanged, so `defer_loading` keeps its refusal.
        """
        return cls(
            id=id,
            description=description,
            image=image,
            app_name=app_name,
            create_app_if_missing=create_app_if_missing,
            sandbox_timeout=sandbox_timeout,
            idle_timeout=idle_timeout,
            working_dir=working_dir,
            env=env,
            warn_if_no_tools=warn_if_no_tools,
            **kwargs,
            authority=authority,
            path_authority=path_authority,
        )

    def backend(self, ref: WorkspaceRef) -> ModalSandboxBackend:
        """Construct a backend for a stored Modal ref without opening the sandbox."""
        return ModalSandboxBackend(ref=ref, working_dir=self.working_dir, env=self.env, client=self.client)

    async def destroy(self, ref: WorkspaceRef) -> None:
        """Terminate a sandbox by ID without running commands or restoring its workspace.

        A sandbox that no longer exists is already gone, so destroying it returns quietly.
        """
        check_provider(ref, 'modal')
        await terminate_sandbox(ref.id, client=self.client)

    def get_workspace(self, ctx: RunContext[AgentDepsT], *, ref: WorkspaceRef | None) -> WorkspaceBackend | None:
        """The backend for this run, in an `AuthorityWorkspace` when `authority` or `path_authority` asks for one."""
        return with_authority(self._run_backend(ctx, ref=ref), self.authority, self.path_authority)

    def _run_backend(self, ctx: RunContext[AgentDepsT], *, ref: WorkspaceRef | None) -> WorkspaceBackend | None:
        """Build a backend without performing Modal I/O."""
        del ctx
        if ref is not None and ref.provider != 'modal':
            return None
        return ModalSandboxBackend(
            ref=ref,
            image=self.image,
            app_name=self.app_name,
            create_app_if_missing=self.create_app_if_missing,
            sandbox_timeout=self.sandbox_timeout,
            idle_timeout=self.idle_timeout,
            working_dir=self.working_dir,
            env=self.env,
            client=self.client,
        )

    async def prepare_tools(self, ctx: RunContext[AgentDepsT], tool_defs: list[ToolDefinition]) -> list[ToolDefinition]:
        # The previous `ModalSandbox` registered its own tools, so `ModalSandbox(image=...)` on its
        # own still builds but now leaves the model without the sandbox. This is the earliest hook
        # that sees the run's tools; it warns once per instance and never changes the tools.
        if (
            self.warn_if_no_tools
            and not self._warned_no_tools
            and not any(
                tool.name == name or tool.name.endswith(f'_{name}')
                for tool in tool_defs
                for name in _WORKSPACE_TOOL_NAMES
            )
        ):
            self._warned_no_tools = True
            agent_name = ctx.agent.name if ctx.agent is not None else None
            warnings.warn(
                _no_workspace_tools_message(agent_name),
                ModalSandboxNoToolsWarning,
                stacklevel=2,
            )
        return tool_defs
