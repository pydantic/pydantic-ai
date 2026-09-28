"""Capability that supplies a Modal sandbox as an agent run's workspace."""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, cast

from typing_extensions import Never

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.exceptions import UserError
from pydantic_ai.tools import AgentDepsT, RunContext, ToolDefinition
from pydantic_ai.workspaces import WorkspaceBackend, WorkspaceRef
from pydantic_ai_harness._warn import HarnessDeprecationWarning, warn_argument_renamed
from pydantic_ai_harness._workspace_provider import check_integer, check_working_dir
from pydantic_ai_harness.modal_sandbox._backend import (
    DEFAULT_APP_NAME,
    DEFAULT_SANDBOX_TIMEOUT,
    ModalSandboxBackend,
    terminate_sandbox,
)

if TYPE_CHECKING:
    import modal

UPGRADE_DOCS_URL = 'https://pydantic.dev/docs/ai/harness/modal-sandbox/#upgrading-from-the-previous-modalsandbox'

# Constructor arguments of the previous `ModalSandbox`, which registered its own `run_command`,
# `read_file`, `write_file`, and `list_directory` tools, that have no counterpart now that the
# capability only supplies `ctx.workspace`. Each maps to the guidance for moving off it. They are
# accepted and ignored with a deprecation warning, except `sandbox_id`, which still attaches:
# ignoring it would run the agent in a new sandbox instead of the one the user named.
_REMOVED_ARGUMENTS: Mapping[str, str] = {
    'sandbox_id': (
        'still attaches to that sandbox for now, but will be removed. Attach per run instead: '
        "`agent.run(..., workspace=WorkspaceRef(provider='modal', id=sandbox_id))`. "
        'Later runs that continue the message history reattach to it without being told.'
    ),
    'session': (
        '`ModalSandboxSession` no longer exists. To share a sandbox you own across runs, pass '
        '`ModalSandboxBackend(sandbox=<modal.Sandbox>)` (or its `WorkspaceRef`) as `workspace=` to '
        '`agent.run()`. The backend never terminates a sandbox; that stays your job.'
    ),
    'default_command_timeout': (
        'command timeouts belong to the tool that runs commands: use `Shell(default_timeout=...)`.'
    ),
    'max_command_timeout': (
        'there is no direct equivalent, as nothing caps a timeout the model asks for. '
        '`Shell(default_timeout=...)` sets the timeout of commands that do not give one, and `sandbox_timeout` '
        'limits the lifetime of a new sandbox, not of an attached one.'
    ),
    'max_output_bytes': (
        'output limits belong to the tools: use `Shell(max_output_chars=...)`, or `ToolOutputLimits` for any tool.'
    ),
    'max_output_lines': (
        'output limits belong to the tools: use `Shell(max_output_chars=...)`, or `ToolOutputLimits` for any tool.'
    ),
    'max_read_bytes': 'file read limits belong to the tool: use `FileSystem(max_read_lines=..., max_read_chars=...)`.',
    'instructions': (
        'the capability no longer adds instructions; `Shell` and `FileSystem` describe their own tools, and any '
        "further guidance belongs in the agent's `instructions`."
    ),
}


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


def _no_workspace_tools_message(agent_name: str | None, *, attached: bool) -> str:
    # Naming the agent tells the user which of several agents lacks the tools.
    run = 'this run' if agent_name is None else f'this run of agent {agent_name!r}'
    # A `sandbox_id=` sandbox is the user's own, which the previous `ModalSandbox` never terminated either.
    lifetime = '' if attached else f'{_LIFETIME_NOTE} '
    return (
        "`ModalSandbox` supplies the Modal sandbox as the run's `ctx.workspace` and registers no tools of its own, "
        f'and {run} has no `Shell` or `FileSystem` tool. Add `Coder()`, or `Shell()` and/or `FileSystem()`, '
        'alongside it. If your own code or tools use `ctx.workspace`, pass `ModalSandbox(warn_if_no_tools=False)` '
        f'to silence this warning. {lifetime}See {UPGRADE_DOCS_URL}'
    )


def _removed_arguments_message(names: list[str]) -> str:
    moves = '\n'.join(f'- `{name}`: {_REMOVED_ARGUMENTS[name]}' for name in names)
    listed = ', '.join(f'{name}=...' for name in names)
    # `sandbox_id` still attaches, so only say 'ignored' when it is not among them, and the
    # sandbox it names is the user's own, never one the previous `ModalSandbox` terminated.
    attached = 'sandbox_id' in names
    ignored = '' if attached else ' and ignored'
    lifetime = '' if attached else f'{_LIFETIME_NOTE}\n'
    return (
        f"`ModalSandbox({listed})` is deprecated{ignored}. `ModalSandbox` now supplies the Modal sandbox as the run's "
        '`ctx.workspace` and registers no tools of its own; add `Shell()` and/or `FileSystem()` alongside it '
        'to give the model command and file tools that run in the sandbox.\n'
        f'{lifetime}'
        f'{moves}\n'
        f'See {UPGRADE_DOCS_URL}'
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

    warn_if_no_tools: bool = True
    """Warn, once per `ModalSandbox` instance, when none of a run's tools has a `Shell` or `FileSystem` tool name.

    Only tool names are checked, so custom tools that reach the sandbox under other names do not count.

    Set it to `False` for agents that reach the sandbox only from their own tools or hooks.
    This flag and its warning go away in the stable harness release.
    """

    _warned_no_tools: bool = field(default=False, init=False, repr=False, compare=False)
    """Whether this instance has already warned that the run has no workspace tools."""

    _sandbox_id_ref: WorkspaceRef | None = field(default=None, init=False, repr=False)
    """The sandbox the deprecated `sandbox_id=` names, attached when the run has no ref of its own."""

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
        workdir: str | None = None,
        **removed_arguments: Never,
    ) -> None:
        # Hand-written, with the same parameters a dataclass would generate, so the previous
        # `ModalSandbox` arguments reach a message that says where each one went. `**removed_arguments: Never`
        # keeps the static signature closed.
        unknown = [argument for argument in removed_arguments if argument not in _REMOVED_ARGUMENTS]
        if unknown:
            raise TypeError(f'ModalSandbox.__init__() got an unexpected keyword argument {unknown[0]!r}')
        # An explicit `None` (say, from a config file) sets nothing, so there is nothing to move off.
        passed = [argument for argument, value in removed_arguments.items() if value is not None]
        if passed:
            warnings.warn(_removed_arguments_message(passed), HarnessDeprecationWarning, stacklevel=2)
        if workdir is not None:
            if working_dir is not None:
                raise UserError('Pass `working_dir` only; `workdir` is its deprecated name.')
            warn_argument_renamed('ModalSandbox', 'workdir', 'working_dir')
            working_dir = workdir
        if defer_loading:
            raise UserError(
                '`ModalSandbox` does not support `defer_loading=True`: '
                'the workspace is selected before deferred capabilities load.'
            )
        # Checked here rather than when the backend first creates a sandbox, so a bad value fails
        # where it is written instead of at the first workspace operation of some later run.
        if type(sandbox_timeout) is not int or not 10 <= sandbox_timeout <= 86_400:
            raise UserError(
                f'sandbox_timeout must be an integer between 10 and 86400 seconds, got {sandbox_timeout!r}.'
            )
        check_integer('idle_timeout', idle_timeout, optional=True)
        check_working_dir(working_dir)
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
        self.warn_if_no_tools = warn_if_no_tools
        self._warned_no_tools = False
        # `removed_arguments` is typed `Never` to close the signature; its values are whatever the caller passed.
        sandbox_id = cast('str | None', removed_arguments.get('sandbox_id'))
        if sandbox_id == '':
            # Treating it as absent would silently create and bill a new sandbox.
            raise UserError('`sandbox_id` must name a Modal sandbox, got an empty string.')
        self._sandbox_id_ref = WorkspaceRef(provider='modal', id=sandbox_id) if sandbox_id is not None else None

    def backend(self, ref: WorkspaceRef) -> ModalSandboxBackend:
        """Construct a backend for a stored Modal ref without opening the sandbox."""
        return ModalSandboxBackend(ref=ref, working_dir=self.working_dir, env=self.env)

    async def destroy(self, ref: WorkspaceRef) -> None:
        """Terminate a sandbox by ID without running commands or restoring its workspace.

        A sandbox that no longer exists is already gone, so destroying it returns quietly.
        """
        if ref.provider != 'modal':
            raise ValueError(f"unsupported workspace provider {ref.provider!r}; expected 'modal'")
        await terminate_sandbox(ref.id)

    def get_workspace(self, ctx: RunContext[AgentDepsT], *, ref: WorkspaceRef | None) -> WorkspaceBackend | None:
        """Build a backend without performing Modal I/O."""
        del ctx
        if ref is not None and ref.provider != 'modal':
            return None
        return ModalSandboxBackend(
            ref=ref or self._sandbox_id_ref,
            image=self.image,
            app_name=self.app_name,
            create_app_if_missing=self.create_app_if_missing,
            sandbox_timeout=self.sandbox_timeout,
            idle_timeout=self.idle_timeout,
            working_dir=self.working_dir,
            env=self.env,
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
                _no_workspace_tools_message(agent_name, attached=self._sandbox_id_ref is not None),
                ModalSandboxNoToolsWarning,
                stacklevel=2,
            )
        return tool_defs
