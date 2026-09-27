"""Capability that supplies a Modal sandbox as an agent run's workspace."""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.exceptions import UserError
from pydantic_ai.tools import AgentDepsT, RunContext, ToolDefinition
from pydantic_ai.workspaces import WorkspaceBackend, WorkspaceRef
from typing_extensions import Never

from pydantic_ai_harness._warn import warn_argument_renamed
from pydantic_ai_harness._workspace_provider import check_integer, check_working_dir
from pydantic_ai_harness.modal_sandbox._backend import (
    DEFAULT_APP_NAME,
    DEFAULT_SANDBOX_TIMEOUT,
    ModalSandboxBackend,
)

if TYPE_CHECKING:
    import modal

UPGRADE_DOCS_URL = 'https://pydantic.dev/docs/ai/harness/modal-sandbox/#upgrading-from-the-previous-modalsandbox'

# Constructor arguments of the previous `ModalSandbox`, which registered its own `run_command`,
# `read_file`, `write_file`, and `list_directory` tools, that have no counterpart now that the
# capability only supplies `ctx.workspace`. Each maps to the guidance for moving off it.
_LEGACY_ARGUMENTS: Mapping[str, str] = {
    'sandbox_id': (
        'attach to an existing sandbox per run instead: '
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
        'the ceiling is gone; the sandbox lifetime (`sandbox_timeout`) bounds every command, and the '
        'model-facing default is `Shell(default_timeout=...)`.'
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

_NO_WORKSPACE_TOOLS_MESSAGE = (
    "`ModalSandbox` supplies the Modal sandbox as the run's `ctx.workspace` and registers no tools of its own, "
    'and this run has no `Shell` or `FileSystem` tool. Add `Coder()`, or `Shell()` and/or `FileSystem()`, '
    'alongside it. If your own code or tools use `ctx.workspace`, pass `ModalSandbox(warn_if_no_tools=False)` '
    f'to silence this warning. See {UPGRADE_DOCS_URL}'
)

_warned_no_workspace_tools = False


def _legacy_argument_message(names: list[str]) -> str:
    moves = '\n'.join(f'- `{name}`: {_LEGACY_ARGUMENTS[name]}' for name in names)
    listed = ', '.join(f'`{name}`' for name in names)
    return (
        f"`ModalSandbox` no longer accepts {listed}. It now supplies the Modal sandbox as the run's "
        '`ctx.workspace` and registers no tools of its own; add `Shell()` and/or `FileSystem()` alongside it '
        'to give the model command and file tools that run in the sandbox.\n'
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
    commands run under `sh -c` in the sandbox's shell environment. Constructor
    arguments of the previous `ModalSandbox`, which bundled its own tools, raise a `UserError`
    that says how to express each one now.
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
    """Absolute directory commands start in and relative paths resolve against; the image's when `None`."""

    env: Mapping[str, str] | None = field(default=None, repr=False)
    """Environment variables every command in the sandbox gets; a command's own `env` is layered on top."""

    warn_if_no_tools: bool = True
    """Warn once per process when none of a run's tools has a `Shell` or `FileSystem` tool name.

    Only tool names are checked, so custom tools that reach the sandbox under other names do not count.

    Set it to `False` for agents that reach the sandbox only from their own tools or hooks.
    This flag and its warning go away in the stable harness release.
    """

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
        **legacy: Never,
    ) -> None:
        # Hand-written, with the same parameters a dataclass would generate, so the previous
        # `ModalSandbox` arguments reach a message that says where each one went. `**legacy: Never`
        # keeps the static signature closed.
        if legacy:
            unknown = [argument for argument in legacy if argument not in _LEGACY_ARGUMENTS]
            if unknown:
                raise TypeError(f'ModalSandbox.__init__() got an unexpected keyword argument {unknown[0]!r}')
            raise UserError(_legacy_argument_message(list(legacy)))
        if workdir is not None:
            if working_dir is not None:
                raise UserError('Pass `working_dir` only; `workdir` is its deprecated name.')
            warn_argument_renamed('ModalSandbox', 'workdir', 'working_dir')
            working_dir = workdir
        if defer_loading:
            raise UserError(
                '`defer_loading` is not supported on `ModalSandbox`: a deferred capability is skipped '
                "when the run's workspace is chosen, so it could never supply the sandbox."
            )
        # Checked here rather than when the backend first creates a sandbox, so a bad value fails
        # where it is written instead of at the first workspace operation of some later run.
        check_integer('sandbox_timeout', sandbox_timeout)
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

    def backend(self, ref: WorkspaceRef) -> ModalSandboxBackend:
        """Construct a backend for a stored Modal ref without opening the sandbox."""
        return ModalSandboxBackend(ref=ref, working_dir=self.working_dir, env=self.env)

    async def destroy(self, ref: WorkspaceRef) -> None:
        """Terminate a sandbox by ID without running commands or restoring its workspace."""
        if ref.provider != 'modal':
            raise ValueError(f"unsupported workspace provider {ref.provider!r}; expected 'modal'")
        import modal

        # A dead sandbox cannot be opened for work, but its ID still identifies the
        # resource to terminate. Do not go through backend.get_client() here.
        sandbox = await modal.Sandbox.from_id.aio(ref.id)
        await sandbox.terminate.aio()

    def get_workspace(self, ctx: RunContext[AgentDepsT], *, ref: WorkspaceRef | None) -> WorkspaceBackend | None:
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
        )

    async def prepare_tools(self, ctx: RunContext[AgentDepsT], tool_defs: list[ToolDefinition]) -> list[ToolDefinition]:
        # The previous `ModalSandbox` registered its own tools, so `ModalSandbox(image=...)` on its
        # own still builds but now leaves the model without the sandbox. This is the earliest hook
        # that sees the run's tools; it warns once per process and never changes the tools.
        global _warned_no_workspace_tools
        if (
            self.warn_if_no_tools
            and not _warned_no_workspace_tools
            and not any(
                tool.name == name or tool.name.endswith(f'_{name}')
                for tool in tool_defs
                for name in _WORKSPACE_TOOL_NAMES
            )
        ):
            _warned_no_workspace_tools = True
            warnings.warn(_NO_WORKSPACE_TOOLS_MESSAGE, UserWarning, stacklevel=2)
        return tool_defs
