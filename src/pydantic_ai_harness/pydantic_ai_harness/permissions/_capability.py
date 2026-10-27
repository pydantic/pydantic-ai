"""Ask before tool calls that write files or run commands, by permission mode."""

from __future__ import annotations

import re
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import KW_ONLY, dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, ValidationError

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.exceptions import ApprovalRequired, UserError
from pydantic_ai.messages import ToolCallPart
from pydantic_ai.tools import (
    AgentDepsT,
    DeferredToolApprovalResult,
    DeferredToolRequests,
    DeferredToolResults,
    RunContext,
    ToolApproved,
    ToolDefinition,
    ToolDenied,
)
from pydantic_ai_harness.filesystem._tool_kinds import CREATE_DIRECTORY_KIND, EDIT_FILE_KIND, WRITE_FILE_KIND

PermissionMode = Literal['full', 'auto', 'read-only']
"""How much the agent may do without asking.

- `'full'`: every call runs without asking.
- `'auto'`: file writes and commands run, except commands matching `Permissions.ask_commands`.
- `'read-only'`: every file write and every command asks first.
"""

PermissionAction = Literal['command', 'write']
"""What a call that asks would do: run a shell command, or write a file."""

DEFAULT_ASK_COMMANDS: tuple[str, ...] = (
    r'\brm\s+(-\S*[rRf]|--recursive\b|--force\b)',
    r'\bgit\s+push\b',
    r'\bgit\s+(reset\s+--hard|clean)\b',
    r'\bsudo\b',
    r'\b(curl|wget|ssh|scp|sftp|rsync|ftp|telnet|nc|ncat)\b',
    r'\b(pip3?|npm|pnpm|yarn|gem|cargo|brew|apt|apt-get)\s+(install|add)\b',
    r'\buv\s+(add|pip\s+install|tool\s+install)\b',
    r'\b(mkfs|dd)\b',
)
"""Regular expressions for the commands `'auto'` asks about: forced or recursive deletes, `git push` and
history rewrites, `sudo`, network clients, and package installs. Each is searched for anywhere in the command."""

_WRITE_KINDS = frozenset({WRITE_FILE_KIND.id, EDIT_FILE_KIND.id, CREATE_DIRECTORY_KIND.id})


class _Asked(BaseModel):
    action: PermissionAction
    subject: str
    reason: str


class _AskedMetadata(BaseModel):
    """What `Permissions` attaches to a call it holds, read back by `permission_requests`."""

    permissions: _Asked


class _Answered(BaseModel):
    remember: bool


class _AnsweredMetadata(BaseModel):
    """What `ask_permissions` attaches to an approved call, read back by `Permissions`."""

    permissions: _Answered


@dataclass(frozen=True, kw_only=True)
class PermissionRequest:
    """A call `Permissions` holds until a person approves it."""

    call: ToolCallPart
    """The call waiting for approval, with the arguments it will run with."""

    action: PermissionAction
    """Whether the call runs a command or writes a file."""

    subject: str
    """The command, or the path written."""

    reason: str
    """Why the current mode asks about this call, for the person deciding."""


@dataclass(frozen=True, kw_only=True)
class PermissionDecision:
    """The answer to one `PermissionRequest`."""

    approved: bool
    """Run the call, or send the model a denial instead."""

    remember: bool = False
    """With `approved`, also run this tool on the same subject without asking, for the rest of the conversation."""

    reason: str | None = None
    """When denied, the person's reason, sent to the model with the denial."""


Approver = Callable[[PermissionRequest], Awaitable[PermissionDecision]]
"""Asks a person about one `PermissionRequest`, such as with a prompt in a UI."""


@dataclass
class Permissions(AbstractCapability[AgentDepsT]):
    """Ask before tool calls that write files or run shell commands, according to a permission mode.

    A call the mode asks about raises [`ApprovalRequired`][pydantic_ai.exceptions.ApprovalRequired], so it
    joins the run's deferred approval flow: a
    [`HandleDeferredToolCalls`][pydantic_ai.capabilities.HandleDeferredToolCalls] handler that calls
    `ask_permissions` puts it to a person inline, and without one the run ends with it in its
    [`DeferredToolRequests`][pydantic_ai.tools.DeferredToolRequests] output.

    File writes are the `FileSystem` tools `write_file`, `edit_file`, and `create_directory`, recognized by
    their tool kind whatever they are named. Commands are the tools whose metadata declares
    `code_arg_language='shell'`, as the `Shell` tools that take a command do; the command is the
    `code_arg_name` argument; when a wrapper removed that name, it is the `command` argument, or else every string
    argument joined. Other tools are not checked.

    `mode` can be a function of the run context. It is read on every call, so a mode changed during a run
    applies from the run's next call.
    """

    mode: PermissionMode | Callable[[RunContext[AgentDepsT]], PermissionMode] = 'full'
    """The permission mode, or a function returning it for a run."""

    _: KW_ONLY

    ask_commands: Sequence[str] = DEFAULT_ASK_COMMANDS
    """Regular expressions for the commands `'auto'` asks about, searched for anywhere in the command."""

    _patterns: tuple[re.Pattern[str], ...] = field(init=False, repr=False, compare=False)
    _remembered: dict[str | None, set[tuple[str, str]]] = field(
        default_factory=dict[str | None, set[tuple[str, str]]], init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        """Compile `ask_commands`, so a bad pattern fails when the capability is built."""
        try:
            self._patterns = tuple(re.compile(pattern) for pattern in self.ask_commands)
        except re.error as error:
            raise UserError(f'Permissions.ask_commands has an invalid pattern: {error}') from error

    @classmethod
    def get_serialization_name(cls) -> str | None:
        """Return the stable name used in agent specs."""
        return 'Permissions'

    def mode_for(self, ctx: RunContext[AgentDepsT]) -> PermissionMode:
        """The permission mode for `ctx`'s run, as it stands now."""
        return self.mode(ctx) if callable(self.mode) else self.mode

    async def before_tool_execute(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: dict[str, Any],
    ) -> dict[str, Any]:
        """Hold a call the mode asks about until it is approved, unless approved for the conversation already."""
        asked = self._asks(ctx, tool_def=tool_def, args=args)
        if asked is None:
            return args
        key = (call.tool_name, asked.subject)
        if ctx.tool_call_approved:
            if _remember(ctx.tool_call_metadata):
                self._remembered.setdefault(ctx.conversation_id, set()).add(key)
            return args
        if key in self._remembered.get(ctx.conversation_id, ()):
            return args
        raise ApprovalRequired(metadata=_AskedMetadata(permissions=asked).model_dump())

    def _asks(self, ctx: RunContext[AgentDepsT], *, tool_def: ToolDefinition, args: dict[str, Any]) -> _Asked | None:
        """What to ask about this call in the current mode, or `None` to run it."""
        mode = self.mode_for(ctx)
        if mode == 'full':
            return None
        if tool_def.tool_kind in _WRITE_KINDS:
            # The file tools only reach files inside their root, so `'auto'` lets them write.
            path = args.get('path')
            if mode == 'read-only' and isinstance(path, str):
                return _Asked(action='write', subject=path, reason='Read-only mode asks before every file write.')
            return None
        command = _command(tool_def, args)
        if command is None:
            return None
        if mode == 'read-only':
            return _Asked(action='command', subject=command, reason='Read-only mode asks before every command.')
        if any(pattern.search(command) for pattern in self._patterns):
            return _Asked(
                action='command',
                subject=command,
                reason='Auto mode asks before commands that delete, push, install, or reach the network.',
            )
        return None


def _command(tool_def: ToolDefinition, args: dict[str, Any]) -> str | None:
    """The shell command a call runs, when its tool declares one."""
    metadata = tool_def.metadata or {}
    if metadata.get('code_arg_language') != 'shell':
        return None
    name = metadata.get('code_arg_name')
    if isinstance(name, str):
        command = args.get(name)
        return command if isinstance(command, str) else None
    # A host that folds the tool into code mode drops `code_arg_name` and keeps the language. Not knowing which
    # argument is the command, check them all, so a tool whose command argument has another name is still held.
    command = args.get('command')
    if isinstance(command, str):
        return command
    return ' '.join(value for value in args.values() if isinstance(value, str)) or None


def _remember(metadata: object) -> bool:
    try:
        return _AnsweredMetadata.model_validate(metadata).permissions.remember
    except ValidationError:
        return False


def permission_requests(requests: DeferredToolRequests) -> list[PermissionRequest]:
    """The approvals in `requests` that `Permissions` holds, in order; approvals others asked for are left out."""
    found: list[PermissionRequest] = []
    for call in requests.approvals:
        try:
            asked = _AskedMetadata.model_validate(requests.metadata.get(call.tool_call_id)).permissions
        except ValidationError:
            continue
        found.append(PermissionRequest(call=call, action=asked.action, subject=asked.subject, reason=asked.reason))
    return found


async def ask_permissions(requests: DeferredToolRequests, approver: Approver) -> DeferredToolResults | None:
    """Put each call `Permissions` holds in `requests` to `approver`, one at a time; `None` when it holds none.

    Call it from the handler of a [`HandleDeferredToolCalls`][pydantic_ai.capabilities.HandleDeferredToolCalls]
    capability. Approvals other capabilities asked for are left for the next handler.
    """
    asked = permission_requests(requests)
    if not asked:
        return None
    approvals: dict[str, DeferredToolApprovalResult] = {}
    metadata: dict[str, dict[str, Any]] = {}
    for request in asked:
        decision = await approver(request)
        call_id = request.call.tool_call_id
        if not decision.approved:
            approvals[call_id] = ToolDenied(denial_message(request, reason=decision.reason))
            continue
        approvals[call_id] = ToolApproved()
        if decision.remember:
            metadata[call_id] = _AnsweredMetadata(permissions=_Answered(remember=True)).model_dump()
    return DeferredToolResults(approvals={**approvals}, metadata=metadata)


def denial_message(request: PermissionRequest, *, reason: str | None = None) -> str:
    """What the model is told when `request` is denied, with the person's `reason` when they gave one."""
    what = 'command' if request.action == 'command' else 'file write'
    message = f'The user denied this {what}, so it did not run.'
    return f'{message} Their reason: {reason}' if reason else message
