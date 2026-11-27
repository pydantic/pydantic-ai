"""Ask before tool calls that write files or run commands, by permission mode and rules."""

from __future__ import annotations

import re
from collections.abc import Awaitable, Callable, MutableSet, Sequence
from dataclasses import KW_ONLY, dataclass, field, replace
from typing import Any, Literal, NamedTuple, cast

from opentelemetry.trace import NoOpTracer, get_current_span
from pydantic import BaseModel, ValidationError

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.exceptions import ApprovalRequired, SkipToolExecution, UserError
from pydantic_ai.messages import ModelMessage, ToolCallPart, ToolReturn, ToolReturnPart
from pydantic_ai.tools import (
    AgentDepsT,
    DeferredToolApprovalResult,
    DeferredToolRequests,
    DeferredToolResults,
    RunContext,
    ToolApproved,
    ToolDefinition,
    ToolDenied,
    matches_tool_selector,
)
from pydantic_ai_harness._durable import in_durable_context
from pydantic_ai_harness._shell_command import shell_command, shell_command_arg
from pydantic_ai_harness.file_system._tool_kinds import CREATE_DIRECTORY_KIND, EDIT_FILE_KIND, WRITE_FILE_KIND
from pydantic_ai_harness.permissions._rules import PermissionRule, PermissionRuleSpec, PermissionVerdict, decide

PermissionMode = Literal['full', 'auto', 'read-only']
"""How much the agent may do without asking.

- `'full'`: every call runs without asking.
- `'auto'`: file writes and commands run, except commands matching `Permissions.ask_commands`.
- `'read-only'`: every file write and every command asks first.
"""

PermissionAction = Literal['command', 'write']
"""What a call that asks would do: run a shell command, or write a file."""

_WHAT: dict[PermissionAction, str] = {'command': 'command', 'write': 'file write'}

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


class _Decision(NamedTuple):
    verdict: PermissionVerdict
    source: Literal['rule', 'mode', 'approved', 'remembered']
    asked: _Asked | None = None
    reason: str | None = None
    what: str = 'tool call'
    """What a denied call would have done, for the model's denial."""


class _AskedMetadata(BaseModel):
    """What `Permissions` attaches to a call it holds, read back by `permission_requests`."""

    permissions: _Asked


class _Answered(BaseModel):
    remember: bool


class _AnsweredMetadata(BaseModel):
    """What `ask_permissions` attaches to an approved call, read back by `Permissions`."""

    permissions: _Answered


class _Remembered(BaseModel):
    conversation_id: str | None
    subject: str


class _RememberedMetadata(BaseModel):
    """What `Permissions` attaches to the return of a call remembered inside a durable run, read back from history."""

    permissions: _Remembered


class RememberedApproval(NamedTuple):
    """A call approved with `remember=True`: the same tool on the same subject runs without asking in its conversation."""

    conversation_id: str | None
    """The conversation the approval holds for."""

    tool_name: str
    """The tool approved."""

    subject: str
    """The command, or the path written, approved."""


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
    """Why the current mode or a rule asks about this call, for the person deciding."""


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
    """Ask before file writes and shell commands by permission mode, and allow, ask, or deny tool calls by rule.

    A call the mode asks about raises [`ApprovalRequired`][pydantic_ai.exceptions.ApprovalRequired], so it
    joins the run's deferred approval flow: a
    [`HandleDeferredToolCalls`][pydantic_ai.capabilities.HandleDeferredToolCalls] whose handler is
    `approval_ask(approver)` puts it to a person inside the run, and without a handler the run ends with
    it in its [`DeferredToolRequests`][pydantic_ai.tools.DeferredToolRequests] output.

    `rules` decide calls to any tool before the mode does: a matching `'allow'` rule runs the call, `'ask'`
    holds it, and `'deny'` refuses it with a `ToolReturnPart` whose `outcome` is `'denied'`, even when it was
    approved. Calls no rule matches follow the mode.

    The mode checks file writes, the `FileSystem` tools `write_file`, `edit_file`, and `create_directory`,
    recognized by their tool kind whatever they are named, and commands, the tools whose metadata declares
    `code_arg_language='shell'` and a `code_arg_name`, as the `Shell` tools that take a command do. Sub-agents'
    runs this capability does not reach are not checked: pass it in `SubAgents(shared_capabilities=[...])`.

    `mode` can be a function of the run context. It is read on every call, so a mode changed during a run
    applies from the run's next call.
    """

    mode: PermissionMode | Callable[[RunContext[AgentDepsT]], PermissionMode] = 'full'
    """The permission mode, or a function returning it for a run."""

    _: KW_ONLY

    rules: Sequence[PermissionRule[AgentDepsT]] = ()
    """Rules that allow, ask about, or deny matching calls, checked before the mode."""

    ask_commands: Sequence[str] = DEFAULT_ASK_COMMANDS
    """Regular expressions for the commands `'auto'` asks about, searched for anywhere in the command."""

    remembered: MutableSet[RememberedApproval] = field(
        default_factory=set[RememberedApproval], repr=False, compare=False
    )
    """Calls approved with `remember=True`, which run without asking.

    The caller owns it: pass one to share it, bound it, or drop it with the conversation it is for. It is
    in-process memory, so inside a durable workflow or flow it is not used: approvals are remembered in the
    message history instead, in the metadata of the approved call's `ToolReturnPart`.
    """

    _patterns: tuple[re.Pattern[str], ...] = field(init=False, repr=False, compare=False)

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

    @classmethod
    def from_spec(
        cls,
        mode: PermissionMode = 'full',
        *,
        rules: Sequence[PermissionRuleSpec] = (),
        ask_commands: Sequence[str] = DEFAULT_ASK_COMMANDS,
        id: str | None = None,
        description: str | None = None,
        defer_loading: bool = False,
    ) -> Permissions[AgentDepsT]:
        """Build the serializable configuration: a fixed mode, and rules as mappings of `PermissionRule` fields."""
        return cls(
            mode,
            rules=[PermissionRule[AgentDepsT](**rule) for rule in rules],
            ask_commands=ask_commands,
            id=id,
            description=description,
            defer_loading=defer_loading,
        )

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
        """Refuse a call a rule denies, and hold one asked about until it is approved, unless remembered."""
        mode = self.mode_for(ctx)
        decision = self._settle(ctx, call.tool_name, await self._decide(ctx, mode, tool_def=tool_def, args=args))
        _trace(ctx, mode, decision)
        if decision.verdict == 'allow':
            return args
        raise _refusal(decision)

    def _settle(self, ctx: RunContext[AgentDepsT], tool_name: str, decision: _Decision) -> _Decision:
        """Allow a call asked about that was approved, remembering it if asked to, or was remembered before."""
        asked = decision.asked
        if decision.verdict != 'ask':
            return decision
        if ctx.tool_call_approved:
            if asked is not None and not in_durable_context(ctx) and _remember(ctx.tool_call_metadata):
                self.remembered.add(RememberedApproval(ctx.conversation_id, tool_name, asked.subject))
            return decision._replace(verdict='allow', source='approved')
        if asked is not None and self._remembered(ctx, tool_name, asked.subject):
            return decision._replace(verdict='allow', source='remembered')
        return decision

    def _remembered(self, ctx: RunContext[AgentDepsT], tool_name: str, subject: str) -> bool:
        if in_durable_context(ctx):
            # In-process memory would not survive a replay or a move to another worker; history does.
            return _remembered_in(
                ctx.messages, _Remembered(conversation_id=ctx.conversation_id, subject=subject), tool_name
            )
        return RememberedApproval(ctx.conversation_id, tool_name, subject) in self.remembered

    async def after_tool_execute(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: dict[str, Any],
        result: Any,
    ) -> Any:
        """Inside a durable run, record a call approved with `remember=True` in its `ToolReturnPart`'s metadata."""
        if not (ctx.tool_call_approved and in_durable_context(ctx) and _remember(ctx.tool_call_metadata)):
            return result
        asked = (await self._decide(ctx, self.mode_for(ctx), tool_def=tool_def, args=args)).asked
        if asked is None or _nests_tool_return(result):
            return result
        tool_return = cast(ToolReturn[Any], result) if isinstance(result, ToolReturn) else ToolReturn[Any](result)
        existing: object = {} if tool_return.metadata is None else tool_return.metadata
        if not isinstance(existing, dict) or 'permissions' in existing:
            return tool_return
        remembered = _RememberedMetadata(
            permissions=_Remembered(conversation_id=ctx.conversation_id, subject=asked.subject)
        )
        return replace(tool_return, metadata={**existing, **remembered.model_dump()})

    async def _decide(
        self, ctx: RunContext[AgentDepsT], mode: PermissionMode, *, tool_def: ToolDefinition, args: dict[str, Any]
    ) -> _Decision:
        """Whether the rules or the mode allow, ask about, or deny this call, before approvals and memory."""
        command = shell_command(tool_def, args)
        rules = [rule for rule in self.rules if await matches_tool_selector(rule.tools, ctx, tool_def)]
        posix = ctx.workspace.platform != 'windows'
        rule = decide(rules, command=command, command_arg=shell_command_arg(tool_def), args=args, posix=posix)
        if rule is not None and rule.verdict != 'allow':
            subject = _subject(tool_def, args, command)
            if rule.verdict == 'ask':
                return _Decision('ask', 'rule', _rule_asks(rule, subject), rule.reason)
            what = 'tool call' if subject is None else _WHAT[subject[0]]
            return _Decision('deny', 'rule', reason=rule.reason, what=what)
        asked = self._mode_asks(mode, tool_def=tool_def, args=args, command=command, allowed=rule is not None)
        if asked is not None:
            return _Decision('ask', 'mode', asked, asked.reason)
        return _Decision('allow', 'mode') if rule is None else _Decision('allow', 'rule', reason=rule.reason)

    def _mode_asks(
        self,
        mode: PermissionMode,
        *,
        tool_def: ToolDefinition,
        args: dict[str, Any],
        command: str | None,
        allowed: bool,
    ) -> _Asked | None:
        """What the mode asks about a call; a rule's allow skips read-only's blanket ask, not `ask_commands`."""
        if mode == 'full':
            return None
        if tool_def.tool_kind in _WRITE_KINDS:
            # The file tools only reach files inside their root, so `'auto'` lets them write.
            path = args.get('path')
            if mode == 'read-only' and not allowed and isinstance(path, str):
                return _Asked(action='write', subject=path, reason='Read-only mode asks before every file write.')
            return None
        if command is None:
            return None
        if mode == 'read-only' and not allowed:
            return _Asked(action='command', subject=command, reason='Read-only mode asks before every command.')
        if any(pattern.search(command) for pattern in self._patterns):
            label = 'Auto' if mode == 'auto' else 'Read-only'
            return _Asked(
                action='command',
                subject=command,
                reason=f'{label} mode asks before commands that delete, push, install, or reach the network.',
            )
        return None


def _subject(
    tool_def: ToolDefinition, args: dict[str, Any], command: str | None
) -> tuple[PermissionAction, str] | None:
    """Whether a call runs a command or writes a file, and which; `None` for any other call."""
    if command is not None:
        return 'command', command
    path = args.get('path')
    if tool_def.tool_kind in _WRITE_KINDS and isinstance(path, str):
        return 'write', path
    return None


def _rule_asks(rule: PermissionRule[Any], subject: tuple[PermissionAction, str] | None) -> _Asked | None:
    """What a rule asks about a call that runs a command or writes a file; `None` for any other call."""
    if subject is None:
        return None
    reason = rule.reason or f'A permission rule asks before this {_WHAT[subject[0]]}.'
    return _Asked(action=subject[0], subject=subject[1], reason=reason)


def _refusal(decision: _Decision) -> SkipToolExecution | ApprovalRequired:
    """The exception that refuses a call `decision` denies, or holds one it asks about."""
    asked, reason = decision.asked, decision.reason
    if decision.verdict == 'deny':
        message = f'A permission rule denied this {decision.what}, so it did not run. Do not retry it.'
        return SkipToolExecution(ToolDenied(f'{message} Reason: {reason}' if reason else message))
    if asked is None:
        # Not a command or file write, so it takes the plain approval flow, like `requires_approval=True`.
        return ApprovalRequired(metadata={'reason': reason} if reason else None)
    return ApprovalRequired(metadata=_AskedMetadata(permissions=asked).model_dump())


def _trace(ctx: RunContext[AgentDepsT], mode: PermissionMode, decision: _Decision) -> None:
    """Record the decision on the open `execute_tool` span; the reason only with `trace_include_content`."""
    span = get_current_span()
    if isinstance(ctx.tracer, NoOpTracer) or not span.is_recording():
        return
    span.set_attributes(
        {'permissions.decision': decision.verdict, 'permissions.source': decision.source, 'permissions.mode': mode}
    )
    if decision.reason is not None and ctx.trace_include_content:
        span.set_attribute('permissions.reason', decision.reason)


def _nests_tool_return(result: object) -> bool:
    """Whether `result` is a list holding a `ToolReturn`, which core rejects with a clear error."""
    return isinstance(result, list) and any(isinstance(item, ToolReturn) for item in cast(list[object], result))


def _remembered_in(messages: Sequence[ModelMessage], remembered: _Remembered, tool_name: str) -> bool:
    for message in messages:
        for part in message.parts:
            if isinstance(part, ToolReturnPart) and part.tool_name == tool_name:
                try:
                    if _RememberedMetadata.model_validate(part.metadata).permissions == remembered:
                        return True
                except ValidationError:
                    continue
    return False


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
    capability, or pass `approval_ask(approver)` to one as its handler. Approvals other capabilities asked
    for are left for the next handler.
    """
    asked = permission_requests(requests)
    if not asked:
        return None
    approvals: dict[str, bool | DeferredToolApprovalResult] = {}
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
    return DeferredToolResults(approvals=approvals, metadata=metadata)


def approval_ask(
    approver: Approver,
) -> Callable[[RunContext[AgentDepsT], DeferredToolRequests, str], Awaitable[DeferredToolResults]]:
    """A [`HandleDeferredToolCalls`][pydantic_ai.capabilities.HandleDeferredToolCalls] handler that asks `approver`.

    `approver` answers each call `Permissions` holds in place, as `ask_permissions` asks it. Approvals other
    capabilities asked for, and external calls, are left pending for the next handler, or for the run's
    `DeferredToolRequests` output.

    The handler takes a resume token, so a durable run records it, and the durable runs of engines without
    external event support, every engine but AWS Lambda so far, refuse it: there, use the two-parameter
    handler `lambda ctx, requests: ask_permissions(requests, approver)`. On Lambda, the handler runs in a
    step that records the answer, so the approver must answer quickly.
    """

    async def ask(ctx: RunContext[AgentDepsT], requests: DeferredToolRequests, token: str) -> DeferredToolResults:
        # Empty results, not `None`, which would make the run wait for a token no one will resolve.
        return await ask_permissions(requests, approver) or requests.build_results()

    return ask


def denial_message(request: PermissionRequest, *, reason: str | None = None) -> str:
    """What the model is told when `request` is denied, with the person's `reason` when they gave one."""
    message = f'The user denied this {_WHAT[request.action]}, so it did not run.'
    return f'{message} Their reason: {reason}' if reason else message
