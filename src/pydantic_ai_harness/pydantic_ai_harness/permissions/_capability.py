"""Ask before tool calls that write files or run commands, by permission mode and rules."""

from __future__ import annotations

import re
from collections.abc import Awaitable, Callable, MutableSet, Sequence
from dataclasses import KW_ONLY, dataclass, field, replace
from typing import Any, Literal, NamedTuple, cast

from pydantic import BaseModel, ValidationError

from pydantic_ai.capabilities import AbstractCapability, ValidatedToolArgs, WrapperCapability
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
from pydantic_ai_harness.permissions._protected import command_locations, matches, write_locations
from pydantic_ai_harness.permissions._rules import (
    COMMAND_LIMIT,
    PermissionRule,
    PermissionRuleSpec,
    command_matches,
    decide,
)

PermissionMode = Literal['full', 'auto', 'read-only']
"""How much the agent may do without asking.

- `'full'`: every call runs without asking.
- `'auto'`: file writes and commands run, except commands matching `Permissions.ask_commands` and changes to
  `Permissions.protected_paths`.
- `'read-only'`: every file write and every command asks first.
"""

_STRICTNESS: dict[PermissionMode, int] = {'full': 0, 'auto': 1, 'read-only': 2}


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

_PROTECTED_REASONS: dict[PermissionAction, str] = {
    'command': 'This command may change a protected path, so only a person can approve it.',
    'write': 'This file write changes a protected path, so only a person can approve it.',
}


class _Asked(BaseModel):
    action: PermissionAction
    subject: str
    reason: str
    person_only: bool = False


class _PersonOnly(BaseModel):
    person_only: bool = False


class _PersonOnlyMetadata(BaseModel):
    """The `person_only` flag of a call `Permissions` holds, whether as a `PermissionRequest` or a plain approval."""

    permissions: _PersonOnly


class _AskedMetadata(BaseModel):
    """What `Permissions` attaches to a call it holds, read back by `permission_requests`."""

    permissions: _Asked


class _Answered(BaseModel):
    remember: bool


class _AnsweredMetadata(BaseModel):
    """What `ask_permissions` attaches to an approved call, read back by `Permissions`."""

    permissions: _Answered


class _Reviewed(BaseModel):
    approved: bool


class _ReviewedMetadata(BaseModel):
    """What an `ApprovalReviewer` attaches to a call it approved, so `Permissions` can tell it from a person's approval."""

    approval_reviewer: _Reviewed


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

    person_only: bool = False
    """Only a person may approve this call: an `ApprovalReviewer` skips it and no session approval covers it.

    `Permissions` sets it for a call that changes one of its `protected_paths` or runs one of its `protected_commands`.
    """


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

    Outside `'full'`, a call that changes one of `protected_paths`, or runs one of `protected_commands`, always
    asks, whatever the rules allow, and an `ApprovalReviewer` leaves it to the person.

    `max_mode` and `managed_rules` are limits set by whoever configures the agent for the people who run it,
    such as an organization's managed settings: `mode` and `rules`, which the person running it may choose,
    can't loosen them. Every `'deny'`, managed or not, is checked before any `'ask'`, so approving a call
    never runs one a rule denies. A call held by a managed rule or by `max_mode` is `person_only`, like a
    protected change.
    """

    mode: PermissionMode | Callable[[RunContext[AgentDepsT]], PermissionMode] = 'full'
    """The permission mode, or a function returning it for a run."""

    _: KW_ONLY

    rules: Sequence[PermissionRule[AgentDepsT]] = ()
    """Rules that allow, ask about, or deny matching calls, checked before the mode."""

    max_mode: PermissionMode | Callable[[RunContext[AgentDepsT]], PermissionMode] = 'full'
    """The widest mode `mode` may take, or a function returning it for a run: a wider `mode` acts as this one.

    It is checked on its own, before the `'ask'` rules in `rules` and before `mode`: only an `'allow'` in
    `managed_rules` skips the blanket ask of a `'read-only'` `max_mode`, so an `'allow'` in `rules` doesn't lift
    a call over it. What it asks about is `person_only`.
    """

    managed_rules: Sequence[PermissionRule[AgentDepsT]] = ()
    """Rules checked before `rules`, which `rules`, approvals and the mode can't loosen.

    A managed `'deny'` refuses a call even when it was approved. A managed `'ask'` holds a call even when a rule
    allows it, and only a person may approve it (`person_only`). A managed `'allow'` skips the blanket ask of a
    `'read-only'` `max_mode`, but not a `'deny'` or `'ask'` in `rules`, a stricter `mode`, or `protected_paths`.
    A command longer than `COMMAND_LIMIT` is refused while any managed `'deny'` applies to commands, since it is
    too long to check against one.
    """

    ask_commands: Sequence[str] = DEFAULT_ASK_COMMANDS
    """Regular expressions for the commands `'auto'` asks about, searched for anywhere in the command."""

    protected_paths: Sequence[str] = ()
    """Globs for paths only a person may approve a change to, such as the agent's own settings.

    Outside `'full'`, a file write to a matching path asks, and so does a command naming one in any word, a
    redirection target included, unless it only runs programs that read files, such as `cat` or `grep`. A rule's
    allow does not let either run, a rule's deny still refuses it, and an `ApprovalReviewer` leaves it to the
    person, who approves each such call on its own: `PermissionDecision(remember=True)` is ignored for it. A glob
    is matched against the path resolved against the working directory, and a relative glob also
    against its end, so `'.git/hooks/*'` covers every repository's hooks. A folder's glob covers only the folder,
    so list `'dir/*'` too for what is in it. Case is ignored. Commands are checked best effort, by their words,
    split as the shell would: a path a command builds at run time, such as with `cd` first, is not seen.

    A command that can't be checked asks: one over 64 KiB, one nesting `-c` or `eval` scripts more than three deep,
    or one with a word holding more than 16 short options or whose braces expand to more than 64 words.
    """

    protected_commands: Sequence[str] = ()
    """Shell commands, as words like an ask rule's `PermissionRule.command`, held like a change to a protected path.

    Use it for a program that changes what `protected_paths` protects, such as the agent's own settings CLI.
    """

    remembered: MutableSet[RememberedApproval] = field(
        default_factory=set[RememberedApproval], repr=False, compare=False
    )
    """Calls approved with `remember=True`, which run without asking.

    The caller owns it: pass one to share it, bound it, or drop it with the conversation it is for. It is
    in-process memory, so inside a durable workflow or flow it is not used: approvals are remembered in the
    message history instead, in the metadata of the approved call's `ToolReturnPart`.
    """

    _patterns: tuple[re.Pattern[str], ...] = field(init=False, repr=False, compare=False)
    _protected_rules: tuple[PermissionRule[AgentDepsT], ...] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Compile `ask_commands` and `protected_commands`, so a bad one fails when the capability is built."""
        try:
            self._patterns = tuple(re.compile(pattern) for pattern in self.ask_commands)
        except re.error as error:
            raise UserError(f'Permissions.ask_commands has an invalid pattern: {error}') from error
        self._protected_rules = tuple(PermissionRule[AgentDepsT]('ask', command=c) for c in self.protected_commands)

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
        max_mode: PermissionMode = 'full',
        managed_rules: Sequence[PermissionRuleSpec] = (),
        ask_commands: Sequence[str] = DEFAULT_ASK_COMMANDS,
        protected_paths: Sequence[str] = (),
        protected_commands: Sequence[str] = (),
        id: str | None = None,
        description: str | None = None,
        defer_loading: bool = False,
    ) -> Permissions[AgentDepsT]:
        """Build the serializable configuration: fixed modes, and rules as mappings of `PermissionRule` fields."""
        return cls(
            mode,
            rules=[PermissionRule[AgentDepsT](**rule) for rule in rules],
            max_mode=max_mode,
            managed_rules=[PermissionRule[AgentDepsT](**rule) for rule in managed_rules],
            ask_commands=ask_commands,
            protected_paths=protected_paths,
            protected_commands=protected_commands,
            id=id,
            description=description,
            defer_loading=defer_loading,
        )

    def mode_for(self, ctx: RunContext[AgentDepsT]) -> PermissionMode:
        """The permission mode for `ctx`'s run, as it stands now: `mode`, held to `max_mode`."""
        return max(_read(self.mode, ctx), _read(self.max_mode, ctx), key=_STRICTNESS.__getitem__)

    async def after_tool_validate(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: ValidatedToolArgs,
    ) -> ValidatedToolArgs:
        """Hold a call for a person as soon as its arguments are valid, when only a person may approve it.

        This is before core defers a tool registered with `requires_approval=True`, and before any
        `before_tool_execute` hook asks, so whichever mechanism defers the call, its request says that only
        a person may approve it: a protected change, a managed `'ask'`, or a hold of `max_mode`.
        """
        if ctx.tool_call_approved or not _holds_for_a_person(self):
            return args
        try:
            asked = await self._asks(ctx, tool_def=tool_def, args=args)
        except SkipToolExecution:
            # Left to `before_tool_execute`, which refuses it.
            return args
        except ApprovalRequired as held:
            # A managed ask on a tool that neither runs a command nor writes a file.
            if person_only_request(held.metadata):
                raise
            return args
        if asked is not None and asked.person_only:
            raise ApprovalRequired(metadata=_AskedMetadata(permissions=asked).model_dump())
        return args

    async def before_tool_execute(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: dict[str, Any],
    ) -> dict[str, Any]:
        """Refuse a call a rule denies, and hold one asked about until it is approved, unless remembered."""
        asked = await self._asks(ctx, tool_def=tool_def, args=args)
        if asked is None:
            return args
        if ctx.tool_call_approved and asked.person_only and _reviewed(ctx.tool_call_metadata):
            # An `ApprovalReviewer` approved a protected call whose request did not say so: ask a person.
            raise ApprovalRequired(metadata=_AskedMetadata(permissions=asked).model_dump())
        if ctx.tool_call_approved:
            if not in_durable_context(ctx) and not asked.person_only and _remember(ctx.tool_call_metadata):
                self.remembered.add(RememberedApproval(ctx.conversation_id, call.tool_name, asked.subject))
            return args
        # A protected change is approved one call at a time: no remembered approval covers it.
        if not asked.person_only and self._is_remembered(ctx, call.tool_name, asked.subject):
            return args
        raise ApprovalRequired(metadata=_AskedMetadata(permissions=asked).model_dump())

    def _is_remembered(self, ctx: RunContext[AgentDepsT], tool_name: str, subject: str) -> bool:
        if in_durable_context(ctx):
            # In-process memory would not survive a replay or a move to another worker; history does.
            remembered = _Remembered(conversation_id=ctx.conversation_id, subject=subject)
            return _remembered_in(ctx.messages, remembered, tool_name)
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
        asked = await self._asks(ctx, tool_def=tool_def, args=args)
        if asked is None or asked.person_only or _nests_tool_return(result):
            return result
        tool_return = cast(ToolReturn[Any], result) if isinstance(result, ToolReturn) else ToolReturn[Any](result)
        existing: object = {} if tool_return.metadata is None else tool_return.metadata
        if not isinstance(existing, dict) or 'permissions' in existing:
            return tool_return
        remembered = _RememberedMetadata(
            permissions=_Remembered(conversation_id=ctx.conversation_id, subject=asked.subject)
        )
        return replace(tool_return, metadata={**existing, **remembered.model_dump()})

    async def _asks(
        self, ctx: RunContext[AgentDepsT], *, tool_def: ToolDefinition, args: dict[str, Any]
    ) -> _Asked | None:
        """What to ask about this call, or `None` to run it; raises `SkipToolExecution` when a rule denies it."""
        command = shell_command(tool_def, args)
        posix = ctx.workspace.platform != 'windows'
        # A managed deny inside quotes stays a deny: the person it restricts mustn't approve their way past it.
        managed = await _decided(
            ctx, self.managed_rules, tool_def=tool_def, args=args, command=command, posix=posix, strict=True
        )
        rule = await _decided(ctx, self.rules, tool_def=tool_def, args=args, command=command, posix=posix)
        subject = _subject(tool_def, args, command)
        # Every deny before any ask: a person approving a managed ask must not run a call a rule denies.
        if managed is not None and managed.verdict == 'deny':
            return _rule_asks(ctx, managed, subject, managed=True)
        if rule is not None and rule.verdict == 'deny':
            return _rule_asks(ctx, rule, subject)
        if managed is not None and managed.verdict == 'ask':
            return _rule_asks(ctx, managed, subject, managed=True)
        person, ceiling = _read(self.mode, ctx), _read(self.max_mode, ctx)
        mode = max(person, ceiling, key=_STRICTNESS.__getitem__)
        if mode != 'full' and (protected := await self._protected(ctx, subject, posix=posix)) is not None:
            return protected
        # The ceiling before the person's own ask rules, so one of theirs can't drop its hold for a person only. Each
        # layer's own allow skips only its own mode's blanket ask.
        held = self._mode_asks(
            ceiling, tool_def=tool_def, args=args, command=command, allowed=managed is not None, ceiling=True
        )
        if held is not None:
            return held
        if rule is not None and rule.verdict == 'ask':
            return _rule_asks(ctx, rule, subject)
        return self._mode_asks(person, tool_def=tool_def, args=args, command=command, allowed=rule is not None)

    async def _protected(
        self, ctx: RunContext[AgentDepsT], subject: tuple[PermissionAction, str] | None, *, posix: bool
    ) -> _Asked | None:
        if subject is None or not await self._protects(ctx, subject, posix=posix):
            return None
        action, text = subject
        return _Asked(action=action, subject=text, reason=_PROTECTED_REASONS[action], person_only=True)

    async def _protects(
        self, ctx: RunContext[AgentDepsT], subject: tuple[PermissionAction, str], *, posix: bool
    ) -> bool:
        """Whether a file write or command changes one of `protected_paths` or runs one of `protected_commands`."""
        action, text = subject
        if not (self.protected_paths or self.protected_commands):
            return False
        if action == 'command' and (
            # Too long to check in reasonable time: it counts as protected.
            len(text) > COMMAND_LIMIT or command_matches(self._protected_rules, text, posix=posix)
        ):
            return True
        if not self.protected_paths:
            return False
        if action == 'write':
            locations = await write_locations(ctx.workspace, text)
        else:
            # Inside a durable run, expanding `~` or a variable from this process's environment could decide
            # differently on replay: such a command counts as protected.
            found = await command_locations(ctx.workspace, text, posix=posix, expand=not in_durable_context(ctx))
            if found is None:
                return True
            locations = found
        pattern = action == 'command'
        return any(matches(location, self.protected_paths, posix=posix, pattern=pattern) for location in locations)

    def _mode_asks(
        self,
        mode: PermissionMode,
        *,
        tool_def: ToolDefinition,
        args: dict[str, Any],
        command: str | None,
        allowed: bool,
        ceiling: bool = False,
    ) -> _Asked | None:
        """What the mode asks about a call; a rule's allow skips read-only's blanket ask, not `ask_commands`.

        With `ceiling`, `mode` is `max_mode`, so what it asks about is for a person only.
        """
        if mode == 'full':
            return None
        label = f'The maximum permission mode, {mode!r},' if ceiling else f'{mode.capitalize()} mode'
        if tool_def.tool_kind in _WRITE_KINDS:
            # The file tools only reach files inside their root, so `'auto'` lets them write.
            path = args.get('path')
            if mode == 'read-only' and not allowed and isinstance(path, str):
                reason = f'{label} asks before every file write.'
                return _Asked(action='write', subject=path, reason=reason, person_only=ceiling)
            return None
        if command is None:
            return None
        if mode == 'read-only' and not allowed:
            reason = f'{label} asks before every command.'
            return _Asked(action='command', subject=command, reason=reason, person_only=ceiling)
        if any(pattern.search(command) for pattern in self._patterns):
            reason = f'{label} asks before commands that delete, push, install, or reach the network.'
            return _Asked(action='command', subject=command, reason=reason, person_only=ceiling)
        return None


def _read(
    mode: PermissionMode | Callable[[RunContext[AgentDepsT]], PermissionMode], ctx: RunContext[AgentDepsT]
) -> PermissionMode:
    return mode(ctx) if callable(mode) else mode


async def _decided(
    ctx: RunContext[AgentDepsT],
    rules: Sequence[PermissionRule[AgentDepsT]],
    *,
    tool_def: ToolDefinition,
    args: dict[str, Any],
    command: str | None,
    posix: bool,
    strict: bool = False,
) -> PermissionRule[AgentDepsT] | None:
    """The rule among `rules` deciding a call, or `None` when none applies; see `decide` for `strict`."""
    applying = [rule for rule in rules if await matches_tool_selector(rule.tools, ctx, tool_def)]
    return decide(
        applying, command=command, command_arg=shell_command_arg(tool_def), args=args, posix=posix, strict=strict
    )


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


def _rule_asks(
    ctx: RunContext[AgentDepsT],
    rule: PermissionRule[Any],
    subject: tuple[PermissionAction, str] | None,
    *,
    managed: bool = False,
) -> _Asked | None:
    """What an `'ask'` rule asks about a call; raises `SkipToolExecution` for a `'deny'` rule.

    A managed rule's ask is for a person only.
    """
    what = 'tool call' if subject is None else _WHAT[subject[0]]
    source = 'A managed permission rule' if managed else 'A permission rule'
    if rule.verdict == 'deny':
        message = f'{source} denied this {what}, so it did not run. Do not retry it.'
        raise SkipToolExecution(ToolDenied(f'{message} Reason: {rule.reason}' if rule.reason else message))
    if subject is None:
        # Not a command or file write, so it takes the plain approval flow, like `requires_approval=True`.
        if ctx.tool_call_approved and not (managed and _reviewed(ctx.tool_call_metadata)):
            return None
        metadata: dict[str, Any] = {'reason': rule.reason} if rule.reason else {}
        if managed:
            metadata |= _PersonOnlyMetadata(permissions=_PersonOnly(person_only=True)).model_dump()
        raise ApprovalRequired(metadata=metadata or None)
    reason = rule.reason or f'{source} asks before this {what}.'
    return _Asked(action=subject[0], subject=subject[1], reason=reason, person_only=managed)


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


def person_only_request(metadata: object) -> bool:
    """Whether a deferred call's metadata says `Permissions` holds it for a person only to approve."""
    try:
        return _PersonOnlyMetadata.model_validate(metadata).permissions.person_only
    except ValidationError:
        return False


def marks_reviewer_approvals(ctx: RunContext[Any]) -> bool:
    """Whether an `ApprovalReviewer` must mark its approvals in this run, as `Permissions` holds calls for a person.

    The run's own capability tree is searched, so a per-run copy, such as one a delegate inherits, counts. A run
    without such a `Permissions` gets no marker, so its tool calls' metadata stays as it was.
    """
    found = False

    def visit(capability: AbstractCapability[Any]) -> None:
        nonlocal found
        while isinstance(capability, WrapperCapability):
            capability = capability.wrapped
        if isinstance(capability, Permissions):
            found |= _holds_for_a_person(capability)

    # A run always has one; only a context built by hand, as for calling the reviewer directly, has none.
    if ctx.root_capability is not None:  # pragma: no branch
        ctx.root_capability.apply(visit)
    return found


def _holds_for_a_person(permissions: Permissions[Any]) -> bool:
    """Whether `permissions` can hold a call that only a person may approve.

    Protected paths and commands, managed rules, and a `max_mode` other than `'full'` can; a `max_mode` function
    counts too, since what it returns isn't known until a call.
    """
    return bool(permissions.protected_paths or permissions.protected_commands or permissions.managed_rules) or (
        permissions.max_mode != 'full'
    )


def reviewed_approval_metadata() -> dict[str, Any]:
    """The metadata an `ApprovalReviewer` attaches to a call it approves, so `Permissions` won't take it for a person's."""
    return _ReviewedMetadata(approval_reviewer=_Reviewed(approved=True)).model_dump()


def _reviewed(metadata: object) -> bool:
    try:
        return _ReviewedMetadata.model_validate(metadata).approval_reviewer.approved
    except ValidationError:
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
        found.append(
            PermissionRequest(
                call=call,
                action=asked.action,
                subject=asked.subject,
                reason=asked.reason,
                person_only=asked.person_only,
            )
        )
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
