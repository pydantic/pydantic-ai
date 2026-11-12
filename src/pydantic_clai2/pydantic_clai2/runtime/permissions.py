"""The `run.permissions` mode: harness `Permissions` on every run, and its approvals asked as questions.

A held call is put to the user as an `ask_user` question with three choices, so the terminal picker and
the browser's question card both answer it; a typed answer denies the call with that reason. Whichever
plugin answers questions binds `approval_handler` with its answerer. A call no handler answered, as in a
session no one can be asked in, is denied by `DenyUnanswered`. A `SubAgents` delegation that the session's
`Permissions` would not reach is refused outside `full` by `KeepDelegatesChecked`.
"""

import difflib
import unicodedata
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ValidationError

from pydantic_ai.capabilities import AbstractCapability, AgentCapability, HandleDeferredToolCalls
from pydantic_ai.exceptions import SkipToolExecution
from pydantic_ai.messages import ToolCallPart
from pydantic_ai.tools import (
    AgentDepsT,
    DeferredToolRequests,
    DeferredToolResults,
    RunContext,
    ToolDefinition,
    ToolDenied,
)
from pydantic_ai_harness.ask_user import AskUserRequest, AskUserResponse, Question, QuestionOption
from pydantic_ai_harness.deny_unanswered import DenyUnanswered
from pydantic_ai_harness.permissions import (
    Approver,
    PermissionDecision,
    PermissionMode,
    PermissionRequest,
    Permissions,
    ask_permissions,
    permission_requests,
)
from pydantic_ai_harness.subagents import SELF_AGENT_NAME, SubAgents
from pydantic_clai2.cli.command_context import CommandContext
from pydantic_clai2.config import PERMISSION_MODES, PERMISSIONS_SETTING as SETTING
from pydantic_clai2.runtime.task_prompts import is_delegate_tool

MODES: dict[PermissionMode, str] = {
    'full': 'Full access: everything runs without asking.',
    'auto': 'Auto: file edits and commands run; commands that delete, push, install, or reach the network ask first.',
    'read-only': 'Read only: every file write and command asks first.',
}

HEADER = 'Approve?'
APPROVE = 'Approve'
APPROVE_SESSION = 'Approve for this session'
DENY = 'Deny'
UNANSWERED = 'No one could be asked to approve it here; /permissions full lets such calls run.'
_QUESTION_LENGTH = 500
"""What an `ask_user` question may hold; a longer command is cut, so the terminal prints it in full first."""


def permissions_command(context: CommandContext, args: list[str]) -> str:
    """Show this session's permission mode, or set it; a running turn's next tool call follows it."""
    if len(args) > 1 or (args and args[0] not in PERMISSION_MODES):
        raise ValueError(f'Usage: /permissions [{"|".join(PERMISSION_MODES)}]')
    if args:
        context.set_setting([SETTING, args[0]])
        return f'Permissions: {MODES[context.settings.permissions]}'
    choices = '\n'.join(f'  /permissions {mode}  {text}' for mode, text in MODES.items())
    return f'Permissions: {MODES[context.settings.permissions]}\n{choices}'


def session_permissions(
    mode: Callable[[], PermissionMode],
) -> 'tuple[Permissions[AgentDepsT], DenyUnanswered[AgentDepsT], KeepDelegatesChecked[AgentDepsT]]':
    """What a session binds: `Permissions` reading `mode` on each call, and its two backstops.

    The runtime keeps them for the session, so approvals remembered for its conversations end with it. It binds the
    `DenyUnanswered` backstop only to runs no one can be asked in: see `unattended_only`.
    """
    permissions = Permissions[AgentDepsT](lambda ctx: mode())
    return (
        permissions,
        DenyUnansweredInSession[AgentDepsT](),
        KeepDelegatesChecked[AgentDepsT](permissions=permissions),
    )


@dataclass(kw_only=True)
class DenyUnansweredInSession(DenyUnanswered[AgentDepsT]):
    """`DenyUnanswered` that tells the model `/permissions full` lets a call `Permissions` held run.

    Other held approvals keep `message`: `/permissions` only changes what `Permissions` holds.
    """

    async def handle_deferred_tool_calls(
        self, ctx: RunContext[AgentDepsT], *, requests: DeferredToolRequests
    ) -> DeferredToolResults | None:
        results = await super().handle_deferred_tool_calls(ctx, requests=requests)
        if results is not None:
            for held in permission_requests(requests):
                results.approvals[held.call.tool_call_id] = ToolDenied(UNANSWERED)
        return results


@dataclass
class KeepDelegatesChecked(AbstractCapability[AgentDepsT]):
    """Refuse a `SubAgents` delegation, outside `full`, whose delegate would run without the session's `Permissions`.

    A delegate runs what is bound to its agent plus the `SubAgents` `shared_capabilities`. CLAI binds its
    plugins, `Permissions` included, to its own agent, so a delegation to `self` carries it; an agent you
    supply gets them per run, and a named sub-agent runs its own agent, so neither does.
    """

    permissions: Permissions[AgentDepsT]

    async def before_tool_execute(
        self, ctx: RunContext[AgentDepsT], *, call: ToolCallPart, tool_def: ToolDefinition, args: dict[str, Any]
    ) -> dict[str, Any]:
        mode = self.permissions.mode_for(ctx)
        if mode == 'full' or not is_delegate_tool(ctx, tool_def) or self._carried(ctx, tool_def, args):
            return args
        raise SkipToolExecution(
            f'Delegation is off while /permissions is {mode}: this delegate would run without the permission '
            'checks. Do the work here instead; /permissions full lets delegates run.'
        )

    def _carried(self, ctx: RunContext[AgentDepsT], tool_def: ToolDefinition, args: dict[str, Any]) -> bool:
        owner = ctx.capabilities[tool_def.capability_id or '']
        assert isinstance(owner, SubAgents)
        if any(shared is self.permissions for shared in owner.shared_capabilities):
            return True
        if args.get('agent_name') != SELF_AGENT_NAME or ctx.agent is None:
            return False
        bound: list[AbstractCapability[Any]] = []
        ctx.agent.root_capability.apply(bound.append)
        return any(capability is self.permissions for capability in bound)


class ApprovalHandler(HandleDeferredToolCalls[AgentDepsT]):
    """Answers the calls `Permissions` holds by asking someone; a session that binds one is attended."""


def approval_handler(approve: Approver) -> ApprovalHandler[AgentDepsT]:
    """Answer the calls `Permissions` holds with `approve`, leaving other deferred calls to later handlers."""

    async def handle(ctx: RunContext[AgentDepsT], requests: DeferredToolRequests) -> DeferredToolResults | None:
        return await ask_permissions(requests, approve)

    return ApprovalHandler[AgentDepsT](handler=handle)


def unattended_only(
    permissions: Sequence[AgentCapability[AgentDepsT]], bound: Sequence[AgentCapability[AgentDepsT]]
) -> tuple[AgentCapability[AgentDepsT], ...]:
    """`permissions` without the session's `DenyUnanswered` when `bound` has an `ApprovalHandler` to ask someone.

    An attended session leaves calls nobody handles, and questions, as they were, instead of denying them.
    """
    if not any(isinstance(capability, ApprovalHandler) for capability in bound):
        return tuple(permissions)
    return tuple(capability for capability in permissions if not isinstance(capability, DenyUnanswered))


def approval_question(request: PermissionRequest) -> AskUserRequest:
    """`request` as a question: approve once, approve for the session, or deny; a typed answer denies with it."""
    text = _question_text(request)
    if len(text) > _QUESTION_LENGTH:
        text = text[: _QUESTION_LENGTH - 1] + '…'
    noun = 'command' if request.action == 'command' else 'file'
    return AskUserRequest(
        id=request.call.tool_call_id,
        questions=(
            Question(
                header=HEADER,
                question=text,
                options=(
                    QuestionOption(label=APPROVE, description='Run it this once.'),
                    QuestionOption(label=APPROVE_SESSION, description=f'Run it on this {noun} without asking again.'),
                    QuestionOption(
                        label=DENY, description='Do not run it. Type a reason instead to send it to the model.'
                    ),
                ),
            ),
        ),
    )


def shown_in_full(request: PermissionRequest) -> bool:
    """Whether `approval_question` shows all of `request`'s command or path, rather than cutting it."""
    return len(_question_text(request)) <= _QUESTION_LENGTH


def _question_text(request: PermissionRequest) -> str:
    if request.action == 'command':
        return _printable(f'{request.reason} Run this command?\n{request.subject}')
    return _printable(f'{request.reason} Write {request.subject}?')


def decision(response: AskUserResponse) -> PermissionDecision:
    """What the user decided: a declined question denies, and a typed answer denies with it as the reason."""
    answer = response.answers[0] if response.answers and not response.cancelled else None
    if answer is None:
        return PermissionDecision(approved=False)
    if answer.custom_answer is not None:
        return PermissionDecision(approved=False, reason=answer.custom_answer)
    picked = answer.selected[0] if answer.selected else DENY
    return PermissionDecision(approved=picked != DENY, remember=picked == APPROVE_SESSION)


class _Edit(BaseModel):
    old_text: str
    new_text: str


class _Write(BaseModel):
    """The text arguments of `write_file` and `edit_file`, whichever a call has."""

    content: str | None = None
    old_text: str | None = None
    new_text: str | None = None
    replacements: list[_Edit] | None = None


def proposed_diff(request: PermissionRequest) -> str | None:
    """The change a held file write would make, as a unified diff of its arguments; `None` for anything else."""
    if request.action != 'write':
        return None
    try:
        write = _Write.model_validate(request.call.args_as_dict())
    except ValidationError:
        return None
    pairs = [('', write.content)] if write.content is not None else []
    if write.old_text is not None and write.new_text is not None:
        pairs.append((write.old_text, write.new_text))
    pairs += [(edit.old_text, edit.new_text) for edit in write.replacements or ()]
    path = request.subject
    diff = ''.join(
        line
        for old, new in pairs
        for line in difflib.unified_diff(
            old.splitlines(keepends=True), new.splitlines(keepends=True), f'a/{path}', f'b/{path}', n=2
        )
    )
    return diff or None


def _printable(text: str) -> str:
    """`text` with only what a question accepts: printable characters, spaces, and line breaks; tabs become spaces."""
    text = text.replace('\t', '    ')
    return ''.join(char for char in text if char == '\n' or char.isprintable() or unicodedata.category(char) == 'Zs')
