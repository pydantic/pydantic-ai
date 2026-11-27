"""The `run.permissions` mode: harness `Permissions` on every run, and its approvals asked as questions.

A held call is put to the user as an `ask_user` question with three choices, so the terminal picker and
the browser's question card both answer it; a typed answer denies the call with that reason. Whichever
plugin answers questions binds `approval_handler` with its answerer, an `ApprovalHandler` whose handler is
harness `approval_ask`. A call no handler answered, as in a session no one can be asked in, is denied by
`DenyUnanswered`. A `SubAgents` delegation that the session's `Permissions` would not reach is refused outside
`full` by `KeepDelegatesChecked`. Outside `full`, a change to Pax's own settings always asks: see `protected_paths`.
"""

import difflib
import glob
import unicodedata
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ValidationError

from pydantic_ai.capabilities import AbstractCapability, AgentCapability, HandleDeferredToolCalls, WrapperCapability
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
    approval_ask,
    permission_requests,
)
from pydantic_ai_harness.sub_agents import SELF_AGENT_NAME, SubAgents
from pydantic_ai_pax.cli.command_context import CommandContext
from pydantic_ai_pax.config import PERMISSION_MODES, PERMISSIONS_SETTING as SETTING
from pydantic_ai_pax.config.project_settings import LEGACY_PROJECT_DIR, PROJECT_DIR
from pydantic_ai_pax.config.settings_store import SettingsStore
from pydantic_ai_pax.paths import LEGACY_DIRECTORY_NAME, config_dir
from pydantic_ai_pax.runtime.task_prompts import is_delegate_tool

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
PROTECTED = "This may change Pax's own settings, which only you can approve."
"""The reason a call `Permissions` holds for `protected_paths` or `PROTECTED_COMMANDS` gives in its question."""

_PROGRAMS = ('pax', 'pydantic-ai-pax', 'clai2', 'pydantic-clai2', 'pydantic_ai_pax', 'pydantic_clai2')
"""The names `pax` runs under: its commands, the deprecated ones, and their modules for `python -m`."""

PROTECTED_COMMANDS = (
    *(
        f'{program} {change}'
        for program in _PROGRAMS
        for change in (
            'config set',
            'config reset',
            'plugins add',
            'plugins enable',
            'plugins disable',
            'plugins remove',
        )
    ),
    # It can set `core.hooksPath`, which points a repository's hooks at any folder, without naming `.git/config`.
    'git config',
)
"""The `pax` subcommands that change its settings, and `git config`, which ask outside `full` like a write to them."""
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


def protected_paths(store: SettingsStore | None = None) -> tuple[str, ...]:
    """Globs for Pax's own settings, which the agent may not change without asking outside `full`.

    The user's Pax folder (its settings database, credential files, and plugins folder) under its name and its
    name before the rename, the `store`'s settings database and plugins folder when they are elsewhere, every
    project's `.pax/` and `.clai/` folders, and every repository's `.git/hooks/` and `.git/config`, each as written
    and resolved.
    """
    folder = config_dir()
    directories = [folder, folder.with_name(LEGACY_DIRECTORY_NAME)]
    if store is not None:
        directories.append(store.plugins_dir)
    patterns: list[str] = []
    for directory in directories:
        for form in _spellings(directory):
            patterns += [form, f'{form}/*']
    if store is not None:
        # The database's `-wal`, `-shm`, and `-journal` files too.
        patterns += [f'{form}*' for form in _spellings(store.path)]
    for name in (PROJECT_DIR.name, LEGACY_PROJECT_DIR.name, '.git/hooks'):
        patterns += [name, f'{name}/*']
    # `core.hooksPath` in a repository's config can point its hooks at any folder, around `.git/hooks/`.
    patterns.append('.git/config')
    return tuple(dict.fromkeys(patterns))


def _spellings(path: Path) -> list[str]:
    """`path` as written and with its links resolved, escaped for a glob."""
    return list(dict.fromkeys(glob.escape(str(form)) for form in (path.absolute(), path.resolve())))


def session_permissions(
    mode: Callable[[], PermissionMode], *, store: SettingsStore | None = None
) -> 'tuple[Permissions[AgentDepsT], DenyUnanswered[AgentDepsT], KeepDelegatesChecked[AgentDepsT]]':
    """What a session binds: `Permissions` reading `mode` on each call, and its two backstops.

    Outside `full`, `Permissions` asks before every change to `protected_paths(store)` and every
    `PROTECTED_COMMANDS` command. The runtime keeps them for the session, so approvals remembered for its
    conversations end with it. It binds the `DenyUnanswered` backstop only to runs no one can be asked in: see
    `unattended_only`.
    """
    permissions = Permissions[AgentDepsT](
        lambda ctx: mode(), protected_paths=protected_paths(store), protected_commands=PROTECTED_COMMANDS
    )
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

    A delegate runs what is bound to its agent plus the `SubAgents` `shared_capabilities`. Pax binds its
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
    return ApprovalHandler[AgentDepsT](approval_ask(approve))


def unattended_only(
    permissions: Sequence[AgentCapability[AgentDepsT]], supplied: Sequence[AgentCapability[AgentDepsT]]
) -> tuple[AgentCapability[AgentDepsT], ...]:
    """`permissions` without the session's `DenyUnansweredInSession` when `supplied` has an `ApprovalHandler`.

    An `ApprovalHandler` inside a `WrapperCapability`, such as the `PluginGuard` around a settings-built plugin
    capability, counts too. An attended session leaves calls nobody handles, and questions, as they were,
    instead of denying them.
    """
    capabilities: list[AbstractCapability[AgentDepsT]] = [
        capability for capability in supplied if isinstance(capability, AbstractCapability)
    ]
    if not any(_is_approval_handler(capability) for capability in capabilities):
        return tuple(permissions)
    return tuple(capability for capability in permissions if not isinstance(capability, DenyUnansweredInSession))


def _is_approval_handler(capability: AbstractCapability[AgentDepsT]) -> bool:
    if isinstance(capability, ApprovalHandler):
        return True
    return isinstance(capability, WrapperCapability) and _is_approval_handler(capability.wrapped)


def approval_question(request: PermissionRequest) -> AskUserRequest:
    """`request` as a question: approve once, approve for the session, or deny; a typed answer denies with it.

    A change to Pax's own settings (`request.person_only`) is approved one call at a time, so it offers no
    approval for the session.
    """
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
                    *(
                        ()
                        if request.person_only
                        else (
                            QuestionOption(
                                label=APPROVE_SESSION, description=f'Run it on this {noun} without asking again.'
                            ),
                        )
                    ),
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


def approval_reason(request: PermissionRequest) -> str:
    """Why `request` asks, as the terminal and the web UI show it: `PROTECTED` for a change to Pax's own settings."""
    return PROTECTED if request.person_only else request.reason


def _question_text(request: PermissionRequest) -> str:
    reason = approval_reason(request)
    if request.action == 'command':
        return _printable(f'{reason} Run this command?\n{request.subject}')
    return _printable(f'{reason} Write {request.subject}?')


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
