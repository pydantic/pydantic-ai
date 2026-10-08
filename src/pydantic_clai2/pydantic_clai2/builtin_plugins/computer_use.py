"""Let the model see and control this machine's screen, asking before every action.

The built-in `computer_use` plugin, off by default: harness `ComputerUse` on a `LocalComputer`, with
`require_approval` answered by core's `HandleDeferredToolCalls` through the inline picker `ask_user` uses.
"""

from collections.abc import Awaitable, Callable, Sequence

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter
from rich.console import RenderableType
from rich.text import Text

from pydantic_ai import AgentStreamEvent, RunContext
from pydantic_ai.capabilities import AgentCapability, HandleDeferredToolCalls
from pydantic_ai.messages import ToolCallPart
from pydantic_ai.tools import DeferredToolRequests, DeferredToolResults, ToolDenied
from pydantic_ai_harness.ask_user import AskUserRequest, AskUserResponse, Question, QuestionOption
from pydantic_ai_harness.computer_use import (
    ClickAction,
    ComputerAction,
    ComputerActionsEvent,
    ComputerUse,
    DragAction,
    KeypressAction,
    MoveAction,
    ScrollAction,
    TypeAction,
    WaitAction,
)
from pydantic_clai2.builtin_plugins.ask_user_menu import TerminalAnswerer
from pydantic_clai2.plugins import FullScreen, Plugin, PluginHost
from pydantic_clai2.ui.rendering import theme
from pydantic_clai2.ui.rendering.tool_output import terminal_text

TOOL_NAME = 'computer'
ALLOW = 'Allow'
ALLOW_SESSION = 'Allow for this session'
DENY = 'Deny'
INSTALL_HINT = (
    'Computer use needs the `computer-use` extra. Reinstall CLAI with it, for example '
    '`uv tool install "pydantic-clai2[computer-use]"`, then run /plugins enable computer_use.'
)

_ACTIONS = TypeAdapter(list[ComputerAction])
_MAX_INLINE_QUESTION = 300
"""Longer requests are listed above the picker; well inside `ask_user`'s question length limit."""

Answerer = Callable[[AskUserRequest], Awaitable[AskUserResponse]]
Show = Callable[[RenderableType], Awaitable[None]]


class ComputerUseSettings(BaseModel):
    """What a `computer_use` declaration may override."""

    model_config = ConfigDict(extra='forbid', frozen=True, strict=True)
    require_approval: bool = Field(
        default=True, description='Ask before every call that clicks, types, scrolls, or moves the pointer.'
    )
    monitor: int = Field(default=1, ge=1, description='The display to drive: 1 is the primary display.')


def describe_action(action: ComputerAction) -> str:
    """One terminal-safe phrase per action, complete: approval must never hide what it approves."""
    if isinstance(action, ClickAction):
        clicks = {1: 'click', 2: 'double-click', 3: 'triple-click'}[action.count]
        button = '' if action.button == 'left' else f'{action.button}-'
        held = f'{"+".join(action.modifiers)}+' if action.modifiers else ''
        text = f'{held}{button}{clicks} ({action.x}, {action.y})'
    elif isinstance(action, MoveAction):
        text = f'move to ({action.x}, {action.y})'
    elif isinstance(action, DragAction):
        text = 'drag ' + ' to '.join(f'({point.x}, {point.y})' for point in action.path)
    elif isinstance(action, ScrollAction):
        text = f'scroll {action.direction} {action.amount} at ({action.x}, {action.y})'
    elif isinstance(action, TypeAction):
        text = f'type {action.text!r}'
    elif isinstance(action, KeypressAction):
        text = 'press ' + '+'.join(action.keys)
    elif isinstance(action, WaitAction):
        text = f'wait {action.seconds:g}s'
    else:
        text = 'screenshot'
    return terminal_text(text, keep='')


def describe_actions(actions: Sequence[ComputerAction]) -> str:
    """The actions in order, comma separated."""
    return ', '.join(describe_action(action) for action in actions)


def _options() -> tuple[QuestionOption, ...]:
    return (
        QuestionOption(label=ALLOW),
        QuestionOption(label=ALLOW_SESSION, description='Stop asking until CLAI restarts or the plugin reloads'),
        QuestionOption(label=DENY),
    )


class Approver:
    """Answer `computer` approvals one call at a time; other tools' approvals are left for someone else.

    A request short enough to read in the picker is asked inline. A longer one is printed in full
    through `show` first, one numbered action per line, and the question refers to that list, so
    nothing the user approves is cut off. When `full_screen` cannot be entered, as in a headless
    run, nobody can answer and the call is denied.
    """

    def __init__(self, answerer: Answerer, show: Show, full_screen: FullScreen) -> None:
        """`allow_all` starts off for every load, so reloading the plugin asks again."""
        self._answerer = answerer
        self._show = show
        self._full_screen = full_screen
        self.allow_all = False

    async def _can_ask(self) -> bool:
        """Whether the terminal can be borrowed; headless CLAI refuses `full_screen` with `RuntimeError`."""
        try:
            async with self._full_screen():
                pass
        except RuntimeError:
            return False
        return True

    async def __call__(self, ctx: RunContext[None], requests: DeferredToolRequests) -> DeferredToolResults | None:
        """Ask about each `computer` call in turn, or approve it once the user allowed the session."""
        calls = [call for call in requests.approvals if call.tool_name == TOOL_NAME]
        if not calls:
            return None
        approvals: dict[str, bool | ToolDenied] = {}
        for call in calls:
            approvals[call.tool_call_id] = True if self.allow_all else await self._decide(call)
        return requests.build_results(approvals=approvals)

    async def _decide(self, call: ToolCallPart) -> bool | ToolDenied:
        if not await self._can_ask():
            return ToolDenied('Computer actions need approval, which is unavailable in this headless session.')
        actions = _ACTIONS.validate_python(call.args_as_dict().get('actions'))
        question = f'Allow the agent to {describe_actions(actions)}?'
        if len(question) > _MAX_INLINE_QUESTION:
            listing = Text(f'The agent wants to run {len(actions)} computer actions:')
            for number, action in enumerate(actions, start=1):
                listing.append(f'\n  {number}. {describe_action(action)}')
            await self._show(listing)
            question = f'Allow the agent to run the {len(actions)} computer actions listed above?'
        request = AskUserRequest(questions=(Question(header='Computer', question=question, options=_options()),))
        response = await self._answerer(request)
        if response.cancelled or not response.answers:
            return ToolDenied('The user declined this computer action. Ask them how to proceed.')
        [answer] = response.answers
        if answer.custom_answer is not None:
            return ToolDenied(f'The user denied this computer action and said: {answer.custom_answer}')
        if answer.selected == (ALLOW_SESSION,):
            self.allow_all = True
            return True
        if answer.selected == (ALLOW,):
            return True
        return ToolDenied('The user denied this computer action. Ask them how to proceed.')


def render_actions(event: ComputerActionsEvent) -> RenderableType:
    """Record what ran under the `computer` header, since the screenshot itself is not shown."""
    text = Text('  ', style=theme.color(theme.MUTED))
    text.append(describe_actions(event.actions[: event.performed]) or 'nothing ran', style=theme.color(theme.MUTED))
    if event.error is not None:
        text.append('\n  failed: ', style=theme.color(theme.ERROR))
        text.append(terminal_text(event.error, keep=''), style=theme.color(theme.ERROR))
    return text


class ComputerUsePlugin(Plugin[ComputerUseSettings]):
    """`ComputerUse` on this machine's display, inline approvals, and a transcript line per call."""

    def __init__(self, host: PluginHost[None], settings: ComputerUseSettings) -> None:
        super().__init__(host, settings)
        try:
            from pydantic_ai_harness.computer_use import LocalComputer
        except ImportError as exc:
            raise ImportError(INSTALL_HINT) from exc
        self.computer_use = ComputerUse[None](
            computer=LocalComputer(monitor=settings.monitor), require_approval=settings.require_approval
        )
        self.approver = Approver(
            TerminalAnswerer(full_screen=host.full_screen, console=host.console), self._show_listing, host.full_screen
        )

    async def _show_listing(self, listing: RenderableType) -> None:
        async with self.host.full_screen():
            self.host.console.print(listing)

    def get_capabilities(self) -> Sequence[AgentCapability[None]]:
        if not self.settings.require_approval:
            return (self.computer_use,)
        return (self.computer_use, HandleDeferredToolCalls[None](handler=self.approver))

    def render(self, event: AgentStreamEvent) -> RenderableType | None:
        return render_actions(event) if isinstance(event, ComputerActionsEvent) else None
