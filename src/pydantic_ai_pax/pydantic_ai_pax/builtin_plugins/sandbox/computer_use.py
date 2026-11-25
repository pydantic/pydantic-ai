"""`/computer-use`'s on/off record, `ComputerUse` for desktop sandboxes, and readable terminal rows for it."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Literal, assert_never

from rich.console import RenderableType
from rich.text import Text

from pydantic_ai import (
    AgentStreamEvent,
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    RunContext,
)
from pydantic_ai.capabilities import AbstractCapability, WrapperCapability
from pydantic_ai.computer_use import (
    Click,
    ComputerCallPart,
    ComputerResult,
    ComputerReturnPart,
    ComputerToolAction,
    CursorPosition,
    Drag,
    HoldKeys,
    MouseDown,
    MouseMove,
    MouseUp,
    PressKeys,
    Screenshot,
    Scroll,
    TypeText,
    Wait,
    Zoom,
)
from pydantic_ai.messages import ModelMessage, ModelResponse
from pydantic_ai.workspaces import WorkspaceRef
from pydantic_ai_harness.computer_use import ActionsPerformedEvent, ComputerUse, SupportsDisplay
from pydantic_ai_pax.ui.rendering import theme
from pydantic_ai_pax.ui.rendering.tool_output import terminal_text, tool_header

_TYPE_PREVIEW_CHARS = 40
"""Maximum characters of a `type_text` action shown in a one-line summary."""

COMPUTER_USE = 'clai2_computer_use'
"""`ModelResponse.metadata` key of a sandbox with a desktop: `True` while `/computer-use` is on, `False` once off.

Pax's own, never sent to a model. `/sandbox` and `/computer-use` save it on the response they record, and every
later response in the sandbox carries it on, so the latest response says both, even after compaction.
"""


def computer_use_state(messages: Sequence[ModelMessage]) -> bool | None:
    """`COMPUTER_USE` on the history's latest response: `None` when its sandbox has no desktop."""
    response = next((message for message in reversed(messages) if isinstance(message, ModelResponse)), None)
    state = None if response is None else (response.metadata or {}).get(COMPUTER_USE)
    return state if isinstance(state, bool) else None


def computer_use_on(messages: Sequence[ModelMessage]) -> bool:
    """Whether `/computer-use` is on, according to the history's latest response."""
    return computer_use_state(messages) is True


"""The actions that only look at the screen, which run whatever `/permissions` is."""


FrameCallback = Callable[[WorkspaceRef, bytes, Sequence[ComputerToolAction], ComputerResult | None], None]
"""Called after each `computer` tool result carrying a screenshot."""


@dataclass(frozen=True, kw_only=True)
class ActionMarker:
    """One visual overlay on a desktop screenshot, showing where an action landed."""

    kind: Literal['click', 'move', 'drag']
    points: tuple[tuple[int, int], ...]
    label: str = ''


def _format_click(action: Click) -> str:
    prefix = '+'.join(action.modifiers)
    button = '' if action.button == 'left' else f'{action.button}-'
    verb = {1: 'click', 2: 'double-click', 3: 'triple-click'}.get(action.count, f'click ×{action.count}')
    name = f'{prefix}+{button}{verb}' if prefix else f'{button}{verb}'
    return f'{name} ({action.x}, {action.y})' if action.x is not None and action.y is not None else name


def _format_scroll(action: Scroll) -> str:
    prefix = '+'.join(action.modifiers)
    verb = f'{prefix}+scroll' if prefix else 'scroll'
    at = f' ({action.x}, {action.y})' if action.x is not None and action.y is not None else ''
    return f'{verb} {action.direction} {action.amount}{at}'


def _format_type(action: TypeText) -> str:
    safe = terminal_text(action.text, keep='')
    preview = safe if len(safe) <= _TYPE_PREVIEW_CHARS else f'{safe[: _TYPE_PREVIEW_CHARS - 1]}…'
    return f'type "{preview}"'


def _format_press(action: PressKeys) -> str:
    combo = '+'.join(action.keys)
    return f'press {combo} ×{action.repeat}' if action.repeat > 1 else f'press {combo}'


def _format_duration(seconds: float) -> str:
    return f'{int(seconds)}s' if seconds.is_integer() else f'{seconds:g}s'


def format_action(action: ComputerToolAction) -> str:
    """A concise, terminal-safe summary of one `computer` action."""
    match action:
        case Click():
            return _format_click(action)
        case MouseMove():
            return f'move ({action.x}, {action.y})'
        case Drag():
            start, end = action.path[0], action.path[-1]
            return f'drag ({start.x}, {start.y}) -> ({end.x}, {end.y})'
        case MouseDown():
            return f'mouse-down {action.button}'
        case MouseUp():
            return f'mouse-up {action.button}'
        case Scroll():
            return _format_scroll(action)
        case TypeText():
            return _format_type(action)
        case PressKeys():
            return _format_press(action)
        case HoldKeys():
            return f'hold {"+".join(action.keys)} ({_format_duration(action.seconds)})'
        case Screenshot():
            return 'screenshot'
        case Wait():
            return f'wait {_format_duration(action.seconds)}'
        case Zoom():
            region = action.region
            return f'zoom ({region.left}, {region.top})-({region.right}, {region.bottom})'
        case CursorPosition():
            return 'cursor-position'
        case _:  # pragma: no cover
            assert_never(action)


def format_actions(actions: Sequence[ComputerToolAction]) -> str:
    """One-line summary of a batch of `computer` actions."""
    return ' · '.join(format_action(action) for action in actions)


def action_markers(actions: Sequence[ComputerToolAction]) -> tuple[ActionMarker, ...]:
    """Extract pointer coordinates from `actions` for the live desktop viewer's overlay."""
    markers: list[ActionMarker] = []
    for action in actions:
        match action:
            case Click(x=x, y=y) if x is not None and y is not None:
                markers.append(ActionMarker(kind='click', points=((x, y),), label=_format_click(action)))
            case MouseMove(x=x, y=y):
                markers.append(ActionMarker(kind='move', points=((x, y),), label=f'move ({x}, {y})'))
            case Drag(path=path):
                markers.append(
                    ActionMarker(
                        kind='drag',
                        points=tuple((point.x, point.y) for point in path),
                        label=format_action(action),
                    )
                )
            case _:
                pass
    return tuple(markers)


def render_computer_event(event: AgentStreamEvent, *, live_url: str | None = None) -> RenderableType | None:
    """Render `computer` tool calls and `ActionsPerformedEvent`s as concise transcript rows.

    With `live_url`, an `ActionsPerformedEvent` row ends in a clickable `watch live ↗` link to the viewer.
    """
    if isinstance(event, FunctionToolCallEvent) and isinstance(event.part, ComputerCallPart):
        summary = format_actions(event.part.actions or ())
        return tool_header(name='computer', argument=summary)
    if isinstance(event, ActionsPerformedEvent):
        result = event.result
        muted = theme.color(theme.MUTED)
        row = Text('  ⎿ ', style=muted)
        if error := result.get('error'):
            row.append(terminal_text(error, keep=''), style=theme.color(theme.ERROR))
        else:
            count = result['performed']
            noun = 'action' if count == 1 else 'actions'
            row.append(f'{count} {noun} · screenshot', style=muted)
        if live_url is not None:
            row.append(' · ', style=muted)
            row.append('watch live ↗', style=f'{theme.color(theme.AQUA)} link {live_url}')
        return row
    return None


def history_frames(
    messages: Sequence[ModelMessage],
) -> list[tuple[bytes, Sequence[ComputerToolAction], ComputerResult | None]]:
    """Screenshots and paired actions already recorded in `messages`, oldest first."""
    calls: dict[str, Sequence[ComputerToolAction]] = {}
    frames: list[tuple[bytes, Sequence[ComputerToolAction], ComputerResult | None]] = []
    for message in messages:
        for part in message.parts:
            if isinstance(part, ComputerCallPart) and part.actions is not None:
                calls[part.tool_call_id] = part.actions
            elif isinstance(part, ComputerReturnPart) and (screenshot := part.screenshot) is not None:
                frames.append((screenshot.data, calls.get(part.tool_call_id, ()), part.result))
    return frames


@dataclass
class _ObservedComputerUse(WrapperCapability[None]):
    """Wraps harness `ComputerUse` to hand each screenshot, with the actions that led to it, to `on_frame`."""

    on_frame: FrameCallback = field(repr=False)
    _calls: dict[str, Sequence[ComputerToolAction]] = field(
        default_factory=dict[str, Sequence[ComputerToolAction]], init=False, repr=False
    )

    async def on_event(self, ctx: RunContext[None], *, event: AgentStreamEvent) -> None:
        if isinstance(event, FunctionToolCallEvent) and isinstance(event.part, ComputerCallPart):
            if event.part.actions is not None:
                self._calls[event.part.tool_call_id] = event.part.actions
        elif isinstance(event, FunctionToolResultEvent) and isinstance(event.part, ComputerReturnPart):
            actions = self._calls.pop(event.part.tool_call_id, ())
            if (screenshot := event.part.screenshot) is not None and (ref := ctx.workspace.ref) is not None:
                self.on_frame(ref, screenshot.data, actions, event.part.result)


@dataclass
class DesktopComputerUse(AbstractCapability[None]):
    """Harness `ComputerUse` for a run whose history has `/computer-use` on and whose workspace has a display.

    Does not override `get_workspace`, so Pax attaches the local folder as usual on other runs.
    """

    on_frame: FrameCallback = field(repr=False)
    full_access: Callable[[], bool] = field(repr=False)
    """Whether `/permissions` is `full`, read at each run: like harness on a read-only workspace, no `computer` tool
    otherwise, as the screen can change anything."""
    _computer_use: ComputerUse[None] = field(default_factory=ComputerUse[None], init=False, repr=False)

    async def for_run(self, ctx: RunContext[None]) -> AbstractCapability[None]:
        backend = ctx.workspace.backend
        if (
            computer_use_on(ctx.messages)
            and self.full_access()
            and isinstance(backend, SupportsDisplay)
            and backend.display is not None
        ):
            return _ObservedComputerUse(await self._computer_use.for_run(ctx), on_frame=self.on_frame)
        return AbstractCapability[None]()
