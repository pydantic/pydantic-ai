"""Mapping the `computer` tool to Anthropic's computer use tools, and their calls back to `computer` calls."""

from __future__ import annotations as _annotations

from dataclasses import dataclass
from typing import Any, TypeGuard, cast

from anthropic.types.beta import (
    BetaComputerToolset20260801Param,
    BetaContentBlock,
    BetaToolComputerUse20250124Param,
    BetaToolComputerUse20251124Param,
    BetaToolUnionParam,
    BetaToolUseBlock,
)

from ...computer_use import DISPLAY_SIZE
from ...messages import ToolCallPart
from ...profiles.anthropic import AnthropicComputerToolVersion
from ...tools import ToolDefinition

COMPUTER_TOOL_BETAS: dict[str, str] = {
    'computer_20251124': 'computer-use-2025-11-24',
    'computer_20250124': 'computer-use-2025-01-24',
}
"""The beta header each earlier computer use tool needs."""


@dataclass(frozen=True)
class NativeComputer:
    """The `computer` tool and the Anthropic computer use tool sent in its place."""

    name: str
    version: AnthropicComputerToolVersion

    @property
    def is_toolset(self) -> bool:
        return self.version == 'computer_toolset_20260801'

    def tool_param(self, tool_def: ToolDefinition) -> BetaToolUnionParam:
        if self.is_toolset:
            return BetaComputerToolset20260801Param(type='computer_toolset_20260801')
        width, height = (tool_def.metadata or {})[DISPLAY_SIZE]
        if self.version == 'computer_20251124':
            return BetaToolComputerUse20251124Param(
                type='computer_20251124',
                name='computer',
                display_width_px=width,
                display_height_px=height,
                enable_zoom=True,
            )
        return BetaToolComputerUse20250124Param(
            type='computer_20250124', name='computer', display_width_px=width, display_height_px=height
        )

    def answers(self, block: BetaContentBlock | None) -> TypeGuard[BetaToolUseBlock]:
        """Whether a block is a call of the computer use tool sent for the `computer` tool."""
        if not isinstance(block, BetaToolUseBlock):
            return False
        if self.is_toolset:
            return block.toolset_name == 'computer'
        return block.name == 'computer' and block.toolset_name is None

    def call_part(self, blocks: list[dict[str, Any]], provider_name: str) -> ToolCallPart:
        """Calls of the computer use tool as one `computer` call, keeping them to send back as they were."""
        member_actions = [map_computer_member(*computer_member(block, self.version)) for block in blocks]
        actions = [action for member in member_actions for action in member]
        provider_details: dict[str, Any] = {
            'computer_blocks': blocks,
            'computer_tool': self.version,
            # How many actions each call became, so results are matched to calls without mapping them again.
            COMPUTER_ACTION_COUNTS: [len(member) for member in member_actions],
        }
        return ToolCallPart(
            self.name,
            # A field Claude left out takes the `computer` tool's default.
            {'actions': [{key: value for key, value in action.items() if value is not None} for action in actions]},
            tool_call_id=blocks[0]['id'],
            provider_name=provider_name,
            provider_details=provider_details,
        )


def computer_member(block: dict[str, Any], version: AnthropicComputerToolVersion) -> tuple[str, dict[str, Any]]:
    """A call's member and input: the earlier tools name the member in the input's `action`."""
    if version == 'computer_toolset_20260801':
        return block['name'], block['input']
    args = {key: value for key, value in block['input'].items() if key != 'action'}
    return block['input'].get('action', ''), args


COMPUTER_ACTION_COUNTS = 'computer_action_counts'
"""The `provider_details` key with how many `computer` tool actions each computer use call became."""

RETURNS_OUTPUT = frozenset({'zoom', 'cursor_position'})
"""The calls whose result has content of its own, before the screenshot every result ends with; each is one action."""

COMPUTER_NOT_EXECUTED = 'Not executed: an earlier computer action in this turn failed.'
"""The result Anthropic asks for on each member call after one that failed."""

_COMPUTER_CLICKS = {
    'left_click': ('left', 1),
    'right_click': ('right', 1),
    'middle_click': ('middle', 1),
    'double_click': ('left', 2),
    'triple_click': ('left', 3),
}


def _point(coordinate: Any) -> dict[str, Any]:
    """A toolset `[x, y]` coordinate as `x` and `y`, or nothing when it's absent.

    Anything else goes through as `x` alone, so validation refuses it instead of the action running at the pointer.
    """
    if coordinate is None:
        return {}
    if isinstance(coordinate, list) and len(cast(list[Any], coordinate)) == 2:
        x, y = cast(list[Any], coordinate)
        return {'x': x, 'y': y}
    return {'x': coordinate}


_COMPUTER_MEMBER_FIELDS: dict[str, frozenset[str]] = {
    **{name: frozenset({'coordinate', 'text'}) for name in _COMPUTER_CLICKS},
    'mouse_move': frozenset({'coordinate', 'text'}),
    'left_click_drag': frozenset({'start_coordinate', 'coordinate', 'text'}),
    'left_mouse_down': frozenset(),
    'left_mouse_up': frozenset(),
    'scroll': frozenset({'coordinate', 'scroll_direction', 'scroll_amount', 'text'}),
    'type': frozenset({'text'}),
    'key': frozenset({'text', 'repeat'}),
    'hold_key': frozenset({'text', 'duration'}),
    'wait': frozenset({'duration'}),
    'zoom': frozenset({'region'}),
    'screenshot': frozenset(),
    'cursor_position': frozenset(),
}
"""The input fields of each computer use member that map onto the `computer` tool."""

_COMPUTER_MEMBER_TYPES: dict[str, str] = {
    **dict.fromkeys(_COMPUTER_CLICKS, 'click'),
    'mouse_move': 'move',
    'left_click_drag': 'drag',
    'left_mouse_down': 'mouse_down',
    'left_mouse_up': 'mouse_up',
    'key': 'keypress',
}
"""The `computer` tool action type of each member whose name differs from it."""


def map_computer_member(name: str, args: dict[str, Any]) -> list[dict[str, Any]]:
    """One computer use call as `computer` tool actions, mapped field by field.

    A member or field that doesn't map fails validation, so the model retries; only the names of what didn't map are
    kept, so the call never runs with part of what the model asked for.
    """
    fields = _COMPUTER_MEMBER_FIELDS.get(name)
    unsupported = sorted(args.keys() - fields) if fields is not None else sorted(args.keys())
    if fields is None or unsupported:
        # Under the tool's own type, so the retry names the fields rather than refusing the member.
        return [{'type': _COMPUTER_MEMBER_TYPES.get(name, name), 'unsupported_fields': unsupported}]
    position = _point(args.get('coordinate'))
    modifiers = [args['text']] if args.get('text') else []  # keys held during a click, move, scroll or drag
    if name in _COMPUTER_CLICKS:
        button, count = _COMPUTER_CLICKS[name]
        return [{'type': 'click', **position, 'button': button, 'count': count, 'modifiers': modifiers}]
    match name:
        case 'mouse_move':
            return [{'type': 'move', **position, 'modifiers': modifiers}]
        case 'left_click_drag':
            # A missing end leaves the path too short to validate, so the model is asked again.
            ends = [args.get('start_coordinate'), args.get('coordinate')]
            path = [_point(point) for point in ends if point is not None]
            return [{'type': 'drag', 'path': path, 'modifiers': modifiers}]
        case 'left_mouse_down':
            return [{'type': 'mouse_down'}]
        case 'left_mouse_up':
            return [{'type': 'mouse_up'}]
        case 'scroll':
            direction = args.get('scroll_direction')
            amount = args.get('scroll_amount')
            return [{'type': 'scroll', **position, 'direction': direction, 'amount': amount, 'modifiers': modifiers}]
        case 'type':
            return [{'type': 'type', 'text': args.get('text', '')}]
        case 'key':
            # xdotool syntax, which Claude writes: chords separated by spaces are pressed one after another.
            chords = str(args.get('text', '')).split() or ['']
            return [{'type': 'keypress', 'keys': [chord], 'repeat': args.get('repeat')} for chord in chords]
        case 'hold_key':
            return [{'type': 'hold_key', 'keys': [args.get('text', '')], 'seconds': args.get('duration')}]
        case 'wait':
            return [{'type': 'wait', 'seconds': args.get('duration')}]
        case 'zoom':
            region = args.get('region')
            if isinstance(region, list) and len(cast(list[Any], region)) == 4:
                left, top, right, bottom = cast(list[Any], region)
                return [{'type': 'zoom', 'region': {'left': left, 'top': top, 'right': right, 'bottom': bottom}}]
            return [{'type': 'zoom', 'region': region}]
        case _:  # `screenshot` and `cursor_position`, the only members left
            return [{'type': name}]
