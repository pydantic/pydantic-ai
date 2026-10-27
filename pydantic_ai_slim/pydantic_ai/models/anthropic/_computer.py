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
        actions = [action for block in blocks for action in map_computer_member(*computer_member(block, self.version))]
        provider_details: dict[str, Any] = {'computer_blocks': blocks, 'computer_tool': self.version}
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
    input = {key: value for key, value in block['input'].items() if key != 'action'}
    return block['input'].get('action', ''), input


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


def map_computer_member(name: str, input: dict[str, Any]) -> list[dict[str, Any]]:
    """One computer use call as `computer` tool actions; an unknown member fails validation, so the model retries."""
    position = _point(input.get('coordinate'))
    modifiers = [input['text']] if input.get('text') else []  # keys held during a click, move, scroll or drag
    if name in _COMPUTER_CLICKS:
        button, count = _COMPUTER_CLICKS[name]
        return [{'type': 'click', **position, 'button': button, 'count': count, 'modifiers': modifiers}]
    match name:
        case 'mouse_move':
            return [{'type': 'move', **position, 'modifiers': modifiers}]
        case 'left_click_drag':
            # A missing end leaves the path too short to validate, so the model is asked again.
            ends = [input.get('start_coordinate'), input.get('coordinate')]
            path = [_point(point) for point in ends if point is not None]
            return [{'type': 'drag', 'path': path, 'modifiers': modifiers}]
        case 'left_mouse_down':
            return [{'type': 'mouse_down'}]
        case 'left_mouse_up':
            return [{'type': 'mouse_up'}]
        case 'scroll':
            direction = input.get('scroll_direction')
            amount = input.get('scroll_amount')
            return [{'type': 'scroll', **position, 'direction': direction, 'amount': amount, 'modifiers': modifiers}]
        case 'type':
            return [{'type': 'type', 'text': input.get('text', '')}]
        case 'key':
            # xdotool syntax, which Claude writes: chords separated by spaces are pressed one after another.
            chords = str(input.get('text', '')).split() or ['']
            return [{'type': 'keypress', 'keys': [chord], 'repeat': input.get('repeat')} for chord in chords]
        case 'hold_key':
            return [{'type': 'hold_key', 'keys': [input.get('text', '')], 'seconds': input.get('duration')}]
        case 'wait':
            return [{'type': 'wait', 'seconds': input.get('duration')}]
        case 'zoom':
            region = input.get('region')
            if isinstance(region, list) and len(cast(list[Any], region)) == 4:
                left, top, right, bottom = cast(list[Any], region)
                return [{'type': 'zoom', 'region': {'left': left, 'top': top, 'right': right, 'bottom': bottom}}]
            return [{'type': 'zoom', 'region': region}]
        case _:  # `screenshot` and `cursor_position` have the tool's own shape; anything else fails validation.
            return [{**input, 'type': name}]
