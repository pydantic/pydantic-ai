"""Send the `computer` tool to Claude as Anthropic's computer use tool, and turn Claude's calls into `computer` calls.

Anthropic's earlier tools, `computer_20250124` and `computer_20251124`, are one tool named `computer`. Each call names
what to do in its `action` field, such as `left_click`. The `computer_toolset_20260801` toolset is a set of tools
instead, one per action, such as a `left_click` tool. This module calls each of those actions a member. Each of
Claude's calls becomes one `computer` call, and each `computer` result goes back as that call's `tool_result`.
"""

from __future__ import annotations as _annotations

import re
from dataclasses import dataclass
from typing import Annotated, Any, TypeGuard

from anthropic.types.beta import (
    BetaComputerToolset20260801Param,
    BetaContentBlock,
    BetaTextBlockParam,
    BetaToolComputerUse20250124Param,
    BetaToolComputerUse20251124Param,
    BetaToolResultBlockParam,
    BetaToolUnionParam,
    BetaToolUseBlock,
)
from pydantic import BaseModel, Field, TypeAdapter, ValidationError
from typing_extensions import TypedDict

from ...computer_use import ComputerCallPart, ComputerReturnPart
from ...messages import ToolCallPart, ToolReturnPart
from ...profiles.anthropic import AnthropicComputerToolVersion
from .._computer_calls import key_from_x11

COMPUTER_TOOL_BETAS: dict[AnthropicComputerToolVersion, str] = {
    'computer_20251124': 'computer-use-2025-11-24',
    'computer_20250124': 'computer-use-2025-01-24',
}
"""The beta header each earlier computer use tool needs."""


class ComputerBlock(TypedDict):
    """One call Claude made to Anthropic's computer use tool, unchanged."""

    id: str
    name: str
    input: dict[str, Any]


class AnthropicComputerReplay(TypedDict):
    """What a `computer` call stores in `provider_details`, so Claude's original call can be sent back unchanged."""

    computer_block: ComputerBlock
    computer_tool_version: AnthropicComputerToolVersion


_REPLAY_ADAPTER = TypeAdapter(AnthropicComputerReplay)


def stored_computer_call(part: ToolCallPart, provider_name: str, sent: NativeComputer) -> ComputerBlock | None:
    """Return Claude's original call behind a `computer` call, or `None` to send it back as a regular tool call.

    The history may come from storage or a UI, so the stored data is validated.

    The part's class isn't checked: a call with an action the `computer` tool doesn't accept becomes a plain
    `ToolCallPart` after a round trip through JSON, and still goes back as Claude's call.

    The original call isn't sent back if a hook changed the part's actions or tool name, or if it came from a
    different tool version than `sent`, since that would tell Claude that its own call ran.
    """
    if part.provider_name != provider_name or part.tool_name != sent.name:
        return None
    try:
        replay = _REPLAY_ADAPTER.validate_python(part.provider_details)
    except ValidationError:
        return None
    block = replay['computer_block']
    if replay['computer_tool_version'] != sent.version:
        return None
    if part.args_as_dict().get('actions') != _call_actions(block, sent.version):
        return None
    return block


def map_computer_result(part: ToolReturnPart, result: BetaToolResultBlockParam) -> BetaToolResultBlockParam:
    """A `computer` call's `tool_result` for a toolset call, which names the toolset it answers.

    A call whose action failed is an error result with only the error's text, as the toolset asks.
    """
    result['toolset_name'] = 'computer'
    if isinstance(part, ComputerReturnPart) and (outcome := part.result) is not None and 'error' in outcome:
        result['is_error'] = True
        result['content'] = [BetaTextBlockParam(type='text', text=outcome['error'])]
    return result


def _call_actions(block: ComputerBlock, version: AnthropicComputerToolVersion) -> list[dict[str, Any]]:
    """The `computer` actions a call becomes, without the fields Claude left out, so the tool's defaults apply."""
    actions = map_member_actions(*member_name_and_input(block, version))
    return [{key: value for key, value in action.items() if value is not None} for action in actions]


@dataclass(frozen=True)
class NativeComputer:
    """The `computer` tool, and the Anthropic computer use tool sent in its place. Built from a `ComputerTool`."""

    name: str
    version: AnthropicComputerToolVersion
    display_size: tuple[int, int]

    @property
    def is_toolset(self) -> bool:
        return self.version == 'computer_toolset_20260801'

    def tool_param(self) -> BetaToolUnionParam:
        if self.is_toolset:
            return BetaComputerToolset20260801Param(type='computer_toolset_20260801')
        width, height = self.display_size
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

    def is_computer_call(self, block: BetaContentBlock | None) -> TypeGuard[BetaToolUseBlock]:
        """Whether `block` is Claude's call to the computer use tool sent in place of the `computer` tool."""
        if not isinstance(block, BetaToolUseBlock):
            return False
        if self.is_toolset:
            return block.toolset_name == 'computer'
        return block.name == 'computer' and block.toolset_name is None

    def map_computer_call(self, block: ComputerBlock, provider_name: str) -> ComputerCallPart:
        """Turn one of Claude's computer use calls into a `computer` call, keeping the call to send back unchanged.

        If the call can't run as Claude asked, its actions include one that the `computer` tool refuses.
        """
        replay = AnthropicComputerReplay(computer_block=block, computer_tool_version=self.version)
        return ComputerCallPart(
            self.name,
            {'actions': _call_actions(block, self.version)},
            tool_call_id=block['id'],
            provider_name=provider_name,
            provider_details=dict(replay),
        )


def member_name_and_input(block: ComputerBlock, version: AnthropicComputerToolVersion) -> tuple[str, dict[str, Any]]:
    """Return the member a call names, and the rest of its input.

    The toolset names the member in the call's tool name. The earlier tools name it in the input's `action` field.
    An `action` that isn't a string names no member and stays in the input, so the conversion reports it as a field
    that doesn't map.
    """
    if version == 'computer_toolset_20260801':
        return block['name'], block['input']
    action = block['input'].get('action', '')
    if not isinstance(action, str):
        return '', block['input']
    return action, {key: value for key, value in block['input'].items() if key != 'action'}


_CLICK_MEMBERS = {
    'left_click': ('left', 1),
    'right_click': ('right', 1),
    'middle_click': ('middle', 1),
    'double_click': ('left', 2),
    'triple_click': ('left', 3),
}


class _MemberInput(BaseModel, extra='forbid', strict=True):
    """A member's input as Anthropic documents it, checked strictly: an unknown field or a number for text fails."""


_Coordinate = Annotated[list[int], Field(min_length=2, max_length=2)]


class _Click(_MemberInput):
    coordinate: _Coordinate | None = None
    text: str | None = None
    """Keys held down during the click. A scroll or a drag takes them the same way."""


class _Move(_MemberInput):
    coordinate: _Coordinate


class _Drag(_MemberInput):
    start_coordinate: _Coordinate
    coordinate: _Coordinate
    text: str | None = None


class _Scroll(_MemberInput):
    coordinate: _Coordinate | None = None
    scroll_direction: str
    scroll_amount: int
    text: str | None = None


class _Type(_MemberInput):
    text: str


class _Key(_MemberInput):
    text: str
    repeat: int | None = None


class _HoldKey(_MemberInput):
    text: str
    duration: float


class _Wait(_MemberInput):
    duration: float


_Region = Annotated[list[int], Field(min_length=4, max_length=4)]


class _Zoom(_MemberInput):
    region: _Region


class _Screenshot(_MemberInput):
    region: _Region | None = None
    """Not documented on `screenshot`, but Claude sends it, in the same frame as `zoom`'s `region`."""


class _ButtonDownOrUp(_MemberInput):
    coordinate: _Coordinate | None = None


class _NoInput(_MemberInput):
    pass


_MEMBER_INPUTS: dict[str, type[_MemberInput]] = {
    **dict.fromkeys(_CLICK_MEMBERS, _Click),
    'mouse_move': _Move,
    'left_click_drag': _Drag,
    'left_mouse_down': _ButtonDownOrUp,
    'left_mouse_up': _ButtonDownOrUp,
    'scroll': _Scroll,
    'type': _Type,
    'key': _Key,
    'hold_key': _HoldKey,
    'wait': _Wait,
    'zoom': _Zoom,
    'screenshot': _Screenshot,
    'cursor_position': _NoInput,
}
"""The input of each member that has a matching `computer` tool action."""


_ACTION_TYPE_BY_MEMBER: dict[str, str] = {
    **dict.fromkeys(_CLICK_MEMBERS, 'click'),
    'left_click_drag': 'drag',
    'left_mouse_down': 'mouse_down',
    'left_mouse_up': 'mouse_up',
    'type': 'type_text',
    'key': 'press_keys',
    'hold_key': 'hold_keys',
}
"""The `computer` tool action type for each member whose name is different."""


def _xy(coordinate: list[int] | None) -> dict[str, int]:
    """A `[x, y]` coordinate as `x` and `y`, or nothing when it's absent."""
    return {} if coordinate is None else {'x': coordinate[0], 'y': coordinate[1]}


_CHORD_SEPARATOR = re.compile(r'(?<=.)\+(?=.)')
"""A `+` between two characters joins keys, as in `ctrl+s`. A lone `+`, or the last one in `ctrl++`, is the plus key."""


def _split_chord(chord: str) -> list[str]:
    """Claude writes a key combination as xdotool does, such as `ctrl+shift+t`, with X11 key names."""
    return [key_from_x11(key) for key in _CHORD_SEPARATOR.split(chord)]


def _modifier_keys(text: str | None) -> list[str]:
    return _split_chord(text) if text else []


def map_member_actions(name: str, args: dict[str, Any]) -> list[dict[str, Any]]:
    """Convert one computer use call into `computer` tool actions.

    A member or field with no matching `computer` action fails validation. The error names only what didn't match.
    The model then retries, so the call never runs halfway.
    """
    action_type = _ACTION_TYPE_BY_MEMBER.get(name, name)
    input_model = _MEMBER_INPUTS.get(name)
    if input_model is None:
        return [{'type': action_type, 'unsupported_fields': sorted(args)}]
    try:
        member = input_model.model_validate(args)
    except ValidationError as error:
        # Keep the member's action type, so the retry message names the bad fields instead of refusing the member.
        fields = sorted({str(detail['loc'][0]) for detail in error.errors() if detail['loc']})
        return [{'type': action_type, 'unsupported_fields': fields}]
    match member:
        case _Click():
            button, count = _CLICK_MEMBERS[name]
            return [
                {
                    'type': 'click',
                    **_xy(member.coordinate),
                    'button': button,
                    'count': count,
                    'modifiers': _modifier_keys(member.text),
                }
            ]
        case _Move():
            return [{'type': 'mouse_move', **_xy(member.coordinate)}]
        case _Drag():
            path = [_xy(member.start_coordinate), _xy(member.coordinate)]
            return [{'type': 'drag', 'path': path, 'modifiers': _modifier_keys(member.text)}]
        case _Scroll():
            return [
                {
                    'type': 'scroll',
                    **_xy(member.coordinate),
                    'direction': member.scroll_direction,
                    'amount': member.scroll_amount,
                    'modifiers': _modifier_keys(member.text),
                }
            ]
        case _Type():
            return [{'type': 'type_text', 'text': member.text}]
        case _Key():
            # Claude writes keys in xdotool syntax: key combinations separated by spaces are pressed one after another.
            chords = member.text.split() or ['']
            return [{'type': 'press_keys', 'keys': _split_chord(chord), 'repeat': member.repeat} for chord in chords]
        case _HoldKey():
            return [{'type': 'hold_keys', 'keys': _split_chord(member.text), 'seconds': member.duration}]
        case _Wait():
            return [{'type': 'wait', 'seconds': member.duration}]
        case _ButtonDownOrUp():
            # Claude's earlier tools press or release the button at a point, so the pointer moves there first.
            move = [{'type': 'mouse_move', **_xy(member.coordinate)}] if member.coordinate else []
            return [*move, {'type': action_type}]
        case _Zoom(region=[left, top, right, bottom]) | _Screenshot(region=[left, top, right, bottom]):
            # A screenshot of a region is a zoom: both capture that part of the screen.
            return [{'type': 'zoom', 'region': {'left': left, 'top': top, 'right': right, 'bottom': bottom}}]
        case _:  # a whole-screen `screenshot`, and `cursor_position`, which takes no input
            return [{'type': action_type}]
