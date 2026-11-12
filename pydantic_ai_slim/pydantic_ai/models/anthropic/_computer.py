"""Mapping the `computer` tool to Anthropic's computer use tools, and their calls back to `computer` calls."""

from __future__ import annotations as _annotations

from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from typing import Annotated, Any, TypeGuard

from anthropic.types.beta import (
    BetaComputerToolset20260801Param,
    BetaContentBlock,
    BetaToolComputerUse20250124Param,
    BetaToolComputerUse20251124Param,
    BetaToolUnionParam,
    BetaToolUseBlock,
)
from pydantic import BaseModel, Field, TypeAdapter, ValidationError
from typing_extensions import TypedDict

from ...computer_use import ComputerCallPart
from ...messages import ToolCallPart
from ...profiles.anthropic import AnthropicComputerToolVersion

COMPUTER_TOOL_BETAS: dict[AnthropicComputerToolVersion, str] = {
    'computer_20251124': 'computer-use-2025-11-24',
    'computer_20250124': 'computer-use-2025-01-24',
}
"""The beta header each earlier computer use tool needs."""


class ComputerBlock(TypedDict):
    """A call of the computer use tool, as Claude made it."""

    id: str
    name: str
    input: dict[str, Any]


class ComputerReplay(TypedDict):
    """What a `computer` call keeps in its `provider_details`, to send back the calls it came from as they were."""

    computer_blocks: list[ComputerBlock]
    computer_tool: AnthropicComputerToolVersion
    computer_action_counts: list[Annotated[int, Field(ge=1)]]
    """How many actions each call became, so results are matched to calls without mapping them again."""


_REPLAY = TypeAdapter(ComputerReplay)


class _CallArgs(TypedDict):
    actions: list[Any]


_CALL_ARGS = TypeAdapter(_CallArgs)


def computer_replay(part: ToolCallPart, provider_name: str) -> ComputerReplay | None:
    """The calls a `computer` call came from, or `None` to send it back as a plain call.

    A history may come from storage or a UI, so what it kept is validated: one count per call, each at least 1,
    adding up to the call's actions. The part's class isn't checked: a call whose actions don't validate, such as one
    with an option the `computer` tool lacks, is a plain `ToolCallPart` after JSON, and still goes back as its calls.
    """
    if part.provider_name != provider_name:
        return None
    try:
        replay = _REPLAY.validate_python(part.provider_details)
        actions = _CALL_ARGS.validate_python(part.args_as_dict())['actions']
    except ValidationError:
        return None
    blocks, counts = replay['computer_blocks'], replay['computer_action_counts']
    return replay if blocks and len(blocks) == len(counts) and sum(counts) == len(actions) else None


@dataclass(frozen=True)
class NativeComputer:
    """The `computer` tool and the Anthropic computer use tool sent in its place, from a `ComputerTool`."""

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

    def answers(self, block: BetaContentBlock | None) -> TypeGuard[BetaToolUseBlock]:
        """Whether a block is a call of the computer use tool sent for the `computer` tool."""
        if not isinstance(block, BetaToolUseBlock):
            return False
        if self.is_toolset:
            return block.toolset_name == 'computer'
        return block.name == 'computer' and block.toolset_name is None

    def call_part(
        self,
        blocks: list[dict[str, Any]],
        provider_name: str,
        *,
        incomplete: Collection[str] = (),
        split: Mapping[str, Sequence[str]] | None = None,
    ) -> ComputerCallPart:
        """Calls of the computer use tool as one `computer` call, keeping them to send back unchanged.

        Nothing runs for a call in `incomplete` (by ID), which didn't arrive whole, such as one cut off at `max_tokens`.
        `split` names, by call ID, the other tools Claude called just before a call: run as one batch, the call would
        run before them, so the batch is refused and Claude asked again. A call whose input isn't an object, or whose
        name isn't text, counts as incomplete, since the SDK doesn't validate responses and Anthropic takes only an
        object and a name back.
        """
        split = split or {}
        malformed = [block['id'] for block in blocks if not _well_formed(block)]
        incomplete = {*incomplete, *malformed}
        checked = [
            ComputerBlock(
                id=block['id'],
                name=block['name'] if _is_text(block['name']) else '',
                input=block['input'] if _is_object(block['input']) else {},
            )
            for block in blocks
        ]
        member_actions = [
            [{'type': _COMPUTER_MEMBER_TYPES.get(name, name), 'incomplete_input': True}]
            if block['id'] in incomplete
            else [{'type': _COMPUTER_MEMBER_TYPES.get(name, name), 'after_tool_calls': list(split[block['id']])}]
            if block['id'] in split
            else map_computer_member(name, args)
            for block in checked
            for name, args in [computer_member(block, self.version)]
        ]
        actions = [action for member in member_actions for action in member]
        replay = ComputerReplay(
            computer_blocks=checked,
            computer_tool=self.version,
            computer_action_counts=[len(member) for member in member_actions],
        )
        return ComputerCallPart(
            self.name,
            # A field Claude left out takes the `computer` tool's default.
            {'actions': [{key: value for key, value in action.items() if value is not None} for action in actions]},
            tool_call_id=checked[0]['id'],
            provider_name=provider_name,
            provider_details=dict(replay),
        )


def _well_formed(block: dict[str, Any]) -> bool:
    return _is_text(block['name']) and _is_object(block['input'])


def _is_text(value: Any) -> bool:
    return isinstance(value, str)


def _is_object(value: Any) -> bool:
    return isinstance(value, dict)


def computer_member(block: ComputerBlock, version: AnthropicComputerToolVersion) -> tuple[str, dict[str, Any]]:
    """A call's member and input: the earlier tools name the member in the input's `action`."""
    if version == 'computer_toolset_20260801':
        return block['name'], block['input']
    action = block['input'].get('action', '')
    if not isinstance(action, str):
        return '', dict(block['input'])  # names the `action` among the fields that don't map
    args = {key: value for key, value in block['input'].items() if key != 'action'}
    return action, args


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


class _Strict(BaseModel, extra='forbid', strict=True):
    """A member's input as the tool documents it, checked as is: a field it lacks or a number for text is refused."""


_Coordinate = Annotated[list[int], Field(min_length=2, max_length=2)]


class _Click(_Strict):
    coordinate: _Coordinate | None = None
    text: str | None = None
    """Keys held during the click, as for a move, scroll or drag."""


class _Move(_Strict):
    coordinate: _Coordinate
    text: str | None = None


class _Drag(_Strict):
    start_coordinate: _Coordinate
    coordinate: _Coordinate
    text: str | None = None


class _Scroll(_Strict):
    coordinate: _Coordinate | None = None
    scroll_direction: str
    scroll_amount: int
    text: str | None = None


class _Type(_Strict):
    text: str


class _Key(_Strict):
    text: str
    repeat: int | None = None


class _HoldKey(_Strict):
    text: str
    duration: float


class _Wait(_Strict):
    duration: float | None = None


class _Zoom(_Strict):
    region: Annotated[list[int], Field(min_length=4, max_length=4)]


class _NoInput(_Strict):
    pass


_MEMBERS: dict[str, type[_Strict]] = {
    **dict.fromkeys(_COMPUTER_CLICKS, _Click),
    'mouse_move': _Move,
    'left_click_drag': _Drag,
    'left_mouse_down': _NoInput,
    'left_mouse_up': _NoInput,
    'scroll': _Scroll,
    'type': _Type,
    'key': _Key,
    'hold_key': _HoldKey,
    'wait': _Wait,
    'zoom': _Zoom,
    'screenshot': _NoInput,
    'cursor_position': _NoInput,
}
"""The input each computer use member takes that maps onto the `computer` tool."""

_COMPUTER_MEMBER_TYPES: dict[str, str] = {
    **dict.fromkeys(_COMPUTER_CLICKS, 'click'),
    'mouse_move': 'move',
    'left_click_drag': 'drag',
    'left_mouse_down': 'mouse_down',
    'left_mouse_up': 'mouse_up',
    'key': 'keypress',
}
"""The `computer` tool action type of each member whose name differs from it."""


def _point(coordinate: list[int] | None) -> dict[str, int]:
    """A `[x, y]` coordinate as `x` and `y`, or nothing when it's absent."""
    return {} if coordinate is None else {'x': coordinate[0], 'y': coordinate[1]}


def _modifiers(text: str | None) -> list[str]:
    return [text] if text else []


def map_computer_member(name: str, args: dict[str, Any]) -> list[dict[str, Any]]:
    """One computer use call as `computer` tool actions.

    A member or field that doesn't map fails validation, naming only what didn't map, so the model retries and the
    call never runs in part.
    """
    kind = _COMPUTER_MEMBER_TYPES.get(name, name)
    schema = _MEMBERS.get(name)
    if schema is None:
        return [{'type': kind, 'unsupported_fields': sorted(args)}]
    try:
        member = schema.model_validate(args)
    except ValidationError as error:
        # Under the tool's own type, so the retry names the fields rather than refusing the member.
        fields = sorted({str(detail['loc'][0]) for detail in error.errors() if detail['loc']})
        return [{'type': kind, 'unsupported_fields': fields}]
    match member:
        case _Click():
            button, count = _COMPUTER_CLICKS[name]
            return [
                {
                    'type': 'click',
                    **_point(member.coordinate),
                    'button': button,
                    'count': count,
                    'modifiers': _modifiers(member.text),
                }
            ]
        case _Move():
            return [{'type': 'move', **_point(member.coordinate), 'modifiers': _modifiers(member.text)}]
        case _Drag():
            path = [_point(member.start_coordinate), _point(member.coordinate)]
            return [{'type': 'drag', 'path': path, 'modifiers': _modifiers(member.text)}]
        case _Scroll():
            return [
                {
                    'type': 'scroll',
                    **_point(member.coordinate),
                    'direction': member.scroll_direction,
                    'amount': member.scroll_amount,
                    'modifiers': _modifiers(member.text),
                }
            ]
        case _Type():
            return [{'type': 'type', 'text': member.text}]
        case _Key():
            # xdotool syntax, which Claude writes: chords separated by spaces are pressed one after another.
            chords = member.text.split() or ['']
            return [{'type': 'keypress', 'keys': [chord], 'repeat': member.repeat} for chord in chords]
        case _HoldKey():
            return [{'type': 'hold_key', 'keys': [member.text], 'seconds': member.duration}]
        case _Wait():
            return [{'type': 'wait', 'seconds': member.duration}]
        case _Zoom():
            left, top, right, bottom = member.region
            return [{'type': 'zoom', 'region': {'left': left, 'top': top, 'right': right, 'bottom': bottom}}]
        case _:  # `left_mouse_down`, `left_mouse_up`, `screenshot` and `cursor_position` take no input
            return [{'type': kind}]
