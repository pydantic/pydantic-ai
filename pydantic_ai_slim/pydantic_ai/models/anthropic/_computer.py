"""Mapping the `computer` tool to Anthropic's computer use tools, and their calls back to `computer` calls."""

from __future__ import annotations as _annotations

from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Annotated, Any, NotRequired, TypeGuard

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
from ...messages import ModelResponsePart, ToolCallPart
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
    computer_refusal: NotRequired[str]
    """Why none of the calls ran, sent to Claude in place of the retry's validation errors."""


_REPLAY = TypeAdapter(ComputerReplay)


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
    except ValidationError:
        return None
    blocks, counts = replay['computer_blocks'], replay['computer_action_counts']
    match part.args_as_dict():
        case {'actions': [*actions]} if blocks and len(blocks) == len(counts) and sum(counts) == len(actions):
            return replay
        case _:
            return None


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
        blocks: list[ComputerBlock],
        provider_name: str,
        *,
        incomplete: Collection[str] = (),
        split: Mapping[str, Sequence[str]] | None = None,
    ) -> ComputerCallPart:
        """Calls of the computer use tool as one `computer` call, keeping them to send back unchanged.

        Nothing runs for a call in `incomplete` (by ID), which didn't arrive whole, such as one cut off at `max_tokens`.
        `split` names, by call ID, the other tools Claude called just before a call: run as one batch, the call would
        run before them, so the batch is refused and Claude asked again.
        """
        split = split or {}
        member_actions = [self._member_actions(block, incomplete, split) for block in blocks]
        actions = [action for member in member_actions for action in member]
        replay = ComputerReplay(
            computer_blocks=blocks,
            computer_tool=self.version,
            computer_action_counts=[len(member) for member in member_actions],
        )
        if between := [f'`{name}`' for names in split.values() for name in names]:
            replay['computer_refusal'] = (
                f'No computer action ran: they run together as one batch, and a call of {" and ".join(between)} came '
                "between them. Make the computer calls before or after the other tools' calls, not between them."
            )
        return ComputerCallPart(
            self.name,
            # A field Claude left out takes the `computer` tool's default.
            {'actions': [{key: value for key, value in action.items() if value is not None} for action in actions]},
            tool_call_id=blocks[0]['id'],
            provider_name=provider_name,
            provider_details=dict(replay),
        )

    def _member_actions(
        self, block: ComputerBlock, incomplete: Collection[str], split: Mapping[str, Sequence[str]]
    ) -> list[dict[str, Any]]:
        """The `computer` tool actions one call becomes, or one the tool refuses if it can't run as asked."""
        name, args = computer_member(block, self.version)
        if block['id'] in incomplete:
            return [{'type': _COMPUTER_MEMBER_TYPES.get(name, name), 'incomplete_input': True}]
        if block['id'] in split:
            return [{'type': _COMPUTER_MEMBER_TYPES.get(name, name), 'after_tool_calls': list(split[block['id']])}]
        return map_computer_member(name, args)


@dataclass
class ComputerBatch:
    """Collects a response's calls of the computer use tool into the `computer` calls they become."""

    computer: NativeComputer | None
    provider_name: str
    blocks: list[ComputerBlock] = field(default_factory=list[ComputerBlock])
    index: int = 0
    """Where in the response's parts the toolset's batch goes: at its first call."""
    between: list[str] = field(default_factory=list[str])
    """The other tools called since the batch's last call, which would run out of order if the batch ran as one."""
    split: dict[str, list[str]] = field(default_factory=dict[str, list[str]])
    incomplete: set[str] = field(default_factory=set[str])
    """The IDs of calls whose input didn't arrive whole, such as in a stream cut off at `max_tokens`."""

    def take(self, block: BetaToolUseBlock, parts: list[ModelResponsePart]) -> bool:
        """Collect `block` if it's a call of the computer use tool; otherwise note it as another tool's call."""
        if self.computer is None or not self.computer.answers(block):
            self.note_other(block.name)
            return False
        call = ComputerBlock(id=block.id, name=block.name, input=block.input)
        if self.computer.is_toolset:
            self.add(call, len(parts))
        else:
            # Each call of an earlier computer use tool is a call of its own.
            parts.append(self.computer.call_part([call], self.provider_name))
        return True

    def add(self, call: ComputerBlock, index: int = 0) -> None:
        """Add a toolset call to the batch; `index` is where in the parts the batch goes if this is its first call."""
        if not self.blocks:
            self.index = index
        self.blocks.append(call)
        if self.between:
            self.split[call['id']] = self.between
            self.between = []

    def note_other(self, tool_name: str) -> None:
        """Note another tool's call, which runs out of order if it comes between the batch's calls."""
        if self.blocks:
            self.between.append(tool_name)

    def call_part(self) -> ComputerCallPart | None:
        """The toolset's calls as one `computer` call, or `None` if there are none."""
        if self.computer is None or not self.blocks:
            return None
        return self.computer.call_part(self.blocks, self.provider_name, incomplete=self.incomplete, split=self.split)

    def insert(self, parts: list[ModelResponsePart]) -> None:
        """Put the toolset's calls in `parts` as one `computer` call, at the first one's place.

        A response's toolset member calls are one batch, run in order until one fails. Text may sit between them;
        another tool call can't, since it would then run after calls Claude made after it.
        """
        if (part := self.call_part()) is not None:
            parts.insert(self.index, part)


def computer_member(block: ComputerBlock, version: AnthropicComputerToolVersion) -> tuple[str, dict[str, Any]]:
    """A call's member and input: the earlier tools name the member in the input's `action`."""
    if version == 'computer_toolset_20260801':
        return block['name'], block['input']
    return block['input'].get('action', ''), {key: value for key, value in block['input'].items() if key != 'action'}


def returns_output(blocks: list[ComputerBlock], index: int) -> bool:
    """Whether a call's result has content of its own, before the screen the last call that ran gets.

    A zoom and the pointer position do, and so does a screenshot before the last call, captured where it is.
    """
    name = blocks[index]['name']
    return name in ('zoom', 'cursor_position') or (name == 'screenshot' and index < len(blocks) - 1)


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
    duration: float


class _Zoom(_Strict):
    region: Annotated[list[int], Field(min_length=4, max_length=4)]


class _Press(_Strict):
    coordinate: _Coordinate | None = None


class _NoInput(_Strict):
    pass


_MEMBERS: dict[str, type[_Strict]] = {
    **dict.fromkeys(_COMPUTER_CLICKS, _Click),
    'mouse_move': _Move,
    'left_click_drag': _Drag,
    'left_mouse_down': _Press,
    'left_mouse_up': _Press,
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
            return [{'type': 'move', **_point(member.coordinate)}]
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
        case _Press():
            # Claude's earlier tools press or release at a point: the pointer moves there first.
            move = [{'type': 'move', **_point(member.coordinate)}] if member.coordinate else []
            return [*move, {'type': kind}]
        case _Zoom():
            left, top, right, bottom = member.region
            return [{'type': 'zoom', 'region': {'left': left, 'top': top, 'right': right, 'bottom': bottom}}]
        case _:  # `screenshot` and `cursor_position` take no input
            return [{'type': kind}]
