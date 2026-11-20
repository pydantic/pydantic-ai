"""Send the `computer` tool to Claude as Anthropic's computer use tool, and turn Claude's calls into `computer` calls.

Anthropic's earlier tools, `computer_20250124` and `computer_20251124`, are one tool named `computer`. Each call names
what to do in its `action` field, such as `left_click`. The `computer_toolset_20260801` toolset is a set of tools
instead, one per action, such as a `left_click` tool. This module calls each of those actions a member.
"""

from __future__ import annotations as _annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from itertools import accumulate
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
from ...messages import FinishReason, ModelResponsePart, ToolCallPart
from ...profiles.anthropic import AnthropicComputerToolVersion
from .. import ModelRequestParameters
from .._computer_calls import UNFINISHED_COMPUTER_ARGS, block_computer_calls

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


class ComputerReplay(TypedDict):
    """What a `computer` call stores in `provider_details`, so Claude's original calls can be sent back unchanged."""

    computer_blocks: Annotated[list[ComputerBlock], Field(min_length=1)]
    computer_tool: AnthropicComputerToolVersion
    computer_action_counts: list[Annotated[int, Field(ge=1)]]
    """How many `computer` actions each of Claude's calls became. Results are matched to calls by these counts."""


_REPLAY = TypeAdapter(ComputerReplay)


def stop_computer_calls_unless_finished(
    parts: Sequence[ModelResponsePart],
    model_request_parameters: ModelRequestParameters,
    finish_reason: FinishReason | None,
) -> list[ModelResponsePart]:
    """Keep every `computer` call in a response cut off by the token limit from running.

    Claude can stop at `max_tokens` partway through a batch, and the finished calls would then run without the rest.
    """
    if finish_reason != 'length':
        return list(parts)
    return block_computer_calls(parts, model_request_parameters)


def computer_replay(part: ToolCallPart, provider_name: str, sent: NativeComputer) -> ComputerReplay | None:
    """Return Claude's original calls behind a `computer` call, or `None` to send it back as a regular tool call.

    The history may come from storage or a UI, so the stored data is validated.

    The part's class isn't checked: a call with an action the `computer` tool doesn't accept becomes a plain
    `ToolCallPart` after a round trip through JSON, and still goes back as Claude's calls.

    The original calls aren't sent back if a hook changed the part's actions or tool name, or if they came from a
    different tool version than `sent`, since that would tell Claude that its own calls ran.
    """
    if part.provider_name != provider_name or part.tool_name != sent.name:
        return None
    try:
        replay = _REPLAY.validate_python(part.provider_details)
    except ValidationError:
        return None
    if replay['computer_tool'] != sent.version:
        return None
    if part.args == UNFINISHED_COMPUTER_ARGS:
        # Blocked because the response was cut off; each call is answered as not run.
        return replay
    blocks, counts = replay['computer_blocks'], replay['computer_action_counts']
    match part.args_as_dict():
        case {'actions': [*actions]} if len(blocks) == len(counts) and sum(counts) == len(actions):
            starts = [0, *accumulate(counts)][:-1]
            members = [actions[start : start + count] for start, count in zip(starts, counts)]
            if all(_made(block, member, replay['computer_tool']) for block, member in zip(blocks, members)):
                return replay
            return None
        case _:
            return None


def _made(block: ComputerBlock, actions: list[Any], version: AnthropicComputerToolVersion) -> bool:
    """Whether `actions` are what Claude's call `block` became, or its refusal for coming after another tool's call.

    If the call now becomes a different number of actions, because an earlier release converted it differently, the
    check passes and the stored count is used.
    """
    if any(isinstance(action, dict) and 'after_tool_calls' in action for action in actions):
        return True
    made = [
        {key: value for key, value in action.items() if value is not None}
        for action in map_computer_member(*computer_member(block, version))
    ]
    return len(made) != len(actions) or actions == made


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

    def answers(self, block: BetaContentBlock | None) -> TypeGuard[BetaToolUseBlock]:
        """Whether `block` is Claude's call to the computer use tool sent in place of the `computer` tool."""
        if not isinstance(block, BetaToolUseBlock):
            return False
        if self.is_toolset:
            return block.toolset_name == 'computer'
        return block.name == 'computer' and block.toolset_name is None

    def call_part(
        self,
        blocks: list[ComputerBlock],
        provider_name: str,
        member_actions: list[list[dict[str, Any]]] | None = None,
    ) -> ComputerCallPart:
        """Turn Claude's computer use calls into one `computer` call, keeping the calls to send back unchanged.

        Pass `member_actions` to reuse calls `member_actions()` already converted.
        """
        if member_actions is None:
            member_actions = [self.member_actions(block) for block in blocks]
        actions = [action for member in member_actions for action in member]
        replay = ComputerReplay(
            computer_blocks=blocks,
            computer_tool=self.version,
            computer_action_counts=[len(member) for member in member_actions],
        )
        return ComputerCallPart(
            self.name,
            # A field Claude left out is dropped, so the `computer` tool's default applies.
            {'actions': [{key: value for key, value in action.items() if value is not None} for action in actions]},
            tool_call_id=blocks[0]['id'],
            provider_name=provider_name,
            provider_details=dict(replay),
        )

    def member_actions(self, block: ComputerBlock, after_tool_calls: Sequence[str] = ()) -> list[dict[str, Any]]:
        """Convert one of Claude's calls into `computer` tool actions.

        If the call can't run as Claude asked, the result is one action that the `computer` tool refuses.

        `after_tool_calls` names the other tools Claude called just before this call. Since the batch runs as one
        `computer` call, this call would run before them, so the batch is refused and Claude is asked again.
        """
        name, args = computer_member(block, self.version)
        if after_tool_calls:
            return [{'type': _COMPUTER_MEMBER_TYPES.get(name, name), 'after_tool_calls': list(after_tool_calls)}]
        return map_computer_member(name, args)


SPLIT_BATCH_REFUSAL = (
    'No computer action ran: they run together as one batch, and another tool call came between them. Make the '
    "computer calls before or after the other tools' calls, not between them."
)
"""What Claude gets instead of the validation errors when another tool call came between its computer calls.

It is fixed text, so nothing from a stored or UI-built history reaches Claude through it.
"""


def splits_batch(part: ToolCallPart) -> bool:
    """Whether a `computer` call that `computer_replay` accepted has an action marked as after another tool's call."""
    if part.args == UNFINISHED_COMPUTER_ARGS:
        return False
    return any(isinstance(action, dict) and 'after_tool_calls' in action for action in part.args_as_dict()['actions'])


@dataclass
class ComputerBatch:
    """Collects Claude's computer use calls in one response and turns them into `computer` calls."""

    computer: NativeComputer | None
    provider_name: str
    blocks: list[ComputerBlock] = field(default_factory=list[ComputerBlock])
    index: int = 0
    """The position in the response's parts where the toolset's batch goes, which is where its first call was."""
    between: list[str] = field(default_factory=list[str])
    """Other tools called since the batch's last call. They would run out of order if the batch ran as one call."""
    actions: list[list[dict[str, Any]]] = field(default_factory=list[list[dict[str, Any]]])
    """Each call's `computer` actions, converted once when the call is added."""

    def take(self, block: BetaToolUseBlock, parts: list[ModelResponsePart]) -> bool:
        """Collect `block` if it's a computer use call. Otherwise record it as another tool's call."""
        if self.computer is None or not self.computer.answers(block):
            self.note_other(block.name)
            return False
        call = ComputerBlock(id=block.id, name=block.name, input=block.input)
        if self.computer.is_toolset:
            self.add(call, len(parts))
        else:
            # With an earlier computer use tool, each call becomes its own `computer` call.
            parts.append(self.computer.call_part([call], self.provider_name))
        return True

    def add(self, call: ComputerBlock, index: int = 0) -> None:
        """Add a toolset call to the batch. If it's the first call, `index` is where the batch goes in the parts."""
        assert self.computer is not None
        if not self.blocks:
            self.index = index
        self.blocks.append(call)
        self.actions.append(self.computer.member_actions(call, self.between))
        self.between = []

    def note_other(self, tool_name: str) -> None:
        """Record another tool's call."""
        if self.blocks:
            self.between.append(tool_name)

    def call_part(self) -> ComputerCallPart | None:
        """The toolset's calls as one `computer` call, or `None` if there are none."""
        if self.computer is None or not self.blocks:
            return None
        return self.computer.call_part(self.blocks, self.provider_name, self.actions)

    def insert(self, parts: list[ModelResponsePart]) -> None:
        """Insert the toolset's calls into `parts` as one `computer` call, where the first call was.

        The batch runs in order until a call fails. Text may come between its calls; another tool call can't, because
        it would then run after calls Claude made after it.
        """
        if (part := self.call_part()) is not None:
            parts.insert(self.index, part)


def computer_member(block: ComputerBlock, version: AnthropicComputerToolVersion) -> tuple[str, dict[str, Any]]:
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


COMPUTER_NOT_EXECUTED = 'Not executed: an earlier computer action in this turn failed.'
"""The result Anthropic asks for on each call that comes after the failed call in a batch."""

COMPUTER_FAILED = 'Failed.'
"""The result sent for the call that failed. The error is only in the result's content, which a hook may change."""

COMPUTER_RESULT_ON_LAST = 'The result is on the last call of this turn.'
"""The result sent for every call except the last, when the result doesn't say which calls ran."""

_COMPUTER_CLICKS = {
    'left_click': ('left', 1),
    'right_click': ('right', 1),
    'middle_click': ('middle', 1),
    'double_click': ('left', 2),
    'triple_click': ('left', 3),
}


class _Strict(BaseModel, extra='forbid', strict=True):
    """A member's input as Anthropic documents it, checked strictly: an unknown field or a number for text fails."""


_Coordinate = Annotated[list[int], Field(min_length=2, max_length=2)]


class _Click(_Strict):
    coordinate: _Coordinate | None = None
    text: str | None = None
    """Keys held down during the click. A scroll or a drag takes them the same way."""


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


_Region = Annotated[list[int], Field(min_length=4, max_length=4)]


class _Zoom(_Strict):
    region: _Region


class _Screenshot(_Strict):
    region: _Region | None = None
    """Not documented on `screenshot`, but Claude sends it, in the same frame as `zoom`'s `region`."""


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
    'screenshot': _Screenshot,
    'cursor_position': _NoInput,
}
"""The input of each member that has a matching `computer` tool action."""


_COMPUTER_MEMBER_TYPES: dict[str, str] = {
    **dict.fromkeys(_COMPUTER_CLICKS, 'click'),
    'left_click_drag': 'drag',
    'left_mouse_down': 'mouse_down',
    'left_mouse_up': 'mouse_up',
    'type': 'type_text',
    'key': 'press_keys',
    'hold_key': 'hold_keys',
}
"""The `computer` tool action type for each member whose name is different."""


def _point(coordinate: list[int] | None) -> dict[str, int]:
    """A `[x, y]` coordinate as `x` and `y`, or nothing when it's absent."""
    return {} if coordinate is None else {'x': coordinate[0], 'y': coordinate[1]}


def _modifiers(text: str | None) -> list[str]:
    return [text] if text else []


def map_computer_member(name: str, args: dict[str, Any]) -> list[dict[str, Any]]:
    """Convert one computer use call into `computer` tool actions.

    A member or field with no matching `computer` action fails validation. The error names only what didn't match.
    The model then retries, so the call never runs halfway.
    """
    kind = _COMPUTER_MEMBER_TYPES.get(name, name)
    schema = _MEMBERS.get(name)
    if schema is None:
        return [{'type': kind, 'unsupported_fields': sorted(args)}]
    try:
        member = schema.model_validate(args)
    except ValidationError as error:
        # Keep the member's action type, so the retry message names the bad fields instead of refusing the member.
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
            return [{'type': 'mouse_move', **_point(member.coordinate)}]
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
            return [{'type': 'type_text', 'text': member.text}]
        case _Key():
            # Claude writes keys in xdotool syntax: key combinations separated by spaces are pressed one after another.
            chords = member.text.split() or ['']
            return [{'type': 'press_keys', 'keys': [chord], 'repeat': member.repeat} for chord in chords]
        case _HoldKey():
            return [{'type': 'hold_keys', 'keys': [member.text], 'seconds': member.duration}]
        case _Wait():
            return [{'type': 'wait', 'seconds': member.duration}]
        case _Press():
            # Claude's earlier tools press or release the button at a point, so the pointer moves there first.
            move = [{'type': 'mouse_move', **_point(member.coordinate)}] if member.coordinate else []
            return [*move, {'type': kind}]
        case _Zoom(region=[left, top, right, bottom]) | _Screenshot(region=[left, top, right, bottom]):
            # A screenshot of a region is a zoom: both capture that part of the screen.
            return [{'type': 'zoom', 'region': {'left': left, 'top': top, 'right': right, 'bottom': bottom}}]
        case _:  # a whole-screen `screenshot`, and `cursor_position`, which takes no input
            return [{'type': kind}]
