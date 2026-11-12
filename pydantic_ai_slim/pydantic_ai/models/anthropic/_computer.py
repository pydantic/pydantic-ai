"""Mapping the `computer` tool to Anthropic's computer use tools, and their calls back to `computer` calls."""

from __future__ import annotations as _annotations

from collections.abc import Callable, Collection, Mapping, Sequence
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
from pydantic import TypeAdapter, ValidationError
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
    computer_action_counts: list[int]
    """How many actions each call became, so results are matched to calls without mapping them again."""


_REPLAY = TypeAdapter(ComputerReplay)


def computer_replay(part: ToolCallPart, provider_name: str) -> ComputerReplay | None:
    """The calls a `computer` call came from, or `None` to send it back as a plain call.

    A history can come back from storage or a UI, so what it kept is validated, and one call per count is required.
    The part's class isn't: a call whose actions don't validate, such as one Claude asked for with an option the
    `computer` tool lacks, is a plain `ToolCallPart` once the history has been through JSON, and still goes back as
    the calls it came from.
    """
    if part.provider_name != provider_name:
        return None
    try:
        replay = _REPLAY.validate_python(part.provider_details)
    except ValidationError:
        return None
    blocks = replay['computer_blocks']
    return replay if blocks and len(blocks) == len(replay['computer_action_counts']) else None


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
        """Calls of the computer use tool as one `computer` call, keeping them to send back as they were.

        A call in `incomplete`, by ID, didn't arrive whole, such as one cut off at `max_tokens`: none of what it
        asked for runs. `split` names, by call ID, the other tools Claude called just before a call: run as one
        batch, the call would run before them, so the batch is refused and Claude asked again. A call whose input
        isn't an object is incomplete too: nothing in it maps, and Anthropic takes only an object back.
        """
        split = split or {}
        incomplete = {*incomplete, *(block['id'] for block in blocks if not isinstance(block['input'], dict))}
        blocks = [block if isinstance(block['input'], dict) else {**block, 'input': {}} for block in blocks]
        member_actions = [
            [{'type': _COMPUTER_MEMBER_TYPES.get(name, name), 'incomplete_input': True}]
            if block['id'] in incomplete
            else [{'type': _COMPUTER_MEMBER_TYPES.get(name, name), 'after_tool_calls': list(split[block['id']])}]
            if block['id'] in split
            else map_computer_member(name, args)
            for block in blocks
            for name, args in [computer_member(block, self.version)]
        ]
        actions = [action for member in member_actions for action in member]
        replay = ComputerReplay(
            computer_blocks=[ComputerBlock(id=b['id'], name=b['name'], input=b['input']) for b in blocks],
            computer_tool=self.version,
            computer_action_counts=[len(member) for member in member_actions],
        )
        return ComputerCallPart(
            self.name,
            # A field Claude left out takes the `computer` tool's default.
            {'actions': [{key: value for key, value in action.items() if value is not None} for action in actions]},
            tool_call_id=blocks[0]['id'],
            provider_name=provider_name,
            provider_details=dict(replay),
        )


def computer_member(block: dict[str, Any], version: AnthropicComputerToolVersion) -> tuple[str, dict[str, Any]]:
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

_REQUIRED_MEMBER_FIELDS: dict[str, frozenset[str]] = {
    'mouse_move': frozenset({'coordinate'}),
    'scroll': frozenset({'scroll_direction', 'scroll_amount'}),
    'type': frozenset({'text'}),
    'key': frozenset({'text'}),
    'hold_key': frozenset({'text', 'duration'}),
    'zoom': frozenset({'region'}),
}
"""The input fields a member can't do without; one that is missing is refused, not filled in."""


def _ints(value: Any, count: int) -> bool:
    return (
        isinstance(value, list)
        and len(cast(list[Any], value)) == count
        and all(type(item) is int for item in cast(list[Any], value))
    )


_MEMBER_FIELD_TYPES: dict[str, Callable[[Any], bool]] = {
    'coordinate': lambda value: _ints(value, 2),
    'start_coordinate': lambda value: _ints(value, 2),
    'region': lambda value: _ints(value, 4),
    'text': lambda value: isinstance(value, str),
    'scroll_direction': lambda value: isinstance(value, str),
    'scroll_amount': lambda value: type(value) is int,
    'repeat': lambda value: type(value) is int,
    'duration': lambda value: type(value) in (int, float),
}
"""Whether each input field has the type the tool documents, checked as is: a number is not text."""


def _unmapped_member_fields(name: str, args: dict[str, Any]) -> list[str] | None:
    """The input fields of a member that don't map, or `None` when every one does.

    A field that isn't one the `computer` tool takes, is missing though required, or has the wrong type doesn't map.
    """
    fields = _COMPUTER_MEMBER_FIELDS.get(name)
    if fields is None:
        return sorted(args.keys())
    unmapped = (args.keys() - fields) | (_REQUIRED_MEMBER_FIELDS.get(name, frozenset()) - args.keys())
    unmapped |= {key for key, value in args.items() if key in fields and not _MEMBER_FIELD_TYPES[key](value)}
    return sorted(unmapped) if unmapped else None


def map_computer_member(name: str, args: dict[str, Any]) -> list[dict[str, Any]]:
    """One computer use call as `computer` tool actions, mapped field by field.

    A member or field that doesn't map fails validation, so the model retries; only the names of what didn't map are
    kept, so the call never runs with part of what the model asked for.
    """
    unsupported = _unmapped_member_fields(name, args)
    if unsupported is not None:
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
            return [{'type': 'type', 'text': args['text']}]
        case 'key':
            # xdotool syntax, which Claude writes: chords separated by spaces are pressed one after another.
            chords = args['text'].split() or ['']
            return [{'type': 'keypress', 'keys': [chord], 'repeat': args.get('repeat')} for chord in chords]
        case 'hold_key':
            return [{'type': 'hold_key', 'keys': [args['text']], 'seconds': args['duration']}]
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
