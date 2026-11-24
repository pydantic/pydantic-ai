"""Send the `computer` tool to Gemini as Google's computer use tool, and turn Gemini's calls into `computer` calls.

Google's tool is a set of predefined functions, such as `click` and `type`, that Gemini calls like function tools.
Positions are on a 0-999 grid over the screenshot, and are converted to pixels here, once, with the size of the
screenshot the request sent. Gemini's original call is kept, so it goes back unchanged.

A call that Gemini's safety checks flag carries a `safety_decision` asking for a person's confirmation. Its
`ComputerCallPart` gets a [`confirmation`][pydantic_ai.computer_use.ComputerCallPart.confirmation], and every response
to it tells Gemini whether the person approved it. See https://ai.google.dev/gemini-api/docs/computer-use#safety-decisions.
"""

from __future__ import annotations as _annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import get_args

from google.genai.types import ComputerUseDict, Environment, ToolDict
from pydantic import BaseModel, JsonValue, TypeAdapter, ValidationError
from typing_extensions import TypedDict

from ..computer_use import ComputerCallPart, Modifier
from ..messages import ModelMessage, ModelResponse, ToolCallPart, ToolReturnPart
from ._computer_calls import key_from_x11


class GeminiFunctionCall(TypedDict):
    """One call Gemini made to a function of Google's computer use tool, unchanged."""

    name: str
    args: dict[str, JsonValue]


class GeminiComputerReplay(TypedDict):
    """What a `computer` call stores in `provider_details`, so Gemini's original call can be sent back unchanged."""

    gemini_call: GeminiFunctionCall
    display_size: tuple[int, int]
    """The size of the screenshot Gemini pointed on, which converted its positions to pixels."""
    held_modifiers: tuple[str, ...]
    """The modifiers Gemini's earlier `key_down` calls held when this call ran, which it ran with."""


_REPLAY_ADAPTER = TypeAdapter(GeminiComputerReplay)


class _FunctionArgs(BaseModel, extra='forbid', strict=True):
    """A function's arguments as Gemini sends them, checked strictly: an unknown field or a string for a number fails."""


class _Position(_FunctionArgs):
    x: int
    y: int


class _DragAndDrop(_FunctionArgs):
    start_x: int
    start_y: int
    end_x: int
    end_y: int


class _Scroll(_FunctionArgs):
    x: int
    y: int
    direction: str
    magnitude_in_wheel_clicks: int = 3


class _Type(_FunctionArgs):
    text: str
    press_enter: bool | None = None


class _Wait(_FunctionArgs):
    seconds: float | None = None


class _Key(_FunctionArgs):
    key: str


class _Hotkey(_FunctionArgs):
    keys: list[str]


class _NoArgs(_FunctionArgs):
    pass


_CLICK_FUNCTIONS = {
    'click': ('left', 1),
    'double_click': ('left', 2),
    'triple_click': ('left', 3),
    'middle_click': ('middle', 1),
    'right_click': ('right', 1),
}

_FUNCTION_ARGS: dict[str, type[_FunctionArgs]] = {
    **dict.fromkeys(_CLICK_FUNCTIONS, _Position),
    'move': _Position,
    'mouse_down': _Position,
    'mouse_up': _Position,
    'drag_and_drop': _DragAndDrop,
    'scroll': _Scroll,
    'type': _Type,
    'wait': _Wait,
    'press_key': _Key,
    'key_down': _Key,
    'key_up': _Key,
    'hotkey': _Hotkey,
    'take_screenshot': _NoArgs,
}
"""The arguments of each function of the desktop environment. Each also takes `intent` and `safety_decision`."""

COMPUTER_FUNCTIONS = frozenset(_FUNCTION_ARGS)
"""The names of the desktop environment's functions, which no function tool may have beside Google's tool."""

_ACTION_TYPE_BY_FUNCTION = {
    **dict.fromkeys(_CLICK_FUNCTIONS, 'click'),
    'move': 'mouse_move',
    'drag_and_drop': 'drag',
    'type': 'type_text',
    'press_key': 'press_keys',
    'hotkey': 'press_keys',
    # A key press or release only changes the modifiers held for later calls, so it runs as a screenshot.
    'key_down': 'screenshot',
    'key_up': 'screenshot',
    'take_screenshot': 'screenshot',
}
"""The `computer` tool action type for each function whose name is different."""


_ActionJson = dict[str, object]
"""One `computer` tool action, as the JSON the `computer` tool validates."""


def _confirmation_reason(args: dict[str, JsonValue]) -> str | None | ValueError:
    """Gemini's reason a call needs a person's confirmation, or `None` when it needs none.

    A decision other than `require_confirmation` gives an error, and the call must not run.
    """
    match args.get('safety_decision'):
        case None:
            return None
        case {'decision': 'require_confirmation', 'explanation': str() as explanation}:
            return explanation
        case _:
            return ValueError('safety_decision')


def _as_modifier(key: str) -> str | None:
    """The modifier `key` names, such as `shift` for `Shift_L`, or `None` for any other key."""
    name = key_from_x11(key)
    return name if name in get_args(Modifier) else None


def _grid_to_pixel(value: int, side: int) -> int:
    """A position on Gemini's 0-999 grid as a pixel on a side `side` pixels long."""
    return value * side // 1000


def map_computer_call(
    name: str, args: dict[str, JsonValue], display_size: tuple[int, int], held: tuple[str, ...] = ()
) -> tuple[list[_ActionJson], str | None, tuple[str, ...]]:
    """Convert one of Gemini's calls, made with the modifiers `held`, into `computer` tool actions.

    Also returns its confirmation reason, if any, and the modifiers held after it, which only a `key_down` or `key_up`
    that maps changes. A function, field or safety decision with no matching `computer` action becomes one action the
    `computer` tool refuses, naming what didn't match. Gemini is then asked again, and nothing runs.
    """
    action_type = _ACTION_TYPE_BY_FUNCTION.get(name, name)
    confirmation = _confirmation_reason(args)
    if isinstance(confirmation, ValueError):
        return [{'type': action_type, 'unsupported_fields': [str(confirmation)]}], None, held
    schema = _FUNCTION_ARGS.get(name)
    rest = {key: value for key, value in args.items() if key not in ('intent', 'safety_decision')}
    if schema is None:
        return [{'type': action_type, 'unsupported_fields': sorted(rest)}], confirmation, held
    try:
        call = schema.model_validate(rest)
    except ValidationError as error:
        fields = sorted({str(detail['loc'][0]) for detail in error.errors() if detail['loc']})
        return [{'type': action_type, 'unsupported_fields': fields}], confirmation, held
    width, height = display_size

    def to_pixels(x: int, y: int) -> _ActionJson:
        return {'x': _grid_to_pixel(x, width), 'y': _grid_to_pixel(y, height)}

    actions = _map_function_actions(name, call, to_pixels, held)
    if isinstance(call, _Key) and name in ('key_down', 'key_up') and (modifier := _as_modifier(call.key)):
        released = tuple(key for key in held if key != modifier)
        held = (*released, modifier) if name == 'key_down' else released
    return actions, confirmation, held


def _map_function_actions(
    name: str, call: _FunctionArgs, to_pixels: Callable[[int, int], _ActionJson], held: tuple[str, ...]
) -> list[_ActionJson]:
    """The `computer` tool actions for Gemini's validated call to the function `name`, with the modifiers `held`."""
    modifiers_field: _ActionJson = {'modifiers': list(held)} if held else {}
    match call:
        case _Position() if name in _CLICK_FUNCTIONS:
            button, count = _CLICK_FUNCTIONS[name]
            return [{'type': 'click', **to_pixels(call.x, call.y), 'button': button, 'count': count, **modifiers_field}]
        case _Position() if name == 'move':
            return [{'type': 'mouse_move', **to_pixels(call.x, call.y)}]
        case _Position():
            # `mouse_down` and `mouse_up` press or release the left button at a point, so the pointer moves there first.
            return [{'type': 'mouse_move', **to_pixels(call.x, call.y)}, {'type': name}]
        case _DragAndDrop():
            return [
                {
                    'type': 'drag',
                    'path': [to_pixels(call.start_x, call.start_y), to_pixels(call.end_x, call.end_y)],
                    **modifiers_field,
                }
            ]
        case _Scroll():
            amount = call.magnitude_in_wheel_clicks
            return [
                {
                    'type': 'scroll',
                    **to_pixels(call.x, call.y),
                    'direction': call.direction,
                    'amount': amount,
                    **modifiers_field,
                }
            ]
        case _Type():
            enter: list[_ActionJson] = [{'type': 'press_keys', 'keys': ['enter']}] if call.press_enter else []
            return [{'type': 'type_text', 'text': call.text}, *enter]
        case _Wait():
            return [{'type': 'wait'} if call.seconds is None else {'type': 'wait', 'seconds': call.seconds}]
        case _Key() | _Hotkey() if name in ('press_key', 'hotkey'):
            # Gemini sends X11 key names, as Claude does.
            keys = [key_from_x11(key) for key in (call.keys if isinstance(call, _Hotkey) else [call.key])]
            # A held modifier that the call names is pressed once.
            return [{'type': 'press_keys', 'keys': [*(key for key in held if key not in keys), *keys]}]
        case _Key() if name == 'key_down' and _as_modifier(call.key) is None:
            return [{'type': 'screenshot', 'unsupported_fields': [f'key {call.key!r}, which is not a modifier']}]
        case _Key():
            # Nothing is pressed on the screen, and Gemini gets a screenshot back: `map_computer_call` holds the key.
            return [{'type': 'screenshot'}]
        case _:  # `take_screenshot` takes no arguments
            return [{'type': 'screenshot'}]


@dataclass
class NativeComputer:
    """The `computer` tool, and Google's computer use tool sent in its place. Built from a `ComputerTool`."""

    name: str
    display_size: tuple[int, int]
    held_modifiers: tuple[str, ...] = ()
    """The modifiers Gemini holds for its next call: each `key_down` that no `key_up` let go of."""

    def restore_held_keys(self, messages: Sequence[ModelMessage]) -> None:
        """Hold the modifiers held after Gemini's last computer use call in `messages`, which stored them.

        A call that didn't run, such as a denied `key_down`, leaves the modifiers held before it.
        """
        parts = (part for m in reversed(messages) if isinstance(m, ModelResponse) for part in reversed(m.parts))
        last = next(
            ((part, r) for part in parts if isinstance(part, ToolCallPart) and (r := self._stored_call(part))), None
        )
        if last is None:
            self.held_modifiers = ()
            return
        part, (_, _, held_after, held_before) = last
        ran = any(
            isinstance(result, ToolReturnPart)
            and result.tool_call_id == part.tool_call_id
            and result.outcome == 'success'
            for m in messages
            for result in m.parts
        )
        self.held_modifiers = held_after if ran else held_before

    @staticmethod
    def tool_dict() -> ToolDict:
        return ToolDict(computer_use=ComputerUseDict(environment=Environment.ENVIRONMENT_DESKTOP))

    @staticmethod
    def is_computer_call(function_name: str | None) -> bool:
        """Whether a function call is Gemini's call to Google's computer use tool."""
        return function_name in COMPUTER_FUNCTIONS

    def to_call_part(
        self, name: str, args: dict[str, JsonValue], tool_call_id: str | None, provider_name: str
    ) -> ComputerCallPart:
        """Turn Gemini's call into a `computer` call in pixels, keeping the call to send back unchanged."""
        actions, confirmation, held = map_computer_call(name, args, self.display_size, self.held_modifiers)
        details: dict[str, JsonValue] = {
            'gemini_call': {'name': name, 'args': args},
            'display_size': list[JsonValue](self.display_size),
            'held_modifiers': list[JsonValue](self.held_modifiers),
        }
        self.held_modifiers = held
        if confirmation is not None:
            details['confirmation'] = confirmation
        part = ComputerCallPart(self.name, {'actions': actions}, provider_name=provider_name, provider_details=details)
        if tool_call_id is not None:
            part.tool_call_id = tool_call_id
        return part

    def stored_computer_call(self, part: ToolCallPart) -> GeminiFunctionCall | None:
        """Gemini's original call behind a `computer` call, or `None` to send it back as a plain function call.

        The history may come from storage or a UI, so the stored data is validated. The original call isn't sent back
        if a hook or an approval changed the part's actions or tool name, since that would tell Gemini that its own
        call ran. The caller checks that the part came from a transport that accepts it back.
        """
        replayed = self._stored_call(part)
        return replayed[0] if replayed and part.args_as_dict() == {'actions': replayed[1]} else None

    def _stored_call(
        self, part: ToolCallPart
    ) -> tuple[GeminiFunctionCall, list[_ActionJson], tuple[str, ...], tuple[str, ...]] | None:
        """Gemini's call stored on a `computer` call, its actions, and the modifiers held after and before it."""
        if part.tool_name != self.name:
            return None
        try:
            replay = _REPLAY_ADAPTER.validate_python(part.provider_details)
        except ValidationError:
            return None
        call = replay['gemini_call']
        actions, _, held = map_computer_call(
            call['name'], call['args'], replay['display_size'], replay['held_modifiers']
        )
        return call, actions, held, replay['held_modifiers']


def map_computer_result(
    response: Mapping[str, object], call: GeminiFunctionCall, part: ToolReturnPart
) -> dict[str, object]:
    """The function response for a replayed call, with the `safety_acknowledgement` Gemini asked for, if it asked.

    Gemini refuses a response to a flagged call that has none, so every response to a flagged call carries one. It is
    `True` when the runner set `confirmed` on the result, because a person approved the call and it ran, even if one of
    its actions failed. A denial, or an approved call that raised before returning a result, sends `False` beside the
    error.
    """
    if _confirmation_reason(call['args']) is None:
        return dict(response)
    match part.metadata:
        # A mapping with `confirmed` set to exactly `True`: a pattern compares `True` by identity.
        case {'confirmed': True}:
            return {**response, 'safety_acknowledgement': True}
        case _:
            return {'error': part.model_response_str(wrap_if_error=False), 'safety_acknowledgement': False}
