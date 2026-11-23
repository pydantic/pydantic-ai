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
from ._computer_calls import x11_key

_COMPUTER_USE_MODELS = ('gemini-4-', 'gemini-3.8-flash', 'gemini-3.5-flash-lite', 'gemini-3-flash-preview')
"""The models Google lists for computer use: https://ai.google.dev/gemini-api/docs/computer-use#model-versions."""


def is_computer_use_model(model_name: str) -> bool:
    """Whether Google lists `model_name` as a computer use model."""
    return model_name.startswith(_COMPUTER_USE_MODELS) and 'image' not in model_name


class GeminiCall(TypedDict):
    """One call Gemini made to a function of Google's computer use tool, unchanged."""

    name: str
    args: dict[str, JsonValue]


class ComputerReplay(TypedDict):
    """What a `computer` call stores in `provider_details`, so Gemini's original call can be sent back unchanged."""

    gemini_call: GeminiCall
    display_size: tuple[int, int]
    """The size of the screenshot Gemini pointed on, which converted its positions to pixels."""
    held: tuple[str, ...]
    """The modifiers Gemini's earlier `key_down` calls held when this call ran, which it ran with."""


_REPLAY = TypeAdapter(ComputerReplay)


class _Strict(BaseModel, extra='forbid', strict=True):
    """A function's arguments as Gemini sends them, checked strictly: an unknown field or a string for a number fails."""


class _At(_Strict):
    x: int
    y: int


class _DragAndDrop(_Strict):
    start_x: int
    start_y: int
    end_x: int
    end_y: int


class _Scroll(_Strict):
    x: int
    y: int
    direction: str
    magnitude_in_wheel_clicks: int | None = None


class _Type(_Strict):
    text: str
    press_enter: bool | None = None


class _Wait(_Strict):
    seconds: float | None = None


class _Key(_Strict):
    key: str


class _Hotkey(_Strict):
    keys: list[str]


class _NoArgs(_Strict):
    pass


_CLICKS = {
    'click': ('left', 1),
    'double_click': ('left', 2),
    'triple_click': ('left', 3),
    'middle_click': ('middle', 1),
    'right_click': ('right', 1),
}

_FUNCTIONS: dict[str, type[_Strict]] = {
    **dict.fromkeys(_CLICKS, _At),
    'move': _At,
    'mouse_down': _At,
    'mouse_up': _At,
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

COMPUTER_FUNCTIONS = frozenset(_FUNCTIONS)
"""The names of the desktop environment's functions, which no function tool may have beside Google's tool."""

_ACTION_TYPES = {
    **dict.fromkeys(_CLICKS, 'click'),
    'move': 'mouse_move',
    'drag_and_drop': 'drag',
    'type': 'type_text',
    'press_key': 'press_keys',
    'hotkey': 'press_keys',
    'key_down': 'screenshot',
    'key_up': 'screenshot',
    'take_screenshot': 'screenshot',
}
"""The `computer` tool action type for each function whose name is different."""


_Confirmation = str | None
_Action = dict[str, object]
"""One `computer` tool action, as the JSON the `computer` tool validates."""


def _confirmation(args: dict[str, JsonValue]) -> _Confirmation | ValueError:
    """Gemini's reason a call needs a person's confirmation, or `None` when it needs none.

    A decision other than `require_confirmation` gives an error, and the call must not run.
    """
    match args.get('safety_decision'):
        case None:
            return None
        case {'decision': 'require_confirmation', **rest}:
            explanation = rest.get('explanation')
            return (
                explanation if isinstance(explanation, str) and explanation else 'Gemini asks to confirm this action.'
            )
        case _:
            return ValueError('safety_decision')


def _modifier(key: str) -> str | None:
    """The modifier `key` names, such as `shift` for `Shift_L`, or `None` for any other key."""
    name = x11_key(key)
    return name if name in get_args(Modifier) else None


def _px(value: int, side: int) -> int:
    """A position on Gemini's 0-999 grid as a pixel on a side `side` pixels long."""
    return value * side // 1000


def map_gemini_call(
    name: str, args: dict[str, JsonValue], display_size: tuple[int, int], held: tuple[str, ...] = ()
) -> tuple[list[_Action], _Confirmation, tuple[str, ...]]:
    """Convert one of Gemini's calls, made with the modifiers `held`, into `computer` tool actions.

    Also returns its confirmation reason, if any, and the modifiers held after it, which only a `key_down` or `key_up`
    that maps changes. A function, field or safety decision with no matching `computer` action becomes one action the
    `computer` tool refuses, naming what didn't match. Gemini is then asked again, and nothing runs.
    """
    kind = _ACTION_TYPES.get(name, name)
    confirmation = _confirmation(args)
    if isinstance(confirmation, ValueError):
        return [{'type': kind, 'unsupported_fields': [str(confirmation)]}], None, held
    schema = _FUNCTIONS.get(name)
    rest = {key: value for key, value in args.items() if key not in ('intent', 'safety_decision')}
    if schema is None:
        return [{'type': kind, 'unsupported_fields': sorted(rest)}], confirmation, held
    try:
        call = schema.model_validate(rest)
    except ValidationError as error:
        fields = sorted({str(detail['loc'][0]) for detail in error.errors() if detail['loc']})
        return [{'type': kind, 'unsupported_fields': fields}], confirmation, held
    width, height = display_size

    def at(x: int, y: int) -> _Action:
        return {'x': _px(x, width), 'y': _px(y, height)}

    actions = _actions(name, call, at, held)
    if isinstance(call, _Key) and name in ('key_down', 'key_up') and (modifier := _modifier(call.key)):
        released = tuple(key for key in held if key != modifier)
        held = (*released, modifier) if name == 'key_down' else released
    return actions, confirmation, held


def _actions(name: str, call: _Strict, at: Callable[[int, int], _Action], held: tuple[str, ...]) -> list[_Action]:
    """The `computer` tool actions for Gemini's validated call to the function `name`, with the modifiers `held`."""
    holding: _Action = {'modifiers': list(held)} if held else {}
    match call:
        case _At() if name in _CLICKS:
            button, count = _CLICKS[name]
            return [{'type': 'click', **at(call.x, call.y), 'button': button, 'count': count, **holding}]
        case _At() if name == 'move':
            return [{'type': 'mouse_move', **at(call.x, call.y), **holding}]
        case _At() if held and name == 'mouse_down':
            return [{'type': name, 'unsupported_fields': ["can't hold a key through mouse_down"]}]
        case _At():
            # `mouse_down` and `mouse_up` press or release the left button at a point, so the pointer moves there first.
            return [{'type': 'mouse_move', **at(call.x, call.y)}, {'type': name}]
        case _DragAndDrop():
            return [{'type': 'drag', 'path': [at(call.start_x, call.start_y), at(call.end_x, call.end_y)], **holding}]
        case _Scroll():
            action: _Action = {'type': 'scroll', **at(call.x, call.y), 'direction': call.direction, **holding}
            if call.magnitude_in_wheel_clicks is not None:
                action['amount'] = call.magnitude_in_wheel_clicks
            return [action]
        case _Type():
            enter: list[_Action] = [{'type': 'press_keys', 'keys': ['enter']}] if call.press_enter else []
            return [{'type': 'type_text', 'text': call.text}, *enter]
        case _Wait():
            return [{'type': 'wait'} if call.seconds is None else {'type': 'wait', 'seconds': call.seconds}]
        case _Key() | _Hotkey() if name in ('press_key', 'hotkey'):
            # Gemini sends X11 key names, as Claude does.
            keys = [x11_key(key) for key in (call.keys if isinstance(call, _Hotkey) else [call.key])]
            # A held modifier that the call names is pressed once.
            return [{'type': 'press_keys', 'keys': [*(key for key in held if key not in keys), *keys]}]
        case _Key() if name == 'key_down' and _modifier(call.key) is None:
            return [{'type': 'screenshot', 'unsupported_fields': [f'key {call.key!r}, which is not a modifier']}]
        case _Key():
            # Nothing is pressed on the screen, and Gemini gets a screenshot back: `map_gemini_call` holds the key.
            return [{'type': 'screenshot'}]
        case _:  # `take_screenshot` takes no arguments
            return [{'type': 'screenshot'}]


@dataclass
class NativeComputer:
    """The `computer` tool, and Google's computer use tool sent in its place. Built from a `ComputerTool`."""

    name: str
    display_size: tuple[int, int]
    held: tuple[str, ...] = ()
    """The modifiers Gemini holds for its next call: each `key_down` that no `key_up` let go of."""

    def hold_after(self, messages: Sequence[ModelMessage]) -> None:
        """Hold the modifiers Gemini held after its last computer use call in `messages`, which stored them."""
        parts = (part for m in reversed(messages) if isinstance(m, ModelResponse) for part in reversed(m.parts))
        last = next((r for part in parts if isinstance(part, ToolCallPart) and (r := self._replayed(part))), None)
        self.held = last[2] if last else ()

    @staticmethod
    def tool() -> ToolDict:
        return ToolDict(computer_use=ComputerUseDict(environment=Environment.ENVIRONMENT_DESKTOP))

    @staticmethod
    def answers(function_name: str | None) -> bool:
        """Whether a function call is Gemini's call to Google's computer use tool."""
        return function_name in COMPUTER_FUNCTIONS

    def call_part(
        self, name: str, args: dict[str, JsonValue], tool_call_id: str | None, provider_name: str
    ) -> ComputerCallPart:
        """Turn Gemini's call into a `computer` call in pixels, keeping the call to send back unchanged."""
        actions, confirmation, held = map_gemini_call(name, args, self.display_size, self.held)
        details: dict[str, JsonValue] = {
            'gemini_call': {'name': name, 'args': args},
            'display_size': list[JsonValue](self.display_size),
            'held': list[JsonValue](self.held),
        }
        self.held = held
        if confirmation is not None:
            details['confirmation'] = confirmation
        part = ComputerCallPart(self.name, {'actions': actions}, provider_name=provider_name, provider_details=details)
        if tool_call_id is not None:
            part.tool_call_id = tool_call_id
        return part

    def replayed_call(self, part: ToolCallPart) -> GeminiCall | None:
        """Gemini's original call behind a `computer` call, or `None` to send it back as a plain function call.

        The history may come from storage or a UI, so the stored data is validated. The original call isn't sent back
        if a hook or an approval changed the part's actions or tool name, since that would tell Gemini that its own
        call ran. The caller checks that the part came from a transport that accepts it back.
        """
        replayed = self._replayed(part)
        return replayed[0] if replayed and part.args_as_dict() == {'actions': replayed[1]} else None

    def _replayed(self, part: ToolCallPart) -> tuple[GeminiCall, list[_Action], tuple[str, ...]] | None:
        """Gemini's call stored on a `computer` call, its actions, and the modifiers held after it, if one is stored."""
        if part.tool_name != self.name:
            return None
        try:
            replay = _REPLAY.validate_python(part.provider_details)
        except ValidationError:
            return None
        call = replay['gemini_call']
        actions, _, held = map_gemini_call(call['name'], call['args'], replay['display_size'], replay['held'])
        return call, actions, held


def acknowledge(response: Mapping[str, object], call: GeminiCall, part: ToolReturnPart) -> dict[str, object]:
    """The function response for a replayed call, with the `safety_acknowledgement` Gemini asked for, if it asked.

    Gemini refuses a response to a flagged call that has none, so every response to a flagged call carries one. It is
    `True` only when the runner set `confirmed` on the result, because a person approved the call and it ran. A
    success without it, such as from a runner that never asks, sends `False` beside the result. Any other result, such
    as a denial or an approved call that then failed, sends `False` beside the error.
    """
    if _confirmation(call['args']) is None:
        return dict(response)
    match part.metadata:
        # A mapping with `confirmed` set to exactly `True`: a pattern compares `True` by identity.
        case {'confirmed': True}:
            return {**response, 'safety_acknowledgement': True}
        case _ if part.outcome == 'success':
            return {**response, 'safety_acknowledgement': False}
        case _:
            return {'error': part.model_response_str(wrap_if_error=False), 'safety_acknowledgement': False}
