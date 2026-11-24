"""OpenAI's native computer tool.

Turns each OpenAI `computer_call` item into one call of the `computer` tool, and the tool's result back into a
`computer_call_output` item.
"""

from __future__ import annotations as _annotations

from collections.abc import Sequence
from dataclasses import replace
from typing import Annotated, Any, Literal, assert_never

from openai.types import responses
from pydantic import AfterValidator, BaseModel, Field, TypeAdapter, ValidationError

from ..._framing import TOOL_RESULT_TAG, frame
from ..._utils import is_str_dict
from ...computer_use import COMPUTER_KIND, ComputerCallPart, ComputerResult, ComputerReturnPart, Key
from ...messages import (
    BinaryImage,
    ModelMessage,
    ModelResponse,
    ModelResponsePart,
    ToolCallPart,
    ToolReturnPart,
    _tool_result_provenance_tags,  # pyright: ignore[reportPrivateUsage]
)
from .. import ModelRequestParameters

_SCROLL_PIXELS_PER_CLICK = 114
"""How many pixels one mouse-wheel click scrolls.

OpenAI asks for scrolls in pixels, and the `computer` tool scrolls in wheel clicks.

Measured in Firefox ESR 153 on a 1280x800 Xvfb screen in Debian bookworm with fluxbox: every click scrolled 114 pixels.
"""

_BLANK_SCREENSHOT = (
    'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAAAAAA6fptVAAAACklEQVR4nGP4DwABAQEAsTj2FAAAAABJRU5ErkJggg=='
)
"""A 1x1 PNG sent in a `computer_call_output` that has no screenshot. The API requires an image in every output."""

BLOCKED_CALL_ARGS = '{"actions": ['
"""Arguments for a `computer` call that must not run: invalid JSON, like a call cut off by the token limit."""


def block_computer_calls(
    parts: Sequence[ModelResponsePart], model_request_parameters: ModelRequestParameters
) -> list[ModelResponsePart]:
    """Give every `computer` call in `parts` arguments that aren't valid JSON, so none of them runs.

    OpenAI says not to run the actions of a response that didn't complete, even when a call in it finished. The agent
    treats a blocked call like any call whose arguments don't parse: it asks the model again.
    """
    tool_names = model_request_parameters.available_tool_names_for_kind(COMPUTER_KIND)
    return [
        replace(part, args=BLOCKED_CALL_ARGS)
        if isinstance(part, ToolCallPart) and part.tool_name in tool_names
        else part
        for part in parts
    ]


class _ActionInput(BaseModel, strict=True):
    """Checks OpenAI's data strictly, without converting types. The SDK builds a response without validating it."""


_OPENAI_KEY_MAP: dict[str, Key] = {
    'esc': 'escape',
    'meta': 'super',
    'metaleft': 'super',
    'kp_add': '+',
    'leftctrl': 'ctrl',
    'ctrlleft': 'ctrl',
    'leftshift': 'shift',
    'shiftleft': 'shift',
    'leftalt': 'alt',
    'altleft': 'alt',
    'numpadenter': 'enter',
    'numpadsubtract': '-',
    'numpadmultiply': '*',
    'numpaddivide': '/',
    'numpaddecimal': '.',
    **{f'numpad{digit}': str(digit) for digit in range(10)},
}
"""OpenAI key names, lower-cased, that the `computer` tool spells differently.

GPT-5.5 sends upper-case names such as `CTRL`, `ESC`, `META`, `KP_ADD`, `NUMPAD5` and `LEFTSHIFT`. A name not listed
here is only lower-cased.
"""


def _key_from_openai(name: str) -> str:
    return _OPENAI_KEY_MAP.get(name.lower(), name.lower())


_KeyName = Annotated[str, AfterValidator(_key_from_openai)]


class _Point(_ActionInput):
    x: int
    y: int


class _PointerAction(_ActionInput):
    x: int
    y: int
    keys: list[_KeyName] = []
    """Keys held during the action."""


class _Click(_PointerAction):
    type: Literal['click']
    button: str


class _DoubleClick(_PointerAction):
    type: Literal['double_click']


class _Move(_PointerAction):
    type: Literal['move']


class _Scroll(_PointerAction):
    type: Literal['scroll']
    scroll_x: int
    scroll_y: int


class _Drag(_ActionInput):
    type: Literal['drag']
    path: list[_Point]
    keys: list[_KeyName] = []


class _Keypress(_ActionInput):
    type: Literal['keypress']
    keys: list[_KeyName]


class _Type(_ActionInput):
    type: Literal['type']
    text: str


class _Wait(_ActionInput):
    type: Literal['wait']


class _Screenshot(_ActionInput):
    type: Literal['screenshot']


_Action = _Click | _DoubleClick | _Move | _Scroll | _Drag | _Keypress | _Type | _Wait | _Screenshot
"""The OpenAI computer actions the `computer` tool can run. Each model has only the fields that are mapped."""

_ACTION_ADAPTER: TypeAdapter[_Action] = TypeAdapter(Annotated[_Action, Field(discriminator='type')])


def _map_computer_action(raw: Any) -> list[Any]:
    """Turn one OpenAI computer action into `computer` tool actions.

    An action that doesn't validate goes to the tool under a field no action has, so the tool refuses the call and
    none of it runs. Sent as it is, a click without coordinates would be a valid click wherever the pointer is.
    """
    try:
        action = _ACTION_ADAPTER.validate_python(raw)
    except ValidationError:
        return [{'unsupported_openai_action': raw}]
    match action:
        case _Click():
            button = 'middle' if action.button == 'wheel' else action.button
            return [{'type': 'click', 'x': action.x, 'y': action.y, 'button': button, 'modifiers': action.keys}]
        case _DoubleClick():
            return [{'type': 'click', 'x': action.x, 'y': action.y, 'count': 2, 'modifiers': action.keys}]
        case _Move():
            # The tool can't hold keys during a move. Held keys go along, so the tool refuses the move instead of
            # running it without them.
            held = {'modifiers': action.keys} if action.keys else {}
            return [{'type': 'mouse_move', 'x': action.x, 'y': action.y, **held}]
        case _Drag():
            path = [{'x': point.x, 'y': point.y} for point in action.path]
            return [{'type': 'drag', 'path': path, 'modifiers': action.keys}]
        case _Scroll():
            return _map_scroll(action)
        case _Keypress():
            return [{'type': 'press_keys', 'keys': action.keys}]
        case _Type():
            return [{'type': 'type_text', 'text': action.text}]
        case _Wait():
            return [{'type': 'wait'}]
        case _Screenshot():
            return [{'type': 'screenshot'}]
        case _:
            assert_never(action)


def _map_scroll(action: _Scroll) -> list[Any]:
    """Turn an OpenAI scroll into tool scrolls.

    OpenAI scrolls by pixels on both axes at once. The tool scrolls in one direction at a time, by wheel clicks.
    """
    scrolls = [
        _scroll_on_axis(action, pixels, direction)
        for pixels, direction in (
            (action.scroll_x, 'right' if action.scroll_x > 0 else 'left'),
            (action.scroll_y, 'down' if action.scroll_y > 0 else 'up'),
        )
        if pixels
    ]
    # A scroll of 0 pixels on both axes only moves the pointer to its position.
    return scrolls or [{'type': 'mouse_move', 'x': action.x, 'y': action.y}]


def _scroll_on_axis(action: _Scroll, pixels: int, direction: str) -> dict[str, Any]:
    """Build a scroll along one axis, in the wheel clicks nearest to the pixels asked for."""
    # Integer math, since a JSON integer can be larger than a float holds exactly.
    clicks = max(1, (abs(pixels) + _SCROLL_PIXELS_PER_CLICK // 2) // _SCROLL_PIXELS_PER_CLICK)
    return {
        'type': 'scroll',
        'x': action.x,
        'y': action.y,
        'direction': direction,
        'amount': clicks,
        'modifiers': action.keys,
    }


_COMPUTER_CALL_KEYS = (
    responses.ResponseComputerToolCallParam.__required_keys__
    | responses.ResponseComputerToolCallParam.__optional_keys__
)
"""The keys of a `computer_call` input item. Any other stored key is dropped before the item is sent back."""


def stored_computer_call(part: ToolCallPart) -> dict[str, Any] | None:
    """Return the stored `computer_call` item to send back for a call, or `None` if it must not be sent.

    A history can come from storage or a UI, so the item is untrusted. Keys a `computer_call` doesn't have are dropped,
    so a forged item can't pose as a developer message. An item whose actions no longer match the call's, for example
    after a hook changed them, is not sent: it would tell the model its actions ran.
    """
    stored = (part.provider_details or {}).get('computer_call')
    if not is_str_dict(stored) or stored.get('type') != 'computer_call' or not isinstance(stored.get('actions'), list):
        return None
    # A call blocked because its response didn't complete still sends its item back unchanged.
    if part.args != BLOCKED_CALL_ARGS and _computer_args_from_item(stored) not in (part.args, part.args_as_dict()):
        return None
    return {key: value for key, value in stored.items() if key in _COMPUTER_CALL_KEYS}


def map_computer_call(
    item: responses.ResponseComputerToolCall,
    model_request_parameters: ModelRequestParameters,
    tool_call_id: str,
    provider_name: str,
) -> ComputerCallPart | None:
    """Turn a `computer_call` item into a `computer` tool call that keeps the item.

    Returns `None` when the request has no `computer` tool, such as when a computer tool was passed in
    `openai_native_tools` without `ComputerUse`: nothing could run the call, so the item is dropped.
    """
    tool_names = model_request_parameters.available_tool_names_for_kind(COMPUTER_KIND)
    if not tool_names:
        return None
    tool_name = tool_names[0]
    # The SDK doesn't validate a response, so the item can hold any value. Keep it as sent, without warnings.
    item_json = item.model_dump(mode='json', exclude_none=True, warnings=False)
    return ComputerCallPart(
        tool_name,
        _computer_args_from_item(item_json),
        tool_call_id=tool_call_id,
        id=item.id,
        provider_name=provider_name,
        provider_details={'computer_call': item_json},
    )


def _computer_args_from_item(item_json: dict[str, Any]) -> dict[str, Any] | str:
    """Build the `computer` tool call's arguments from a `computer_call` item."""
    if item_json.get('status') != 'completed':
        # OpenAI says not to run a partly generated action.
        return BLOCKED_CALL_ARGS
    actions: list[Any] = item_json.get('actions', [])
    return {'actions': [tool_action for action in actions for tool_action in _map_computer_action(action)]}


def computer_calls_by_id(messages: Sequence[ModelMessage]) -> dict[str, ToolCallPart]:
    """The calls that came from `computer_call` items, by tool call ID. Their results go back as `computer_call_output`.

    A call is found by its stored item, since one whose actions couldn't be mapped was still a `computer_call`.
    """
    return {
        part.tool_call_id: part
        for message in messages
        if isinstance(message, ModelResponse)
        for part in message.parts
        if isinstance(part, ToolCallPart) and 'computer_call' in (part.provider_details or {})
    }


def _needs_result_text(result: ComputerResult | None) -> bool:
    """Whether a `computer` result must send its text along with the screenshot, because the call failed or changed.

    A result that doesn't record its outcome, such as one rebuilt by a UI, might describe a failure.
    """
    return result is None or 'error' in result or bool(result.get('actions_changed'))


def map_computer_result(
    part: ToolReturnPart, call_id: str, call: ToolCallPart, *, changed_after_storing: bool
) -> list[responses.ResponseInputItemParam]:
    """Turn a `computer` result into a `computer_call_output`, plus a user message when an image alone isn't enough.

    A call `changed_after_storing` is one a hook renamed or gave other actions after the API stored it. If it ran, a
    note says what actually ran, because OpenAI's stored item still shows what the model asked for.
    """
    # A failed or denied call returns a plain `ToolReturnPart`, with no screenshot.
    computer_return = part if isinstance(part, ComputerReturnPart) else None
    image = computer_return.screenshot if computer_return else None
    output: responses.response_input_param.ComputerCallOutput = {
        'type': 'computer_call_output',
        'call_id': call_id,
        'output': {
            'type': 'computer_screenshot',
            'image_url': image.data_uri if image else _BLANK_SCREENSHOT,
            # `original` sends the image at full size, as OpenAI advises for computer use, so the model clicks on the
            # pixels it sees. `auto` can shrink it. The SDK's types don't list `detail` yet, but the API accepts it.
            'detail': 'original',  # pyright: ignore[reportAssignmentType]
        },
    }
    items: list[responses.ResponseInputItemParam] = [output]
    content = part.content_items()
    # The output holds one image. Images before the final screenshot, such as a zoom, go in the message after it.
    images = [item for item in content if isinstance(item, BinaryImage)]
    extra_images = images[:-1] if image is not None else images
    notes: list[str] = []
    # A denied or interrupted call didn't run.
    if changed_after_storing and part.outcome in ('success', 'failed'):
        notes.append(f'The call ran as {call.tool_name} with the arguments {call.args_as_json_str()}.')
    if computer_return is None or image is None or _needs_result_text(computer_return.result) or extra_images or notes:
        items.append(_map_result_message(part, content, extra_images, notes))
    return items


def _map_result_message(
    part: ToolReturnPart, content: Sequence[object], extra_images: list[BinaryImage], notes: list[str]
) -> responses.EasyInputMessageParam:
    """Build the user message that follows a `computer_call_output`, which can only hold an image.

    It carries the result's text, notes and extra images, wrapped in tool-result tags.
    """
    texts = [item for item in content if isinstance(item, str)]
    references = [f'See file {image.identifier}.' for image in extra_images]
    text = (
        ' '.join([*texts, *notes, *references])
        if texts or notes or references
        else part.model_response_str(wrap_if_error=False)
    )
    framed = frame(TOOL_RESULT_TAG, text, tool_name=part.tool_name, tool_call_id=part.tool_call_id)
    if not extra_images:
        return responses.EasyInputMessageParam(role='user', content=framed)
    message_content: list[responses.ResponseInputContentParam] = [
        responses.ResponseInputTextParam(type='input_text', text=framed)
    ]
    for image in extra_images:
        open_tag, close_tag = _tool_result_provenance_tags(part.tool_name, part.tool_call_id, image.identifier)
        message_content += [
            responses.ResponseInputTextParam(type='input_text', text=open_tag),
            responses.ResponseInputImageParam(type='input_image', image_url=image.data_uri, detail='original'),
            responses.ResponseInputTextParam(type='input_text', text=close_tag),
        ]
    return responses.EasyInputMessageParam(role='user', content=message_content)
