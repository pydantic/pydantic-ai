"""OpenAI's native computer tool.

Turns OpenAI's `computer_call` items into calls of the `computer` tool, and the tool's results back into
`computer_call_output` items.
"""

from __future__ import annotations as _annotations

from collections.abc import Sequence
from dataclasses import replace
from typing import Annotated, Any, Literal, assert_never

from openai.types import responses
from openai.types.responses.response_status import ResponseStatus
from pydantic import BaseModel, Field, TypeAdapter, ValidationError

from ..._framing import TOOL_RESULT_TAG, frame
from ..._utils import is_str_dict
from ...computer_use import COMPUTER_KIND, ComputerCallPart, ComputerResult, ComputerReturnPart
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


class _Strict(BaseModel, extra='forbid', strict=True):
    """Checks OpenAI's data strictly, without converting types. The SDK builds a response without validating it."""


class _OpenAIPoint(_Strict):
    x: int
    y: int


class _OpenAIPointerAction(_Strict):
    x: int
    y: int
    keys: list[str] = []
    """Keys held during the action."""


class _OpenAIClick(_OpenAIPointerAction):
    type: Literal['click']
    button: str


class _OpenAIDoubleClick(_OpenAIPointerAction):
    type: Literal['double_click']


class _OpenAIMove(_OpenAIPointerAction):
    type: Literal['move']


class _OpenAIScroll(_OpenAIPointerAction):
    type: Literal['scroll']
    scroll_x: int
    scroll_y: int


class _OpenAIDrag(_Strict):
    type: Literal['drag']
    path: list[_OpenAIPoint]
    keys: list[str] = []


class _OpenAIKeypress(_Strict):
    type: Literal['keypress']
    keys: list[str]


class _OpenAIType(_Strict):
    type: Literal['type']
    text: str


class _OpenAIWait(_Strict):
    type: Literal['wait']


class _OpenAIScreenshot(_Strict):
    type: Literal['screenshot']


_OpenAIAction = (
    _OpenAIClick
    | _OpenAIDoubleClick
    | _OpenAIMove
    | _OpenAIScroll
    | _OpenAIDrag
    | _OpenAIKeypress
    | _OpenAIType
    | _OpenAIWait
    | _OpenAIScreenshot
)
"""The OpenAI computer actions the `computer` tool can run. Each model has only the fields that are mapped."""

_OPENAI_ACTION: TypeAdapter[_OpenAIAction] = TypeAdapter(Annotated[_OpenAIAction, Field(discriminator='type')])


def _map_computer_action(raw: Any) -> list[tuple[dict[str, Any], str | None]]:
    """Turn one OpenAI computer action into `computer` tool actions, each with a note if it runs differently.

    An invalid action becomes one the tool refuses, naming the invalid fields, so none of the call runs.
    """
    try:
        action = _OPENAI_ACTION.validate_python(raw)
    except ValidationError as error:
        return [(_refused_action(error), None)]
    match action:
        case _OpenAIClick():
            button = 'middle' if action.button == 'wheel' else action.button
            return [({'type': 'click', 'x': action.x, 'y': action.y, 'button': button, 'modifiers': action.keys}, None)]
        case _OpenAIDoubleClick():
            return [({'type': 'click', 'x': action.x, 'y': action.y, 'count': 2, 'modifiers': action.keys}, None)]
        case _OpenAIMove():
            return [({'type': 'mouse_move', 'x': action.x, 'y': action.y, 'modifiers': action.keys}, None)]
        case _OpenAIDrag():
            path = [{'x': point.x, 'y': point.y} for point in action.path]
            return [({'type': 'drag', 'path': path, 'modifiers': action.keys}, None)]
        case _OpenAIScroll():
            return _map_scroll(action)
        case _OpenAIKeypress():
            return [({'type': 'press_keys', 'keys': action.keys}, None)]
        case _OpenAIType():
            return [({'type': 'type_text', 'text': action.text}, None)]
        case _OpenAIWait():
            return [({'type': 'wait'}, None)]
        case _OpenAIScreenshot():
            return [({'type': 'screenshot'}, None)]
        case _:
            assert_never(action)


_TOOL_TYPES: dict[object, str] = {
    'double_click': 'click',
    'move': 'mouse_move',
    'keypress': 'press_keys',
    'type': 'type_text',
}
"""The `computer` tool's name for each OpenAI action type that has a different name."""


def _refused_action(error: ValidationError) -> dict[str, Any]:
    """Build an action the tool refuses, naming the fields that failed validation."""
    # A field error's location starts with the action type. An unknown type comes back as the error's tag instead,
    # and an action that isn't an object has neither.
    details = error.errors()
    fields = sorted({str(detail['loc'][1]) for detail in details if len(detail['loc']) > 1})
    kind = details[0]['loc'][0] if details[0]['loc'] else (details[0].get('ctx') or {}).get('tag')
    # The tool's name for the type makes its refusal name the bad fields instead of the type.
    return {'type': _TOOL_TYPES.get(kind, kind), 'unsupported_fields': fields}


def _map_scroll(action: _OpenAIScroll) -> list[tuple[dict[str, Any], str | None]]:
    """Turn an OpenAI scroll into tool scrolls.

    OpenAI scrolls by pixels on both axes at once. The tool scrolls in one direction at a time, by wheel clicks.
    """
    scrolls = [
        _scroll(action, axis, pixels, direction)
        for axis, pixels, direction in (
            ('scroll_x', action.scroll_x, 'right' if action.scroll_x > 0 else 'left'),
            ('scroll_y', action.scroll_y, 'down' if action.scroll_y > 0 else 'up'),
        )
        if pixels
    ]
    # A scroll of 0 pixels on both axes only moves the pointer to its position.
    return scrolls or [({'type': 'mouse_move', 'x': action.x, 'y': action.y, 'modifiers': action.keys}, None)]


def _scroll(action: _OpenAIScroll, axis: str, pixels: int, direction: str) -> tuple[dict[str, Any], str | None]:
    """Build a scroll along one axis in wheel clicks, with a note when the clicks don't match the pixels asked for."""
    # Integer math, since a JSON integer can be larger than a float holds exactly.
    clicks = max(1, (abs(pixels) + _SCROLL_PIXELS_PER_CLICK // 2) // _SCROLL_PIXELS_PER_CLICK)
    scroll = {
        'type': 'scroll',
        'x': action.x,
        'y': action.y,
        'direction': direction,
        'amount': clicks,
        'modifiers': action.keys,
    }
    if clicks * _SCROLL_PIXELS_PER_CLICK == abs(pixels):
        return scroll, None
    return scroll, f'{axis}={pixels} ran as {clicks} wheel clicks of about {_SCROLL_PIXELS_PER_CLICK} pixels each.'


_COMPUTER_CALL_KEYS = (
    responses.ResponseComputerToolCallParam.__required_keys__
    | responses.ResponseComputerToolCallParam.__optional_keys__
)
"""The keys of a `computer_call` input item. Any other stored key is dropped before the item is sent back."""


_UNFINISHED_ARGS = '{"actions": ['
"""Arguments for a call that must not run: invalid JSON, like a call cut off by the token limit."""


def stored_computer_call(part: ToolCallPart) -> dict[str, Any] | None:
    """Return the stored `computer_call` item to send back for a call, or `None` if it must not be sent.

    A history can come from storage or a UI, so the item is untrusted. Keys a `computer_call` doesn't have are dropped,
    so a forged item can't pose as a developer message. An item whose actions no longer match the call's, for example
    after a hook changed them, is not sent: it would tell the model its actions ran.
    """
    stored = (part.provider_details or {}).get('computer_call')
    if not is_str_dict(stored) or stored.get('type') != 'computer_call':
        return None
    # A call blocked because its response didn't complete still sends its item back unchanged.
    if part.args != _UNFINISHED_ARGS and _call_args(stored) not in (part.args, part.args_as_dict()):
        return None
    return {key: value for key, value in stored.items() if key in _COMPUTER_CALL_KEYS}


def map_computer_call(
    item: responses.ResponseComputerToolCall,
    model_request_parameters: ModelRequestParameters,
    tool_call_id: str,
    provider_name: str,
) -> ComputerCallPart | None:
    """Turn a `computer_call` item into a `computer` tool call that keeps the item, or `None` without the tool."""
    # A real request always has the tool. The stream parity test replays recorded streams without tools.
    tool_name = next(iter(model_request_parameters.available_tool_names_for_kind(COMPUTER_KIND)), None)
    if tool_name is None:
        return None
    # The SDK doesn't validate a response, so the item can hold any value. Keep it as sent, without warnings.
    dumped = item.model_dump(mode='json', exclude_none=True, warnings=False)
    return ComputerCallPart(
        tool_name,
        _call_args(dumped),
        tool_call_id=tool_call_id,
        id=item.id,
        provider_name=provider_name,
        provider_details={'computer_call': dumped},
    )


def stop_computer_calls_unless_completed(
    parts: Sequence[ModelResponsePart],
    model_request_parameters: ModelRequestParameters,
    *,
    status: ResponseStatus | None,
    refused: bool,
) -> list[ModelResponsePart]:
    """Block every `computer` call in a response unless it completed and the model didn't refuse.

    OpenAI says not to run the actions of an incomplete, failed or refused response, even when a call in it finished.
    A stream learns the status only after sending its calls, so this runs on the final parts. A blocked call gets
    invalid JSON arguments, so the agent asks for it again like any call whose arguments don't parse (or raises
    `IncompleteToolCall` if it was cut off by the token limit). No status counts as completed, since some
    OpenAI-compatible APIs send none. Calls to a deferred `computer` tool are blocked too, since tool search can reveal
    it in the same response.
    """
    if status in (None, 'completed') and not refused:
        return list(parts)
    tool_names = {tool.name for tool in model_request_parameters.function_tools if tool.tool_kind == COMPUTER_KIND.id}
    return [
        replace(part, args=_UNFINISHED_ARGS)
        if isinstance(part, ToolCallPart) and part.tool_name in tool_names
        else part
        for part in parts
    ]


def _call_args(dumped: dict[str, Any]) -> dict[str, Any] | str:
    """Build the `computer` tool call's arguments from a `computer_call` item."""
    if dumped.get('status') != 'completed':
        # OpenAI says not to run a partly generated action.
        return _UNFINISHED_ARGS
    return {'actions': [action for action, _ in _map_actions(dumped)]}


def _map_actions(dumped: dict[str, Any]) -> list[tuple[dict[str, Any], str | None]]:
    """Turn a `computer_call` item's actions into `computer` tool actions, each with its note on a pixel scroll."""
    actions: list[Any] = dumped.get('actions', [])
    if not isinstance(actions, list):
        # A malformed call: none of it runs.
        actions = [None]
    return [tool_action for action in actions for tool_action in _map_computer_action(action)]


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


def _scroll_notes(call: ToolCallPart, result: ComputerResult | None) -> list[str]:
    """Rebuild the notes on the pixel scrolls that ran, from the stored item's validated numbers.

    Notes aren't stored as text on the call, since a history from storage or a UI could put any text there. A changed
    call gets none, since its summary covers it, and neither does a result that doesn't record its outcome.
    """
    if result is None or result.get('changed') or not (stored := stored_computer_call(call)):
        return []
    notes = [note for _, note in _map_actions(stored)]
    return [note for note in notes[: result['performed']] if note]


def _needs_summary(result: ComputerResult | None) -> bool:
    """Whether a `computer` result must send its text along with the screenshot, because the call failed or changed.

    A result that doesn't record its outcome, such as one rebuilt by a UI, might describe a failure.
    """
    return result is None or 'error' in result or bool(result.get('changed'))


def map_computer_call_output(
    part: ToolReturnPart, call_id: str, call: ToolCallPart, *, changed: bool
) -> list[responses.ResponseInputItemParam]:
    """Turn a `computer` result into a `computer_call_output`, plus a user message when an image alone isn't enough.

    A `changed` call is one a hook renamed or gave other actions after the API stored it. If it ran, a note says what
    actually ran, because OpenAI's stored item still shows what the model asked for.
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
    # Read a successful result through its validated content, so content stored as a JSON string reads as a list.
    content = (computer_return.typed_content if computer_return else None) or part.content_items()
    # The output holds one image. Images before the final screenshot, such as a zoom, go in the message after it.
    images = [item for item in content if isinstance(item, BinaryImage)]
    extra_images = images[:-1] if image is not None else images
    notes = _scroll_notes(call, computer_return.result) if computer_return else []
    # A denied or interrupted call didn't run.
    if changed and part.outcome in ('success', 'failed'):
        notes.append(f'The call ran as {call.tool_name} with the arguments {call.args_as_json_str()}.')
    if computer_return is None or image is None or _needs_summary(computer_return.result) or extra_images or notes:
        items.append(_summary_message(part, content, extra_images, notes))
    return items


def _summary_message(
    part: ToolReturnPart, content: Sequence[object], extra_images: list[BinaryImage], notes: list[str]
) -> responses.EasyInputMessageParam:
    """Build the user message that follows a `computer_call_output`, which can only hold an image.

    It carries the result's text, notes and extra images, wrapped in tool-result tags.
    """
    texts = [item for item in content if isinstance(item, str)]
    references = [f'See file {image.identifier}.' for image in extra_images]
    note = (
        ' '.join([*texts, *notes, *references])
        if texts or notes or references
        else part.model_response_str(wrap_if_error=False)
    )
    framed = frame(TOOL_RESULT_TAG, note, tool_name=part.tool_name, tool_call_id=part.tool_call_id)
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
