"""OpenAI's native `computer` tool: its `computer_call` items as calls of the `computer` tool, and back."""

from __future__ import annotations as _annotations

from collections.abc import Sequence
from typing import Annotated, Any, Literal, assert_never

from openai.types import responses
from pydantic import BaseModel, Field, TypeAdapter, ValidationError

from ..._framing import TOOL_RESULT_TAG, frame
from ..._utils import is_str_dict
from ...computer_use import COMPUTER_KIND, ComputerCallPart, ComputerResult, ComputerReturnPart
from ...messages import BinaryImage, ModelMessage, ModelResponse, ToolCallPart, ToolReturnPart
from .. import ModelRequestParameters

_SCROLL_PIXELS_PER_CLICK = 114
"""OpenAI scrolls by pixels and the `computer` tool by wheel clicks; one click scrolls about this far.

Measured in Firefox ESR 153 on `DockerContainer(desktop=True)`'s 1280x800 desktop: every click scrolled 114 pixels.
"""

_BLANK_SCREENSHOT = (
    'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAAAAAA6fptVAAAACklEQVR4nGP4DwABAQEAsTj2FAAAAABJRU5ErkJggg=='
)
"""A 1x1 PNG for a `computer_call_output` with no screenshot to send, since the API requires one."""


class _Strict(BaseModel, extra='forbid', strict=True):
    """OpenAI's own data, checked as is: the SDK builds a response without validating it."""


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
"""The OpenAI computer actions the `computer` tool can run, each with only the fields it maps."""

_OPENAI_ACTION: TypeAdapter[_OpenAIAction] = TypeAdapter(Annotated[_OpenAIAction, Field(discriminator='type')])


def _map_computer_action(raw: Any) -> list[tuple[dict[str, Any], str | None]]:
    """One OpenAI computer action as `computer` tool actions, each with a note when it runs differently than asked.

    An invalid action becomes one the tool refuses, naming the bad fields, so none of it runs and the model retries.
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
            return [({'type': 'move', 'x': action.x, 'y': action.y, 'modifiers': action.keys}, None)]
        case _OpenAIDrag():
            path = [{'x': point.x, 'y': point.y} for point in action.path]
            return [({'type': 'drag', 'path': path, 'modifiers': action.keys}, None)]
        case _OpenAIScroll():
            return _map_scroll(action)
        case _OpenAIKeypress():
            return [({'type': 'keypress', 'keys': action.keys}, None)]
        case _OpenAIType():
            return [({'type': 'type', 'text': action.text}, None)]
        case _OpenAIWait():
            return [({'type': 'wait'}, None)]
        case _OpenAIScreenshot():
            return [({'type': 'screenshot'}, None)]
        case _:
            assert_never(action)


def _refused_action(error: ValidationError) -> dict[str, Any]:
    """An action the tool refuses, naming the fields that didn't validate."""
    # A field's error location starts with the action type. An error with no location is about the action as a
    # whole: a type that isn't one comes back as the error's tag, and an action that isn't an object has none.
    details = error.errors()
    fields = sorted({str(detail['loc'][1]) for detail in details if len(detail['loc']) > 1})
    kind = details[0]['loc'][0] if details[0]['loc'] else (details[0].get('ctx') or {}).get('tag')
    # Under the tool's own type, so the retry names the fields rather than refusing the type.
    return {'type': 'click' if kind == 'double_click' else kind, 'unsupported_fields': fields}


def _map_scroll(action: _OpenAIScroll) -> list[tuple[dict[str, Any], str | None]]:
    """OpenAI scrolls by pixels on both axes at once; the tool scrolls one way at a time, in wheel clicks."""
    scrolls = [
        _scroll(action, axis, pixels, direction)
        for axis, pixels, direction in (
            ('scroll_x', action.scroll_x, 'right' if action.scroll_x > 0 else 'left'),
            ('scroll_y', action.scroll_y, 'down' if action.scroll_y > 0 else 'up'),
        )
        if pixels
    ]
    # A scroll by nothing only moves the pointer there.
    return scrolls or [({'type': 'move', 'x': action.x, 'y': action.y, 'modifiers': action.keys}, None)]


def _scroll(action: _OpenAIScroll, axis: str, pixels: int, direction: str) -> tuple[dict[str, Any], str | None]:
    """A scroll along one axis in wheel clicks, with a note when the clicks don't match the pixels asked for."""
    # Rounded half up in whole numbers: any JSON integer is valid, and a float can't hold them all.
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
"""The keys a `computer_call` input item has; any other stored key is dropped before the item is sent back."""


def stored_computer_call(part: ToolCallPart) -> dict[str, Any] | None:
    """The `computer_call` item a call came as, to send back, or `None` to send the call back plain.

    A history may come from storage or a UI, so an item of another type is not sent, nor is any key a `computer_call`
    doesn't have: otherwise a forged item could be a message from the developer. The API judges the values.
    """
    stored = (part.provider_details or {}).get('computer_call')
    if not is_str_dict(stored) or stored.get('type') != 'computer_call':
        return None
    return {key: value for key, value in stored.items() if key in _COMPUTER_CALL_KEYS}


def map_computer_call(
    item: responses.ResponseComputerToolCall,
    model_request_parameters: ModelRequestParameters,
    tool_call_id: str,
    provider_name: str,
) -> ComputerCallPart | None:
    """A `computer_call` as a call of the `computer` tool, keeping the item to send back as it was.

    `None` when the request has no `computer` tool to run it.
    """
    if not (tool_name := model_request_parameters.tool_name_for_kind(COMPUTER_KIND)):
        return None
    # The SDK doesn't validate a response, so the item can hold anything; it is kept as sent, without a warning.
    dumped = item.model_dump(mode='json', exclude_none=True, warnings=False)
    args: dict[str, Any] | str
    if item.status != 'completed':
        # Cut off or still being generated: like a truncated function call, its arguments don't parse, so none of it
        # runs. OpenAI says not to run a partially generated action.
        args = '{"actions": ['
    else:
        args = {'actions': [action for action, _ in _map_actions(dumped)]}
    return ComputerCallPart(
        tool_name,
        args,
        tool_call_id=tool_call_id,
        id=item.id,
        provider_name=provider_name,
        provider_details={'computer_call': dumped},
    )


def _map_actions(dumped: dict[str, Any]) -> list[tuple[dict[str, Any], str | None]]:
    """A `computer_call` item's actions as `computer` tool actions, each with its note on how a pixel scroll runs."""
    actions: list[Any] = dumped.get('actions', [])
    if not isinstance(actions, list):
        # Not a list, even if it is one action: the call is malformed, so none of it runs.
        actions = [None]
    return [tool_action for action in actions for tool_action in _map_computer_action(action)]


def computer_calls_by_id(messages: Sequence[ModelMessage]) -> dict[str, ToolCallPart]:
    """The `computer_call`s this API made, by tool call ID, so their outputs can read them.

    A call is found by the item it came as, not by valid arguments: a call whose actions didn't map was still a
    `computer_call`, and its output must be a `computer_call_output`.
    """
    return {
        part.tool_call_id: part
        for message in messages
        if isinstance(message, ModelResponse)
        for part in message.parts
        if isinstance(part, ToolCallPart) and 'computer_call' in (part.provider_details or {})
    }


def _scroll_notes(call: ToolCallPart, result: ComputerResult | None) -> list[str]:
    """The notes on the pixel scrolls that ran, made again from the stored item's validated numbers.

    Not kept as text on the call, where a history from storage or a UI could put any text. Only the first `performed`
    actions ran. Nothing is said for a call changed before it ran (its summary covers it) or a result without its
    outcome.
    """
    if result is None or result.get('changed') or not (stored := stored_computer_call(call)):
        return []
    notes = [note for _, note in _map_actions(stored)]
    return [note for note in notes[: result['performed']] if note]


def _needs_summary(result: ComputerResult | None) -> bool:
    """Whether a `computer` result needs its text beside the screenshot, which alone would say the call ran.

    A result without its outcome, such as one a UI rebuilt, may describe a failure. A failed or changed call says so
    in its text.
    """
    return result is None or 'error' in result or bool(result.get('changed'))


def map_computer_call_output(
    part: ToolReturnPart, call_id: str, call: ToolCallPart
) -> list[responses.ResponseInputItemParam]:
    """A `computer` result as a `computer_call_output`, followed by a message when it needs more than an image.

    The API needs a screenshot in every output, so a result without one sends a blank image instead.
    """
    # A failed or denied call is a base part, without a screenshot.
    computer_return = part if isinstance(part, ComputerReturnPart) else None
    image = computer_return.screenshot if computer_return else None
    output: responses.response_input_param.ComputerCallOutput = {
        'type': 'computer_call_output',
        'call_id': call_id,
        'output': {
            'type': 'computer_screenshot',
            'image_url': image.data_uri if image else _BLANK_SCREENSHOT,
            # `original` keeps every pixel the model clicks on (OpenAI's advice for computer use); `auto` may
            # downscale. The SDK's types don't list `detail` yet, though the API accepts it.
            'detail': 'original',  # pyright: ignore[reportAssignmentType]
        },
    }
    items: list[responses.ResponseInputItemParam] = [output]
    # A successful result is read as validated content, so content stored as a JSON string reads the same as a list.
    content = (computer_return.typed_content if computer_return else None) or part.content_items()
    # Images before the closing screenshot, such as a zoom, can't go in the output.
    images = [item for item in content if isinstance(item, BinaryImage)]
    extra_images = images[:-1] if image is not None else images
    scroll_notes = _scroll_notes(call, computer_return.result) if computer_return else []
    if (
        computer_return is None
        or image is None
        or _needs_summary(computer_return.result)
        or extra_images
        or scroll_notes
    ):
        items.append(_summary_message(part, content, extra_images, scroll_notes))
    return items


def _summary_message(
    part: ToolReturnPart, content: Sequence[object], extra_images: list[BinaryImage], scroll_notes: list[str]
) -> responses.EasyInputMessageParam:
    """The output can only carry an image: say what didn't run, or why the image is blank.

    The note is the tool's result, so it is framed as one on the user channel, as for any tool result that can't go
    natively.
    """
    texts = [item for item in content if isinstance(item, str)]
    note = ' '.join([*texts, *scroll_notes]) if texts or scroll_notes else part.model_response_str(wrap_if_error=False)
    framed = frame(TOOL_RESULT_TAG, note, tool_name=part.tool_name, tool_call_id=part.tool_call_id)
    if not extra_images:
        return responses.EasyInputMessageParam(role='user', content=framed)
    message_content: list[responses.ResponseInputContentParam] = [
        responses.ResponseInputTextParam(type='input_text', text=framed),
        *(
            responses.ResponseInputImageParam(type='input_image', image_url=extra_image.data_uri, detail='original')
            for extra_image in extra_images
        ),
    ]
    return responses.EasyInputMessageParam(role='user', content=message_content)
