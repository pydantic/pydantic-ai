"""OpenAI's native `computer` tool: its `computer_call` items as calls of the `computer` tool, and back."""

from __future__ import annotations as _annotations

from collections.abc import Sequence
from typing import Annotated, Any, Literal, assert_never

from openai.types import responses
from pydantic import BaseModel, Field, TypeAdapter, ValidationError

from ..._framing import TOOL_RESULT_TAG, frame
from ...computer_use import (
    _SAFETY_CHECKS,  # pyright: ignore[reportPrivateUsage]
    ComputerCallPart,
    ComputerResult,
    ComputerReturnPart,
    SafetyCheck,
    _safety_checks,  # pyright: ignore[reportPrivateUsage]
)
from ...messages import BinaryImage, ModelMessage, ModelResponse, ToolCallPart, ToolReturnPart

_SCROLL_PIXELS_PER_CLICK = 100
"""OpenAI scrolls by pixels and the `computer` tool by wheel clicks; one click scrolls about this far."""

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


_OPENAI_ACTION: TypeAdapter[
    _OpenAIClick
    | _OpenAIDoubleClick
    | _OpenAIMove
    | _OpenAIScroll
    | _OpenAIDrag
    | _OpenAIKeypress
    | _OpenAIType
    | _OpenAIWait
    | _OpenAIScreenshot
] = TypeAdapter(
    Annotated[
        _OpenAIClick
        | _OpenAIDoubleClick
        | _OpenAIMove
        | _OpenAIScroll
        | _OpenAIDrag
        | _OpenAIKeypress
        | _OpenAIType
        | _OpenAIWait
        | _OpenAIScreenshot,
        Field(discriminator='type'),
    ]
)
"""The OpenAI computer actions the `computer` tool can run, each with only the fields it maps."""

_LIST = TypeAdapter(list[Any], config={'strict': True})


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


_SCROLL_NOTES = 'computer_scroll_notes'
"""The `provider_details` key of a note per `computer` tool action, on how a pixel scroll ran in wheel clicks."""

_SCROLL_NOTES_TA = TypeAdapter(list[str | None])

_STORED_ITEM_TA = TypeAdapter(dict[str, Any])

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
    try:
        stored = _STORED_ITEM_TA.validate_python((part.provider_details or {}).get('computer_call'))
    except ValidationError:
        return None
    if stored.get('type') != 'computer_call':
        return None
    return {key: value for key, value in stored.items() if key in _COMPUTER_CALL_KEYS}


def map_computer_call(
    item: responses.ResponseComputerToolCall, tool_name: str, tool_call_id: str, provider_name: str
) -> ComputerCallPart:
    """A `computer_call` as a call of the `computer` tool, keeping the item to send back as it was."""
    # The SDK doesn't validate a response, so the item can hold anything; it is kept as sent, without a warning.
    dumped = item.model_dump(mode='json', exclude_none=True, warnings=False)
    provider_details: dict[str, Any] = {'computer_call': dumped}
    # Read by `ComputerCallPart.safety_checks`, and sent back as acknowledged once the call is approved. A value that
    # isn't a list of checks, even a falsy one like `{}` or an explicit `null`, is kept so the call still waits for
    # approval.
    if 'pending_safety_checks' in item.model_fields_set and (checks := dumped.get('pending_safety_checks')) != []:
        provider_details[_SAFETY_CHECKS] = checks
    args: dict[str, Any] | str
    if item.status != 'completed':
        # Cut off or still being generated: like a truncated function call, its arguments don't parse, so none of it
        # runs. OpenAI says not to run a partially generated action.
        args = '{"actions": ['
    else:
        args = _map_actions(dumped, provider_details)
    return ComputerCallPart(
        tool_name,
        args,
        tool_call_id=tool_call_id,
        id=item.id,
        provider_name=provider_name,
        provider_details=provider_details,
    )


def _map_actions(dumped: dict[str, Any], provider_details: dict[str, Any]) -> dict[str, Any]:
    """A `computer_call` item's actions as the `computer` tool's arguments, noting scrolls in `provider_details`."""
    if 'actions' in dumped:
        raw = dumped['actions']
    elif 'action' in dumped:
        # As in OpenAI's SDK, a single-action call without its safety checklist is refused: its checks may be lost.
        raw = [dumped['action']] if 'pending_safety_checks' in dumped else None
    else:
        raw = []
    try:
        actions = _LIST.validate_python(raw)
    except ValidationError:
        # Not a list, even if it is one action: the call is malformed, so none of it runs.
        actions = [None]
    mapped = [tool_action for action in actions for tool_action in _map_computer_action(action)]
    notes = [note for _, note in mapped]
    if any(notes):
        # Written now, while what the model asked for is known, and read when the result is sent back.
        provider_details[_SCROLL_NOTES] = notes
    # A call with no actions only asks to look.
    return {'actions': [tool_action for tool_action, _ in mapped] or [{'type': 'screenshot'}]}


def computer_calls_by_id(messages: list[ModelMessage]) -> dict[str, ToolCallPart]:
    """Each `computer_call` this API made, by tool call ID, so its output can read it.

    Found by the item it came as, not by valid arguments: a call whose actions didn't map was still a `computer_call`,
    and its output must be a `computer_call_output`.
    """
    return {
        part.tool_call_id: part
        for message in messages
        if isinstance(message, ModelResponse)
        for part in message.parts
        if isinstance(part, ToolCallPart) and 'computer_call' in (part.provider_details or {})
    }


def _scroll_notes(call: ToolCallPart | None, result: ComputerResult | None) -> list[str]:
    """The notes on the pixel scrolls that ran, kept on the call when it was made.

    Only the first `performed` actions ran. Nothing is said for a call changed before it ran (its summary covers it),
    a result without its outcome, or notes that don't validate, since a history may come from storage or a UI.
    """
    if call is None or result is None or result.get('changed'):
        return []
    try:
        notes = _SCROLL_NOTES_TA.validate_python((call.provider_details or {}).get(_SCROLL_NOTES, []))
    except ValidationError:
        return []
    return [note for note in notes[: result['performed']] if note]


def _needs_summary(result: ComputerResult | None) -> bool:
    """Whether a `computer` result needs its text beside the screenshot, which alone would say the call ran.

    A result without its outcome, such as one a UI rebuilt, may describe a failure. A failed or changed call says so
    in its text.
    """
    return result is None or 'error' in result or bool(result.get('changed'))


def map_computer_call_output(
    part: ToolReturnPart, call_id: str, call: ToolCallPart | None
) -> list[responses.ResponseInputItemParam]:
    """A `computer` result as a `computer_call_output`, followed by a message when it needs more than an image.

    The API needs a screenshot in every output, so a result without one sends a blank image instead. The call's
    safety checks are acknowledged only when it ran, which means it was approved.
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
    # Read from the stored details, as the approval gate does: a restored call whose arguments didn't validate is a
    # plain `ToolCallPart`, and may still have run with corrected arguments.
    if computer_return and call is not None and (checks := _safety_checks(call.provider_details)):
        output['acknowledged_safety_checks'] = _acknowledged(checks)
    items: list[responses.ResponseInputItemParam] = [output]
    # A successful result is read as validated content, so content stored as a JSON string reads the same as a list.
    content = (computer_return.typed_content if computer_return else None) or part.content_items()
    # Images before the closing screenshot, such as a zoom, can't go in the output.
    images = [item for item in content if isinstance(item, BinaryImage)]
    others = images[:-1] if image is not None else images
    conversions = _scroll_notes(call, computer_return.result) if computer_return else []
    if image is None or computer_return is None or _needs_summary(computer_return.result) or others or conversions:
        items.append(_summary_message(part, content, others, conversions))
    return items


def _acknowledged(
    checks: list[SafetyCheck],
) -> list[responses.response_input_param.ComputerCallOutputAcknowledgedSafetyCheck]:
    return [
        responses.response_input_param.ComputerCallOutputAcknowledgedSafetyCheck(
            id=check.id or '', code=check.code, message=check.message
        )
        for check in checks
    ]


def _summary_message(
    part: ToolReturnPart, content: Sequence[object], others: list[BinaryImage], conversions: list[str]
) -> responses.EasyInputMessageParam:
    """The output can only carry an image: say what didn't run, or why the image is blank.

    The note is the tool's result, so it is framed as one on the user channel, as for any tool result that can't go
    natively.
    """
    texts = [item for item in content if isinstance(item, str)]
    note = ' '.join([*texts, *conversions]) if texts or conversions else part.model_response_str(wrap_if_error=False)
    framed = frame(TOOL_RESULT_TAG, note, tool_name=part.tool_name, tool_call_id=part.tool_call_id)
    if not others:
        return responses.EasyInputMessageParam(role='user', content=framed)
    message_content: list[responses.ResponseInputContentParam] = [
        responses.ResponseInputTextParam(type='input_text', text=framed),
        *(
            responses.ResponseInputImageParam(type='input_image', image_url=other.data_uri, detail='original')
            for other in others
        ),
    ]
    return responses.EasyInputMessageParam(role='user', content=message_content)
