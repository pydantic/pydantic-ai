"""A history dumped to a UI protocol and loaded back maps to the request body it mapped to before, on every model.

The property is adapter-level: it runs each adapter's `dump_messages` and `load_messages` in Python, with the
installed SDK and the default dump options. What a frontend sends back is decided by its client library, not by
these two methods (`@ag-ui/client` leaves `activity` messages out of every request it sends), so `tests/ui_roundtrip`,
which drives the real JavaScript clients, stays the authority on what a client loses: see "Round trips are checked
with the real JavaScript clients" in `pydantic_ai_slim/pydantic_ai/ui/AGENTS.md`.

A round trip loses some shapes, by design or by a known gap. `LOSSES` names, per adapter, each loss that moves a
body, with an `apply` that takes the shape out of a history the way the round trip does, and where the loss is
documented. The property applies an adapter's losses to the drawn history in table order, then compares the body
mapped from that history with the body mapped from its round trip, as JSON text: key order counts, as it does for
the provider's prompt cache. That is the body the mapper builds, not the SDK's own wire bytes, the same caveat as
`tests/test_cache_prefix_stability.py`.

Three documented losses move no body, so they have no entry:

- A response's own provider fields (`provider_name`, `model_name`, `provider_response_id`, `provider_url`,
  `provider_details`), which both `dump_messages` docstrings list: no mapper may read them, as
  https://github.com/pydantic/pydantic-ai/pull/10642 settled for the carried-thinking tag. The one exception, where
  `repair_messages` merges responses that have none, is `responses around change records`.
- `ToolCallPart.args` coming back as a dict, or as compact JSON, which both docstrings note: the mappers send the
  same JSON either way.
- Tool returns coming back in the order of their calls (the `project` docstring in
  `tests/ui_roundtrip/test_ui_roundtrip.py`): the mappers send results in call order.
"""

from __future__ import annotations as _annotations

import asyncio
import json
from collections.abc import Callable
from dataclasses import dataclass, replace
from functools import partial

import pydantic_core
import pytest
from hypothesis import given
from pydantic import JsonValue

from pydantic_ai.messages import (
    CapabilityLoadDeltaPart,
    CapabilityPart,
    FilePart,
    InstructionBaselinePart,
    InstructionDeltaPart,
    ModelMessage,
    ModelRequest,
    ModelRequestPart,
    ModelResponse,
    ModelResponsePart,
    ModelSettingsDeltaPart,
    NativeToolCallPart,
    NativeToolReturnPart,
    SpeechPart,
    SystemPromptPart,
    TextPart,
    ToolAvailabilityDeltaPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.tools import ToolDenied
from pydantic_ai.ui.vercel_ai import VercelAIAdapter

from .conftest import try_import
from .history_mappers import BODY_CASES, Body, MapperCase
from .history_strategies import IGNORE_RANDOM_STATE_NOTE, PROPERTY_SETTINGS, histories

with try_import() as ag_ui_imports_successful:
    from pydantic_ai.ui.ag_ui import AGUIAdapter

pytestmark = IGNORE_RANDOM_STATE_NOTE

History = list[ModelMessage]

_ISSUE_DRAFT = 'to file: UI round trips change the request for eight history shapes'


@dataclass(frozen=True)
class Loss:
    """A shape a UI round trip loses, and where that is documented."""

    apply: Callable[[History], History]
    """Takes the shape out of a history, removing or normalizing it as the round trip does."""
    reason: str


def _edit_parts(
    history: History,
    *,
    request: Callable[[list[ModelRequestPart]], list[ModelRequestPart]] = list,
    response: Callable[[list[ModelResponsePart]], list[ModelResponsePart]] = list,
) -> History:
    """`history` with each message's parts edited.

    A message an edit leaves empty dumps to nothing, so it is left out, and the messages on either side of it load
    back as one, as `load_messages` adds consecutive parts of one kind to one message.
    """
    edited: History = []
    for message in history:
        previous = edited[-1] if edited else None
        if isinstance(message, ModelRequest):
            parts = request(list(message.parts))
            if isinstance(previous, ModelRequest):
                edited[-1] = replace(previous, parts=[*previous.parts, *parts])
            elif parts:
                edited.append(replace(message, parts=parts))
        else:
            parts = response(list(message.parts))
            if isinstance(previous, ModelResponse):
                edited[-1] = replace(previous, parts=[*previous.parts, *parts])
            elif parts:
                edited.append(replace(message, parts=parts))
    return edited


def _drop(*kinds: type) -> Callable[[History], History]:
    """A loss that removes every part of `kinds`."""
    return partial(
        _edit_parts,
        request=lambda parts: [part for part in parts if not isinstance(part, kinds)],
        response=lambda parts: [part for part in parts if not isinstance(part, kinds)],
    )


def _without_instructions(history: History) -> History:
    return [
        replace(message, instructions=None) if isinstance(message, ModelRequest) else message for message in history
    ]


_CHANGE_RECORDS = (ToolAvailabilityDeltaPart, CapabilityLoadDeltaPart, ModelSettingsDeltaPart)


def _synthetic_around_change_records(history: History) -> History:
    """The responses next to a request made only of change records, without their provider fields."""
    around = {
        neighbor
        for index, message in enumerate(history)
        if isinstance(message, ModelRequest) and all(isinstance(part, _CHANGE_RECORDS) for part in message.parts)
        for neighbor in (index - 1, index + 1)
    }
    return [
        replace(
            message,
            provider_name=None,
            model_name=None,
            provider_response_id=None,
            provider_url=None,
            provider_details=None,
        )
        if index in around and isinstance(message, ModelResponse)
        else message
        for index, message in enumerate(history)
    ]


def _without_request_scoped_system_prompts(parts: list[ModelRequestPart]) -> list[ModelRequestPart]:
    return [part for part in parts if not (isinstance(part, SystemPromptPart) and part.scope == 'request')]


def _without_unpaired_native_returns(parts: list[ModelResponsePart]) -> list[ModelResponsePart]:
    """Without each `NativeToolReturnPart` whose `tool_call_id` names no `NativeToolCallPart` of the response."""
    call_ids = {part.tool_call_id for part in parts if isinstance(part, NativeToolCallPart)}
    return [part for part in parts if not isinstance(part, NativeToolReturnPart) or part.tool_call_id in call_ids]


def _lone_texts_as_strings(parts: list[ModelRequestPart]) -> list[ModelRequestPart]:
    return [
        replace(part, content=part.content[0])
        if isinstance(part, UserPromptPart)
        and not isinstance(part.content, str)
        and len(part.content) == 1
        and isinstance(part.content[0], str)
        else part
        for part in parts
    ]


def _merged_text_runs(parts: list[ModelResponsePart]) -> list[ModelResponsePart]:
    """Each run of consecutive `TextPart`s as one, with the first part's other fields."""
    merged: list[ModelResponsePart] = []
    for part in parts:
        previous = merged[-1] if merged else None
        if isinstance(part, TextPart) and isinstance(previous, TextPart):
            merged[-1] = replace(previous, content=previous.content + part.content)
        else:
            merged.append(part)
    return merged


def _joined_system_prompt_runs(parts: list[ModelRequestPart]) -> list[ModelRequestPart]:
    """Each run of consecutive `SystemPromptPart`s as one, their contents joined by newlines."""
    joined: list[ModelRequestPart] = []
    for part in parts:
        previous = joined[-1] if joined else None
        if isinstance(part, SystemPromptPart) and isinstance(previous, SystemPromptPart):
            joined[-1] = replace(previous, content=f'{previous.content}\n{part.content}')
        else:
            joined.append(part)
    return joined


def _error_content_as_text(parts: list[ModelRequestPart]) -> list[ModelRequestPart]:
    """Error outcomes' non-text content as the text it renders to, except a retry's list, which comes back."""
    return [
        replace(part, content=part.model_response_str(wrap_if_error=False))
        if isinstance(part, ToolReturnPart)
        and part.outcome != 'success'
        and not isinstance(part.content, str)
        and not (part.outcome == 'retried' and isinstance(part.content, list))
        else part
        for part in parts
    ]


def _empty_denials_as_the_default(parts: list[ModelRequestPart]) -> list[ModelRequestPart]:
    return [
        replace(part, content=ToolDenied().message)
        if isinstance(part, ToolReturnPart) and part.outcome == 'denied' and part.content == ''
        else part
        for part in parts
    ]


def _json_text_returns_parsed(parts: list[ModelRequestPart]) -> list[ModelRequestPart]:
    """Each tool return whose text is a JSON object or array, with that value as its content."""
    edited: list[ModelRequestPart] = []
    for part in parts:
        if isinstance(part, ToolReturnPart) and isinstance(part.content, str):
            try:
                parsed: JsonValue = pydantic_core.from_json(part.content)
            except ValueError:
                parsed = None
            if isinstance(parsed, dict | list):
                part = replace(part, content=parsed)
        edited.append(part)
    return edited


def _without_part_ids_and_details(parts: list[ModelResponsePart]) -> list[ModelResponsePart]:
    edited: list[ModelResponsePart] = []
    for part in parts:
        if isinstance(part, TextPart | ToolCallPart):
            edited.append(replace(part, id=None, provider_name=None, provider_details=None))
        elif isinstance(part, NativeToolCallPart):
            edited.append(replace(part, id=None, provider_details=None))
        elif isinstance(part, NativeToolReturnPart):
            edited.append(replace(part, provider_details=None))
        else:
            edited.append(part)
    return edited


def _native_returns_after_the_calls(parts: list[ModelResponsePart]) -> list[ModelResponsePart]:
    """Each `NativeToolReturnPart` after the run of calls its call is in, as one AG-UI assistant message and its
    tool messages load back."""
    call_ids = {part.tool_call_id for part in parts if isinstance(part, NativeToolCallPart)}
    returns = {part.tool_call_id: part for part in parts if isinstance(part, NativeToolReturnPart)}
    edited: list[ModelResponsePart] = []
    pending: list[ModelResponsePart] = []
    for part in parts:
        if isinstance(part, NativeToolReturnPart) and part.tool_call_id in call_ids:
            continue
        if isinstance(part, NativeToolCallPart | ToolCallPart):
            edited.append(part)
            if isinstance(part, NativeToolCallPart) and part.tool_call_id in returns:
                pending.append(returns[part.tool_call_id])
        else:
            edited.extend(pending)
            pending.clear()
            edited.append(part)
    return [*edited, *pending]


_INSTRUCTIONS = Loss(
    _without_instructions,
    '`docs/ui/vercel-ai.md` and `docs/ui/ag-ui.md`, "System prompts and instructions": instructions are injected '
    'fresh on every request, never persisted; `docs/ui/overview.md`, "Trust model for client-submitted messages", '
    'resets `ModelRequest.instructions` on the way back in',
)
_REQUEST_SCOPED_SYSTEM_PROMPTS = Loss(
    partial(_edit_parts, request=_without_request_scoped_system_prompts),
    '`docs/ui/overview.md`, "Trust model for client-submitted messages": request-scoped system prompts are left out '
    'of `dump_messages` altogether; `agent_docs/message-history-and-caching.md`, "UI round trips are faithful"',
)
_INSTRUCTION_RECORDS = Loss(
    _drop(InstructionDeltaPart, InstructionBaselinePart),
    '`docs/ui/overview.md`, "Trust model for client-submitted messages": UI message dumps omit the operator records '
    'of append-on-change instruction history',
)
_CAPABILITY_PARTS = Loss(
    _drop(CapabilityPart),
    '`sanitize_messages` docstring: the UI adapters never send a `CapabilityPart` to a client, which stays in '
    "server-side history until it has a carrier whose trust a client can't change "
    '(https://github.com/pydantic/pydantic-ai/issues/11029)',
)
_SPEECH_PARTS = Loss(
    _drop(SpeechPart),
    f'undocumented ({_ISSUE_DRAFT}): both adapters skip `SpeechPart` in requests and responses ("Realtime '
    'audio parts are not rendered", the `SpeechPart()` cases), while `Model.prepare_messages` sends it as a prompt or '
    'text',
)
_RESPONSES_AROUND_CHANGE_RECORDS = Loss(
    _synthetic_around_change_records,
    f"undocumented ({_ISSUE_DRAFT}): both `dump_messages` docstrings say a response's own provider fields "
    "aren't carried, which leaves every reloaded response synthetic, and `repair_messages` merges consecutive synthetic "
    'responses: two responses around a request that `Model.prepare_messages` empties, one made only of change '
    'records, are sent as one after the round trip and as two before it',
)
_UNPAIRED_NATIVE_RETURNS = Loss(
    partial(_edit_parts, response=_without_unpaired_native_returns),
    f'undocumented ({_ISSUE_DRAFT}): both adapters dump a native tool return with the call of the same '
    "`tool_call_id`, so one with its own id, as older OpenAI tool search histories store it (`tso_` for the call's "
    '`tsc_`), is dropped',
)
_LONE_TEXT_PROMPTS = Loss(
    partial(_edit_parts, request=_lone_texts_as_strings),
    f'undocumented ({_ISSUE_DRAFT}): `UserPromptPart([text])`, as `agent.run([text])` records it, reloads '
    'as `UserPromptPart(text)`, which several mappers send as a plain string instead of a list of parts',
)

# Each adapter's removals come first, so the losses that merge parts see what a removal leaves adjacent: the property
# round-trips the cut history, whose own dump no longer has the removed parts between them.
LOSSES: dict[str, dict[str, Loss]] = {
    'vercel-ai': {
        'instructions': _INSTRUCTIONS,
        'request-scoped system prompts': _REQUEST_SCOPED_SYSTEM_PROMPTS,
        'instruction records': _INSTRUCTION_RECORDS,
        'capability parts': _CAPABILITY_PARTS,
        'speech parts': _SPEECH_PARTS,
        'unpaired native returns': _UNPAIRED_NATIVE_RETURNS,
        'responses around change records': _RESPONSES_AROUND_CHANGE_RECORDS,
        'lone text prompts': _LONE_TEXT_PROMPTS,
        'consecutive text parts': Loss(
            partial(_edit_parts, response=_merged_text_runs),
            '`VercelAIAdapter.dump_messages` docstring: consecutive `TextPart`s come back as one, with the first '
            "part's `id` and `provider_details`; that the joined text has no separator is undocumented "
            f'({_ISSUE_DRAFT})',
        ),
        'error content': Loss(
            partial(_edit_parts, request=_error_content_as_text),
            '`VercelAIAdapter.dump_messages` docstring: `errorText` can only hold the rendering of an error outcome, '
            "and only a retry's list rides along beside it (`_retried_structured_content` docstring)",
        ),
        # After `error content`, which can leave a denial's content empty.
        'empty denial reasons': Loss(
            partial(_edit_parts, request=_empty_denials_as_the_default),
            f'undocumented ({_ISSUE_DRAFT}): a denied return with empty content reloads with '
            "`ToolDenied`'s default message (`_denial_reason` docstring), which `VercelAIAdapter.dump_messages` "
            "doesn't list",
        ),
    },
    'ag-ui': {
        'instructions': _INSTRUCTIONS,
        'request-scoped system prompts': _REQUEST_SCOPED_SYSTEM_PROMPTS,
        'instruction records': _INSTRUCTION_RECORDS,
        'capability parts': _CAPABILITY_PARTS,
        'speech parts': _SPEECH_PARTS,
        'file parts': Loss(
            _drop(FilePart),
            '`AGUIAdapter.dump_messages` docstring: `FilePart` is dropped unless `preserve_file_data=True`',
        ),
        'unpaired native returns': _UNPAIRED_NATIVE_RETURNS,
        'responses around change records': _RESPONSES_AROUND_CHANGE_RECORDS,
        'system prompt runs': Loss(
            partial(_edit_parts, request=_joined_system_prompt_runs),
            '`AGUIAdapter._dump_request_parts` docstring: a run of consecutive system prompts becomes one '
            '`SystemMessage`',
        ),
        'lone text prompts': _LONE_TEXT_PROMPTS,
        'consecutive text parts': Loss(
            partial(_edit_parts, response=_merged_text_runs),
            f'undocumented ({_ISSUE_DRAFT}): `_dump_text_part` concatenates adjacent `TextPart`s into one '
            "message's content, and `AGUIAdapter.dump_messages` doesn't list it",
        ),
        'json text tool returns': Loss(
            partial(_edit_parts, request=_json_text_returns_parsed),
            '`tests/test_ag_ui.py::test_tool_return_json_scalar_string_stays_string` docstring: a string return '
            'shaped like a JSON object or array is wire-indistinguishable from a structured return, so it reloads as '
            'one',
        ),
        'part ids and provider details': Loss(
            partial(_edit_parts, response=_without_part_ids_and_details),
            '`AGUIAdapter.dump_messages` docstring: `TextPart` and `ToolCallPart` lose `id`, `provider_name` and '
            '`provider_details`, `NativeToolCallPart` its `id` and `provider_details`, `NativeToolReturnPart` its '
            '`provider_details`',
        ),
        'native returns after the calls': Loss(
            partial(_edit_parts, response=_native_returns_after_the_calls),
            f'undocumented ({_ISSUE_DRAFT}): a native tool return rides as a tool message after its '
            "call's assistant message, so it reloads after every call in that message, not right after its own; the "
            '`AGUIAdapter.dump_messages` docstring says it is emitted directly after its call',
        ),
    },
}
"""Each adapter's losses that move a body, applied in this order."""


@dataclass(frozen=True)
class RoundTrip:
    id: str
    run: Callable[[History], History]
    marks: tuple[pytest.MarkDecorator, ...] = ()


def _vercel_ai(history: History) -> History:
    return VercelAIAdapter.load_messages(VercelAIAdapter.dump_messages(history))


def _ag_ui(history: History) -> History:
    return AGUIAdapter.load_messages(AGUIAdapter.dump_messages(history))


ROUND_TRIPS = [
    RoundTrip('vercel-ai', _vercel_ai),
    RoundTrip(
        'ag-ui',
        _ag_ui,
        (pytest.mark.skipif(not ag_ui_imports_successful(), reason='ag-ui-protocol not installed'),),
    ),
]


def _text(body: Body) -> str:
    return json.dumps({'system': body.system, 'messages': body.messages})


# Sync, like the body properties in `tests/test_history_properties.py`: Hypothesis ends a draw that runs out of data
# with a `BaseException`, which raised inside an async test breaks the anyio test runner for every later example.
@pytest.mark.parametrize('case', [pytest.param(case, id=case.id, marks=case.marks) for case in BODY_CASES])
@pytest.mark.parametrize(
    'round_trip', [pytest.param(round_trip, id=round_trip.id, marks=round_trip.marks) for round_trip in ROUND_TRIPS]
)
@PROPERTY_SETTINGS
@given(history=histories())
def test_round_trip_keeps_the_body(round_trip: RoundTrip, case: MapperCase, history: History):
    """Less the adapter's `LOSSES`, a history maps to the same body after `dump_messages` and `load_messages`."""
    cut = history
    for loss in LOSSES[round_trip.id].values():
        cut = loss.apply(cut)
    reloaded = round_trip.run(cut)
    with asyncio.Runner() as runner:
        expected = _text(runner.run(case.body(cut)))
        actual = _text(runner.run(case.body(reloaded)))
    assert actual == expected
