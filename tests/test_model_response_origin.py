"""The `origin` an agent run stamps on its model-stream events, and its wire form.

Durable execution engines' execution attempts are tested in their own suites, like `tests/durable_exec/temporal/`.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import replace

import pydantic
import pytest
from inline_snapshot import snapshot

from pydantic_ai import (
    Agent,
    AgentStreamEvent,
    FinalResultEvent,
    ModelResponseOrigin,
    PartDeltaEvent,
    PartEndEvent,
    PartStartEvent,
    TextPart,
    TextPartDelta,
)
from pydantic_ai.capabilities import ModelFallback
from pydantic_ai.exceptions import ModelAPIError
from pydantic_ai.messages import ModelMessage
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.run import AgentRunResultEvent


async def failure_stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
    raise ModelAPIError(model_name='failing', message='boom')
    yield ''  # pragma: no cover


async def success_stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
    yield 'hello'


async def _stream_events(agent: Agent[None, str]) -> tuple[list[AgentStreamEvent], str]:
    """The run's model-stream events, and its `run_id`."""
    async with agent.run_stream_events('x') as stream:
        events = [event async for event in stream]
    result_event = events.pop()
    assert isinstance(result_event, AgentRunResultEvent)
    return events, result_event.result.run_id


def _origin(event: AgentStreamEvent) -> ModelResponseOrigin | None:
    assert isinstance(event, PartStartEvent | PartDeltaEvent | PartEndEvent | FinalResultEvent)
    return event.origin


async def test_model_fallback_streams_the_next_model_as_the_next_attempt():
    """`ModelFallback` moving on before the first event streams the model it moved to as attempt 2."""
    agent = Agent(
        FunctionModel(stream_function=failure_stream),
        capabilities=[ModelFallback(FunctionModel(stream_function=success_stream))],
    )
    events, run_id = await _stream_events(agent)

    assert events == snapshot(
        [
            PartStartEvent(index=0, part=TextPart(content='hello')),
            FinalResultEvent(tool_name=None, tool_call_id=None),
            PartEndEvent(index=0, part=TextPart(content='hello')),
        ]
    )
    origin = ModelResponseOrigin(run_id=run_id, run_step=1, attempt=2)
    assert [_origin(event) for event in events] == [origin, origin, origin]
    assert origin.response_id == f'{run_id}:1:2'


async def test_fallback_model_candidates_are_one_attempt():
    """A `FallbackModel` moves on inside one model request, before its first event, so its candidates share one attempt."""
    agent = Agent(
        FallbackModel(FunctionModel(stream_function=failure_stream), FunctionModel(stream_function=success_stream))
    )
    events, run_id = await _stream_events(agent)

    assert {_origin(event) for event in events} == {ModelResponseOrigin(run_id=run_id, run_step=1)}


def test_supersedes():
    """A later attempt or execution of the same request step supersedes an earlier one, and nothing else does."""
    first = ModelResponseOrigin(run_id='run', run_step=1)
    re_executed = ModelResponseOrigin(run_id='run', run_step=1, execution_attempt=2)
    next_attempt = ModelResponseOrigin(run_id='run', run_step=1, attempt=2)

    assert re_executed.supersedes(first)
    assert re_executed.response_id == first.response_id == 'run:1:1'
    assert next_attempt.supersedes(re_executed)
    assert next_attempt.response_id == 'run:1:2'
    assert not first.supersedes(re_executed)
    assert not first.supersedes(first)
    assert not ModelResponseOrigin(run_id='run', run_step=2).supersedes(first)
    assert not ModelResponseOrigin(run_id='other', run_step=1, attempt=2).supersedes(first)


_EVENTS = pydantic.TypeAdapter(AgentStreamEvent)


@pytest.mark.parametrize(
    'event',
    [
        PartStartEvent(index=0, part=TextPart(content='Hi')),
        PartDeltaEvent(index=0, delta=TextPartDelta(content_delta=' there')),
        PartEndEvent(index=0, part=TextPart(content='Hi there')),
        FinalResultEvent(tool_name=None, tool_call_id=None),
    ],
    ids=lambda event: event.event_kind,
)
def test_origin_round_trips(event: PartStartEvent | PartDeltaEvent | PartEndEvent | FinalResultEvent):
    """An event with an `origin` round-trips through `AgentStreamEvent`, and one without serializes without the key."""
    assert 'origin' not in _EVENTS.dump_python(event, mode='json')

    event = replace(event, origin=ModelResponseOrigin(run_id='run', run_step=3, attempt=2, execution_attempt=4))
    serialized = _EVENTS.dump_python(event, mode='json')
    assert serialized['origin'] == {'run_id': 'run', 'run_step': 3, 'attempt': 2, 'execution_attempt': 4}
    assert _origin(_EVENTS.validate_python(serialized)) == event.origin


def test_origin_is_left_out_of_equality_and_repr():
    """Events compare and print as they did before `origin` existed, so identical streams of two runs compare equal."""
    event = PartStartEvent(index=0, part=TextPart(content='Hi'))
    stamped = replace(event, origin=ModelResponseOrigin(run_id='run', run_step=1))
    assert stamped == event
    assert repr(stamped) == repr(event) == "PartStartEvent(index=0, part=TextPart(content='Hi'))"


def test_events_without_origin_serialize_as_before():
    """An event without an `origin` serializes byte for byte as it did before the field existed."""
    assert _EVENTS.dump_json(PartStartEvent(index=0, part=TextPart(content='Hi'))) == snapshot(
        b'{"index":0,"part":{"content":"Hi","id":null,"provider_name":null,"provider_details":null,"citations":null,"part_kind":"text"},"previous_part_kind":null,"canonical_system":null,"event_kind":"part_start"}'
    )
    assert _EVENTS.dump_json(FinalResultEvent(tool_name=None, tool_call_id=None)) == snapshot(
        b'{"tool_name":null,"tool_call_id":null,"event_kind":"final_result"}'
    )
