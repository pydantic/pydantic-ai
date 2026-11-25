"""The `ModelResponseStartEvent` and `ModelResponseEndEvent` an agent run opens and closes each streamed response with.

Not VCR tests: the boundaries are framework-side events that no provider response carries, and no provider can be
made to fail before or after its first event, or be cancelled mid-stream, on cue. Durable execution engines'
re-executions are tested in their own suites, like `tests/durable_exec/temporal/test_workflow_streams.py`, a
`wrap_model_request` hook that replaces a response mid-stream in `tests/test_capability_hooks.py`, and the UI
adapters' handling in `tests/test_vercel_ai.py` and `tests/test_ag_ui.py`.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from unittest.mock import ANY

import pytest
from dirty_equals import IsDatetime, IsStr
from inline_snapshot import snapshot

from pydantic_ai import (
    Agent,
    AgentStreamEvent,
    ModelMessage,
    ModelResponse,
    ModelResponseEndEvent,
    ModelResponseOrigin,
    ModelResponseStartEvent,
    PartDeltaEvent,
    PartEndEvent,
    PartStartEvent,
    RequestUsage,
    TextPart,
)
from pydantic_ai.capabilities import ModelFallback
from pydantic_ai.exceptions import ModelAPIError
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.run import AgentRunResultEvent


def _boundaries(events: list[AgentStreamEvent]) -> list[ModelResponseStartEvent | ModelResponseEndEvent]:
    return [event for event in events if isinstance(event, ModelResponseStartEvent | ModelResponseEndEvent)]


async def test_each_response_is_opened_and_closed_once():
    """A run that streams two responses opens and closes each once, and each end carries the response history keeps."""
    agent = Agent(TestModel())

    @agent.tool_plain
    def get_answer() -> str:
        return '42'

    events: list[AgentStreamEvent] = []
    async with agent.run_stream_events('hello') as stream:
        async for event in stream:
            if isinstance(event, AgentRunResultEvent):
                result = event.result
            else:
                events.append(event)

    assert [type(event).__name__ for event in events] == snapshot(
        [
            'ModelResponseStartEvent',
            'PartStartEvent',
            'PartEndEvent',
            'ModelResponseEndEvent',
            'FunctionToolCallEvent',
            'FunctionToolResultEvent',
            'ModelResponseStartEvent',
            'PartStartEvent',
            'FinalResultEvent',
            'PartDeltaEvent',
            'PartDeltaEvent',
            'PartEndEvent',
            'ModelResponseEndEvent',
        ]
    )
    first = ModelResponseOrigin(run_id=result.run_id, run_step=1)
    second = ModelResponseOrigin(run_id=result.run_id, run_step=2)
    boundaries = _boundaries(events)
    assert [event.origin for event in boundaries] == [first, first, second, second]
    assert boundaries[0] == ModelResponseStartEvent(origin=first, model_name='test', provider_name='test')
    # A response's events all carry the origin its boundaries do.
    assert {event.origin for event in events if isinstance(event, PartStartEvent | PartDeltaEvent | PartEndEvent)} == {
        first,
        second,
    }

    # Each end carries the response the run records, as streamed.
    ends = [event for event in boundaries if isinstance(event, ModelResponseEndEvent)]
    assert [event.status for event in ends] == ['complete', 'complete']
    responses = [message for message in result.all_messages() if isinstance(message, ModelResponse)]
    assert [event.response for event in ends] == responses


async def test_a_fallback_response_replaces_the_attempt_it_moved_on_from():
    """`ModelFallback` moving on before the first event streams the next attempt, which names the one it replaces."""

    async def failure_stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
        raise ModelAPIError(model_name='failing', message='boom')
        yield ''  # pragma: no cover

    async def success_stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
        yield 'hello'

    agent = Agent(
        FunctionModel(stream_function=failure_stream),
        capabilities=[ModelFallback(FunctionModel(stream_function=success_stream))],
    )
    events: list[AgentStreamEvent] = []
    async with agent.run_stream_events('x') as stream:
        async for event in stream:
            events.append(event)
    result_event = events.pop()
    assert isinstance(result_event, AgentRunResultEvent)
    run_id = result_event.result.run_id

    # The attempt that failed before its first event streamed nothing, so only the one that replaced it is opened.
    start, end = _boundaries(events)
    assert start == ModelResponseStartEvent(
        origin=ModelResponseOrigin(run_id=run_id, run_step=1, attempt=2),
        model_name='function::success_stream',
        provider_name=None,
        replaces=f'{run_id}:1:1',
    )
    assert start.origin == ModelResponseOrigin(run_id=run_id, run_step=1, attempt=2)
    assert start.replaces == ModelResponseOrigin(run_id=run_id, run_step=1).response_id
    assert end.origin == start.origin
    assert end.status == 'complete'


async def test_a_cancelled_response_ends_interrupted_with_what_it_streamed():
    """`AgentStream.cancel()` ends the response as `'interrupted'`, carrying the partial response."""
    agent = Agent(TestModel(custom_output_text='Hello there, world'))
    events: list[AgentStreamEvent] = []
    async with agent.iter('x') as run:
        node = await run.next(run.next_node)
        assert Agent.is_model_request_node(node)
        async with node.stream(run.ctx) as stream:
            async for event in stream:
                events.append(event)
                if isinstance(event, PartStartEvent):
                    await stream.cancel()

    start, end = _boundaries(events)
    assert end.origin == start.origin
    assert end == snapshot(
        ModelResponseEndEvent(
            origin=ANY,
            status='interrupted',
            response=ModelResponse(
                parts=[TextPart(content='Hello there, world')],
                usage=RequestUsage(input_tokens=51, output_tokens=3),
                model_name='test',
                model_id='test:test',
                timestamp=IsDatetime(),
                provider_name='test',
                run_id=IsStr(),
                conversation_id=IsStr(),
                state='interrupted',
            ),
        )
    )


async def test_a_stream_that_fails_after_output_ends_failed():
    """A model stream that raises after streaming ends the response as `'failed'`, before the error reaches the run."""

    async def stream_function(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
        yield 'Hello'
        raise RuntimeError('connection lost')

    agent = Agent(FunctionModel(stream_function=stream_function))
    events: list[AgentStreamEvent] = []
    with pytest.raises(RuntimeError, match='connection lost'):
        async with agent.run_stream_events('x') as stream:
            async for event in stream:
                events.append(event)

    start, end = _boundaries(events)
    assert end.origin == start.origin
    assert end == snapshot(
        ModelResponseEndEvent(
            origin=ANY,
            status='failed',
            response=ModelResponse(
                parts=[TextPart(content='Hello')],
                usage=RequestUsage(input_tokens=50, output_tokens=1),
                model_name='function::stream_function',
                model_id='function:function::stream_function',
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
                state='interrupted',
            ),
            error='RuntimeError: connection lost',
        )
    )
