"""The `origin` an agent run stamps on its model-stream events.

Not VCR tests: `origin` is framework-side metadata that no provider response carries, so a recording can't verify
the stamping, and no provider can be made to fail before its first event on cue for a fallback to move on from.
Durable execution engines' execution attempts are tested in their own suites, like `tests/durable_exec/temporal/`,
and the wire form of `origin` in `tests/test_messages.py`.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

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
    events: list[AgentStreamEvent] = []
    async with agent.run_stream_events('x') as stream:
        async for event in stream:
            if isinstance(event, AgentRunResultEvent):
                return events, event.result.run_id
            events.append(event)
    raise AssertionError('the run ended without a result')  # pragma: no cover


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
