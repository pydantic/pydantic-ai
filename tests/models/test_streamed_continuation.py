"""End-to-end tests for streamed model continuations driven through the full agent graph.

These are unit tests rather than VCR tests for two reasons: `FunctionModel` cannot emit a
*suspended streaming* segment (the input a real continuation needs), and a cassette wouldn't
reliably protect this behavior anyway — our VCR matchers aren't sensitive to the reindex payload,
so a regression in the accumulate-vs-replace offset boundary could still match an existing recording
and pass green. So these tests use a small scripted streaming model (`_ScriptedModel`) whose
`request`/`request_stream` return a `suspended → complete` chain. Driving
`agent.run`/`agent.run_stream`/`agent.iter` against it exercises the continuation loop inside
`ModelRequestNode`: the streamed composite stitches every segment into one `AgentStream`,
model-request hooks fire once around the whole chain, usage is summed once, and cancellation kills
the server-side job without requesting further segments.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from typing import Any, Literal

import pytest

from pydantic_ai import Agent
from pydantic_ai._run_context import RunContext
from pydantic_ai.capabilities import Hooks
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ModelResponseStreamEvent,
    PartDeltaEvent,
    PartStartEvent,
    TextPart,
    UserPromptPart,
)
from pydantic_ai.models import Model, ModelRequestContext, ModelRequestParameters, StreamedResponse
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.wrapper import WrapperModel
from pydantic_ai.settings import ModelSettings
from pydantic_ai.usage import RequestUsage
from pydantic_graph import End

from .._inline_snapshot import snapshot

pytestmark = pytest.mark.anyio

_TIMESTAMP = datetime(2024, 1, 1, tzinfo=timezone.utc)


@dataclass
class _StreamSegment:
    """One scripted `request_stream` call: the text parts it streams and the segment's metadata."""

    texts: list[str]
    state: Literal['suspended', 'complete']
    provider_response_id: str
    input_tokens: int
    output_tokens: int
    suspended_retry_delay: float | None = None


@dataclass
class _ScriptedStreamedResponse(StreamedResponse):
    """Streams a segment's text parts through the parts manager, exposing scripted metadata via `get()`."""

    _segment: _StreamSegment = field(default=None)  # type: ignore[assignment]

    def __post_init__(self) -> None:
        segment = self._segment
        # Populate metadata up front so the composite can resolve replace-vs-accumulate on the first
        # reindexable event, mirroring a real provider stream that knows its id from the start.
        self.provider_response_id = segment.provider_response_id
        self.state = segment.state
        self.suspended_retry_delay = segment.suspended_retry_delay
        self._usage = RequestUsage(input_tokens=segment.input_tokens, output_tokens=segment.output_tokens)

    async def _get_event_iterator(self) -> AsyncIterator[ModelResponseStreamEvent]:
        for index, text in enumerate(self._segment.texts):
            # An empty delta opens the part (`PartStartEvent`); the content follows (`PartDeltaEvent`).
            for event in self._parts_manager.handle_text_delta(vendor_part_id=index, content=''):
                yield event
            for event in self._parts_manager.handle_text_delta(vendor_part_id=index, content=text):
                yield event

    async def close_stream(self) -> None:
        pass

    @property
    def model_name(self) -> str:
        return 'scripted'

    @property
    def provider_name(self) -> str:
        return 'scripted'

    @property
    def provider_url(self) -> str:
        return 'scripted'

    @property
    def timestamp(self) -> datetime:
        return _TIMESTAMP


class _ScriptedModel(Model):
    """A `Model` whose `request`/`request_stream` replay scripted `suspended → complete` chains."""

    def __init__(
        self,
        *,
        responses: list[ModelResponse] | None = None,
        segments: list[_StreamSegment] | None = None,
    ) -> None:
        super().__init__()
        self._responses = responses or []
        self._segments = segments or []
        self.request_stream_calls = 0
        self.cancelled: list[ModelResponse] = []

    @property
    def model_name(self) -> str:
        return 'scripted'

    @property
    def system(self) -> str:
        return 'scripted'

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        return self._responses.pop(0)

    @asynccontextmanager
    async def request_stream(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
        run_context: RunContext[Any] | None = None,
    ) -> AsyncGenerator[StreamedResponse]:
        self.request_stream_calls += 1
        segment = self._segments.pop(0)
        yield _ScriptedStreamedResponse(model_request_parameters, _segment=segment)

    async def cancel_suspended_response(self, response: ModelResponse) -> None:
        self.cancelled.append(response)


def _suspended(*, texts: list[str], provider_response_id: str, input_tokens: int, output_tokens: int) -> ModelResponse:
    return ModelResponse(
        parts=[TextPart(content=t) for t in texts],
        model_name='scripted',
        provider_name='scripted',
        provider_response_id=provider_response_id,
        usage=RequestUsage(input_tokens=input_tokens, output_tokens=output_tokens),
        state='suspended',
        timestamp=_TIMESTAMP,
    )


async def _collect_stream_events(
    agent: Agent[None, str], prompt: str, **iter_kwargs: Any
) -> list[ModelResponseStreamEvent]:
    """Drive `agent.iter`, streaming each `ModelRequestNode`, and collect the raw stream events."""
    events: list[ModelResponseStreamEvent] = []
    async with agent.iter(prompt, **iter_kwargs) as run:
        node = run.next_node
        while not isinstance(node, End):
            if Agent.is_model_request_node(node):
                async with node.stream(run.ctx) as stream:
                    async for event in stream:
                        events.append(event)
            node = await run.next(node)
    return events


def _part_indices(events: list[ModelResponseStreamEvent]) -> list[tuple[str, int]]:
    return [
        (type(event).__name__, event.index) for event in events if isinstance(event, (PartStartEvent, PartDeltaEvent))
    ]


async def test_streamed_accumulate_offsets_part_indices() -> None:
    """Anthropic-shape accumulate: the second segment's parts get offset indices with no collision."""
    model = _ScriptedModel(
        segments=[
            _StreamSegment(
                texts=['A', 'B'], state='suspended', provider_response_id='r1', input_tokens=5, output_tokens=2
            ),
            _StreamSegment(texts=['C'], state='complete', provider_response_id='r2', input_tokens=3, output_tokens=4),
        ]
    )
    assert model.model_name == 'scripted'
    assert model.system == 'scripted'
    agent = Agent(model)

    events = await _collect_stream_events(agent, 'go')

    # First segment keeps indices [0, 1]; the second segment is offset past them to [2].
    assert _part_indices(events) == snapshot(
        [
            ('PartStartEvent', 0),
            ('PartDeltaEvent', 0),
            ('PartStartEvent', 1),
            ('PartDeltaEvent', 1),
            ('PartStartEvent', 2),
            ('PartDeltaEvent', 2),
        ]
    )
    assert model.request_stream_calls == 2


async def test_streamed_replace_reuses_part_indices() -> None:
    """OpenAI-shape replace (same provider_response_id): the second segment reuses the index space."""
    model = _ScriptedModel(
        segments=[
            _StreamSegment(texts=['X'], state='suspended', provider_response_id='r1', input_tokens=5, output_tokens=2),
            _StreamSegment(
                texts=['X', 'Y'], state='complete', provider_response_id='r1', input_tokens=8, output_tokens=6
            ),
        ]
    )
    agent = Agent(model)

    events = await _collect_stream_events(agent, 'go')

    # Offset stays 0: the second segment reuses indices [0, 1] rather than appending.
    assert _part_indices(events) == snapshot(
        [
            ('PartStartEvent', 0),
            ('PartDeltaEvent', 0),
            ('PartStartEvent', 0),
            ('PartDeltaEvent', 0),
            ('PartStartEvent', 1),
            ('PartDeltaEvent', 1),
        ]
    )


async def test_streamed_continuation_merges_into_one_message() -> None:
    """The stitched stream produces a single final message with all accumulated parts in order."""
    model = _ScriptedModel(
        segments=[
            _StreamSegment(
                texts=['first '], state='suspended', provider_response_id='r1', input_tokens=5, output_tokens=2
            ),
            _StreamSegment(
                texts=['second'], state='complete', provider_response_id='r2', input_tokens=3, output_tokens=4
            ),
        ]
    )
    agent = Agent(model)

    async with agent.run_stream('go') as result:
        output = await result.get_output()

    assert output == snapshot('first second')
    messages = result.all_messages()
    assert len(messages) == snapshot(2)  # one request, one merged response
    merged = messages[-1]
    assert isinstance(merged, ModelResponse)
    assert [p.content for p in merged.parts if isinstance(p, TextPart)] == snapshot(['first ', 'second'])
    assert merged.state == 'complete'


@pytest.mark.parametrize('stream', [False, True])
async def test_hooks_fire_once_around_continuation_chain(stream: bool) -> None:
    """`before`/`after`/`wrap` model-request hooks fire exactly once across a suspended → complete chain.

    `after_model_request` must receive the final merged response, not an intermediate suspended one.
    """
    hooks = Hooks()
    calls: list[str] = []
    after_responses: list[ModelResponse] = []

    @hooks.on.before_model_request
    async def _before(ctx: RunContext[Any], request_context: ModelRequestContext) -> ModelRequestContext:
        calls.append('before')
        return request_context

    @hooks.on.model_request
    async def _wrap(ctx: RunContext[Any], *, request_context: ModelRequestContext, handler: Any) -> ModelResponse:
        calls.append('wrap')
        return await handler(request_context)

    @hooks.on.after_model_request
    async def _after(
        ctx: RunContext[Any], *, request_context: ModelRequestContext, response: ModelResponse
    ) -> ModelResponse:
        calls.append('after')
        after_responses.append(response)
        return response

    def _make_model() -> _ScriptedModel:
        if stream:
            return _ScriptedModel(
                segments=[
                    _StreamSegment(
                        texts=['a'], state='suspended', provider_response_id='r1', input_tokens=5, output_tokens=2
                    ),
                    _StreamSegment(
                        texts=['b'], state='complete', provider_response_id='r2', input_tokens=3, output_tokens=4
                    ),
                ]
            )
        return _ScriptedModel(
            responses=[
                _suspended(texts=['a'], provider_response_id='r1', input_tokens=5, output_tokens=2),
                ModelResponse(
                    parts=[TextPart(content='b')],
                    model_name='scripted',
                    provider_response_id='r2',
                    usage=RequestUsage(input_tokens=3, output_tokens=4),
                ),
            ]
        )

    agent = Agent(_make_model(), capabilities=[hooks])

    if stream:
        async with agent.run_stream('go') as result:
            await result.get_output()
    else:
        await agent.run('go')

    assert calls == snapshot(['before', 'wrap', 'after'])
    # The single `after_model_request` call sees the final merged (complete) response.
    assert len(after_responses) == 1
    assert after_responses[0].state == 'complete'
    assert [p.content for p in after_responses[0].parts if isinstance(p, TextPart)] == snapshot(['a', 'b'])


@pytest.mark.parametrize('stream', [False, True])
async def test_usage_summed_once_across_segments(stream: bool) -> None:
    """A multi-segment run sums usage exactly once, with `requests == 1` (continuations aren't steps)."""
    if stream:
        model: _ScriptedModel = _ScriptedModel(
            segments=[
                _StreamSegment(
                    texts=['a'], state='suspended', provider_response_id='r1', input_tokens=10, output_tokens=3
                ),
                _StreamSegment(
                    texts=['b'], state='suspended', provider_response_id='r2', input_tokens=8, output_tokens=4
                ),
                _StreamSegment(
                    texts=['c'], state='complete', provider_response_id='r3', input_tokens=6, output_tokens=5
                ),
            ]
        )
    else:
        model = _ScriptedModel(
            responses=[
                _suspended(texts=['a'], provider_response_id='r1', input_tokens=10, output_tokens=3),
                _suspended(texts=['b'], provider_response_id='r2', input_tokens=8, output_tokens=4),
                ModelResponse(
                    parts=[TextPart(content='c')],
                    model_name='scripted',
                    provider_response_id='r3',
                    usage=RequestUsage(input_tokens=6, output_tokens=5),
                ),
            ]
        )
    agent = Agent(model)

    if stream:
        async with agent.run_stream('go') as result:
            await result.get_output()
        usage = result.usage
    else:
        run_result = await agent.run('go')
        usage = run_result.usage

    assert usage.requests == snapshot(1)
    assert usage.input_tokens == snapshot(24)
    assert usage.output_tokens == snapshot(12)


async def test_cancel_mid_continuation_cancels_job_and_stops() -> None:
    """Cancelling between segments cancels the server-side job once and requests no further segment."""
    model = _ScriptedModel(
        segments=[
            _StreamSegment(
                texts=['a', 'b'],
                state='suspended',
                provider_response_id='r1',
                input_tokens=5,
                output_tokens=2,
                suspended_retry_delay=0.0,
            ),
            _StreamSegment(texts=['c'], state='complete', provider_response_id='r2', input_tokens=3, output_tokens=4),
        ]
    )
    agent = Agent(model)

    async with agent.iter('go') as run:
        node = run.next_node
        while not isinstance(node, End):  # pragma: no branch
            if Agent.is_model_request_node(node):
                async with node.stream(run.ctx) as stream:
                    iterator = stream.__aiter__()
                    await iterator.__anext__()  # start the first (suspended) segment
                    await stream.cancel()
                    async for _ in iterator:  # draining must not request the second segment
                        pass
                    assert stream.response.state == 'interrupted'
                break
            node = await run.next(node)

    assert len(model.cancelled) == 1
    assert model.cancelled[0].provider_response_id == 'r1'
    assert model.request_stream_calls == 1  # the second segment was never requested


@pytest.mark.parametrize('stream', [False, True])
async def test_resume_from_trailing_suspended_history(stream: bool) -> None:
    """A history ending in a suspended response resumes via continuation; hooks fire once."""
    hooks = Hooks()
    calls: list[str] = []

    @hooks.on.before_model_request
    async def _before(ctx: RunContext[Any], request_context: ModelRequestContext) -> ModelRequestContext:
        calls.append('before')
        return request_context

    @hooks.on.after_model_request
    async def _after(
        ctx: RunContext[Any], *, request_context: ModelRequestContext, response: ModelResponse
    ) -> ModelResponse:
        calls.append('after')
        return response

    suspended = _suspended(texts=['partial '], provider_response_id='r1', input_tokens=5, output_tokens=2)
    history: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart(content='go')]),
        suspended,
    ]

    if stream:
        model: Model = _ScriptedModel(
            segments=[
                _StreamSegment(
                    texts=['done'], state='complete', provider_response_id='r2', input_tokens=3, output_tokens=4
                ),
            ]
        )
    else:

        def _model_fn(messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
            return ModelResponse(parts=[TextPart('done')], provider_response_id='r2')

        model = FunctionModel(_model_fn)

    agent = Agent(model, capabilities=[hooks])

    if stream:
        async with agent.run_stream(message_history=history) as result:
            output = await result.get_output()
        new_messages = result.new_messages()
    else:
        run_result = await agent.run(message_history=history)
        output = run_result.output
        new_messages = run_result.new_messages()

    assert 'done' in output
    # Only the final merged response is new (the suspended response was resumed, not re-recorded).
    assert len(new_messages) == snapshot(1)
    merged = new_messages[0]
    assert isinstance(merged, ModelResponse)
    assert merged.state == 'complete'
    # Hooks fire once around the resumed chain.
    assert calls == snapshot(['before', 'after'])


async def test_new_prompt_after_trailing_suspended_history_errors() -> None:
    """A new prompt on top of a suspended-ending history is rejected, not silently started as a fresh turn.

    Falling through to a new request would abandon the paused turn and leak the provider's server-side
    job; the caller must resume it (run with the suspended history and no new prompt) first.
    """
    history: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart(content='go')]),
        _suspended(texts=['partial '], provider_response_id='r1', input_tokens=5, output_tokens=2),
    ]
    agent = Agent(FunctionModel(lambda m, i: ModelResponse(parts=[TextPart('done')])))  # pragma: no cover

    with pytest.raises(UserError, match='ends in a suspended response'):
        await agent.run('new prompt', message_history=history)


async def test_error_on_first_streamed_segment_propagates() -> None:
    """An error raised while iterating the first streamed segment surfaces cleanly out of the node."""

    @dataclass
    class _ExplodingStream(StreamedResponse):
        async def _get_event_iterator(self) -> AsyncIterator[ModelResponseStreamEvent]:
            raise RuntimeError('stream exploded')
            yield  # pragma: no cover

        async def close_stream(self) -> None:
            pass

        @property
        def model_name(self) -> str:
            return 'scripted'

        @property
        def provider_name(self) -> str:
            return 'scripted'

        @property
        def provider_url(self) -> str:
            return 'scripted'

        @property
        def timestamp(self) -> datetime:
            return _TIMESTAMP

    class _ExplodingModel(_ScriptedModel):
        @asynccontextmanager
        async def request_stream(
            self,
            messages: list[ModelMessage],
            model_settings: ModelSettings | None,
            model_request_parameters: ModelRequestParameters,
            run_context: RunContext[Any] | None = None,
        ) -> AsyncGenerator[StreamedResponse]:
            yield _ExplodingStream(model_request_parameters)

    agent = Agent(_ExplodingModel())

    with pytest.raises(RuntimeError, match='stream exploded'):
        async with agent.run_stream('go'):
            pass


@pytest.mark.parametrize('stream', [False, True])
async def test_resume_history_without_preceding_request(stream: bool) -> None:
    """Resuming a history whose base messages contain no `ModelRequest` leaves `resumed_request` unset.

    A completed response precedes the suspended one, with no `ModelRequest` anywhere in the base
    history, so the reverse scan for the resumed request finds nothing and falls through.
    """
    prior = ModelResponse(parts=[TextPart('earlier')], model_name='scripted', provider_response_id='r0')
    suspended = _suspended(texts=['partial '], provider_response_id='r1', input_tokens=5, output_tokens=2)
    history: list[ModelMessage] = [prior, suspended]

    if stream:
        model: Model = _ScriptedModel(
            segments=[
                _StreamSegment(
                    texts=['done'], state='complete', provider_response_id='r2', input_tokens=3, output_tokens=4
                ),
            ]
        )
    else:

        def _model_fn(messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
            return ModelResponse(parts=[TextPart('done')], provider_response_id='r2')

        model = FunctionModel(_model_fn)

    agent = Agent(model)

    if stream:
        async with agent.run_stream(message_history=history) as result:
            output = await result.get_output()
    else:
        output = (await agent.run(message_history=history)).output

    assert 'done' in output


async def test_resume_hook_dropping_suspended_response_errors() -> None:
    """A `before_model_request` hook that strips the trailing suspended response breaks the resume precondition."""
    hooks = Hooks()

    @hooks.on.before_model_request
    async def _before(ctx: RunContext[Any], request_context: ModelRequestContext) -> ModelRequestContext:
        # Remove the trailing suspended response so the processed history no longer ends in one.
        return replace(request_context, messages=request_context.messages[:-1])

    suspended = _suspended(texts=['partial '], provider_response_id='r1', input_tokens=5, output_tokens=2)
    history: list[ModelMessage] = [ModelRequest(parts=[UserPromptPart(content='go')]), suspended]

    agent = Agent(FunctionModel(lambda m, i: ModelResponse(parts=[TextPart('done')])), capabilities=[hooks])

    with pytest.raises(UserError, match='must end with a suspended'):
        await agent.run(message_history=history)


async def test_streaming_wrap_error_propagates() -> None:
    """A `model_request` (wrap) hook raising a non-retry error in the streaming short-circuit propagates it."""
    hooks = Hooks()

    @hooks.on.model_request
    async def _wrap(ctx: RunContext[Any], *, request_context: ModelRequestContext, handler: Any) -> ModelResponse:
        raise RuntimeError('wrap boom')  # never calls the handler → streaming short-circuit path

    model = _ScriptedModel(
        segments=[
            _StreamSegment(texts=['x'], state='complete', provider_response_id='r1', input_tokens=1, output_tokens=1),
        ]
    )
    agent = Agent(model, capabilities=[hooks])

    with pytest.raises(RuntimeError, match='wrap boom'):
        async with agent.run_stream('go'):
            pass


async def test_cancel_through_wrapper_model_delegates() -> None:
    """Cancelling a continuation on a `WrapperModel` forwards `cancel_suspended_response` to the wrapped model."""
    inner = _ScriptedModel(
        segments=[
            _StreamSegment(
                texts=['a', 'b'],
                state='suspended',
                provider_response_id='r1',
                input_tokens=5,
                output_tokens=2,
                suspended_retry_delay=0.0,
            ),
            _StreamSegment(texts=['c'], state='complete', provider_response_id='r2', input_tokens=3, output_tokens=4),
        ]
    )
    agent = Agent(WrapperModel(inner))

    async with agent.run_stream('go') as result:
        async for _ in result.stream_text(delta=True, debounce_by=None):  # pragma: no branch
            break  # first (suspended) segment now in flight
        await result.cancel()

    assert len(inner.cancelled) == 1
    assert inner.cancelled[0].provider_response_id == 'r1'
