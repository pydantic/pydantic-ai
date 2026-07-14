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

import asyncio
from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from typing import Any, Literal

import pytest

from pydantic_ai import Agent, capture_run_messages, set_agent_graph_sleep
from pydantic_ai._agent_graph import _resolve_interrupted_stream_state  # pyright: ignore[reportPrivateUsage]
from pydantic_ai._run_context import RunContext
from pydantic_ai.capabilities import Hooks
from pydantic_ai.exceptions import UnexpectedModelBehavior, UserError
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
    agent = Agent(FunctionModel(lambda m, i: ModelResponse(parts=[TextPart('done')])))

    with pytest.raises(UserError, match='ends in a suspended response'):
        await agent.run('new prompt', message_history=history)


async def test_nonstream_continuation_cancel_failure_preserves_original_error() -> None:
    """A failing cancel during a non-streamed continuation must not mask the error that aborted the run.

    If `request()` raises mid-continuation and cancelling the suspended job then also raises (e.g. a
    transport error from the provider's cancel call), the caller must still see the original error, and
    the cancel must have been attempted (best-effort teardown of the server-side job).
    """

    class _CancelFailsModel(WrapperModel):
        def __init__(self, wrapped: Model) -> None:
            super().__init__(wrapped)
            self.cancel_attempts = 0

        async def cancel_suspended_response(self, response: ModelResponse) -> None:
            self.cancel_attempts += 1
            raise RuntimeError('cleanup failure')

    calls = 0

    def fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal calls
        calls += 1
        if calls == 1:
            return ModelResponse(parts=[TextPart('paused')], state='suspended')
        raise RuntimeError('original failure')

    model = _CancelFailsModel(FunctionModel(fn))
    agent = Agent(model)

    with pytest.raises(RuntimeError, match='original failure'):
        await agent.run('go')

    assert model.cancel_attempts == 1


class _RecordingCancelModel(WrapperModel):
    """A `WrapperModel` that records every `cancel_suspended_response` call."""

    def __init__(self, wrapped: Model) -> None:
        super().__init__(wrapped)
        self.cancelled: list[ModelResponse] = []

    async def cancel_suspended_response(self, response: ModelResponse) -> None:
        self.cancelled.append(response)


async def test_nonstream_cancel_during_retry_sleep_cancels_job() -> None:
    """A cancellation during the inter-poll retry sleep cancels the suspended job and propagates the error.

    The sleep between continuation polls sits outside the request's cancel guard, so a `CancelledError`
    raised while parked in it must still tear down the server-side job before propagating, or the
    background job leaks.
    """

    def fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart('paused')], state='suspended', suspended_retry_delay=0.5)

    model = _RecordingCancelModel(FunctionModel(fn))
    agent = Agent(model)

    async def _cancel_during_sleep(delay: float) -> None:
        raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError), set_agent_graph_sleep(_cancel_during_sleep):
        await agent.run('go')

    assert len(model.cancelled) == 1
    assert model.cancelled[0].state == 'suspended'


async def test_nonstream_max_continuations_cancels_job(monkeypatch: pytest.MonkeyPatch) -> None:
    """Hitting the max-continuations limit cancels the still-suspended job before raising, so it doesn't leak."""
    monkeypatch.setattr('pydantic_ai._agent_graph.MAX_CONTINUATIONS', 2)

    def fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart('paused')], state='suspended')

    model = _RecordingCancelModel(FunctionModel(fn))
    agent = Agent(model)

    with pytest.raises(UnexpectedModelBehavior, match='suspended more than the maximum'):
        await agent.run('go')

    assert len(model.cancelled) == 1
    assert model.cancelled[0].state == 'suspended'


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


@pytest.mark.parametrize('stream', [False, True])
async def test_background_poll_survives_past_max_continuations(stream: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    """A replace-style (same-id) background poll chain isn't killed by the small `MAX_CONTINUATIONS` cap.

    Each poll of a still-pending background job re-fetches the *same* job (shared `provider_response_id`),
    so it must not count against the accumulate ceiling that guards a runaway `pause_turn` model — a
    healthy background job that legitimately runs long has to be allowed to finish.
    """
    monkeypatch.setattr('pydantic_ai._agent_graph.MAX_CONTINUATIONS', 3)

    poll_count = 8  # far past the strict cap of 3
    if stream:
        segments = [
            _StreamSegment(texts=[], state='suspended', provider_response_id='job1', input_tokens=1, output_tokens=0)
            for _ in range(poll_count)
        ]
        segments.append(
            _StreamSegment(
                texts=['done'], state='complete', provider_response_id='job1', input_tokens=1, output_tokens=1
            )
        )
        model = _ScriptedModel(segments=segments)
    else:
        responses = [
            ModelResponse(
                parts=[],
                model_name='scripted',
                provider_response_id='job1',
                usage=RequestUsage(input_tokens=1, output_tokens=0),
                state='suspended',
                timestamp=_TIMESTAMP,
            )
            for _ in range(poll_count)
        ]
        responses.append(
            ModelResponse(
                parts=[TextPart('done')],
                model_name='scripted',
                provider_response_id='job1',
                usage=RequestUsage(input_tokens=1, output_tokens=1),
            )
        )
        model = _ScriptedModel(responses=responses)

    agent = Agent(model)

    if stream:
        async with agent.run_stream('go') as result:
            output = await result.get_output()
    else:
        output = (await agent.run('go')).output

    assert 'done' in output
    # The chain completed without raising `UnexpectedModelBehavior` and without cancelling the healthy job.
    assert model.cancelled == []


async def test_streamed_background_detach_records_suspended_and_resumes() -> None:
    """A streamed background run detached mid-flight records `state='suspended'` and resumes later.

    Kicking off a long background job over streaming, walking away (stop iterating, no `cancel()`), and
    resuming from the persisted history must work — consistent with the non-streaming path. Detaching
    leaves the server-side job alive (no cancel), records the trailing response as resumable
    `'suspended'`, and feeding that history back into a fresh run issues the continuation.
    """
    model = _ScriptedModel(
        segments=[
            _StreamSegment(
                texts=['partial '], state='suspended', provider_response_id='job1', input_tokens=5, output_tokens=2
            ),
            # Same id → a background poll would replace, not accumulate; never reached (we detach first).
            _StreamSegment(
                texts=['done'], state='complete', provider_response_id='job1', input_tokens=8, output_tokens=6
            ),
        ]
    )
    agent = Agent(model)

    recorded: list[ModelMessage] = []
    async with agent.iter('go') as run:
        node = run.next_node
        while not isinstance(node, End):  # pragma: no branch
            if Agent.is_model_request_node(node):
                async with node.stream(run.ctx) as stream:
                    iterator = stream.__aiter__()
                    await iterator.__anext__()  # first (suspended) background segment now in flight
                    # Detach: stop iterating WITHOUT cancelling.
                assert stream.response.state == 'suspended'
                recorded = run.ctx.state.message_history[:]
                break
            node = await run.next(node)

    # Detaching left the server-side job alive (no cancel) and requested no further poll.
    assert model.cancelled == []
    assert model.request_stream_calls == 1
    # The recorded trailing response is a resumable suspended response.
    assert isinstance(recorded[-1], ModelResponse)
    assert recorded[-1].state == 'suspended'
    assert recorded[-1].provider_response_id == 'job1'

    # Resume: feed the persisted suspended history to a fresh run; it must issue the continuation.
    resume_model = _ScriptedModel(
        segments=[
            _StreamSegment(
                texts=['done'], state='complete', provider_response_id='job1', input_tokens=8, output_tokens=6
            ),
        ]
    )
    resume_agent = Agent(resume_model)
    async with resume_agent.run_stream(message_history=recorded) as result:
        output = await result.get_output()
    new_messages = result.new_messages()

    assert 'done' in output
    assert resume_model.request_stream_calls == 1  # it resumed the turn rather than erroring or starting fresh
    # Only the final merged (complete) response is new; the suspended response was resumed, not re-recorded.
    assert len(new_messages) == snapshot(1)
    assert isinstance(new_messages[-1], ModelResponse)
    assert new_messages[-1].state == 'complete'


async def test_nonstream_model_change_chain_counts_against_strict_ceiling(monkeypatch: pytest.MonkeyPatch) -> None:
    """A non-streamed *model-change* replace chain is bound by the strict `MAX_CONTINUATIONS`, not the backstop.

    Each suspended segment reports a different `model_name`, so the merge *replaces* — but as *fresh*
    generation, not a same-id re-poll of one background job. It must therefore count against the strict
    cap (the runaway guard) rather than inherit the generous same-id backstop, and cancel the still-live
    job before raising.
    """
    monkeypatch.setattr('pydantic_ai._agent_graph.MAX_CONTINUATIONS', 2)

    def _suspended_model_change(model_name: str, provider_response_id: str) -> ModelResponse:
        return ModelResponse(
            parts=[TextPart('x')],
            model_name=model_name,
            provider_response_id=provider_response_id,
            usage=RequestUsage(input_tokens=1, output_tokens=1),
            state='suspended',
            timestamp=_TIMESTAMP,
        )

    # Three fresh suspensions under different models — past the strict cap of 2, which the generous same-id
    # backstop (10_000) would otherwise let sail through.
    model = _ScriptedModel(
        responses=[
            _suspended_model_change('m1', 'r1'),
            _suspended_model_change('m2', 'r2'),
            _suspended_model_change('m3', 'r3'),
        ]
    )
    agent = Agent(model)

    with pytest.raises(UnexpectedModelBehavior, match=r"Model response 'r3' was suspended more than the maximum"):
        await agent.run('go')

    # The still-suspended job was cancelled before the raise, so it doesn't leak.
    assert len(model.cancelled) == 1
    assert model.cancelled[0].provider_response_id == 'r3'


async def test_call_tools_node_rejects_suspended_response() -> None:
    """Driving `CallToolsNode` with a suspended response raises `UserError` instead of finalizing on partial output.

    After a streamed background run is detached under `agent.iter`, the trailing suspended response is
    persisted and `ModelRequestNode.run()` hands it to a `CallToolsNode`. A user who keeps driving the
    graph would otherwise have `CallToolsNode` treat the partial text as completed output — ending the run
    on a mid-turn answer while the server-side job keeps running. The guard rejects it, symmetric with
    `UserPromptNode`'s suspended guard.
    """
    model = _ScriptedModel(
        segments=[
            _StreamSegment(
                texts=['partial '], state='suspended', provider_response_id='job1', input_tokens=5, output_tokens=2
            ),
        ]
    )
    agent = Agent(model)

    async with agent.iter('go') as run:
        user_prompt_node = run.next_node
        assert not isinstance(user_prompt_node, End)
        node = await run.next(user_prompt_node)  # UserPromptNode → ModelRequestNode
        assert Agent.is_model_request_node(node)
        async with node.stream(run.ctx) as stream:
            iterator = stream.__aiter__()
            await iterator.__anext__()  # first (suspended) background segment now in flight
            # Detach: stop iterating WITHOUT cancelling.
        assert stream.response.state == 'suspended'

        # Keep driving: `ModelRequestNode.run()` returns a `CallToolsNode` holding the suspended response.
        call_tools_node = await run.next(node)
        assert Agent.is_call_tools_node(call_tools_node)
        with pytest.raises(UserError, match='suspended model response'):
            await run.next(call_tools_node)


async def test_run_stream_background_output_is_complete() -> None:
    """`run_stream` over a suspended→complete background job yields the COMPLETE output, not the partial.

    The suspended first segment streams partial text (and its own recognized `FinalResultEvent`), but the
    turn's final output is the terminal segment's — `run_stream`'s `get_output()` drains the whole chain
    before validating, so it never finalizes the run on the mid-turn partial.
    """
    model = _ScriptedModel(
        segments=[
            _StreamSegment(
                texts=['partial'], state='suspended', provider_response_id='job1', input_tokens=1, output_tokens=1
            ),
            # Same id → the terminal segment *replaces* the partial with the full answer.
            _StreamSegment(
                texts=['complete answer'],
                state='complete',
                provider_response_id='job1',
                input_tokens=1,
                output_tokens=1,
            ),
        ]
    )
    agent = Agent(model)

    async with agent.run_stream('go') as result:
        output = await result.get_output()

    assert output == snapshot('complete answer')
    assert model.request_stream_calls == 2  # it polled the job to completion, not stopped on the partial


async def test_run_stream_early_break_records_suspended_and_resumes() -> None:
    """Walking away from a streamed background job (`run_stream` break, no `cancel()`) keeps it resumable.

    Exiting `run_stream` after an early break sends `GeneratorExit` into the model-request stream. Only the
    graph can tell that apart from a genuine downstream failure: it records the trailing response as
    resumable `'suspended'` (not `'interrupted'`) and leaves the server-side job alive, mirroring the
    non-streaming detach. Feeding the persisted history to a fresh run issues the continuation.
    """
    model = _ScriptedModel(
        segments=[
            _StreamSegment(
                texts=['partial '], state='suspended', provider_response_id='job1', input_tokens=5, output_tokens=2
            ),
            _StreamSegment(
                texts=['done'], state='complete', provider_response_id='job1', input_tokens=8, output_tokens=6
            ),
        ]
    )
    agent = Agent(model)

    with capture_run_messages() as messages:
        async with agent.run_stream('go') as result:
            async for _ in result.stream_text(delta=True, debounce_by=None):  # pragma: no branch
                break  # walk away mid background job WITHOUT cancelling

    # Detaching left the server-side job alive (no cancel) and requested no further poll.
    assert model.cancelled == []
    assert model.request_stream_calls == 1
    # The persisted trailing response is a resumable suspended response.
    assert isinstance(messages[-1], ModelResponse)
    assert messages[-1].state == 'suspended'
    assert messages[-1].provider_response_id == 'job1'

    # Resume: feed the persisted suspended history to a fresh run; it issues the continuation.
    resume_model = _ScriptedModel(
        segments=[
            _StreamSegment(
                texts=['done'], state='complete', provider_response_id='job1', input_tokens=8, output_tokens=6
            ),
        ]
    )
    async with Agent(resume_model).run_stream(message_history=messages) as result:
        output = await result.get_output()
    assert 'done' in output
    assert resume_model.request_stream_calls == 1


async def test_run_stream_downstream_error_interrupts_and_cancels_job() -> None:
    """A genuine downstream error during `run_stream` interrupts the turn and cancels the still-live job.

    Unlike a walk-away detach, an exception raised by the consumer is a real failure: the graph records the
    trailing response as non-resumable `'interrupted'` and best-effort cancels the server-side job so it
    doesn't leak, mirroring the non-streaming cancel-on-error policy.
    """
    model = _ScriptedModel(
        segments=[
            _StreamSegment(
                texts=['partial '], state='suspended', provider_response_id='job1', input_tokens=5, output_tokens=2
            ),
            _StreamSegment(
                texts=['done'], state='complete', provider_response_id='job1', input_tokens=8, output_tokens=6
            ),
        ]
    )
    agent = Agent(model)

    with capture_run_messages() as messages:
        with pytest.raises(ValueError, match='downstream boom'):
            async with agent.run_stream('go') as result:
                async for _ in result.stream_text(delta=True, debounce_by=None):  # pragma: no branch
                    raise ValueError('downstream boom')

    assert isinstance(messages[-1], ModelResponse)
    assert messages[-1].state == 'interrupted'  # non-resumable
    # The still-live background job was cancelled best-effort so it doesn't leak.
    assert len(model.cancelled) == 1
    assert model.cancelled[0].provider_response_id == 'job1'


async def test_iter_node_stream_early_break_records_suspended() -> None:
    """Breaking out of a `ModelRequestNode` stream via `agent.iter` mid-suspended-segment detaches.

    Unlike the `run_stream` early break (which records via the composite's own detach path), exiting a
    `node.stream(...)` context sends `GeneratorExit` into the graph's model-request stream, where
    `_resolve_interrupted_stream_state` records the trailing response as resumable `'suspended'` and leaves
    the server-side job alive — the graph-level detach branch.
    """
    model = _ScriptedModel(
        segments=[
            _StreamSegment(
                texts=['partial '], state='suspended', provider_response_id='job1', input_tokens=5, output_tokens=2
            ),
            _StreamSegment(
                texts=['done'], state='complete', provider_response_id='job1', input_tokens=8, output_tokens=6
            ),
        ]
    )
    agent = Agent(model)

    with capture_run_messages() as messages:
        async with agent.iter('go') as run:
            node = run.next_node
            while not isinstance(node, End):
                if Agent.is_model_request_node(node):
                    async with node.stream(run.ctx) as stream:
                        async for _ in stream:  # pragma: no branch
                            break  # walk away mid suspended segment, without cancelling
                    break  # stop driving the run
                node = await run.next(node)

    assert model.cancelled == []  # detach left the job alive
    assert isinstance(messages[-1], ModelResponse)
    assert messages[-1].state == 'suspended'  # resumable
    assert messages[-1].provider_response_id == 'job1'


async def test_resolve_interrupted_stream_state_detach_vs_error() -> None:
    """`_resolve_interrupted_stream_state`: a `GeneratorExit` walk-away with a suspended partial is a
    resumable detach (`'suspended'`, job left alive); any other error is a genuine failure (`'interrupted'`,
    job best-effort cancelled).

    Unit-tested directly on the graph helper: whether the `GeneratorExit`-vs-error branch fires depends on
    stream-teardown ordering (which handle is closed first) that the `run_stream`/`iter` early-break tests
    above don't deterministically drive, so the branch itself is pinned here.
    """
    model = _ScriptedModel(segments=[])
    suspended = _suspended(texts=['partial '], provider_response_id='job1', input_tokens=1, output_tokens=1)

    assert await _resolve_interrupted_stream_state(model, GeneratorExit(), suspended) == 'suspended'
    assert model.cancelled == []  # detach leaves the job alive

    assert await _resolve_interrupted_stream_state(model, RuntimeError('boom'), suspended) == 'interrupted'
    assert model.cancelled == [suspended]  # genuine error cancels the still-live job


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
