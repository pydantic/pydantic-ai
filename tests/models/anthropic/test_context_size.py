"""The context size of an Anthropic response whose server-side tool loop sampled the model several times.

A server-side tool loop (web search, web fetch, code execution) runs several sampling passes inside one request, and
the top-level `usage` sums them: that is what Anthropic bills, but no pass was that large. A beta that returns
`usage.iterations` (here `compact-2026-01-12`) reports each pass, and the last `message` pass's input, cached tokens
included, is the context the response leaves behind. `RunContext.context_window_used`, the per-request input token
limit and harness compaction read that instead of the sum.
"""

from __future__ import annotations as _annotations

from collections.abc import Callable

import pytest

from pydantic_ai import Agent, ModelMessage, ModelResponse, NativeToolCallPart, RunContext
from pydantic_ai.capabilities import NativeTool
from pydantic_ai.native_tools import WebSearchTool
from pydantic_ai.usage import RunUsage, UsageLimits

from ...conftest import RequestCapture, try_import
from ..conftest import AnthropicModelFactory
from ..test_anthropic import MockAnthropic, completion_message

with try_import() as imports_successful:
    from anthropic.types.beta import (
        BetaCompactionIterationUsage,
        BetaFallbackMessageIterationUsage,
        BetaIterationsUsage,
        BetaMessage,
        BetaMessageDeltaUsage,
        BetaMessageIterationUsage,
        BetaRawContentBlockStartEvent,
        BetaRawContentBlockStopEvent,
        BetaRawMessageDeltaEvent,
        BetaRawMessageStartEvent,
        BetaRawMessageStopEvent,
        BetaRawMessageStreamEvent,
        BetaTextBlock,
        BetaUsage,
    )
    from anthropic.types.beta.beta_raw_message_delta_event import Delta

    from pydantic_ai.models.anthropic import AnthropicModel, AnthropicModelSettings
    from pydantic_ai.providers.anthropic import AnthropicProvider

pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='anthropic not installed'),
    pytest.mark.vcr,
]

THREE_SEARCHES = (
    'Run three separate web searches, one after another, each in its own search call: '
    '(1) the population of Lyon, (2) the height of the Eiffel Tower, (3) the founding year of ETH Zurich. '
    'Then answer all three in one short sentence.'
)


def _web_search_agent(model: AnthropicModel, *, iterations: bool) -> Agent[None, str]:
    settings = AnthropicModelSettings(anthropic_betas=['compact-2026-01-12']) if iterations else None
    return Agent(model, capabilities=[NativeTool(WebSearchTool())], model_settings=settings)


def _context_window_used(model: AnthropicModel, messages: list[ModelMessage]) -> float | None:
    return RunContext(deps=None, model=model, usage=RunUsage(), messages=messages).context_window_used


# The recorded loop: the web searches the model ran (the streamed run searched once more than asked), the summed input
# and output of every pass, and the last `message` iteration's input (none of these requests read or wrote the cache).
SERVER_TOOL_LOOP_USAGE: dict[bool, tuple[int, int, int, int]] = {
    False: (3, 33_450, 920, 10_668),
    True: (4, 76_814, 1_366, 28_504),
}


def _web_searches(response: ModelResponse) -> int:
    """The `web_search` calls in `response`, not the `code_execution` calls Claude Haiku 5.5 can run them from."""
    return sum(isinstance(part, NativeToolCallPart) and part.tool_name == 'web_search' for part in response.parts)


@pytest.mark.parametrize('stream', [False, True])
async def test_server_tool_loop_reads_context_from_last_pass(
    allow_model_requests: None, anthropic_api_key: str, stream: bool
):
    """Sequential web searches run several passes; the context is the last pass's input, not their sum.

    The per-request limit sits between the two, so it holds only when the last pass is read as the context.
    """
    model = AnthropicModel('claude-haiku-5-5', provider=AnthropicProvider(api_key=anthropic_api_key))
    agent = _web_search_agent(model, iterations=True)
    usage_limits = UsageLimits(per_request_input_tokens_limit=30_000)

    if stream:
        async with agent.run_stream(THREE_SEARCHES, usage_limits=usage_limits) as result:
            await result.get_output()
        messages = result.all_messages()
    else:
        # Under the SDK's non-streaming output limit, so the response arrives as one JSON message rather than a
        # stream `AnthropicModel.request` reads behind the scenes.
        settings = AnthropicModelSettings(max_tokens=4096)
        messages = (await agent.run(THREE_SEARCHES, usage_limits=usage_limits, model_settings=settings)).all_messages()

    response = messages[-1]
    assert isinstance(response, ModelResponse)
    web_searches, summed_input_tokens, output_tokens, last_pass_input_tokens = SERVER_TOOL_LOOP_USAGE[stream]
    assert _web_searches(response) == web_searches
    # Usage keeps the sum Anthropic bills.
    assert response.usage.input_tokens == summed_input_tokens > 30_000 > last_pass_input_tokens
    assert response.usage.output_tokens == output_tokens
    assert response.metadata == {'__pydantic_ai__': {'context_input_tokens': last_pass_input_tokens}}
    assert _context_window_used(model, messages) == (last_pass_input_tokens + output_tokens) / 1_000_000


async def test_server_tool_loop_without_iterations_reads_summed_usage(
    allow_model_requests: None, anthropic_api_key: str
):
    """Without a beta that returns `usage.iterations`, the passes aren't reported, so the context reads the sum."""
    model = AnthropicModel('claude-haiku-5-5', provider=AnthropicProvider(api_key=anthropic_api_key))
    messages = (await _web_search_agent(model, iterations=False).run(THREE_SEARCHES)).all_messages()

    response = messages[-1]
    assert isinstance(response, ModelResponse)
    assert _web_searches(response) == 3
    assert (response.usage.input_tokens, response.usage.output_tokens) == (65_215, 2_296)
    assert response.metadata is None
    assert _context_window_used(model, messages) == (65_215 + 2_296) / 1_000_000


FIFTEEN_SEARCHES = (
    'Run fifteen separate web searches, one after another, each in its own search call, and do not answer until all '
    'of them are done: the population of (1) Lyon, (2) Porto, (3) Ghent, (4) Graz, (5) Turku, (6) Bergen, (7) Brno, '
    '(8) Split, (9) Malmo, (10) Basel, (11) Leiden, (12) Cork, (13) Tartu, (14) Kaunas, (15) Varna. '
    'Then list the fifteen numbers.'
)


# The recorded `usage`: each segment's summed input, and the last segment's last `message` iteration's input (no
# cache reads or writes). The first segment paused after 11 passes, the second finished after 5.
PAUSE_TURN_SEGMENT_INPUT_TOKENS = (419_501, 469_089)
PAUSE_TURN_LAST_PASS_INPUT_TOKENS = 106_549


async def test_pause_turn_reads_context_from_last_segment(
    allow_model_requests: None, anthropic_model: AnthropicModelFactory, request_capture: RequestCapture
):
    """A turn the server-side tool loop paused reads the context of its last segment's last pass.

    Each `pause_turn` continuation is a separate request with its own `usage.iterations`; the merged response sums
    their usage, and its context is the last pass of the segment that finished the turn.
    """
    # Claude 5 models run the `20260209` web search loop to `end_turn` without pausing, so this stays on Haiku 4.5.
    model = anthropic_model('claude-haiku-4-5', capture=True)
    messages = (await _web_search_agent(model, iterations=True).run(FIFTEEN_SEARCHES)).all_messages()

    assert len(request_capture.bodies('/v1/messages')) == len(PAUSE_TURN_SEGMENT_INPUT_TOKENS)
    response = messages[-1]
    assert isinstance(response, ModelResponse)
    assert response.state == 'complete'
    assert response.usage.input_tokens == sum(PAUSE_TURN_SEGMENT_INPUT_TOKENS)
    assert response.metadata == {'__pydantic_ai__': {'context_input_tokens': PAUSE_TURN_LAST_PASS_INPUT_TOKENS}}


def _iteration(input_tokens: int, output_tokens: int) -> BetaMessageIterationUsage:
    return BetaMessageIterationUsage(
        type='message',
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_read_input_tokens=0,
        cache_creation_input_tokens=0,
    )


def _segment_stream(message: BetaMessage) -> list[BetaRawMessageStreamEvent]:
    """The stream events of `message`: the start event reports the first pass, the delta the summed usage."""
    return [
        BetaRawMessageStartEvent(
            type='message_start',
            message=message.model_copy(
                update={'content': [], 'stop_reason': None, 'usage': BetaUsage(input_tokens=100, output_tokens=0)}
            ),
        ),
        BetaRawContentBlockStartEvent(type='content_block_start', index=0, content_block=message.content[0]),
        BetaRawContentBlockStopEvent(type='content_block_stop', index=0),
        BetaRawMessageDeltaEvent(
            type='message_delta',
            delta=Delta(stop_reason=message.stop_reason),
            usage=BetaMessageDeltaUsage(
                input_tokens=message.usage.input_tokens,
                output_tokens=message.usage.output_tokens,
                iterations=message.usage.iterations,
            ),
        ),
        BetaRawMessageStopEvent(type='message_stop'),
    ]


@pytest.mark.parametrize('stream', [False, True])
@pytest.mark.parametrize(
    ('done_last_pass_input_tokens', 'context_input_tokens'),
    [pytest.param(None, 500, id='done-without-iterations'), pytest.param(450, 450, id='done-with-iterations')],
)
async def test_pause_turn_segment_reads_its_own_context(
    allow_model_requests: None, stream: bool, done_last_pass_input_tokens: int | None, context_input_tokens: int
):
    """The segment that finishes a `pause_turn` chain doesn't inherit the last pass an earlier segment reported.

    The continuation merges the segments' metadata shallowly, so without a reading of its own the merged response would
    keep the first segment's last pass under both segments' summed usage: a segment that reports `usage.iterations`
    reads its own last pass, one that doesn't reads its own input. Not a VCR test: a `pause_turn` continuation resends
    the same betas, so the recorded chain reports `iterations` on every segment, and it isn't streamed.
    """
    paused = completion_message(
        [BetaTextBlock(text='paused', type='text')],
        BetaUsage(input_tokens=100 + 400, output_tokens=20, iterations=[_iteration(100, 10), _iteration(400, 10)]),
    )
    paused.stop_reason = 'pause_turn'
    done_iterations: BetaIterationsUsage | None = (
        None if done_last_pass_input_tokens is None else [_iteration(done_last_pass_input_tokens, 5)]
    )
    done = completion_message(
        [BetaTextBlock(text='done', type='text')],
        BetaUsage(input_tokens=500, output_tokens=5, iterations=done_iterations),
    )
    done.id = '456'
    if stream:
        client = MockAnthropic.create_stream_mock([_segment_stream(paused), _segment_stream(done)])
    else:
        client = MockAnthropic.create_mock([paused, done])
    model = AnthropicModel('claude-haiku-4-5', provider=AnthropicProvider(anthropic_client=client))
    agent = Agent(model)

    if stream:
        async with agent.run_stream('hello') as result:
            await result.get_output()
        messages = result.all_messages()
    else:
        messages = (await agent.run('hello')).all_messages()

    response = messages[-1]
    assert isinstance(response, ModelResponse)
    assert (response.usage.input_tokens, response.metadata) == (
        1000,
        {'__pydantic_ai__': {'context_input_tokens': context_input_tokens}},
    )


def _trailing_compaction() -> list[BetaMessageIterationUsage | BetaCompactionIterationUsage]:
    compaction = BetaCompactionIterationUsage(
        type='compaction', input_tokens=900, output_tokens=50, cache_read_input_tokens=0, cache_creation_input_tokens=0
    )
    return [_iteration(100, 10), _iteration(400, 10), compaction]


def _fallback_message() -> list[BetaMessageIterationUsage | BetaFallbackMessageIterationUsage]:
    fallback = BetaFallbackMessageIterationUsage(
        type='fallback_message',
        model='claude-sonnet-4-5',
        input_tokens=300,
        output_tokens=10,
        cache_read_input_tokens=0,
        cache_creation_input_tokens=0,
    )
    return [_iteration(100, 10), fallback]


def _cached_last_message() -> list[BetaMessageIterationUsage]:
    cached = BetaMessageIterationUsage(
        type='message', input_tokens=40, output_tokens=10, cache_read_input_tokens=200, cache_creation_input_tokens=50
    )
    return [_iteration(100, 10), cached]


@pytest.mark.parametrize(
    ('iterations', 'context_input_tokens'),
    [
        pytest.param(_trailing_compaction, 400, id='trailing-compaction'),
        pytest.param(_fallback_message, 300, id='fallback-message'),
        pytest.param(_cached_last_message, 40 + 200 + 50, id='cached-last-message'),
    ],
)
async def test_context_is_last_message_or_fallback_message_pass(
    allow_model_requests: None, iterations: Callable[[], BetaIterationsUsage], context_input_tokens: int
):
    """The context is the last `message` or `fallback_message` pass; a `compaction` entry isn't, even the last one.

    A `compaction` entry measures the compaction, not the context it leaves, and `fallback_message` replaces `message`
    as the last entry of a turn a fallback model served. The pass's cache reads and writes count toward its context.
    Not a VCR test: a server-side model fallback can't be triggered on demand, a response that ends with a compaction
    needs a history past the compaction threshold, and the recorded turns read and write no cache.
    """
    message = completion_message(
        [BetaTextBlock(text='done', type='text')],
        BetaUsage(input_tokens=500, output_tokens=20, iterations=iterations()),
    )
    model = AnthropicModel(
        'claude-haiku-4-5', provider=AnthropicProvider(anthropic_client=MockAnthropic.create_mock(message))
    )

    response = (await Agent(model).run('hello')).all_messages()[-1]

    assert isinstance(response, ModelResponse)
    assert response.metadata == {'__pydantic_ai__': {'context_input_tokens': context_input_tokens}}
