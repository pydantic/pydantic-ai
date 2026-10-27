"""`FallbackCompaction(background_start=...)`: summarize ahead of the trigger, swap the summary in when due."""

import asyncio
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass, field

import anyio
import pytest
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from pydantic_ai import Agent, ModelHTTPError, RunContext, UserError
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import (
    CompactionPart,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.models import ModelRequestContext
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.instrumented import InstrumentationSettings
from pydantic_ai.usage import RequestUsage
from pydantic_ai_harness.compaction import FallbackCompaction, SlidingWindowCompaction, SummarizingCompaction
from pydantic_ai_harness.compaction._fallback_compaction import MAX_BACKGROUND_COMPACTIONS

READINESS_WAIT_TIMEOUT = 10

TRIGGER = 2_000
"""`max_tokens` of every chain here; `background_start=0.5` starts the summary above 1,000."""

OLD = 'old ' * 4_000
"""The opening prompt: about 4,000 tokens of text, which the summary replaces."""

OVER = 'more ' * 800
"""A prompt of about 1,000 tokens: after a 1,150-token response, it takes the next request over the trigger."""


@dataclass
class Summarizer:
    """The summary model: waits for `release`, then answers, raises while `failures` remain, or is cancelled."""

    release: asyncio.Event = field(default_factory=asyncio.Event)
    started: asyncio.Event = field(default_factory=asyncio.Event)
    failures: int = 0
    cancel: bool = False
    calls: int = 0
    finished: int = 0
    cancelled: int = 0

    async def __call__(self, messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        self.calls += 1
        self.started.set()
        try:
            await self.release.wait()
        except asyncio.CancelledError:
            self.cancelled += 1
            raise
        if self.cancel:
            self.cancel = False
            raise asyncio.CancelledError
        if self.failures:
            self.failures -= 1
            raise ModelHTTPError(status_code=503, model_name='summarizer', body=None)
        self.finished += 1
        return ModelResponse(parts=[TextPart(f'summary {self.calls}')])

    async def wait_started(self) -> None:
        with anyio.fail_after(READINESS_WAIT_TIMEOUT):
            await self.started.wait()


@dataclass
class Main:
    """The conversation's model. Each response reports `input_tokens`, the context size the trigger reads."""

    input_tokens: list[int]
    tool_first: bool = False
    answer_after: Summarizer | None = None
    """When set, the final answer waits until this summarizer has started."""

    async def __call__(self, messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        usage = RequestUsage(input_tokens=self.input_tokens.pop(0))
        last = messages[-1]
        if self.tool_first and not any(isinstance(part, ToolReturnPart) for part in last.parts):
            return ModelResponse(parts=[ToolCallPart('lookup', {}, tool_call_id='call-1')], usage=usage)
        if self.answer_after is not None:
            await self.answer_after.wait_started()
        return ModelResponse(parts=[TextPart('done')], usage=usage)


@dataclass
class Requesting(AbstractCapability[None]):
    """Sets `event` as a request starts, before the compaction listed after it runs."""

    event: asyncio.Event = field(default_factory=asyncio.Event)

    async def before_model_request(
        self, ctx: RunContext[None], request_context: ModelRequestContext
    ) -> ModelRequestContext:
        self.event.set()
        return request_context

    async def wait(self) -> None:
        with anyio.fail_after(READINESS_WAIT_TIMEOUT):
            await self.event.wait()


def chain(summarizer: Summarizer, *, background_start: float | None = 0.5) -> FallbackCompaction[None]:
    return FallbackCompaction(
        fallback_chain=[
            SummarizingCompaction(
                model=FunctionModel(summarizer, model_name='summarizer'),
                max_messages=1,
                keep_messages=1,
                preserve_first_user_message=False,
            ),
            SlidingWindowCompaction(max_messages=1, keep_messages=1),
        ],
        max_tokens=TRIGGER,
        background_start=background_start,
    )


def agent(
    main: Main,
    compaction: FallbackCompaction[None],
    name: str = 'main',
    *,
    before: tuple[AbstractCapability[None], ...] = (),
) -> Agent[None, str]:
    built = Agent(FunctionModel(main, model_name=name), deps_type=type(None), capabilities=[*before, compaction])

    @built.tool_plain
    def lookup() -> str:
        return 'found it'

    return built


def opening(input_tokens: int) -> list[ModelMessage]:
    """A first turn, from an earlier run, whose response reported `input_tokens`."""
    return [
        ModelRequest.user_text_prompt(OLD),
        ModelResponse(parts=[TextPart('ok')], usage=RequestUsage(input_tokens=input_tokens)),
    ]


def summaries(messages: list[ModelMessage]) -> list[str | None]:
    return [
        part.content
        for message in messages
        if isinstance(message, ModelRequest)
        for part in message.parts
        if isinstance(part, CompactionPart)
    ]


@contextmanager
def traced() -> Generator[InMemorySpanExporter]:
    spans = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(spans))
    Agent.instrument_all(InstrumentationSettings(tracer_provider=provider))
    try:
        yield spans
    finally:
        Agent.instrument_all(False)


def named(spans: InMemorySpanExporter, name: str) -> list[ReadableSpan]:
    return [span for span in spans.get_finished_spans() if span.name == name]


async def start_background(summarizer: Summarizer, compaction: FallbackCompaction[None]) -> list[ModelMessage]:
    """One run that starts a background summary and ends while it is still being written."""
    first = await agent(Main([1_150]), compaction).run('next', message_history=opening(1_100), conversation_id='c')
    await summarizer.wait_started()
    return first.all_messages()


@pytest.mark.parametrize(('input_tokens', 'starts'), [(990, False), (1_100, True)])
async def test_starts_above_the_background_threshold_only(input_tokens: int, starts: bool) -> None:
    summarizer = Summarizer()
    compaction = chain(summarizer)
    result = await agent(Main([input_tokens]), compaction).run(
        'next', message_history=opening(input_tokens), conversation_id='c'
    )
    assert ('c' in compaction.background_conversations) == starts
    if starts:
        await summarizer.wait_started()
    assert summarizer.calls == int(starts)
    assert len(result.all_messages()) == 4, 'the run went on without waiting for the summary'
    await compaction.aclose()


async def test_off_by_default_compacts_only_at_the_trigger() -> None:
    summarizer = Summarizer()
    summarizer.release.set()
    compaction = chain(summarizer, background_start=None)
    first = await agent(Main([1_900]), compaction).run('next', message_history=opening(1_900), conversation_id='c')
    assert 'c' not in compaction.background_conversations and summarizer.calls == 0
    second = await agent(Main([500]), compaction).run(
        'more ' * 200, message_history=first.all_messages(), conversation_id='c'
    )
    assert summarizer.calls == 1
    assert summaries(second.all_messages()) == ['summary 1']


async def test_held_summary_swaps_in_and_keeps_what_came_after_verbatim() -> None:
    summarizer = Summarizer()
    compaction = chain(summarizer)
    with traced() as spans:
        first = await agent(Main([1_150, 1_200], tool_first=True), compaction).run(
            'next', message_history=opening(1_100), conversation_id='c'
        )
        await summarizer.wait_started()
        assert 'c' in compaction.background_conversations and summarizer.finished == 0
        # The run called a tool and answered while the summary was still being written.
        added = first.all_messages()[3:]
        assert [type(message) for message in added] == [ModelResponse, ModelRequest, ModelResponse]
        assert first.usage.requests == 2, 'nothing has finished yet to count'

        summarizer.release.set()
        second = await agent(Main([500]), compaction).run(
            OVER, message_history=first.all_messages(), conversation_id='c'
        )
    summary, kept, *rest = second.all_messages()
    assert summaries([summary]) == ['summary 1']
    assert kept == first.all_messages()[2], 'the request the snapshot ended with is the tail the summary kept'
    assert rest[:3] == added, 'the tool call and its return, added during the summary, are kept verbatim'
    assert summarizer.calls == 1, 'no second summary: the held one was used'
    assert 'c' not in compaction.background_conversations
    assert first.usage.requests == 3, 'the summary request counts toward the run that started it'
    assert second.usage.requests == 1
    [background] = named(spans, 'background_compaction')
    assert dict(background.attributes or {}) == {
        'compaction.strategy': 'SummarizingCompaction',
        'compaction.messages_before': 3,
    }
    [compacted] = named(spans, 'compact_messages')
    assert compacted.attributes is not None and compacted.attributes['compaction.background'] == 'used'


async def test_waits_for_a_summary_still_being_written() -> None:
    summarizer = Summarizer()
    compaction = chain(summarizer)
    history = await start_background(summarizer, compaction)
    requesting = Requesting()
    run = asyncio.create_task(
        agent(Main([500]), compaction, before=(requesting,)).run(OVER, message_history=history, conversation_id='c')
    )
    await requesting.wait()
    assert not run.done() and summarizer.finished == 0
    summarizer.release.set()
    second = await run
    assert summarizer.calls == 1
    assert summaries(second.all_messages()) == ['summary 1']


async def test_cancelled_wait_leaves_the_summary_for_later() -> None:
    summarizer = Summarizer()
    compaction = chain(summarizer)
    history = await start_background(summarizer, compaction)
    requesting = Requesting()
    run = asyncio.create_task(
        agent(Main([500]), compaction, before=(requesting,)).run(OVER, message_history=history, conversation_id='c')
    )
    await requesting.wait()
    run.cancel()
    await asyncio.wait((run,))
    assert 'c' in compaction.background_conversations and summarizer.cancelled == 0
    summarizer.release.set()
    second = await agent(Main([500]), compaction).run(OVER, message_history=history, conversation_id='c')
    assert summarizer.calls == 1
    assert summaries(second.all_messages()) == ['summary 1']


async def test_discarding_while_a_request_waits_falls_back_to_blocking_compaction() -> None:
    summarizer = Summarizer()
    compaction = chain(summarizer)
    history = await start_background(summarizer, compaction)
    requesting = Requesting()
    run = asyncio.create_task(
        agent(Main([500]), compaction, before=(requesting,)).run(OVER, message_history=history, conversation_id='c')
    )
    await requesting.wait()
    await compaction.aclose()
    assert summarizer.cancelled == 1
    summarizer.release.set()
    second = await run
    assert summarizer.calls == 2 and summarizer.finished == 1
    assert summaries(second.all_messages()) == ['summary 2']
    assert not compaction.background_conversations


async def compact_after(
    summarizer: Summarizer, prompt: str, *, model: str = 'main', outcome: str = 'stale'
) -> list[ModelMessage]:
    """Start a background summary in one run, then let `prompt` push the next run over the trigger."""
    compaction = chain(summarizer)
    history = await start_background(summarizer, compaction)
    summarizer.release.set()
    with traced() as spans:
        second = await agent(Main([500]), compaction, model).run(prompt, message_history=history, conversation_id='c')
    assert 'c' not in compaction.background_conversations
    [compacted] = named(spans, 'compact_messages')
    assert compacted.attributes is not None and compacted.attributes['compaction.background'] == outcome
    return second.all_messages()


async def test_too_much_new_content_falls_back_to_blocking_compaction() -> None:
    """The new prompt alone keeps the swapped-in history above the trigger."""
    summarizer = Summarizer()
    messages = await compact_after(summarizer, 'more ' * 6_000)
    assert summarizer.calls == 2, 'the held summary was dropped and the chain summarized again'
    assert summaries(messages) == ['summary 2']


async def test_model_switch_falls_back_to_blocking_compaction() -> None:
    summarizer = Summarizer()
    messages = await compact_after(summarizer, OVER, model='other')
    assert summarizer.calls == 2
    assert summaries(messages) == ['summary 2']


async def test_summary_failure_falls_back_to_blocking_compaction() -> None:
    summarizer = Summarizer(failures=1)
    messages = await compact_after(summarizer, OVER, outcome='failed')
    assert summarizer.calls == 2 and summarizer.finished == 1
    assert summaries(messages) == ['summary 2']


async def test_cancelled_summary_falls_back_to_blocking_compaction() -> None:
    summarizer = Summarizer(cancel=True)
    messages = await compact_after(summarizer, OVER, outcome='failed')
    assert summarizer.calls == 2 and summarizer.finished == 1
    assert summaries(messages) == ['summary 2']


async def test_rewritten_history_drops_the_summary() -> None:
    summarizer = Summarizer()
    compaction = chain(summarizer)
    history = await start_background(summarizer, compaction)
    # A side run with a history of its own leaves it alone.
    await agent(Main([10]), compaction).run('side', conversation_id='c')
    assert 'c' in compaction.background_conversations and summarizer.cancelled == 0
    # Rewinding the conversation below the background threshold cancels it.
    rewound = [history[0], ModelResponse(parts=[TextPart('ok')], usage=RequestUsage(input_tokens=200))]
    await agent(Main([210]), compaction).run('again', message_history=rewound, conversation_id='c')
    assert 'c' not in compaction.background_conversations and summarizer.cancelled == 1


async def test_another_history_above_the_threshold_replaces_the_summary() -> None:
    summarizer = Summarizer()
    compaction = chain(summarizer)
    await start_background(summarizer, compaction)
    other = [
        ModelRequest.user_text_prompt('elsewhere'),
        ModelResponse(parts=[TextPart('ok')], usage=RequestUsage(input_tokens=1_100)),
    ]
    summarizer.started.clear()
    await agent(Main([1_150]), compaction).run('next', message_history=other, conversation_id='c')
    await summarizer.wait_started()
    assert summarizer.cancelled == 1 and summarizer.calls == 2
    assert 'c' in compaction.background_conversations
    await compaction.aclose()
    assert summarizer.cancelled == 2


async def test_a_conversations_first_run_leaves_its_summary_running() -> None:
    """The summary started in the run that began the conversation can finish before the next run."""
    summarizer = Summarizer()
    compaction = chain(summarizer)
    main = Main([1_100, 1_200], tool_first=True, answer_after=summarizer)
    result = await agent(main, compaction).run('go', conversation_id='c')
    assert summarizer.calls == 1, 'started on the second request'
    assert 'c' in compaction.background_conversations and summarizer.cancelled == 0
    assert len(result.all_messages()) == 4
    await compaction.discard_background_compaction('c')
    assert summarizer.cancelled == 1


async def test_a_run_ending_over_the_trigger_keeps_its_summary_for_the_next_request() -> None:
    summarizer = Summarizer()
    compaction = chain(summarizer)
    await agent(Main([2_100]), compaction).run('next', message_history=opening(1_100), conversation_id='c')
    await summarizer.wait_started()
    assert 'c' in compaction.background_conversations and summarizer.calls == 1
    await compaction.aclose()


async def test_take_background_compaction_outside_a_run() -> None:
    summarizer = Summarizer()
    compaction = chain(summarizer)
    history = [*await start_background(summarizer, compaction), ModelRequest.user_text_prompt('later')]
    summarizer.release.set()
    main = FunctionModel(Main([]), model_name='main')
    spans = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(spans))
    compacted = await compaction.take_background_compaction(
        history, model=main, conversation_id='c', tracer=provider.get_tracer('test')
    )
    assert compacted is not None and summaries(compacted) == ['summary 1']
    [span] = named(spans, 'compact_messages')
    assert span.attributes is not None and span.attributes['compaction.background'] == 'used'
    assert compacted[-3:] == history[-3:]
    assert summarizer.calls == 1 and 'c' not in compaction.background_conversations
    assert await compaction.take_background_compaction(history, model=main, conversation_id='c') is None


async def test_take_background_compaction_drops_a_stale_one() -> None:
    summarizer = Summarizer()
    compaction = chain(summarizer)
    history = await start_background(summarizer, compaction)
    other = FunctionModel(Main([]), model_name='other')
    assert await compaction.take_background_compaction(history, model=other, conversation_id='c') is None
    assert 'c' not in compaction.background_conversations and summarizer.cancelled == 1


async def test_discard_and_aclose_cancel_running_work() -> None:
    summarizer = Summarizer()
    compaction = chain(summarizer)
    before = asyncio.all_tasks()
    await start_background(summarizer, compaction)
    await compaction.discard_background_compaction('c')
    await compaction.discard_background_compaction('c')
    assert 'c' not in compaction.background_conversations and summarizer.cancelled == 1
    summarizer.started.clear()
    await start_background(summarizer, compaction)
    await compaction.aclose()
    assert 'c' not in compaction.background_conversations and summarizer.cancelled == 2
    assert asyncio.all_tasks() == before


async def test_a_run_ending_in_the_background_band_starts_a_summary_for_the_next_run() -> None:
    summarizer = Summarizer()
    compaction = chain(summarizer)
    first = await agent(Main([1_200]), compaction).run('next', message_history=opening(900), conversation_id='c')
    await summarizer.wait_started()
    assert 'c' in compaction.background_conversations and summarizer.calls == 1, 'started when the run ended'
    summarizer.release.set()
    second = await agent(Main([500]), compaction).run(
        'more ' * 900, message_history=first.all_messages(), conversation_id='c'
    )
    assert summaries(second.all_messages()) == ['summary 1'] and summarizer.calls == 1
    await compaction.aclose()


async def test_holds_a_bounded_number_of_conversations() -> None:
    """Conversations that end without a discard, such as one-shot runs, never pile up past the bound."""
    summarizer = Summarizer()
    compaction = chain(summarizer)
    conversations = [f'c{index}' for index in range(MAX_BACKGROUND_COMPACTIONS + 1)]
    for conversation_id in conversations:
        await agent(Main([1_150]), compaction).run(
            'next', message_history=opening(1_100), conversation_id=conversation_id
        )
    assert compaction.background_conversations == frozenset(conversations[1:]), 'the oldest was cancelled'
    # Replacing a held conversation's summary reuses its slot instead of evicting another.
    other = [
        ModelRequest.user_text_prompt('elsewhere'),
        ModelResponse(parts=[TextPart('ok')], usage=RequestUsage(input_tokens=1_100)),
    ]
    await agent(Main([1_150]), compaction).run('next', message_history=other, conversation_id=conversations[3])
    assert compaction.background_conversations == frozenset(conversations[1:])
    await compaction.aclose()


async def test_focus_copy_holds_nothing() -> None:
    summarizer = Summarizer()
    compaction = chain(summarizer)
    await start_background(summarizer, compaction)
    assert not compaction.with_focus('auth').background_conversations
    await compaction.aclose()


async def test_rejected_inside_a_durable_workflow() -> None:
    class DBOSDurability(AbstractCapability[None]):
        in_durable_context = True

    DBOSDurability.__module__ = 'pydantic_ai.durable_exec.dbos'
    compaction = chain(Summarizer())
    with pytest.raises(UserError, match='durable workflow or flow'):
        await agent(Main([]), compaction, before=(DBOSDurability(),)).run('hi')


@pytest.mark.parametrize('start', [0.0, 1.0])
def test_background_start_is_a_fraction_below_one(start: float) -> None:
    with pytest.raises(ValueError, match='background_start must be'):
        FallbackCompaction(
            fallback_chain=[SlidingWindowCompaction(max_messages=1)], max_tokens=1, background_start=start
        )


def test_background_start_needs_a_trigger() -> None:
    with pytest.raises(ValueError, match='needs a max_tokens or max_fraction trigger'):
        FallbackCompaction(fallback_chain=[SlidingWindowCompaction(max_messages=1)], background_start=0.5)
