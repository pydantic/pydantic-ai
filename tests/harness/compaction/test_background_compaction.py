"""`FallbackCompaction(background_start=...)`: summarize ahead of the trigger, swap the summary in when due."""

import asyncio
from dataclasses import dataclass, field

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from pydantic_ai import Agent, ModelHTTPError, RunContext
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.models import ModelRequestContext
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.instrumented import InstrumentationSettings
from pydantic_ai.usage import RequestUsage
from pydantic_ai_harness.compaction import (
    FallbackCompaction,
    SlidingWindowCompaction,
    SummarizingCompaction,
    compact_now,
)

TRIGGER = 2_000
"""`max_tokens` of every chain here; `background_start=0.5` starts the summary above 1,000."""

OLD = 'old ' * 4_000
"""The opening prompt: about 4,000 tokens of text, which the summary replaces."""

OVER = 'more ' * 800
"""A prompt of about 1,000 tokens: after a 1,150-token response, it takes the next request over the trigger."""


@dataclass
class Summarizer:
    """The summary model: waits for `release`, then answers, or raises while `failures` remain."""

    release: asyncio.Event = field(default_factory=asyncio.Event)
    started: asyncio.Event = field(default_factory=asyncio.Event)
    failures: int = 0
    calls: int = 0
    finished: int = 0

    async def __call__(self, messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        self.calls += 1
        self.started.set()
        await self.release.wait()
        if self.failures:
            self.failures -= 1
            raise ModelHTTPError(status_code=503, model_name='summarizer', body=None)
        self.finished += 1
        return ModelResponse(parts=[TextPart(f'summary {self.calls}')])


@dataclass
class Main:
    """The conversation's model. Each response reports `input_tokens`, the context size the trigger reads."""

    input_tokens: list[int]
    tool_first: bool = False

    def __call__(self, messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        usage = RequestUsage(input_tokens=self.input_tokens.pop(0))
        last = messages[-1]
        if self.tool_first and not any(isinstance(part, ToolReturnPart) for part in last.parts):
            return ModelResponse(parts=[ToolCallPart('lookup', {}, tool_call_id='call-1')], usage=usage)
        return ModelResponse(parts=[TextPart('done')], usage=usage)


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


def agent(main: Main, compaction: FallbackCompaction[None], name: str = 'main') -> Agent[None, str]:
    built = Agent(FunctionModel(main, model_name=name), deps_type=type(None), capabilities=[compaction])

    @built.tool_plain
    def lookup() -> str:
        return 'found it'

    return built


def opening(input_tokens: int) -> list[ModelMessage]:
    """A first turn whose response reported `input_tokens`."""
    return [
        ModelRequest.user_text_prompt(OLD),
        ModelResponse(parts=[TextPart('ok')], usage=RequestUsage(input_tokens=input_tokens)),
    ]


def summaries(messages: list[ModelMessage]) -> list[str]:
    return [
        part.content
        for message in messages
        if isinstance(message, ModelRequest)
        for part in message.parts
        if isinstance(part, SystemPromptPart)
    ]


@pytest.mark.parametrize(('input_tokens', 'starts'), [(990, False), (1_100, True)])
async def test_starts_above_the_background_threshold_only(input_tokens: int, starts: bool) -> None:
    summarizer = Summarizer()
    compaction = chain(summarizer)
    result = await agent(Main([input_tokens]), compaction).run(
        'next', message_history=opening(input_tokens), conversation_id='c'
    )
    assert compaction.has_background_compaction('c') == starts
    if starts:
        await summarizer.started.wait()
    assert summarizer.calls == int(starts)
    assert len(result.all_messages()) == 4, 'the run went on without waiting for the summary'
    await compaction.aclose()


async def test_off_by_default_compacts_only_at_the_trigger() -> None:
    summarizer = Summarizer()
    summarizer.release.set()
    compaction = chain(summarizer, background_start=None)
    first = await agent(Main([1_900]), compaction).run('next', message_history=opening(1_900), conversation_id='c')
    assert not compaction.has_background_compaction('c') and summarizer.calls == 0
    second = await agent(Main([500]), compaction).run(
        'more ' * 200, message_history=first.all_messages(), conversation_id='c'
    )
    assert summarizer.calls == 1
    assert summaries(second.all_messages()) == ['Summary of previous conversation:\n\nsummary 1']


async def test_held_summary_swaps_in_and_keeps_what_came_after_verbatim() -> None:
    summarizer = Summarizer()
    compaction = chain(summarizer)
    main = Main([1_150, 1_200], tool_first=True)
    first = await agent(main, compaction).run('next', message_history=opening(1_100), conversation_id='c')
    await summarizer.started.wait()
    assert compaction.has_background_compaction('c') and summarizer.finished == 0
    # The run called a tool and answered while the summary was still being written.
    added = first.all_messages()[3:]
    assert [type(message) for message in added] == [ModelResponse, ModelRequest, ModelResponse]

    summarizer.release.set()
    spans = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(spans))
    Agent.instrument_all(InstrumentationSettings(tracer_provider=provider))
    try:
        second = await agent(Main([500]), compaction).run(
            OVER, message_history=first.all_messages(), conversation_id='c'
        )
    finally:
        Agent.instrument_all(False)
    messages = second.all_messages()
    summary, kept, *rest = messages
    assert summaries([summary]) == ['Summary of previous conversation:\n\nsummary 1']
    assert kept == first.all_messages()[2], 'the request the snapshot ended with is the tail the summary kept'
    assert rest[:3] == added, 'the tool call and its return, added during the summary, are kept verbatim'
    assert summarizer.calls == 1, 'no second summary: the held one was used'
    assert not compaction.has_background_compaction('c')
    assert second.usage.requests == 2, 'the summary request is billed to the run that used it'
    [span] = [span for span in spans.get_finished_spans() if span.name == 'compact_messages']
    assert span.attributes is not None and span.attributes['compaction.background'] == 'used'


@dataclass
class Requesting(AbstractCapability[None]):
    """Sets `event` as a request starts, before the compaction listed after it runs."""

    event: asyncio.Event = field(default_factory=asyncio.Event)

    async def before_model_request(
        self, ctx: RunContext[None], request_context: ModelRequestContext
    ) -> ModelRequestContext:
        self.event.set()
        return request_context


async def test_waits_for_a_summary_still_being_written() -> None:
    summarizer = Summarizer()
    compaction = chain(summarizer)
    first = await agent(Main([1_150]), compaction).run('next', message_history=opening(1_100), conversation_id='c')
    await summarizer.started.wait()
    requesting = Requesting()
    waiting = Agent(
        FunctionModel(Main([500]), model_name='main'), deps_type=type(None), capabilities=[requesting, compaction]
    )
    run = asyncio.create_task(waiting.run(OVER, message_history=first.all_messages(), conversation_id='c'))
    await requesting.event.wait()
    assert not run.done() and summarizer.finished == 0
    summarizer.release.set()
    second = await run
    assert summarizer.calls == 1
    assert summaries(second.all_messages()) == ['Summary of previous conversation:\n\nsummary 1']


async def test_cancelled_wait_leaves_the_summary_for_later() -> None:
    summarizer = Summarizer()
    compaction = chain(summarizer)
    first = await agent(Main([1_150]), compaction).run('next', message_history=opening(1_100), conversation_id='c')
    requesting = Requesting()
    waiting = Agent(
        FunctionModel(Main([500]), model_name='main'), deps_type=type(None), capabilities=[requesting, compaction]
    )
    run = asyncio.create_task(waiting.run(OVER, message_history=first.all_messages(), conversation_id='c'))
    await requesting.event.wait()
    run.cancel()
    await asyncio.wait((run,))
    assert compaction.has_background_compaction('c')
    summarizer.release.set()
    second = await agent(Main([500]), compaction).run(OVER, message_history=first.all_messages(), conversation_id='c')
    assert summarizer.calls == 1
    assert summaries(second.all_messages()) == ['Summary of previous conversation:\n\nsummary 1']


async def test_another_history_above_the_threshold_replaces_the_summary() -> None:
    summarizer = Summarizer()
    compaction = chain(summarizer)
    await agent(Main([1_150]), compaction).run('next', message_history=opening(1_100), conversation_id='c')
    await summarizer.started.wait()
    [job] = compaction._background.values()  # pyright: ignore[reportPrivateUsage]
    other = [
        ModelRequest.user_text_prompt('elsewhere'),
        ModelResponse(parts=[TextPart('ok')], usage=RequestUsage(input_tokens=1_100)),
    ]
    await agent(Main([1_150]), compaction).run('next', message_history=other, conversation_id='c')
    assert job.task.cancelled() and summarizer.calls == 2
    await compaction.aclose()


async def test_a_run_ending_over_the_trigger_starts_nothing() -> None:
    summarizer = Summarizer()
    summarizer.release.set()
    compaction = chain(summarizer)
    await agent(Main([2_100]), compaction).run('next', message_history=opening(2_100), conversation_id='c')
    assert summarizer.calls == 1, 'the blocking compaction'
    assert not compaction.has_background_compaction('c')


async def compact_after(summarizer: Summarizer, prompt: str, *, model: str = 'main') -> list[ModelMessage]:
    """Start a background summary in one run, then let `prompt` push the next run over the trigger."""
    compaction = chain(summarizer)
    first = await agent(Main([1_150]), compaction).run('next', message_history=opening(1_100), conversation_id='c')
    await summarizer.started.wait()
    summarizer.release.set()
    second = await agent(Main([500]), compaction, model).run(
        prompt, message_history=first.all_messages(), conversation_id='c'
    )
    assert not compaction.has_background_compaction('c')
    return second.all_messages()


async def test_too_much_new_content_falls_back_to_blocking_compaction() -> None:
    summarizer = Summarizer()
    messages = await compact_after(summarizer, 'more ' * 6_000)
    assert summarizer.calls == 2, 'the held summary was dropped and the chain summarized again'
    assert summaries(messages) == ['Summary of previous conversation:\n\nsummary 2']


async def test_model_switch_falls_back_to_blocking_compaction() -> None:
    summarizer = Summarizer()
    messages = await compact_after(summarizer, OVER, model='other')
    assert summarizer.calls == 2
    assert summaries(messages) == ['Summary of previous conversation:\n\nsummary 2']


async def test_summary_failure_falls_back_to_blocking_compaction() -> None:
    summarizer = Summarizer(failures=1)
    messages = await compact_after(summarizer, OVER)
    assert summarizer.calls == 2 and summarizer.finished == 1
    assert summaries(messages) == ['Summary of previous conversation:\n\nsummary 2']


async def test_cancelled_summary_falls_back_to_blocking_compaction() -> None:
    summarizer = Summarizer()
    compaction = chain(summarizer)
    first = await agent(Main([1_150]), compaction).run('next', message_history=opening(1_100), conversation_id='c')
    await summarizer.started.wait()
    [job] = compaction._background.values()  # pyright: ignore[reportPrivateUsage]
    job.task.cancel()
    await asyncio.wait((job.task,))
    assert job.task.cancelled()
    summarizer.release.set()
    second = await agent(Main([500]), compaction).run(OVER, message_history=first.all_messages(), conversation_id='c')
    assert summarizer.calls == 2 and summarizer.finished == 1
    assert summaries(second.all_messages()) == ['Summary of previous conversation:\n\nsummary 2']


async def test_rewritten_history_drops_the_summary() -> None:
    summarizer = Summarizer()
    compaction = chain(summarizer)
    first = await agent(Main([1_150]), compaction).run('next', message_history=opening(1_100), conversation_id='c')
    await summarizer.started.wait()
    [job] = compaction._background.values()  # pyright: ignore[reportPrivateUsage]
    # A side run with a history of its own leaves it alone.
    await agent(Main([10]), compaction).run('side', conversation_id='c')
    assert compaction.has_background_compaction('c')
    # Rewinding the conversation below the background threshold cancels it.
    rewound = [first.all_messages()[0], ModelResponse(parts=[TextPart('ok')], usage=RequestUsage(input_tokens=200))]
    await agent(Main([210]), compaction).run('again', message_history=rewound, conversation_id='c')
    assert not compaction.has_background_compaction('c') and job.task.cancelled()


async def test_compact_uses_the_held_summary() -> None:
    summarizer = Summarizer()
    compaction = chain(summarizer)
    first = await agent(Main([1_150]), compaction).run('next', message_history=opening(1_100), conversation_id='c')
    summarizer.release.set()
    history = [*first.all_messages(), ModelRequest.user_text_prompt('later')]
    compacted = await compact_now(
        compaction, history, model=FunctionModel(Main([]), model_name='main'), conversation_id='c'
    )
    assert summaries(compacted) == ['Summary of previous conversation:\n\nsummary 1']
    assert compacted[-3:] == history[-3:]
    assert summarizer.calls == 1


async def test_aclose_cancels_running_work() -> None:
    summarizer = Summarizer()
    compaction = chain(summarizer)
    before = asyncio.all_tasks()
    await agent(Main([1_150]), compaction).run('next', message_history=opening(1_100), conversation_id='c')
    await summarizer.started.wait()
    [job] = compaction._background.values()  # pyright: ignore[reportPrivateUsage]
    await compaction.aclose()
    assert job.task.cancelled() and not compaction.has_background_compaction('c')
    assert asyncio.all_tasks() == before


async def test_focus_copy_holds_nothing() -> None:
    summarizer = Summarizer()
    compaction = chain(summarizer)
    await agent(Main([1_150]), compaction).run('next', message_history=opening(1_100), conversation_id='c')
    assert not compaction.with_focus('auth').has_background_compaction('c')
    await compaction.aclose()


@pytest.mark.parametrize('start', [0.0, 1.0])
def test_background_start_is_a_fraction_below_one(start: float) -> None:
    with pytest.raises(ValueError, match='background_start must be'):
        FallbackCompaction(
            fallback_chain=[SlidingWindowCompaction(max_messages=1)], max_tokens=1, background_start=start
        )


def test_background_start_needs_a_trigger() -> None:
    with pytest.raises(ValueError, match='needs a max_tokens or max_fraction trigger'):
        FallbackCompaction(fallback_chain=[SlidingWindowCompaction(max_messages=1)], background_start=0.5)
