"""`NativeCompaction`, requesting compaction from a run, and the generic compaction profile flags.

These run `Agent`s on `FunctionModel`s that compact with a stand-in for a provider's `compact_messages`: what's under
test is how `NativeCompaction` maps a `CompactionRequest` onto the model and back, and how a run's handles and
`compact=` request compaction. The provider calls themselves are recorded in `tests/models/test_native_compaction.py`.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import pytest

from pydantic_ai import (
    Agent,
    AgentRunResultEvent,
    CompactionEndEvent,
    CompactionPart,
    CompactionStartEvent,
    FunctionToolCallEvent,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    RunContext,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.agent import AgentRunResult, CompactionOptions, WrapperAgent
from pydantic_ai.capabilities import (
    AbstractCapability,
    CompactionRequest,
    CompactionResult,
    ModelFallback,
    NativeCompaction,
    WrapRunHandler,
    on_event,
)
from pydantic_ai.exceptions import ModelAPIError, UnexpectedModelBehavior, UserError
from pydantic_ai.models import ModelRequestContext
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, DeltaToolCalls, FunctionModel
from pydantic_ai.models.wrapper import WrapperModel
from pydantic_ai.profiles import CompactionInstructionsMode, ModelProfile
from pydantic_ai.usage import RequestUsage

from ._inline_snapshot import snapshot
from .conftest import IsDatetime, try_import

with try_import() as openai_available:
    from pydantic_ai.models.openai import OpenAICompaction

COMPACTION_USAGE = RequestUsage(input_tokens=1000, output_tokens=50)
TIMESTAMP = datetime(2026, 1, 1, tzinfo=UTC)


class CompactingModel(FunctionModel):
    """A `FunctionModel` that compacts on demand, like a provider's `compact_messages`, recording what it was asked."""

    def __init__(
        self,
        model_name: str,
        *,
        supports: bool = True,
        instructions: CompactionInstructionsMode | None = 'replaces_prompt',
        fails: bool = False,
        parts: list[TextPart] | None = None,
    ) -> None:
        super().__init__(
            self._respond,
            stream_function=self._stream,
            model_name=model_name,
            profile=ModelProfile(supports_compaction_on_demand=supports, compaction_instructions=instructions),
        )
        self.fails = fails
        self.parts = parts
        self.compacted: list[tuple[list[ModelMessage], str | None]] = []
        self.seen: list[list[ModelMessage]] = []

    def _respond(self, messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        self.seen.append(messages)
        return ModelResponse(parts=[TextPart(f'answered by {self.model_name}')])

    async def _stream(self, messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
        self.seen.append(messages)
        yield f'answered by {self.model_name}'

    async def compact_messages(
        self, request_context: ModelRequestContext, *, instructions: str | None = None
    ) -> ModelResponse:
        self.compacted.append((list(request_context.messages), instructions))
        if self.fails:
            raise ModelAPIError(model_name=self.model_name, message='compaction rate limited')
        parts = self.parts if self.parts is not None else [CompactionPart(content=f'summary by {self.model_name}')]
        return ModelResponse(parts=[*parts], usage=COMPACTION_USAGE, model_name=self.model_name, timestamp=TIMESTAMP)


@dataclass
class RequestOnce(AbstractCapability[Any]):
    """Requests compaction for the run's first request."""

    focus: str | None = None

    async def before_run(self, ctx: RunContext[Any]) -> None:
        ctx.request_compaction(reason='threshold', focus=self.focus)


@dataclass
class Summarize(AbstractCapability[Any]):
    """The next capability in the chain: compacts any request into a summary of its own."""

    calls: list[CompactionRequest] = field(default_factory=list[CompactionRequest])

    async def compact_history(self, ctx: RunContext[Any], request: CompactionRequest) -> CompactionResult | None:
        self.calls.append(request)
        part = CompactionPart(content='summarized')
        return CompactionResult(
            messages=[ModelRequest(parts=[part]), request.messages[-1]], part=part, handler='summarize'
        )


@dataclass
class RecordEvents(AbstractCapability[Any]):
    events: list[CompactionStartEvent | CompactionEndEvent] = field(
        default_factory=list[CompactionStartEvent | CompactionEndEvent]
    )

    @on_event(CompactionStartEvent, CompactionEndEvent)
    async def _record(self, ctx: RunContext[Any], event: CompactionStartEvent | CompactionEndEvent) -> None:
        self.events.append(event)


def history() -> list[ModelMessage]:
    return [
        ModelRequest(parts=[UserPromptPart('My name is Bob.', timestamp=TIMESTAMP)], timestamp=TIMESTAMP),
        ModelResponse(parts=[TextPart('Hi Bob!')], timestamp=TIMESTAMP),
    ]


async def test_native_compaction_compacts_with_the_model_and_keeps_the_turn():
    """The model compacts everything before the request being answered, which follows its compaction response."""
    model = CompactingModel('primary')
    agent = Agent(model, capabilities=[RequestOnce(), NativeCompaction()])
    result = await agent.run('What is my name?', message_history=history())

    assert model.compacted == [(history(), None)]
    assert result.all_messages() == snapshot(
        [
            ModelResponse(
                parts=[CompactionPart(content='summary by primary')],
                usage=RequestUsage(input_tokens=1000, output_tokens=50),
                model_name='primary',
                timestamp=IsDatetime(),
                metadata={'__pydantic_ai__': {'context_input_tokens': 0}},
            ),
            ModelRequest(
                parts=[UserPromptPart(content='What is my name?', timestamp=IsDatetime())],
                timestamp=IsDatetime(),
                run_id=result.run_id,
                conversation_id=result.conversation_id,
            ),
            ModelResponse(
                parts=[TextPart(content='answered by primary')],
                usage=RequestUsage(input_tokens=54, output_tokens=3),
                model_name='primary',
                model_id='function:primary',
                timestamp=IsDatetime(),
                run_id=result.run_id,
                conversation_id=result.conversation_id,
            ),
        ]
    )
    [record] = result.compactions
    assert (record.handler, record.model_name, record.messages_before, record.messages_after, record.kept_from) == (
        'native',
        'primary',
        3,
        2,
        1,
    )
    assert result.usage.input_tokens == 1054


async def test_native_compaction_keeps_a_tool_call_with_its_result():
    """A tool result can't be sent without its call, so the response that made the call is kept too."""
    model = CompactingModel('primary')
    calls = 0

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal calls
        calls += 1
        if calls == 1:
            return ModelResponse(parts=[ToolCallPart('compact_now', {}, tool_call_id='call')])
        return ModelResponse(parts=[TextPart('done')])

    model.function = respond
    agent = Agent(model, capabilities=[NativeCompaction()])

    @agent.tool
    def compact_now(ctx: RunContext[Any]) -> str:
        ctx.request_compaction()
        return 'compacting'

    result = await agent.run('Compact, please.', message_history=history())

    # Everything before the response with the call: the history and the request that asked for it.
    [(compacted, _)] = model.compacted
    assert compacted[:2] == history()
    assert [part.content for part in compacted[2].parts if isinstance(part, UserPromptPart)] == ['Compact, please.']
    messages = result.all_messages()
    assert [type(message).__name__ for message in messages] == [
        'ModelResponse',
        'ModelResponse',
        'ModelRequest',
        'ModelResponse',
    ]
    assert isinstance(messages[0].parts[0], CompactionPart)
    assert messages[1].parts == [ToolCallPart('compact_now', {}, tool_call_id='call')]
    assert isinstance(messages[2].parts[0], ToolReturnPart)
    assert result.compactions[0].kept_from == 1


async def test_native_compaction_leaves_an_unsupported_model_to_the_next_capability():
    model = CompactingModel('primary', supports=False)
    summarize = Summarize()
    agent = Agent(model, capabilities=[RequestOnce(), NativeCompaction(), summarize])
    result = await agent.run('hi', message_history=history())

    assert model.compacted == []
    assert len(summarize.calls) == 1
    assert result.compactions[0].handler == 'summarize'


async def test_native_compaction_leaves_a_fallback_model_to_the_next_capability():
    """A `FallbackModel` has no profile of its own, so which of its models would compact isn't known."""
    model = CompactingModel('primary')
    summarize = Summarize()
    agent = Agent(FallbackModel(model), capabilities=[RequestOnce(), NativeCompaction(), summarize])
    await agent.run('hi', message_history=history())

    assert model.compacted == []
    assert len(summarize.calls) == 1


async def test_native_compaction_has_nothing_to_compact_before_the_first_request():
    model = CompactingModel('primary')
    summarize = Summarize()
    agent = Agent(model, capabilities=[RequestOnce(), NativeCompaction(), summarize])
    await agent.run('hi')

    assert model.compacted == []
    assert len(summarize.calls) == 1


@pytest.mark.parametrize(
    ('instructions', 'focus_as_system_prompt', 'sent'),
    [
        pytest.param('replaces_prompt', False, 'the name', id='replaces-prompt'),
        pytest.param('system_prompt', False, None, id='system-prompt-declines'),
        pytest.param('system_prompt', True, 'the name', id='system-prompt-sent'),
        pytest.param(None, True, None, id='no-instructions-declines'),
    ],
)
async def test_native_compaction_maps_the_focus(
    instructions: CompactionInstructionsMode | None, focus_as_system_prompt: bool, sent: str | None
):
    """A focus goes to a model that wraps it in its summarization prompt; one that would lose the cache declines.

    Declining leaves the focused request to the next capability, which can keep the cache, unless
    `focus_as_system_prompt` accepts the cost.
    """
    model = CompactingModel('primary', instructions=instructions)
    summarize = Summarize()
    agent = Agent(
        model,
        capabilities=[
            RequestOnce(focus=' the name '),
            NativeCompaction(focus_as_system_prompt=focus_as_system_prompt),
            summarize,
        ],
    )
    await agent.run('hi', message_history=history())

    if sent is None:
        assert model.compacted == []
        assert summarize.calls[0].focus == ' the name '
    else:
        assert model.compacted == [(history(), sent)]
        assert summarize.calls == []


async def test_native_compaction_ignores_a_blank_focus():
    model = CompactingModel('primary', instructions='system_prompt')
    agent = Agent(model, capabilities=[RequestOnce(focus='  '), NativeCompaction()])
    await agent.run('hi', message_history=history())

    assert model.compacted == [(history(), None)]


async def test_native_compaction_compacts_for_the_fallback_candidate_not_the_step_model():
    """Under `ModelFallback`, the compaction is for the candidate serving the attempt, not `ctx.model`.

    The primary's compaction call is rate limited, which fails its attempt, so the fallback compacts for itself.
    """
    primary = CompactingModel('primary', fails=True)
    fallback = CompactingModel('fallback')
    step_models: list[str] = []

    @dataclass
    class StepModel(AbstractCapability[Any]):
        async def compact_history(self, ctx: RunContext[Any], request: CompactionRequest) -> CompactionResult | None:
            step_model = ctx.model
            assert isinstance(step_model, CompactingModel)
            step_models.append(f'{step_model.model_name} for {request.model.model_name}')
            return None

    agent = Agent(primary, capabilities=[RequestOnce(), StepModel(), NativeCompaction(), ModelFallback(fallback)])
    result = await agent.run('hi', message_history=history())

    assert step_models == ['primary for primary', 'primary for fallback']
    assert [messages for messages, _ in primary.compacted] == [history()]
    assert [messages for messages, _ in fallback.compacted] == [history()]
    assert primary.seen == []
    assert result.compactions[0].model_name == 'fallback'
    assert result.output == 'answered by fallback'


async def test_native_compaction_without_a_compaction_part():
    model = CompactingModel('primary', parts=[TextPart('not a summary')])
    agent = Agent(model, capabilities=[RequestOnce(), NativeCompaction()])
    with pytest.raises(UnexpectedModelBehavior, match="'primary' returned no compaction part"):
        await agent.run('hi', message_history=history())


def tool_then_text(model: CompactingModel) -> None:
    """Make `model` call a tool on its first request and answer on its second."""

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        model.seen.append(messages)
        if len(model.seen) == 1:
            return ModelResponse(parts=[ToolCallPart('look_up', {}, tool_call_id='call')])
        return ModelResponse(parts=[TextPart('done')])

    async def stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str | DeltaToolCalls]:
        model.seen.append(messages)
        if len(model.seen) == 1:
            yield {0: DeltaToolCall(name='look_up', json_args='{}', tool_call_id='call')}
        else:
            yield 'done'

    model.function = respond
    model.stream_function = stream


def look_up() -> str:
    return 'Bob'


async def test_run_compact_with_a_focus_reaches_the_handler():
    """`compact=` compacts before the run's first request, with the focus the options carry."""
    model = CompactingModel('primary')
    events = RecordEvents()
    agent = Agent(model, capabilities=[events, NativeCompaction()])

    result = await agent.run('What is my name?', message_history=history(), compact={'focus': 'the name'})

    assert model.compacted == [(history(), 'the name')]
    [record] = result.compactions
    assert (record.reason, record.handler, record.messages_before, record.kept_from) == ('manual', 'native', 3, 1)
    assert record.summarized == history()
    start, end = events.events
    assert isinstance(start, CompactionStartEvent) and isinstance(end, CompactionEndEvent)
    assert (start.reason, start.focus) == ('manual', 'the name')
    # The first request was answered from the compacted history.
    [first_request] = model.seen
    assert isinstance(first_request[0], ModelResponse)
    assert isinstance(first_request[0].parts[0], CompactionPart)
    assert result.output == 'answered by primary'


@pytest.mark.parametrize(
    ('compact', 'reason'),
    [
        pytest.param(True, 'manual', id='true'),
        pytest.param({}, 'manual', id='empty'),
        pytest.param({'reason': 'threshold'}, 'threshold', id='reason'),
    ],
)
async def test_run_compact_with_a_function_handler(compact: bool | CompactionOptions, reason: str):
    summarize = Summarize()
    agent = Agent(CompactingModel('primary', supports=False), capabilities=[summarize])

    result = await agent.run('hi', message_history=history(), compact=compact)

    [request] = summarize.calls
    assert (request.reason, request.focus) == (reason, None)
    assert [record.reason for record in result.compactions] == [reason]


async def test_run_without_compact_does_not_compact():
    summarize = Summarize()
    agent = Agent(CompactingModel('primary', supports=False), capabilities=[summarize])

    result = await agent.run('hi', message_history=history(), compact=False)

    assert summarize.calls == []
    assert result.compactions == []


async def test_run_compact_runs_the_full_lifecycle():
    """The compacting run is a run like any other: `after_run` gets its result, with the compaction on it."""
    calls: list[str] = []

    @dataclass
    class Lifecycle(AbstractCapability[Any]):
        async def before_run(self, ctx: RunContext[Any]) -> None:
            calls.append('before_run')

        async def wrap_run(self, ctx: RunContext[Any], *, handler: WrapRunHandler) -> AgentRunResult[Any]:
            calls.append('wrap_run')
            return await handler()

        async def after_run(self, ctx: RunContext[Any], *, result: AgentRunResult[Any]) -> AgentRunResult[Any]:
            calls.append(f'after_run with {[record.handler for record in result.compactions]}')
            return result

    agent = Agent(CompactingModel('primary'), capabilities=[Lifecycle(), NativeCompaction()])
    result = await agent.run('hi', message_history=history(), compact=True)

    assert calls == ['wrap_run', 'before_run', "after_run with ['native']"]
    assert len(result.compactions) == 1


def test_run_sync_compact():
    model = CompactingModel('primary')
    agent = Agent(model, capabilities=[NativeCompaction()])

    result = agent.run_sync('hi', message_history=history(), compact={'focus': 'the name'})

    assert model.compacted == [(history(), 'the name')]
    assert len(result.compactions) == 1


def test_run_stream_sync_compact():
    model = CompactingModel('primary')
    agent = Agent(model, capabilities=[NativeCompaction()])

    result = agent.run_stream_sync('hi', message_history=history(), compact=True)
    result.get_output()

    assert model.compacted == [(history(), None)]
    assert len(result.compactions) == 1


async def test_run_stream_compact():
    model = CompactingModel('primary')
    agent = Agent(model, capabilities=[NativeCompaction()])

    async with agent.run_stream('hi', message_history=history(), compact=True) as stream:
        assert await stream.get_output() == 'answered by primary'

    assert len(model.compacted) == 1
    assert [record.handler for record in stream.compactions] == ['native']


async def test_run_stream_events_compact():
    model = CompactingModel('primary')
    agent = Agent(model, capabilities=[NativeCompaction()])

    async with agent.run_stream_events('hi', message_history=history(), compact=True) as stream:
        events = [event async for event in stream]

    assert len(model.compacted) == 1
    assert isinstance(final := events[-1], AgentRunResultEvent)
    assert len(final.result.compactions) == 1


async def test_wrapper_agent_run_compact():
    model = CompactingModel('primary')
    agent = WrapperAgent(Agent(model, capabilities=[NativeCompaction()]))

    result = await agent.run('hi', message_history=history(), compact=True)

    assert len(model.compacted) == 1
    assert len(result.compactions) == 1


async def test_agent_run_request_compaction_mid_run():
    """`AgentRun.request_compaction()` between steps of an `iter()` loop compacts before the next request."""
    model = CompactingModel('primary')
    tool_then_text(model)
    agent = Agent(model, capabilities=[NativeCompaction()], tools=[look_up])

    async with agent.iter('What is my name?', message_history=history()) as agent_run:
        async for node in agent_run:
            if Agent.is_call_tools_node(node):
                agent_run.request_compaction(reason='threshold', focus='the name')

    # The tool call's response and its result are kept; everything before them is compacted.
    [(compacted, focus)] = model.compacted
    assert focus == 'the name'
    assert compacted[:2] == history()
    assert [part.content for part in compacted[2].parts if isinstance(part, UserPromptPart)] == ['What is my name?']
    assert agent_run.result is not None
    [record] = agent_run.result.compactions
    assert (record.reason, record.handler) == ('threshold', 'native')
    second_request = model.seen[1]
    assert isinstance(second_request[0].parts[0], CompactionPart)
    assert second_request[1].parts == [ToolCallPart('look_up', {}, tool_call_id='call')]

    with pytest.raises(UserError, match='The run has ended'):
        agent_run.request_compaction()


async def test_run_stream_events_request_compaction():
    """The `run_stream_events()` iterator requests compaction once started, like `AgentRun`."""
    model = CompactingModel('primary')
    tool_then_text(model)
    agent = Agent(model, capabilities=[NativeCompaction()], tools=[look_up])

    async with agent.run_stream_events('What is my name?', message_history=history()) as events:
        with pytest.raises(UserError):
            events.request_compaction()
        async for event in events:
            if isinstance(event, FunctionToolCallEvent):
                events.request_compaction(focus='the name')

    assert [focus for _, focus in model.compacted] == ['the name']
    assert events.result is not None
    assert len(events.result.compactions) == 1


@pytest.mark.skipif(not openai_available(), reason='openai not installed')
async def test_stateful_openai_compaction_leaves_requested_compaction_to_the_next_capability():
    """Only `OpenAICompaction`'s stateless mode compacts on request: the stateful one leaves it to OpenAI."""
    summarize = Summarize()
    agent = Agent(CompactingModel('primary'), capabilities=[RequestOnce(), OpenAICompaction(), summarize])
    result = await agent.run('hi', message_history=history())

    assert result.compactions[0].handler == 'summarize'


@pytest.mark.skipif(not openai_available(), reason='openai not installed')
async def test_stateless_openai_compaction_leaves_another_model_to_the_next_capability():
    """Stateless `OpenAICompaction` serves only an OpenAI Responses model, however it's wrapped."""
    summarize = Summarize()
    model = WrapperModel(CompactingModel('primary'))
    agent = Agent(model, capabilities=[RequestOnce(), OpenAICompaction(message_count_threshold=100), summarize])
    result = await agent.run('hi', message_history=history())

    assert result.compactions[0].handler == 'summarize'
