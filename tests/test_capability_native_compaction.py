"""`NativeCompaction`, `Agent.compact()` and the generic compaction profile flags.

These run `Agent`s on `FunctionModel`s that compact with a stand-in for a provider's `compact_messages`: what's under
test is how `NativeCompaction` maps a `CompactionRequest` onto the model and back, and how `Agent.compact()` runs the
hooks outside a run. The provider calls themselves are recorded in `tests/models/test_native_compaction.py`.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from typing import Any

import pytest

from pydantic_ai import (
    Agent,
    CompactionEndEvent,
    CompactionPart,
    CompactionStartEvent,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    RunContext,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.agent import AgentRunResult, WrapperAgent
from pydantic_ai.capabilities import (
    AbstractCapability,
    CompactionRequest,
    CompactionResult,
    ModelFallback,
    NativeCompaction,
    WrapModelRequestHandler,
    WrapRunHandler,
    on_event,
)
from pydantic_ai.exceptions import ModelAPIError, RetryModelRequest, UnexpectedModelBehavior, UserError
from pydantic_ai.models import ModelRequestContext
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.models.function import AgentInfo, FunctionModel
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
    cancel: bool = False
    events: list[CompactionStartEvent | CompactionEndEvent] = field(
        default_factory=list[CompactionStartEvent | CompactionEndEvent]
    )

    @on_event(CompactionStartEvent, CompactionEndEvent)
    async def _record(self, ctx: RunContext[Any], event: CompactionStartEvent | CompactionEndEvent) -> None:
        if self.cancel and isinstance(event, CompactionStartEvent):
            event.cancel()
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


async def test_agent_compact_runs_the_hooks_and_events_outside_a_run():
    model = CompactingModel('primary')
    events = RecordEvents()
    agent = Agent(model, instructions='Be brief.', capabilities=[events, NativeCompaction()])

    result = await agent.compact(history(), focus='the name')

    assert result is not None
    assert result.messages == snapshot(
        [
            ModelResponse(
                parts=[CompactionPart(content='summary by primary')],
                usage=RequestUsage(input_tokens=1000, output_tokens=50),
                model_name='primary',
                timestamp=IsDatetime(),
                metadata={'__pydantic_ai__': {'context_input_tokens': 0}},
            )
        ]
    )
    assert (result.handler, result.kept_from, result.usage) == ('native', 1, COMPACTION_USAGE)
    assert result.record is not None
    assert result.record.replaced == history()
    assert result.record.summarized == history()
    assert (
        result.record.reason,
        result.record.model_name,
        result.record.messages_before,
        result.record.messages_after,
        result.record.kept_from,
    ) == ('manual', 'primary', 2, 1, 1)
    assert model.compacted == [(history(), 'the name')]
    # Nothing but the compaction was sent.
    assert model.seen == []
    assert [type(event).__name__ for event in events.events] == ['CompactionStartEvent', 'CompactionEndEvent']
    start, end = events.events
    assert isinstance(start, CompactionStartEvent) and isinstance(end, CompactionEndEvent)
    assert (start.reason, start.focus, start.model_name, start.messages_before) == ('manual', 'the name', 'primary', 3)
    assert (end.handler, end.usage) == ('native', COMPACTION_USAGE)

    # The conversation continues from the compacted history.
    continued = await Agent(model).run('What is my name?', message_history=result.messages)
    assert isinstance(model.seen[0][0], ModelResponse)
    assert continued.output == 'answered by primary'


async def test_agent_compact_through_a_wrapper_agent():
    """`compact()` is on every agent: a `WrapperAgent` compacts through the agent it wraps."""
    model = CompactingModel('primary')
    agent = WrapperAgent(Agent(model, capabilities=[NativeCompaction()]))

    result = await agent.compact(history())

    assert result is not None
    assert model.compacted == [(history(), None)]


async def test_agent_compact_run_lifecycle():
    """`before_run` and `wrap_run` fire, and the run ends without a result: `after_run` doesn't fire."""
    calls: list[str] = []

    @dataclass
    class Lifecycle(AbstractCapability[Any]):
        async def before_run(self, ctx: RunContext[Any]) -> None:
            calls.append('before_run')

        async def wrap_run(self, ctx: RunContext[Any], *, handler: WrapRunHandler) -> AgentRunResult[Any]:
            calls.append('wrap_run')
            try:
                return await handler()
            except asyncio.CancelledError:
                calls.append('wrap_run cancelled')
                raise

        async def after_run(self, ctx: RunContext[Any], *, result: AgentRunResult[Any]) -> AgentRunResult[Any]:
            calls.append('after_run')  # pragma: no cover
            return result  # pragma: no cover

    agent = Agent(CompactingModel('primary'), capabilities=[Lifecycle(), NativeCompaction()])
    assert await agent.compact(history()) is not None
    assert calls == ['wrap_run', 'before_run', 'wrap_run cancelled']


async def test_agent_compact_keeps_a_trailing_request():
    """A history that ends with a request compacts for answering it, and keeps it."""
    model = CompactingModel('primary')
    agent = Agent(model, capabilities=[NativeCompaction()])
    messages = [*history(), ModelRequest(parts=[UserPromptPart('What is my name?')])]

    result = await agent.compact(messages, reason='threshold')

    assert result is not None
    assert model.compacted == [(history(), None)]
    assert len(result.messages) == 2
    assert result.messages[1].parts == messages[2].parts
    assert result.record is not None
    assert (result.record.reason, result.record.messages_before, result.record.kept_from) == ('threshold', 3, 1)
    assert result.record.summarized == history()


async def test_agent_compact_keeps_the_given_messages_as_they_were():
    """Like a run, it compacts the history as the request shows it, and returns the given originals of what it kept.

    A `prepare_model_request` hook's substitution is what's compacted, and its insertion only for this request isn't
    compacted or returned.
    """
    model = CompactingModel('primary')

    @dataclass
    class RequestOnly(AbstractCapability[Any]):
        async def prepare_model_request(
            self, ctx: RunContext[Any], request_context: ModelRequestContext
        ) -> ModelRequestContext:
            first, *middle, last = request_context.messages
            substituted = replace(first, parts=[UserPromptPart('My name is [redacted].', timestamp=TIMESTAMP)])
            reminder = ModelRequest(parts=[UserPromptPart('A reminder for this request only.')])
            request_context.messages = [substituted, *middle, reminder, last]
            return request_context

    agent = Agent(model, capabilities=[RequestOnly(), NativeCompaction()])
    question = ModelRequest(parts=[UserPromptPart('What is my name?', timestamp=TIMESTAMP)], timestamp=TIMESTAMP)

    result = await agent.compact([*history(), question])

    assert result is not None
    [(compacted, _)] = model.compacted
    assert [part.content for part in compacted[0].parts if isinstance(part, UserPromptPart)] == [
        'My name is [redacted].'
    ]
    assert compacted[1:] == history()[1:]
    assert len(result.messages) == 2
    assert result.messages[-1].parts == question.parts


async def test_agent_compact_with_a_capability_for_the_call():
    model = CompactingModel('primary')
    agent = Agent(model)

    assert await agent.compact(history()) is None
    result = await agent.compact(history(), capabilities=[NativeCompaction()])
    assert result is not None
    assert result.handler == 'native'


async def test_agent_compact_returns_none_when_nothing_compacts():
    """Neither a model that can't compact nor a cancelled compaction leaves a result, and nothing is sent."""
    unsupported = CompactingModel('primary', supports=False)
    agent = Agent(unsupported, capabilities=[NativeCompaction()])
    assert await agent.compact(history()) is None

    model = CompactingModel('primary')
    events = RecordEvents(cancel=True)
    agent = Agent(model, capabilities=[events, NativeCompaction()])
    assert await agent.compact(history()) is None
    assert model.compacted == []
    assert model.seen == unsupported.seen == []


async def test_agent_compact_through_wrap_model_request():
    """`wrap_model_request` wraps the compaction like a model request, and sees its response."""
    model = CompactingModel('primary')
    wrapped: list[ModelResponse] = []

    @dataclass
    class Wrap(AbstractCapability[Any]):
        async def wrap_model_request(
            self, ctx: RunContext[Any], *, request_context: ModelRequestContext, handler: WrapModelRequestHandler
        ) -> ModelResponse:
            response = await handler(request_context)
            wrapped.append(response)
            return response

    agent = Agent(model, capabilities=[Wrap(), NativeCompaction()])
    result = await agent.compact(history())
    assert result is not None
    assert [response.parts for response in wrapped] == [[result.part]]

    unsupported = Agent(CompactingModel('primary', supports=False), capabilities=[Wrap(), NativeCompaction()])
    assert await unsupported.compact(history()) is None
    assert len(wrapped) == 1


async def test_agent_compact_moves_on_to_the_model_a_hook_asks_for():
    """A `prepare_model_request` hook's `RetryModelRequest`, like `ModelFallback`'s sticky one, picks the model."""
    primary = CompactingModel('primary')
    other = CompactingModel('other')

    @dataclass
    class Redirect(AbstractCapability[Any]):
        async def prepare_model_request(
            self, ctx: RunContext[Any], request_context: ModelRequestContext
        ) -> ModelRequestContext:
            if request_context.model is primary:
                raise RetryModelRequest(model=other)
            return request_context

    agent = Agent(primary, capabilities=[Redirect(), NativeCompaction()])
    result = await agent.compact(history())

    assert result is not None
    assert result.record is not None
    assert result.record.model_name == 'other'
    assert primary.compacted == []
    assert len(other.compacted) == 1


async def test_agent_compact_raises_a_compaction_error():
    agent = Agent(CompactingModel('primary', fails=True), capabilities=[NativeCompaction()])
    with pytest.raises(ModelAPIError, match='compaction rate limited'):
        await agent.compact(history())


@pytest.mark.parametrize(
    'messages',
    [
        pytest.param([], id='empty'),
        pytest.param(
            [ModelRequest(parts=[UserPromptPart('hi')]), ModelResponse(parts=[ToolCallPart('tool', {})])],
            id='tool-calls',
        ),
        pytest.param(
            [ModelRequest(parts=[UserPromptPart('hi')]), ModelResponse(parts=[TextPart('hm')], state='suspended')],
            id='suspended',
        ),
    ],
)
async def test_agent_compact_rejects_a_history_it_cannot_compact(messages: list[ModelMessage]):
    agent = Agent(CompactingModel('primary'), capabilities=[NativeCompaction()])
    with pytest.raises(UserError):
        await agent.compact(messages)


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
