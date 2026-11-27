"""`AbstractAgent.compact_history()`: a run that ends after its compaction step, with no output.

Not VCR tests: what's under test is how core runs the compaction step and ends the run, with `FunctionModel`s
standing in for a provider, none of which is sent a request.
"""

from __future__ import annotations

from collections.abc import AsyncIterable
from dataclasses import dataclass, field
from typing import Any

import pytest

from pydantic_ai import (
    Agent,
    AgentStreamEvent,
    CompactionEndEvent,
    CompactionPart,
    CompactionRunResult,
    CompactionStartEvent,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    RunContext,
    TextPart,
    ToolCallPart,
    UserPromptPart,
)
from pydantic_ai.agent import AgentRunResult, WrapperAgent
from pydantic_ai.capabilities import (
    AbstractCapability,
    CompactionRequest,
    CompactionResult,
    ModelFallback,
    NativeCompaction,
    WrapRunHandler,
    on_event,
)
from pydantic_ai.exceptions import ModelAPIError, UserError
from pydantic_ai.models import ModelRequestContext
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.profiles import ModelProfile
from pydantic_ai.usage import RequestUsage

from ._inline_snapshot import snapshot

COMPACTION_USAGE = RequestUsage(input_tokens=1000, output_tokens=50)


def not_sent(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    raise AssertionError('`compact_history` sends no model request')  # pragma: no cover


def history() -> list[ModelMessage]:
    return [
        ModelRequest(parts=[UserPromptPart('Our API uses PostgreSQL 17.')]),
        ModelResponse(parts=[TextPart('Noted.')]),
        ModelRequest(parts=[UserPromptPart('And pgbouncer in front of it.')]),
        ModelResponse(parts=[TextPart('Noted too.')]),
    ]


@dataclass
class Summarize(AbstractCapability[Any]):
    """Summarizes all but the last message the request ends with, keeping that tail as it was."""

    calls: list[CompactionRequest] = field(default_factory=list[CompactionRequest])

    async def compact_history(self, ctx: RunContext[Any], request: CompactionRequest) -> CompactionResult | None:
        self.calls.append(request)
        part = CompactionPart(content=f'summary, about {request.focus}')
        return CompactionResult(
            messages=[ModelRequest(parts=[part]), *request.messages[-2:]],
            part=part,
            handler='summarize',
            usage=COMPACTION_USAGE,
        )


@dataclass
class Lifecycle(AbstractCapability[Any]):
    """Records the run lifecycle hooks and the compaction events, in order."""

    calls: list[str] = field(default_factory=list[str])
    results: list[AgentRunResult[Any]] = field(default_factory=list[AgentRunResult[Any]])

    async def before_run(self, ctx: RunContext[Any]) -> None:
        self.calls.append('before_run')

    async def wrap_run(self, ctx: RunContext[Any], *, handler: WrapRunHandler) -> AgentRunResult[Any]:
        self.calls.append('wrap_run:start')
        result = await handler()
        self.calls.append('wrap_run:end')
        return result

    async def after_run(self, ctx: RunContext[Any], *, result: AgentRunResult[Any]) -> AgentRunResult[Any]:
        self.calls.append('after_run')
        self.results.append(result)
        return result

    async def before_model_request(
        self, ctx: RunContext[Any], request_context: ModelRequestContext
    ) -> ModelRequestContext:
        self.calls.append('before_model_request')
        return request_context

    async def after_model_request(
        self, ctx: RunContext[Any], *, request_context: ModelRequestContext, response: ModelResponse
    ) -> ModelResponse:
        self.calls.append('after_model_request')  # pragma: no cover
        return response  # pragma: no cover

    @on_event(CompactionStartEvent, CompactionEndEvent)
    async def _event(self, ctx: RunContext[Any], event: CompactionStartEvent | CompactionEndEvent) -> None:
        self.calls.append(type(event).__name__)


def shapes(messages: list[ModelMessage]) -> list[str]:
    return [
        f'{type(message).__name__}: {", ".join(type(part).__name__ for part in message.parts)}' for message in messages
    ]


@pytest.mark.parametrize('streamed', [False, True])
async def test_the_whole_lifecycle_runs(streamed: bool):
    """It's a real run: the lifecycle hooks and the compaction events fire, and `after_run` gets its result."""
    lifecycle = Lifecycle()
    agent = Agent(FunctionModel(not_sent), capabilities=[lifecycle, Summarize()])
    streamed_events: list[str] = []

    async def handler(ctx: RunContext[Any], events: AsyncIterable[AgentStreamEvent]) -> None:
        async for event in events:
            streamed_events.append(type(event).__name__)

    result = await agent.compact_history(history(), event_stream_handler=handler if streamed else None)

    assert lifecycle.calls == snapshot(
        [
            'wrap_run:start',
            'before_run',
            'before_model_request',
            'CompactionStartEvent',
            'CompactionEndEvent',
            'after_run',
            'wrap_run:end',
        ]
    )
    if streamed:
        assert streamed_events == ['CompactionStartEvent', 'CompactionEndEvent']
    [run_result] = lifecycle.results
    assert run_result.output is None
    assert run_result.all_messages() == result.all_messages()
    assert run_result.compactions == result.compactions


async def test_focus_and_reason_reach_the_handler():
    """The focus reaches the `compact_history` hooks exactly as with `compact=`."""
    summarize = Summarize()
    agent = Agent(FunctionModel(not_sent, model_name='primary'), capabilities=[summarize])

    await agent.compact_history(history(), focus='the infrastructure decisions', reason='cache_expiry')

    [request] = summarize.calls
    assert (request.reason, request.focus, request.conversation_request.model.model_name) == (
        'cache_expiry',
        'the infrastructure decisions',
        'primary',
    )
    # The conversation's request, which the run adds to the history to compact it with, as `compact=` does.
    assert shapes(request.messages) == snapshot(
        [
            'ModelRequest: UserPromptPart',
            'ModelResponse: TextPart',
            'ModelRequest: UserPromptPart',
            'ModelResponse: TextPart',
            'ModelRequest: ',
        ]
    )


async def test_focus_reaches_a_function_models_native_compaction():
    """With `NativeCompaction`, the focus reaches the model's `compact_messages` as its instructions."""
    compacted: list[tuple[int, str | None]] = []

    class CompactingModel(FunctionModel):
        async def compact_messages(
            self, request_context: ModelRequestContext, *, instructions: str | None = None
        ) -> ModelResponse:
            compacted.append((len(request_context.messages), instructions))
            return ModelResponse(parts=[CompactionPart(content='a summary', provider_name='function')])

    model = CompactingModel(
        not_sent, profile=ModelProfile(supports_compaction_on_demand=True, compaction_instructions='replaces_prompt')
    )
    agent = Agent(model, capabilities=[NativeCompaction()])

    result = await agent.compact_history(history(), focus='the database')

    assert len(compacted) == 1 and compacted[0][1] is not None and 'the database' in compacted[0][1]
    assert [record.handler for record in result.compactions] == ['native']
    assert shapes(result.all_messages()) == snapshot(['ModelResponse: CompactionPart'])
    assert result.new_messages() == result.all_messages()


async def test_result_has_the_compacted_messages_records_and_usage():
    agent = Agent(FunctionModel(not_sent), capabilities=[Summarize()])
    given = history()

    result = await agent.compact_history(given, focus='the database')

    assert isinstance(result, CompactionRunResult)
    # The empty request the run compacted with isn't kept: no response answers it.
    assert result.all_messages() == [
        ModelRequest(parts=[CompactionPart(content='summary, about the database')]),
        given[-1],
    ]
    assert result.new_messages() == [ModelRequest(parts=[CompactionPart(content='summary, about the database')])]
    [record] = result.compactions
    assert (record.handler, record.reason, record.messages_before, record.messages_after, record.kept_from) == (
        'summarize',
        'manual',
        4,
        2,
        1,
    )
    assert (record.part_message_index, record.part_index) == (0, 0)
    assert record.replaced == given
    assert record.summarized == given[:3]
    # The compaction is billed, but it isn't a request step.
    assert result.usage.requests == 0
    assert (result.usage.input_tokens, result.usage.output_tokens) == (1000, 50)
    assert result.run_id and result.conversation_id
    assert result.metadata is None
    assert result.all_messages_json().startswith(b'[{')
    assert result.new_messages_json().startswith(b'[{')

    # The compacted history continues the conversation.
    def answer(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart(f'answered from {shapes(messages)}')])

    continued = await agent.run('Which database?', message_history=result.all_messages(), model=FunctionModel(answer))
    assert continued.output == snapshot(
        "answered from ['ModelRequest: CompactionPart', 'ModelResponse: TextPart', 'ModelRequest: UserPromptPart']"
    )


@dataclass
class Decline(AbstractCapability[Any]):
    async def compact_history(self, ctx: RunContext[Any], request: CompactionRequest) -> CompactionResult | None:
        return None


@dataclass
class CancelStart(AbstractCapability[Any]):
    @on_event(CompactionStartEvent)
    async def _cancel(self, ctx: RunContext[Any], event: CompactionStartEvent) -> None:
        event.cancel()


@pytest.mark.parametrize('capability', [Decline(), CancelStart()], ids=['declined', 'cancelled'])
async def test_nothing_to_compact_returns_the_history_unchanged(capability: AbstractCapability[Any]):
    """When no capability compacts, the history comes back as it was given, with no compactions."""
    lifecycle = Lifecycle()
    agent = Agent(FunctionModel(not_sent), instructions='Be brief.', capabilities=[lifecycle, capability])
    given = history()

    result = await agent.compact_history(given)

    assert result.all_messages() == given
    assert result.compactions == []
    assert result.new_messages() == []
    assert result.usage.requests == 0
    assert 'after_run' in lifecycle.calls


async def test_a_history_ending_in_a_request_keeps_it():
    """A request the history ends with is the one the compaction keeps, and it stays when nothing is compacted."""
    given = [*history(), ModelRequest(parts=[UserPromptPart('Which database?')])]

    unchanged = await Agent(FunctionModel(not_sent)).compact_history(given)
    assert unchanged.all_messages()[:-1] == given[:-1]
    # Like any run continuing from the request a history ends with, it stamps that request with its IDs.
    last = unchanged.all_messages()[-1]
    assert isinstance(last, ModelRequest) and last.parts == given[-1].parts
    assert last.run_id == unchanged.run_id

    summarize = Summarize()
    result = await Agent(FunctionModel(not_sent), capabilities=[summarize]).compact_history(given)
    assert shapes(result.all_messages()) == snapshot(
        ['ModelRequest: CompactionPart', 'ModelResponse: TextPart', 'ModelRequest: UserPromptPart']
    )
    assert result.compactions[0].messages_before == 5


@pytest.mark.parametrize(
    'given, match',
    [
        ([], 'There is no message history to compact'),
        (
            [ModelRequest(parts=[UserPromptPart('hi')]), ModelResponse(parts=[ToolCallPart('lookup', {})])],
            'ends with unprocessed tool calls',
        ),
        (
            [ModelRequest(parts=[UserPromptPart('hi')]), ModelResponse(parts=[TextPart('...')], state='suspended')],
            'ends in a suspended response',
        ),
    ],
    ids=['empty', 'pending-tool-calls', 'suspended'],
)
async def test_histories_it_cant_compact_are_refused(given: list[ModelMessage], match: str):
    summarize = Summarize()
    agent = Agent(FunctionModel(not_sent), capabilities=[summarize])
    with pytest.raises(UserError, match=match):
        await agent.compact_history(given)
    assert summarize.calls == []


async def test_a_failed_compaction_falls_back_to_the_next_model():
    """A compaction that fails is the attempt's error, so a fallback model's compaction ends the run instead."""

    @dataclass
    class FailForPrimary(AbstractCapability[Any]):
        async def compact_history(self, ctx: RunContext[Any], request: CompactionRequest) -> CompactionResult | None:
            if request.conversation_request.model.model_name == 'primary':
                raise ModelAPIError(model_name='primary', message='compaction rate limited')
            return None

    summarize = Summarize()
    agent = Agent(
        FunctionModel(not_sent, model_name='primary'),
        capabilities=[FailForPrimary(), summarize, ModelFallback(FunctionModel(not_sent, model_name='fallback'))],
    )
    result = await agent.compact_history(history())
    assert [request.conversation_request.model.model_name for request in summarize.calls] == ['fallback']
    assert [record.model_name for record in result.compactions] == ['fallback']

    failing = Agent(FunctionModel(not_sent, model_name='primary'), capabilities=[FailForPrimary()])
    with pytest.raises(ModelAPIError, match='compaction rate limited'):
        await failing.compact_history(history())


async def test_a_run_started_by_a_handler_is_a_regular_run():
    """Only the `compact_history` run ends after compacting: a summarizer agent its handler runs answers as usual."""
    summarizer = Agent(FunctionModel(lambda messages, info: ModelResponse(parts=[TextPart('a summary')])))

    @dataclass
    class SummarizeWithAgent(AbstractCapability[Any]):
        async def compact_history(self, ctx: RunContext[Any], request: CompactionRequest) -> CompactionResult | None:
            summary = await summarizer.run('Summarize the conversation.')
            part = CompactionPart(content=summary.output)
            return CompactionResult(
                messages=[ModelRequest(parts=[part]), request.messages[-1]], part=part, handler='agent'
            )

    result = await Agent(FunctionModel(not_sent), capabilities=[SummarizeWithAgent()]).compact_history(history())
    assert result.all_messages() == [ModelRequest(parts=[CompactionPart(content='a summary')])]
    assert result.compactions[0].kept_from == 1


async def test_a_wrapper_agent_compacts_through_the_wrapped_agent():
    summarize = Summarize()
    agent = WrapperAgent(Agent(FunctionModel(not_sent), capabilities=[summarize]))
    result = await agent.compact_history(history(), focus='the database')
    assert [request.focus for request in summarize.calls] == ['the database']
    assert [record.handler for record in result.compactions] == ['summarize']


async def test_a_later_run_is_not_compaction_only():
    """The run after `compact_history` in the same context answers as usual."""
    agent = Agent(
        FunctionModel(lambda messages, info: ModelResponse(parts=[TextPart('answered')])), capabilities=[Summarize()]
    )
    compacted = await agent.compact_history(history())
    result = await agent.run('Which database?', message_history=compacted.all_messages())
    assert result.output == 'answered'
