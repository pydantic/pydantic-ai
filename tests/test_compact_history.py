"""`AbstractAgent.compact_history()`: a run that ends after its compaction step, with no output.

Not VCR tests: what's under test is how core runs the compaction step and ends the run, with `FunctionModel`s
standing in for a provider, none of which is sent a request.
"""

from __future__ import annotations

import asyncio
import copy
from collections.abc import AsyncIterable
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass, field
from typing import Any

import pytest

from pydantic_ai import (
    Agent,
    AgentStreamEvent,
    CallToolsNode,
    CompactionEndEvent,
    CompactionPart,
    CompactionRunResult,
    CompactionStartEvent,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    RunContext,
    SystemPromptPart,
    TextPart,
    ToolCallPart,
    UserPromptPart,
)
from pydantic_ai.agent import AbstractAgent, AgentRunResult, WrapperAgent
from pydantic_ai.capabilities import (
    AbstractCapability,
    CompactionRequest,
    CompactionResult,
    Hooks,
    Instrumentation,
    ModelFallback,
    NativeCompaction,
    ProcessHistory,
    WrapModelRequestHandler,
    WrapRunHandler,
    on_event,
)
from pydantic_ai.exceptions import (
    ModelAPIError,
    ModelRetry,
    RunCancelled,
    SkipModelRequest,
    UsageLimitExceeded,
    UserError,
)
from pydantic_ai.models import ModelRequestContext
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.instrumented import InstrumentationSettings
from pydantic_ai.profiles import ModelProfile
from pydantic_ai.result import FinalResult
from pydantic_ai.usage import RequestUsage, UsageLimits
from pydantic_graph import End

from ._inline_snapshot import snapshot
from .conftest import try_import

with try_import() as logfire_imports_successful:
    from logfire.testing import CaptureLogfire

COMPACTION_USAGE = RequestUsage(input_tokens=1000, output_tokens=50)


def warns_without_a_hook(no_hook: bool) -> AbstractContextManager[object]:
    """Expect the warning a compaction request no capability can serve gets, when the agent has no `compact_history` hook."""
    if no_hook:
        return pytest.warns(UserWarning, match='Compaction was requested')
    return nullcontext()


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


@pytest.mark.parametrize(
    'capabilities',
    [[Decline()], [CancelStart(), Summarize()], []],
    ids=['declined', 'cancelled', 'no-hook'],
)
async def test_nothing_to_compact_returns_the_history_unchanged(capabilities: list[AbstractCapability[Any]]):
    """When no capability compacts, the history comes back as it was given, with no compactions.

    A hook can decline, or a listener cancel the compaction's start. An agent with no `compact_history` hook at all
    can't compact, so no compaction starts and the run warns that the request was dropped. Either way, the run
    still ends after its compaction step, and no model request is sent.
    """
    lifecycle = Lifecycle()
    agent = Agent(FunctionModel(not_sent), instructions='Be brief.', capabilities=[lifecycle, *capabilities])
    given = history()

    with warns_without_a_hook(not capabilities):
        result = await agent.compact_history(given)

    assert result.all_messages() == given
    assert result.compactions == []
    assert result.new_messages() == []
    assert result.usage.requests == 0
    assert 'after_run' in lifecycle.calls
    assert 'after_model_request' not in lifecycle.calls
    if capabilities:
        assert lifecycle.calls.count('CompactionStartEvent') == 1
    else:
        # Nothing could serve the request, so no compaction starts.
        assert 'CompactionStartEvent' not in lifecycle.calls


async def test_a_history_ending_in_a_request_keeps_it():
    """A request the history ends with is the one the compaction keeps, and it stays when nothing is compacted."""
    given = [*history(), ModelRequest(parts=[UserPromptPart('Which database?')])]

    with warns_without_a_hook(True):
        unchanged = await Agent(FunctionModel(not_sent)).compact_history(given)
    assert unchanged.compactions == []
    assert unchanged.all_messages()[:-1] == given[:-1]
    # Like any run continuing from the request a history ends with, it stamps that request with its IDs.
    last = unchanged.all_messages()[-1]
    assert isinstance(last, ModelRequest) and last.parts == given[-1].parts
    assert last.run_id == unchanged.run_id
    # It was in the given history, so it isn't new, though the run stamped it.
    assert unchanged.new_messages() == []

    summarize = Summarize()
    result = await Agent(FunctionModel(not_sent), capabilities=[summarize]).compact_history(given)
    assert shapes(result.all_messages()) == snapshot(
        ['ModelRequest: CompactionPart', 'ModelResponse: TextPart', 'ModelRequest: UserPromptPart']
    )
    assert result.compactions[0].messages_before == 5
    assert result.new_messages() == result.all_messages()[:1]


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


@pytest.mark.parametrize('compacts', [True, False], ids=['compacted', 'nothing-compacted'])
async def test_a_changed_appending_system_prompt_isnt_added_to_the_history(compacts: bool):
    """The request the run adds to compact with is dropped even when it holds a changed system prompt's statement.

    An `on_change='append'` dynamic system prompt that changed since the history's last request is stated again in
    that request. The next run on the returned history re-evaluates the prompt against it, so it still appends the
    statement, and the record counts only the messages the caller passed.
    """
    seen: list[list[ModelMessage]] = []

    def answer(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        seen.append(messages)
        return ModelResponse(parts=[TextPart('answered')])

    calls: list[CompactionRequest] = []

    class SummarizeAfterThePrompt(AbstractCapability[Any]):
        async def compact_history(self, ctx: RunContext[Any], request: CompactionRequest) -> CompactionResult | None:
            calls.append(request)
            first = request.messages[0]
            standing = [part for part in first.parts if isinstance(part, SystemPromptPart)]
            part = CompactionPart(content='summary')
            return CompactionResult(
                messages=[ModelRequest(parts=[*standing, part]), *request.messages[-2:]], part=part, handler='summarize'
            )

    agent = Agent(FunctionModel(answer), capabilities=[SummarizeAfterThePrompt()] if compacts else [])
    state = {'plan': 'one'}

    @agent.system_prompt(dynamic=True, on_change='append')
    def plan() -> str:
        return f'The plan is {state["plan"]}.'

    given = (await agent.run('Hello')).all_messages()
    state['plan'] = 'two'
    with warns_without_a_hook(not compacts):
        result = await agent.compact_history(given)

    if compacts:
        assert shapes(result.all_messages()) == snapshot(
            ['ModelRequest: SystemPromptPart, CompactionPart', 'ModelResponse: TextPart']
        )
        assert result.all_messages()[1] is given[-1]
        [record] = result.compactions
        assert (record.messages_before, record.messages_after, record.kept_from) == (len(given), 2, 1)
        assert (record.part_message_index, record.part_index) == (0, 1)
        assert record.replaced == given
        # What the handler was given still ends with the request the run compacted with, which states the change.
        [request] = calls
        assert shapes(request.messages)[-1] == 'ModelRequest: SystemPromptPart'
    else:
        assert result.all_messages() == given
        assert result.compactions == []
    assert result.new_messages() == result.all_messages()[: 1 if compacts else 0]

    await agent.run('And now?', message_history=result.all_messages())
    # The model is sent the appended statement, framed, at the start of the new request.
    sent = [str(part.content) for part in seen[-1][-1].parts if isinstance(part, UserPromptPart)]
    assert ['The plan is two.' in text for text in sent] == [True, False]


@pytest.mark.parametrize('compacts', [True, False], ids=['compacted', 'nothing-compacted'])
async def test_a_message_a_hook_enqueues_is_kept_and_the_added_request_isnt(compacts: bool):
    """A message a `before_model_request` hook enqueues is kept, as the request the compaction answers.

    It comes after the request the run added to compact with, so a compaction that keeps only the request being
    answered summarizes the added request away: the record still counts it among the messages it replaced, but not
    among the compacted ones. When nothing compacts, the added request is dropped and the enqueued one kept, so the
    next run sends it to the model.
    """

    class Remember(AbstractCapability[Any]):
        async def before_model_request(
            self, ctx: RunContext[Any], request_context: ModelRequestContext
        ) -> ModelRequestContext:
            ctx.enqueue('remember this')
            return request_context

    class SummarizeKeepingTheLast(AbstractCapability[Any]):
        async def compact_history(self, ctx: RunContext[Any], request: CompactionRequest) -> CompactionResult | None:
            part = CompactionPart(content='summary')
            return CompactionResult(messages=[ModelRequest(parts=[part]), request.messages[-1]], part=part, handler='s')

    capabilities: list[AbstractCapability[Any]] = [Remember()]
    if compacts:
        capabilities.append(SummarizeKeepingTheLast())
    agent = Agent(FunctionModel(not_sent), capabilities=capabilities)
    given = history()

    with warns_without_a_hook(not compacts):
        result = await agent.compact_history(given)

    remembered = result.all_messages()[-1]
    assert isinstance(remembered, ModelRequest)
    assert [part.content for part in remembered.parts if isinstance(part, UserPromptPart)] == ['remember this']
    if compacts:
        assert shapes(result.all_messages()) == snapshot(
            ['ModelRequest: CompactionPart', 'ModelRequest: UserPromptPart']
        )
        [record] = result.compactions
        # The compaction replaced the given history and the enqueued request; the added request isn't counted.
        assert (record.messages_before, record.messages_after, record.kept_from) == (len(given) + 1, 2, 1)
        assert (record.part_message_index, record.part_index) == (0, 0)
        assert record.replaced == [*given, remembered]
        assert record.summarized == given
        # The summary and the queued message, neither of which was in the given history.
        assert result.new_messages() == result.all_messages()
    else:
        assert result.all_messages()[:-1] == given
        assert result.compactions == []
        assert result.new_messages() == [remembered]


@pytest.mark.parametrize('result_shape', ['copied-tail', 'merged-summary'])
async def test_a_handler_that_doesnt_return_the_added_request_itself(result_shape: str):
    """A handler can return a copy of the request it's given, or fold its summary into that request.

    A copy of a kept message is committed as the original, so the added request is still dropped. A summary folded
    into the added request replaces it: the message holding the summary stays, and only the record of what the
    compaction replaced loses the added request.
    """

    class Compact(AbstractCapability[Any]):
        async def compact_history(self, ctx: RunContext[Any], request: CompactionRequest) -> CompactionResult | None:
            part = CompactionPart(content='summary')
            messages: list[ModelMessage]
            if result_shape == 'copied-tail':
                messages = [
                    ModelRequest(parts=[part]),
                    copy.deepcopy(request.messages[-2]),
                    copy.deepcopy(request.messages[-1]),
                ]
            else:
                messages = [ModelRequest(parts=[part])]
            return CompactionResult(messages=messages, part=part, handler='compact')

    given = history()
    result = await Agent(FunctionModel(not_sent), capabilities=[Compact()]).compact_history(given)

    [record] = result.compactions
    assert record.replaced == given
    assert record.messages_before == len(given)
    if result_shape == 'copied-tail':
        assert result.all_messages() == [ModelRequest(parts=[CompactionPart(content='summary')]), given[-1]]
        # The kept response is the given one, not the handler's copy.
        assert result.all_messages()[1] is given[-1]
        assert (record.messages_after, record.kept_from) == (2, 1)
    else:
        assert result.all_messages() == [ModelRequest(parts=[CompactionPart(content='summary')])]
        assert (record.messages_after, record.kept_from) == (1, 1)
    assert (record.part_message_index, record.part_index) == (0, 0)
    assert result.new_messages() == result.all_messages()[:1]


@pytest.mark.parametrize('position', ['ahead-of-the-kept-tail', 'ahead-of-the-summary'])
async def test_a_handler_that_writes_the_added_request_out_ahead_of_what_it_keeps(position: str):
    """The added request a handler writes out ahead of what it keeps is dropped, and the record's indexes move down.

    A message a hook queued comes after the added request, so it's the request being answered, which the handler keeps.
    """

    class Remember(AbstractCapability[Any]):
        async def before_model_request(
            self, ctx: RunContext[Any], request_context: ModelRequestContext
        ) -> ModelRequestContext:
            ctx.enqueue('remember this')
            return request_context

    class WriteOut(AbstractCapability[Any]):
        async def compact_history(self, ctx: RunContext[Any], request: CompactionRequest) -> CompactionResult | None:
            *_, added, remembered = request.messages
            part = CompactionPart(content='summary')
            summary = ModelRequest(parts=[part])
            prefix = [summary, added] if position == 'ahead-of-the-kept-tail' else [added, summary]
            return CompactionResult(messages=[*prefix, remembered], part=part, handler='write-out', kept_from=2)

    given = history()
    result = await Agent(FunctionModel(not_sent), capabilities=[Remember(), WriteOut()]).compact_history(given)

    assert shapes(result.all_messages()) == snapshot(['ModelRequest: CompactionPart', 'ModelRequest: UserPromptPart'])
    remembered = result.all_messages()[-1]
    [record] = result.compactions
    assert (record.messages_before, record.messages_after, record.kept_from) == (len(given) + 1, 2, 1)
    assert (record.part_message_index, record.part_index) == (0, 0)
    assert record.replaced == [*given, remembered]
    assert record.summarized == given
    assert result.new_messages() == result.all_messages()


@pytest.mark.skipif(not logfire_imports_successful(), reason='logfire not installed')
@pytest.mark.parametrize('streamed', [False, True])
async def test_an_instrumented_compact_history_run_records_no_error(capfire: CaptureLogfire, streamed: bool):
    """No span of a `compact_history()` run has an error status or an exception event: ending after compacting isn't one.

    The step's model request span is still opened, by the `wrap_model_request` chain the compaction runs in, and is
    closed as for a request that wasn't sent.
    """
    agent = Agent(
        FunctionModel(not_sent, model_name='primary'),
        capabilities=[Instrumentation(settings=InstrumentationSettings()), Summarize()],
    )

    async def handler(ctx: RunContext[Any], events: AsyncIterable[AgentStreamEvent]) -> None:
        async for _ in events:
            pass

    await agent.compact_history(history(), event_stream_handler=handler if streamed else None)

    spans = [
        span
        for span in capfire.exporter.exported_spans
        # Logfire also exports a pending span when each span starts, before its status is set.
        if (span.attributes or {}).get('logfire.span_type') != 'pending_span'
    ]
    assert [(span.name, span.status.status_code.name, [event.name for event in span.events]) for span in spans] == (
        snapshot([('chat primary', 'UNSET', []), ('invoke_agent agent', 'UNSET', [])])
    )


async def test_a_wrap_model_request_that_catches_exceptions_doesnt_continue_the_run():
    """Ending after the compaction step isn't an exception a `wrap_model_request` hook catching `Exception` can swallow.

    The same hook does catch a failed request, as a regular run of the agent shows.
    """
    caught: list[str] = []

    def fail(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        raise ModelAPIError(model_name='primary', message='unavailable')

    class Guardrail(AbstractCapability[Any]):
        async def wrap_model_request(
            self, ctx: RunContext[Any], *, request_context: ModelRequestContext, handler: WrapModelRequestHandler
        ) -> ModelResponse:
            try:
                return await handler(request_context)
            except Exception as error:
                caught.append(type(error).__name__)
                return ModelResponse(parts=[TextPart('fallback')])

    agent = Agent(FunctionModel(fail), capabilities=[Guardrail(), Summarize()])
    result = await agent.compact_history(history())

    assert caught == []
    assert shapes(result.all_messages()) == snapshot(['ModelRequest: CompactionPart', 'ModelResponse: TextPart'])
    assert [record.handler for record in result.compactions] == ['summarize']

    answered = await agent.run('Which database?', message_history=result.all_messages())
    assert (answered.output, caught) == ('fallback', ['ModelAPIError'])


async def test_an_agent_whose_run_never_starts_an_agent_run_cant_compact():
    """`compact_history()` through a `run()` that never reaches an `Agent` run fails, rather than report no compaction.

    Like an adapter that answers from somewhere else: its run doesn't compact, so its result isn't a compaction's.
    """
    answered: list[str] = []

    class Adapter(WrapperAgent[None, str]):
        async def run(self, *args: Any, **kwargs: Any) -> AgentRunResult[Any]:
            answered.append('answered elsewhere')
            return AgentRunResult(output='answered elsewhere')

    summarize = Summarize()
    adapter = Adapter(Agent(FunctionModel(not_sent), capabilities=[summarize]))

    with pytest.raises(UserError, match=r'needs the `run\(\)` of Adapter to reach an `Agent` run'):
        # The path a custom `AbstractAgent` subclass gets; `WrapperAgent` itself forwards to the agent it wraps.
        await AbstractAgent[None, str].compact_history(adapter, history())
    # Its own `run()` ran, and the wrapped agent sent no model request and didn't compact.
    assert answered == ['answered elsewhere']
    assert summarize.calls == []


@pytest.mark.parametrize('streamed', [False, True])
@pytest.mark.parametrize('answered_by', ['wrap_model_request', 'before_model_request'])
@pytest.mark.parametrize('answer', ['text', 'tool call'])
async def test_a_capability_that_answers_the_request_itself_means_nothing_is_compacted(
    streamed: bool, answered_by: str, answer: str
):
    """A step that gets a response without reaching the compaction ends the run, like a hook that declines.

    A `wrap_model_request` that returns without calling its handler, like a cache, or a `before_model_request` that
    raises `SkipModelRequest`, answers the request itself, so nothing reaches the compaction. The run still never
    continues the conversation: the response is discarded, with the request the run added, and no tool it calls runs.
    """
    ran: list[str] = []
    response = ModelResponse(
        parts=[ToolCallPart('lookup', {}, tool_call_id='call')] if answer == 'tool call' else [TextPart('cached')]
    )

    class AnswerItself(AbstractCapability[Any]):
        async def before_model_request(
            self, ctx: RunContext[Any], request_context: ModelRequestContext
        ) -> ModelRequestContext:
            if answered_by == 'before_model_request':
                raise SkipModelRequest(response)
            return request_context

        async def wrap_model_request(
            self, ctx: RunContext[Any], *, request_context: ModelRequestContext, handler: WrapModelRequestHandler
        ) -> ModelResponse:
            if answered_by == 'wrap_model_request':
                return response
            return await handler(request_context)

    summarize = Summarize()
    lifecycle = Lifecycle()
    agent = Agent(FunctionModel(not_sent), capabilities=[lifecycle, AnswerItself(), summarize])

    @agent.tool_plain
    def lookup() -> str:
        ran.append('lookup')  # pragma: no cover
        return 'found'  # pragma: no cover

    async def handler(ctx: RunContext[Any], events: AsyncIterable[AgentStreamEvent]) -> None:
        async for _ in events:
            pass

    given = history()
    result = await agent.compact_history(given, event_stream_handler=handler if streamed else None)

    assert result.all_messages() == given
    assert result.compactions == []
    assert result.new_messages() == []
    assert result.usage.requests == 0
    assert (ran, summarize.calls) == ([], [])
    [run_result] = lifecycle.results
    assert run_result.output is None


class EndsTheStepItself(AbstractCapability[Any]):
    """Ends a `compact_history()` run's step some way other than compacting, before or around the compaction."""

    def __init__(self, how: str) -> None:
        self.how = how

    async def before_run(self, ctx: RunContext[Any]) -> None:
        if self.how == 'submit_output from before_run':
            await ctx.submit_output(next(iter(ctx.output_tools)), {'response': 'submitted'})

    async def before_model_request(
        self, ctx: RunContext[Any], request_context: ModelRequestContext
    ) -> ModelRequestContext:
        if self.how == 'ModelRetry from before_model_request':
            raise ModelRetry('try again')
        if self.how == 'submit_output from before_model_request':
            await ctx.submit_output(next(iter(ctx.output_tools)), {'response': 'submitted'})
        return request_context

    async def wrap_model_request(
        self, ctx: RunContext[Any], *, request_context: ModelRequestContext, handler: WrapModelRequestHandler
    ) -> ModelResponse:
        if self.how != 'wrap_model_request swallows the end':
            return await handler(request_context)
        try:
            return await handler(request_context)
        except BaseException:
            return ModelResponse(parts=[ToolCallPart('lookup', {}, tool_call_id='call')])


@pytest.mark.parametrize('streamed', [False, True])
@pytest.mark.parametrize(
    'how',
    [
        'ModelRetry from before_model_request',
        'submit_output from before_model_request',
        'submit_output from before_run',
        'wrap_model_request swallows the end',
    ],
)
async def test_a_step_that_ends_another_way_ends_the_run_with_the_history_as_given(streamed: bool, how: str):
    """However a `compact_history()` run's step ends, the run ends there and never continues the conversation.

    A retry would make another step with its feedback in the history, a submitted output would end the run with an
    output, and a hook that swallows the end of the step continues with its own response. Each ends the run as a
    hook that declines does: with the history as given, no compactions, and no tool run, unless the compaction ran
    first, as it does before the step ends around it.
    """
    ran: list[str] = []
    summarize = Summarize()
    lifecycle = Lifecycle()
    agent = Agent(FunctionModel(not_sent), capabilities=[lifecycle, EndsTheStepItself(how), summarize])

    @agent.tool_plain
    def lookup() -> str:
        ran.append('lookup')  # pragma: no cover
        return 'found'  # pragma: no cover

    async def handler(ctx: RunContext[Any], events: AsyncIterable[AgentStreamEvent]) -> None:
        async for _ in events:
            pass

    given = history()
    result = await agent.compact_history(given, event_stream_handler=handler if streamed else None)

    if how == 'wrap_model_request swallows the end':
        # The compaction ran and was kept before the hook swallowed the end of the step, so the run ends compacted.
        assert shapes(result.all_messages()) == snapshot(['ModelRequest: CompactionPart', 'ModelResponse: TextPart'])
        assert [record.handler for record in result.compactions] == ['summarize']
    else:
        assert result.all_messages() == given
        assert result.compactions == []
        assert summarize.calls == []
    assert result.usage.requests == 0
    assert ran == []
    assert 'after_model_request' not in lifecycle.calls
    [run_result] = lifecycle.results
    assert run_result.output is None


async def test_an_error_hook_that_recovers_from_a_failed_compaction_ends_the_run_with_the_history_as_given():
    """A response an `on_model_request_error` hook recovers a failed compaction with isn't acted on."""

    class FailToCompact(AbstractCapability[Any]):
        async def compact_history(self, ctx: RunContext[Any], request: CompactionRequest) -> CompactionResult | None:
            raise ModelAPIError(model_name='primary', message='compaction rate limited')

    async def recover(ctx: RunContext[Any], *, request_context: ModelRequestContext, error: Exception) -> ModelResponse:
        return ModelResponse(parts=[TextPart('recovered')])

    agent = Agent(FunctionModel(not_sent), capabilities=[FailToCompact(), Hooks(model_request_error=recover)])
    given = history()
    result = await agent.compact_history(given)

    assert result.all_messages() == given
    assert result.compactions == []
    assert result.usage.requests == 0


@pytest.mark.parametrize('replacement', ['tool execution', 'end'])
async def test_a_node_hook_that_replaces_the_model_request_step(replacement: str):
    """A node hook that replaces the run's model request step can't have it run tools; one that ends it, ends it."""
    ran: list[str] = []

    class Replace(AbstractCapability[Any]):
        async def before_node_run(self, ctx: RunContext[Any], *, node: Any) -> Any:
            if replacement == 'tool execution' and Agent.is_model_request_node(node):
                return CallToolsNode[Any, Any](ModelResponse(parts=[ToolCallPart('lookup', {}, tool_call_id='call')]))
            return node

        async def wrap_node_run(self, ctx: RunContext[Any], *, node: Any, handler: Any) -> Any:
            if replacement == 'end' and Agent.is_model_request_node(node):
                return End(FinalResult('ended'))
            return await handler(node)

    summarize = Summarize()
    agent = Agent(FunctionModel(not_sent), capabilities=[Replace(), summarize])

    @agent.tool_plain
    def lookup() -> str:
        ran.append('lookup')  # pragma: no cover
        return 'found'  # pragma: no cover

    given = history()
    if replacement == 'end':
        result = await agent.compact_history(given)
        assert result.all_messages() == given
        assert result.compactions == []
    else:
        with pytest.raises(UserError, match='reached tool execution'):
            await agent.compact_history(given)
    assert (ran, summarize.calls) == ([], [])


@pytest.mark.parametrize('compacts', [True, False], ids=['compacted', 'nothing-compacted'])
async def test_history_rewrites_by_hooks(compacts: bool):
    """What a hook changes only for the request isn't kept; what a history processor changes is, as in any run.

    `ProcessHistory` makes its result the run's history, so a `compact_history()` run returns the processed history
    when nothing compacts, like the message a hook queues.
    """

    async def note(ctx: RunContext[Any], request_context: ModelRequestContext) -> ModelRequestContext:
        request_context.messages = [*request_context.messages[:-1], ModelRequest(parts=[UserPromptPart('only now')])]
        return request_context

    capabilities: list[AbstractCapability[Any]] = [
        ProcessHistory(lambda messages: messages[2:]),
        Hooks(before_model_request=note),
    ]
    if compacts:
        capabilities.append(Summarize())
    agent = Agent(FunctionModel(not_sent), capabilities=capabilities)
    given = history()

    with warns_without_a_hook(not compacts):
        result = await agent.compact_history(given)

    if compacts:
        assert shapes(result.all_messages()) == snapshot(['ModelRequest: CompactionPart', 'ModelResponse: TextPart'])
        [record] = result.compactions
        # The processed history is what the compaction replaced: the request-only note isn't in it.
        assert record.replaced == given[2:]
    else:
        assert result.all_messages() == given[2:]
        assert result.compactions == []


async def test_a_usage_limit_the_compaction_crosses_fails_the_run():
    agent = Agent(FunctionModel(not_sent), capabilities=[Summarize()])
    with pytest.raises(UsageLimitExceeded):
        await agent.compact_history(history(), usage_limits=UsageLimits(input_tokens_limit=10))


async def test_cancelling_the_run_from_the_compaction_cancels_it():
    """A compaction that cancels the run ends it cancelled, rather than with the compaction."""

    class CancelWhileCompacting(AbstractCapability[Any]):
        async def compact_history(self, ctx: RunContext[Any], request: CompactionRequest) -> CompactionResult | None:
            ctx.cancel()
            await asyncio.sleep(0)
            return None  # pragma: no cover

    agent = Agent(FunctionModel(not_sent), capabilities=[CancelWhileCompacting()])
    with pytest.raises(RunCancelled):
        await agent.compact_history(history())
