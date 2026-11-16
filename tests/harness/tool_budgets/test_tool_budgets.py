"""Tests for `ToolBudgets`, driven through `Agent(..., capabilities=[ToolBudgets(...)])`."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import pytest
from inline_snapshot import snapshot
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from pydantic_ai import Agent, DeferredToolRequests, DeferredToolResults, ModelRetry, RunContext
from pydantic_ai.capabilities import AbstractCapability, HandleDeferredToolCalls, ValidatedToolArgs
from pydantic_ai.exceptions import ApprovalRequired, CallDeferred, SkipToolExecution, UsageLimitExceeded, UserError
from pydantic_ai.messages import (
    CompactionPart,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ModelResponsePart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.instrumented import InstrumentationSettings
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.usage import UsageLimits
from pydantic_ai_harness import SummarizingCompaction, ToolBudgets
from tests.function_model_prefix import COMPACTION_REWRITES_HISTORY

from ._search_agent import refusal


def _scripted(*responses: Sequence[ModelResponsePart]) -> FunctionModel:
    """A model that sends `responses` in order, then answers `'done'`."""
    steps = iter(responses)

    def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=list(next(steps, [TextPart('done')])))

    return FunctionModel(model)


def _search(query: str, tool_call_id: str | None = None) -> ToolCallPart:
    return ToolCallPart('search', {'query': query}, tool_call_id=tool_call_id or f'search-{query}')


def _lookup(tool_call_id: str, key: str = 'k') -> ToolCallPart:
    return ToolCallPart('lookup', {'key': key}, tool_call_id=tool_call_id)


def _poll(tool_call_id: str) -> ToolCallPart:
    return ToolCallPart('poll', {'job': 'j1'}, tool_call_id=tool_call_id)


def _agent(
    model: FunctionModel, *capabilities: AbstractCapability[None], ran: list[str] | None = None
) -> Agent[None, str]:
    agent = Agent(model, deps_type=type(None), capabilities=list(capabilities), retries=5)
    log = [] if ran is None else ran

    @agent.tool_plain
    def search(query: str, limit: int = 10) -> str:
        log.append(f'search:{query}')
        if query == 'flaky':
            raise ModelRetry('Try again')
        return f'results for {query}'

    @agent.tool_plain
    def poll(job: str) -> str:
        log.append(f'poll:{job}')
        return 'pending'

    return agent


def _returns(messages: list[ModelMessage]) -> list[tuple[str, object, object]]:
    return [
        (part.tool_call_id, part.content, part.metadata)
        for message in messages
        if isinstance(message, ModelRequest)
        for part in message.parts
        if isinstance(part, ToolReturnPart)
    ]


class TestToolBudgets:
    async def test_per_tool_budget_answers_the_model(self):
        ran: list[str] = []
        model = _scripted([_search('a')], [_search('b')], [_poll('p1')], [_search('c')])
        agent = _agent(model, ToolBudgets(per_tool={'search': 2}), ran=ran)

        result = await agent.run('Research it')

        assert result.output == 'done'
        assert ran == ['search:a', 'search:b', 'poll:j1']
        assert _returns(result.all_messages()) == snapshot(
            [
                ('search-a', 'results for a', None),
                ('search-b', 'results for b', None),
                ('p1', 'pending', None),
                (
                    'search-c',
                    '2 of 2 `search` calls used; no more are allowed in this run.',
                    {'origin': 'pydantic-ai-harness.tool_budgets.refusal.v1', 'reason': 'per_tool'},
                ),
            ]
        )

    async def test_total_budget_spans_tools(self):
        ran: list[str] = []
        model = _scripted([_search('a')], [_poll('p1')], [_poll('p2')])
        agent = _agent(model, ToolBudgets(total=2), ran=ran)

        result = await agent.run('Research it')

        assert ran == ['search:a', 'poll:j1']
        assert _returns(result.all_messages())[-1] == snapshot(
            ('p2', '2 of 2 tool calls used; no more are allowed in this run.', refusal('total'))
        )

    async def test_refuses_a_repeat_of_a_listed_tool_only(self):
        ran: list[str] = []
        model = _scripted(
            [ToolCallPart('search', {'query': 'a', 'limit': 5}, tool_call_id='s1')],
            [ToolCallPart('search', '{"limit": 5, "query": "a"}', tool_call_id='s2')],
            [ToolCallPart('search', {'query': 'a', 'limit': 6}, tool_call_id='s3')],
            [_poll('p1')],
            [_poll('p2')],
        )
        agent = _agent(model, ToolBudgets(per_tool={'search': 2}, refuse_repeats={'search'}), ran=ran)

        result = await agent.run('Research it')

        assert ran == ['search:a', 'search:a', 'poll:j1', 'poll:j1']
        assert _returns(result.all_messages())[1] == snapshot(
            (
                's2',
                '`search` already ran with these exact arguments in this run (tool call `s1`); its result, or a '
                "summary of it, is earlier in this conversation. Don't call it again with the same arguments.",
                refusal('repeat'),
            )
        )

    async def test_refused_repeat_does_not_use_the_budget(self):
        ran: list[str] = []
        model = _scripted([_search('a', 's1')], [_search('a', 's2')], [_search('b', 's3')])
        agent = _agent(model, ToolBudgets(per_tool={'search': 2}, refuse_repeats={'search'}), ran=ran)

        await agent.run('Research it')

        assert ran == ['search:a', 'search:b']

    async def test_parallel_calls_cannot_overshoot(self):
        ran: list[str] = []
        model = _scripted([_search('a'), _search('b'), _search('c'), _search('a', 'again')])
        agent = _agent(model, ToolBudgets(per_tool={'search': 2}, refuse_repeats={'search'}), ran=ran)

        result = await agent.run('Research it')

        assert sorted(ran) == ['search:a', 'search:b']
        assert {call_id: metadata for call_id, _, metadata in _returns(result.all_messages())} == {
            'search-a': None,
            'search-b': None,
            'search-c': refusal('per_tool'),
            'again': refusal('repeat'),
        }

    async def test_failed_call_uses_the_budget_but_can_be_repeated(self):
        ran: list[str] = []
        model = _scripted([_search('flaky', 's1')], [_search('flaky', 's2')], [_search('b', 's3')])
        agent = _agent(model, ToolBudgets(per_tool={'search': 2}, refuse_repeats={'search'}), ran=ran)

        result = await agent.run('Research it')

        assert ran == ['search:flaky', 'search:flaky']
        assert _returns(result.all_messages())[-1][2] == refusal('per_tool')

    async def test_call_skipped_by_another_capability_is_not_counted(self):
        class SkipFirst(AbstractCapability[None]):
            async def before_tool_execute(
                self, ctx: RunContext[None], *, call: ToolCallPart, tool_def: ToolDefinition, args: ValidatedToolArgs
            ) -> ValidatedToolArgs:
                if call.tool_call_id in ('s1', 'p1'):
                    raise SkipToolExecution('skipped')
                return args

        ran: list[str] = []
        model = _scripted([_search('a', 's1'), _poll('p1')], [_search('a', 's2'), _poll('p2')], [_search('b', 's3')])
        budgets = ToolBudgets(per_tool={'search': 1}, total=2, refuse_repeats={'search'})
        agent = _agent(model, budgets, SkipFirst(), ran=ran)

        result = await agent.run('Research it')

        assert sorted(ran) == ['poll:j1', 'search:a']
        assert [metadata for _, _, metadata in _returns(result.all_messages())] == [
            None,
            None,
            None,
            None,
            refusal('per_tool'),
        ]

    async def test_call_approved_inline_is_counted_once(self):
        class AskFirst(AbstractCapability[None]):
            async def before_tool_execute(
                self, ctx: RunContext[None], *, call: ToolCallPart, tool_def: ToolDefinition, args: ValidatedToolArgs
            ) -> ValidatedToolArgs:
                if not ctx.tool_call_approved:
                    raise ApprovalRequired
                return args

        def approve(ctx: RunContext[None], requests: DeferredToolRequests) -> DeferredToolResults:
            return requests.build_results(approve_all=True)

        ran: list[str] = []
        model = _scripted([_search('a', 's1')], [_search('b', 's2')])
        budgets = ToolBudgets(per_tool={'search': 1}, refuse_repeats={'search'})
        agent = _agent(model, budgets, AskFirst(), HandleDeferredToolCalls(approve), ran=ran)

        result = await agent.run('Research it')

        assert ran == ['search:a']
        assert [metadata for _, _, metadata in _returns(result.all_messages())] == [
            None,
            refusal('per_tool'),
        ]

    async def test_call_deferred_and_resolved_inline_is_counted(self):
        """A tool that defers may already have started its work, so the call counts and stays a repeat."""

        def resolve(ctx: RunContext[None], requests: DeferredToolRequests) -> DeferredToolResults:
            return DeferredToolResults(calls={call.tool_call_id: 'started' for call in requests.calls})

        ran: list[str] = []
        model = _scripted([_lookup('l1')], [_lookup('l2')], [_lookup('l3', 'other')], [_lookup('l4', 'third')])
        budgets = ToolBudgets(per_tool={'lookup': 2}, refuse_repeats={'lookup'})
        agent = _agent(model, budgets, HandleDeferredToolCalls(resolve), ran=ran)

        @agent.tool_plain
        def lookup(key: str) -> str:
            ran.append(f'lookup:{key}')
            raise CallDeferred

        result = await agent.run('Look it up')

        assert ran == ['lookup:k', 'lookup:other']
        assert [metadata for _, _, metadata in _returns(result.all_messages())] == [
            None,
            refusal('repeat'),
            None,
            refusal('per_tool'),
        ]

    async def test_runs_do_not_share_counts(self):
        ran: list[str] = []
        agent = _agent(
            FunctionModel(lambda messages, info: _first_search_then_done(messages)),
            ToolBudgets(per_tool={'search': 1}, refuse_repeats={'search'}),
            ran=ran,
        )

        first = await agent.run('Research it')
        await agent.run('Research it again', message_history=first.all_messages())

        assert ran == ['search:a', 'search:a']

    @pytest.mark.moves_cache_prefix(reason=COMPACTION_REWRITES_HISTORY)
    async def test_counts_survive_compaction(self):
        def summarize(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            return ModelResponse(parts=[TextPart('Searched for a and b.')])

        steps = iter([[_search('a')], [_search('b')], [_search('a', 'again')], [_search('c')]])
        seen: list[list[tuple[str, object, object]]] = []

        def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            seen.append(_returns(messages))
            return ModelResponse(parts=next(steps, [TextPart('done')]))

        ran: list[str] = []
        compaction = SummarizingCompaction(model=FunctionModel(summarize), max_messages=4, keep_messages=1)
        agent = _agent(
            FunctionModel(model), ToolBudgets(per_tool={'search': 2}, refuse_repeats={'search'}), compaction, ran=ran
        )

        result = await agent.run('Research it')

        assert any(isinstance(part, CompactionPart) for message in result.all_messages() for part in message.parts)
        assert ran == ['search:a', 'search:b']
        # The model asked for the repeat after compaction had summarized the first call away.
        assert [call_id for call_id, _, _ in seen[2]] == ['search-b']
        assert [returns[-1][2] for returns in seen[3:]] == [refusal('repeat'), refusal('per_tool')]

    async def test_usage_limits_stay_the_hard_ceiling(self):
        """A refused call is still a call the model made, so it counts toward `tool_calls_limit`."""
        model = _scripted([_search('a')], [_search('a', 'again')], [_search('b')])
        agent = _agent(model, ToolBudgets(refuse_repeats={'search'}))

        with pytest.raises(UsageLimitExceeded):
            await agent.run('Research it', usage_limits=UsageLimits(tool_calls_limit=2))

    async def test_records_a_span(self):
        exporter = InMemorySpanExporter()
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        agent = _agent(_scripted([_search('a')], [_search('b')]), ToolBudgets(total=1))
        agent.instrument = InstrumentationSettings(tracer_provider=provider)

        await agent.run('Research it')

        [span] = [span for span in exporter.get_finished_spans() if span.name == 'tool call refused']
        assert dict(span.attributes or {}) == snapshot(
            {'tool_budgets.reason': 'total', 'tool_budgets.tool_name': 'search'}
        )

    @pytest.mark.parametrize(
        ('create', 'message'),
        [
            (
                lambda: ToolBudgets(per_tool={'search': 0}),
                "`ToolBudgets.per_tool['search']` must be at least 1, got 0.",
            ),
            (lambda: ToolBudgets(total=0), '`ToolBudgets.total` must be at least 1, got 0.'),
        ],
    )
    def test_rejects_an_empty_budget(self, create: Callable[[], ToolBudgets[None]], message: str):
        with pytest.raises(UserError) as exc_info:
            create()
        assert str(exc_info.value) == message


def _first_search_then_done(messages: list[ModelMessage]) -> ModelResponse:
    if isinstance(messages[-1], ModelRequest) and any(isinstance(p, ToolReturnPart) for p in messages[-1].parts):
        return ModelResponse(parts=[TextPart('done')])
    return ModelResponse(parts=[_search('a')])
