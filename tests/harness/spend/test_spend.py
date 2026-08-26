"""Tests for the `spend` capability."""

from __future__ import annotations

import json
import warnings
from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation, localcontext
from typing import Any

import pytest
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import NoOpTracer, Tracer
from pydantic_ai import Agent
from pydantic_ai.agent.spec import AgentSpec
from pydantic_ai.capabilities import (
    AbstractCapability,
    CapabilityOrdering,
    CombinedCapability,
    Hooks,
    WrapModelRequestHandler,
    WrapperCapability,
)
from pydantic_ai.exceptions import ModelRetry, SkipModelRequest, UsageLimitExceeded, UserError
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models import ModelRequestContext, ModelRequestParameters
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import RunContext
from pydantic_ai.usage import RequestUsage, RunUsage

from pydantic_ai_harness import HarnessDeprecationWarning
from pydantic_ai_harness.guardrails import GuardrailResult, InputGuardrail
from pydantic_ai_harness.spend import (
    Budget,
    InMemorySpendStore,
    RedisSpendStore,
    SpendCompositionWarning,
    SpendEntry,
    SpendLimitExceeded,
    SpendLimits,
    SpendSnapshot,
    Spent,
    UnpricedModelError,
    UnpricedModelWarning,
)

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return 'asyncio'


_EPOCH = datetime(2026, 7, 26, 12, 0, tzinfo=timezone.utc)


class Clock:
    """A clock the tests move by hand, so windows and expiry are deterministic."""

    def __init__(self, now: datetime = _EPOCH) -> None:
        self.now = now

    def __call__(self) -> datetime:
        return self.now

    def advance(self, delta: timedelta) -> None:
        self.now += delta


def _run_ctx(
    *,
    run_id: str | None = 'run-1',
    conversation_id: str | None = 'conv-1',
    deps: Any = None,
    trace_include_content: bool = False,
    tracer: Tracer | None = None,
    root_capability: AbstractCapability[None] | None = None,
) -> RunContext[Any]:
    return RunContext(
        deps=deps,
        model=TestModel(),
        usage=RunUsage(),
        run_id=run_id,
        conversation_id=conversation_id,
        trace_include_content=trace_include_content,
        tracer=tracer if tracer is not None else NoOpTracer(),
        root_capability=root_capability,
    )


def _request_context() -> ModelRequestContext:
    return ModelRequestContext(
        model=TestModel(),
        messages=[],
        model_settings=None,
        model_request_parameters=ModelRequestParameters(),
    )


def _response(
    *,
    input_tokens: int = 1000,
    output_tokens: int = 100,
    model_name: str | None = 'gpt-4.1',
    provider_name: str | None = 'openai',
    provider_response_id: str | None = None,
) -> ModelResponse:
    return ModelResponse(
        parts=[TextPart(content='ok')],
        usage=RequestUsage(input_tokens=input_tokens, output_tokens=output_tokens),
        model_name=model_name,
        provider_name=provider_name,
        provider_response_id=provider_response_id,
    )


async def _record(
    guard: SpendLimits[Any],
    *,
    ctx: RunContext[Any] | None = None,
    response: ModelResponse | None = None,
    **kwargs: Any,
) -> ModelResponse:
    """Drive one accrual by standing in for the provider call the capability wraps."""
    recorded = response if response is not None else _response(**kwargs)

    async def handler(request_context: ModelRequestContext) -> ModelResponse:
        return recorded

    return await guard.wrap_model_request(
        ctx if ctx is not None else _run_ctx(),
        request_context=_request_context(),
        handler=handler,
    )


def _spec_budgets(*entries: Any) -> list[Any]:
    """Budget entries as a spec really delivers them: parsed YAML, unvalidated.

    `BudgetSpec` types the schema `from_spec` publishes to an editor; nothing enforces
    it at the boundary, which is the whole of what these tests cover.
    """
    return list(entries)


async def _gate(guard: SpendLimits[Any], *, ctx: RunContext[Any] | None = None) -> None:
    await guard.before_model_request(ctx if ctx is not None else _run_ctx(), _request_context())


def _recording_tracer() -> tuple[Tracer, InMemorySpanExporter]:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return provider.get_tracer('test'), exporter


def _only_span(exporter: InMemorySpanExporter) -> ReadableSpan:
    spans = exporter.get_finished_spans()
    assert len(spans) == 1, f'expected one span, got {[s.name for s in spans]}'
    return spans[0]


def _scripted_usage() -> FunctionModel:
    """A model whose every response carries the same usage, for counting."""

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart(content='ok')], usage=RequestUsage(input_tokens=1000, output_tokens=100))

    return FunctionModel(respond)


def _agent(guard: SpendLimits[None], *, usage: RequestUsage | None = None) -> Agent[None, str]:
    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(
            parts=[TextPart(content='ok')],
            usage=usage if usage is not None else RequestUsage(input_tokens=1000, output_tokens=100),
        )

    return Agent(FunctionModel(respond), deps_type=type(None), capabilities=[guard])


def _no_price(response: ModelResponse) -> Decimal | None:
    """A price override that declines, so the registry is consulted instead."""
    return None


async def _call_get_spend(guard: SpendLimits[None]) -> str:
    """Drive `get_spend` the way the model would, and return what it reported."""
    calls = iter(
        [
            ModelResponse(
                parts=[ToolCallPart(tool_name='get_spend', args={}, tool_call_id='c1')],
                usage=RequestUsage(input_tokens=1000, output_tokens=100),
            ),
            ModelResponse(parts=[TextPart(content='done')]),
        ]
    )
    agent = Agent(FunctionModel(lambda messages, info: next(calls)), deps_type=type(None), capabilities=[guard])
    result = await agent.run('hi')
    returns = [p for m in result.all_messages() for p in m.parts if isinstance(p, ToolReturnPart)]
    assert len(returns) == 1
    return str(returns[0].content)


class TestBudgetConfiguration:
    """`Budget` rejects configurations whose store key would be ambiguous."""

    def test_name_must_not_be_empty(self):
        with pytest.raises(UserError, match='must be non-empty'):
            Budget(name='')

    def test_name_must_not_contain_the_separator(self):
        with pytest.raises(UserError, match='must not contain'):
            Budget(name='a|b')

    @pytest.mark.parametrize('warn_at', [0.0, 1.5, -0.2])
    def test_warn_at_must_be_a_fraction(self, warn_at: float):
        with pytest.raises(UserError, match='fraction'):
            Budget(usd=Decimal('1'), warn_at=warn_at)

    def test_warn_at_without_a_ceiling_is_refused(self):
        """A fraction of nothing can never fire, so it is a mistake, not a setting."""
        with pytest.raises(UserError, match='could never fire'):
            Budget(window='total', warn_at=0.8)

    @pytest.mark.parametrize('amount', ['NaN', 'Infinity', '-Infinity'])
    def test_a_non_finite_usd_ceiling_is_refused(self, amount: str):
        """`NaN <= 0` raises `InvalidOperation`, and an infinity reads as a ceiling nothing reaches."""
        with pytest.raises(UserError, match='must be a finite amount'):
            Budget(usd=Decimal(amount))

    def test_retain_defaults_to_the_window_horizon(self):
        assert Budget(window='conversation').ttl == timedelta(days=30)

    def test_retain_forever_never_expires(self):
        """A conversation resumed past the default horizon would otherwise start again from zero."""
        assert Budget(window='conversation', retain='forever').ttl is None

    def test_retain_may_name_its_own_horizon(self):
        assert Budget(window='run', retain=timedelta(hours=2)).ttl == timedelta(hours=2)

    def test_a_misspelt_retain_policy_is_refused(self):
        """The annotation is a `Literal`, which nothing enforces at run time."""
        with pytest.raises(UserError, match='must be a timedelta or one of'):
            Budget(retain='forevr')  # pyright: ignore[reportArgumentType]

    @pytest.mark.parametrize('retain', [timedelta(0), timedelta(seconds=-1)])
    def test_a_non_positive_retain_is_refused(self, retain: timedelta):
        with pytest.raises(UserError, match='must be a positive duration'):
            Budget(retain=retain)

    @pytest.mark.parametrize(
        ('kwargs', 'match'),
        [
            ({'usd': Decimal('-5')}, 'usd must be positive'),
            ({'tokens': -100}, 'tokens must be positive'),
            ({'usd': Decimal('0')}, 'usd must be positive'),
            ({'tokens': 0}, 'tokens must be positive'),
        ],
    )
    def test_a_non_positive_ceiling_is_refused(self, kwargs: Any, match: str):
        """Zero has the negative case's property, and in a spec it far more likely means "no limit"."""
        with pytest.raises(UserError, match=match):
            Budget(**kwargs)

    def test_a_budget_without_limits_only_counts(self):
        assert Budget().enforces is False
        assert Budget(usd=Decimal('1')).enforces is True
        assert Budget(tokens=1).enforces is True

    def test_only_a_lifetime_window_is_kept_forever(self):
        assert Budget(window='total').ttl is None
        assert Budget(window='day').ttl == timedelta(hours=48)

    def test_a_conversation_counter_expires_on_a_long_horizon(self):
        """Its bucket never rolls over, so it needs a horizon; unbounded would grow the store forever."""
        assert Budget(window='conversation').ttl == timedelta(days=30)


class TestWindows:
    """A window rolls over by producing a different key, not by resetting a counter."""

    async def test_day_rolls_over_to_a_fresh_counter(self):
        clock = Clock()
        guard = SpendLimits(budgets=[Budget(window='day')], clock=clock, price=lambda r: Decimal('1'))

        await _record(guard)
        assert (await guard.status())[0].spent.usd == Decimal('1')

        clock.advance(timedelta(days=1))
        assert (await guard.status())[0].spent.usd == Decimal('0')

    async def test_month_bucket_is_year_and_month(self):
        clock = Clock()
        guard = SpendLimits(budgets=[Budget(window='month')], clock=clock)
        await _record(guard)

        assert (await guard.status())[0].key.endswith('|2026-07')

    async def test_total_never_rolls_over(self):
        clock = Clock()
        guard = SpendLimits(budgets=[Budget(window='total')], clock=clock, price=lambda r: Decimal('1'))
        await _record(guard)
        clock.advance(timedelta(days=400))

        assert (await guard.status())[0].spent.usd == Decimal('1')

    async def test_run_window_keys_on_the_run_id(self):
        guard = SpendLimits(budgets=[Budget(window='run')], price=lambda r: Decimal('1'))
        await _record(guard, ctx=_run_ctx(run_id='a'))
        await _record(guard, ctx=_run_ctx(run_id='b'))

        statuses = await guard.status(_run_ctx(run_id='a'))
        assert statuses[0].spent.usd == Decimal('1')

    async def test_conversation_window_keys_on_the_conversation_id(self):
        guard = SpendLimits(budgets=[Budget(window='conversation')], price=lambda r: Decimal('1'))
        await _record(guard, ctx=_run_ctx(conversation_id='c1'))
        await _record(guard, ctx=_run_ctx(conversation_id='c1'))

        assert (await guard.status(_run_ctx(conversation_id='c1')))[0].spent.requests == 2

    @pytest.mark.parametrize(
        ('window', 'ctx_kwargs'),
        [('run', {'run_id': None}), ('conversation', {'conversation_id': None})],
    )
    async def test_a_missing_identity_is_refused_rather_than_shared(self, window: Any, ctx_kwargs: Any):
        guard = SpendLimits(budgets=[Budget(window=window)])

        with pytest.raises(UserError, match='reports none'):
            await _record(guard, ctx=_run_ctx(**ctx_kwargs))


class TestKeyCollisions:
    """Two budgets share a counter only when they mean to."""

    async def test_budgets_sharing_a_window_share_one_counter(self):
        """A USD and a token ceiling on the same window are two limits on one counter."""
        guard = SpendLimits(
            budgets=[Budget(usd=Decimal('10'), window='day'), Budget(tokens=99_999, window='day')],
            price=lambda r: Decimal('1'),
        )
        await _record(guard)

        usd_budget, token_budget = await guard.status()
        assert usd_budget.key == token_budget.key
        assert usd_budget.spent.usd == Decimal('1')
        assert usd_budget.spent.requests == 1

    async def test_a_shared_counter_is_read_once_per_request(self):
        """Two ceilings on one window are one round trip, which matters against a network store."""
        reads: list[Sequence[str]] = []

        class Counting(InMemorySpendStore):
            async def get_many(self, keys: Sequence[str]) -> Mapping[str, Spent]:
                reads.append(list(keys))
                return await super().get_many(keys)

        guard = SpendLimits(
            budgets=[Budget(usd=Decimal('10'), window='day'), Budget(tokens=99_999, window='day')],
            store=Counting(),
        )
        await _gate(guard)

        assert [len(keys) for keys in reads] == [1]

    async def test_a_run_id_cannot_impersonate_another_window(self):
        """Bucket values are not drawn from disjoint sets, so the window is part of the key."""
        guard = SpendLimits(budgets=[Budget(window='run'), Budget(window='total')])
        ctx = _run_ctx(run_id='total')

        run_budget, total_budget = await guard.status(ctx)
        assert run_budget.key != total_budget.key

    async def test_a_run_and_a_conversation_with_one_id_stay_apart(self):
        guard = SpendLimits(budgets=[Budget(window='run'), Budget(window='conversation')])
        ctx = _run_ctx(run_id='same', conversation_id='same')

        run_budget, conversation_budget = await guard.status(ctx)
        assert run_budget.key != conversation_budget.key


class TestScope:
    """`scope` partitions one counter; it does not select a store."""

    async def test_a_scope_that_returns_a_non_string_is_refused(self):
        """A tenant id is often an int or a UUID, and the annotation alone does not stop one."""
        guard = SpendLimits(
            budgets=[Budget(usd=Decimal('1'), scope=lambda ctx: ctx.deps)],  # pyright: ignore[reportArgumentType, reportUnknownLambdaType, reportUnknownArgumentType, reportUnknownMemberType]
            price=lambda r: Decimal('0.4'),
        )

        with pytest.raises(UserError, match='must return a string; got int'):
            await _record(guard, ctx=_run_ctx(deps=7))

    async def test_tenants_count_separately(self):
        guard = SpendLimits(
            budgets=[Budget(usd=Decimal('1'), scope=lambda ctx: str(ctx.deps))],
            price=lambda r: Decimal('0.4'),
        )
        await _record(guard, ctx=_run_ctx(deps='alice'))
        await _record(guard, ctx=_run_ctx(deps='alice'))
        await _record(guard, ctx=_run_ctx(deps='bob'))

        assert (await guard.status(scope='alice'))[0].spent.usd == Decimal('0.8')
        assert (await guard.status(scope='bob'))[0].spent.usd == Decimal('0.4')

    async def test_an_unscoped_budget_ignores_the_requested_scope(self):
        guard = SpendLimits(budgets=[Budget(window='day')], price=lambda r: Decimal('1'))
        await _record(guard)

        assert (await guard.status(scope='anything'))[0].spent.usd == Decimal('1')

    @pytest.mark.parametrize('resolved', ['', 'a|b', '*'])
    async def test_a_scope_key_that_would_collide_is_refused(self, resolved: str):
        """`'*'` included: it is how an unscoped budget is keyed, so it would share that counter."""
        guard = SpendLimits(budgets=[Budget(scope=lambda ctx: resolved)])

        with pytest.raises(UserError, match='must be non-empty and must not be'):
            await _record(guard)

    async def test_a_run_context_and_an_explicit_scope_together_are_refused(self):
        """Resolving one over the other silently reports one tenant's money under another's name."""
        guard = SpendLimits(
            budgets=[Budget(usd=Decimal('5'), scope=lambda ctx: str(ctx.deps), name='tenant')],
            price=lambda r: Decimal('1'),
        )
        await _record(guard, ctx=_run_ctx(deps='tenant-a'))
        ctx = _run_ctx(deps='tenant-a')

        with pytest.raises(UserError, match='not both'):
            await guard.status(ctx, scope='tenant-b')
        with pytest.raises(UserError, match='not both'):
            await guard.exhausted(ctx, scope='tenant-b')

        assert '|tenant-a|' in (await guard.status(ctx))[0].key
        assert (await guard.status(ctx))[0].spent.usd == Decimal('1')
        assert (await guard.status(scope='tenant-b'))[0].spent.usd == Decimal('0')


class TestEnforcement:
    """The gate refuses the next request once a window is spent."""

    async def test_a_usd_ceiling_stops_the_run(self):
        guard = SpendLimits(budgets=[Budget(usd=Decimal('0.02'))], price=lambda r: Decimal('0.01'))
        agent = _agent(guard)

        await agent.run('hi')
        await agent.run('hi')
        with pytest.raises(SpendLimitExceeded, match=r'spent \$0.02 of \$0.02'):
            await agent.run('hi')

    async def test_a_token_ceiling_stops_the_run(self):
        guard = SpendLimits(budgets=[Budget(tokens=1000)])
        agent = _agent(guard)

        await agent.run('hi')
        with pytest.raises(SpendLimitExceeded, match='used 1100 of 1000 tokens'):
            await agent.run('hi')

    async def test_it_is_catchable_as_a_usage_limit(self):
        guard = SpendLimits(budgets=[Budget(usd=Decimal('0.001'))], price=lambda r: Decimal('1'))
        await _record(guard)

        with pytest.raises(UsageLimitExceeded):
            await _gate(guard)

    async def test_a_counting_budget_never_stops_the_run(self):
        guard = SpendLimits(budgets=[Budget(window='total')], price=lambda r: Decimal('1000'))
        agent = _agent(guard)

        await agent.run('hi')
        await agent.run('hi')

        assert (await guard.status())[0].spent.requests == 2

    async def test_no_budgets_means_the_store_is_not_called_at_all(self):
        """Not "called with nothing": a store may treat an empty batch as a caller error.

        A `SpendLimits` with no budgets has no key to read and no entry to apply, so the
        round trip buys nothing on a store that answers it and fails outright on one that
        does not.
        """

        class Exploding(InMemorySpendStore):
            async def get_many(self, keys: Sequence[str]) -> Mapping[str, Spent]:
                raise AssertionError('the gate read with no budgets configured')  # pragma: no cover

            async def add_many(self, entries: Sequence[SpendEntry]) -> Mapping[str, Spent]:
                raise AssertionError('the gate wrote with no budgets configured')  # pragma: no cover

        guard = SpendLimits[None](store=Exploding())

        await _gate(guard)
        await _record(guard)

        assert await guard.status() == ()

    async def test_budgets_survive_across_runs(self):
        guard = SpendLimits(budgets=[Budget(usd=Decimal('0.01'), window='day')], price=lambda r: Decimal('0.01'))
        agent = _agent(guard)

        await agent.run('hi')
        with pytest.raises(SpendLimitExceeded):
            await agent.run('hi')


class _RetryOnceInnermost(AbstractCapability[None]):
    """Rejects the first response it has already awaited, from the innermost tier."""

    seen: int = 0

    def get_ordering(self) -> CapabilityOrdering:
        return CapabilityOrdering(position='innermost')

    async def wrap_model_request(
        self,
        ctx: RunContext[None],
        *,
        request_context: ModelRequestContext,
        handler: WrapModelRequestHandler,
    ) -> ModelResponse:
        response = await handler(request_context)
        self.seen += 1
        if self.seen == 1:
            raise ModelRetry('try again')
        return response


class _InnermostWithAWrapper(AbstractCapability[None]):
    """Innermost with a `wrap_model_request` of its own, which is all the report is about."""

    def get_ordering(self) -> CapabilityOrdering:
        return CapabilityOrdering(position='innermost')

    async def wrap_model_request(
        self,
        ctx: RunContext[None],
        *,
        request_context: ModelRequestContext,
        handler: WrapModelRequestHandler,
    ) -> ModelResponse:
        return await handler(request_context)


class _InnermostRejector(AbstractCapability[None]):
    """Innermost, and rejects every response it has already awaited."""

    def get_ordering(self) -> CapabilityOrdering:
        return CapabilityOrdering(position='innermost')

    async def wrap_model_request(
        self,
        ctx: RunContext[None],
        *,
        request_context: ModelRequestContext,
        handler: WrapModelRequestHandler,
    ) -> ModelResponse:
        await handler(request_context)
        raise RuntimeError('rejected after the provider had already been paid')


class _InnermostWithoutAWrapper(AbstractCapability[None]):
    """Innermost, like `ToolGuardrail`, but with no `wrap_model_request` of its own."""

    def get_ordering(self) -> CapabilityOrdering:
        return CapabilityOrdering(position='innermost')


class _DurabilityLookalike(AbstractCapability[None]):
    """Carries the attribute names a durability capability carries, without being one."""

    engine_name = 'not a durable engine'
    in_durable_context = False

    def get_ordering(self) -> CapabilityOrdering:
        return CapabilityOrdering(position='innermost')

    async def wrap_model_request(
        self,
        ctx: RunContext[None],
        *,
        request_context: ModelRequestContext,
        handler: WrapModelRequestHandler,
    ) -> ModelResponse:
        return await handler(request_context)


class _InnermostWrapper(WrapperCapability[None]):
    """A wrapper that reaches the innermost tier and leaves `wrap_model_request` delegating.

    `WrapperCapability.apply` registers a wrapper over a leaf as itself, not as the leaf, so a
    bare wrapper does not inherit the wrapped capability's `innermost` position. Declaring it
    is what puts a wrapper after `SpendLimits` at all.
    """

    def get_ordering(self) -> CapabilityOrdering:
        return CapabilityOrdering(position='innermost')


class _WrapperWithItsOwnWrapper(_InnermostWrapper):
    """A wrapper subclass that supplies `wrap_model_request` instead of delegating it."""

    async def wrap_model_request(
        self,
        ctx: RunContext[None],
        *,
        request_context: ModelRequestContext,
        handler: WrapModelRequestHandler,
    ) -> ModelResponse:
        return await handler(request_context)


class _HooksWithItsOwnWrapper(Hooks[None]):
    """A `Hooks` subclass that supplies the method instead of dispatching to a registry."""

    async def wrap_model_request(
        self,
        ctx: RunContext[None],
        *,
        request_context: ModelRequestContext,
        handler: WrapModelRequestHandler,
    ) -> ModelResponse:
        return await handler(request_context)


async def _passthrough(
    ctx: RunContext[None],
    /,
    *,
    request_context: ModelRequestContext,
    handler: WrapModelRequestHandler,
) -> ModelResponse:
    return await handler(request_context)


class TestOrdering:
    """Nothing may reject a response before it is counted."""

    async def test_a_response_rejected_by_a_later_capability_is_still_counted(self):
        """The accrual runs inside the wrap chain, so every `after_model_request` is outside it."""

        class RejectFirst(AbstractCapability[None]):
            seen: int = 0

            async def after_model_request(
                self, ctx: RunContext[None], *, request_context: ModelRequestContext, response: ModelResponse
            ) -> ModelResponse:
                self.seen += 1
                if self.seen == 1:
                    raise ModelRetry('try again')
                return response

        guard = SpendLimits(budgets=[Budget(window='total')], price=lambda r: Decimal('1'))
        agent = Agent(_scripted_usage(), deps_type=type(None), capabilities=[guard, RejectFirst()])
        result = await agent.run('hi')

        assert result.usage.requests == 2
        assert (await guard.status())[0].spent.requests == 2

    async def test_a_response_a_wrapper_retries_on_is_still_counted(self):
        """A wrapper that rejects a response it awaited retries before `after_model_request` ever runs.

        Ordering cannot reach this one: the rejecting capability is not innermost and is
        listed *before* `SpendLimits`, so it wraps outside the accrual either way. Accruing
        in `after_model_request` left the provider billing two requests and the counter
        seeing one.
        """

        class RetryOnce(AbstractCapability[None]):
            seen: int = 0

            async def wrap_model_request(
                self,
                ctx: RunContext[None],
                *,
                request_context: ModelRequestContext,
                handler: WrapModelRequestHandler,
            ) -> ModelResponse:
                response = await handler(request_context)
                self.seen += 1
                if self.seen == 1:
                    raise ModelRetry('try again')
                return response

        guard = SpendLimits(budgets=[Budget(window='total')], price=lambda r: Decimal('1'))
        agent = Agent(_scripted_usage(), deps_type=type(None), capabilities=[RetryOnce(), guard])
        result = await agent.run('hi')

        assert result.usage.requests == 2
        spent = (await guard.status())[0].spent
        assert spent.requests == 2
        assert spent.usd == Decimal('2')

    async def test_a_request_the_provider_never_saw_is_not_counted(self):
        """`SkipModelRequest` reaches `after_model_request` but never reaches the wrapped handler."""

        class ServeFromCache(AbstractCapability[None]):
            async def before_model_request(
                self, ctx: RunContext[None], request_context: ModelRequestContext
            ) -> ModelRequestContext:
                raise SkipModelRequest(ModelResponse(parts=[TextPart(content='cached')]))

        guard = SpendLimits(budgets=[Budget(window='total')], price=lambda r: Decimal('1'))
        agent = Agent(_scripted_usage(), deps_type=type(None), capabilities=[ServeFromCache(), guard])
        result = await agent.run('hi')

        assert result.output == 'cached'
        assert (await guard.status())[0].spent == Spent()

    async def test_an_innermost_capability_listed_after_leaves_a_billed_response_uncounted(self):
        """Innermost members are not ordered among themselves, so the later one nests further in.

        The provider bills both requests and the counter sees one. This is the arrangement
        `get_ordering` cannot rule out, and the reason it reports itself.
        """
        guard = SpendLimits(budgets=[Budget(window='total')], price=lambda r: Decimal('1'))
        agent = Agent(_scripted_usage(), deps_type=type(None), capabilities=[guard, _RetryOnceInnermost()])

        with pytest.warns(SpendCompositionWarning):
            result = await agent.run('hi')

        assert result.usage.requests == 2
        assert (await guard.status())[0].spent.requests == 1

    async def test_listing_spend_limits_last_counts_every_billed_response(self):
        """The documented fix: the rejecting capability wraps outside the accrual again."""
        guard = SpendLimits(budgets=[Budget(window='total')], price=lambda r: Decimal('1'))
        agent = Agent(_scripted_usage(), deps_type=type(None), capabilities=[_RetryOnceInnermost(), guard])

        result = await agent.run('hi')

        assert result.usage.requests == 2
        assert (await guard.status())[0].spent.requests == 2

    def test_it_declares_innermost(self):
        assert SpendLimits[None]().get_ordering() == CapabilityOrdering(position='innermost')


class TestCompositionWarning:
    """The arrangement that can leave a billed response uncounted reports itself."""

    async def test_it_names_the_nested_capability_and_the_fix(self):
        guard = SpendLimits[None](budgets=[Budget(window='total')])
        agent = Agent(_scripted_usage(), deps_type=type(None), capabilities=[guard, _InnermostRejector()])

        with pytest.warns(SpendCompositionWarning, match=r'_InnermostRejector.*List `SpendLimits` last'):
            with pytest.raises(RuntimeError):
                await agent.run('hi')

    async def test_it_reports_one_arrangement_once_across_runs(self):
        guard = SpendLimits[None](budgets=[Budget(window='total')])
        agent = Agent(_scripted_usage(), deps_type=type(None), capabilities=[guard, _InnermostWithAWrapper()])

        with warnings.catch_warnings(record=True) as reported:
            warnings.simplefilter('always')
            await agent.run('hi')
            await agent.run('hi')

        assert [str(w.message) for w in reported if w.category is SpendCompositionWarning] == [
            'These capabilities are listed after `SpendLimits`, so they wrap inside it: _InnermostWithAWrapper. '
            'If one of them rejects a response it has already awaited, the provider billed that response and '
            'the accrual never sees it. This reads the ordering, not what those capabilities do with it. '
            'List `SpendLimits` last among the innermost capabilities to rule it out.'
        ]

    async def test_listing_spend_limits_last_reports_nothing(self):
        guard = SpendLimits[None](budgets=[Budget(window='total')])
        agent = Agent(_scripted_usage(), deps_type=type(None), capabilities=[_InnermostWithAWrapper(), guard])

        await agent.run('hi')

    async def test_a_capability_with_no_wrapper_of_its_own_is_not_reported(self):
        """Nesting only matters for a capability that can reject the response on the way out."""
        guard = SpendLimits[None](budgets=[Budget(window='total')])
        agent = Agent(_scripted_usage(), deps_type=type(None), capabilities=[guard, _InnermostWithoutAWrapper()])

        await agent.run('hi')

    async def test_a_hooks_capability_is_not_reported(self):
        """`Hooks` defines the method unconditionally, so its definition cannot answer the question.

        The cost is a missed report for a `Hooks` that did register a `model_request` hook,
        which is preferred over reporting one that did not: that arrangement is correct, and
        the user could only silence the warning by changing correct code.
        """
        guard = SpendLimits[None](budgets=[Budget(window='total')])
        hooks = Hooks[None](ordering=CapabilityOrdering(position='innermost'), model_request=_passthrough)
        agent = Agent(_scripted_usage(), deps_type=type(None), capabilities=[guard, hooks])

        await agent.run('hi')

    async def test_a_durable_execution_capability_is_not_reported(self):
        """It routes the request into a durable unit rather than rejecting what comes back.

        Core also requires its dispatch to be the innermost wrapper, so listing `SpendLimits`
        after it is the one correction a reader must not make. What `SpendLimits` does not
        support under a durable engine is reported separately, by refusing the workflow clock.
        """
        pytest.importorskip('temporalio')
        from pydantic_ai.durable_exec.temporal import TemporalDurability

        guard = SpendLimits[None](budgets=[Budget(window='total')])
        agent = Agent(
            _scripted_usage(),
            name='durable',
            deps_type=type(None),
            capabilities=[guard, TemporalDurability[None]()],
        )

        await agent.run('hi')

    async def test_a_capability_that_only_looks_durable_is_still_reported(self):
        """The exclusion matches the durability base type, not attributes anything could carry."""
        guard = SpendLimits[None](budgets=[Budget(window='total')])
        agent = Agent(_scripted_usage(), deps_type=type(None), capabilities=[guard, _DurabilityLookalike()])

        with pytest.warns(SpendCompositionWarning, match='_DurabilityLookalike'):
            await agent.run('hi')

    async def test_a_hooks_subclass_with_its_own_wrapper_is_reported(self):
        guard = SpendLimits[None](budgets=[Budget(window='total')])
        hooks = _HooksWithItsOwnWrapper(ordering=CapabilityOrdering(position='innermost'))
        agent = Agent(_scripted_usage(), deps_type=type(None), capabilities=[guard, hooks])

        with pytest.warns(SpendCompositionWarning, match='_HooksWithItsOwnWrapper'):
            await agent.run('hi')

    async def test_a_capability_added_for_one_run_is_reported(self):
        """`agent.run(capabilities=...)` is why the chain is read per run rather than at binding."""
        guard = SpendLimits[None](budgets=[Budget(window='total')])
        agent = Agent(_scripted_usage(), deps_type=type(None), capabilities=[guard])

        with pytest.warns(SpendCompositionWarning, match='_InnermostWithAWrapper'):
            await agent.run('hi', capabilities=[_InnermostWithAWrapper()])

    async def test_a_run_that_adds_one_later_is_still_reported(self):
        """A safe first run must not mark every chain that follows it as read.

        Nothing is reported on the first run, so a flag set by merely having checked would
        suppress the second. What is remembered is the arrangement, and the first run has none.
        """
        guard = SpendLimits[None](budgets=[Budget(window='total')])
        agent = Agent(_scripted_usage(), deps_type=type(None), capabilities=[guard])

        await agent.run('hi')
        with pytest.warns(SpendCompositionWarning, match='_InnermostWithAWrapper'):
            await agent.run('hi', capabilities=[_InnermostWithAWrapper()])

    async def test_a_second_arrangement_reports_even_though_the_first_already_warned(self):
        """Remembering that a warning fired is not the same as remembering which arrangement fired it.

        A flag set when the warning fires passes both tests above, and loses this one: the same
        `SpendLimits` instance is surrounded by a different capability on the second run, which
        is a different arrangement and has never been reported.
        """
        guard = SpendLimits[None](budgets=[Budget(window='total')])
        agent = Agent(_scripted_usage(), deps_type=type(None), capabilities=[guard])

        with pytest.warns(SpendCompositionWarning, match='_InnermostWithAWrapper'):
            await agent.run('hi', capabilities=[_InnermostWithAWrapper()])
        with pytest.warns(SpendCompositionWarning, match='_InnermostRejector'):
            with pytest.raises(RuntimeError):
                await agent.run('hi', capabilities=[_InnermostRejector()])

    async def test_an_arrangement_escalated_to_an_error_is_refused_on_every_run(self):
        """Escalating the category is a refusal, so it cannot stop refusing after one run.

        `warnings.warn` raises under `filterwarnings('error', ...)`, so an arrangement recorded
        before the call would be marked reported by the run the raise came from and skipped
        after it. Recording it after the call returns is what keeps the second run refused.
        """
        guard = SpendLimits[None](budgets=[Budget(window='total')])
        agent = Agent(_scripted_usage(), deps_type=type(None), capabilities=[guard])

        with warnings.catch_warnings():
            warnings.simplefilter('error', SpendCompositionWarning)
            for _ in range(2):
                with pytest.raises(SpendCompositionWarning):
                    await agent.run('hi', capabilities=[_InnermostWithAWrapper()])

    async def test_a_wrapper_is_answered_on_what_it_wraps(self):
        """`WrapperCapability.wrap_model_request` only delegates, so defining it says nothing."""
        guard = SpendLimits[None](budgets=[Budget(window='total')])
        agent = Agent(
            _scripted_usage(),
            deps_type=type(None),
            capabilities=[guard, _InnermostWrapper(_InnermostWithoutAWrapper())],
        )

        await agent.run('hi')

    async def test_a_wrapper_over_a_real_wrapper_is_reported(self):
        guard = SpendLimits[None](budgets=[Budget(window='total')])
        agent = Agent(
            _scripted_usage(),
            deps_type=type(None),
            capabilities=[guard, _InnermostWrapper(_InnermostWithAWrapper())],
        )

        with pytest.warns(SpendCompositionWarning, match='_InnermostWrapper'):
            await agent.run('hi')

    async def test_a_wrapper_subclass_with_its_own_wrapper_is_reported(self):
        """Overriding the method supplies one, so the wrapped capability stops being the answer."""
        guard = SpendLimits[None](budgets=[Budget(window='total')])
        agent = Agent(
            _scripted_usage(),
            deps_type=type(None),
            capabilities=[guard, _WrapperWithItsOwnWrapper(_InnermostWithoutAWrapper())],
        )

        with pytest.warns(SpendCompositionWarning, match='_WrapperWithItsOwnWrapper'):
            await agent.run('hi')

    async def test_a_sequential_input_guardrail_is_reported_although_it_cannot_under_count(self):
        """The shipped default trips the check, and the docs say so.

        `InputGuardrail(parallel=False)` runs its guard before calling the handler, so it never
        holds a billed response to reject. The report reads the ordering, not `parallel`.
        """
        guard = SpendLimits[None](budgets=[Budget(window='total')])
        agent = Agent(
            _scripted_usage(),
            deps_type=type(None),
            capabilities=[guard, InputGuardrail[None](guard=lambda ctx, text: GuardrailResult.allow())],
        )

        with pytest.warns(SpendCompositionWarning, match='InputGuardrail'):
            await agent.run('hi')

    async def test_nothing_is_reported_without_a_capability_chain(self):
        """`RunContext.root_capability` is unset outside a run, leaving nothing to compare against."""
        await _gate(SpendLimits[None](budgets=[Budget(window='total')]))

    async def test_nothing_is_reported_when_the_chain_does_not_list_it(self):
        """A chain this `SpendLimits` has no position in leaves nothing to compare against."""
        guard = SpendLimits[None](budgets=[Budget(window='total')])
        ctx = _run_ctx(root_capability=CombinedCapability[None]([_InnermostWithAWrapper()]))

        await _gate(guard, ctx=ctx)

    async def test_a_wrapped_spend_limits_is_still_located_in_the_chain(self):
        """The chain holds the wrapper, and the accrual it delegates to runs at that position.

        Comparing chain members by identity alone reads this as "not in the chain" and reports
        nothing, while the rejector listed after the wrapper still nests inside the accrual:
        the provider bills the response and the counter never sees it.
        """
        guard = SpendLimits[None](budgets=[Budget(window='total')], price=lambda r: Decimal('1'))
        agent = Agent(
            _scripted_usage(),
            deps_type=type(None),
            capabilities=[_InnermostWrapper(guard), _InnermostRejector()],
        )

        with pytest.warns(SpendCompositionWarning, match='_InnermostRejector'):
            with pytest.raises(RuntimeError):
                await agent.run('hi')

        assert (await guard.status())[0].spent.requests == 0


class TestPricing:
    """What a response cost, and whether that number is real."""

    async def test_the_registry_prices_a_known_model(self):
        guard = SpendLimits(budgets=[Budget(window='total')])
        await _record(guard)

        status = (await guard.status())[0]
        assert status.spent.usd > 0
        assert status.spent.unpriced_requests == 0

    async def test_the_price_override_wins(self):
        guard = SpendLimits(budgets=[Budget(window='total')], price=lambda r: Decimal('7'))
        await _record(guard)

        assert (await guard.status())[0].spent.usd == Decimal('7')

    async def test_an_override_returning_none_falls_through_to_the_registry(self):
        guard = SpendLimits(budgets=[Budget(window='total')], price=_no_price)
        await _record(guard)

        assert (await guard.status())[0].spent.usd > 0

    @pytest.mark.parametrize(
        'kwargs',
        [{'model_name': None}, {'model_name': 'not-a-real-model', 'provider_name': None}],
    )
    async def test_an_unpriceable_response_still_counts_tokens(self, kwargs: Any):
        guard = SpendLimits(budgets=[Budget(window='total')])
        await _record(guard, **kwargs)

        status = (await guard.status())[0]
        assert status.spent.usd == Decimal('0')
        assert status.spent.tokens == 1100
        assert status.spent.unpriced_requests == 1

    async def test_a_usage_shape_the_registry_rejects_is_treated_as_unpriced(self):
        """`genai-prices` raises ValueError on some shapes; letting it escape would skip `on_unpriced`."""
        guard = SpendLimits(budgets=[Budget(window='total')])
        response = ModelResponse(
            parts=[TextPart(content='x')],
            usage=RequestUsage(input_tokens=10, cache_read_tokens=500, output_tokens=5),
            model_name='gpt-4.1',
            provider_name='openai',
        )
        await _record(guard, response=response)

        spent = (await guard.status())[0].spent
        assert spent.unpriced_requests == 1
        assert spent.tokens == 15

    async def test_a_negative_price_is_refused(self):
        """A credit would move a budget away from its ceiling, so the gate would never close."""
        guard = SpendLimits(budgets=[Budget(window='total')], price=lambda r: Decimal('-1'))

        with pytest.raises(UserError, match='returned a negative amount'):
            await _record(guard)

    @pytest.mark.parametrize('amount', ['NaN', 'Infinity', '-Infinity'])
    async def test_a_non_finite_price_is_refused(self, amount: str):
        """`NaN < 0` raises `InvalidOperation`, and an infinity would exhaust every budget at once."""
        guard = SpendLimits(budgets=[Budget(window='total')], price=lambda r: Decimal(amount))

        with pytest.raises(UserError, match='returned a non-finite amount'):
            await _record(guard)

    @pytest.mark.parametrize(('amount', 'message'), [('-1', 'negative amount'), ('NaN', 'non-finite amount')])
    async def test_a_rejected_price_still_records_the_response(self, amount: str, message: str):
        """The request was billed by the provider whatever the pricing function returned."""
        seen: list[SpendSnapshot] = []
        guard = SpendLimits(
            budgets=[Budget(window='total', name='audit')],
            price=lambda r: Decimal(amount),
            on_spend=seen.append,
        )

        with pytest.raises(UserError, match=message):
            await _record(guard)

        spent = (await guard.status())[0].spent
        assert spent.requests == 1
        assert spent.tokens > 0
        assert spent.usd == Decimal(0)
        assert spent.unpriced_requests == 1
        assert len(seen) == 1

    async def test_an_unpriced_response_warns_once_per_model_against_a_usd_ceiling(self):
        """A USD ceiling cannot be reached by requests nothing can price, so it says so -- once."""
        guard = SpendLimits(budgets=[Budget(usd=Decimal('5'), window='day')])

        with pytest.warns(UnpricedModelWarning, match='not-a-real-model') as caught:
            await _record(guard, model_name='not-a-real-model', provider_name=None)
            await _record(guard, model_name='not-a-real-model', provider_name=None)

        assert len(caught) == 1

    async def test_a_second_unpriced_model_warns_separately(self):
        guard = SpendLimits(budgets=[Budget(usd=Decimal('5'), window='day')])

        with pytest.warns(UnpricedModelWarning) as caught:
            await _record(guard, model_name='unknown-one', provider_name=None)
            await _record(guard, model_name='unknown-two', provider_name=None)

        assert len(caught) == 2

    async def test_a_response_with_no_model_name_warns_under_one_label(self):
        guard = SpendLimits(budgets=[Budget(usd=Decimal('5'), window='day')])

        with pytest.warns(UnpricedModelWarning, match='<unnamed>') as caught:
            await _record(guard, model_name=None)
            await _record(guard, model_name=None)

        assert len(caught) == 1

    async def test_a_token_only_ceiling_does_not_warn(self):
        """Tokens are counted whether or not a price was found, so that ceiling still holds."""
        guard = SpendLimits(budgets=[Budget(tokens=5000, window='day')])

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            await _record(guard, model_name='not-a-real-model', provider_name=None)

        assert [w for w in caught if issubclass(w.category, UnpricedModelWarning)] == []

    async def test_raising_does_not_also_warn(self):
        """`on_unpriced='raise'` already stops the run; a warning beside it would be noise."""
        guard = SpendLimits(budgets=[Budget(usd=Decimal('5'), window='day')], on_unpriced='raise')

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            with pytest.raises(UnpricedModelError):
                await _record(guard, model_name='not-a-real-model', provider_name=None)

        assert [w for w in caught if issubclass(w.category, UnpricedModelWarning)] == []

    async def test_a_priced_response_does_not_warn(self):
        guard = SpendLimits(budgets=[Budget(usd=Decimal('5'), window='day')])

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            await _record(guard)

        assert [w for w in caught if issubclass(w.category, UnpricedModelWarning)] == []

    async def test_raising_on_a_model_the_registry_does_not_know(self):
        guard = SpendLimits(budgets=[Budget(window='total')], on_unpriced='raise')

        with pytest.raises(UnpricedModelError, match='not-a-real-model'):
            await _record(guard, model_name='not-a-real-model', provider_name=None)

    async def test_a_refused_response_is_still_counted(self):
        """The tokens were really spent, so dropping them would understate a token ceiling."""
        guard = SpendLimits(budgets=[Budget(tokens=5000, window='day')], on_unpriced='raise')

        with pytest.raises(UnpricedModelError):
            await _record(guard, model_name=None)

        spent = (await guard.status())[0].spent
        assert spent.tokens == 1100
        assert spent.requests == 1
        assert spent.unpriced_requests == 1

    async def test_a_refused_response_still_reaches_on_spend(self):
        """An audit that skipped exactly the unpriced responses would miss the ones that matter."""
        seen: list[SpendSnapshot] = []
        guard = SpendLimits(budgets=[Budget(window='total')], on_unpriced='raise', on_spend=seen.append)

        with pytest.raises(UnpricedModelError):
            await _record(guard, model_name=None)

        assert len(seen) == 1
        assert seen[0].priced is False
        assert seen[0].usage.input_tokens == 1000

    async def test_raising_on_an_unpriceable_response(self):
        guard = SpendLimits(budgets=[Budget(window='total')], on_unpriced='raise')

        with pytest.raises(UnpricedModelError, match='<unnamed>'):
            await _record(guard, model_name=None)


class TestConfigurationFromData:
    """Values that arrive as plain data are checked, not trusted."""

    @pytest.mark.parametrize('window', ['weekly', 'daily', ''])
    def test_an_unknown_window_is_refused(self, window: str):
        """A `Literal` is not enforced at runtime, so a spec typo would reach `assert_never`."""
        with pytest.raises(UserError, match='window must be one of'):
            SpendLimits[None].from_spec(budgets=_spec_budgets({'window': window}))

    @pytest.mark.parametrize('policy', ['raises', 'Zero', ''])
    def test_an_unknown_unpriced_policy_is_refused(self, policy: str):
        """Anything but `'raise'` behaves as `'zero'`, so a typo would quietly make responses free."""
        with pytest.raises(UserError, match='on_unpriced must be one of'):
            SpendLimits[None](on_unpriced=policy)  # pyright: ignore[reportArgumentType]


class TestObservability:
    """Push through `on_spend`, pull through `status`."""

    async def test_on_spend_receives_the_response_usage_verbatim(self):
        seen: list[SpendSnapshot] = []
        guard = SpendLimits(budgets=[Budget(window='total')], on_spend=seen.append)
        await _record(guard)

        assert len(seen) == 1
        assert seen[0].usage.input_tokens == 1000
        assert seen[0].priced is True
        assert seen[0].budgets[0].budget.window == 'total'

    async def test_on_spend_may_be_async(self):
        seen: list[Decimal] = []

        async def record(snapshot: SpendSnapshot) -> None:
            seen.append(snapshot.usd)

        guard = SpendLimits[None](price=lambda r: Decimal('3'), on_spend=record)
        await _record(guard)

        assert seen == [Decimal('3')]

    async def test_status_reports_what_is_left(self):
        guard = SpendLimits(
            budgets=[Budget(usd=Decimal('10'), tokens=5000, warn_at=0.5)],
            price=lambda r: Decimal('6'),
        )
        await _record(guard)

        status = (await guard.status())[0]
        assert status.remaining_usd == Decimal('4')
        assert status.remaining_tokens == 3900
        assert status.warning is True
        assert status.exhausted is False

    async def test_a_token_ceiling_reports_as_exhausted(self):
        guard = SpendLimits(budgets=[Budget(tokens=1000)])
        await _record(guard)

        status = (await guard.status())[0]
        assert status.remaining_tokens == -100
        assert status.exhausted is True
        assert status.remaining_usd is None

    async def test_a_token_warning_fires_without_a_usd_limit(self):
        guard = SpendLimits(budgets=[Budget(tokens=2000, warn_at=0.5)])
        await _record(guard)

        assert (await guard.status())[0].warning is True

    async def test_no_warning_threshold_means_no_warning(self):
        guard = SpendLimits(budgets=[Budget(usd=Decimal('0.0001'))], price=lambda r: Decimal('1'))
        await _record(guard)

        status = (await guard.status())[0]
        assert status.warning is False
        assert status.exhausted is True

    async def test_status_without_a_run_omits_what_it_cannot_resolve(self):
        guard = SpendLimits(
            budgets=[
                Budget(name='daily', window='day'),
                Budget(name='per-run', window='run'),
                Budget(name='per-tenant', scope=lambda ctx: 'acme'),
            ]
        )

        assert [s.budget.name for s in await guard.status()] == ['daily']
        assert [s.budget.name for s in await guard.status(scope='acme')] == ['daily', 'per-tenant']
        assert [s.budget.name for s in await guard.status(_run_ctx())] == ['daily', 'per-run', 'per-tenant']


class TestTracing:
    """Only a refusal is worth a span."""

    async def test_a_refusal_records_a_span(self):
        tracer, exporter = _recording_tracer()
        guard = SpendLimits(budgets=[Budget(usd=Decimal('0.001'), window='day')], price=lambda r: Decimal('1'))
        await _record(guard)

        with pytest.raises(SpendLimitExceeded):
            await _gate(guard, ctx=_run_ctx(tracer=tracer))

        span = _only_span(exporter)
        assert span.name == 'spend budget exhausted'
        assert dict(span.attributes or {}) == {'spend.budget': 'default', 'spend.window': 'day'}

    async def test_the_scope_key_is_content(self):
        tracer, exporter = _recording_tracer()
        guard = SpendLimits(
            budgets=[Budget(usd=Decimal('0.001'), scope=lambda ctx: str(ctx.deps))],
            price=lambda r: Decimal('1'),
        )
        await _record(guard, ctx=_run_ctx(deps='acme'))

        with pytest.raises(SpendLimitExceeded):
            await _gate(guard, ctx=_run_ctx(deps='acme', tracer=tracer, trace_include_content=True))

        assert dict(_only_span(exporter).attributes or {})['spend.scope'] == 'acme'

    async def test_accrual_records_nothing(self):
        tracer, exporter = _recording_tracer()
        guard = SpendLimits(budgets=[Budget(window='total')])
        await _record(guard, ctx=_run_ctx(tracer=tracer))

        assert exporter.get_finished_spans() == ()


class TestInMemoryStore:
    """The default store, and how it forgets."""

    async def test_an_unwritten_key_reads_as_zero(self):
        assert await InMemorySpendStore().get('nothing') == Spent()

    async def test_a_lifetime_key_is_never_dropped(self):
        clock = Clock()
        store = InMemorySpendStore(clock=clock)
        await store.add('k', usd=Decimal('1'), tokens=1, requests=1, unpriced=0, ttl=None)
        clock.advance(timedelta(days=999))

        assert (await store.get('k')).usd == Decimal('1')

    async def test_an_expired_key_is_dropped_on_access(self):
        clock = Clock()
        store = InMemorySpendStore(clock=clock)
        await store.add('k', usd=Decimal('1'), tokens=1, requests=1, unpriced=0, ttl=timedelta(hours=1))
        clock.advance(timedelta(hours=2))

        assert await store.get('k') == Spent()

    async def test_a_rolled_over_key_is_swept_rather_than_kept(self):
        """A day key is never read again once the day turns, so only a sweep can drop it."""
        clock = Clock()
        store = InMemorySpendStore(clock=clock)
        await store.add('monday', usd=Decimal('1'), tokens=1, requests=1, unpriced=0, ttl=timedelta(hours=48))
        clock.advance(timedelta(days=3))

        await store.add('thursday', usd=Decimal('1'), tokens=1, requests=1, unpriced=0, ttl=timedelta(hours=48))

        assert len(store) == 1

    async def test_expiring_a_key_on_read_happens_under_the_lock(self):
        """`get` deletes the key it finds expired, which unlocked races the sweep inside `add`.

        Asserted through the lock rather than by racing threads: the interleaving that breaks
        it is real (`RuntimeError: dictionary changed size during iteration`) but not
        reproducible on demand, and a test that fails one run in fifty is not a regression test.
        """
        store = InMemorySpendStore(clock=(clock := Clock()))
        await store.add('k', usd=Decimal('1'), tokens=1, requests=1, unpriced=0, ttl=timedelta(hours=1))
        clock.advance(timedelta(hours=2))
        held: list[bool] = []
        real_lock = store._lock  # pyright: ignore[reportPrivateUsage]

        class _RecordingLock:
            def __enter__(self) -> None:
                real_lock.acquire()
                held.append(True)

            def __exit__(self, *exc_info: object) -> None:
                real_lock.release()

        object.__setattr__(store, '_lock', _RecordingLock())

        assert await store.get('k') == Spent()
        assert held == [True], 'the expiring read ran outside the lock'

    async def test_the_sweep_is_amortised_but_length_still_excludes_dead_keys(self):
        """A full scan under the lock on every write blocks the loop once the dict is large."""
        clock = Clock()
        store = InMemorySpendStore(clock=clock)
        await store.add('monday', usd=Decimal('1'), tokens=1, requests=1, unpriced=0, ttl=timedelta(hours=48))
        clock.advance(timedelta(days=3))
        await store.add('thursday', usd=Decimal('1'), tokens=1, requests=1, unpriced=0, ttl=timedelta(hours=48))

        assert len(store) == 1
        assert await store.get('monday') == Spent()

    async def test_dead_entries_are_physically_dropped_once_the_sweep_runs(self):
        """`__len__` hides dead entries either way, so residency is read by rewinding the clock.

        An entry the sweep dropped stays gone when the clock goes back; one merely filtered by
        `__len__` counts again. That is the difference between reclaiming the memory and only
        hiding it, without reading the store's internals.
        """
        clock = Clock()
        store = InMemorySpendStore(clock=clock, sweep_every=4)
        for index in range(4):
            await store.add(f'k{index}', usd=Decimal('1'), tokens=1, requests=1, unpriced=0, ttl=timedelta(hours=1))
        clock.advance(timedelta(hours=2))
        assert len(store) == 0

        for index in range(4):
            await store.add(f'n{index}', usd=Decimal('1'), tokens=1, requests=1, unpriced=0, ttl=timedelta(hours=1))

        assert len(store) == 4
        clock.now = _EPOCH
        assert len(store) == 4, 'the rolled-over entries were hidden rather than dropped'

    async def test_a_reconciler_may_post_a_negative_delta(self):
        store = InMemorySpendStore()
        await store.add('k', usd=Decimal('5'), tokens=10, requests=1, unpriced=0, ttl=None)
        corrected = await store.add('k', usd=Decimal('-2'), tokens=0, requests=0, unpriced=0, ttl=None)

        assert corrected == Spent(usd=Decimal('3'), tokens=10, requests=1, unpriced_requests=0)

    async def test_every_entry_of_a_response_lands_together(self):
        store = InMemorySpendStore()

        totals = await store.add_many(
            [
                SpendEntry(key='day', usd=Decimal('1'), tokens=5, requests=1),
                SpendEntry(key='month', usd=Decimal('1'), tokens=5, requests=1),
            ]
        )

        assert totals == {
            'day': Spent(usd=Decimal('1'), tokens=5, requests=1),
            'month': Spent(usd=Decimal('1'), tokens=5, requests=1),
        }
        assert await store.get_many(['day', 'month']) == totals

    async def test_a_repeated_token_is_applied_once(self):
        """The in-process equivalent of the marker `RedisSpendStore` claims."""
        store = InMemorySpendStore()
        entry = SpendEntry(key='k', usd=Decimal('1'), requests=1, token='resp-1')

        await store.add_many([entry])
        totals = await store.add_many([entry])

        assert totals == {'k': Spent(usd=Decimal('1'), requests=1)}

    async def test_dedup_can_be_turned_off(self):
        store = InMemorySpendStore(dedup_retain=None)
        entry = SpendEntry(key='k', usd=Decimal('1'), requests=1, token='resp-1')

        await store.add_many([entry])
        totals = await store.add_many([entry])

        assert totals == {'k': Spent(usd=Decimal('2'), requests=2)}

    async def test_a_call_that_fails_does_not_consume_the_token(self):
        """The token is recorded once the counter has moved, not before it.

        Recorded first, a call that then failed left the token consumed and the retry
        that could have recorded the response was skipped as a replay of it. Two
        infinities are the smallest way to make the addition itself fail; any failure
        between the two writes has the same shape.
        """
        store = InMemorySpendStore()
        await store.add_many([SpendEntry(key='k', usd=Decimal('Infinity'))])

        with pytest.raises(InvalidOperation):
            await store.add_many([SpendEntry(key='k', usd=Decimal('-Infinity'), requests=1, token='resp-1')])

        retried = await store.add_many([SpendEntry(key='k', requests=1, token='resp-1')])

        assert retried['k'].requests == 1

    async def test_a_failing_entry_leaves_the_whole_response_unapplied(self):
        """The split write `add_many` exists to prevent, inside one process rather than across a network.

        Two infinities are the smallest way to make one entry's arithmetic raise; any
        failure part-way through a response has the same shape.
        """
        store = InMemorySpendStore()
        await store.add_many([SpendEntry(key='month', usd=Decimal('Infinity'))])

        with pytest.raises(InvalidOperation):
            await store.add_many(
                [
                    SpendEntry(key='day', usd=Decimal('1'), requests=1),
                    SpendEntry(key='month', usd=Decimal('-Infinity'), requests=1),
                ]
            )

        assert (await store.get_many(['day']))['day'] == Spent()

    async def test_a_token_is_remembered_no_longer_than_its_counter(self):
        """A marker outliving its window would skip a replay against a counter that rolled over."""
        clock = Clock()
        store = InMemorySpendStore(clock=clock, dedup_retain=timedelta(hours=24))
        entry = SpendEntry(key='k', usd=Decimal('1'), requests=1, ttl=timedelta(hours=1), token='resp-1')
        await store.add_many([entry])

        clock.advance(timedelta(hours=2))

        assert await store.add_many([entry]) == {'k': Spent(usd=Decimal('1'), requests=1)}

    async def test_one_token_twice_in_a_call_is_applied_once(self):
        store = InMemorySpendStore()
        entry = SpendEntry(key='k', usd=Decimal('1'), requests=1, token='resp-1')

        assert await store.add_many([entry, entry]) == {'k': Spent(usd=Decimal('1'), requests=1)}

    async def test_an_application_s_own_decimal_precision_does_not_round_the_counter(self):
        """A store that took the caller's context would round the counter and then drift from it.

        The rounded total is what the next response is added to, so the error compounds rather
        than showing up once. `_snapshot.summed` pins the arithmetic for both stores.
        """
        store = InMemorySpendStore()
        entry = SpendEntry(key='k', usd=Decimal('0.000123456'), requests=1)

        with localcontext() as context:
            context.prec = 3
            first = await store.add_many([entry])
            second = await store.add_many([entry])

        assert first['k'].usd == Decimal('0.000123456')
        assert second['k'].usd == Decimal('0.000246912')

    async def test_a_token_past_its_horizon_is_forgotten(self):
        """`dedup_retain` is the window a replay is recognised in, not the counter's lifetime.

        The counter here never expires, so it is the marker's own horizon that decides:
        past it the response counts again. Remembering every token instead would grow with
        traffic, which is the price the bound buys.
        """
        clock = Clock()
        store = InMemorySpendStore(clock=clock, sweep_every=2, dedup_retain=timedelta(hours=1))
        entry = SpendEntry(key='k', usd=Decimal('1'), requests=1, token='resp-1')
        await store.add_many([entry])

        clock.advance(timedelta(hours=2))
        await store.add_many([SpendEntry(key='other', requests=1)])
        totals = await store.add_many([entry])

        assert totals == {'k': Spent(usd=Decimal('2'), requests=2)}


class FakeRedis:
    """The two coroutines `RedisSpendStore` uses, over a dict.

    `eval` stands in for the server running the script: the same increments, the same
    per-entry markers and the same expiry, applied as one step, which is what Redis
    guarantees. It reproduces the *result* the script is meant to produce and does not
    run the Lua, so what it pins is the store's contract rather than the script's
    correctness -- `integration_tests/redis/` is where the script itself is executed.

    Totals come back as strings, because that is what keeps a counter past `2**53`
    exact once a real server has put it through Lua.

    `HINCRBY` leaves an existing expiry alone, which is why a zero `ttl` has to clear
    one rather than skip the call: modelled here so a `retain='forever'` budget that
    kept a stale horizon shows up as a failure.
    """

    _FIELDS = ('usd_nanos', 'tokens', 'requests', 'unpriced')

    def __init__(self, *, bytes_keys: bool = False) -> None:
        self.hashes: dict[str, dict[str, int]] = {}
        self.expiries: dict[str, int] = {}
        self.markers: dict[str, int] = {}
        self.calls: list[str] = []
        self._bytes_keys = bytes_keys

    def _out(self, value: str) -> str | bytes:
        return value.encode() if self._bytes_keys else value

    async def hgetall(self, name: str) -> Mapping[str | bytes, str | bytes]:
        return {self._out(field): self._out(str(value)) for field, value in self.hashes.get(name, {}).items()}

    async def eval(self, script: str, numkeys: int, *keys_and_args: str | int) -> Sequence[Sequence[str | bytes]]:
        self.calls.append(script)
        keys = [str(key) for key in keys_and_args[:numkeys]]
        arguments = [int(argument) for argument in keys_and_args[numkeys:]]
        count = arguments[0]
        rows: list[Sequence[str | bytes]] = []
        for index in range(count):
            key, marker = keys[index], keys[count + index]
            usd, tokens, requests, unpriced, ttl, marker_ttl = arguments[1 + index * 6 : 7 + index * 6]
            if marker_ttl <= 0 or marker not in self.markers:
                fields = self.hashes.setdefault(key, {})
                for field, amount in zip(self._FIELDS, (usd, tokens, requests, unpriced)):
                    fields[field] = fields.get(field, 0) + amount
                if ttl > 0:
                    self.expiries[key] = ttl
                else:
                    self.expiries.pop(key, None)
                # Written after the increments, not claimed before them: an increment that
                # errors must not leave a marker that would skip the retry.
                if marker_ttl > 0:
                    self.markers[marker] = marker_ttl
            totals = self.hashes.get(key, {})
            rows.append([self._out(str(totals.get(field, 0))) for field in self._FIELDS])
        return rows


class TestRedisStore:
    """A counter several workers share, without a Redis dependency."""

    async def test_an_absent_hash_reads_as_zero(self):
        assert await RedisSpendStore(FakeRedis()).get('k') == Spent()

    @pytest.mark.parametrize('bytes_keys', [False, True])
    async def test_a_round_trip_keeps_the_exact_amount(self, bytes_keys: bool):
        client = FakeRedis(bytes_keys=bytes_keys)
        store = RedisSpendStore(client)

        added = await store.add('k', usd=Decimal('0.000123456'), tokens=7, requests=1, unpriced=1, ttl=None)
        assert added == Spent(usd=Decimal('0.000123456'), tokens=7, requests=1, unpriced_requests=1)
        assert await store.get('k') == added

    async def test_repeated_adds_do_not_drift(self):
        """A price with a fractional sub-unit, since a whole one cannot detect rounding at all."""
        store = RedisSpendStore(FakeRedis())
        price = Decimal('0.000000675')  # a cheap model's real per-request cost
        for _ in range(100_000):
            await store.add('k', usd=price, tokens=0, requests=1, unpriced=0, ttl=None)

        assert (await store.get('k')).usd == price * 100_000

    async def test_a_ttl_is_applied_and_the_key_is_namespaced(self):
        client = FakeRedis()
        store = RedisSpendStore(client, prefix='acme')
        await store.add('k', usd=Decimal('1'), tokens=1, requests=1, unpriced=0, ttl=timedelta(hours=2))

        assert client.expiries == {'{acme}:k': 7200}

    async def test_every_key_carries_the_prefix_as_a_hash_tag(self):
        """A cluster hashes a tagged key by the tag alone, which is what puts a store's keys in one slot.

        Untagged, a day and a month window land in different slots and the script that
        applies them together is refused with `CROSSSLOT`.
        """
        client = FakeRedis()
        store = RedisSpendStore(client, prefix='acme')
        await store.add_many(
            [
                SpendEntry(key='b|day|*|2026-08-04', requests=1, token='r1'),
                SpendEntry(key='b|month|*|2026-08', requests=1, token='r1'),
            ]
        )

        assert set(client.hashes) == {'{acme}:b|day|*|2026-08-04', '{acme}:b|month|*|2026-08'}
        assert all(marker.startswith('{acme}:|dedup|') for marker in client.markers)

    @pytest.mark.parametrize('prefix', ['{acme}', ''])
    def test_a_prefix_that_would_break_the_hash_tag_is_refused(self, prefix: str):
        """A brace moves the tag; an empty prefix leaves `{}`, which is not a tag at all.

        Redis Cluster reads `{}` as no tag and hashes the whole key, so two windows land in
        different slots and the script that applies them together is refused.
        """
        with pytest.raises(UserError, match='must be non-empty and must not contain braces'):
            RedisSpendStore(FakeRedis(), prefix=prefix)

    async def test_a_response_is_one_round_trip_and_one_unit_of_work(self):
        """Split across commands, a failure between them leaves a window holding part of a response."""
        client = FakeRedis()
        await RedisSpendStore(client).add('k', usd=Decimal('1'), tokens=1, requests=1, unpriced=0, ttl=None)

        assert len(client.calls) == 1
        assert 'HINCRBY' in client.calls[0]
        assert 'EXPIRE' in client.calls[0]

    async def test_every_window_of_a_response_is_one_script(self):
        """The point of `add_many`: a failure cannot leave the day counted and the month not."""
        client = FakeRedis()
        totals = await RedisSpendStore(client).add_many(
            [
                SpendEntry(key='day', usd=Decimal('1'), tokens=5, requests=1, ttl=timedelta(hours=48)),
                SpendEntry(key='month', usd=Decimal('1'), tokens=5, requests=1, ttl=timedelta(days=62)),
            ]
        )

        assert len(client.calls) == 1
        assert totals == {
            'day': Spent(usd=Decimal('1'), tokens=5, requests=1),
            'month': Spent(usd=Decimal('1'), tokens=5, requests=1),
        }
        assert client.expiries == {
            '{pydantic-ai-harness:spend}:day': 172_800,
            '{pydantic-ai-harness:spend}:month': 5_356_800,
        }

    async def test_nothing_to_apply_is_not_a_round_trip(self):
        client = FakeRedis()

        assert await RedisSpendStore(client).add_many([]) == {}
        assert client.calls == []

    async def test_a_total_past_lua_s_exact_integer_range_stays_exact(self):
        """The counters are read back as strings, so nothing in the path is a double.

        Redis stores them exactly whatever happens: `HINCRBY` is 64-bit integer
        arithmetic and the increment arrives as a string. It is the *reply* that would
        round, once a Lua number holds it.
        """
        client = FakeRedis()
        client.hashes['{pydantic-ai-harness:spend}:k'] = {'usd_nanos': 2**53, 'tokens': 0, 'requests': 0, 'unpriced': 0}
        store = RedisSpendStore(client)

        added = await store.add('k', usd=Decimal('0.000000001'), tokens=0, requests=1, unpriced=0, ttl=None)

        assert added.usd == Decimal('9007199.254740993')
        assert (await store.get('k')).usd == Decimal('9007199.254740993')

    async def test_a_repeated_token_is_applied_once(self):
        """A durable engine re-executing the accrual hands back the same response."""
        store = RedisSpendStore(FakeRedis())
        entry = SpendEntry(key='k', usd=Decimal('1'), tokens=5, requests=1, token='resp-1')

        first = await store.add_many([entry])
        second = await store.add_many([entry])

        assert first == second == {'k': Spent(usd=Decimal('1'), tokens=5, requests=1)}

    async def test_a_different_token_is_applied(self):
        """The marker is per response, not a lock on the key."""
        store = RedisSpendStore(FakeRedis())

        await store.add_many([SpendEntry(key='k', usd=Decimal('1'), requests=1, token='resp-1')])
        totals = await store.add_many([SpendEntry(key='k', usd=Decimal('1'), requests=1, token='resp-2')])

        assert totals == {'k': Spent(usd=Decimal('2'), requests=2)}

    async def test_a_token_is_scoped_to_its_window(self):
        """One response reaches several windows, so a marker one window claimed cannot cover another."""
        store = RedisSpendStore(FakeRedis())

        await store.add_many([SpendEntry(key='day', usd=Decimal('1'), requests=1, token='resp-1')])
        totals = await store.add_many([SpendEntry(key='month', usd=Decimal('1'), requests=1, token='resp-1')])

        assert totals == {'month': Spent(usd=Decimal('1'), requests=1)}

    async def test_a_marker_is_held_no_longer_than_its_counter(self):
        """A marker outliving its window would skip a replay against a counter that rolled over,
        and the window would read as zero rather than as the response it should hold."""
        client = FakeRedis()
        store = RedisSpendStore(client, dedup_retain=timedelta(hours=24))

        await store.add_many([SpendEntry(key='k', requests=1, ttl=timedelta(hours=1), token='resp-1')])

        assert set(client.markers.values()) == {3600}
        assert client.expiries == {'{pydantic-ai-harness:spend}:k': 3600}

    async def test_dedup_can_be_turned_off(self):
        """A deployment that would rather not hold a marker per response can say so."""
        store = RedisSpendStore(FakeRedis(), dedup_retain=None)
        entry = SpendEntry(key='k', usd=Decimal('1'), requests=1, token='resp-1')

        await store.add_many([entry])
        totals = await store.add_many([entry])

        assert totals == {'k': Spent(usd=Decimal('2'), requests=2)}

    async def test_an_entry_with_no_token_is_always_applied(self):
        """A reconciler posting the same correction twice means it twice."""
        store = RedisSpendStore(FakeRedis())
        entry = SpendEntry(key='k', usd=Decimal('-1'))

        await store.add_many([entry])
        totals = await store.add_many([entry])

        assert totals == {'k': Spent(usd=Decimal('-2'))}

    async def test_a_counter_written_before_the_hash_tag_is_still_read(self):
        """Nothing an upgrade strands: the old key is read when the tagged one is absent."""
        client = FakeRedis()
        client.hashes['pydantic-ai-harness:spend:k'] = {'usd_nanos': 3_000_000_000, 'tokens': 8, 'requests': 2}

        assert await RedisSpendStore(client).get('k') == Spent(usd=Decimal('3'), tokens=8, requests=2)

    async def test_a_counter_written_before_the_hash_tag_is_added_to_the_one_after_it(self):
        """The old name is read alongside the new one, so an upgrade counts both."""
        client = FakeRedis()
        client.hashes['pydantic-ai-harness:spend:k'] = {'usd_nanos': 3_000_000_000, 'tokens': 8, 'requests': 2}
        store = RedisSpendStore(client)

        totals = await store.add_many([SpendEntry(key='k', usd=Decimal('1'), tokens=1, requests=1)])

        assert totals == {'k': Spent(usd=Decimal('4'), tokens=9, requests=3)}
        assert await store.get('k') == Spent(usd=Decimal('4'), tokens=9, requests=3)

    async def test_the_old_counter_is_never_added_twice(self):
        """Summed rather than moved, so repeating the read cannot repeat the amount."""
        client = FakeRedis()
        client.hashes['pydantic-ai-harness:spend:k'] = {'usd_nanos': 3_000_000_000, 'requests': 1}
        store = RedisSpendStore(client)

        await store.add_many([SpendEntry(key='k', usd=Decimal('1'), requests=1)])
        totals = await store.add_many([SpendEntry(key='k', usd=Decimal('1'), requests=1)])

        assert totals == {'k': Spent(usd=Decimal('5'), requests=3)}

    async def test_a_write_to_the_old_name_after_the_new_one_exists_still_counts(self):
        """A rolling deploy leaves workers on an earlier release writing the old name for a while.

        Moving the old counter once would have read it before that write and never looked
        again, so the spend a still-running old worker recorded would be enforced against
        nothing.
        """
        client = FakeRedis()
        store = RedisSpendStore(client)
        await store.add_many([SpendEntry(key='k', usd=Decimal('1'), requests=1)])

        client.hashes['pydantic-ai-harness:spend:k'] = {'usd_nanos': 2_000_000_000, 'requests': 1}

        assert await store.get('k') == Spent(usd=Decimal('3'), requests=2)
        totals = await store.add_many([SpendEntry(key='k', usd=Decimal('1'), requests=1)])
        assert totals == {'k': Spent(usd=Decimal('4'), requests=3)}

    async def test_one_key_twice_in_a_call_keeps_the_old_counter(self):
        """Two entries on one key report the running total, and neither loses the old name."""
        client = FakeRedis()
        client.hashes['pydantic-ai-harness:spend:k'] = {'usd_nanos': 3_000_000_000, 'requests': 1}
        store = RedisSpendStore(client)

        totals = await store.add_many(
            [
                SpendEntry(key='k', usd=Decimal('1'), requests=1),
                SpendEntry(key='k', usd=Decimal('1'), requests=1),
            ]
        )

        assert totals == {'k': Spent(usd=Decimal('5'), requests=3)}

    async def test_two_keys_that_would_share_a_marker_are_both_applied(self):
        """A key and a token joined on a separator collide; length-prefixed they cannot.

        `key='a|b'` with `token='c'` and `key='a'` with `token='b|c'` are different
        responses against different windows, and the second was dropped as a replay.
        """
        store = RedisSpendStore(FakeRedis())

        await store.add_many([SpendEntry(key='a|b', usd=Decimal('1'), requests=1, token='c')])
        totals = await store.add_many([SpendEntry(key='a', usd=Decimal('1'), requests=1, token='b|c')])

        assert totals == {'a': Spent(usd=Decimal('1'), requests=1)}

    @pytest.mark.parametrize(
        ('retain', 'expected'),
        [
            (timedelta(milliseconds=500), 1),
            (timedelta(seconds=1), 1),
            (timedelta(seconds=90), 90),
            # Rounding on `total_seconds()` truncated the remainder to milliseconds before
            # taking the ceiling, so a horizon with a finer tail expired early.
            (timedelta(seconds=1, microseconds=1), 2),
            (timedelta(seconds=2, microseconds=999), 3),
        ],
    )
    async def test_a_horizon_is_rounded_up_never_down(self, retain: timedelta, expected: int):
        """`EXPIRE` takes seconds and the script reads zero as "keep"; rounding down would never expire."""
        client = FakeRedis()
        await RedisSpendStore(client).add('k', usd=Decimal('1'), tokens=1, requests=1, unpriced=0, ttl=retain)

        assert client.expiries == {'{pydantic-ai-harness:spend}:k': expected}

    async def test_moving_a_budget_to_forever_clears_the_old_horizon(self):
        """`HINCRBY` leaves an expiry in place, so skipping `EXPIRE` is not the same as no expiry.

        A counter written under a finite `retain` and then reconfigured to `'forever'` kept
        expiring on the horizon nothing in the configuration mentions any more, handing the
        ceiling back on a schedule. `InMemorySpendStore` drops the expiry on the next write;
        this is the same behavior.
        """
        client = FakeRedis()
        store = RedisSpendStore(client)
        await store.add('k', usd=Decimal('1'), tokens=1, requests=1, unpriced=0, ttl=timedelta(hours=1))
        assert client.expiries == {'{pydantic-ai-harness:spend}:k': 3600}

        await store.add('k', usd=Decimal('1'), tokens=1, requests=1, unpriced=0, ttl=None)

        assert client.expiries == {}
        assert 'PERSIST' in client.calls[-1]

    async def test_it_drives_the_gate(self):
        guard = SpendLimits(
            budgets=[Budget(usd=Decimal('0.02'), window='day')],
            store=RedisSpendStore(FakeRedis()),
            price=lambda r: Decimal('0.01'),
        )
        agent = _agent(guard)

        await agent.run('hi')
        await agent.run('hi')
        with pytest.raises(SpendLimitExceeded):
            await agent.run('hi')

    async def test_a_budget_named_for_the_marker_sentinel_still_counts(self):
        """A marker's name starts with the separator, which no budget key can.

        `store_key` is `name|window|scope|bucket` and `Budget.name` is refused empty, so a
        budget key never begins with one. That is what keeps counters and markers from naming
        each other in the namespace they share, and it means no configuration is off limits.
        """
        guard = SpendLimits(
            budgets=[Budget(usd=Decimal('10'), window='day', name='dedup')],
            store=RedisSpendStore(FakeRedis()),
            price=lambda r: Decimal('1'),
        )

        await _record(guard)

        spent = (await guard.status())[0].spent
        assert (spent.usd, spent.requests) == (Decimal('1'), 1)

    async def test_a_marker_cannot_be_named_by_a_budget_key(self):
        """The separator prefix, asserted on the names themselves rather than on a symptom."""
        client = FakeRedis()
        store = RedisSpendStore(client)

        await store.add_many([SpendEntry(key='dedup|1:a|1:b', usd=Decimal('1'), requests=1, token='t')])

        counters = [name for name in client.hashes]
        markers = list(client.markers)
        assert counters == ['{pydantic-ai-harness:spend}:dedup|1:a|1:b']
        assert markers == ['{pydantic-ai-harness:spend}:|dedup|13:dedup|1:a|1:b|1:t']
        assert not set(counters) & set(markers)

    async def test_an_application_s_own_decimal_precision_does_not_round_the_total(self):
        """The counter comes back exact, and merging the pre-hash-tag one must keep it that way."""
        client = FakeRedis()
        store = RedisSpendStore(client)
        client.hashes[f'{store.prefix}:k'] = {'usd_nanos': 1, 'requests': 1}

        with localcontext() as context:
            context.prec = 3
            added = await store.add_many([SpendEntry(key='k', usd=Decimal('0.000123456'), requests=1)])
            read = await store.get_many(['k'])

        assert added['k'].usd == Decimal('0.000123457')
        assert read['k'].usd == Decimal('0.000123457')


class _LegacyStore:
    """A store written against the released `SpendStore`: one key per call, and no token."""

    def __init__(self) -> None:
        self.writes: list[str] = []
        self._counters = InMemorySpendStore()

    async def get(self, key: str) -> Spent:
        return await self._counters.get(key)

    async def add(
        self,
        key: str,
        *,
        usd: Decimal,
        tokens: int,
        requests: int,
        unpriced: int,
        ttl: timedelta | None,
    ) -> Spent:
        self.writes.append(key)
        return await self._counters.add(key, usd=usd, tokens=tokens, requests=requests, unpriced=unpriced, ttl=ttl)


class TestBatchAccrual:
    """One response reaches every window it counts against as one unit of work."""

    async def test_every_window_is_one_call(self):
        """Applied one at a time, a failure between them left the day counted and the month not."""
        calls: list[Sequence[str]] = []

        class Counting(InMemorySpendStore):
            async def add_many(self, entries: Sequence[SpendEntry]) -> Mapping[str, Spent]:
                calls.append([entry.key for entry in entries])
                return await super().add_many(entries)

        guard = SpendLimits(
            budgets=[Budget(usd=Decimal('100'), window='day'), Budget(usd=Decimal('500'), window='month')],
            store=Counting(),
            price=lambda r: Decimal('1'),
        )
        await _record(guard)

        assert [len(keys) for keys in calls] == [2]
        assert [status.spent.usd for status in await guard.status()] == [Decimal('1'), Decimal('1')]

    async def test_windows_sharing_a_counter_are_one_entry(self):
        """A USD and a token ceiling on one window still add the response once."""
        calls: list[Sequence[str]] = []

        class Counting(InMemorySpendStore):
            async def add_many(self, entries: Sequence[SpendEntry]) -> Mapping[str, Spent]:
                calls.append([entry.key for entry in entries])
                return await super().add_many(entries)

        guard = SpendLimits(
            budgets=[Budget(usd=Decimal('10'), window='day'), Budget(tokens=99_999, window='day')],
            store=Counting(),
            price=lambda r: Decimal('1'),
        )
        await _record(guard)

        assert [len(keys) for keys in calls] == [1]
        assert (await guard.status())[0].spent.usd == Decimal('1')


class TestIdempotentAccrual:
    """A durable engine re-executes the hooks around a response it already holds."""

    async def test_a_response_the_provider_identified_survives_a_new_run_id(self):
        """DBOS recovery and a Prefect flow retry replay the response under a fresh `run_id`."""
        guard = SpendLimits(budgets=[Budget(window='total')], price=lambda r: Decimal('1'))
        response = _response(provider_response_id='resp-1')

        await _record(guard, ctx=_run_ctx(run_id='first'), response=response)
        await _record(guard, ctx=_run_ctx(run_id='second'), response=response)

        assert (await guard.status())[0].spent == Spent(usd=Decimal('1'), tokens=1100, requests=1)

    async def test_a_response_with_no_provider_id_needs_a_stable_run_id(self):
        """The documented limit: without a response id the token falls back to the run's own.

        `_agent_graph.resolve_run_id` honours a `run_id` the caller passes, which is how a
        replayed accrual stays idempotent for a provider that reports none. A fresh id is a
        different response as far as the token can tell.
        """
        guard = SpendLimits(budgets=[Budget(window='total')], price=lambda r: Decimal('1'))
        response = _response()

        await _record(guard, ctx=_run_ctx(run_id='same'), response=response)
        await _record(guard, ctx=_run_ctx(run_id='same'), response=response)
        await _record(guard, ctx=_run_ctx(run_id='other'), response=response)

        assert (await guard.status())[0].spent == Spent(usd=Decimal('2'), tokens=2200, requests=2)

    async def test_an_empty_provider_response_id_is_not_an_identity(self):
        """`ChatCompletion.id` is a plain required `str`, so a server can answer `""`.

        Read for presence rather than truth, every response from that provider names the
        same token and everything after the first is dropped as a replay of it -- the brake
        releasing late, which is the direction this capability exists to avoid. Core reads
        the same field for truth (`models/_continuation.py`, `models/openai.py`).
        """
        guard = SpendLimits(budgets=[Budget(window='total')], price=lambda r: Decimal('1'))

        for _ in range(3):
            await _record(guard, response=_response(provider_response_id=''))

        assert (await guard.status())[0].spent.requests == 3

    async def test_a_provider_that_repeats_one_id_does_not_collapse_its_responses(self):
        """The response timestamp joins the id, so a broken id does not stand alone.

        A server returning a constant `id` would otherwise have everything after the first
        response dropped as a replay of it, which is spend the ceiling never sees. A genuine
        replay hands back the response object it checkpointed, so its timestamp is unchanged
        and `test_a_response_the_provider_identified_survives_a_new_run_id` still holds.
        """
        guard = SpendLimits(budgets=[Budget(window='total')], price=lambda r: Decimal('1'))

        for _ in range(3):
            await _record(guard, response=_response(provider_response_id='resp-const'))

        assert (await guard.status())[0].spent.requests == 3

    async def test_two_responses_of_one_run_both_count(self):
        """The marker identifies a response, so it must not swallow the next one."""
        guard = SpendLimits(budgets=[Budget(window='total')], price=lambda r: Decimal('1'))

        await _record(guard, response=_response(provider_response_id='resp-1'))
        await _record(guard, response=_response(provider_response_id='resp-2'))

        assert (await guard.status())[0].spent.requests == 2

    async def test_two_responses_that_would_share_a_token_both_count(self):
        """A provider name and a response id joined on a separator can collide.

        `provider_name='a|b'` with id `'c'` and `provider_name='a'` with id `'b|c'` named
        the same response, so the second one's spend was dropped as a replay of the first.
        """
        guard = SpendLimits(budgets=[Budget(window='total')], price=lambda r: Decimal('1'))

        await _record(guard, response=_response(provider_name='a|b', provider_response_id='c'))
        await _record(guard, response=_response(provider_name='a', provider_response_id='b|c'))

        assert (await guard.status())[0].spent.requests == 2

    async def test_the_same_id_from_two_providers_is_two_responses(self):
        """Nothing makes a provider's request id unique across providers."""
        guard = SpendLimits(budgets=[Budget(window='total')], price=lambda r: Decimal('1'))

        await _record(guard, response=_response(provider_name='openai', provider_response_id='1'))
        await _record(guard, response=_response(provider_name='anthropic', provider_response_id='1'))

        assert (await guard.status())[0].spent.requests == 2


class TestDeprecatedStore:
    """A store written against the released `SpendStore` keeps working, and says what it costs."""

    def test_it_warns_once_naming_what_is_lost(self):
        """`HarnessDeprecationWarning` rather than `DeprecationWarning`, which is the whole warning.

        Python filters `DeprecationWarning` out by default unless it is triggered from
        `__main__`, and this one is raised inside the package, so a library caller would
        see nothing at all. `HarnessDeprecationWarning` derives from `UserWarning`, which
        is shown by default and which the docs give one recipe for silencing.
        """
        with pytest.warns(HarnessDeprecationWarning) as warned:
            SpendLimits[None](budgets=[Budget(window='total')], store=_LegacyStore())

        assert len(warned) == 1
        message = str(warned[0].message)
        assert 'one window at a time' in message
        assert 'removed in 0.28.0' in message

    def test_a_batch_store_is_not_warned_about(self):
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            SpendLimits[None](budgets=[Budget(window='total')], store=InMemorySpendStore())

    async def test_it_still_accrues_and_gates(self):
        with pytest.warns(HarnessDeprecationWarning):
            guard = SpendLimits(
                budgets=[Budget(usd=Decimal('0.02'), window='day')],
                store=_LegacyStore(),
                price=lambda r: Decimal('0.01'),
            )
        agent = _agent(guard)

        await agent.run('hi')
        await agent.run('hi')
        with pytest.raises(SpendLimitExceeded):
            await agent.run('hi')

    async def test_it_is_driven_one_window_per_call(self):
        """Which is what it is warned about: two windows are two writes, not one."""
        store = _LegacyStore()
        with pytest.warns(HarnessDeprecationWarning):
            guard = SpendLimits(
                budgets=[Budget(window='day'), Budget(window='month')],
                store=store,
                price=lambda r: Decimal('1'),
            )
        await _record(guard)

        assert len(store.writes) == 2

    async def test_a_replayed_response_counts_twice(self):
        """`SpendEntry.token` has nowhere to go on a store that never sees it."""
        with pytest.warns(HarnessDeprecationWarning):
            guard = SpendLimits(budgets=[Budget(window='total')], store=_LegacyStore(), price=lambda r: Decimal('1'))
        response = _response(provider_response_id='resp-1')

        await _record(guard, response=response)
        await _record(guard, response=response)

        assert (await guard.status())[0].spent.requests == 2


class TestUnreachableOverrides:
    """A subclass that overrode the single-key pair is told it is no longer on the path."""

    def test_a_single_key_override_is_reported(self):
        """`SpendLimits` drives `add_many`, so an audit bolted onto `add` stops running.

        Before the batch pair existed that override *was* the path, so a subclass upgrading
        into this loses whatever it added and gets nothing back saying so.
        """

        class Audited(InMemorySpendStore):
            async def add(
                self,
                key: str,
                *,
                usd: Decimal,
                tokens: int,
                requests: int,
                unpriced: int,
                ttl: timedelta | None,
            ) -> Spent:
                raise AssertionError('never reached, which is what the warning is about')  # pragma: no cover

        with pytest.warns(HarnessDeprecationWarning, match='never called'):
            Audited()

    async def test_a_subclass_that_moved_to_the_batch_pair_is_silent(self):
        """And its batch override really is driven, which is what makes the move the fix."""
        applied: list[int] = []

        class Moved(InMemorySpendStore):
            async def add(
                self,
                key: str,
                *,
                usd: Decimal,
                tokens: int,
                requests: int,
                unpriced: int,
                ttl: timedelta | None,
            ) -> Spent:
                raise AssertionError('never reached, and no longer claimed to be')  # pragma: no cover

            async def add_many(self, entries: Sequence[SpendEntry]) -> Mapping[str, Spent]:
                applied.append(len(entries))
                return await super().add_many(entries)

        with warnings.catch_warnings():
            warnings.simplefilter('error')
            store = Moved()

        await store.add_many([SpendEntry(key='k', usd=Decimal('1'), requests=1)])

        assert applied == [1]

    def test_the_stores_themselves_are_silent(self):
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            InMemorySpendStore()
            RedisSpendStore(FakeRedis())

    def test_the_redis_store_reports_it_too(self):
        class Mirrored(RedisSpendStore):
            async def get(self, key: str) -> Spent:
                raise AssertionError('never reached, which is what the warning is about')  # pragma: no cover

        with pytest.warns(HarnessDeprecationWarning, match='never called'):
            Mirrored(FakeRedis())


class TestReportedPrecision:
    """What a caller is told stays exact whatever precision the application set."""

    async def test_a_lowered_precision_does_not_round_what_status_reports(self):
        """`Spent.usd` and the two numbers derived from it are pinned together.

        Rounded, a `warn_at` crossing reports the wrong side of itself and `remaining_usd`
        contradicts the `spent` on the same dataclass. Enforcement is unaffected -- `_check`
        compares directly and rounding cannot cross zero -- but the reading ships wrong.
        """
        store = InMemorySpendStore()
        guard = SpendLimits[None](budgets=[Budget(usd=Decimal('1234.56789'), window='total', warn_at=0.8)], store=store)
        await store.add_many([SpendEntry(key=(await guard.status())[0].key, usd=Decimal('987.654321'), requests=1)])

        with localcontext() as context:
            context.prec = 4
            status = (await guard.status())[0]

        assert status.spent.usd == Decimal('987.654321')
        assert status.remaining_usd == Decimal('246.913569')
        assert status.warning is True


class TestToolset:
    """The agent-facing tool is off unless asked for."""

    def test_no_toolset_by_default(self):
        assert SpendLimits[None]().get_toolset() is None

    async def test_the_tool_is_offered_to_the_model(self):
        offered: list[list[str]] = []

        def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            offered.append(sorted(tool.name for tool in info.function_tools))
            return ModelResponse(parts=[TextPart(content='done')])

        guard = SpendLimits(budgets=[Budget(window='total')], expose_tools=True)
        agent = Agent(FunctionModel(respond), deps_type=type(None), capabilities=[guard])
        await agent.run('hi')

        assert offered == [['get_spend']]

    async def test_the_tool_reports_each_budget(self):
        guard = SpendLimits[None](
            budgets=[Budget(name='daily', usd=Decimal('10'), window='day'), Budget(name='lifetime', window='total')],
            price=lambda r: Decimal('2'),
            expose_tools=True,
        )

        report = await _call_get_spend(guard)

        assert 'daily (day): $2 spent, $8 left' in report
        assert 'lifetime (total): $2 spent, no limit' in report

    async def test_the_tool_says_so_when_nothing_is_budgeted(self):
        assert await _call_get_spend(SpendLimits[None](expose_tools=True)) == 'No budgets are configured.'


class TestExhaustedGate:
    """The pre-flight check a durable workflow makes, which must not pass by inspecting nothing."""

    async def test_it_refuses_to_answer_about_a_budget_it_cannot_read(self):
        guard = SpendLimits(
            budgets=[Budget(usd=Decimal('5'), scope=lambda ctx: str(ctx.deps), name='tenant')],
            price=lambda r: Decimal('1'),
        )

        with pytest.raises(UserError, match='would pass having inspected nothing'):
            await guard.exhausted()

    async def test_it_answers_when_the_scope_is_named(self):
        guard = SpendLimits(
            budgets=[Budget(usd=Decimal('0.5'), scope=lambda ctx: str(ctx.deps), name='tenant')],
            price=lambda r: Decimal('1'),
        )
        await _record(guard, ctx=_run_ctx(deps='acme'))

        assert await guard.exhausted(scope='acme') is True
        assert await guard.exhausted(scope='other') is False

    async def test_status_still_reports_what_it_can(self):
        """The lenient reading a cost display wants stays lenient."""
        guard = SpendLimits(
            budgets=[Budget(usd=Decimal('5'), scope=lambda ctx: str(ctx.deps), name='tenant')],
            price=lambda r: Decimal('1'),
        )

        assert await guard.status() == ()


class TestNegativeCells:
    """Each of these asserted only its positive half, leaving the mutant that inverts it alive."""

    async def test_raise_lets_a_priced_response_through(self):
        """`on_unpriced='raise'` is about the unpriced ones; dropping `not priced` failed every run."""
        guard = SpendLimits[None](
            budgets=[Budget(usd=Decimal('100'))],
            price=lambda response: Decimal('1'),
            on_unpriced='raise',
        )

        result = await _agent(guard).run('hi')

        assert result.output == 'ok'

    async def test_a_scope_stays_out_of_the_span_without_content(self):
        """Dropping the `trace_include_content` half puts a tenant id in every trace."""
        tracer, exporter = _recording_tracer()
        guard = SpendLimits(
            budgets=[Budget(usd=Decimal('0.001'), scope=lambda ctx: str(ctx.deps))],
            price=lambda r: Decimal('1'),
        )
        await _record(guard, ctx=_run_ctx(deps='acme'))

        with pytest.raises(SpendLimitExceeded):
            await _gate(guard, ctx=_run_ctx(deps='acme', tracer=tracer))

        assert 'spend.scope' not in dict(_only_span(exporter).attributes or {})

    async def test_a_budget_below_its_warning_fraction_does_not_warn(self):
        """Only the crossed case was asserted, so `warning=True` for every budget also passed."""
        guard = SpendLimits(budgets=[Budget(usd=Decimal('100'), warn_at=0.8)], price=lambda r: Decimal('1'))
        await _record(guard)

        [status] = await guard.status()

        assert status.warning is False

    async def test_a_budget_over_its_warning_fraction_warns(self):
        guard = SpendLimits(budgets=[Budget(usd=Decimal('100'), warn_at=0.8)], price=lambda r: Decimal('90'))
        await _record(guard)

        [status] = await guard.status()

        assert status.warning is True


class TestDuplicateBudgets:
    """Sharing a counter is a feature; two ceilings of one kind on it is not."""

    def test_two_budgets_with_the_same_usd_ceiling_slot_are_refused(self):
        """They read as independent limits and behave as the smaller one."""
        with pytest.raises(UserError, match='would share one counter'):
            SpendLimits[None](budgets=[Budget(usd=Decimal('5')), Budget(usd=Decimal('100'))])

    def test_a_collision_a_later_budget_displaced_is_still_refused(self):
        """Remembering one budget per slot missed any collision the next budget overwrote."""
        with pytest.raises(UserError, match='would share one counter'):
            SpendLimits[None](
                budgets=[
                    Budget(usd=Decimal('100'), window='day'),
                    Budget(usd=Decimal('2000'), window='month'),
                    Budget(usd=Decimal('5'), window='day'),
                ]
            )

    def test_two_different_scope_callables_under_one_name_are_refused(self):
        """Two lambdas are two dimensions, and nothing stops them returning the same string."""
        with pytest.raises(UserError, match='different `scope` callables'):
            SpendLimits[None](
                budgets=[
                    Budget(usd=Decimal('5'), window='day', scope=lambda ctx: 'acme', name='tenant'),
                    Budget(usd=Decimal('100'), window='day', scope=lambda ctx: 'acme', name='tenant'),
                ],
            )

    def test_two_scope_dimensions_sharing_a_name_are_refused_across_metrics(self):
        """A USD tenant budget and a token user budget merge whenever the two ids coincide."""
        with pytest.raises(UserError, match='different `scope` callables'):
            SpendLimits[None](
                budgets=[
                    Budget(usd=Decimal('5'), window='day', scope=lambda ctx: 'acme', name='shared'),
                    Budget(tokens=1000, window='day', scope=lambda ctx: 'acme', name='shared'),
                ],
            )

    async def test_one_scope_callable_may_carry_both_ceilings(self):
        """Sharing the callable is how a scoped window deliberately shares its counter."""

        def scope(ctx: RunContext[Any]) -> str:
            return str(ctx.deps)

        guard = SpendLimits(
            budgets=[
                Budget(usd=Decimal('5'), window='day', scope=scope, name='tenant'),
                Budget(tokens=1000, window='day', scope=scope, name='tenant'),
            ],
            price=lambda r: Decimal('1'),
        )
        await _record(guard, ctx=_run_ctx(deps='acme'))

        usd_budget, token_budget = await guard.status(scope='acme')
        assert usd_budget.key == token_budget.key
        assert usd_budget.spent.usd == Decimal('1')
        assert token_budget.spent.tokens == 1100

    def test_a_usd_and_a_token_ceiling_may_share_a_counter(self):
        guard = SpendLimits[None](budgets=[Budget(usd=Decimal('5')), Budget(tokens=100)])

        assert len(guard.budgets) == 2

    def test_different_names_keep_them_apart(self):
        guard = SpendLimits[None](
            budgets=[Budget(usd=Decimal('5'), name='tight'), Budget(usd=Decimal('100'), name='loose')]
        )

        assert len(guard.budgets) == 2

    def test_co_keyed_budgets_disagreeing_on_retain_are_refused(self):
        """One counter has one expiry, so declaration order would decide when it rolls over."""
        with pytest.raises(UserError, match='different `retain` values'):
            SpendLimits[None](
                budgets=[
                    Budget(usd=Decimal('5'), window='conversation', name='cap'),
                    Budget(tokens=1000, window='conversation', name='cap', retain='forever'),
                ]
            )

    def test_co_keyed_budgets_agreeing_on_retain_are_allowed(self):
        guard = SpendLimits[None](
            budgets=[
                Budget(usd=Decimal('5'), window='conversation', name='cap', retain='forever'),
                Budget(tokens=1000, window='conversation', name='cap', retain='forever'),
            ]
        )

        assert len(guard.budgets) == 2

    def test_the_same_name_on_different_windows_may_differ_in_retain(self):
        """Different windows are different counters, so neither decides the other's expiry."""
        guard = SpendLimits[None](
            budgets=[
                Budget(usd=Decimal('5'), window='day', name='cap'),
                Budget(usd=Decimal('50'), window='month', name='cap', retain='forever'),
            ]
        )

        assert len(guard.budgets) == 2

    def test_deferred_loading_is_refused(self):
        """Both hooks are skipped until the model loads it, so the brake would not be holding."""
        with pytest.raises(UserError, match='`defer_loading` is not supported'):
            SpendLimits[None](budgets=[Budget(usd=Decimal('5'))], defer_loading=True, id='spend')


class TestSpec:
    """`Agent.from_spec` covers the fields a spec can express, and refuses the rest."""

    def test_the_spec_name_is_pinned(self):
        """Declared rather than inherited, so renaming the class cannot move the spec API."""
        assert SpendLimits.get_serialization_name() == 'SpendLimits'

    def test_budgets_are_built_from_mappings(self):
        guard = SpendLimits[None].from_spec(
            budgets=[{'usd': '100', 'window': 'day'}, {'tokens': 5000, 'name': 'tokens'}],
            on_unpriced='raise',
        )

        assert guard.budgets[0] == Budget(usd=Decimal('100'), window='day')
        assert guard.budgets[1] == Budget(tokens=5000, name='tokens')
        assert guard.on_unpriced == 'raise'

    def test_a_callable_field_is_refused(self):
        with pytest.raises(UserError, match=r"\['on_spend', 'price'\]"):
            SpendLimits[None].from_spec(price=_no_price, on_spend=print)

    def test_a_budget_must_be_a_mapping(self):
        with pytest.raises(UserError, match='must be a mapping'):
            SpendLimits[None].from_spec(budgets=_spec_budgets('100'))

    def test_every_number_a_spec_carries_is_coerced(self):
        """Only `usd` was converted, so the others reached `__post_init__` and compared str to int."""
        guard = SpendLimits[None].from_spec(budgets=_spec_budgets({'usd': '1', 'tokens': '5000', 'warn_at': '0.8'}))

        assert guard.budgets[0] == Budget(usd=Decimal('1'), tokens=5000, warn_at=0.8)

    def test_a_number_that_is_not_a_number_is_named(self):
        with pytest.raises(UserError, match="budget 'tokens' is not a number"):
            SpendLimits[None].from_spec(budgets=_spec_budgets({'tokens': 'lots'}))

    def test_a_budget_scope_is_refused(self):
        with pytest.raises(UserError, match='cannot be expressed in a spec'):
            SpendLimits[None].from_spec(budgets=_spec_budgets({'scope': 'tenant'}))

    def test_an_unknown_spec_field_is_named(self):
        """`**unsupported` keeps the callable message specific, so it must not swallow the rest."""
        with pytest.raises(UserError, match=r"no spec field\(s\) \['budget'\]"):
            SpendLimits[None].from_spec(budget=[])

    def test_the_schema_carries_the_configuration_the_docs_document(self):
        """A `*args/**kwargs` signature published the bare name, marking every documented block invalid."""
        schema = json.dumps(AgentSpec.model_json_schema_with_capabilities([SpendLimits]), sort_keys=True)

        assert '"budgets"' in schema
        assert '"on_unpriced"' in schema
        assert '"expose_tools"' in schema
        # The four runtime-only fields take callables or a live store and have no spec form.
        for runtime_only in ('"store"', '"price"', '"on_spend"', '"clock"', '"scope"'):
            assert runtime_only not in schema, runtime_only

    def test_the_schema_describes_a_budget_rather_than_an_open_object(self):
        """A `Sequence[Budget]` cannot generate -- `scope` is a callable -- so entries take `BudgetSpec`."""
        definitions = AgentSpec.model_json_schema_with_capabilities([SpendLimits])['$defs']

        assert definitions['spec_params_SpendLimits']['properties']['budgets']['items'] == {
            '$ref': '#/$defs/BudgetSpec'
        }
        entry = definitions['BudgetSpec']
        assert set(entry['properties']) == {'usd', 'tokens', 'window', 'warn_at', 'name', 'retain'}
        # A price given as a string so YAML cannot round it through a float.
        assert {'type': 'string'} in entry['properties']['usd']['anyOf']
        assert entry['additionalProperties'] is False

    async def test_the_documented_yaml_loads_and_the_budget_it_names_holds(self):
        """The README's example, through the real loader rather than through `from_spec` directly."""
        agent = Agent.from_spec(
            {
                'model': 'test',
                'capabilities': [
                    {
                        'SpendLimits': {
                            'budgets': [
                                {'usd': '100', 'window': 'day'},
                                {'usd': '2000', 'window': 'month', 'warn_at': 0.8},
                            ],
                            'on_unpriced': 'raise',
                        }
                    }
                ],
            },
            custom_capability_types=[SpendLimits],
        )

        # `on_unpriced: raise` reached the capability: `TestModel` names no model, so
        # nothing can price its response.
        with pytest.raises(UnpricedModelError):
            await agent.run('hi')

    async def test_a_spec_budget_refuses_a_request_once_it_is_spent(self):
        """A ceiling small enough to trip, so the loaded budget is shown to be wired to the gate."""
        agent = Agent.from_spec(
            {
                'model': 'test',
                'capabilities': [{'SpendLimits': {'budgets': [{'tokens': 1, 'window': 'total'}]}}],
            },
            custom_capability_types=[SpendLimits],
        )
        await agent.run('hi')

        with pytest.raises(SpendLimitExceeded, match='tokens'):
            await agent.run('hi')


class TestDurableClock:
    """Temporal's workflow sandbox restricts `datetime.now`, which these hooks read."""

    async def test_a_restricted_clock_is_re_raised_naming_the_replay_problem(self):
        """The sandbox refuses a symptom; the message names the replay behind it.

        A caller told only to pass the module through would silence the error and get a counter
        Temporal replays. Matched by class name so the translation costs no `temporalio` import;
        the fake stands in for the real exception, which `tests/spend/test_temporal.py`
        exercises end to end.
        """

        class RestrictedWorkflowAccessError(Exception):
            pass

        def restricted() -> datetime:
            raise RestrictedWorkflowAccessError('Cannot access datetime.datetime.now.__call__')

        guard = SpendLimits[None](budgets=[Budget(usd=Decimal('5'))], clock=restricted)

        with pytest.raises(UserError, match='not safe to run inside a Temporal workflow'):
            await _gate(guard)
        with pytest.raises(UserError, match='exhausted'):
            await guard.status()

    async def test_any_other_clock_failure_is_left_alone(self):
        """Only the sandbox's refusal is translated; a broken clock still reports itself."""

        def broken() -> datetime:
            raise ZeroDivisionError('the clock is broken')

        guard = SpendLimits[None](budgets=[Budget(usd=Decimal('5'))], clock=broken)

        with pytest.raises(ZeroDivisionError, match='the clock is broken'):
            await _gate(guard)
