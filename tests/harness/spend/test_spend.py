"""Tests for the `spend` capability."""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any

import pytest
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import NoOpTracer, Tracer
from pydantic_ai import Agent
from pydantic_ai.capabilities import AbstractCapability, CapabilityOrdering
from pydantic_ai.exceptions import ModelRetry, UsageLimitExceeded, UserError
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models import ModelRequestContext, ModelRequestParameters
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import RunContext
from pydantic_ai.usage import RequestUsage, RunUsage

from pydantic_ai_harness.spend import (
    Budget,
    InMemorySpendStore,
    RedisSpendStore,
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
) -> RunContext[Any]:
    return RunContext(
        deps=deps,
        model=TestModel(),
        usage=RunUsage(),
        run_id=run_id,
        conversation_id=conversation_id,
        trace_include_content=trace_include_content,
        tracer=tracer if tracer is not None else NoOpTracer(),
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
) -> ModelResponse:
    return ModelResponse(
        parts=[TextPart(content='ok')],
        usage=RequestUsage(input_tokens=input_tokens, output_tokens=output_tokens),
        model_name=model_name,
        provider_name=provider_name,
    )


async def _record(guard: SpendLimits[Any], *, ctx: RunContext[Any] | None = None, **kwargs: Any) -> ModelResponse:
    response = _response(**kwargs)
    return await guard.after_model_request(
        ctx if ctx is not None else _run_ctx(),
        request_context=_request_context(),
        response=response,
    )


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
        reads: list[str] = []

        class Counting(InMemorySpendStore):
            async def get(self, key: str) -> Spent:
                reads.append(key)
                return await super().get(key)

        guard = SpendLimits(
            budgets=[Budget(usd=Decimal('10'), window='day'), Budget(tokens=99_999, window='day')],
            store=Counting(),
        )
        await _gate(guard)

        assert len(reads) == 1

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

    async def test_no_budgets_means_no_store_access(self):
        class Exploding(InMemorySpendStore):
            async def get(self, key: str) -> Spent:  # pragma: no cover - must never be reached
                raise AssertionError('the gate read the store with no budgets configured')

        guard = SpendLimits(store=Exploding())
        await _gate(guard)

    async def test_budgets_survive_across_runs(self):
        guard = SpendLimits(budgets=[Budget(usd=Decimal('0.01'), window='day')], price=lambda r: Decimal('0.01'))
        agent = _agent(guard)

        await agent.run('hi')
        with pytest.raises(SpendLimitExceeded):
            await agent.run('hi')


class TestOrdering:
    """Nothing may reject a response before it is counted."""

    async def test_a_response_rejected_by_a_later_capability_is_still_counted(self):
        """`after_model_request` runs innermost first, so anywhere else the counter loses a paid response."""

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

    def test_it_declares_innermost(self):
        assert SpendLimits[None]().get_ordering() == CapabilityOrdering(position='innermost')


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
        await guard.after_model_request(_run_ctx(), request_context=_request_context(), response=response)

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
            SpendLimits[None].from_spec(budgets=[{'window': window}])

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

        guard = SpendLimits(price=lambda r: Decimal('3'), on_spend=record)
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


class FakeRedis:
    """The two coroutines `RedisSpendStore` uses, over a dict.

    `eval` stands in for the server running the script: the same four increments
    and the same expiry, applied as one step, which is what Redis guarantees.
    """

    def __init__(self, *, bytes_keys: bool = False) -> None:
        self.hashes: dict[str, dict[str, int]] = {}
        self.expiries: dict[str, int] = {}
        self.calls: list[str] = []
        self._bytes_keys = bytes_keys

    async def hgetall(self, name: str) -> Mapping[str | bytes, str | bytes]:
        fields = self.hashes.get(name, {})
        if self._bytes_keys:
            return {k.encode(): str(v).encode() for k, v in fields.items()}
        return {k: str(v) for k, v in fields.items()}

    async def eval(self, script: str, numkeys: int, *keys_and_args: str | int) -> Sequence[int]:
        self.calls.append(script)
        name = str(keys_and_args[0])
        usd, tokens, requests, unpriced, ttl = (int(argument) for argument in keys_and_args[1:])
        fields = self.hashes.setdefault(name, {})
        if fields.get('usd_nanos', 0) + usd >= 2**53:
            raise RuntimeError('spend usd counter would pass the exact-integer range')
        for field, amount in (
            ('usd_nanos', usd),
            ('tokens', tokens),
            ('requests', requests),
            ('unpriced', unpriced),
        ):
            fields[field] = fields.get(field, 0) + amount
        if ttl > 0:
            self.expiries[name] = ttl
        return [fields['usd_nanos'], fields['tokens'], fields['requests'], fields['unpriced']]


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

        assert client.expiries == {'acme:k': 7200}

    async def test_a_response_is_one_round_trip_and_one_unit_of_work(self):
        """Split across commands, a failure between them leaves a window holding part of a response."""
        client = FakeRedis()
        await RedisSpendStore(client).add('k', usd=Decimal('1'), tokens=1, requests=1, unpriced=0, ttl=None)

        assert len(client.calls) == 1
        assert 'HINCRBY' in client.calls[0]
        assert 'EXPIRE' in client.calls[0]

    async def test_a_total_past_lua_s_exact_integer_range_is_refused(self):
        """Checked against the running total, not the increment: small adds reach it too."""
        store = RedisSpendStore(FakeRedis())
        half = Decimal('4600000')
        await store.add('k', usd=half, tokens=0, requests=1, unpriced=0, ttl=None)

        with pytest.raises(RuntimeError, match='exact-integer range'):
            await store.add('k', usd=half, tokens=0, requests=1, unpriced=0, ttl=None)

    @pytest.mark.parametrize(
        ('retain', 'expected'),
        [(timedelta(milliseconds=500), 1), (timedelta(seconds=1), 1), (timedelta(seconds=90), 90)],
    )
    async def test_a_sub_second_horizon_still_expires(self, retain: timedelta, expected: int):
        """`EXPIRE` takes seconds and the script reads zero as "keep"; truncating would never expire."""
        client = FakeRedis()
        await RedisSpendStore(client).add('k', usd=Decimal('1'), tokens=1, requests=1, unpriced=0, ttl=retain)

        assert client.expiries == {'pydantic-ai-harness:spend:k': expected}

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
            SpendLimits[None].from_spec(budgets=['100'])

    def test_every_number_a_spec_carries_is_coerced(self):
        """Only `usd` was converted, so the others reached `__post_init__` and compared str to int."""
        guard = SpendLimits[None].from_spec(budgets=[{'usd': '1', 'tokens': '5000', 'warn_at': '0.8'}])

        assert guard.budgets[0] == Budget(usd=Decimal('1'), tokens=5000, warn_at=0.8)

    def test_a_number_that_is_not_a_number_is_named(self):
        with pytest.raises(UserError, match="budget 'tokens' is not a number"):
            SpendLimits[None].from_spec(budgets=[{'tokens': 'lots'}])

    def test_a_budget_scope_is_refused(self):
        with pytest.raises(UserError, match='cannot be expressed in a spec'):
            SpendLimits[None].from_spec(budgets=[{'scope': 'tenant'}])


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
