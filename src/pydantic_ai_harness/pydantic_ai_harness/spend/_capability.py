"""Track what an agent spends, and stop it when a budget is gone.

`UsageLimits` in Pydantic AI caps tokens, requests and cost for the duration of
one run. `SpendLimits` covers what that leaves: periods longer than a run,
partitioning by tenant or user, and a counter that several worker processes
share. It prices each response with
[`ModelResponse.cost()`][pydantic_ai.messages.ModelResponse.cost], adds it to
every configured window, and refuses the next request once a window is spent.

The gate is local and immediate. Provider usage APIs and observability backends
aggregate after the fact and are read by polling, so a number there moves only
once the requests behind it have already been made -- enough to reconcile a
counter, not enough to stop the request a runaway loop is about to make.
"""

from __future__ import annotations

import inspect
import warnings
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import TYPE_CHECKING, Any, Literal, TypeGuard

from pydantic_ai.capabilities import AbstractCapability, CapabilityOrdering
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import ModelResponse
from pydantic_ai.tools import AgentDepsT, RunContext

from pydantic_ai_harness.spend._budget import Budget, BudgetSpec, bucket, delimited, scope_key, store_key
from pydantic_ai_harness.spend._composition import warn_about_inner_wrappers
from pydantic_ai_harness.spend._exceptions import SpendLimitExceeded, UnpricedModelError, UnpricedModelWarning
from pydantic_ai_harness.spend._snapshot import BudgetStatus, SpendSnapshot, Spent, money_precision
from pydantic_ai_harness.spend._store import (
    BatchSpendStore,
    InMemorySpendStore,
    SpendEntry,
    SpendStore,
    as_batch_store,
    utc_now,
)

if TYPE_CHECKING:
    from pydantic_ai.capabilities import WrapModelRequestHandler
    from pydantic_ai.models import ModelRequestContext
    from pydantic_ai.toolsets import AgentToolset


SpendCallback = Callable[[SpendSnapshot], None | Awaitable[None]]
"""Called after each model response with what it cost and where the budgets stand."""

PriceFunc = Callable[[ModelResponse], Decimal | None]
"""Prices a response. Return `None` to fall back to the `genai-prices` registry."""

_RUN_SCOPED_WINDOWS = ('run', 'conversation')
_UNPRICED_POLICIES = frozenset({'zero', 'raise'})


@dataclass
class SpendLimits(AbstractCapability[AgentDepsT]):
    """Accumulate spend per window and refuse a request once a window is exhausted.

    ```python
    from decimal import Decimal

    from pydantic_ai import Agent
    from pydantic_ai_harness.spend import Budget, SpendLimits

    agent = Agent(
        'openai:gpt-5.4',
        capabilities=[SpendLimits(budgets=[Budget(usd=Decimal('100'), window='day')])],
    )
    ```

    With no budgets the capability only reports, through `on_spend`. Add a
    `Budget` with no ceiling to keep a running total that never blocks.

    What the gate guarantees: no request **starts** after a budget is
    exhausted. What it does not: that spend stays under the ceiling. The
    request that crosses the line completes, and concurrent runs can each pass
    the check before any of them records anything. This is a brake on a runaway
    loop, not an accounting ledger.

    State lives across runs on purpose, so `for_run` is left alone: a daily
    budget that reset every run would not be a daily budget. Per-run isolation
    comes from `Budget(window='run')`, whose key carries the run id.

    Durable execution: not supported inside a durable workflow, on Temporal, DBOS,
    or Prefect. The hooks run in orchestration code, while the model request beside
    them is a durable unit restored from its checkpoint, so re-execution replays the
    accrual without replaying the request it counted. `SpendEntry.token` identifies
    the response, so a store implementing `BatchSpendStore` applies a replayed accrual
    once rather than twice -- within `dedup_retain`, and not at all through the adapter
    that drives a deprecated `SpendStore`, which has nowhere to put the token. What is
    left is the other direction: recovery that lands in a fresh worker holding a fresh
    `InMemorySpendStore` finds neither the counter nor the marker that guards it, and
    admits more than the budget allows. Temporal stops earlier than any of that, on
    the wall clock `before_model_request` reads to pick the window, which the sandbox
    restricts and `PydanticAIPlugin` does not pass `pydantic_ai_harness` through.
    `exhausted()` works without a `RunContext` so a workflow can at least be refused
    admission on what is already recorded -- but it reserves nothing, so it is a gate
    on the door, not a budget on what happens inside. Tracked in
    <https://github.com/pydantic/pydantic-ai-harness/issues/531>.
    """

    budgets: Sequence[Budget[AgentDepsT]] = ()
    """Windows to accumulate against, and which of them can refuse a request."""

    store: SpendStore | BatchSpendStore = field(default_factory=InMemorySpendStore)
    """Where counters live. The default holds them for the lifetime of the process.

    A store implementing only the deprecated `SpendStore` pair is driven one window per
    call through an adapter, which warns once at construction about what that costs.
    """

    price: PriceFunc | None = None
    """Prices a response before the registry is consulted.

    Returning `None` falls through to `genai-prices`. This is the way to charge
    a self-hosted model, or a negotiated rate the public registry does not know.

    An amount must be finite and not negative. Anything else fails the run with
    `UserError`, after the response's tokens and request count have been recorded:
    a credit would move a budget away from its ceiling, and a NaN or an infinity
    is a broken pricing function rather than a price.
    """

    on_spend: SpendCallback | None = None
    """Called with a `SpendSnapshot` after each response. May be sync or async."""

    on_unpriced: Literal['zero', 'raise'] = 'zero'
    """What to do when a response cannot be priced.

    `'zero'` counts it as free and increments `Spent.unpriced_requests`, so the
    gap shows up instead of disappearing. `'raise'` fails the run with
    `UnpricedModelError`. Tokens are counted either way, so a token ceiling
    still holds for a model the registry does not know.
    """

    expose_tools: bool = False
    """Offer the agent a `get_spend` tool.

    Off by default: a tool costs schema tokens on every request, and most
    applications want the number on a screen rather than in the model's context.
    """

    clock: Callable[[], datetime] = utc_now
    """Supplies the time that day and month windows are derived from.

    It does not reach a default-constructed `store`, which keeps its own `utc_now` for
    expiry. Both remain absolute instants, so a custom clock buckets on one and expires on
    the other; pass the same callable to the store when that matters.
    """

    _warned_unpriced: set[str] = field(default_factory=set[str], init=False, repr=False, compare=False)
    """Model names already reported by `UnpricedModelWarning`, so each reports once.

    Instance-level and never reset, matching the capability's own posture that state
    outlives a run: a per-run set would warn again on every run for the same model.
    """

    _reported_arrangements: set[str] = field(default_factory=set[str], init=False, repr=False, compare=False)
    """Capability arrangements already reported by `SpendCompositionWarning`, so each reports once.

    Instance-level and never reset, like `_warned_unpriced`. Keyed on the arrangement rather
    than being a single flag because `agent.run(capabilities=...)` can put a different chain
    around this instance on each run, and a flag set by a safe first run would hide the rest.
    """

    _store: BatchSpendStore = field(init=False, repr=False, compare=False)
    """`store` as something that takes every window at once, adapting a legacy one."""

    def __post_init__(self) -> None:
        """Reject an `on_unpriced` that arrived as plain data and is not one of the two policies.

        Anything other than `'raise'` behaves as `'zero'`, so a typo in a spec
        would quietly turn unpriced responses free instead of failing the run.
        """
        # Budgets sharing a name, window and scope share a counter deliberately, which is how
        # one window carries a USD and a token ceiling. Two ceilings of the SAME kind on one
        # counter is not that: it reads as two independent limits and behaves as the smaller
        # one. Comparing against a single remembered budget per slot missed any collision a
        # later budget displaced, so the slot carries every attribute the counter key does.
        if self.defer_loading is True:
            # Both hooks are skipped while a deferred capability is unloaded, so an
            # exhausted budget would let requests through until the model happened to
            # load it, and the responses it missed would never be counted. A brake that
            # the thing being braked decides when to apply is not a brake.
            raise UserError(
                '`defer_loading` is not supported on `SpendLimits`: the enforcement and accounting '
                'hooks do not run until the capability is loaded, so an exhausted budget would not '
                'stop a request and the requests made meanwhile would go uncounted.'
            )

        # Two budgets that share a name and window but declare different scope callables are
        # different dimensions -- per tenant and per user, say -- keyed only by what each
        # callable returns. Nothing stops those returning the same string, and the counter
        # that results mixes the two, because every update writes every metric. Sharing one
        # callable is how a USD and a token ceiling deliberately share a counter.
        scoped: dict[tuple[str, str], Budget[AgentDepsT]] = {}
        for budget in self.budgets:
            if budget.scope is None:
                continue
            slot = (budget.name, budget.window)
            prior = scoped.get(slot)
            if prior is not None and prior.scope is not budget.scope:
                raise UserError(
                    f'Budgets named {budget.name!r} on the same window declare different `scope` callables, '
                    'so they would share one counter whenever the two return the same string, mixing the '
                    'dimensions. Give them different `name`s, or pass the same callable to both.'
                )
            scoped[slot] = budget

        # Budgets sharing a key share one counter, and one counter has one expiry: the accrual
        # writes whichever `retain` the first of them carries, so declaration order would decide
        # when a `'forever'` ceiling quietly rolls over. Rejected rather than reconciled -- taking
        # the longest would silently extend the shorter budget's horizon, which is the same class
        # of surprise pointing the other way.
        retention: dict[tuple[str, str, bool], Budget[AgentDepsT]] = {}
        for budget in self.budgets:
            counter = (budget.name, budget.window, budget.scope is None)
            prior = retention.get(counter)
            if prior is not None and prior.retain != budget.retain:
                raise UserError(
                    f'Budgets named {budget.name!r} on the same window and scope share one counter but set '
                    f'different `retain` values ({prior.retain!r} and {budget.retain!r}), so the order they '
                    'are listed in would decide when that counter expires. Give them the same `retain`, or '
                    'different `name`s.'
                )
            retention[counter] = budget

        seen: set[tuple[str, str, str, int]] = set()
        for budget in self.budgets:
            for kind, ceiling in (('usd', budget.usd), ('tokens', budget.tokens)):
                if ceiling is None:
                    continue
                slot = (budget.name, kind, budget.window, id(budget.scope))
                if slot in seen:
                    raise UserError(
                        f'Two budgets named {budget.name!r} both set a `{kind}` ceiling on the same window and '
                        'scope, so they would share one counter and only the smaller would ever apply. '
                        'Give them different `name`s.'
                    )
                seen.add(slot)
        if self.on_unpriced not in _UNPRICED_POLICIES:
            raise UserError(
                f'SpendLimits.on_unpriced must be one of {sorted(_UNPRICED_POLICIES)}; got {self.on_unpriced!r}.'
            )
        # Resolved once, and last, so a configuration that was going to be refused is
        # refused before a deprecated store is reported.
        self._store = as_batch_store(self.store)

    @classmethod
    def get_serialization_name(cls) -> str | None:
        """Serialization name for agent-spec support."""
        return 'SpendLimits'

    def get_ordering(self) -> CapabilityOrdering:
        """Sit innermost, so the accrual happens as close to the provider call as ordering allows.

        Innermost puts this capability's `wrap_model_request` inside every capability outside
        that tier, so their wrappers -- and every capability's `after_model_request` -- run
        outside the accrual and cannot reject a response the counter has not already seen.

        This orders against non-innermost capabilities only. Innermost members are not
        ordered among themselves, and the one listed later nests further in, so another
        innermost capability placed after this one still wraps inside it. `InputGuardrail` is
        the one that reaches a billed response before the counter does. List
        `SpendLimits` last among innermost capabilities where that matters; closing it
        outright is <https://github.com/pydantic/pydantic-ai-harness/issues/534>.
        """
        return CapabilityOrdering(position='innermost')

    def get_toolset(self) -> AgentToolset[AgentDepsT] | None:
        """Offer `get_spend` when `expose_tools` is set."""
        if not self.expose_tools:
            return None
        from pydantic_ai_harness.spend._toolset import build_toolset

        return build_toolset(self)

    async def before_model_request(
        self,
        ctx: RunContext[AgentDepsT],
        request_context: ModelRequestContext,
    ) -> ModelRequestContext:
        """Refuse the request if any budget with a ceiling is already spent.

        Also where the arrangement `get_ordering` cannot rule out is reported. The sorted
        chain is readable from `RunContext.root_capability` from `before_run` onward but not
        before it: `for_agent` sees only the capabilities the agent was constructed with, and
        `ctx.root_capability` is still `None` in `for_run`, so neither covers a capability
        added through `agent.run(capabilities=...)`. `before_run` would serve as well, since
        the chain is fixed for a run; the read sits here to stay on the request path, beside
        the accrual it is about. Re-reading per request costs nothing because
        `_reported_arrangements` makes it idempotent, and keying on the arrangement rather
        than on having reported is what covers a chain that differs between runs.
        """
        warn_about_inner_wrappers(ctx.root_capability, self, self._reported_arrangements)
        enforcing = [(budget, key) for budget, key in self._keyed(ctx) if budget.enforces]
        read = await self._read(list(dict.fromkeys(key for _, key in enforcing)))
        for budget, key in enforcing:
            self._check(budget, read[key], ctx)
        return request_context

    async def wrap_model_request(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        request_context: ModelRequestContext,
        handler: WrapModelRequestHandler,
    ) -> ModelResponse:
        """Price what the provider returned and add it to every window, before an outer capability can reject it.

        The accrual belongs here rather than in `after_model_request` because
        `after_model_request` runs outside this chain, once the whole chain has returned.
        A capability whose own `wrap_model_request` awaits the response and then raises
        `ModelRetry` sends the run straight to a fresh request, and the response it
        rejected -- generated, billed, and kept in history -- is never counted. Ordering
        cannot close that: the rejecting wrapper does not have to be innermost, and one
        listed *before* this capability still nests outside it.

        Wrapping is also why a request the provider never saw is not charged for.
        `SkipModelRequest` from an earlier `before_model_request` reaches
        `after_model_request` with a response the run never paid for; it does not reach
        `handler`, so nothing accrues here.
        """
        response = await handler(request_context)
        usd, priced, price_error = self._price_of(response)
        keyed = self._keyed(ctx)
        token = self._dedup_token(ctx, response)
        entries: dict[str, SpendEntry] = {}
        for budget, key in keyed:
            # Budgets sharing a name, window, and scope share a counter, which is
            # how one window carries both a USD and a token ceiling. Adding the
            # response once per budget would double-count it and halve them both.
            if key not in entries:
                entries[key] = SpendEntry(
                    key=key,
                    usd=usd,
                    tokens=response.usage.total_tokens,
                    requests=1,
                    unpriced=0 if priced else 1,
                    ttl=budget.ttl,
                    token=token,
                )
        # Every window in one call, so a failure cannot leave the response counted
        # against the day and not the month. Nothing to apply is not a call: see `_read`.
        accrued: Mapping[str, Spent] = await self._store.add_many(list(entries.values())) if entries else {}
        statuses = [_status(budget, key, accrued[key]) for budget, key in keyed]

        if self.on_spend is not None:
            snapshot = SpendSnapshot(
                model=response.model_name,
                usage=response.usage,
                usd=usd,
                priced=priced,
                budgets=tuple(statuses),
            )
            result = self.on_spend(snapshot)
            if inspect.isawaitable(result):
                await result

        if price_error is not None:
            # Raised after the store and `on_spend` have seen the response, for the reason
            # `_price_of` gives. Ahead of the `on_unpriced` handling below, which would
            # otherwise report a broken pricing function as an unknown model. Raising from
            # a wrapper rather than from `after_model_request` does not change that: the
            # accrual is already committed above.
            raise UserError(f'`SpendLimits.price` {price_error} for a response.')

        if not priced and self.on_unpriced == 'zero' and any(budget.usd is not None for budget in self.budgets):
            # Only this combination is silent: the response adds nothing in dollars, so a
            # USD ceiling can never be reached by requests the registry cannot price. A
            # token ceiling still holds, so it is not warned about.
            model = response.model_name or '<unnamed>'
            if model not in self._warned_unpriced:
                self._warned_unpriced.add(model)
                warnings.warn(
                    f'No price for model {model}, so it counts as $0 against a USD budget. '
                    "Supply `SpendLimits.price` to price it, or set `on_unpriced='raise'` to "
                    'stop the run instead. Token ceilings are unaffected.',
                    UnpricedModelWarning,
                    stacklevel=2,
                )

        if not priced and self.on_unpriced == 'raise':
            # Raised last, after the store is updated and `on_spend` has seen the
            # response. The request happened and its tokens were really spent, so
            # dropping them would leave a token ceiling understating what the
            # model was asked to do, and an audit that skipped exactly the
            # unpriced responses would be missing the ones worth knowing about.
            raise UnpricedModelError(
                f'No price for model {response.model_name or "<unnamed>"}. Supply `SpendLimits.price`, '
                "or set on_unpriced='zero' to count the request as free."
            )
        return response

    async def status(
        self,
        ctx: RunContext[AgentDepsT] | None = None,
        *,
        scope: str | None = None,
    ) -> tuple[BudgetStatus, ...]:
        """Where each budget stands.

        Inside a run, pass `ctx` and every budget resolves. Without one -- the
        reading a cost display wants, and the check to make before starting a
        durable workflow whose hooks cannot reach a shared store -- budgets on a
        `run` or `conversation` window are omitted, since those periods have no
        meaning outside a run, and a budget declaring a `scope` is omitted
        unless `scope` names the partition to read, since its callable has no
        run context to resolve against.

        Reach for [`exhausted`][pydantic_ai_harness.spend.SpendLimits.exhausted] when the
        answer gates something: `any(s.exhausted for s in ...)` over a tuple that happens to
        be empty is a brake that reads as enforcement and inspects nothing, and a `SpendLimits`
        whose budgets are all scoped returns exactly that tuple.
        """
        statuses, _ = await self._resolve(ctx, scope)
        return statuses

    async def _resolve(
        self, ctx: RunContext[AgentDepsT] | None, scope: str | None
    ) -> tuple[tuple[BudgetStatus, ...], tuple[str, ...]]:
        """The readable budgets, and the names of the ones this call cannot resolve.

        `ctx` and `scope` are two answers to the same question, so supplying both is refused
        rather than resolved by precedence: a run's own scope silently winning would report
        one tenant's money under another tenant's name, and the caller has no way to tell.
        """
        if ctx is not None and scope is not None:
            raise UserError(
                'Pass either a run context or `scope=`, not both: `ctx` already names the partition to read, '
                'so a second answer here would be reporting one scope under the name of another. '
                'Drop `scope=` to read the run, or drop `ctx` to read another partition.'
            )
        now = self._now()
        keyed: list[tuple[Budget[AgentDepsT], str]] = []
        unresolved: list[str] = []
        for budget in self.budgets:
            if ctx is None and (budget.window in _RUN_SCOPED_WINDOWS or (budget.scope is not None and scope is None)):
                unresolved.append(budget.name)
                continue
            keyed.append((budget, self._key(budget, ctx, now, scope)))
        read = await self._read(list(dict.fromkeys(key for _, key in keyed)))
        return tuple(_status(budget, key, read[key]) for budget, key in keyed), tuple(unresolved)

    async def exhausted(
        self,
        ctx: RunContext[AgentDepsT] | None = None,
        *,
        scope: str | None = None,
    ) -> bool:
        """Whether any budget this call can read is exhausted, refusing to guess about the rest.

        An admission check, and only that. It reads the counters; it reserves nothing and
        records nothing, so work started on the strength of it goes unmeasured unless
        something else accrues it. That makes it the pre-flight option on every durable
        engine, and what the next caller reads differs. Under Temporal nothing accrues
        inside the workflow at all, because the sandbox refuses the clock these hooks read,
        so the counter still holds the admission reading. Under DBOS and Prefect the accrual
        does run, and `SpendEntry.token` keeps a replay of it from counting twice, so the
        counter tracks the workflow as long as recovery still reaches the store that accrued.
        Either way this is a floor on runaway spend already recorded, not a ceiling on what
        the workflow goes on to spend.

        `status()` omits what it cannot resolve, and `any(...)` over the remainder is a brake
        that silently checks nothing when every budget is scoped -- so this raises instead,
        naming the budgets that need a `scope` or a `ctx`.
        """
        statuses, unresolved = await self._resolve(ctx, scope)
        if unresolved:
            raise UserError(
                f'Cannot read budget(s) {sorted(unresolved)} without a run context or a `scope`, so this check '
                'would pass having inspected nothing. Pass `scope=` for a scoped budget, or call it inside a run.'
            )
        return any(status.exhausted for status in statuses)

    @classmethod
    def from_spec(
        cls,
        *,
        budgets: Sequence[BudgetSpec] = (),
        on_unpriced: Literal['zero', 'raise'] = 'zero',
        expose_tools: bool = False,
        id: str | None = None,
        description: str | None = None,
        defer_loading: bool = False,
        **unsupported: Any,
    ) -> SpendLimits[Any]:
        """Build from an agent spec, covering the fields a spec can express.

        Every parameter is named because that signature is what core reads to generate
        the spec's JSON schema: `build_schema_types` drops `*args`/`**kwargs`, so a
        catch-all signature publishes the bare string `'SpendLimits'` and an editor
        marks every documented `budgets:` block as invalid even though it loads.

        `budgets` arrive as mappings and become `Budget` instances, with `usd`
        accepted as a string so YAML cannot round a price through a float. The
        callables and the store have no spec representation and are rejected
        rather than dropped: a spec that promises per-tenant scoping and does
        not deliver it is worse than a spec that refuses to load. `**unsupported`
        stays so that rejection keeps naming the field; core drops it from the schema,
        so it costs nothing there.
        """
        callables = sorted({'store', 'price', 'on_spend', 'clock'} & unsupported.keys())
        if callables:
            raise UserError(
                f'SpendLimits cannot be built from a spec with {callables}: these take callables or a live '
                'store. Construct the capability in code to use them.'
            )
        if unsupported:
            raise UserError(f'SpendLimits has no spec field(s) {sorted(unsupported)}.')
        return cls(
            budgets=[_budget_from_spec(entry) for entry in budgets],
            on_unpriced=on_unpriced,
            expose_tools=expose_tools,
            id=id,
            description=description,
            defer_loading=defer_loading,
        )

    def _now(self) -> datetime:
        """The current time, naming the real problem when Temporal's sandbox refuses the clock.

        These hooks run in workflow code, and the default clock calls `datetime.now`, which
        Temporal's workflow sandbox restricts. The sandbox's own error names
        `datetime.datetime.now` and not what it means here, so it is translated. Matched by
        class name rather than by importing `temporalio`, which this package does not depend
        on, and which core's own `pydantic_ai/durable_exec/AGENTS.md` rules out: "Prefer generic
        capabilities/toolsets/models extension points over engine-specific escape hatches."

        The message leads with the unsafety rather than the passthrough that silences it: the
        sandbox is refusing a symptom, and a caller who only removes the symptom gets a
        counter that Temporal replays.
        """
        try:
            return self.clock()
        except Exception as error:
            if type(error).__name__ != 'RestrictedWorkflowAccessError':
                raise
            raise UserError(
                'SpendLimits is not safe to run inside a Temporal workflow. Its hooks run in '
                'workflow code rather than in the model activity, and the clock they read to pick '
                'the budget window is what the sandbox stopped here; a workflow day and a key day '
                'diverge under time-skipping even once it is let through. Refuse the workflow '
                'admission before starting it instead, '
                'with `exhausted()` -- which reads the counters and does not move them, so it '
                'bounds what has already been recorded rather than what the workflow will spend. '
                'See https://github.com/pydantic/pydantic-ai-harness/issues/531'
            ) from error

    def _key(
        self,
        budget: Budget[AgentDepsT],
        ctx: RunContext[AgentDepsT] | None,
        now: datetime,
        scope: str | None,
    ) -> str:
        """The store key this budget accumulates under right now."""
        bucket_id = bucket(budget.window, ctx, now)
        if bucket_id is None:  # pragma: no cover - callers filter run-scoped windows out first
            raise UserError(f"Budget {budget.name!r} uses window='{budget.window}', which needs a run.")
        return store_key(budget, bucket_id, scope_key(budget, ctx, scope))

    def _dedup_token(self, ctx: RunContext[AgentDepsT], response: ModelResponse) -> str:
        """Identifies the response, so an accrual a durable engine re-executes lands once.

        These hooks run in orchestration code, which DBOS re-executes when it recovers a
        workflow and Prefect re-executes when a flow retries. The model request itself is
        the durable unit, so both hand back the response they already have while the
        accrual around it runs again, and the window counts one response twice.

        The token therefore has to come from the response rather than from the run.
        `provider_response_id` is the provider's own identifier for the request, kept in
        the checkpointed response, and paired with the provider name because two providers
        can mint the same string. When a provider reports none, the fallback identifies
        the response by where it sat in the run and when it arrived:
        `ModelResponse.timestamp` is set as the response is built, inside the durable unit,
        and `run_step` is incremented by `ModelRequestNode._prepare_request` before any of
        these hooks run. That pair is replay-stable only as far as `ctx.run_id` is, which
        is a fresh UUID7 unless the caller passes one -- so a provider that reports no
        response id needs a `run_id` chosen by the caller for the accrual to be idempotent.

        `ModelResponse.run_id` is not usable here: `fill_run_metadata` stamps it after this
        wrapper returns, so it is still `None` at this point.

        The parts are `delimited` rather than joined, because a provider id and a
        caller-supplied run id can contain anything: joined on a separator,
        `provider_name='a|b'` with id `'c'` and `provider_name='a'` with id `'b|c'` name
        the same response, and the second one's spend is dropped as a replay of the first.

        An empty `provider_response_id` takes the fallback rather than the primary path.
        The field is a plain `str` on the wire -- `ChatCompletion.id` is required and
        unnormalised -- so an OpenAI-compatible server answering `"id": ""` would otherwise
        name every one of its responses the same, and every response after the first would
        be dropped as a replay of it. Core reads the same field for truth rather than for
        presence where it matters (`models/_continuation.py`, `models/openai.py`).

        The timestamp joins the id rather than only standing in for it, so a server that
        reports the *same* non-empty id for different responses does not have them collapse
        into one. It costs the primary path nothing: a replayed response is the same object
        checkpointed and handed back, so its timestamp is the one it was built with, which
        is the property the fallback below already rests on. `ModelResponse.timestamp` is
        the client's own `now_utc()` rather than a provider field (`models/openai.py` keeps
        the provider's coarse `created` in `provider_details`), so it separates responses at
        microsecond resolution. What is left is two responses sharing an id *and* a
        microsecond; tracked in
        <https://github.com/pydantic/pydantic-ai-harness/issues/693>.
        """
        stamp = response.timestamp.isoformat()
        if response.provider_response_id:
            return delimited(response.provider_name or '', response.provider_response_id, stamp)
        return delimited(ctx.run_id or '', str(ctx.run_step), stamp)

    def _check(self, budget: Budget[AgentDepsT], spent: Spent, ctx: RunContext[AgentDepsT]) -> None:
        """Raise if `spent` has reached either of the budget's ceilings."""
        if budget.usd is not None and spent.usd >= budget.usd:
            self._refuse(budget, ctx, f'spent ${spent.usd} of ${budget.usd}')
        if budget.tokens is not None and spent.tokens >= budget.tokens:
            self._refuse(budget, ctx, f'used {spent.tokens} of {budget.tokens} tokens')

    def _refuse(self, budget: Budget[AgentDepsT], ctx: RunContext[AgentDepsT], detail: str) -> None:
        """Record the refusal as a span and raise."""
        attributes: dict[str, str] = {'spend.budget': budget.name, 'spend.window': budget.window}
        if ctx.trace_include_content and budget.scope is not None:
            # A scope key is usually a tenant or user id, and a trace has a wider
            # audience than the application that produced it.
            attributes['spend.scope'] = budget.scope(ctx)
        ctx.tracer.start_span('spend budget exhausted', attributes=attributes).end()
        raise SpendLimitExceeded(f'Budget {budget.name!r} exhausted for this {budget.window}: {detail}')

    async def _read(self, keys: Sequence[str]) -> Mapping[str, Spent]:
        """What each key holds, without asking the store about none of them.

        A `SpendLimits` configured to report rather than enforce, or one whose budgets this
        call cannot resolve, has no key to read. Asking anyway spends a round trip on a
        store that answers it and fails outright on one that treats an empty batch as a
        caller error, and neither buys anything. `add_many` is guarded the same way.
        """
        return await self._store.get_many(keys) if keys else {}

    def _keyed(self, ctx: RunContext[AgentDepsT]) -> list[tuple[Budget[AgentDepsT], str]]:
        """Each budget paired with the store key it accumulates under right now.

        No collision check here. Every part of a key is fixed at construction -- `name`,
        `window`, and the scope callable -- and `__post_init__` rejects the combinations
        that would collide, so two budgets share a key only where sharing one is the
        point. Checking again would repeat that work on every model request.
        """
        now = self._now()
        return [(budget, self._key(budget, ctx, now, None)) for budget in self.budgets]

    def _price_of(self, response: ModelResponse) -> tuple[Decimal, bool, str | None]:
        """What the response cost, whether that number is real, and why it was rejected.

        A rejected amount is reported rather than raised so the caller can finish
        accruing the response first. The request happened and its tokens were really
        spent, so dropping them would leave a token ceiling understating what the model
        was asked to do -- the same reasoning `on_unpriced='raise'` already follows.
        """
        if self.price is not None:
            supplied = self.price(response)
            if supplied is not None:
                if not supplied.is_finite():
                    # Checked before the comparison below, which raises `InvalidOperation`
                    # on a NaN rather than returning False. An infinity would pass that
                    # comparison and then exhaust every budget it reached at once.
                    return Decimal(0), False, f'returned a non-finite amount ({supplied})'
                if supplied < 0:
                    # A credit would move a budget away from its ceiling, which
                    # turns a bug in the pricing function into a gate that never
                    # closes. Corrections belong in the store, not here.
                    return Decimal(0), False, f'returned a negative amount ({supplied})'
                return supplied, True, None
        if response.model_name:
            try:
                return response.cost().total_price, True, None
            except (LookupError, ValueError):
                # LookupError: the registry has no entry. ValueError: it rejects the
                # usage shape. Either way this is a pricing failure, and letting it
                # escape would skip `on_unpriced` and drop the accrual with it.
                pass
        return Decimal(0), False, None


def _status(budget: Budget[Any], key: str, spent: Spent) -> BudgetStatus:
    """Pair a budget with what its window has accumulated.

    `remaining_usd` under `money_precision` for the same reason the counter itself is: an
    application that lowered `Decimal` precision for its own arithmetic would otherwise be
    told a rounded number by a store that holds an exact one.
    """
    with money_precision():
        remaining_usd = None if budget.usd is None else budget.usd - spent.usd
    remaining_tokens = None if budget.tokens is None else budget.tokens - spent.tokens
    return BudgetStatus(
        budget=budget,
        key=key,
        spent=spent,
        remaining_usd=remaining_usd,
        remaining_tokens=remaining_tokens,
        warning=_warning(budget, spent),
        exhausted=(remaining_usd is not None and remaining_usd <= 0)
        or (remaining_tokens is not None and remaining_tokens <= 0),
    )


def _warning(budget: Budget[Any], spent: Spent) -> bool:
    """Whether spend has crossed the budget's warning fraction.

    The USD product is taken under `money_precision`: rounded at the application's, a
    crossing near the fraction reports the wrong side of it.
    """
    if budget.warn_at is None:
        return False
    fraction = Decimal(str(budget.warn_at))
    with money_precision():
        if budget.usd is not None and spent.usd >= budget.usd * fraction:
            return True
    return budget.tokens is not None and spent.tokens >= budget.tokens * fraction


def _is_spec_mapping(entry: object) -> TypeGuard[Mapping[str, Any]]:
    """Whether a spec entry has the shape a `Budget` can be built from.

    Only the mapping shape is checked; a non-string key fails against
    `Budget`'s own signature, in an error that names the field.
    """
    return isinstance(entry, Mapping)


def _budget_from_spec(entry: Any) -> Budget[Any]:
    """Build one `Budget` from its spec mapping."""
    if not _is_spec_mapping(entry):
        raise UserError(f'Each SpendLimits budget in a spec must be a mapping; got {entry!r}.')
    fields = dict(entry)
    if 'scope' in fields:
        raise UserError('A SpendLimits budget scope is a callable and cannot be expressed in a spec.')
    # Every number a spec can carry, not just `usd`: the others reached
    # `Budget.__post_init__` as strings and compared str to int there, which
    # surfaces as a bare TypeError naming no field.
    for name, convert in (('usd', Decimal), ('tokens', int), ('warn_at', float)):
        if name in fields:
            try:
                fields[name] = convert(str(fields[name]))
            except (TypeError, ValueError) as error:
                raise UserError(f'SpendLimits budget {name!r} is not a number: {fields[name]!r}.') from error
    return Budget(**fields)
