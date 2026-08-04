---
title: Spend
description: Track what an agent costs and refuse the next request once a budget is spent, with windows longer than a run, per-tenant scopes, and a counter shared across worker processes.
---

# Spend

Track what an agent costs, and stop it when a budget is gone.

> The API may change between releases. Where practical, breaking changes ship with a deprecation warning.

## The problem

A loop that calls a model until a condition it never reaches will keep calling until something stops it. `UsageLimits` in Pydantic AI is that stop for one run: it caps tokens and requests, in token counts, for the duration of a single `run()`. What it does not cover is money, a period longer than one run, a per-tenant share of a shared allowance, or a counter that several worker processes agree on. A daily ceiling spread across a queue's workers is exactly the case where each worker independently believes it has the whole budget.

Provider usage APIs do not close that gap. They are billing and observability pipelines: usage is aggregated after the fact and read by polling, so a number there moves only once the requests behind it have already been made. That is enough to reconcile a ledger and not enough to refuse the request a runaway loop is about to make.

## The solution

`SpendLimits` prices every model response with [`ModelResponse.cost()`](https://pydantic.dev/docs/ai/api/messages/), adds it to each window you configure, and refuses the next request once a window is spent.

```python
from decimal import Decimal

from pydantic_ai import Agent
from pydantic_ai_harness.spend import Budget, SpendLimits

agent = Agent(
    'openai:gpt-5.4',
    capabilities=[SpendLimits(budgets=[Budget(usd=Decimal('100'), window='day')])],
)
```

Past $100 in a UTC day, the next request raises `SpendLimitExceeded`.

## Budgets

A budget is a ceiling, a period, and optionally a partition. They compose, so several apply at once:

```python
from decimal import Decimal

from pydantic_ai_harness.spend import Budget, SpendLimits

SpendLimits(
    budgets=[
        Budget(usd=Decimal('5'), window='run'),  # one runaway run
        Budget(usd=Decimal('100'), window='day'),  # the whole deployment, per day
        Budget(usd=Decimal('2000'), window='month', warn_at=0.8),
        Budget(usd=Decimal('10'), window='day', scope=lambda ctx: ctx.deps.tenant_id, name='tenant'),
    ]
)
```

| Field | Meaning |
|---|---|
| `usd` / `tokens` | ceilings; set either, both, or neither |
| `window` | `run`, `conversation`, `day`, `month`, `total` |
| `scope` | derives a partition key from the run, so tenants count separately; typed against the agent's `deps` |
| `warn_at` | fraction past which `BudgetStatus.warning` is set; never blocks |
| `name` | distinguishes budgets sharing a window and scope |
| `retain` | how long the counter is kept after its last write; `'window default'`, `'forever'`, or a `timedelta` |

A window rolls over by producing a different store key rather than by resetting a counter, so a new day is simply a new key and nothing has to run at midnight. A `total` counter never expires. `run` and `conversation` buckets never roll over either, so expiry there hands back the ceiling rather than starting a new period -- but each mints a key per run or per conversation, so they carry a long horizon (24 hours and 30 days) instead, past which the counter is dropped rather than kept forever. That default is a compromise, and it is visible: a conversation resumed past its horizon starts from zero again, so set `retain='forever'` where a conversation ceiling has to hold for as long as the conversation does, and clean the keys up some other way.

Budgets that share a `name`, `window`, and `scope` share one counter, which is how a single window carries both a USD and a token ceiling. The response is added to that counter once, not once per budget. Two budgets that share a `name` and `window` but declare *different* `scope` callables are refused at construction: they are different dimensions -- per tenant and per user, say -- and nothing stops the two returning the same string, which would merge them into one counter. Give them different names, or pass the same callable to both. Budgets that do share a counter must also agree on `retain`: one counter has one expiry, and the accrual writes whichever of them is listed first, so disagreeing would let declaration order decide when a `'forever'` ceiling rolls over.

`Budget` is generic in the agent's dependency type, so a `scope` is checked against it: pass the capability to an `Agent` with a `deps_type` and a scope reaching for a field those deps do not have is a type error rather than an `AttributeError` on the first request.

**A budget with no ceiling is a counter.** It accumulates and reports and never refuses anything, which is how per-tenant accounting with no cap is expressed:

```python
from pydantic_ai_harness.spend import Budget, SpendLimits

SpendLimits(budgets=[Budget(window='month', scope=lambda ctx: ctx.deps.tenant_id, name='chargeback')])
```

## What the gate guarantees

No request **starts** after a budget is exhausted.

Not: that spend stays under the ceiling. The request that crosses the line completes, and concurrent runs can each pass the check before any of them records anything. Three further gaps are worth knowing rather than discovering: a stream the caller abandons part-way never reaches the accounting hook, so its tokens are billed by the provider and invisible here; a capability that answers from a cache without calling a provider is charged the registry price for the response it returns; and a continuation chain (Anthropic `pause_turn`, OpenAI background mode) arrives at the hook as one merged response, which is what Pydantic AI counts as one request too, so its segments are priced on summed usage rather than one at a time -- the difference only shows where pricing is tiered rather than linear. Treat this as a brake on a runaway loop, not as an accounting ledger; reconcile against the provider's own numbers if you need the second thing.

## Reading the numbers

```python
from decimal import Decimal

from pydantic_ai_harness.spend import Budget, SpendLimits, SpendSnapshot


def show(snapshot: SpendSnapshot) -> None:
    print(f'{snapshot.model} cost ${snapshot.usd}')
    for status in snapshot.budgets:
        print(f'  {status.budget.name}: ${status.remaining_usd} left')


SpendLimits(budgets=[Budget(usd=Decimal('100'))], on_spend=show)
```

`on_spend` fires after every response, sync or async, with a `SpendSnapshot` -- including one that `on_unpriced='raise'` is about to reject, since a report that skipped exactly the unpriced responses would be missing the ones worth knowing about. It carries the response's `usage` unchanged, so cache reads and writes are available without this capability modelling them.

`status()` reads the same numbers without a run, which is what a cost display in a UI wants:

```python
async def report(limits: SpendLimits[None]) -> None:
    for status in await limits.status(scope='acme'):
        print(status.budget.name, status.spent.usd, status.exhausted)
```

Without a run context, budgets on a `run` or `conversation` window are omitted, and so is a budget declaring a `scope` unless `scope=` names the partition to read. Pass `ctx` inside a run and every budget resolves.

Set `expose_tools=True` to give the agent a `get_spend` tool. It is off by default: a tool costs schema tokens on every request, and most applications want the number on a screen rather than in the model's context.

## Sharing a counter across processes

The default store keeps counters in the process, which catches a runaway loop inside one worker and does nothing for a budget spread across a queue. `RedisSpendStore` is the shared counter:

```python
from decimal import Decimal

from redis.asyncio import Redis

from pydantic_ai_harness.spend import Budget, RedisSpendStore, SpendLimits

store = RedisSpendStore(Redis.from_url('redis://localhost'))
limits = SpendLimits(budgets=[Budget(usd=Decimal('100'), window='day')], store=store)
```

It adds no dependency: `RedisClient` is a protocol of the two coroutines used, so any compatible client satisfies it. Amounts are stored as integer billionths of a dollar rather than through `INCRBYFLOAT`, which accumulates rounding error over the tens of thousands of requests a busy day produces. Billionths rather than millionths because the residue does not average out: an agent repeats requests of near-identical shape, so the same fraction rounds the same way every time.

Each window is applied as one Lua script, in one round trip, so no other client sees a window holding part of a response and the exact-integer ceiling is checked before anything is written. That guarantee is per window, not across them: a response counting against a day budget and a month budget is two scripts, so a failure between the two leaves the day counted and the month not. Widening it needs a store operation that takes every window at once, tracked in [#536](https://github.com/pydantic/pydantic-ai-harness/issues/536).

A failure *after* the server has run a script does not say whether it committed -- the connection can drop once `EVAL` has landed -- so an `add` that errors leaves the outcome unknown rather than untried. Nothing retries it: counting a billed response twice is a direction the brake survives, and counting it zero times is not.

The cost of doing it server-side is a ceiling -- the counters pass through Lua, whose numbers stop being exact integers above `2**53` billionths, about **$9,007,199** against a single key. Past that the store raises rather than rounding silently. Settling a protocol without that ceiling is [#532](https://github.com/pydantic/pydantic-ai-harness/issues/532).

The default store is built per capability, so two `SpendLimits` instances do not quietly share one counter. Pass the same store object to both when you want them to.

A store that fails does not fail quietly. An error reading the counter refuses the request, which is the safe direction. An error writing it propagates out of the run after the model has already answered and been charged. That is deliberate: a swallowed write would drift the counter down and weaken the gate, which is worse than a visible failure. If your deployment would rather keep the answer than the count, wrap the store and decide there.

Any object with `get` and `add` works, so a Postgres or DynamoDB counter is a small class rather than a fork.

## Pricing

Prices come from [genai-prices](https://github.com/pydantic/genai-prices) via `ModelResponse.cost()`, per response: cache and tier pricing are per request, so summing usage across requests and pricing the total gives the wrong number.

A model the registry does not know -- a local deployment, a negotiated rate -- is handled by `price`:

```python
from decimal import Decimal

from pydantic_ai_harness.spend import SpendLimits

SpendLimits(price=lambda response: Decimal('0.002') if response.model_name == 'internal-7b' else None)
```

An amount returned by `price` must be finite and not negative. Anything else -- a credit, a `NaN`, an infinity -- fails the run with `UserError`, because a credit moves a budget away from its ceiling and the other two are a broken pricing function rather than a price. The response is still recorded first: it was billed by the provider whatever the function returned, so its tokens and request count are accrued and `on_spend` fires before the error is raised.

Returning `None` falls through to the registry. When nothing can price a response, `on_unpriced` decides: `'zero'` (the default) counts it as free and increments `Spent.unpriced_requests` so the gap is visible, and `'raise'` fails the run with `UnpricedModelError`. Either way the response is recorded first and the tokens are counted, so a token ceiling still holds for a model with no price and an application that catches the error does not carry on against an understated counter. Under `'zero'` a USD ceiling is the one that cannot hold: nothing priceable accrues, so no number of such requests reaches it. That combination -- `'zero'` plus a `usd` budget -- warns once per model with `UnpricedModelWarning`, rather than once per request. If callers choose the model, prefer `'raise'` or supply `price`.

## Composition

State lives across runs deliberately, so `for_run` is not overridden: a daily budget that reset every run would not be a daily budget. Per-run isolation comes from `Budget(window='run')`, whose key carries the run id.

`defer_loading=True` is refused. A deferred capability's hooks do not run until the model loads it, so an exhausted budget would not stop a request and the requests made meanwhile would go uncounted -- a brake the thing being braked decides when to apply.

The accrual happens in `wrap_model_request`, immediately around the provider call, and the capability declares itself innermost so that wrapper is the innermost one. Every other capability's wrapper, and every capability's `after_model_request`, therefore runs outside it and cannot reject a response the counter has not already seen.

`after_model_request` is the wrong hook for this. It runs once the whole wrap chain has returned, so a capability whose own `wrap_model_request` awaits the response and then raises `ModelRetry` sends the run straight to a fresh request and the rejected one -- generated, billed, kept in history -- is never counted. Ordering cannot reach that case: the rejecting capability need not be innermost, and one listed *before* `SpendLimits` still wraps outside it.

Wrapping also means a request the provider never saw is not charged for. `SkipModelRequest` from an earlier capability's `before_model_request` reaches `after_model_request` with a response the run never paid for, but never reaches the wrapped handler.

What is left is siblings. Pydantic AI orders innermost capabilities against non-innermost ones only, and among themselves the one listed *later* nests further in. `TemporalDurability` and `InputGuardrail` also declare themselves innermost, so either of them listed after `SpendLimits` wraps inside it and can still reject a billed response before it is counted. List `SpendLimits` last among your innermost capabilities when that matters. Closing it outright needs a way to order innermost capabilities against each other, or the public composition-validation hook being decided in [pydantic-ai#5477](https://github.com/pydantic/pydantic-ai/issues/5477); tracked in [#534](https://github.com/pydantic/pydantic-ai-harness/issues/534).

**Durable execution.** `SpendLimits` is not supported inside a Temporal workflow.

The capability hooks run in workflow code; only the model request itself is the activity. Temporal replays workflow code, so it replays the accrual with it, and a window ends up counting the same response more than once -- one `$1` model activity leaves `$2` in the store once the workflow replays. The day and month buckets have the same problem from the other side: they come from a wall clock the workflow sandbox restricts, and under time-skipping the workflow's day and the key's day drift apart.

The sandbox refusing that clock is what surfaces this first, and `SpendLimits` translates the error into what it means rather than into the setting that silences it. Passing the package through the sandbox removes the message, not the replay.

Refuse the workflow **admission** before starting it instead. That is why `exhausted()` works without a `RunContext`:

```python
async def start_if_funded(limits: SpendLimits[None], tenant_id: str) -> None:
    if await limits.exhausted(scope=tenant_id):
        raise RuntimeError('daily budget exhausted')
    await workflow_handle.execute(...)
```

`exhausted` rather than `any(s.exhausted for s in await limits.status(...))`: `status` omits
the budgets it cannot resolve, and `any()` over what is left is a brake that passes having
inspected nothing -- which is exactly what a `SpendLimits` whose budgets are all scoped returns when
the scope is missing. `exhausted` raises there instead, naming the budgets that need a
`scope` or a run context. Use `status` for a reading, `exhausted` for a decision.

Making the accrual replay-safe needs the store write to happen inside an activity, which the
capability cannot arrange without depending on `temporalio` and detecting the engine -- so it
belongs in Pydantic AI core rather than here. Tracked in
[#531](https://github.com/pydantic/pydantic-ai-harness/issues/531).

For a per-run token ceiling and nothing else, Pydantic AI's own
[`UsageLimits(total_tokens_limit=...)`](https://pydantic.dev/docs/ai/core-concepts/agent/#usage-limits)
does the same job in-process with no store and no capability. Reach for `Budget(tokens=...,
window='run')` when the same configuration also has to express money, a longer window, a
tenant scope, or a counter shared between processes.

## Tracing

A refusal emits a `spend budget exhausted` span with `spend.budget` and `spend.window`. Accrual emits nothing: a span per model request would double the size of a trace without adding a decision. `spend.scope` is attached only when `RunContext.trace_include_content` is set, since a scope key is usually a tenant or user id and a trace has a wider audience than the application that produced it.

## Specs

`Agent.from_spec` supports the part of the configuration a spec can express:

```yaml
- SpendLimits:
    budgets:
      - {usd: '100', window: day}
      - {usd: '2000', window: month, warn_at: 0.8}
    on_unpriced: raise
```

`store`, `price`, `on_spend`, `clock`, and a budget's `scope` take callables or live objects. A spec naming them is rejected rather than silently ignored, because a spec that promises per-tenant scoping and does not deliver it is worse than one that refuses to load.

The fields above are what `SpendLimits.from_spec` names in its signature, which is also what Pydantic AI reads to generate the spec's JSON schema -- so an editor following the `$schema` line completes and validates them. `BudgetSpec` is the entry shape, exported for anyone building a spec in code.

Source: [`pydantic_ai_harness/spend/`](https://github.com/pydantic/pydantic-ai-harness/tree/main/pydantic_ai_harness/spend/).

## API reference

::: pydantic_ai_harness.spend.SpendLimits

::: pydantic_ai_harness.spend.Budget

::: pydantic_ai_harness.spend.SpendSnapshot

::: pydantic_ai_harness.spend.BudgetStatus

::: pydantic_ai_harness.spend.Spent

::: pydantic_ai_harness.spend.SpendStore

::: pydantic_ai_harness.spend.InMemorySpendStore

::: pydantic_ai_harness.spend.RedisSpendStore

::: pydantic_ai_harness.spend.SpendLimitExceeded

::: pydantic_ai_harness.spend.UnpricedModelError
