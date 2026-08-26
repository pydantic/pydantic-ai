---
title: Spend
description: Track what an agent costs and refuse the next request once a budget is spent, with windows longer than a run, per-tenant scopes, and a counter shared across worker processes.
---

# Spend

Track what an agent costs, and stop it when a budget is gone.

> While Pydantic AI Harness is on 0.x releases, the API may change between minor releases; when it does, deprecation warnings and release-note migration guidance tell you (or your agent) exactly how to upgrade. See the [version policy](index.md#version-policy).

## The problem

A loop that calls a model until a condition it never reaches will keep calling until something stops it. `UsageLimits` in Pydantic AI is that stop for one run: it caps tokens and requests, in token counts, for the duration of a single `run()`. What it does not cover is money, a period longer than one run, a per-tenant share of a shared allowance, or a counter that several worker processes agree on. A daily ceiling spread across a queue's workers is exactly the case where each worker independently believes it has the whole budget.

Provider usage APIs do not close that gap. They are billing and observability pipelines: usage is aggregated after the fact and read by polling, so a number there moves only once the requests behind it have already been made. That is enough to reconcile a ledger and not enough to refuse the request a runaway loop is about to make.

## The solution

`SpendLimits` prices every model response with [`ModelResponse.cost()`](https://pydantic.dev/docs/ai/api/messages/), adds it to each window you configure, and refuses the next request once a window is spent.

```python
from decimal import Decimal

from pydantic_ai import Agent
from pydantic_ai_harness import SpendLimits
from pydantic_ai_harness.spend import Budget

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

from pydantic_ai_harness import SpendLimits
from pydantic_ai_harness.spend import Budget

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
from pydantic_ai_harness import SpendLimits
from pydantic_ai_harness.spend import Budget

SpendLimits(budgets=[Budget(window='month', scope=lambda ctx: ctx.deps.tenant_id, name='chargeback')])
```

## What the gate guarantees

No request **starts** after a budget is exhausted.

Not: that spend stays under the ceiling. The request that crosses the line completes, and concurrent runs can each pass the check before any of them records anything. Three further gaps are worth knowing rather than discovering: a stream the caller abandons part-way never reaches the accounting hook, so its tokens are billed by the provider and invisible here; a capability that answers from a cache without calling a provider is charged the registry price for the response it returns; and a continuation chain (Anthropic `pause_turn`, OpenAI background mode) arrives at the hook as one merged response, which is what Pydantic AI counts as one request too, so its segments are priced on summed usage rather than one at a time -- the difference only shows where pricing is tiered rather than linear. Treat this as a brake on a runaway loop, not as an accounting ledger; reconcile against the provider's own numbers if you need the second thing.

## Reading the numbers

```python
from decimal import Decimal

from pydantic_ai_harness import SpendLimits
from pydantic_ai_harness.spend import Budget, SpendSnapshot


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

from pydantic_ai_harness import SpendLimits
from pydantic_ai_harness.spend import Budget, RedisSpendStore

store = RedisSpendStore(Redis.from_url('redis://localhost'))
limits = SpendLimits(budgets=[Budget(usd=Decimal('100'), window='day')], store=store)
```

It adds no dependency: `RedisClient` is a protocol of the two coroutines used, so any compatible client satisfies it. Amounts are stored as integer billionths of a dollar rather than through `INCRBYFLOAT`, which accumulates rounding error over the tens of thousands of requests a busy day produces. Billionths rather than millionths because the residue does not average out: an agent repeats requests of near-identical shape, so the same fraction rounds the same way every time.

Every window a response counts against is applied as one Lua script, so no other client sees the response part-applied: not across the four counters of one window, and not across the windows themselves. A response counting against a day budget and a month budget is one script rather than two, so there is no failure between them to leave the day counted and the month not. The one exception is the overflow below: it aborts the script where it happens and leaves the windows already applied, which takes a counter near $9.22 billion to reach. Each key also costs a second read for as long as the compatibility fallback below is in place.

A failure *after* the server has run a script does not say whether it committed -- the connection can drop once `EVAL` has landed -- so a write that errors leaves the outcome unknown rather than untried. Nothing retries it: counting a billed response twice is a direction the brake survives, and counting it zero times is not.

The counters do not round. `HINCRBY` is 64-bit integer arithmetic and takes its increment as a string, so what Redis holds is exact, and the totals come back as bulk strings read with `HMGET` rather than as the integer replies `HINCRBY` returns -- those become Lua numbers, which are doubles, and would round a total past `2**53` billionths on the way out. What is left is `HINCRBY`'s own range: a counter passing the signed 64-bit range, around **$9.22 billion** against a single key, which Redis refuses before writing that field.

Keys are `{prefix}:budget-key`, with the braces around the prefix literal. That is a Redis Cluster hash tag, so the slot comes from the prefix alone and every key of one store lands in the same slot, which is what lets one script take several of them. The cost is that a cluster cannot spread a store's keys across its nodes, and that a `prefix` of your own is refused at construction if it is empty or carries a brace of its own, either of which stops the tag being read as one. `BudgetStatus.key` reports the budget key without the prefix, so what it shows is unchanged. The dedup markers below share that namespace, so a key beginning `dedup|` is refused: it would name a marker, which holds a string, and the counter would fail with `WRONGTYPE` rather than accumulate. No budget produces such a key -- the second segment of one is always a window -- so this only reaches a caller driving the store itself.

A counter written by an earlier release, under the untagged name, is read alongside the tagged one and added to it, so an upgrade needs no migration step. Added rather than moved: a move would have to decide when it is complete, and nothing here can know that, since a worker still on the old release can write to the old name at any point in a rolling deploy and a move that already ran would never pick that up. The cost is one extra read per key, on reads and writes alike.

The compatibility only runs one way, which is worth planning around on three counts. A **rolling deploy** under-counts while it lasts: an upgraded worker sees both names, but one still on the old release reads only the untagged one and cannot see what the upgraded workers have written, so it admits requests against a total that is missing them. A **downgrade** loses the tagged counter outright for the same reason, and this release never writes the old name, so nothing carries back. And the old key's **expiry is frozen** at whatever the last old-release write set: nothing here refreshes it, so it goes when it goes and the total drops by what it held. Keep the deploy short, and treat a downgrade as a reset rather than a rollback.

The fallback goes away in 0.28.0. A counter still living under the old name stops being counted at that point, so a window set to `retain='forever'`, or one whose horizon outlasts the gap between the two releases, is worth moving by hand before then.

`add_many` carries a token identifying the response, and `RedisSpendStore` reads a marker for it before the increments and writes it after them, inside the same script. `InMemorySpendStore` remembers the same tokens under its lock, so the default store behaves the same way. A durable engine that re-executes the accrual around a response it already holds -- DBOS recovering a workflow, a Prefect flow retry -- therefore adds it once rather than twice. The token is the provider's own response id where the provider reports one; without one it falls back to the run id, the step, and the response timestamp, which survives a replay only for a `run_id` the caller chose rather than the generated default. Markers are held for `dedup_retain`, a field on both stores, an hour by default, or for the window's own horizon where that is shorter, and cost one small key per response per window. That horizon is the window a replay is recognised in, not the counter's lifetime: a response replayed later than it is counted again, which is the direction to err in, since a brake that trips early survives and one that releases late does not. Set `dedup_retain=None` to hold no markers and apply every entry, which also gives up the guarantee. Recognition starts at the upgrade either way: a response an earlier release counted left no marker behind, so a replay of that one is counted again.

The default store is built per capability, so two `SpendLimits` instances do not quietly share one counter. Pass the same store object to both when you want them to.

A store that fails does not fail quietly. An error reading the counter refuses the request, which is the safe direction. An error writing it propagates out of the run after the model has already answered and been charged. That is deliberate: a swallowed write would drift the counter down and weaken the gate, which is worse than a visible failure. If your deployment would rather keep the answer than the count, wrap the store and decide there.

Any object with `get_many` and `add_many` works, so a Postgres or DynamoDB counter is a small class rather than a fork. Four obligations come with writing one. Return a total for every key you were handed, keyed by `SpendEntry.key`, including one you skipped as a replay: `SpendLimits` indexes the result by key, so a missing one is a `KeyError` part-way through a run. Read a key that was never written as zero rather than leaving it out. Skip an entry whose `token` has already been applied to that key, or an accrual a durable engine re-executes is counted twice. And apply the whole call or none of it -- the guarantee at the top of this section is only as good as the backend behind it, and a store that commits each entry as it goes puts back the split write this seam exists to remove. Neither method is ever handed an empty sequence, so there is no such case to answer for.

`SpendStore`, the single-key `get` and `add` pair released in 0.17.0, is deprecated and removed in 0.28.0: a store of that shape still works, driven one window per call, and emits one `HarnessDeprecationWarning` when the `SpendLimits` holding it is constructed, naming what that costs -- windows applied one at a time, and no token to recognise a repeat by. The single-key `get` and `add` on `InMemorySpendStore` and `RedisSpendStore` go at that release too. A direct call to either does not warn, so this is the notice; reach for `get_many` and `add_many` instead. A subclass that *overrode* one of them without also overriding the batch pair is warned when the store itself is constructed, because that case loses behavior rather than just naming a deprecated method: `SpendLimits` drives `get_many` and `add_many`, so an override on `get` or `add` is never called and whatever it added -- an audit, a mirrored write -- stops happening. Move it onto `get_many` or `add_many`, which is also what makes the warning stop.

## Pricing

Prices come from [genai-prices](https://github.com/pydantic/genai-prices) via `ModelResponse.cost()`, per response: cache and tier pricing are per request, so summing usage across requests and pricing the total gives the wrong number.

A model the registry does not know -- a local deployment, a negotiated rate -- is handled by `price`:

```python
from decimal import Decimal

from pydantic_ai_harness import SpendLimits

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

A replay on another engine is already handled: `SpendEntry.token` identifies the response, so DBOS
recovery and a Prefect flow retry apply it once rather than twice. Temporal is not that case. The run
stops before any accrual, on the clock `SpendLimits` reads to pick the window, so closing it needs a
deterministic clock as well as a store write the capability can make durable without depending on
`temporalio` and detecting the engine -- which is why it belongs in Pydantic AI core rather than here.
Tracked in [#531](https://github.com/pydantic/pydantic-ai-harness/issues/531).

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

::: pydantic_ai_harness.spend.BatchSpendStore

::: pydantic_ai_harness.spend.SpendEntry

::: pydantic_ai_harness.spend.SpendStore

::: pydantic_ai_harness.spend.InMemorySpendStore

::: pydantic_ai_harness.spend.RedisSpendStore

::: pydantic_ai_harness.spend.SpendLimitExceeded

::: pydantic_ai_harness.spend.UnpricedModelError
