---
title: Spend
description: "Set daily, monthly, or per-tenant USD and token budgets for a Pydantic AI agent, track LLM cost per response, and stop requests once a budget is spent."
---

# Spend

Track what an agent costs, and stop it when a budget is gone.

![SpendLimits checks every budget window before each model request and adds the response's price after it. Once the day window is spent, the next request raises SpendLimitExceeded and is not sent. The store can be in memory, Redis or PostgreSQL; the shared stores give several workers one counter.](../img/harness/spend.svg)

> While Pydantic AI Harness is on 0.x releases, the API may change between minor releases; when it does, deprecation warnings and release-note migration guidance tell you (or your agent) exactly how to upgrade. See the [version policy](index.md#version-policy).

## The problem

A loop that calls a model until a condition it never reaches keeps calling until something stops it. Pydantic AI's [`UsageLimits`][pydantic_ai.usage.UsageLimits] caps tokens, requests and cost for a single `run()`. It does not cover a period longer than one run, a per-tenant share of an allowance, or a counter shared by several workers: with a daily ceiling spread across a queue, each worker sees only its own spend.

Provider usage APIs are polled and report requests only after they are made. That is enough to reconcile a bill, not to refuse a runaway loop's next request.

## The solution

[`SpendLimits`][pydantic_ai_harness.spend.SpendLimits] adds each response's price to every configured window and refuses the next request once a window is spent. [Pricing](#pricing) covers how a response is priced.

```python
from decimal import Decimal

from pydantic_ai import Agent
from pydantic_ai_harness import SpendLimits
from pydantic_ai_harness.spend import Budget

agent = Agent(
    '{{ openai:medium }}',
    capabilities=[SpendLimits(budgets=[Budget(usd=Decimal('100'), window='day')])],
)
```

Past $100 in a UTC day, the next request raises [`SpendLimitExceeded`][pydantic_ai_harness.spend.SpendLimitExceeded].

## Budgets

A [`Budget`][pydantic_ai_harness.spend.Budget] is a ceiling, a period, and optionally a partition. Several apply at once:

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
| `window` | `run`, `conversation`, `day` (default), `month`, `total` |
| `scope` | derives a partition key from the run, so tenants count separately; typed against the agent's `deps` |
| `warn_at` | fraction past which `BudgetStatus.warning` is set; never blocks |
| `name` | distinguishes budgets sharing a window and scope; defaults to `'default'` |
| `retain` | how long the counter is kept after its last write: `'window default'` (default), `'forever'`, or a `timedelta` |

A window rolls over by producing a new store key, so nothing resets at midnight. A `total` counter never expires. `run` and `conversation` windows never roll over, but mint a key per run or conversation, so they expire 24 hours and 30 days after their last write; that expiry hands back the ceiling, and a conversation resumed past its horizon starts from zero. Set `retain='forever'` where a conversation ceiling must hold for the whole conversation, and clean the keys up some other way.

Budgets sharing a `name`, `window`, and `scope` share one counter, which is how one window carries both a USD and a token ceiling; each response is added once. Construction refuses:

- the same `name` and `window` with different `scope` callables: two dimensions (per tenant, per user) that would merge whenever both return the same string. Use different names, or one callable.
- a shared counter with different `retain` values: one counter has one expiry, so declaration order would decide when a `'forever'` ceiling rolls over.
- two `usd` (or two `tokens`) ceilings on one counter: only the smaller would apply.

`Budget` is generic in the agent's dependency type, so with a `deps_type`, a `scope` reading a missing field is a type error, not an `AttributeError` on the first request.

A budget with no ceiling counts and reports but never refuses, for per-tenant accounting with no cap:

```python
from pydantic_ai_harness import SpendLimits
from pydantic_ai_harness.spend import Budget

SpendLimits(budgets=[Budget(window='month', scope=lambda ctx: ctx.deps.tenant_id, name='chargeback')])
```

## What the gate guarantees

Once a recorded response spends a window, `SpendLimits` refuses the next request.

It does not keep spend under the ceiling: the request that crosses the line completes, and concurrent runs can each pass the check before any records anything. A stream stopped part-way is charged for the usage reported so far, which can fall short of the bill, since many providers report output tokens only in the last event. Reconcile against the provider's numbers for exact totals.

## Reading the numbers

```python
from decimal import Decimal

from pydantic_ai import Agent
from pydantic_ai_harness import SpendLimits
from pydantic_ai_harness.spend import Budget, SpendRecordedEvent

limits = SpendLimits(budgets=[Budget(usd=Decimal('100'))])
agent = Agent('{{ anthropic:medium }}', capabilities=[limits])


@agent.on_event(SpendRecordedEvent)
async def show(ctx, event):
    print(f'{event.model} cost ${event.usd}')
```

`SpendRecordedEvent` carries the response usage and serializable budget readings. It is emitted after each response `SpendLimits` records, including one `on_unpriced='raise'` is about to reject. A response a [`FallbackModel`][pydantic_ai.models.fallback.FallbackModel] or the `ModelFallback` capability rejected gets its own event (`failed_attempt=True`) and `on_spend` call with that attempt's model and usage; see [Composition](#composition). Under durable execution an event can be delivered again although the journaled accrual ran once, so keep listeners that write audit or billing records idempotent.

`on_spend` is deprecated but still works, with the same idempotency requirement; move its body to a `SpendRecordedEvent` subscription.

`status()` reads the same numbers outside a run, for a cost display in a UI:

```python {names="defined"}
from pydantic_ai_harness import SpendLimits


async def report(limits: SpendLimits[None]) -> None:
    for status in await limits.status(scope='acme'):
        print(status.budget.name, status.spent.usd, status.exhausted)
```

Without a run context, `run` and `conversation` budgets are omitted, and so are scoped budgets unless `scope=` names the partition. Inside a run, pass `ctx` instead and every budget resolves; passing both is refused.

`expose_tools=True` gives the agent a `get_spend` tool. It is off by default: a tool costs schema tokens on every request, and most applications want the number on a screen, not in the model's context.

## Reacting to a threshold

Spend events are reports, not approval points: they can follow the response carrying the final answer. To act before a request, read `status(ctx)` in a capability's `before_model_request` and hold the run until someone decides:

```python {names="defined"}
import asyncio
from dataclasses import dataclass
from decimal import Decimal

from pydantic_ai import Agent
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.models import ModelRequestContext
from pydantic_ai.tools import RunContext

from pydantic_ai_harness import SpendLimits
from pydantic_ai_harness.spend import Budget

limits = SpendLimits[None](budgets=[Budget(usd=Decimal('100'), warn_at=0.8)])
approvals: asyncio.Queue[bool] = asyncio.Queue()


@dataclass
class ApproveBeforeSpending(AbstractCapability[None]):
    async def before_model_request(
        self, ctx: RunContext[None], request_context: ModelRequestContext
    ) -> ModelRequestContext:
        if any(status.warning for status in await limits.status(ctx)) and not await approvals.get():
            raise RuntimeError('spending past the warning threshold was not approved')
        return request_context


agent = Agent('{{ openai:medium }}', deps_type=type(None), capabilities=[limits, ApproveBeforeSpending()])
```

The gate sees the previous response, which `SpendLimits` accrued inside `wrap_model_request`. It also gates a run's first request, so a threshold crossed by an earlier run carries into the next. A capability listed after it can still skip the request with [`SkipModelRequest`][pydantic_ai.exceptions.SkipModelRequest], so an approval here does not prove a request followed.

The pause holds a coroutine, so it lasts only as long as the process. To pause across processes, gate at a tool call: [deferred tool calls](../deferred-tools.md) end the run with a serializable approval request that a later run resumes. Or stop the run and resume it from its message history, as below.

`budgets` is read on every request, so replacing it after a refusal continues the work against a larger ceiling. Counters are keyed on `name`, `window`, `scope` and the current period, not the ceiling, so what is spent carries over.

A refusal can land mid-run, after tool calls have run and been paid for. Re-running the prompt would repeat them and their side effects; instead, [`capture_run_messages`][pydantic_ai.capture_run_messages] holds the partial history, and a run given that history and no new prompt continues from the refused request.

```python {names="defined"}
import dataclasses
from decimal import Decimal

from pydantic_ai import Agent, capture_run_messages

from pydantic_ai_harness import SpendLimits
from pydantic_ai_harness.spend import Budget, SpendLimitExceeded

limits = SpendLimits[None](budgets=[Budget(usd=Decimal('1'), name='daily')])
agent = Agent('{{ openai:medium }}', deps_type=type(None), capabilities=[limits])


async def ask(prompt: str, ceiling: Decimal) -> str:
    with capture_run_messages() as messages:
        try:
            return (await agent.run(prompt)).output
        except SpendLimitExceeded:
            (budget,) = limits.budgets
            limits.budgets = [dataclasses.replace(budget, usd=ceiling)]
            resumable = list(messages)
    return (await agent.run(message_history=resumable)).output
```

Assigning `budgets` skips the constructor's checks on budget combinations, so keep the same names, windows, and scopes. The new ceiling applies to every run sharing this `SpendLimits`, and nothing lowers it again.

## Sharing a counter across processes

The default [`InMemorySpendStore`][pydantic_ai_harness.spend.InMemorySpendStore] keeps counters in the process: it catches a runaway loop inside one worker and does nothing for a budget spread across a queue. [`RedisSpendStore`][pydantic_ai_harness.spend.RedisSpendStore] and [`PostgresSpendStore`][pydantic_ai_harness.spend.PostgresSpendStore] share one counter across workers:

```python
from decimal import Decimal

from redis.asyncio import Redis

from pydantic_ai_harness import SpendLimits
from pydantic_ai_harness.spend import Budget, RedisSpendStore

store = RedisSpendStore(Redis.from_url('redis://localhost'))
limits = SpendLimits(budgets=[Budget(usd=Decimal('100'), window='day')], store=store)
```

`RedisClient` is a protocol of the two coroutines the store uses, so any compatible client works and the harness adds no dependency. Amounts are integer billionths of a dollar: `INCRBYFLOAT` accumulates rounding error over a busy day's tens of thousands of requests, and millionths would drift too, because an agent's near-identical requests round the same fraction the same way every time.

One Lua script applies every window of a response, so no client sees it part-applied, within one window's four counters or across windows. The exception is overflow, which aborts the script where it happens and keeps the windows already applied; it needs a counter near $9.22 billion. A write that errors has an unknown outcome, not an untried one: the connection can drop after `EVAL` has landed. Nothing retries it.

Counters do not round: `HINCRBY` is 64-bit integer arithmetic on an increment passed as a string, and totals come back from `HMGET` as bulk strings, because Lua would turn `HINCRBY`'s integer replies into doubles that round past `2**53` billionths. Redis refuses a write past the signed 64-bit range, about $9.22 billion per key.

Keys are `{prefix}:budget-key` with literal braces, a Redis Cluster hash tag: all of a store's keys share one slot, so one script can update several, and a cluster cannot spread them across nodes. An empty `prefix` or one containing a brace breaks the tag and is refused at construction. `BudgetStatus.key` omits the prefix. Dedup markers share the namespace under a `|dedup` prefix, which no budget key can start with; only a caller driving the store directly with such a key would collide with a marker and fail with `WRONGTYPE`.

Counters an earlier release wrote under the untagged name are read and added to the tagged total, so upgrading needs no migration. They are added, not moved, because an old-release worker can still write the old name during a rolling deploy. This costs one extra read per key, on reads and writes. The compatibility runs one way only:

- A rolling deploy under-counts while it lasts: old-release workers read only the untagged name, missing what upgraded workers wrote. Keep the deploy short.
- A downgrade loses the tagged counter, which old releases never read. Treat it as a reset.
- Nothing refreshes the old key's expiry, so when it expires the total drops by what it held.

The fallback goes away in 0.28.0, and counters still under the old name stop counting then. Before that, move by hand any window with `retain='forever'` or a horizon longer than the gap between the two releases.

`add_many` carries a token identifying the response, and every bundled store applies a token once per key within `dedup_retain` (see below), recording a marker atomically with the increments (in Redis's script, in Postgres's transaction, or under the in-memory store's lock while its process lives). The token combines the run id and step with a digest of replay-stable response content, usage, and provider identity, excluding clock-derived values and arbitrary provider bookkeeping. It catches recovery that presents an entry without consulting the journal only if the original run and its recovery pass the same `run_id` to `Agent.run`; otherwise Pydantic AI creates a fresh id and the store cannot recognise the entry. Ordinary durable replay is covered by the journaled `_accrue` operation regardless.

Markers cost one small key or row per response per window and are held for `dedup_retain` (a constructor argument on every bundled store, one hour by default), or the window's horizon if shorter. A response presented again later is counted again, an over-count rather than an under-count. `dedup_retain=None` holds no markers and applies every entry. A response counted before the upgrade left no marker, so presenting it again counts it again.

`PostgresSpendStore` suits a deployment that already runs PostgreSQL (alongside a Postgres-backed durable execution engine, say) and would rather not add Redis:

```python {names="defined" typecheck="skip - asyncpg's pooled connections are untyped proxies, see #10853"}
from decimal import Decimal

import asyncpg

from pydantic_ai_harness import SpendLimits
from pydantic_ai_harness.spend import Budget, PostgresSpendStore


async def build_limits() -> tuple[SpendLimits[None], asyncpg.Pool]:
    pool = await asyncpg.create_pool('postgres://localhost/app')
    limits = SpendLimits(budgets=[Budget(usd=Decimal('100'), window='day')], store=PostgresSpendStore(pool))
    return limits, pool
```

Call `build_limits` at startup and close the pool at shutdown; the store does not manage it. It takes the driver-neutral `PostgresPool` protocol from `pydantic_ai_harness.media`, which an `asyncpg` pool satisfies, so the harness adds no driver dependency. Install the driver yourself:

```bash
pip/uv-add asyncpg
```

The store creates two tables on first use: `{table}_counters`, one row per window, and `{table}_dedup`, one row per remembered token and window. Both are keyed by a SHA-256 digest of the budget key (and token), so a caller-supplied id of any length fits the index; the counters table also keeps the key as UTF-8 `BYTEA`, since `TEXT` cannot hold a NUL. `table` defaults to `'spend'` and must be lowercase letters, digits and underscores, start with a letter or underscore, and be at most 47 characters, so derived names fit PostgreSQL's 63-byte identifier limit.

On its first call, each store instance checks for both tables and, if either is missing, creates them with `CREATE TABLE IF NOT EXISTS` under `pg_advisory_xact_lock`, so workers starting together do not collide. Only creation needs `CREATE` on the schema; afterwards `SELECT`, `INSERT`, `UPDATE` and `DELETE` on the tables are enough. There are no migrations: an existing table of either name with a different layout is left alone and the first call fails on a missing column, so pass `table=` if the name is taken. Amounts are integer billionths in a `BIGINT`, with the same range as Redis, and the server refuses a counter passing it.

One transaction applies every window of a response with its dedup markers, so any failure, overflow included, rolls back the whole response and its markers. A concurrent replay of the same token waits on the first transaction's marker and skips once that commits. That wait is `READ COMMITTED` behavior, so the write sets that level itself; under a stricter pool default, the second writer would fail with a serialization error. All markers are claimed before any counter, each in a fixed order, so transactions taking the same rows wait instead of deadlocking. As with Redis, a connection dropped after the commit was sent leaves the outcome unknown, and nothing retries it.

Horizons use the server's clock, as Redis expiry does, so every worker agrees on when a window rolls over. A horizon runs from when its row is written (`clock_timestamp()`), so a write that waited on a lock still gets its full `ttl`. A rolled-over window reads as zero and restarts on its next write. Every `sweep_every` entries it writes (256 by default), a store instance deletes expired counters and markers, after the write commits and skipping rows another transaction holds; a failed sweep is logged as a warning, not raised. The count is in memory per instance, so a process writing fewer than `sweep_every` entries never sweeps; lower `sweep_every`, or delete rows past `expires_at` on your own schedule. `dedup_retain`, and an entry's `ttl` when you drive the store yourself, must be positive or `None`: a zero horizon would write already-expired rows.

The default store is built per capability, so two `SpendLimits` share a counter only if given the same store object. `InMemorySpendStore` loses its counters and deduplication markers with its worker, so durable workflows that can recover on another worker need a shared store.

Store errors are not swallowed. A read error refuses the request, the safe direction. A write error propagates out of the run after the model has answered and been charged, because a swallowed write would leave the counter low and admit requests that should be refused. To keep the answer instead of the count, wrap the store and decide there.

Any object with `get_many` and `add_many` ([`BatchSpendStore`][pydantic_ai_harness.spend.BatchSpendStore]) works, so a DynamoDB counter is a small class, not a fork. Neither method is called with an empty sequence. A custom store must:

- return a total for every key it was handed, keyed by `SpendEntry.key`, including keys it skipped as replays. A missing total raises `UserError` naming this contract, or a non-deterministic scope during durable replay, as the cause.
- read a never-written key as zero rather than leaving it out.
- skip an entry whose `token` was already applied to that key, or recovery outside the durable journal can count one response twice.
- apply the whole call or none of it, or it brings back the split write (day counted, month not) that the batch call exists to prevent.

[`SpendStore`][pydantic_ai_harness.spend.SpendStore], the single-key `get` and `add` pair from 0.17.0, is deprecated. Such a store still works, one window per call, and `SpendLimits` emits one `HarnessDeprecationWarning` at construction naming both losses: windows are applied one at a time, and the token is dropped, so recovery that cannot consult the durable journal has no store-side deduplication. `get` and `add` on `InMemorySpendStore` and `RedisSpendStore` are deprecated too and warn when called. A subclass overriding `get` or `add` but not the batch pair is warned at construction: `SpendLimits` never calls those overrides, so whatever they added (an audit, a mirrored write) stops happening until moved onto `get_many` or `add_many`.

## Pricing

When `price` is unset or returns `None`, each complete response is priced on its own with [genai-prices](https://github.com/pydantic/genai-prices) via [`ModelResponse.cost()`][pydantic_ai.messages.ModelResponse.cost]; cache and tier pricing are per request, so pricing summed usage would be wrong. For a response with omitted usage, any known `response.usage.cost` subtotal is recorded instead.

`price` covers a model the registry does not know, such as a local deployment or a negotiated rate:

```python
from decimal import Decimal

from pydantic_ai_harness import SpendLimits

SpendLimits(price=lambda response: Decimal('0.002') if response.model_name == 'internal-7b' else None)
```

`price` is called first, and a finite, non-negative amount is the complete response price, even when usage is incomplete. Any other amount (negative, `NaN` or infinite), or an exception from `price`, fails the run with `UserError`: a credit would move a budget away from its ceiling, and a `NaN` or infinity is a broken pricing function. The provider billed the response either way, so its tokens and request count are accrued and `on_spend` fires first. When one lifecycle committed several provider responses, an error accruing one response (pricing, reporting, or the store write) waits until the rest accrue.

`None` falls through to the registry. With incomplete usage, a known cost subtotal is recorded but the response stays unpriced, since its full cost is unknown. `on_unpriced` then applies: `'zero'` (the default) increments `Spent.unpriced_requests` and continues; `'raise'` fails the run with [`UnpricedModelError`][pydantic_ai_harness.spend.UnpricedModelError] after recording the subtotal. With `'zero'` and a `usd` budget, [`UnpricedModelWarning`][pydantic_ai_harness.spend.UnpricedModelWarning] is emitted once per model with no complete price. If callers choose the model, prefer `'raise'` or supply `price`.

Known token totals for incomplete usage are lower bounds, even when `price` supplies the cost. When a response omits usage and a token budget is configured, `SpendLimits` emits `UsageLimitUnavailableWarning` if the run can continue; check the provider's usage reporting, or promote the warning to an error to stop when the token ceiling cannot be verified.

Under durable execution, `price` and each budget's `scope` must be deterministic for the same response and run context. Pricing runs in orchestration, outside the journaled accrual, so a changed price can make `on_spend` disagree with the recorded counter or turn a successful recovery into a pricing error. A changed scope selects a different store key than the recorded accrual, which `SpendLimits` reports as a `UserError` naming the determinism requirement.

## Composition

State outlives a run on purpose, so `for_run` is not overridden: a daily budget that reset every run would not be a daily budget. For per-run isolation use `Budget(window='run')`, whose key carries the run id.

`defer_loading=True` is refused: a deferred capability's hooks do not run until the model loads it, so an exhausted budget would not stop a request and the requests made meanwhile would go uncounted.

`SpendLimits` accrues every response the provider billed, even one a hook nested inside it rejects or replaces, so its position among capabilities doesn't change what it counts. A cached response, or one supplied with `SkipModelRequest`, never reached the provider and is not charged.

Core merges a continuation chain (Anthropic `pause_turn`, OpenAI background mode) into one response, which `on_spend` and `SpendRecordedEvent` report once with its combined usage. The counters instead grow at each segment boundary by what the merged response adds over what is already applied, so `price` also sees each intermediate response:

- A boundary `price` cannot price adds no USD and the merged response catches up, so pricing errors and `on_unpriced` apply to the merged response alone.
- Tokens add up to the merged response, and USD to the highest amount any boundary was priced at, so a boundary priced above the merged response is not taken back as a credit.
- A later boundary that adds nothing, such as an idle background poll, is not written.
- A run resuming a turn from a history ending in its `'suspended'` response is charged only for what was billed beyond it, since the earlier run paid for that response.

So a `run` window holds what each run billed, and `conversation`, `day`, `month` and `total` windows hold what the provider billed for the turn, once.

A response a `FallbackModel` rejected with a `fallback_on` response handler was billed too. Each one in `ModelResponse.failed_attempts` is accrued as its own request, ahead of the response carrying it. When every model fails, the attempts on the `FallbackExceptionGroup` are accrued the same way, even if an error hook recovers. A response the [`ModelFallback`](../capabilities/model-fallback.md) capability rejected is accrued as a failed attempt too, with its parts and `provider_url`. An attempt that raised before the provider answered reported no usage and is not charged. `Spent.requests` counts these attempts; core's `RunUsage.requests` counts only responses the agent acted on.

`price` sees an attempt a `FallbackModel` recorded as a `ModelResponse` with the attempt's model, provider, usage and timestamp, but no parts and no `provider_url`, so a `price` reading the output or the endpoint must allow for that. When `price` returns `None`, the cost core recorded on the attempt is used, unless it is negative or not finite, in which case the registry prices it.

**Durable execution.** `SpendLimits` supports Pydantic AI durability capabilities. Its clock read, counter read, and accrual are separate durable operations: Temporal reads the clock in an activity, not in workflow orchestration, and DBOS and Prefect record the same boundaries in their own durable units. On replay, each operation's recorded result is returned without reading the clock or store again, so the response is accrued once and the window key comes from the recorded clock value. When a continuation chain fails partway, a retry replays the completed segments and their accruals, so only newly requested segments reach the store. A `FallbackModel` runs its whole chain in one durable unit, and its `FallbackExceptionGroup` does not cross the durable boundary, so only the rejected attempts it recorded on a response are charged. The `ModelFallback` capability makes each attempt its own durable model operation and records rejected responses on the workflow side, so they are charged as without durable execution.

Attach the durability capability to the same agent as `SpendLimits`. Without `TemporalDurability`, a `SpendLimits` agent run inside a Temporal workflow reads the clock in workflow orchestration, and the sandbox error is translated into advice to attach durability.

The journal covers replay while its records are available. The store token covers recovery that presents the same `SpendEntry` without consulting the recorded accrual, applying it once within `dedup_retain`. A deprecated `SpendStore` drops the token and cannot provide this second layer, and `InMemorySpendStore` provides it only within one process, so use a shared `BatchSpendStore`, such as `RedisSpendStore` or `PostgresSpendStore`, when recovery can land on another worker.

`exhausted()` is a workflow admission check that needs no `RunContext`:

```python {names="defined"}
from collections.abc import Awaitable, Callable

from pydantic_ai_harness import SpendLimits


async def start_if_funded(
    limits: SpendLimits[None], tenant_id: str, start_workflow: Callable[[], Awaitable[object]]
) -> None:
    if await limits.exhausted(scope=tenant_id):
        raise RuntimeError('daily budget exhausted')
    await start_workflow()
```

Prefer it to `any(s.exhausted for s in await limits.status(...))`: `status` omits budgets it cannot resolve, so with every budget scoped and no `scope` given, `any()` returns `False` having checked nothing, while `exhausted` raises `UserError` naming the budgets that need a `scope` or a run context. It reserves nothing; the durable operations then account for the workflow's model responses.

For a ceiling on one run only, Pydantic AI's own [`UsageLimits`](../agent.md#usage-limits) works in-process with no store, and adds input-token limits (cumulative and per request) that `SpendLimits` lacks. Use `Budget(tokens=..., window='run')` when the same configuration also needs a longer window, a tenant scope, or a counter shared between processes.

## Tracing

A refusal emits a `spend budget exhausted` span with `spend.budget` and `spend.window`. Accrual emits nothing: a span per model request would double a trace without recording a decision. `spend.scope` is attached only when `RunContext.trace_include_content` is set, since a scope is usually a tenant or user id and traces have a wider audience than the application.

## Specs

`Agent.from_spec` supports the part of the configuration a spec can express:

```yaml
- SpendLimits:
    budgets:
      - {usd: '100', window: day}
      - {usd: '2000', window: month, warn_at: 0.8}
    on_unpriced: raise
```

`store`, `price`, `on_spend`, `clock`, and a budget's `scope` take callables or live objects; a spec naming them is rejected, not ignored.

`SpendLimits.from_spec` names these fields in its signature, from which Pydantic AI generates the spec's JSON schema, so an editor following the `$schema` line completes and validates them. `BudgetSpec` is the exported entry shape for building a spec in code.

Source: [`pydantic_ai_harness/spend/`](https://github.com/pydantic/pydantic-ai/tree/main/src/pydantic_ai_harness/pydantic_ai_harness/spend/).

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

::: pydantic_ai_harness.spend.PostgresSpendStore

::: pydantic_ai_harness.spend.SpendLimitExceeded

::: pydantic_ai_harness.spend.UnpricedModelError

::: pydantic_ai_harness.spend.UnpricedModelWarning
