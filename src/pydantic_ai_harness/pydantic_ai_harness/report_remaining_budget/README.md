# Report Remaining Budget

Tell the model, on every model request, how much of the run's usage limits and the agent's spend budgets is left, so it can pace its work and answer before a limit stops the run. It only discloses the budget: core [`UsageLimits`](https://pydantic.dev/docs/ai/core-concepts/agent/#usage-limits) and [Spend Limits](../spend_limits/) still enforce it.

[Source](https://github.com/pydantic/pydantic-ai/tree/main/src/pydantic_ai_harness/pydantic_ai_harness/report_remaining_budget/)

## Usage

uv:

```bash
uv add pydantic-ai-harness
```

pip:

```bash
pip install pydantic-ai-harness
```

```python
from decimal import Decimal

from pydantic_ai import Agent
from pydantic_ai_harness import ReportRemainingBudget, SpendLimits
from pydantic_ai_harness.spend_limits import Budget

spend = SpendLimits(budgets=[Budget(usd=Decimal(10), window='day')])
agent = Agent('anthropic:claude-opus-5-5', capabilities=[spend, ReportRemainingBudget(spend_limits=spend)])
```

Without `spend_limits`, only the run's limits are reported. The run's figures come from the `UsageLimits` the run enforces, such as `agent.run(..., usage_limits=UsageLimits(request_limit=20, total_tokens_limit=200_000))` (`RunContext.usage_limits`), so they are not configured twice.

## What the model sees

Each model request carries a status like this:

```text
Budget remaining:
- this run: 8 requests including this one, about 38k total tokens, $1.42
- today: $12.50
- this month (tenant): about 1.2M tokens
```

- **This run** lists each `UsageLimits` ceiling that is set: requests (counting the one being sent), tool calls, input, output and total tokens, and the USD `cost_limit`. `per_request_input_tokens_limit` caps one request's context rather than a running total, so it is not reported. With no limit set, the line is left out.
- **Spend budgets** get one line per `Budget` with a `usd` or `tokens` ceiling, labelled by window (`spend limit for this run`, `this conversation`, `today`, `this month`, `in total`) and by `name` unless it is `'default'`, so a run-window, daily or per-tenant spend remainder is not read as the run's usage limits. A budget with neither ceiling only counts spend and is left out.
- Token counts below 1,000 are exact; larger ones are rounded down (`about 1.3k`, `about 38k`, `about 1.2M`), so the status stays short. USD is rounded down to the cent. Remainders never go below zero, and rounding never reports more than is left.
- A `cost_limit` counts a run with no price data as having spent nothing; core warns with `CostNotFoundWarning` in that case.

A run without `usage_limits=` still has core's default `request_limit=50`, so that is reported. Nothing is sent only when no limit applies: `UsageLimits(request_limit=None)` with no other limit, and no `spend_limits`.

## Caching

The status is enqueued from `before_model_request` as a `RemainingBudgetPart`, a request-scoped [capability part](https://pydantic.dev/docs/ai/capabilities/custom/#capability-parts) in the application's voice, which the model sees as a [request-scoped system prompt](https://pydantic.dev/docs/ai/core-concepts/message-history/#request-scoped-system-prompts), the same channel [Warn Near Limits](../compaction/) uses. Find statuses in history with `isinstance(part, RemainingBudgetPart)`, imported from `pydantic_ai_harness.report_remaining_budget`. It stays in the message history, so history records what each request was told. Anthropic is sent every earlier status and clears them with `clear_at`, so the prefix holds; other providers get each status at the end of its own request only, behind the cache breakpoints, so the prefix before it is still read from cache.

## Composition

- Pass the same `SpendLimits` instance the agent carries. Its store is read on every model request, which is one extra read per request for a shared store such as Redis or Postgres.
- [Warn Near Limits](../compaction/) adds an escalating warning once a threshold is crossed; the two compose.
- Several `ReportRemainingBudget` on one agent each send their own status.

## Telemetry

It emits no spans: it makes no decision, and the status it sends is part of the model request that core's instrumentation already records.
