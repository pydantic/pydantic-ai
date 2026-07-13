---
title: Planning
description: Give an agent a structured, self-updating task plan through a single write_plan tool, without ever invalidating the prompt cache.
---

# Planning

`Planning` gives the model a structured, self-updating task plan through a single `write_plan` tool -- and surfaces the current plan back to the model every turn without ever invalidating the prompt cache.

[Source](https://github.com/pydantic/pydantic-ai-harness/tree/main/pydantic_ai_harness/planning/)

> The API may change between releases. Where practical, breaking changes ship with a deprecation warning.

## The problem

Long agentic runs drift: the model loses track of what it set out to do and what's left. The usual fix -- keep a running plan and re-inject it into the system prompt each turn -- invalidates the prompt cache. The system prompt sits at the front of the request, so every plan edit changes the cached prefix and forces the whole conversation to be re-processed at full token price.

## The solution

`Planning` gives the model one tool, `write_plan`, that owns the plan (whole-plan replacement -- pass the full list every call, no indices). The current plan is surfaced back to the model as an ephemeral reminder appended to the tail of each request, behind a cache breakpoint:

- The reminder is added in `wrap_model_request`, which runs *after* the durable history is persisted, so it reaches the model but is never written to `message_history`. No reminders accumulate across turns.
- A `CachePoint` is placed immediately *before* the reminder, so the cached prefix (tools + system + real conversation) stays byte-identical turn over turn. Only the reminder falls outside the cache.

So the plan stays current in the model's view while the cached prefix is never invalidated; the only added cost is re-reading the reminder each turn.

## Usage

Construct an `Agent` with `Planning()` in its `capabilities`. The `write_plan` tool is registered automatically, and the static usage guidance is added to the system prompt:

```python
from pydantic_ai import Agent
from pydantic_ai_harness.planning import Planning

agent = Agent('anthropic:claude-sonnet-4-6', capabilities=[Planning()])

result = agent.run_sync('Refactor the auth module and add tests.')
print(result.output)
```

## The tool

| Tool | Purpose |
|---|---|
| `write_plan(items)` | Create or replace the full plan. The model passes the entire ordered list every time, including unchanged, completed, and cancelled steps. |

Each item is a `content` string plus a `status` (`pending`, `in_progress`, `completed`, `cancelled`). The convention -- stated in the guidance and noted in the tool's reply -- is to keep exactly one step `in_progress`.

There is no `get_plan` tool: the current plan is already in the model's context via the tail reminder every turn.

## Why whole-plan replacement

Addressing steps by mutable integer index (insert/remove/reorder) is error-prone for both the code (index bookkeeping) and the model (indices it just saw can go stale within a turn). Restating the whole plan each call removes that: there are no indices to track, and a later call can't corrupt partial state. For short plans the token cost is negligible.

## Caching guarantee

The plan is never injected into the system prompt or instructions. Static usage guidance goes there (cache-stable); only the mutable plan rides the ephemeral tail reminder. Across turns:

- the durable history grows append-only and is replayed byte-identically, so the whole prefix is a cache hit;
- the reminder and its `CachePoint` live only in the per-request copy, so they can't invalidate anything and aren't persisted.

`CachePoint` is supported on Anthropic and Amazon Bedrock; on providers without prompt caching it's simply ignored (nothing to bust).

## Configuration

```python
from pydantic_ai_harness.planning import Planning

Planning(
    guidance=None,      # static system-prompt guidance; None = default, '' = omit
    cache_ttl='5m',     # TTL for the cache breakpoint before the reminder ('5m' | '1h')
)
```

- `guidance` -- static planning guidance added to the system prompt. It is identical on every request, so it stays cache-stable. Leave it as `None` for the built-in default, or set `''` to omit guidance entirely.
- `cache_ttl` -- TTL for the cache breakpoint placed before the plan reminder. One of `'5m'` or `'1h'`.

## Observing the plan

Plan state is per-run (a fresh, isolated plan each run via `for_run`), so it doesn't live on the `Planning()` instance you construct. To see the final plan, read the most recent `write_plan` tool return from the run's messages -- its content is the rendered plan:

```python
from pydantic_ai import Agent
from pydantic_ai.messages import ToolReturnPart
from pydantic_ai_harness.planning import Planning

agent = Agent('anthropic:claude-sonnet-4-6', capabilities=[Planning()])

result = agent.run_sync('Refactor the auth module and add tests.')
plans = [
    part.content
    for message in result.all_messages()
    for part in message.parts
    if isinstance(part, ToolReturnPart) and part.tool_name == 'write_plan'
]
latest_plan = plans[-1] if plans else None
print(latest_plan)
```

## Composition

`Planning` contributes a single leaf toolset (`write_plan`), some static instructions, and a `wrap_model_request` hook. It does not wrap or intercept other toolsets, so it composes cleanly alongside other capabilities and your own tools in the same `Agent(..., capabilities=[...])`.

The tail reminder is only appended when the last message in the request is a `ModelRequest` and the plan is non-empty, so an empty plan adds nothing to the request. Because the reminder and its `CachePoint` live only in the per-request copy and never enter the durable history, `Planning` is safe with message-history replay and does not accumulate stale reminders across turns.

## Agent spec (YAML/JSON)

`Planning` works with Pydantic AI's [agent spec](/ai/core-concepts/agent-spec/) feature for defining agents in YAML or JSON:

```yaml
# agent.yaml
model: anthropic:claude-sonnet-4-6
capabilities:
  - Planning: {}
```

```python
from pydantic_ai import Agent
from pydantic_ai_harness.planning import Planning

agent = Agent.from_file('agent.yaml', custom_capability_types=[Planning])
result = agent.run_sync('...')
print(result.output)
```

Pass `custom_capability_types` so the spec loader knows how to instantiate `Planning`. Arguments can be passed in the YAML too:

```yaml
capabilities:
  - Planning:
      cache_ttl: '1h'
```

## Further reading

- [Pydantic AI capabilities](/ai/core-concepts/capabilities/)
- [Hooks](/ai/core-concepts/hooks/) -- `wrap_model_request` is the ephemeral injection point used here
- [Anthropic prompt caching](https://docs.claude.com/en/docs/build-with-claude/prompt-caching)
- [Code Mode](code-mode.md) -- another prompt-cache-aware harness capability

## API reference

::: pydantic_ai_harness.planning.Planning
