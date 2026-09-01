# Planning

Give an agent a structured, self-updating task list -- without ever invalidating the prompt cache. Optionally persist it, break steps into subtasks with dependencies, and react to changes through events.

[Source](https://github.com/pydantic/pydantic-ai-harness/tree/main/pydantic_ai_harness/planning/)

> [!NOTE]
> This capability incorporates the task-list features of the standalone [`pydantic-ai-todo`](https://github.com/vstorm-co/pydantic-ai-todo) library -- persistent stores, subtasks, dependencies, and events -- which it supersedes. If you are migrating from `pydantic-ai-todo`, the tools are renamed (`write_todos` -> `write_plan`, `read_todos` -> `read_plan`, `add_todo` -> `add_task`, `update_todo_status(es)` -> `update_task_status(es)`, `remove_todo` -> `remove_task`; subtask tools keep their names).

## The problem

Long agentic runs drift: the model loses track of what it set out to do and what's left. The usual fix -- keep a running plan and re-inject it into the system prompt each turn -- invalidates the prompt cache. The system prompt sits at the front of the request, so every plan edit changes the cached prefix and forces the whole conversation to be re-processed at full token price.

## The solution

`Planning` gives the model a small toolset that owns the plan. The current plan is surfaced back to the model as an *ephemeral* reminder appended to the tail of each request, with a cache breakpoint after its stable opening tag:

- The reminder is added after the durable history is persisted, so it reaches the model but is never written to `message_history`. No reminders accumulate across turns.
- A `CachePoint` follows the stable `<plan-reminder>` opening tag, so the cached prefix (tools + system + real conversation + that tag) stays byte-identical turn over turn. Only the mutable reminder content falls outside the cache.

So the plan stays current in the model's view while the cached prefix is never invalidated; the only added cost is re-reading the reminder each turn.

```python
from pydantic_ai import Agent
from pydantic_ai_harness import Planning

agent = Agent('anthropic:claude-sonnet-4-6', capabilities=[Planning()])

result = agent.run_sync('Refactor the auth module and add tests.')
print(result.output)
```

## The tools

| Tool | Purpose |
|---|---|
| `write_plan(items)` | Create or replace the full plan (whole-list replacement -- no indices to track). |
| `read_plan()` | Read the current plan with step ids and a progress summary. |
| `add_task(content, active_form)` | Append a single `pending` step. |
| `update_task_status(task_id, status)` | Move one step between statuses by id. |
| `update_task_statuses(updates)` | Apply several status changes in one call, validated all-or-nothing. |
| `remove_task(task_id)` | Delete a step by id. |

Each step is a `content` string, an optional present-continuous `active_form` label, and a `status` (`pending`, `in_progress`, `completed`, `cancelled`). The convention -- stated in the guidance and the tools' replies -- is to keep exactly one step `in_progress`.

All six are registered by default. `tools=` narrows that to an allowlist, and the built-in guidance follows it:

```python
Planning(tools=['write_plan'])  # whole-plan replacement only -- one tool, no step ids to track
```

Naming a tool the current mode does not register raises `ValueError`, as does an unknown key in `descriptions`.

### Subtasks and dependencies

Pass `enable_subtasks=True` to add three more tools and the `blocked` status:

| Tool | Purpose |
|---|---|
| `add_subtask(parent_id, content, active_form)` | Add a child step under a parent. |
| `set_dependency(task_id, depends_on_id)` | Make one step wait for another; the dependent step is auto-`blocked` until its prerequisite is resolved (completed or cancelled). Self-dependencies, cycles, and duplicates are rejected. |
| `get_available_tasks()` | List steps with no incomplete dependencies -- the ones that can start now. |

## Persistence

By default the plan lives in memory for the duration of a single run (a fresh, isolated plan per run via `for_run`). Pass a `store` to persist it, or a `store_resolver` to pick one per run:

```python
from pydantic_ai_harness import Planning
from pydantic_ai_harness.planning import SqlitePlanStore

agent_store = SqlitePlanStore('plan.db', session='user-123')
planning = Planning(store=agent_store)
```

Built-in stores: `InMemoryPlanStore` (default), `SqlitePlanStore` (local file, session-scoped), `PostgresPlanStore` (server database over a caller-owned asyncpg pool), and `RedisPlanStore` (over a caller-owned `redis.asyncio` client). The Postgres and Redis stores take a client you already own, so the harness carries no database driver dependency. Any object implementing the `PlanStore` protocol works. `SqlitePlanStore` requires a file-backed database; use `InMemoryPlanStore` for ephemeral plans rather than `':memory:'`.

The tail reminder reads the store on every model request, so a store that raises fails the run rather than degrading -- the reminder is not best-effort. That is deliberate: a plan the model can no longer see is not a state to continue running in silently. Retry and fallback policy belongs to the store, not to `Planning`, and `PlanStore` is a protocol precisely so you can wrap one:

```python
class BestEffort:
    """Serve the last known plan when the backing store is unreachable."""

    def __init__(self, inner: PlanStore) -> None:
        self._inner, self._last = inner, []

    async def get_items(self) -> list[PlanItem]:
        try:
            self._last = await self._inner.get_items()
        except ConnectionError:
            pass
        return self._last

    # ... delegate the other five methods to `self._inner`
```

### Planning and executing in separate runs

A shared store is the whole handoff mechanism between two runs. One agent writes the plan, a second one executes it, and the plan is the only state that crosses between them:

```python
store = SqlitePlanStore('plan.db', session='issue-403')

planner = Agent('anthropic:claude-opus-4-7', capabilities=[Planning(store=store)])
executor = Agent('anthropic:claude-sonnet-4-6', capabilities=[Planning(store=store)])

await planner.run('Investigate the issue and write a plan. Do not implement anything.')
await executor.run('Implement the plan.')
```

The executor starts with no `message_history`, so it never pays for the planner's investigation. Its first request carries only the new prompt plus the plan reminder, which the capability rebuilds from the store. That is why the two agents can run on different models: a large-context model can do the reading and the reasoning, and a smaller one can execute against the resulting checklist.

The planner's read-only discipline is a property of how you configure that agent (which toolsets it gets, and what its instructions say), not something the capability enforces.

## Events

Give a store a `PlanEventEmitter` to react to changes -- surface progress in a UI, mirror steps to a tracker, notify a channel on completion:

```python
from pydantic_ai_harness.planning import InMemoryPlanStore, PlanEventEmitter

emitter = PlanEventEmitter()

@emitter.on_completed
async def announce(event):
    print('done:', event.item.content)

store = InMemoryPlanStore(event_emitter=emitter)
```

Emitted types: `created`, `updated`, `status_changed`, `completed`, `deleted`.

Events come from the granular tools (`add_task`, `update_task_status`, `add_subtask`, ...). `write_plan` is a bulk whole-plan replacement and is **event-silent**, so a UI driven purely off events won't see plans the model builds or rewrites with `write_plan`. Read the plan after the run too, or steer the model toward the granular tools when you need live event coverage.

## Why whole-plan replacement

Addressing steps by mutable integer index (insert/remove/reorder) is error-prone for both the code and the model. `write_plan` restates the whole plan each call, so there are no indices to track. Granular edits (`add_task`, `update_task_status`, `remove_task`) instead reference the stable `id` shown by `read_plan`.

## Caching guarantee

The plan is never injected into the system prompt or instructions. Static usage guidance goes there (cache-stable); only the mutable plan rides the ephemeral tail reminder. Set `inject=False` to disable the reminder entirely. Pydantic AI maps `CachePoint` for models whose profiles support prompt caching; on other models it is ignored.

With a durable-execution capability attached, the plan read used to build that reminder is a
journaled capability operation. Replay reuses the recorded plan instead of reading the store again.
`Planning` carries the stable default `id='planning'`, so durable recovery works without
configuration.

## Configuration

```python
from pydantic_ai_harness import Planning

Planning(
    guidance=None,           # static system-prompt guidance; None = default, '' = omit
    cache_ttl='5m',          # TTL for the cache breakpoint after the stable opening tag ('5m' | '1h')
    store=None,              # None = fresh in-memory plan per run; or a PlanStore to persist
    enable_subtasks=False,   # add subtask/dependency tools and the 'blocked' status
    inject=True,             # surface the current plan as a cache-safe tail reminder
    tools=None,              # None = every tool the mode registers; or an allowlist of names
    descriptions=None,       # optional per-tool description overrides, keyed by tool name
)
```

## Agent spec (YAML/JSON)

`Planning` works with Pydantic AI's [agent spec](https://ai.pydantic.dev/agent-spec/):

```yaml
# agent.yaml
model: anthropic:claude-sonnet-4-6
capabilities:
  - Planning: {}
```

```python
from pydantic_ai import Agent
from pydantic_ai_harness import Planning

agent = Agent.from_file('agent.yaml', custom_capability_types=[Planning])
```

## Further reading

- [Pydantic AI capabilities](https://ai.pydantic.dev/capabilities/)
- [Anthropic prompt caching](https://docs.claude.com/en/docs/build-with-claude/prompt-caching)
