# Planning

Give an agent a structured, self-updating task list -- without rewriting the cached prompt prefix. Optionally persist it, break steps into subtasks with dependencies, and react to changes through events.

[Source](https://github.com/pydantic/pydantic-ai/tree/main/src/pydantic_ai_harness/pydantic_ai_harness/planning/)

> [!NOTE]
> This capability incorporates the task-list features of the standalone [`pydantic-ai-todo`](https://github.com/vstorm-co/pydantic-ai-todo) library -- persistent stores, subtasks, dependencies, and events -- which it supersedes. If you are migrating from `pydantic-ai-todo`, the tools are renamed (`write_todos` -> `write_plan`, `read_todos` -> `read_plan`, `add_todo` -> `add_task`, `update_todo_status(es)` -> `update_task_status(es)`, `remove_todo` -> `remove_task`; subtask tools keep their names).

## The problem

Long agentic runs drift: the model loses track of what it set out to do and what's left. The usual fix -- keep a running plan and re-inject it into the system prompt each turn -- invalidates the prompt cache. The system prompt sits at the front of the request, so every plan edit changes the cached prefix and forces the whole conversation to be re-processed at full token price.

## The solution

`Planning` gives the model a small toolset that owns the plan. The current plan is surfaced back as a reminder that `Planning` appends to `message_history` whenever the plan differs from the last reminder there:

- An unchanged plan adds nothing, and a changed plan appends one reminder with the current plan. Clearing a plan appends a reminder saying there is no plan, so the old one isn't left as the latest.
- History stays append-only, so each request is a prefix of the next. That includes caches that store past the explicit breakpoint, such as Anthropic automatic caching and OpenAI's server-side prefix caching, which match as far as the request bytes agree.
- A `CachePoint` goes on the last user content of each request once a reminder exists, on that request's copy only, so breakpoints don't accumulate in `message_history`.

The cost is one reminder in history per plan change. In a long run the latest reminder can sit well behind the end of the conversation; the model can call `read_plan` to see the current plan.

As with all capability cache breakpoints, provider mapping applies: OpenAI models only receive the `CachePoint` when the model profile enables explicit cache control.

Note that the anchor lands on the last `UserPromptPart` present in the request. A capability listed before `Planning` that appends user content each request (for example `SystemReminders`) displaces the anchor onto that part, so the prefix stays cache-stable only while that content is stable across turns.

So the model sees each version of the plan once, and the cached prefix grows with the conversation instead of being written again on every request.

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

Built-in stores: `InMemoryPlanStore` (default), `SqlitePlanStore` (local file, session-scoped), `PostgresPlanStore` (server database over a caller-owned asyncpg pool), and `RedisPlanStore` (over a caller-owned `redis.asyncio` client). The Postgres and Redis stores take a client you already own, so the harness carries no database driver dependency. Any object implementing the `PlanStore` protocol works. `SqlitePlanStore` keeps its database on the machine running the agent, not in the workspace. It requires a file-backed database; use `InMemoryPlanStore` for ephemeral plans rather than `':memory:'`.

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

Subscribe to typed plan events to react to changes made through the `Planning` tools:

```python
from pydantic_ai import Agent
from pydantic_ai_harness import Planning
from pydantic_ai_harness.planning import PlanCompletedEvent

agent = Agent('anthropic:claude-sonnet-4-6', capabilities=[Planning()])

@agent.on_event(PlanCompletedEvent)
async def announce(ctx, event):
    print('done:', event.item.content)
```

The family contains
`PlanCreatedEvent`, `PlanUpdatedEvent`, `PlanStatusChangedEvent`, `PlanCompletedEvent`, and
`PlanDeletedEvent`; each carries the affected `item` and, for updates, `previous_state`.

Run events come from planning tool paths, including `write_plan`. Direct application mutations on a
`PlanStore` have no run context and do not produce run events. `PlanEventEmitter`, `EventCallback`,
and store `event_emitter` parameters remain supported but are deprecated.

## Why whole-plan replacement

Addressing steps by mutable integer index (insert/remove/reorder) is error-prone for both the code and the model. `write_plan` restates the whole plan each call, so there are no indices to track. Granular edits (`add_task`, `update_task_status`, `remove_task`) instead reference the stable `id` shown by `read_plan`.

## Caching guarantee

The plan is never injected into the system prompt or instructions. Static usage guidance goes there (cache-stable); the plan itself is appended to `message_history` as a reminder when it changes, so nothing already sent is rewritten. Reminders are stored with the rest of the conversation and returned by `new_messages()`, so a UI that renders message history shows them as user content. Set `inject=False` to disable them. Pydantic AI maps `CachePoint` for models whose profiles support prompt caching; on other models it is ignored.

With a durable-execution capability attached, the plan read used to build that reminder is a
journaled capability operation. Replay reuses the recorded plan instead of reading the store again.
`Planning` carries the stable default `id='planning'`, so durable recovery works without
configuration.

## Configuration

```python
from pydantic_ai_harness import Planning

Planning(
    guidance=None,           # static system-prompt guidance; None = default, '' = omit
    cache_ttl='5m',          # TTL for the cache breakpoint anchored on the last durable user content ('5m' | '1h')
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
