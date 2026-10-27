# Planning

Give an agent a structured, self-updating task list -- without ever invalidating the prompt cache. Optionally persist it, break steps into subtasks with dependencies, and react to changes through events.

[Source](https://github.com/pydantic/pydantic-ai/tree/main/src/pydantic_ai_harness/pydantic_ai_harness/planning/)

## The problem

Long agentic runs drift: the model loses track of what it set out to do and what's left. The usual fix -- keep a running plan and re-inject it into the system prompt each turn -- invalidates the prompt cache. The system prompt sits at the front of the request, so every plan edit changes the cached prefix and forces the whole conversation to be re-processed at full token price.

## The solution

`Planning` gives the model a small toolset that owns the plan. The current plan is surfaced back to the model as an *ephemeral* reminder appended to the tail of each request, led by the capability's single cache breakpoint:

- The reminder is added after the durable history is persisted, so it reaches the model but is never written to `message_history`. No reminders accumulate across turns.
- The reminder opens with a `CachePoint`, which providers attach to the content right before it: the last user prompt, or in a tool loop the latest tool result. That content is what the next request starts with, so on Anthropic, Bedrock and OpenAI each request reads the previous one's prefix back from cache, tool results included, and only the reminder is sent uncached. On OpenRouter the breakpoint stops at the previous user message, so the assistant turns and tool results after it stay outside the explicit prefix.

As with all capability cache breakpoints, provider mapping applies: OpenAI models only receive the `CachePoint` when the model profile enables explicit prompt cache breakpoints, OpenRouter attaches it to the previous user message rather than to a tool result, and models without explicit breakpoints (such as Google's) ignore it. With Anthropic automatic caching (`anthropic_cache`), the automatic breakpoint lands on the reminder, which the next request doesn't contain; the `Planning` breakpoint right before it is the one the next request reads back. A request with no content before the reminder (an `instructions`-only first request) gets the reminder without a breakpoint, since there is no prefix to protect.

A capability listed before `Planning` that also appends a tail part each request puts its content between the durable history and the plan reminder. The `Planning` breakpoint then covers that content too, so its cache entry is only read back while that content stays the same. The earlier capability's own breakpoint still marks the durable history.

So the plan stays current in the model's view while the conversation before the reminder is read from cache, up to where the provider puts the breakpoint; the only added cost is re-reading the reminder each turn.

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

The plan is never injected into the system prompt or instructions. Static usage guidance goes there (cache-stable); only the mutable plan rides the ephemeral tail reminder. Set `inject=False` to disable the reminder entirely. Pydantic AI maps `CachePoint` for models whose profiles support prompt caching; on other models it is ignored.

With a durable-execution capability attached, every plan store call is a journaled capability
operation: the plan read used to build that reminder, and each read and write the plan tools make.
Replay reuses the recorded result instead of calling the store again, so recovering a run doesn't
append a step to a persistent store a second time. Under an engine that runs function tools in workflow code,
like DBOS, a plan tool call runs alone, without overlapping other tool calls, and emits its run
events after its last store call. So its recorded store calls keep the same order on replay,
including relative to the steps an `event_stream_handler` runs for those events.
`Planning` carries the stable default `id='planning'`, so durable recovery works without
configuration.
Engines that run tools and capability operations in a separate worker, like Temporal, don't
carry the run's in-memory plan there: each call resolves its store from the run context. Pass a
persistent `store` or a `store_resolver` (such as `SqlitePlanStore` or `PostgresPlanStore`) to keep
the plan across steps.

## Configuration

```python
from pydantic_ai_harness import Planning

Planning(
    guidance=None,           # static system-prompt guidance; None = default, '' = omit
    cache_ttl='5m',          # TTL for the cache breakpoint leading the plan reminder ('5m' | '1h')
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

## Relationship to `pydantic-ai-todo`

`Planning` incorporates the task-list features of [`pydantic-ai-todo`](https://github.com/vstorm-co/pydantic-ai-todo), a community package by [Vstorm](https://vstorm.co/) -- persistent stores, subtasks, dependencies, and events -- which it supersedes. If you are migrating from `pydantic-ai-todo`, the tools are renamed (`write_todos` -`write_plan`, `read_todos` -`read_plan`, `add_todo` -`add_task`, `update_todo_status(es)` -`update_task_status(es)`, `remove_todo` -`remove_task`; subtask tools keep their names).

## Further reading

- [Pydantic AI capabilities](https://ai.pydantic.dev/capabilities/)
- [Anthropic prompt caching](https://docs.claude.com/en/docs/build-with-claude/prompt-caching)
