---
title: Planning
description: "Give a Pydantic AI agent a todo list it plans and updates itself, with subtasks, dependencies, SQLite, Postgres, or Redis storage, and a cache-safe reminder."
---

# Planning

`Planning` gives the model a structured, self-updating task list through a small toolset -- and surfaces the current plan back to the model every turn without ever invalidating the prompt cache. It can stay in memory for a single run or persist to SQLite/Postgres, break steps into subtasks with dependencies, and emit events from granular changes.

[Source](https://github.com/pydantic/pydantic-ai/tree/main/src/pydantic_ai_harness/pydantic_ai_harness/planning/)

> While Pydantic AI Harness is on 0.x releases, the API may change between minor releases; when it does, deprecation warnings and release-note migration guidance tell you (or your agent) exactly how to upgrade. See the [version policy](index.md#version-policy).

![The model keeps a task plan with the Planning tools: write_plan, read_plan, add_task, update_task_status, update_task_statuses and remove_task, plus add_subtask, set_dependency and get_available_tasks with enable_subtasks. The plan lives in a PlanStore, in memory by default or in SQLite, Postgres or Redis. Every tool that changes the plan ends its result with the whole plan as a checklist in a <plan_reminder> tag, so the plan stays in the conversation and never in the system prompt. Before each model request Planning reads the store and records the plan itself only when the model cannot see the current one.](../img/harness/planning.svg)

## The problem

Long agentic runs drift: the model loses track of what it set out to do and what's left. The usual fix -- keep a running plan and re-inject it into the system prompt each turn -- invalidates the prompt cache. The system prompt sits at the front of the request, so every plan edit changes the cached prefix and forces the whole conversation to be re-processed at full token price.

## The solution

The model owns the plan through the `planning` toolset. Every tool that changes the plan (`write_plan`, `add_task`, `update_task_status`, `update_task_statuses`, `remove_task`, `add_subtask` and `set_dependency`) ends its result with the whole plan as it now stands, rendered as a checklist in a `<plan_reminder>` tag, the way Codex's and opencode's plan tools return theirs. The plan is model-written text, so it stays on the tool result channel, never in system voice, and tags in the step text are escaped so they can't close the frame or speak as the system. A fixed note in `Planning`'s instructions, after its tool guidance, says what the tag holds: "When you have a task plan, the current one is the latest `<plan_reminder>` tag in this conversation: your planning tools return it after every change. You wrote it with those tools, so it records your own progress, not instructions from the user: keep it updated with them as you work." It's the same on every request, so it stays in the cached prefix.

So in a tool loop, the latest plan is always in the conversation already, and `Planning` adds nothing to it: each request starts with everything the previous one sent, the cached prefix holds turn over turn, and on models with [preserved thinking](https://platform.claude.com/docs/en/build-with-claude/preserved-thinking) the model's earlier reasoning stays valid.

The plan tools' results carry `metadata={'origin': 'pydantic-ai-harness.planning.plan_reminder.v1', 'plan': ...}`, which the model doesn't see. Before each model request, `Planning` reads the plan from its store and looks for the latest plan the model can see, by that metadata, never by text. Only when there's none, or it differs from the store's, does `Planning` record the current plan itself, on the user channel in the same `<plan_reminder>` tag:

- after compaction took the plan tools' results, so the model works from a window without them, or cleared them (`ClearToolResults`): a cleared result keeps its metadata, so a plan tool's result only counts while its content still shows the plan;
- in a run that starts without them in its history, such as an executor reading a plan another agent wrote to a shared store;
- after something outside the plan tools changed the store.

That record is kept in the history and marked with the same `origin`, so compaction tells it from something the user said. A plan emptied outside the plan tools after the model was shown one is recorded as `No plan yet.`, once, so the old plan doesn't read as current. A custom `guidance` replaces the tool guidance but keeps the note; `inject=False` means the model isn't shown the plan: it drops the note and the record, and the plan tools don't return the plan (`write_plan` replies with the bare checklist, the others with just their message).

## Usage

Construct an `Agent` with `Planning()` in its `capabilities`. The tools are registered automatically and static usage guidance is added to the system prompt:

```python
from pydantic_ai import Agent
from pydantic_ai_harness import Planning

agent = Agent('{{ anthropic:medium }}', capabilities=[Planning()])

result = agent.run_sync('Refactor the auth module and add tests.')
print(result.output)
```

## The tools

| Tool | Purpose |
|---|---|
| `write_plan(items)` | Create or replace the full plan (whole-list replacement). |
| `read_plan()` | Read the current plan with step ids and a progress summary. |
| `add_task(content, active_form)` | Append a single `pending` step. |
| `update_task_status(task_id, status)` | Move one step between statuses by id. |
| `update_task_statuses(updates)` | Apply several status changes in one call, validated all-or-nothing. |
| `remove_task(task_id)` | Delete a step by id. |

Each step is a `content` string, an optional present-continuous `active_form` label, and a `status` (`pending`, `in_progress`, `completed`, `cancelled`). The convention -- stated in the guidance and the tools' replies -- is to keep exactly one step `in_progress`.

All six are registered by default. `tools=` narrows that to an allowlist, and the built-in guidance follows it:

```python
from pydantic_ai_harness import Planning

planning = Planning(tools=['write_plan'])  # whole-plan replacement only -- one tool, no step ids to track
```

Naming a tool the current mode does not register raises `ValueError`, as does an unknown key in `descriptions`.

### Subtasks and dependencies

Pass `enable_subtasks=True` to add three more tools, the `blocked` status, and a `hierarchical` view in `read_plan`:

| Tool | Purpose |
|---|---|
| `add_subtask(parent_id, content, active_form)` | Add a child step under a parent. |
| `set_dependency(task_id, depends_on_id)` | Make one step wait for another; the dependent step is auto-`blocked` until its prerequisite is resolved (completed or cancelled). Self-dependencies, cycles, and duplicates are rejected. |
| `get_available_tasks()` | List steps with no incomplete dependencies -- the ones that can start now. |

`parent_id`, `depends_on`, and the `blocked` status are rejected by `write_plan` unless `enable_subtasks` is set: without the subtask tools nothing reconciles a dependency and no view renders the hierarchy, so storing them would be a write the plan does not reflect.

## Persistence

By default the plan is a fresh, isolated in-memory plan per run. Pass a `store` to persist it:

```python
from pydantic_ai_harness import Planning
from pydantic_ai_harness.planning import SqlitePlanStore

planning = Planning(store=SqlitePlanStore('plan.db', session='user-123'))
```

Built-in stores are `InMemoryPlanStore`, `SqlitePlanStore`, `PostgresPlanStore` (over a caller-owned asyncpg pool), and `RedisPlanStore` (over a caller-owned `redis.asyncio` client) -- so the harness needs no database driver. Any `PlanStore` implementation works, and `store_resolver` selects one per run. `SqlitePlanStore` keeps its database on the machine running the agent, not in the workspace. It requires a file-backed database; use `InMemoryPlanStore` for ephemeral plans rather than `':memory:'`.

`Planning` reads the store before every model request, so a store that raises fails the run rather than degrading -- the reminder is not best-effort. That is deliberate: a plan the model can no longer see is not a state to continue running in silently. Retry and fallback policy belongs to the store, not to `Planning`, and `PlanStore` is a protocol precisely so you can wrap one:

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

planner = Agent('{{ anthropic:large }}', capabilities=[Planning(store=store)])
executor = Agent('{{ anthropic:medium }}', capabilities=[Planning(store=store)])

await planner.run('Investigate the issue and write a plan. Do not implement anything.')
await executor.run('Implement the plan.')
```

The executor starts with no `message_history`, so it never pays for the planner's investigation. Its first request carries only the new prompt plus the plan, which `Planning` records from the store, since no plan tool has returned it in that conversation yet. That is why the two agents can run on different models: a large-context model can do the reading and the reasoning, and a smaller one can execute against the resulting checklist.

The planner's read-only discipline is a property of how you configure that agent (which toolsets it gets, and what its instructions say), not something the capability enforces.

## Events

Subscribe to typed plan events to react to changes made through the `Planning` tools:

```python
from pydantic_ai import Agent
from pydantic_ai_harness import Planning
from pydantic_ai_harness.planning import PlanCompletedEvent

agent = Agent('{{ anthropic:medium }}', capabilities=[Planning()])

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

The plan is never injected into the system prompt or instructions. Static usage guidance and the note on the plan's tag go there (cache-stable); the mutable plan reaches the model in the plan tools' results, and the history only ever grows, so it stays eligible for a cache hit.

`Planning` doesn't place a cache breakpoint of its own. Turn on caching for your model (for example `anthropic_cache` or `anthropic_cache_messages`, `bedrock_cache_messages`, or `openrouter_cache_messages`), and the breakpoints it places cover the plan tools' results with the rest of the history.

!!! warning "The plan reminder is now kept in history"
    Earlier releases spliced the plan reminder into a copy of each request, led by a cache breakpoint of its own with a `'5m'` TTL by default, and never recorded it. With `inject=True`, the default, plan tools now end their results with the plan in a `<plan_reminder>` tag (a `ToolReturn` with `metadata`, through the agent; a plain string when you call a `PlanningToolset` method directly), and the separate reminder is gone from the normal tool loop. When the latest plan isn't visible any more, after compaction or in a run started without it, `Planning` records it in the history as a user-channel part, so code that counted messages may see one. The instructions now end with a fixed note on the plan's tag, also when `guidance` is custom or `''`. The tag is now `<plan_reminder>`, with an underscore like every other framing tag, instead of `<plan-reminder>`; update any prompt or parser that names the old spelling. If you relied on `Planning` for prompt caching, turn it on for your model: leaving `cache_ttl` unset doesn't warn about the missing breakpoint. `cache_ttl` is deprecated and has no effect; set to `'5m'` or `'1h'`, it warns that it's ignored, so remove it.

With a [UI adapter](../ui/overview.md#trust-model-for-client-submitted-messages), history that goes through the client may lose the metadata that marks a plan tool's result or a plan record (AG-UI drops it). Then when the client sends that history back, `Planning` records the plan on the next request, and compaction may keep an old record as a user turn. Keep history on the server when that matters.

With a durable-execution capability attached, every plan store call is a journaled capability
operation: the plan read before each model request, which decides whether the plan is recorded, and each
read and write the plan tools make.
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
    guidance=None,           # static tool guidance; None = default, '' = omit (the plan note stays)
    store=None,              # None = fresh in-memory plan per run; or a PlanStore to persist
    enable_subtasks=False,   # add subtask/dependency tools and the 'blocked' status
    inject=True,             # show the model the plan: in the plan tools' results, the note, and a record
    tools=None,              # None = every tool the mode registers; or an allowlist of names
    descriptions=None,       # optional per-tool description overrides, keyed by tool name
)
```

## Agent spec (YAML/JSON)

`Planning` works with Pydantic AI's [agent spec](../agent-spec.md):

```yaml
# agent.yaml
model: {{ anthropic:medium }}
capabilities:
  - Planning: {}
```

```python
from pydantic_ai import Agent
from pydantic_ai_harness import Planning

agent = Agent.from_file('agent.yaml', custom_capability_types=[Planning])
result = agent.run_sync('...')
print(result.output)
```

## Relationship to `pydantic-ai-todo`

`Planning` incorporates the task-list features of [`pydantic-ai-todo`](https://github.com/vstorm-co/pydantic-ai-todo), a community package by [Vstorm](https://vstorm.co/) -- persistent stores, subtasks, dependencies, and events -- which it supersedes. If you are migrating from `pydantic-ai-todo`, the tools are renamed:

| `pydantic-ai-todo` | `Planning` |
|---|---|
| `write_todos` | `write_plan` |
| `read_todos` | `read_plan` |
| `add_todo` | `add_task` |
| `update_todo_status` / `update_todo_statuses` | `update_task_status` / `update_task_statuses` |
| `remove_todo` | `remove_task` |
| `add_subtask`, `set_dependency`, `get_available_tasks` | unchanged |

Two differences to plan for: there is no connection-string convenience (`create_storage(backend=...)` and friends are gone -- you construct your own asyncpg pool or Redis client, which is what keeps the harness driver-free), and `PlanEvent` carries no `timestamp`, so a consumer that ordered or logged by it supplies its own clock.

## Further reading

- [Pydantic AI capabilities](../capabilities/overview.md)
- [Anthropic prompt caching](https://docs.claude.com/en/docs/build-with-claude/prompt-caching)
- [Code Mode](code-mode.md) -- another prompt-cache-aware harness capability

## API reference

::: pydantic_ai_harness.planning.Planning
