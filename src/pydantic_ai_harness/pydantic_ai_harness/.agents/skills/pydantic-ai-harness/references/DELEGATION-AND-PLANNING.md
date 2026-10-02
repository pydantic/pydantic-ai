# Delegation and Planning

Capabilities that structure long or multi-part work: `Planning` (a model-owned task list),
`SubAgents` (one `delegate_task` tool over named child agents), `DynamicWorkflow` (the model
scripts sub-agent calls in a sandbox), `Advisor` (consult a second model mid-run), and
`BackgroundTools` (slow tools run while the agent keeps going). All import from
`pydantic_ai_harness`; only `DynamicWorkflow` needs an extra.

## Which one to use

| Need | Use |
| --- | --- |
| Parent calls one specific agent from a tool it writes itself, full control over prompt and deps | Core agent delegation (`await other.run(..., usage=ctx.usage)` inside a tool; see the `building-pydantic-ai-agents` skill, ORCHESTRATION-AND-INTEGRATIONS.md) |
| Model picks among several named specialists, occasionally, judging each result before the next | `SubAgents` |
| Fan-out, chaining, voting, or retry loops over sub-agents, where intermediate results should not enter the parent context | `DynamicWorkflow` |
| A second opinion from another (often stronger) model, not a task hand-off | `Advisor` |
| Keep a checklist across a long run, or hand a plan from a planner run to an executor run | `Planning` |
| A slow tool should not block the agent | `BackgroundTools` |

Start with `SubAgents` when unsure; its sub-agents move to a `DynamicWorkflow` catalog unchanged.

## Planning

Gives the model plan tools and re-shows the current plan as an ephemeral reminder at the tail of each
request (never written to `message_history`, never in the system prompt), so plan edits do not break
the prompt cache.

```python
from pydantic_ai import Agent

from pydantic_ai_harness import Planning

agent = Agent('test', capabilities=[Planning()])
```

Tools: `write_plan(items)` (whole-list replace), `read_plan()`, `add_task(content, active_form)`,
`update_task_status(task_id, status)`, `update_task_statuses(updates)` (all-or-nothing),
`remove_task(task_id)`. `enable_subtasks=True` adds `add_subtask`, `set_dependency`,
`get_available_tasks` and the `blocked` status. Statuses: `pending`, `in_progress`, `completed`,
`cancelled` (+ `blocked`).

One `write_plan` item (the `PlanItem` model from `pydantic_ai_harness.planning`); `parent_id` and
`depends_on` (a list of ids) need `enable_subtasks=True`:

```python
from pydantic_ai_harness.planning import PlanItem

item = {
    'id': 'mig',  # optional; generated when omitted
    'content': 'Add the database migration',  # the only required field
    'status': 'in_progress',  # default 'pending'
    'active_form': 'Adding the database migration',  # shown while in progress
}
print(PlanItem.model_validate(item).status.value)
#> in_progress
```

Key parameters:

- `guidance=None`: `None` = built-in guidance, `''` = none, a string replaces it.
- `store=None`: `None` = fresh in-memory plan per run. Pass a `PlanStore` to persist.
- `store_resolver=None`: `Callable[[RunContext], PlanStore]`, wins over `store` (per-tenant stores).
- `enable_subtasks=False`, `inject=True` (the tail reminder), `cache_ttl='5m'` (`'5m'` or `'1h'`).
- `tools=None`: allowlist, e.g. `tools=['write_plan']`. Naming a tool the mode does not register
  raises `ValueError`; so does an unknown key in `descriptions`.

Persistence and planner/executor hand-off through a shared store:

```python
from pydantic_ai import Agent

from pydantic_ai_harness import Planning
from pydantic_ai_harness.planning import SqlitePlanStore

store = SqlitePlanStore('plan.db', session='issue-403')
planner = Agent('test', capabilities=[Planning(store=store)])
executor = Agent('test', capabilities=[Planning(store=store)])
# await planner.run('Write a plan. Do not implement.'); await executor.run('Implement the plan.')
```

Stores: `InMemoryPlanStore()`, `SqlitePlanStore(database='.agent-plan.db', *, session='default',
table='plan_items')`, `PostgresPlanStore(pool, *, session=..., table=...)` over your asyncpg pool,
`RedisPlanStore(client, *, session=..., key_prefix='plan', expire_seconds=None)` over your
`redis.asyncio` client. The harness installs no driver.

Gotchas:

- `SqlitePlanStore(':memory:')` raises `ValueError`; use `InMemoryPlanStore`. The SQLite file lives on
  the agent's host, not in the workspace.
- The reminder reads the store on every model request; a store that raises fails the run. Wrap the
  store yourself for fallback behaviour.
- `parent_id`, `depends_on`, and `blocked` are rejected by `write_plan` unless `enable_subtasks=True`.
- Events: `PlanCreatedEvent`, `PlanUpdatedEvent`, `PlanStatusChangedEvent`, `PlanCompletedEvent`,
  `PlanDeletedEvent` from `pydantic_ai_harness.planning`, via `@agent.on_event(...)`. Only mutations
  made through the tools emit them; direct `PlanStore` calls do not. Store `event_emitter=` is deprecated.
- The planner's "do not implement" discipline comes from its instructions and toolsets, not from `Planning`.

## SubAgents

One `delegate_task(agent_name, task)` tool. Each call runs the named agent in a fresh run (no parent
history) and returns `str(result.output)`. The roster is listed in the system prompt.

```python
from pydantic_ai import Agent
from pydantic_ai.usage import UsageLimits

from pydantic_ai_harness import SubAgent, SubAgents

researcher = Agent('test', name='researcher', description='Researches a topic')
writer = Agent('test', name='writer', description='Turns notes into prose')

orchestrator = Agent(
    'test',
    capabilities=[
        SubAgents(
            agents=[
                SubAgent(researcher, usage_limits=UsageLimits(request_limit=20)),
                SubAgent(writer, timeout_seconds=300, max_calls=2),
            ],
        )
    ],
)
```

`SubAgent(agent, name=None, description=None, models=None, usage_limits=None, timeout_seconds=None,
max_calls=None, on_failure=None, contain_errors=None)` sets per-delegate controls.

`SubAgents` parameters that change behaviour:

- `agent_folders=None` (default): no disk loading. Pass `'agents'` to load every Claude-style `*.md`
  and Codex-style `*.toml` under `.agents/agents/`, `.claude/agents/`, and `.codex/agents/` (in that
  order) from the run's workspace. TOML needs `name`, `description`, and `developer_instructions`;
  unknown keys, including sandbox or permission settings, skip the file with a warning. A run without
  a workspace skips it. A sequence of paths reads exactly those (and fails at run start without a workspace).
  Pass `workspace=LocalWorkspaceBackend('/app')` to read definitions from elsewhere.
- `forward_usage=True`: children share the parent's usage, so a parent `usage_limits` bounds the tree.
  A `SubAgent.usage_limits` gives that child isolated accounting (its tokens stop aggregating).
- `inherit_tools=False`: `True` is deprecated. Bind tools directly to explicit child agents, use
  `include_self=True` for a fresh run of the agent with all capabilities and tools bound to it, or
  use `tool_resolver` to give disk agents tools.
- `shared_capabilities=()`: capabilities applied to every child run.
- `models={}`: menu of `key -> model | ModelOption(model, description=, settings=)`. When set,
  `delegate_task` gains a `model` enum argument; `SubAgent(models=['fast'])` restricts a delegate.
- `include_self=False`: `True` lists the running agent as delegate `self` (fresh run with all bound
  capabilities). `max_depth=3` caps nesting, counting the top-level run.
- `tool_retries=2`, `contain_errors=False`, `event_stream_handler=None`, `tool_name='delegate_task'`.
- Disk agents: `agent_overrides={'name': AgentOverride(model=..., effort='high')}` and
  `tool_resolver` (maps frontmatter `tools` names to toolsets); from `pydantic_ai_harness.subagents`.

Gotchas:

- Every delegate needs a name (the agent's `name` or `SubAgent(name=...)`), else `ValueError` at
  construction. Duplicate names raise; with `include_self=True`, so does a delegate named `self` (a disk agent named
  `self` is skipped with a warning).
- Children receive the parent's `deps` (same `AgentDepsT`) and run in the parent's workspace.
- In tests, override a child's model by nesting overrides:
  `with parent.override(model=m1), child.override(model=m2):` (`SubAgent` runs the same `Agent`).
- Timeout, own-budget exhaustion, and `max_calls` exhaustion are soft: a steering message comes back as
  the tool result. Child `ModelRetry`/`UnexpectedModelBehavior` becomes a parent `ModelRetry`. Other
  crashes abort the parent unless `contain_errors=True`.
- `include_self=True` passed in `agent.run(capabilities=...)` raises `UserError`; bind it on the `Agent`.
- Not agent-spec serializable (holds live agents). Events: `DelegationStartEvent`,
  `DelegationEndEvent` (`outcome` is `ok`/`timeout`/`budget`/`failed`/`contained`).

## DelegationReports

Delivers finished background-task reports from a `DelegationTasks` owner into a parent run, as
automated, untrusted task data rather than user messages.

```python
from pydantic_ai_harness.subagents import DelegationReports, DelegationTasks

tasks = DelegationTasks()
reports = DelegationReports(tasks, conversation_id='conversation-1')
```

- `priority='when_idle'` (default) lets an active parent finish its current work first.
- For a host-started, report-only continuation, use `priority='asap'` with `agent.run(None, ...)`, so
  the reports reach the first model request with no synthetic user prompt.
- Reports are marked delivered once consumed and are not replayed. The host owns idle wake-up.

## DynamicWorkflow

The model gets one `run_workflow` tool and writes Python (Monty sandbox) in which each sub-agent is an
`async` function called as `await name(task='...')`. Only the script's last expression returns to the
parent.

```bash
uv add "pydantic-ai-harness[dynamic-workflow]"
```

```python
from pydantic_ai import Agent

from pydantic_ai_harness import DynamicWorkflow

reviewer = Agent('test', name='reviewer', description='Reviews code for bugs.')
summarizer = Agent('test', name='summarizer', description='Summarizes findings.')

orchestrator = Agent(
    'test',
    capabilities=[DynamicWorkflow(agents=[reviewer, summarizer], max_agent_calls=20)],
)
```

What the model writes (illustrative):

```python {test="skip" lint="skip"}
import asyncio

reports = await asyncio.gather(reviewer(task='Review auth.py ...'), reviewer(task='Review db.py ...'))
await summarizer(task='Summarize:\n' + '\n\n'.join(reports))
```

All parameters are keyword-only: `agents` (required), `tool_name='run_workflow'`,
`max_agent_calls=50` (exact host-enforced ceiling on sub-agent runs per parent run, shared across all
`run_workflow` calls), `max_retries=3`, `forward_usage=True`, `inherit_model=False` (children keep their own
models; `True` makes them follow the parent run's resolved model, e.g. after a per-run model override), `sub_agent_usage_limits=None` (applied to each child run),
`resource_limits=None` (backstop 256 MB, no time cap; dict merges, `'unlimited'` disables),
`id`/`description`/`defer_loading`.

Gotchas:

- Agent `name` must be a valid Python identifier; `WorkflowAgent(agent, name=..., description=...)`
  (from `pydantic_ai_harness.dynamic_workflow`) renames without editing the agent.
- `task` is keyword-only in the script. A structured `output_type` arrives as a `dict`: `r['field']`.
- The parent `run(usage_limits=...)` is not forwarded into children; use `sub_agent_usage_limits` or
  `max_agent_calls`. With shared usage and concurrent fan-out, token limits are best-effort.
- Workflows do not nest: do not give catalog agents `DynamicWorkflow`.
- Sandbox: no third-party imports, no clock/randomness/filesystem/env. A child failure raises
  `RuntimeError` in the script; uncaught, the whole script retries.
- Past `max_agent_calls` a call raises in the script, and whatever the script does next (even a
  `try`/`except` that finishes), `run_workflow` returns `{'error': '...exhausted its sub-agent call
  budget...', 'last_error': ..., 'completed': [...]}`; later calls in that run are refused.
- Children do not inherit the parent's workspace: give each its own workspace capability. Children
  gathered in parallel on one checkout can overwrite each other's edits.
- `defer_loading=True` requires a stable `id`. `workflow.reveal(agent)` adds a sub-agent mid-run.
- Not agent-spec serializable.

## Advisor

Lets the executor consult another model. Uses the provider-native advisor tool when executor and
advisor share an Anthropic or OpenRouter provider (string model names only), else a local `advisor`
function tool backed by a separate agent.

```python
from pydantic import BaseModel
from pydantic_ai import Agent

from pydantic_ai_harness import Advisor


class Decision(BaseModel):
    proceed: bool
    risk_score: float


agent = Agent('test', capabilities=[Advisor('anthropic:claude-opus-5-5', max_uses=1)])
typed = Agent('test', capabilities=[Advisor('openai:gpt-5.6-sol', output_type=Decision)])
```

`Advisor(model, *, mode='auto', output_type=str, max_uses=None, max_tokens=None, caching=None,
forward_history=False)`. `mode`: `'auto'`, `'native'` (provider tool; history gets provider-native
parts), or `'local'` (always the `advisor` function tool; history gets a `ToolCallPart` and
`ToolReturnPart`).

The executor decides when to consult. To require a consultation before a verdict, use
`mode='local'` and check this run's history in an output validator:

```python {test="skip" lint="skip"}
@agent.output_validator
def advised(ctx: RunContext[None], output: str) -> str:
    responses = [m for m in ctx.messages if isinstance(m, ModelResponse) and m.run_id == ctx.run_id]
    if output == 'APPROVE' and not any(p.part_kind == 'tool-call' and p.tool_name == 'advisor' for m in responses for p in m.parts):
        raise ModelRetry('Call `advisor` before approving.')
    return output
```

Gotchas:

- Raise at construction: `max_uses < 1`, `max_tokens < 1024`, a non-`str` `output_type` with
  `mode='native'`, `mode='native'` without an `anthropic:`/`openrouter:` string, OpenRouter native
  with `max_uses`. `mode='native'` with an executor on a different provider raises `UserError` at run time.
- A `Model` instance, a non-default `output_type`, or `max_uses` on OpenRouter select local execution.
- Local path: the advisor sees only the consultation prompt (plus completed history with
  `forward_history=True`), never the executor's deps or tools. Its usage counts toward parent limits.
- One `Advisor` per agent (tool name `advisor`). For durable runs use `mode='native'`; local
  execution is unsupported there.

## BackgroundTools

Selected tools start, return a "running in background (task id)" message immediately, and deliver the
result as a follow-up message when done.

```python
import asyncio

from pydantic_ai import Agent

from pydantic_ai_harness import BackgroundTools

agent = Agent('test', capabilities=[BackgroundTools()])


@agent.tool_plain(metadata={'background': True})
async def slow_research(query: str) -> str:
    await asyncio.sleep(60)
    return f'Findings for {query!r}'
```

`BackgroundTools(tools={'background': True})` takes any `ToolSelector`: `'all'`, names, metadata
dict, or `lambda ctx, td: ...`. `metadata={'background': 'optional'}` adds a `run_in_background`
argument the model sets per call (your function never receives it; a tool with its own
`run_in_background` parameter raises `UserError`).

Gotchas:

- `run_stream()`/`run_stream_sync()` wait for background tools but do not deliver results; use
  `run()`, `run_sync()`, `run_stream_events()`, or fully consume `agent.iter()`.
- The run waits for pending tools; a pause or early stop cancels them. Async tools must honour
  cancellation; sync tools cannot be interrupted.
- Delivered results skip tool-result/tool-error hooks; an unexpected tool failure shows the model only the error type.
- Sequential tools and realtime sessions run normally. Do not call `ctx.enqueue()` inside a durable
  activity.

## See also

- https://pydantic.dev/docs/ai/harness/planning/
- https://pydantic.dev/docs/ai/harness/subagents/
- https://pydantic.dev/docs/ai/harness/dynamic-workflow/
- https://pydantic.dev/docs/ai/harness/advisor/
- https://pydantic.dev/docs/ai/harness/background-tools/
- https://pydantic.dev/docs/ai/guides/multi-agent-applications/
- https://pydantic.dev/docs/ai/capabilities/on-demand/
