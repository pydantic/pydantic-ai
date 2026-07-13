---
title: Dynamic Workflow
description: Let an orchestrator agent coordinate a catalog of sub-agents by writing one sandboxed Python script -- fan-out, chaining, voting, and retry loops in a single tool call.
---

# Dynamic Workflow

`DynamicWorkflow` is for the case where the coordination *between* sub-agents is the actual work. Say you have a few specialists -- one reviews code, one summarizes findings, one writes the final note. Each is easy to call on its own; the hard part is the choreography: review three files at once, keep only the reports that found something, summarize those, and hand the summary to the writer. Reach for this capability when that orchestration involves fan-out, chaining, voting, or retry loops that you do not want to run one model turn at a time, with every intermediate result flowing back through the orchestrator's context.

!!! note "Import path"
    Import this capability from its submodule -- there is no top-level `pydantic_ai_harness` re-export:

    ```python
    from pydantic_ai_harness.dynamic_workflow import DynamicWorkflow
    ```

> The API may change between releases. Where practical, breaking changes ship with a deprecation warning.

## The idea

The usual way to coordinate sub-agents is one tool call per step. The agent calls the reviewer and waits, reads the result, calls the reviewer again, waits again, and so on. Every intermediate result travels back into the agent's context, and every step that depends on the previous one is a separate model turn.

`DynamicWorkflow` takes a different route. You hand it a catalog of named sub-agents, and it gives the model a single tool, `run_workflow`. Inside that tool the model writes ordinary Python: each of your sub-agents is an `async` function it can call, loop over, and combine. The script runs to completion in one tool call, and only its final value comes back to the model. The choreography moves out of the conversation and into code.

If you have met [Code Mode](code-mode.md), this will feel familiar -- the same [Monty](https://github.com/pydantic/monty) sandbox and the same idea: write a script instead of many tool calls. The difference is what the script gets to call. In Code Mode it calls the agent's own tools; here it calls whole sub-agents.

## How this relates to Subagents

The harness has two delegation capabilities. They trade in the same currency -- named, isolated sub-agent runs -- but at different altitudes:

- [`SubAgents`](subagents.md) exposes one `delegate_task(agent_name, task)` tool. Each delegation is its own tool call and its own model turn. It is the right fit when delegations are occasional, or when each result needs the parent's judgment before the next one.
- `DynamicWorkflow` moves the choreography into a script. Fan-out, chaining, voting, and retry loops all run inside one tool call, and intermediate results never enter the parent's context.

Start with `SubAgents` if you are not sure. A `delegate_task` orchestrator converts to a workflow catalog without changing the sub-agents themselves.

## Installation

The script runs inside the Monty sandbox, so install the extra:

```bash
uv add "pydantic-ai-harness[dynamic-workflow]"
```

## Your first workflow

Two sub-agents, one orchestrator:

```python
from pydantic_ai import Agent
from pydantic_ai_harness.dynamic_workflow import DynamicWorkflow

reviewer = Agent('openai:gpt-5', name='reviewer', description='Reviews code for bugs.')
summarizer = Agent('openai:gpt-5', name='summarizer', description='Summarizes findings.')

orchestrator = Agent(
    'openai:gpt-5',
    capabilities=[DynamicWorkflow(agents=[reviewer, summarizer])],
)
```

`reviewer` and `summarizer` are plain agents -- the same `Agent` you already know. Their `name` becomes the function name the model calls in the script, so pick names that are valid Python identifiers. Their `description` tells the model what each one is for; write it the way you would document a function. `DynamicWorkflow(agents=[...])` bundles them into one capability and hands the orchestrator a single `run_workflow` tool.

## What the model does with it

When the orchestrator decides to use the tool, it does not call your sub-agents one at a time. It writes a script:

```python
import asyncio

reports = await asyncio.gather(
    reviewer(task="Review auth.py for bugs:\n<file contents>"),
    reviewer(task="Review parser.py for bugs:\n<file contents>"),
)
await summarizer(task="Summarize these review findings:\n" + "\n\n".join(reports))
```

The parts that matter:

- Each sub-agent is an `async` function. You call it with `await`.
- You pass the work as a single keyword argument, `task`. Always by keyword -- `reviewer(task="...")`, not `reviewer("...")`.
- `asyncio.gather(...)` runs the two reviews concurrently instead of one after the other.
- The last expression's value becomes the result the model sees. The intermediate `reports` list never leaves the sandbox.

Each call is a full `Agent.run`, with its own model loop, message history, tools, and typed output. Two things follow: calls are **isolated** (a sub-agent remembers nothing from an earlier call, so put everything it needs into `task`), and calls **cost tokens and take time** (which is why this capability gives you budgets, below).

## Sub-agents can return structured data

A sub-agent returns whatever its `output_type` produces. The default is a string, but give a sub-agent a Pydantic model and the script receives a `dict`:

```python
from pydantic import BaseModel

class Score(BaseModel):
    value: int
    reason: str

critic = Agent('openai:gpt-5', name='critic', description='Scores an answer 0-10.', output_type=Score)
```

Inside the script the model reads the fields by subscript, the way it would read a JSON object:

```python
result = await critic(task="Score this answer: ...")
result["value"]   # not result.value
```

The catalog the model sees renders each output type as a `TypedDict`, so it knows the fields and reads them by subscript on its own.

## How results come back

The value of the script's last expression becomes the tool result -- the model does not `print()` it.

| The script... | The model receives |
| --- | --- |
| ends in a value, no print | that value directly (or `{}` if it is `None`) |
| prints and ends in a value | `{"output": "<printed text>", "result": <value>}` |
| prints and ends in `None` | `{"output": "<printed text>"}` |

`print()` is for debug logging; it stringifies, so let the last expression carry the real result.

## Choosing sub-agent models

By default each sub-agent uses the model it was constructed with. Set `inherit_model=True` when the host passes a per-run model override to the parent agent (for example from a `/model` command) and every sub-agent dispatch should follow that resolved parent model. Leave it `False` when a sub-agent is deliberately pinned to a different model.

## Keeping it safe: budgets

A sub-agent is non-deterministic, costs tokens, and can fan out into more sub-agents. `DynamicWorkflow` gives you a hard count ceiling, token budgets, and a guard against runaway sandbox scripts.

### `max_agent_calls` -- an exact count

```python
DynamicWorkflow(agents=[...], max_agent_calls=50)  # 50 is the default
```

A hard, host-enforced ceiling on the number of sub-agent runs in one parent run. It is one budget shared across every `run_workflow` call in that run, and it holds exactly even when the script fans out with `asyncio.gather`. When the budget runs out, the workflow stops calling sub-agents and returns a terminal result that includes the sub-agent results that did complete, so nothing you already paid for is wasted. This is the only knob that bounds the number of runs exactly.

### `sub_agent_usage_limits` and `forward_usage` -- bounding cost

`sub_agent_usage_limits` is a `UsageLimits` applied to each sub-agent run. `forward_usage` controls whether the whole tree shares one usage counter:

| `forward_usage` | Counter | What the limit means |
| --- | --- | --- |
| `True` (default) | the parent's `usage` is shared across the tree | a tree-wide cap. Under concurrent fan-out it is best-effort: several sub-agents can pass the check before any of them adds to the count. |
| `False` | each sub-agent run counts on its own | per-run limits. A per-run `total_tokens_limit` of `T` with `max_agent_calls` of `N` bounds the tree to roughly `N * T` tokens. |

!!! warning "The parent `run()` usage limit is not forwarded"
    The `usage_limits` you pass to the parent `run()` is not forwarded into sub-agents -- it is re-checked only at the parent's own request boundaries. To bound sub-agents, set `sub_agent_usage_limits`; for an exact ceiling on the number of runs, use `max_agent_calls`.

### `resource_limits` -- guarding the script itself

These limits guard the orchestration script's own memory and allocations, not the sub-agents it calls. The default backstop is 256 MB and 50 million allocations, with no time limit.

```python
DynamicWorkflow(agents=[...], resource_limits={'max_duration_secs': 30})
```

`max_duration_secs` measures the time your script spends running sandbox code, not wall-clock time. While the script waits on a sub-agent it is suspended and that time does not count, so the cap will not fire on a normal workflow no matter how long the sub-agents take. Its one job is catching a pure-CPU runaway -- a `while True:` loop that never awaits, which none of the sub-agent budgets can stop because it never calls a sub-agent. Pass `'unlimited'` to remove every limit, or a partial dict that merges onto the backstop so you override only the caps you name.

### Workflows do not nest

A sub-agent cannot start its own workflow; a nested `run_workflow` call returns a terminal error instead of running. The practical rule: do not give the sub-agents in your catalog the `DynamicWorkflow` capability. They are the leaves of the orchestration, not orchestrators.

## Renaming a sub-agent: `WorkflowAgent`

By default a sub-agent shows up under its own `name` and `description`. To give it a different name or description for one workflow without editing the agent itself, wrap it in a `WorkflowAgent`:

```python
from pydantic_ai_harness.dynamic_workflow import WorkflowAgent

DynamicWorkflow(
    agents=[
        WorkflowAgent(
            reviewer,
            name='check',
            description='Checks one code change and returns actionable review findings.',
        ),
    ],
)
```

Now the model calls `check(task=...)`. Passing a bare agent is shorthand for wrapping it in a `WorkflowAgent` with no overrides.

## Adding sub-agents mid-run: `reveal()`

The catalog is fixed when a run starts, which keeps it in the prompt-cache prefix across turns. To make a new sub-agent available during a run (say once a fixer agent has been provisioned), keep a reference to the `DynamicWorkflow` instance and call `reveal()`:

```python
workflow = DynamicWorkflow(agents=[reviewer])
orchestrator = Agent('openai:gpt-5', deps_type=MyDeps, capabilities=[workflow])

# later, from the host or from another tool:
workflow.reveal(fixer)
```

The revealed sub-agent becomes callable on the next step; the model learns about it through a short announcement message that carries the new function's signature. The `run_workflow` description itself stays frozen at the agents present when the run started, so a runtime reveal never moves the prompt-cache prefix. `reveal()` is append-only and validates immediately -- a missing name, an invalid identifier, a reserved keyword, or a name collision raises `UserError` at the call site.

## Loading it only when needed: `defer_loading`

`DynamicWorkflow` carries a fair amount of instruction text, and most turns do not need it. Keep it collapsed to a one-line entry until the model actually loads it:

```python
DynamicWorkflow(
    agents=[reviewer, summarizer],
    id='workflow',
    defer_loading=True,
)
```

`defer_loading=True` needs a stable `id`. See [on-demand capabilities](/ai/core-concepts/capabilities/#on-demand-capabilities) for the full picture.

## What runs in the sandbox

The script runs in Monty, a subset of Python. Knowing the edges matters:

- No class definitions, and no third-party libraries.
- Useful standard-library modules: `asyncio`, `math`, `json`, `re`, `typing`. Import what you use; other modules are unavailable or stubbed.
- No wall-clock or timing primitives -- no `asyncio.sleep`, no `datetime.now()`, no `time`.
- `asyncio.gather(...)` runs sub-agents concurrently but does not support `return_exceptions=True`.

Before a script runs it is statically type-checked against the sub-agent signatures. A misspelled function, a positional `task`, or a wrong-typed argument costs one retry, but no sub-agent budget and no sandbox execution.

!!! warning "Errors abort the whole script"
    A sub-agent that raises cannot be caught inside the script -- one failure aborts the whole script and the model retries it. Write scripts where sub-agents do not depend on catching each other's errors. If a script fails after some sub-agents already finished, the retry prompt lists those completed results, so the model can reuse them as plain values instead of paying for the same calls again.

## Observability

The [Logfire](https://pydantic.dev/logfire) trace is the best way to see what a workflow did. Each sub-agent run appears nested under the `run_workflow` span, and the span carries the exact `code` argument the model wrote, so you can read the script it actually ran. Until first-class progress streaming ships, set `event_stream_handler` on each sub-agent `Agent` to watch sub-agent runs inside the one tool call.

## API

```python
DynamicWorkflow(                  # all parameters are keyword-only
    agents=[...],                 # Sequence[AbstractAgent | WorkflowAgent], required
    tool_name='run_workflow',
    max_agent_calls=50,
    max_retries=3,
    forward_usage=True,
    inherit_model=False,          # True -> sub-agents run with the parent run's resolved model
    sub_agent_usage_limits=None,  # UsageLimits per sub-agent run; None -> pydantic-ai default
    resource_limits=None,         # None -> backstop (256 MB, 50M allocs, no time cap);
                                  # 'unlimited' -> off; a dict is merged onto the backstop
    id=None,                      # required when defer_loading=True
    description=None,             # one-line catalog entry shown while deferred
    defer_loading=False,
)

workflow.reveal(agent)            # AbstractAgent | WorkflowAgent; validates before appending

WorkflowAgent(
    agent,                        # Agent, required, positional
    name=None,                    # sandbox function name; falls back to agent.name
    description=None,             # function docstring; falls back to agent.description
)
```

`DynamicWorkflowToolset` and `WorkflowResourceLimits` are also exported from the module for advanced use.

Source: [`pydantic_ai_harness/dynamic_workflow/`](https://github.com/pydantic/pydantic-ai-harness/tree/main/pydantic_ai_harness/dynamic_workflow/).

## Further reading

- [Code Mode](code-mode.md) -- the same sandbox, calling the agent's own tools instead of sub-agents.
- [Subagents](subagents.md) -- one-delegation-per-tool-call sub-agents, without the scripted choreography.
- [Rewriting Bun in Rust](https://bun.com/blog/bun-in-rust) (Bun) -- the same pattern at scale, via Claude Code's dynamic workflows.
- [Capabilities](/ai/core-concepts/capabilities/) and [on-demand capabilities](/ai/core-concepts/capabilities/#on-demand-capabilities).
