---
name: pydantic-ai-render-workflows
description: >-
  Use when a Pydantic AI agent's own job is long-running or distributed: a
  research agent that spends minutes writing a report, a batch document
  pipeline, a monitor or scheduled job, parallel model and tool calls, or work
  that must outlive the HTTP request that started it. Covers running those
  agents on Render Workflows, where each supported operation is a registered
  task definition and each invocation is a child task run with its own retry,
  timeout, and managed compute. Use when the user mentions Render Workflows,
  background agent jobs, fan-out, or per-step retries and timeouts. Do not wait
  until they ask to deploy a web service on Render. Do not use for
  Temporal-style replay, checkpoint resume, or crash recovery.
license: MIT
---

# Pydantic AI on Render Workflows

`RenderWorkflows` runs a Pydantic AI agent on a [Render Workflows](https://render.com/docs/workflows) app.
Constructing the agent registers Render task definitions for supported operations (model requests,
static function-tool validation and calls, dynamic/MCP toolsets, event delivery, `@durable_operation`
methods). Inside a workflow, each supported invocation starts a child task run. Use it when the agent's own job is
long-running or distributed; Render retries those task runs rather than replaying the agent loop.

The supported contract is narrower than the agent surface, so read
[the capability README](https://github.com/pydantic/pydantic-ai/tree/main/src/pydantic_ai_harness/pydantic_ai_harness/render/#readme)
before designing around this. The points below are the ones that change a design.

## Install

Follow the README's installation steps to keep Harness, Pydantic AI, and Graph versions aligned. The `render`
extra requires Render SDK 1.2.0 or later. Until a release includes the integration, use the checkout instructions.

## Required boundary

Wrap `agent.run(...)` in `@workflows.task` on the same `Workflows` app passed to `RenderWorkflows`.
Attaching the capability alone does not route calls through Render. A plain `@app.task` does not activate
it. Attach exactly one `RenderWorkflows` per agent.

```python {test="skip"}
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel
from render import TaskContext, Workflows

from pydantic_ai_harness import RenderWorkflows

app = Workflows()
workflows = RenderWorkflows(app)
agent = Agent(TestModel(), name='support', capabilities=[workflows])


@workflows.task
async def support(ctx: TaskContext, prompt: str) -> str:
    del ctx
    return (await agent.run(prompt)).output
```

Generated operations are ordinary registered Render tasks. A caller with Workflow API access can invoke them
directly and bypass checks that exist only in the entry task. Treat those callers as trusted. Keep Render API
credentials on the server and expose an application endpoint that authorizes users and submits validated inputs.

## Task names for capability toolsets

Render task names are persisted workflow identity, so every registered leaf toolset needs a stable `id`. An
explicit `id` set at construction wins. Pydantic AI has no public API for assigning one later, so an unnamed
capability-owned leaf stays inline and receives no independent Render task. An unnamed supported leaf attached
directly by the application is refused before any task registers. Two explicit duplicate IDs reach Pydantic
AI's own uniqueness check.

## Delegation and large tool outputs

`SubAgents` keeps its unnamed `delegate_task` tool inline. To run a delegate's supported model and tool
operations as Render child task runs, explicitly construct that child Agent with `RenderWorkflows` using the
same `Workflows` app as the parent. Successful child operations return usage deltas and buffered events in
the JSON result envelope; the caller applies them once and preserves event order. `max_calls` remains correct
within one active parent task run, but is not a global budget across root-task retries or processes.
Immediate capability events cannot preserve their synchronous decision semantics in a child task and fail
closed; keep tools that emit them inline.

`ToolOutputLimits` also keeps its unnamed helper tool inline. Its `Spill` mode is filesystem-backed: task
runs and root-task retries can use different processes or filesystems, so persisted large outputs need shared
storage. Inside a workflow, return bounded JSON and
keep large artifacts in external durable storage, returning a key the read side can fetch.

## Other design constraints

- Task options (retry, timeout, plan) are fixed at registration. A resolver can assign different options to
  statically known function tools in one named `FunctionToolset`; each eligible tool then receives its own
  task definitions. Returning `False` keeps that function tool inline. Dynamic and MCP tools stay per-toolset.
- Everything crossing the boundary must be JSON encodable, including `deps`, and the arguments of one task
  run must fit Render's documented 4 MB argument cap. Model instances do not cross; their ids do.
- Current callers use protocol v2. New workers accept v1 requests and return effect-free v1 results, so
  deploy workers before sending v2 work during an upgrade.
- Render keeps task state for 30 days, so prompts, responses, tool arguments and results, and deps are
  retained. Read secrets from environment variables inside the child task instead of sending them through.
- Streaming is buffered at the model-task boundary; provider tokens do not stream live across `ctx.run(...)`.
- Retries are task-level. A child-task retry can repeat that task's side effects; an entry-task retry starts
  `agent.run` again. This is not replay or checkpoint resume.
