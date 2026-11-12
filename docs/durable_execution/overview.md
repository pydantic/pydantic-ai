---
description: "Add durable execution to Pydantic AI agents so runs survive crashes, restarts and API failures, using Temporal, DBOS, Prefect, Restate or another engine."
---

# Durable Execution

Pydantic AI agents run durably, surviving crashes and restarts, on Temporal, DBOS, Prefect, Restate, AWS Lambda, Absurd, pgtask and Apache Airflow.

Use durable execution in production when a run in flight has to survive its process stopping, through a deploy, a crash or a transient API failure, or when a workflow waits hours or days for a person or another system. A durable engine journals the agent's model requests and the tool calls it wraps, and a restarted run replays the completed ones from the journal and retries the one in flight. Which tool calls (function tools, [MCP](../mcp/client.md) servers) an engine wraps for you and how it delivers [streaming](../agent.md#streaming-all-events) differ per engine; each page below says.

!!! note "Durability is not storage"
    A durable engine keeps one run alive when it crashes partway through a step, with retries and journaled model and tool calls. Saving a run, resuming it later and forking it from an earlier step need no engine: [`StepPersistence`](../harness/step-persistence.md) from Pydantic AI Harness snapshots every settled step to a store you choose, the way a checkpointer does, and [Persistence](../persistence.md) lays out the options.

Pydantic AI supports nine durable execution engines, plus a [builder](./backends.md) for any other engine. Each page below has a complete example:

| Engine | Install | Make an agent durable with |
|---|---|---|
| [Temporal](./temporal.md) | `pydantic-ai-slim[temporal]` | `TemporalDurability` from `pydantic_ai.durable_exec.temporal` |
| [DBOS](./dbos.md) | `pydantic-ai-slim[dbos]` | `DBOSDurability` from `pydantic_ai.durable_exec.dbos` |
| [Prefect](./prefect.md) | `pydantic-ai-slim[prefect]` | `PrefectDurability` from `pydantic_ai.durable_exec.prefect` |
| [Restate](./restate.md) | `restate_sdk[serde]` | `RestateAgent` from `restate.ext.pydantic` |
| [AWS Lambda](../harness/aws-lambda.md) | `pydantic-ai-harness[aws-lambda]` | `AWSLambdaDurability` from `pydantic_ai_harness.aws_lambda` |
| [Absurd](../harness/absurd.md) | `pydantic-ai-harness[absurd]` | `AbsurdDurability` from `pydantic_ai_harness.absurd` |
| [pgtask](../harness/pgtask.md) | `pydantic-ai-harness[pgtask]` | `PGTaskDurability` from `pydantic_ai_harness.pgtask` |
| [Azure Durable Task](../harness/azure-durable-task.md) | `pydantic-ai-harness[azure-durable-task]` | `AzureDurableTaskDurability` from `pydantic_ai_harness.azure_durable_task` |
| [Apache Airflow](./airflow.md) | `apache-airflow-providers-common-ai` | `AgentOperator(..., durable=True)` from `airflow.providers.common.ai.operators.agent` |

Temporal, DBOS, Prefect, Restate, and AWS Lambda are co-maintained by the Pydantic and vendor teams. AWS Lambda, Absurd, pgtask, and Azure Durable Task ship in [Pydantic AI Harness](../harness/index.md); Restate and Airflow ship in the engine's own package.

!!! warning "One durable execution engine per agent"
    Each engine journals the model requests and the tool calls it wraps as its own durable units, so attaching a second one, whether to the agent, for a single run, or through an agent spec, raises a [`UserError`][pydantic_ai.exceptions.UserError] before either is bound.

[Batched model requests](../batch.md#durable-execution) are recorded differently on each engine: see that section before batching under durable execution.

Capability authors can also move custom hook work into engine activities, steps, or tasks with [durable capability operations](../capabilities/custom.md#durable-capability-operations).

[Append-on-change instructions](../agent.md#appending-instruction-changes) derive their baseline and replacements from persisted history. Instruction callbacks still follow the engine's replay rules: use a durable operation for external reads so replay uses the same values and produces the same history shape.

Third-party runtime authors can use the stable [durable execution backend builder](./backends.md)
to integrate another engine without importing Pydantic AI internals.

Harness workspace capabilities such as `Coder`, `Shell`, and `FileSystem` also run inside a durable agent; the
[workspace guide](../workspace.md#durable-execution) shows how, and what works on each engine.

## Running a tool outside the durable unit

An engine runs each function tool call in a durable unit (an activity, step or task) so a restarted run reuses its recorded result. To run a tool in the workflow code itself instead, set `metadata={'durable': False}` on it. The tool then runs again on every replay, so keep it deterministic: no I/O, clock or randomness of its own. What it does through a [workspace](../workspace.md#durable-execution) call or a durable agent's run is still recorded, unit by unit. Use it for a tool that has to see live run state, or that starts work which should be recorded step by step rather than as one unit:

```python {title="durable_opt_out.py" test="skip"}
from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import LocalWorkspace
from pydantic_ai.durable_exec.temporal import TemporalDurability
from pydantic_ai.toolsets import FunctionToolset

toolset = FunctionToolset(id='files')


@toolset.tool(metadata={'durable': False})  # (1)!
async def copy_file(ctx: RunContext, source: str, destination: str) -> str:
    content = await ctx.workspace.read_bytes(source)  # (2)!
    await ctx.workspace.write_bytes(destination, content)
    return f'Copied {source} to {destination}.'


agent = Agent(
    '{{ openai:medium }}',
    name='files',
    toolsets=[toolset],
    capabilities=[LocalWorkspace('.'), TemporalDurability()],  # (3)!
)
```

1. The same metadata keeps the tool out of its durable unit on every engine.
2. Each workspace call is recorded as a unit of its own, so a run that restarts after the read reuses its result and only writes.
3. Swap in any other engine's durability capability; the tool's metadata stays the same.

To set the metadata on tools you don't define yourself, such as a capability's, use [`SetToolMetadata(tools=[...], durable=False)`][pydantic_ai.capabilities.SetToolMetadata]. `durable=True` puts back the default.

Only async function tools can leave their durable unit this way. On a sync tool, an [MCP](../mcp/client.md) tool or a dynamic toolset's tool that the engine runs in a durable unit, `{'durable': False}` raises a [`UserError`][pydantic_ai.exceptions.UserError] when the tool is called, because those run in a thread or perform I/O. So does a value other than `True` or `False`. DBOS runs function tools in workflow code without wrapping them, so it checks neither on them: keep an agent you may move to another engine to async tools.

An engine's own metadata key wins when a tool sets it, so `metadata={'temporal': ActivityConfig(...), 'durable': False}` still runs the tool in a Temporal activity. Engine keys also accept the engine's unit config, and some allow tools the engine-neutral key refuses. `{'durable': False}` in turn wins over per-tool config set in code, such as Azure Durable Task's and Render Workflows' `resolve_tool_options`:

| Engine | Where `{'durable': False}` runs the tool | Engine's own key |
|---|---|---|
| [Temporal](./temporal.md#per-tool-activity-config) | workflow code | `'temporal'` |
| [DBOS](./dbos.md) | workflow code, where DBOS already runs function tools, so it doesn't check the key on them | none; `{'dbos': {'child_workflow': ...}}` selects a [child workflow](./dbos.md#child-workflows) |
| [Prefect](./prefect.md) | flow code | `'prefect'` |
| [AWS Lambda](../harness/aws-lambda.md) | handler code, outside any step | `'aws_lambda'` |
| [Azure Durable Task](../harness/azure-durable-task.md) | orchestrator code | `'azure_durable_task'` |
| [Absurd](../harness/absurd.md) | task handler code, outside any step | none |
| [pgtask](../harness/pgtask.md) | task code, outside any step | none |
| [Render Workflows](../harness/render-workflows.md) | the parent task run | `'render_workflows'`, whose `False` only lets a per-run toolset through; per tool, `resolve_tool_options` can return `False` |

Restate and Apache Airflow ship their own integrations; see their pages for how they wrap tools.
