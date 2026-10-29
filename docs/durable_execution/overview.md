---
description: "Add durable execution to Pydantic AI agents so runs survive crashes, restarts and API failures, using Temporal, DBOS, Prefect, Restate or another engine."
---

# Durable Execution

Use durable execution in production when a run in flight has to survive its process stopping, through a deploy, a crash or a transient API failure, or when a workflow waits hours or days for a person or another system. A durable engine journals the agent's model requests and the tool calls it wraps, and a restarted run replays the completed ones from the journal and retries the one in flight. Which tool calls (function tools, [MCP](../mcp/client.md) servers) an engine wraps for you and how it delivers [streaming](../agent.md#streaming-all-events) differ per engine; each page below says.

!!! note "Durability is not storage"
    A durable engine keeps one run alive when it crashes partway through a step, with retries and journaled model and tool calls. Saving a run, resuming it later and forking it from an earlier step need no engine: [`StepPersistence`](../harness/step-persistence.md) from Pydantic AI Harness snapshots every settled step to a store you choose, the way a checkpointer does, and [Persistence](../persistence.md) lays out the options.

Pydantic AI supports eight durable execution engines, plus a [builder](./backends.md) for any other engine. Each page below has a complete example:

| Engine | Install | Make an agent durable with |
|---|---|---|
| [Temporal](./temporal.md) | `pydantic-ai-slim[temporal]` | `TemporalDurability` from `pydantic_ai.durable_exec.temporal` |
| [DBOS](./dbos.md) | `pydantic-ai-slim[dbos]` | `DBOSDurability` from `pydantic_ai.durable_exec.dbos` |
| [Prefect](./prefect.md) | `pydantic-ai-slim[prefect]` | `PrefectDurability` from `pydantic_ai.durable_exec.prefect` |
| [Restate](./restate.md) | `restate_sdk[serde]` | `RestateAgent` from `restate.ext.pydantic` |
| [AWS Lambda](../harness/aws-lambda.md) | `pydantic-ai-harness[aws-lambda]` | `AWSLambdaDurability` from `pydantic_ai_harness.aws_lambda` |
| [Absurd](../harness/absurd.md) | `pydantic-ai-harness[absurd]` | `AbsurdDurability` from `pydantic_ai_harness.absurd` |
| [pgtask](../harness/pgtask.md) | `pydantic-ai-harness[pgtask]` | `PGTaskDurability` from `pydantic_ai_harness.pgtask` |
| [Apache Airflow](./airflow.md) | `apache-airflow-providers-common-ai` | `AgentOperator(..., durable=True)` from `airflow.providers.common.ai.operators.agent` |

Temporal, DBOS, Prefect, Restate, and AWS Lambda are co-maintained by the Pydantic and vendor teams. AWS Lambda, Absurd, and pgtask ship in [Pydantic AI Harness](../harness/index.md); Restate and Airflow ship in the engine's own package.

!!! warning "One durable execution engine per agent"
    Each engine journals the model requests and the tool calls it wraps as its own durable units, so attaching a second one, whether to the agent, for a single run, or through an agent spec, raises a [`UserError`][pydantic_ai.exceptions.UserError] before either is bound.

[Batched model requests](../batch.md#durable-execution) are recorded differently on each engine: see that section before batching under durable execution.

Capability authors can also move custom hook work into engine activities, steps, or tasks with [durable capability operations](../capabilities/custom.md#durable-capability-operations).

[Append-on-change instructions](../agent.md#appending-instruction-changes) derive their baseline and replacements from persisted history. Instruction callbacks still follow the engine's replay rules: use a durable operation for external reads so replay uses the same values and produces the same history shape.

Third-party runtime authors can use the stable [durable execution backend builder](./backends.md)
to integrate another engine without importing Pydantic AI internals.

Harness workspace capabilities such as `Coder`, `Shell`, and `FileSystem` also run inside a durable agent; the
[workspace guide](../workspace.md#durable-execution) shows how, and what works on each engine.
