---
description: "Add durable execution to Pydantic AI agents so runs survive crashes, restarts and API failures, using Temporal, DBOS, Prefect, Restate or another engine."
---

# Durable Execution

Pydantic AI allows you to build durable agents that can preserve their progress across transient API failures and application errors or restarts, and handle long-running, asynchronous, and human-in-the-loop workflows with production-grade reliability. Durable agents have full support for [streaming](../agent.md#streaming-all-events) and [MCP](../mcp/client.md), with the added benefit of fault tolerance.

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
| [Kitaru](./kitaru.md) | `kitaru[pydantic-ai]` | `KitaruAgent` from `kitaru.adapters.pydantic_ai` |
| [Apache Airflow](./airflow.md) | `apache-airflow-providers-common-ai` | `AgentOperator(..., durable=True)` from `airflow.providers.common.ai.operators.agent` |

Temporal, DBOS, Prefect, Restate, and AWS Lambda are co-maintained by the Pydantic and vendor teams. AWS Lambda, Absurd, and pgtask ship in [Pydantic AI Harness](../harness/index.md); Restate, Kitaru, and Airflow ship in the engine's own package.

!!! warning "One durable execution engine per agent"
    Each engine wraps every model request and tool call as its own durable unit, so attaching a second one, whether to the agent, for a single run, or through an agent spec, raises a [`UserError`][pydantic_ai.exceptions.UserError] before either is bound.

Capability authors can also move custom hook work into engine activities, steps, or tasks with [durable capability operations](../capabilities/custom.md#durable-capability-operations).

Third-party runtime authors can use the stable [durable execution backend builder](./backends.md)
to integrate another engine without importing Pydantic AI internals.

Harness workspace capabilities such as `Coder`, `Shell`, and `FileSystem` also run inside a durable agent; the
[workspace guide](../workspace.md#durable-execution) shows how, and what works on each engine.
