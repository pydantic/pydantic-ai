# Durable Execution

Capability authors can also move custom hook work into engine activities, steps, or tasks with [durable capability operations](../capabilities/custom.md#durable-capability-operations).

Third-party runtime authors can use the stable [durable execution backend builder](./backends.md)
to integrate another engine without importing Pydantic AI internals.

Pydantic AI allows you to build durable agents that can preserve their progress across transient API failures and application errors or restarts, and handle long-running, asynchronous, and human-in-the-loop workflows with production-grade reliability. Durable agents have full support for [streaming](../agent.md#streaming-all-events) and [MCP](../mcp/client.md), with the added benefit of fault tolerance.

Pydantic AI officially supports five durable execution solutions, co-maintained by the Pydantic and vendor teams:

- [Temporal](./temporal.md)
- [DBOS](./dbos.md)
- [Prefect](./prefect.md)
- [Restate](./restate.md)
- [AWS Lambda durable functions](https://pydantic.dev/docs/ai/harness/aws-lambda/)

Additional external SDK integrations:

- [Kitaru](./kitaru.md)
- [Apache Airflow](./airflow.md)

## Enqueue Messages from Tools

Tools wrapped as durable units by [`TemporalDurability`][pydantic_ai.durable_exec.temporal.TemporalDurability], [`DBOSDurability`][pydantic_ai.durable_exec.dbos.DBOSDurability], or [`PrefectDurability`][pydantic_ai.durable_exec.prefect.PrefectDurability] can call [`ctx.enqueue()`][pydantic_ai.tools.RunContext.enqueue]. This includes function tools, dynamic-toolset tools, and an MCP toolset's `process_tool_call` hook when their calls are wrapped by the durability capability.

Queued messages are recorded with the tool result and delivered to the agent after the durable unit completes. Recovery or a cache hit restores them with their original IDs and priorities. Loading the same recorded result more than once adds each message only once to a reconstructed agent run, even if the message has already been consumed. A new run can consume those recorded messages again.

If an attempt fails without a recorded tool result, its queued messages are discarded. Tool control-flow outcomes such as [`ModelRetry`][pydantic_ai.exceptions.ModelRetry] are recorded results, so their queued messages are restored before the outcome is handled. Tools running outside a durable unit keep their usual enqueue behavior.

Model, tool-discovery, argument-validation, and event-handler durable units still reject `ctx.enqueue()`, as do the deprecated standalone durable toolset wrappers. Enqueue from workflow or flow code when the operation cannot record messages. Custom event delivery through [`ctx.emit()`][pydantic_ai.tools.RunContext.emit] has separate engine-specific behavior; see [Temporal](./temporal.md#streaming), [DBOS](./dbos.md#streaming), and [Prefect](./prefect.md#streaming).
