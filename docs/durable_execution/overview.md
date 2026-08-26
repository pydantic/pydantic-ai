# Durable Execution

Capability authors can also move custom hook work into engine activities, steps, or tasks with [durable capability operations](../capabilities/custom.md#durable-capability-operations).

Third-party runtime authors can use the stable [durable execution backend builder](./backends.md)
to integrate another engine without importing Pydantic AI internals.

Pydantic AI allows you to build durable agents that can preserve their progress across transient API failures and application errors or restarts, and handle long-running, asynchronous, and human-in-the-loop workflows with production-grade reliability. Durable agents have full support for [streaming](../agent.md#streaming-all-events) and [MCP](../mcp/client.md), with the added benefit of fault tolerance.

Pydantic AI officially supports four durable execution solutions:

- [Temporal](./temporal.md)
- [DBOS](./dbos.md)
- [Prefect](./prefect.md)
- [Restate](./restate.md)

These integrations are co-maintained by the Pydantic and vendor teams. The Temporal, DBOS, and Prefect integrations ship with Pydantic AI as [capabilities](../capabilities/overview.md) you attach to an agent; the [Restate](./restate.md) integration lives in the Restate SDK and builds only on Pydantic AI's public interface, so it can also serve as a reference for integrating with other durable systems.

Additional external SDK integrations:

- [Kitaru](./kitaru.md)
- [Apache Airflow](./airflow.md)
