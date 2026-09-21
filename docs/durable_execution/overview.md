# Durable Execution

Capability authors can also move custom hook work into engine activities, steps, or tasks with [durable capability operations](../capabilities/custom.md#durable-capability-operations).

Third-party runtime authors can use the stable [durable execution backend builder](./backends.md)
to integrate another engine without importing Pydantic AI internals.

Pydantic AI allows you to build durable agents that can preserve their progress across transient API failures and application errors or restarts, and handle long-running, asynchronous, and human-in-the-loop workflows with production-grade reliability. Durable agents have full support for [streaming](../agent.md#streaming-all-events) and [MCP](../mcp/client.md), with the added benefit of fault tolerance.

!!! note "Durability is not storage"
    A durable engine keeps one run alive across crashes and restarts. It does not store your chat threads: saving a conversation and picking it up later is a different problem with a much lighter answer, laid out in [Storage](../storage.md).

Pydantic AI supports seven durable execution solutions, plus a [builder](./backends.md) for any other engine. Five are co-maintained by the Pydantic and vendor teams:

- [Temporal](./temporal.md)
- [DBOS](./dbos.md)
- [Prefect](./prefect.md)
- [Restate](./restate.md)
- [AWS Lambda](https://pydantic.dev/docs/ai/harness/aws-lambda/)

Additional external SDK integrations:

- [Kitaru](./kitaru.md)
- [Apache Airflow](./airflow.md)
