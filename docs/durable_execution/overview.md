# Durable Execution

Pydantic AI allows you to build durable agents that can preserve their progress across transient API failures and application errors or restarts, and handle long-running, asynchronous, and human-in-the-loop workflows with production-grade reliability. Durable agents have full support for [streaming](../agents.md#streaming-all-events) and [MCP](../mcp/client.md), with the added benefit of fault tolerance.

Pydantic AI natively supports two durable execution solutions:

- [Temporal](./temporal.md)
- [DBOS](./dbos.md)

These integrations only uses Pydantic AI's public interface, so they also serve as a reference for integrating with other durable systems.
