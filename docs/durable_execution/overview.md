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

## How tool calls run durably

Each engine wraps your toolsets so that, by default, every tool call executes inside the engine's durable unit (a Temporal activity, Prefect task, DBOS step, and so on). The shared scaffolding lives in `pydantic_ai.durable_exec._toolset` and is specialized per toolset kind:

- `DurableFunctionToolset` wraps `FunctionToolset` (your `@agent.tool` functions).
- `DurableDynamicToolset` wraps toolsets supplied at run time via `DynamicToolset`.
- `DurableMCPToolset` wraps `MCPToolset` connections.

These wrappers are applied automatically by the engine-specific agent classes; you do not construct them by hand. They exist so the engine can run one tool call per durable unit and replay it deterministically after a failure.

Two extension points control per-tool behavior:

- `resolve_tool_config` maps each tool to either a durable config mapping (merged into the engine's per-operation config, e.g. a Temporal `ActivityConfig`) or `False` to run the tool inline, outside any durable unit. Engines that restrict inline execution reject it here with their own error wording (for example, Temporal requires async tools and forbids inline MCP tools).
- `lifecycle` controls when the wrapped toolset is entered relative to the durable context. Its value is engine- and toolset-specific: `'enter-outside-durable'`, `'enter-always'`, or `'enter-never'` (dynamic toolsets, whose members are managed per step).

If you need custom hook work to survive retries, move it into a durable unit too — see [durable capability operations](../capabilities/custom.md#durable-capability-operations).
