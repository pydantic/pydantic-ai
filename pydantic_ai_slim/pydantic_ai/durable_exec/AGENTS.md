# durable_exec/ Guidelines

Durable execution integrations are first-class compatibility targets.

- Treat Temporal, DBOS, Prefect, Restate, and similar engines as compatibility checks for core agent semantics, not peripheral adapters.
- Preserve run context, dependencies, message history, retries, model/profile selection, and toolset lifecycle across durable boundaries.
- Avoid hidden ordering assumptions, non-serializable state, and runtime-only closures unless the durable wrapper explicitly owns them.
- Prefer generic capabilities/toolsets/models extension points over engine-specific escape hatches.
- When changing graph/tool/output/streaming/MCP behavior, check whether durable wrappers need matching updates and add workflow-level tests where external runtime behavior matters.
