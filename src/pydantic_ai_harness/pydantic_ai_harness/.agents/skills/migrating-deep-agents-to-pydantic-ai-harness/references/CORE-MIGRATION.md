# Core Migration

Read this reference when the source depends on Deep Agents construction, middleware, tools, state, outputs, or tracing. Confirm behavior against the locked source and target versions.

Deep Agents is an opinionated harness over LangChain's agent loop and the LangGraph runtime. `create_deep_agent` assembles the final prompt, middleware, tools, state, and runtime configuration, so translating its arguments one by one does not preserve that contract.

| Deep Agents contract | Pydantic AI candidate | Required check |
| --- | --- | --- |
| `create_deep_agent` | `Agent(...)` with explicit tools, toolsets, capabilities, outputs, and settings | Snapshot the effective prompt, tool schemas, profile behavior, result, errors, and limits. |
| LangChain `BaseChatModel` | a Pydantic AI provider model or model string | Do not pass the LangChain model through. Translate transport, endpoint, authentication, provider settings, model profiles, retries, and limits. Construct and validate every configured model branch. |
| Deep Agents harness and provider profiles | explicit agent configuration plus Pydantic AI model settings and profiles | Trace provider/model lookup and merge order. Preserve prompt layers, tool descriptions and exclusions, middleware changes, general-purpose-child settings, and model-construction defaults; a Pydantic AI model profile does not configure the Harness composition. |
| graph `.invoke()`, `.ainvoke()`, `.stream()`, `.astream()`, or event callbacks | `run_sync`, `run`, `run_stream`, `run_stream_events`, `iter`, and `event_stream_handler` | Use `run_stream` for final-output streaming; use `run_stream_events` or `iter` when the contract needs the complete tool/event lifecycle. Preserve sync, async, result, message, error, and event shapes at the application boundary. |
| `context_schema` | `deps_type` and `RunContext` | Dependencies are runtime resources and identity, not checkpointed state. |
| custom `state_schema` and reducers | capability-owned run state, an application repository, or `pydantic_graph` | Classify each field's lifecycle and merge semantics. |
| custom or replaced middleware | a core capability, `Hooks`, a wrapper toolset, or a focused Harness capability | Match hook timing, ordering, name-based replacement or exclusion, request mutation, retry, and failure behavior on main and child stacks. |
| structured response | Pydantic `output_type` and an explicit output mode | Test invalid output, final-tool behavior, and streaming. |
| graph cache and provider prompt caching | application caching plus the selected Pydantic AI provider behavior | Separate graph-step caching from provider prompt caching; verify keys, scope, invalidation, cached content, usage, and replay behavior. |
| LangSmith, Langfuse, or other callbacks | retain the existing system, or use OpenTelemetry or Logfire after agreement | Compare correlation, dashboards, evaluations, retention, export, and privacy. Preserve conversation, run, child, tool-call, and external-job identities separately. |

Adapt LangChain `@tool` and `StructuredTool` objects to typed Pydantic functions or toolsets. Preserve their names, schemas, return values, and errors rather than passing framework objects through unchanged.

For MCP, use `pydantic_ai.capabilities.MCP` when provider-native or client-side selection is needed; use `pydantic_ai.mcp.MCPToolset` for client-side MCP connections. Preserve transport, authentication, tool filtering, structured content, connection and session lifetime, elicitation, sampling, retries, and tracing at the integration boundary.

For plain chains, LCEL, or direct LangGraph code in a mixed project, use `$migrating-langchain-to-pydantic-ai` when it is available.

Primary sources: [Deep Agents architecture](https://github.com/langchain-ai/deepagents/blob/main/libs/ARCHITECTURE.md), [`create_deep_agent` source](https://github.com/langchain-ai/deepagents/blob/main/libs/deepagents/deepagents/graph.py), [profiles](https://docs.langchain.com/oss/python/deepagents/profiles), [Pydantic AI capabilities](https://pydantic.dev/docs/ai/capabilities/overview/), and [hooks](https://pydantic.dev/docs/ai/core-concepts/hooks/).
