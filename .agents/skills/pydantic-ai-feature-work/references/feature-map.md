# Pydantic AI Slim Feature Map

Purpose: orient learning and issue triage around the user-facing feature surface of `pydantic-ai-slim`.

Primary docs entry points:

- Agents: `docs/agent.md`
- Dependencies: `docs/dependencies.md`
- Function tools: `docs/tools.md`
- Advanced tools: `docs/tools-advanced.md`
- Toolsets: `docs/toolsets.md`
- Deferred tools: `docs/deferred-tools.md`
- Native tools: `docs/native-tools.md`
- Common tools: `docs/common-tools.md`
- Output: `docs/output.md`
- Capabilities: `docs/capabilities.md`
- Models/providers: `docs/models/overview.md`
- Message history: `docs/message-history.md`
- MCP: `docs/mcp/overview.md`, `docs/mcp/client.md`, `docs/mcp/fastmcp-client.md`, `docs/mcp/server.md`
- Input/multimodal: `docs/input.md`
- Thinking: `docs/thinking.md`
- Hooks: `docs/hooks.md`
- Direct model requests: `docs/direct.md`
- Testing: `docs/testing.md`
- Embeddings: `docs/embeddings.md`
- Logfire/OpenTelemetry: `docs/logfire.md`
- UI event streams: `docs/ui/overview.md`, `docs/ui/ag-ui.md`, `docs/ui/vercel-ai.md`
- Durable execution: `docs/durable_execution/overview.md`
- Agent specs/config: `docs/agent-spec.md`
- Extensibility: `docs/extensibility.md`

## Feature Groups

### Agent Runtime

- `Agent` is the main user API: model, instructions, tools/toolsets, capabilities, deps, output type, settings, retries, usage limits, metadata.
- Run modes: `run`, `run_sync`, `run_stream`, `run_stream_sync`, `run_stream_events`, `iter`.
- Agent runs can be continued with message history and `conversation_id`.
- V2 execution direction: `run_stream_events` is becoming an async context manager that yields an async iterator, so early exits can deterministically clean up background run tasks. Source: [PR #5440](https://github.com/pydantic/pydantic-ai/pull/5440).

### Instructions, Deps, Context

- Instructions can be static, dynamic, templated, agent-level, run-level, capability-provided, or toolset-provided.
- Dependencies are runtime values exposed via `RunContext`.
- `RunContext` also carries model, usage, messages, prompt, run step, retry state, tool call info, validation context, and agent reference.

### Tools

- Function tools are regular Python functions exposed to the model.
- Registration surfaces: `@agent.tool`, `@agent.tool_plain`, `tools=[...]`, `Tool(...)`, `FunctionToolset`.
- Tool schema comes from signatures, annotations, Pydantic schema generation, and docstrings.
- Tools support retries, validation, timeouts, sequential execution, advanced returns, multimodal returns, custom schemas, dynamic prepare functions, and custom args validators.
- V2 execution direction: model-requested tools execute in parallel by default. Per-tool `sequential=True` acts as a barrier in the tool-call batch; `end_strategy` decides what to do after final output appears, not whether tools are globally sequential. Source: [PR #5339](https://github.com/pydantic/pydantic-ai/pull/5339).
- Function-tool retries can suppress internally produced final results so retry semantics stay coherent when output and function calls are processed in the same batch. Source: [PR #5339](https://github.com/pydantic/pydantic-ai/pull/5339).

### Toolsets

- Toolsets are reusable collections of tools with lifecycle and instructions.
- Built-in composition/wrapper types: combined, filtered, prefixed, renamed, prepared, approval-required, deferred-loading, return-schema inclusion, metadata setting, external, wrapper.
- Toolsets are the main abstraction for MCP servers and third-party tool collections.

### Deferred Tools

- Deferred tool calls cover human approval and external execution.
- Inline resolution uses `HandleDeferredToolCalls`.
- Stop-the-world resolution returns `DeferredToolRequests`, then resumes with `DeferredToolResults` and message history.
- V2 execution direction: deferred calls remain batched with normal tool-call processing, including sequential barriers and parallel execution where allowed. Source: [PR #5339](https://github.com/pydantic/pydantic-ai/pull/5339).

### Output

- Output can be text, structured data, image, output function result, deferred requests, or optional `None`.
- Structured output modes:
  - `ToolOutput`: output represented as tool calls.
  - `NativeOutput`: provider-native JSON schema output.
  - `PromptedOutput`: schema instructions plus text parsing.
  - `TextOutput`: text passed through a processing function.
- Output validators and output functions can ask the model to retry via `ModelRetry`.
- Streaming output supports partial validation and partial output processing.

### Capabilities

- Capabilities bundle reusable behavior: instructions, model settings, toolsets, native tools, wrapper toolsets, and lifecycle hooks.
- Built-ins include Thinking, Hooks, Instrumentation, WebSearch, WebFetch, ImageGeneration, MCP, ToolSearch, PrepareTools, PrepareOutputTools, PrefixTools, NativeTool, Toolset, IncludeToolReturnSchemas, SetToolMetadata, HandleDeferredToolCalls, ProcessHistory, ProcessEventStream, ThreadExecutor.
- Provider-adaptive capabilities choose native provider tools when supported and local function-tool fallbacks otherwise.
- V2 direction: capabilities are the migration destination for cross-cutting agent knobs. `Instrumentation(...)`, `ProcessEventStream(...)`, and `PrepareTools(...)` replace constructor-level `instrument=`, `event_stream_handler=`, and `prepare_tools=` as the primary shape. Sources: [PR #5434](https://github.com/pydantic/pydantic-ai/pull/5434), [PR #5475](https://github.com/pydantic/pydantic-ai/pull/5475).
- Provider-adaptive defaults are intentionally capability-specific. `WebSearch` and `WebFetch` default to provider-native tools when possible and require explicit `local=...` for local fallback behavior; `MCP` defaults local-only because credentials and transport usually live client-side, with provider-native MCP as an explicit opt-in. Source: [PR #5333](https://github.com/pydantic/pydantic-ai/pull/5333).

### Models, Providers, Profiles

- Model classes translate normalized Pydantic AI messages/settings/tools into provider API calls and translate responses back.
- Providers own authentication, SDK clients, base URLs, and HTTP client lifecycle.
- Profiles encode model-family behavior: schema quirks, supported native tools, structured output defaults, thinking support, return-schema support, caching/instruction handling.
- Model ID strings use `provider:model`, with `infer_model` resolving model class, provider, and profile.
- Provider API compatibility is core product work, not a side integration. The library's role is to expose provider functionality through one ergonomic API while preserving provider-specific capability, metadata, and escape hatches where needed.
- Important provider functionality includes tool calling, structured output modes, native tools, thinking/reasoning, prompt caching, message compaction, multimodal input/output, streaming, usage, token counting, retries/errors, and provider-specific settings.
- V2 direction: `ModelProfile` is moving toward a `TypedDict(total=False)` shape merged with `merge_profile`, with effective profile state resolved from defaults, provider/model profile facts, and user overrides. Source: [PR #5481](https://github.com/pydantic/pydantic-ai/pull/5481).
- V2 provider routing removes several deprecated aliases and fallbacks: `openai:` defaults to the Responses API, bare model-name fallback is dropped, old Google/Grok/OpenAI aliases are removed, and `vendor_*` terminology gives way to `provider_*`. Sources: [PR #5469](https://github.com/pydantic/pydantic-ai/pull/5469), [PR #5464](https://github.com/pydantic/pydantic-ai/pull/5464), [PR #5479](https://github.com/pydantic/pydantic-ai/pull/5479), [PR #5460](https://github.com/pydantic/pydantic-ai/pull/5460), [PR #5468](https://github.com/pydantic/pydantic-ai/pull/5468), [PR #5476](https://github.com/pydantic/pydantic-ai/pull/5476).

### Native Tools

- Provider-executed tools include WebSearch, XSearch, CodeExecution, ImageGeneration, WebFetch, Memory, MCPServer, FileSearch.
- Native tool support varies by provider and model profile.
- Native tool parts are preserved in normalized message history where provider adapters expose them.
- V2 naming direction: `native_*` is the canonical vocabulary; older `builtin_*` shims are removed. Output-tool events are also split from function-tool events rather than being treated as one tool-event family. Sources: [PR #5396](https://github.com/pydantic/pydantic-ai/pull/5396), [PR #5332](https://github.com/pydantic/pydantic-ai/pull/5332).

### Common Tools

- Local function-tool implementations include DuckDuckGo search, web fetch, Tavily, Exa, image generation.
- These are either directly usable or local fallbacks for provider-adaptive capabilities.

### MCP

- Agent-side MCP client toolsets: standard MCP SDK clients over Streamable HTTP, SSE, stdio.
- FastMCP client toolset: higher-level connection/config/OAuth/tool-transformation path.
- Provider-native remote MCP: model provider connects to the MCP server.
- Pydantic AI agents can also run inside MCP servers and can support MCP sampling.
- V2 direction: `MCPToolset` is the canonical public client toolset. Deprecated `MCPServer*`, `FastMCPToolset`, and `load_mcp_servers` surfaces are removed, and durable wrappers target `MCPToolset` directly. Source: [PR #5337](https://github.com/pydantic/pydantic-ai/pull/5337).

### Streaming and Events

- Provider streams yield normalized part events.
- Agent streams layer output detection and validation on top.
- Event streams expose model response parts, tool call/result events, final-result events, and run result events.
- V2 direction: output-tool events are distinct from function-tool events, which matters for UI adapters, snapshots, and message-history replay. Source: [PR #5332](https://github.com/pydantic/pydantic-ai/pull/5332).

### Observability

- Logfire/OpenTelemetry integration is exposed through instrumentation settings/capability.
- Instrumentation covers agent runs, model requests, tool calls, output validation/processing, usage, and messages depending on config.

### Testing

- `TestModel` simulates model behavior for deterministic agent tests.
- `FunctionModel` lets tests inspect request messages/params and return custom model responses.
- Model request recording/cassettes cover provider integrations.

### Embeddings

- Embedding models/providers are separate from agent chat models but follow similar provider/wrapper/instrumentation patterns.
- Supported providers include OpenAI, Google, Cohere, VoyageAI, Bedrock, Sentence Transformers, and test/instrumented wrappers.

### UI, Direct, Durable, Specs

- UI adapters convert agent/event streams into AG-UI and Vercel AI shapes.
- Direct model requests expose lower-level one-shot model calls outside the agent loop.
- Durable execution wrappers adapt models/toolsets/agents for Temporal, DBOS, Prefect, and related durable orchestration engines.
- Durable execution engines are first-class compatibility targets. Changes to agent run semantics, tool execution, context propagation, model selection, retries, streaming, MCP/toolsets, or capability ordering should be checked against durable wrappers.
- There is an open direction to standardize durable object / durable orchestration integration via a capability. Even before that lands, durable-engine compatibility should stay visible in design decisions.
- Agent specs serialize/load agent configuration and spec-compatible capabilities.
