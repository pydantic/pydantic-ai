# Pydantic AI Slim Internals Model

Purpose: orient code reading and maintainer decisions around the implementation architecture of `pydantic-ai-slim`.

Primary source entry points:

- Public exports: `pydantic_ai_slim/pydantic_ai/__init__.py`
- Agent facade: `pydantic_ai_slim/pydantic_ai/agent/__init__.py`
- Agent abstract run methods: `pydantic_ai_slim/pydantic_ai/agent/abstract.py`
- Graph runtime: `pydantic_ai_slim/pydantic_ai/_agent_graph.py`
- Run/result wrappers: `pydantic_ai_slim/pydantic_ai/run.py`, `pydantic_ai_slim/pydantic_ai/result.py`
- Messages: `pydantic_ai_slim/pydantic_ai/messages.py`
- Models: `pydantic_ai_slim/pydantic_ai/models/__init__.py`, `pydantic_ai_slim/pydantic_ai/models/*.py`
- Providers: `pydantic_ai_slim/pydantic_ai/providers/*.py`
- Profiles: `pydantic_ai_slim/pydantic_ai/profiles/*.py`
- Tools: `pydantic_ai_slim/pydantic_ai/tools.py`
- Tool manager: `pydantic_ai_slim/pydantic_ai/tool_manager.py`
- Toolsets: `pydantic_ai_slim/pydantic_ai/toolsets/*.py`
- Output internals: `pydantic_ai_slim/pydantic_ai/_output.py`, `pydantic_ai_slim/pydantic_ai/output.py`
- Capabilities: `pydantic_ai_slim/pydantic_ai/capabilities/*.py`
- Durable execution: `pydantic_ai_slim/pydantic_ai/durable_exec/*`

## Core Shape

The runtime is a typed agent graph, not a monolithic loop.

1. `Agent` stores configuration and constructs run-time graph deps.
2. `AbstractAgent.run()` and friends drive `Agent.iter()`.
3. `Agent.iter()` builds a graph with `UserPromptNode`, `ModelRequestNode`, and `CallToolsNode`.
4. The graph mutates `GraphAgentState` and reads services/config from `GraphAgentDeps`.

## Graph Nodes

### `UserPromptNode`

Responsibilities:

- Clean and initialize message history.
- Add system prompt parts and user prompt parts.
- Reevaluate dynamic system prompts in prior messages.
- Resume unprocessed model tool calls.
- Apply provided `DeferredToolResults`.
- Decide whether to start with model request or tool-call processing.

### `ModelRequestNode`

Responsibilities:

- Append the request to history.
- Increment run step.
- Prepare the active `ToolManager` for the step.
- Resolve instructions from agent/capabilities/toolsets.
- Build `ModelRequestParameters`: function tools, output tools, native tools, output mode/object, instruction parts, text/image allowance.
- Resolve model settings and run capability `before_model_request`.
- Normalize/prepare message history for provider adapters.
- Enforce usage limits and optional token counting.
- Call `model.request()` or `model.request_stream()` through capability middleware.
- Append model response and transition to `CallToolsNode`.

### `CallToolsNode`

Responsibilities:

- Interpret a `ModelResponse`.
- Handle empty/thinking-only/content-filter/length cases.
- Prefer tool calls over text when present.
- Extract text, images, native tool events, compaction text, and function/output tool calls.
- Validate/process text/image output.
- Dispatch tool calls through `process_tool_calls`.
- End with `FinalResult` or create next `ModelRequestNode`.

## Tool Dispatch

`process_tool_calls` is the dispatch center.

Current v2 execution direction:

1. Classify every model tool call by tool kind: output, function, external, unapproved, unknown.
2. Validate arguments before execution where possible.
3. Segment the call list around `sequential=True` barriers.
4. Launch output, function, external, and unknown-call handling together inside each segment.
5. Emit progress events and append resulting message parts in the order they are emitted.
6. Apply `end_strategy` to decide what to do after final output exists:
   - `early`: do not wait for avoidable remaining work once final output exists.
   - `graceful`: let function-tool work complete while suppressing extra output tools.
   - `exhaustive`: process all relevant output and function calls, first valid output wins.
7. Collect tool return parts, retry prompt parts, user content parts, deferred calls, and metadata.
8. Resolve deferred calls inline through capability hooks when possible.
9. Either produce final `DeferredToolRequests` or return tool parts for the next model request.

Source: [PR #5339](https://github.com/pydantic/pydantic-ai/pull/5339).

Implication: `_call_tools` is becoming the deferred-handler-resolution path, not the conceptual main execution engine. The mental model should center on `process_tool_calls`, with sequential tools as barriers and output tools as peers in the same batch rather than a pre-pass before function tools.

`ToolManager` is per run step. It owns active tools, validation, execution, retry counters, capability tool hooks, deferred-call resolution, and parallel execution mode decisions.

## Toolsets

`AbstractToolset` is the collection/execution interface:

- `for_run`
- `for_run_step`
- `__aenter__` / `__aexit__`
- `get_instructions`
- `get_tools`
- `call_tool`

Wrapper toolsets modify listing, names, metadata, schemas, approval/defer behavior, or execution without changing leaf toolsets.

## Output Pipeline

Public markers in `output.py` are user API. `_output.py` builds internal schemas and processors.

Important internal concepts:

- `OutputSchema`: complete interpretation of accepted output.
- `BaseOutputProcessor`: validates/processes a concrete output path.
- `OutputToolset`: exposes structured output as output tools.
- Output hooks: capability validation/process wrappers plus agent output validators.

Output is semantically parallel to tools, but output hooks and retry budgets are separate from function-tool hooks and retries.

V2 event-stream direction: output-tool events are distinct from function-tool events, so tests/docs should snapshot output events as their own protocol surface rather than assuming every tool-looking event is a function-tool event. Source: [PR #5332](https://github.com/pydantic/pydantic-ai/pull/5332).

## Message Model

`messages.py` is the internal protocol.

Requests contain:

- system prompt parts
- user prompt parts
- tool return parts
- retry prompt parts
- instruction parts

Responses contain:

- text parts
- thinking parts
- compaction parts
- file parts
- function tool calls
- native tool calls/returns

Streaming uses deltas and part events to build the same final response shape. Provider adapters should translate to/from this protocol, not leak provider-specific structure into graph logic except through dedicated metadata fields.

V2 message-capture direction: captured messages can carry `state='complete' | 'interrupted'`. Streaming exceptions can still preserve a partial `ModelResponse`; interruption/cancellation during tool processing can preserve a partial `ModelRequest` with completed tool returns. Source: [PR #5364](https://github.com/pydantic/pydantic-ai/pull/5364).

## Model Layer

`Model` is the normalized interface:

- `request`
- `request_stream`
- `count_tokens`
- `compact_messages`
- `prepare_request`
- `prepare_messages`
- `customize_request_parameters`
- `profile`

`prepare_request` merges model settings, applies profile JSON-schema transformations, resolves thinking, deduplicates native tools, selects structured output mode, injects prompted-output instructions, checks capability support, and resolves native/local tool swaps.

V2 profile direction: `ModelProfile` is moving from dataclass-style attributes toward a sparse `TypedDict(total=False)` shape. Code that reads profile facts should use `.get(...)`-style defaults and `merge_profile(...)` rather than assuming a concrete class instance or complete attribute set. Effective profile behavior is resolved from default profile facts, provider/model profile inference, and user profile overrides. Source: [PR #5481](https://github.com/pydantic/pydantic-ai/pull/5481).

Provider adapters should focus on wire mapping:

- normalized messages to provider request shape
- normalized tools/output/native tools to provider-specific params
- provider responses/streams back to `ModelResponse` and stream events
- provider-specific settings classes and metadata

Provider compatibility is a central abstraction responsibility:

- Keep normalized core APIs ergonomic while allowing provider-specific capabilities to surface through typed settings, native tools, profiles, provider metadata, and model-specific classes.
- Preserve functionality across providers where possible; when impossible, expose capability facts and fail clearly rather than silently dropping behavior.
- Avoid scattering provider checks through graph/tool/output logic. Prefer profile flags, provider/model overrides, native tool support sets, and typed settings.

## Provider and Profile Layer

Provider classes own auth, clients, base URLs, and model profile inference.

Profiles are where model-family capability facts belong:

- structured output support/default mode
- tool support
- native tool support
- thinking support
- JSON schema transformer
- return schema support
- prompted-output template
- provider/model quirks that are intrinsic to the model family

Provider-specific API behavior belongs on provider/model classes, not in generic graph code.

V2 routing direction: deprecated provider/model aliases are intentionally being removed while canonical provider routing remains central. Important removals include bare model fallback, old Google provider/model prefixes, GrokProvider in favor of XaiProvider, old OpenAI aliases, and `vendor_*` naming. Sources: [PR #5464](https://github.com/pydantic/pydantic-ai/pull/5464), [PR #5479](https://github.com/pydantic/pydantic-ai/pull/5479), [PR #5460](https://github.com/pydantic/pydantic-ai/pull/5460), [PR #5468](https://github.com/pydantic/pydantic-ai/pull/5468), [PR #5476](https://github.com/pydantic/pydantic-ai/pull/5476).

## Durable Execution Layer

Durable execution integrations are first-class compatibility targets, not peripheral adapters.

Current code lives under `pydantic_ai_slim/pydantic_ai/durable_exec/` for Temporal, DBOS, and Prefect.

Their role:

- Adapt agent/model/toolset execution to durable orchestration constraints.
- Preserve run context and dependencies across durable boundaries.
- Make tool execution resumable/replay-safe where the engine requires it.
- Wrap models/toolsets/MCP servers/function toolsets without changing the user-facing agent concepts.
- Keep observability and retries coherent when work is split into durable activities/tasks.

Design implications:

- Changes to graph node semantics, tool call ordering, output finalization, deferred tools, streaming, model selection, `RunContext`, toolset lifecycle, and capability ordering can break durable integrations.
- Durable wrappers should benefit from generic extension points. Capability-based integration is the preferred direction for standardizing durable object / durable orchestration behavior.
- Compatibility checks should include whether a feature can be represented through serializable specs/config, stable toolset IDs, deterministic activity names, and replay-safe context handling.
- V2 MCP direction: durable MCP wrappers target `MCPToolset` directly after removal of separate MCP server and FastMCP wrapper families. Source: [PR #5337](https://github.com/pydantic/pydantic-ai/pull/5337).
- V2 typing direction: default type variables move from `None` to `object` for deps/state-style generics, which matters for durable wrappers and type-safe context propagation. Source: [PR #5307](https://github.com/pydantic/pydantic-ai/pull/5307).

## Capabilities

Capabilities are middleware plus configuration contributors.

They can provide:

- instructions
- model settings
- toolsets
- native tools
- wrapper toolsets
- prepare-tools hooks
- run/node/model/tool/output/event/deferred hooks

`CombinedCapability` composes them in middleware order. Forward hooks run outer-to-inner; after/wrap unwinding runs inner-to-outer where appropriate.

Use capabilities for cross-cutting behavior that should compose across models, tools, and runs.

V2 capability consolidation: `Instrumentation`, `ProcessEventStream`, and `PrepareTools` are the canonical home for behavior previously exposed as direct `Agent` constructor knobs. This keeps graph construction smaller and makes durable/spec integrations reason about one composable extension surface. Sources: [PR #5434](https://github.com/pydantic/pydantic-ai/pull/5434), [PR #5475](https://github.com/pydantic/pydantic-ai/pull/5475).

## Maintainer Layering Heuristics

- If it changes user-facing agent construction/run API: start in `agent/`.
- If it changes loop semantics: start in `_agent_graph.py`.
- If it changes provider wire shape: start in `models/{provider}.py`.
- If it changes model-family support facts: start in `profiles/{provider}.py`.
- If it changes cross-cutting behavior: prefer `capabilities/`.
- If it changes tool collection behavior: prefer `toolsets/` or `ToolManager`.
- If it changes normalized data shape: start in `messages.py` or `_output.py`.
- If it requires provider-specific data retention: use `provider_details`/metadata fields, not overloaded strings or args.
