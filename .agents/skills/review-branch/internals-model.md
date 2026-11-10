# Pydantic AI Slim Internals Model

Purpose: orient code reading and layering decisions in Pydantic AI Slim and its integrated Harness/CLAI 2 layers. Verified against `upstream/main` at `f6478ca7d6f65df4531ed892098d808af8431e48` (2026-11-10).

The repo's own CE outranks this file. `agent_docs/pydantic-ai-slim.md` owns the ownership map ("Ownership"), the contracts to check before editing ("Compatibility Checks"), the design rules, and the event-stream and cancellation internals. Each directory under `pydantic_ai_slim/pydantic_ai/` with an `AGENTS.md` owns its local rules. This file adds the runtime walkthrough those files do not give.

Primary source entry points:

- Public exports: `pydantic_ai_slim/pydantic_ai/__init__.py`
- Agent facade: `pydantic_ai_slim/pydantic_ai/agent/__init__.py`
- Agent abstract run methods: `pydantic_ai_slim/pydantic_ai/agent/abstract.py`
- Graph runtime: `pydantic_ai_slim/pydantic_ai/_agent_graph.py`
- Tool-call processing: `pydantic_ai_slim/pydantic_ai/_tool_execution.py`
- Run context: `pydantic_ai_slim/pydantic_ai/_run_context.py`
- Run/result wrappers: `pydantic_ai_slim/pydantic_ai/run.py`, `pydantic_ai_slim/pydantic_ai/result.py`
- Messages: `pydantic_ai_slim/pydantic_ai/messages.py`
- Models: `pydantic_ai_slim/pydantic_ai/models/__init__.py`, `pydantic_ai_slim/pydantic_ai/models/_abstract.py`, `pydantic_ai_slim/pydantic_ai/models/*.py`
- Providers: `pydantic_ai_slim/pydantic_ai/providers/*.py`
- Profiles: `pydantic_ai_slim/pydantic_ai/profiles/*.py`
- Tools: `pydantic_ai_slim/pydantic_ai/tools.py`
- Tool manager: `pydantic_ai_slim/pydantic_ai/tool_manager.py`
- Toolsets: `pydantic_ai_slim/pydantic_ai/toolsets/*.py`
- Output internals: `pydantic_ai_slim/pydantic_ai/_output.py`, `pydantic_ai_slim/pydantic_ai/output.py`
- Capabilities: `pydantic_ai_slim/pydantic_ai/capabilities/*.py`
- Durable execution: `pydantic_ai_slim/pydantic_ai/durable_exec/*`
- Workspaces: `pydantic_ai_slim/pydantic_ai/workspaces/*`
- Realtime: `pydantic_ai_slim/pydantic_ai/realtime/*`
- UI adapters: `pydantic_ai_slim/pydantic_ai/ui/*`
- Agent specs: `pydantic_ai_slim/pydantic_ai/agent/spec.py`; `docs/agent-spec.md`

## Core Shape

The runtime is a typed agent graph, not a monolithic loop.

1. `Agent` stores configuration. `Agent.iter()` is the graph-run facade: `_prepare_run()` resolves per-run inputs into `_PreparedAgentRun`, whose `open()` owns resource entry, capability lifecycle, recovery and cleanup. Details: `agent_docs/pydantic-ai-slim.md` → "Ownership".
2. `AbstractAgent.run()` and friends drive `Agent.iter()`.
3. The graph comes from `build_agent_graph()`, which calls an `lru_cache`d `_build_agent_graph(name)` built with `pydantic_graph.GraphBuilder`. It has four nodes: `UserPromptNode`, `ModelRequestNode`, `CallToolsNode`, `SetFinalResult`.
4. The graph mutates `GraphAgentState` (per-run mutable state) and reads services and config from `GraphAgentDeps` (per-run inputs: model, toolsets, capabilities, output schema, end strategy, model selector).

The graph object depends on the agent `name` alone and is shared by every run and every agent with that name. Keep it that way: agent config and run state go on `GraphAgentDeps` / `GraphAgentState`, never into graph construction or node classes.

## Agent Specs

Agent specs are a declarative construction path alongside direct `Agent(...)` construction. `AgentSpec` parses YAML/JSON configuration; `Agent.from_spec()` and `Agent.from_file()` build agents, and run/override APIs can apply a spec at runtime. Use `docs/agent-spec.md` for schema, loading, and precedence details.

## Graph Nodes

### `UserPromptNode`

Responsibilities:

- Clean and repair message history (dangling tool calls, orphaned tool results, consecutive requests).
- Add system prompt parts and user prompt parts.
- Reevaluate dynamic system prompts in prior messages.
- Resume unprocessed model tool calls.
- Resume a suspended response (Anthropic `pause_turn`, OpenAI background mode) when history ends in one.
- Apply provided `DeferredToolResults`.
- Decide whether to start with a model request or tool-call processing.

### `ModelRequestNode`

Responsibilities:

- Append the request to history.
- Increment run step.
- Select the step's model: `_select_model()` calls the model selector once per step. The selector is a callable a capability returns from `get_model` (`SelectModel` is the built-in). It receives a `ModelSelectionContext` whose `messages` end with the request being routed and whose `prompt` is the run prompt.
- Prepare the active `ToolManager` for the step.
- Resolve instructions from agent/capabilities/toolsets.
- Build `ModelRequestParameters`: function tools, output tools, native tools, output mode/object, instruction parts, revealed and deferred tool names, text/image allowance.
- Resolve model settings and run capability `before_model_request`.
- Normalize/prepare message history for provider adapters.
- Enforce usage limits and optional token counting.
- Call `model.request()` or `model.request_stream()` through capability middleware. A capability can raise `SkipModelRequest` to short-circuit.
- Continue suspended responses transparently, merging segments into one completed `ModelResponse`.
- Append model response and transition to `CallToolsNode`.

### `CallToolsNode`

Responsibilities:

- Interpret a `ModelResponse`.
- Handle empty/thinking-only/content-filter/length cases.
- Prefer tool calls over text when present.
- Extract text, images, native tool events, compaction text, and function/output tool calls.
- Validate/process text/image output.
- Dispatch tool calls through `process_tool_calls` (`_tool_execution.py`).
- End through `SetFinalResult` / `End` with a `FinalResult`, or create the next `ModelRequestNode`.

### `SetFinalResult`

Ends the graph immediately with a `FinalResult` that a streaming response already produced.

## Tool Dispatch

`process_tool_calls` in `_tool_execution.py` is the dispatch center. It picks a processor by `end_strategy` and runs it: `_EarlyProcessor`, `_GracefulProcessor` (default), `_ExhaustiveProcessor`, all subclasses of `_ToolCallProcessor`. `_ToolCallProcessor._call_tools` validates and executes one batch of function/unknown calls; it is not the conceptual center.

Semantics (from `process_tool_calls` and `_ToolCallProcessor`):

1. Classify every model tool call by kind: output, function, external, unapproved, unknown.
2. Segment the calls around `sequential=True` barriers. A barrier runs alone: calls emitted before it finish first, calls after it start once it finishes. `parallel_execution_mode('sequential')` makes every call a barrier.
3. Apply `end_strategy`:
   - `early`: output tools run sequentially and stop at the first success. Function tools run only if every output tool failed; otherwise they are stubbed as not executed.
   - `graceful` (default): calls run in emission order, function tools in parallel within a segment. Output tools run sequentially and stop at the first success; later output tools are skipped.
   - `exhaustive`: every call runs in parallel. The first valid output in emission order wins; the rest still execute.
4. Retry wins: under `graceful` and `exhaustive`, a function tool that asks for a retry (`ModelRetry` or failed argument validation, recorded as a `ToolReturnPart` with `outcome='retried'`) suppresses the final result. Output-tool and unknown-tool retries do not (`_ToolCallProcessor._is_retry_wins_trigger`; the `process_tool_calls` docstring still says "function/unknown"). The rule does not apply to a final result passed in by `run_stream`.
5. Collect tool return parts (retried calls included), user content parts, deferred calls, and metadata.
6. Collect unresolved `external` / `unapproved` calls and resolve them as one batch at the end of the step, inline through capability hooks when possible.
7. Either produce `DeferredToolRequests` or return tool parts for the next model request.

`process_tool_calls` in `_tool_execution.py` owns per-strategy scheduling, barriers, and tool-call retry/return handling. `CallToolsNode` in `_agent_graph.py` owns response-level precedence between text/image output and tool calls, including the `end_strategy='early'` case where plain function calls are co-emitted with a final output.

`ToolManager` is per run step (`for_run_step`). It owns active tools, validation, execution, retry counters, capability tool hooks, deferred-call resolution, and the parallel execution mode.

## Toolsets

`AbstractToolset` is the collection/execution interface:

- `for_run`
- `for_run_step`
- `__aenter__` / `__aexit__`
- `get_instructions`
- `get_tools`
- `call_tool`
- `get_tool_for_tool_def`: returns the tool for a definition this toolset already produced. Durable execution uses it to rebuild a tool inside a durable unit. The default lists the tools, which is always correct; override it only when the toolset can build the tool from the definition alone (`MCPToolset`, whose listing is a network round trip). `DynamicToolset` delegates it to the resolved toolset.

Wrapper toolsets modify listing, names, metadata, schemas, approval/defer behavior, or execution without changing leaf toolsets. Forwarding rules: `toolsets/AGENTS.md`.

A `DynamicToolset` built with `per_run_step=False` resolves once per run; under durable execution the run holds it like an `MCPToolset` session.

## Output Pipeline

Public markers in `output.py` are user API (`ToolOutput`, `NativeOutput`, `PromptedOutput`, `TextOutput`, `Choices`, `Choice`, `BoolCriteria`). `_output.py` builds internal schemas and processors.

Important internal concepts:

- `OutputSchema` and its subclasses (`AutoOutputSchema`, `TextOutputSchema`, `ImageOutputSchema`, `NativeOutputSchema`, `PromptedOutputSchema`, `ToolOutputSchema`): complete interpretation of accepted output.
- `BaseOutputProcessor` (`ObjectOutputProcessor`, `UnionOutputProcessor`, `TextOutputProcessor`, `TextFunctionOutputProcessor`): validates/processes a concrete output path.
- `OutputToolset`: exposes structured output as output tools.
- Output hooks: capability validation/process wrappers plus agent output validators.

Output is semantically parallel to tools, but output hooks and retry budgets are separate from function-tool hooks and retries.

Output-tool calls emit `OutputToolCallEvent` / `OutputToolResultEvent`, not the function-tool events. Snapshot them as their own protocol surface.

## Message Model

`messages.py` is the normalized protocol. `ModelMessagesTypeAdapter` (de)serializes histories.

Request parts (`ModelRequestPart`):

- `SystemPromptPart`, `UserPromptPart`
- `SpeechPart` (realtime user audio)
- `ToolReturnPart`: a tool-call retry is the call's `ToolReturnPart` with `outcome='retried'`
- `RetryFeedbackPart`: retry feedback that answers no tool call (output that failed validation, `ModelRetry` from an output validator or function, no usable output)
- `RetryPromptPart`: deprecated. A stored `'retry-prompt'` part loads as a `ToolReturnPart` or a `RetryFeedbackPart`.
- `ToolSearchReturnPart`, `LoadCapabilityReturnPart` (typed `ToolReturnPart` subclasses)
- `ToolAvailabilityDeltaPart`: records tools that became available mid-conversation without invalidating the prompt cache
- `CapabilityLoadDeltaPart`, `ModelSettingsDeltaPart`: record on-demand capabilities loaded without a `load_capability` call, and model-setting changes, from that point on
- `InstructionDeltaPart`, `InstructionBaselinePart`: record instruction changes and the instructions an `on_change='append'` window opened with
- `CompactionPart` (also a response part)

Instructions travel as `InstructionPart`s on `ModelRequestParameters` and as the rendered `ModelRequest.instructions` string. History records only their changes, through `InstructionDeltaPart` and `InstructionBaselinePart`.

Response parts (`ModelResponsePart`):

- `TextPart`, `ThinkingPart`, `CompactionPart`, `FilePart`, `SpeechPart`
- `ToolCallPart`, plus typed subclasses `ToolSearchCallPart` and `LoadCapabilityCallPart`
- `NativeToolCallPart` / `NativeToolReturnPart`, plus `NativeToolSearchCallPart` / `NativeToolSearchReturnPart`

Persisted tags for native parts stay `builtin-tool-call`, `builtin-tool-return`, `builtin-tool-search-call`, `builtin-tool-search-return`. Changing a tag breaks stored histories.

Message state:

- `ModelResponse.state` is `ModelResponseState`: `'complete'`, `'incomplete'` (still streaming), `'suspended'` (paused mid-turn, continued transparently), `'interrupted'` (stream cancelled or realtime barge-in).
- `ModelRequest.state` is `'complete'` or `'interrupted'`. Cancellation during tool processing can preserve a partial request with completed tool returns.
- `ModelResponse.workspace_ref` records the workspace environment. It is never sent to the model. `sanitize_messages` resets it by default (`strip_workspace_refs=True`), because a client that sets it could point a run at another environment.

`repair_messages()` is the public deterministic, idempotent provider-validity repair. The graph performs corresponding cleanup automatically; callers can use the helper before appending a prompt to history with unanswered tool calls. `sanitize_messages()` has a different boundary: use it for untrusted client-supplied history. By default it strips system prompts, unsafe file references/uploads, workspace refs, and unresolved tail tool calls. UI adapters apply this sanitization. Full contract: `docs/message-history.md` → "Making histories provider-valid".

Streaming uses deltas and part events to build the same final response shape. Provider adapters should translate to/from this protocol, not leak provider-specific structure into graph logic except through dedicated metadata fields.

The version policy lets minor releases add message parts, stream events, and optional fields. Consumers must code defensively against new parts.

## Model Layer

`AbstractModel` (`models/_abstract.py`) is the base shared by `Model` and `RealtimeModel`: `model_name`, `system`, `base_url`, `model_id`, `context_window`, `label`, async context management. `RunContext.model` is typed `AbstractModel`.

`Model` is the normalized request/response interface:

- `request`
- `request_stream`
- `count_tokens`
- `compact_messages`
- `cancel_suspended_response`, `continuation_delay` (suspended-response continuation)
- `prepare_request`
- `prepare_messages`
- `customize_request_parameters`
- `profile`, `context_window`, `tool_deferral_mode`, `tool_addition_mode`, `supported_native_tools`

`prepare_request` merges model settings, applies profile JSON-schema transformations, resolves thinking, deduplicates native tools, selects structured output mode, injects prompted-output instructions, checks capability support, and resolves native/local tool swaps. When forcing the output tool would stop the model thinking (`forced_tool_choice_disables_thinking`), the default structured output mode falls back to native output. Per-provider forced-tool-choice handling reads the profile through `models/_tool_choice.py`.

`ModelProfile` is a sparse `TypedDict(total=False)`. Read profile facts with `.get(...)` defaults and combine defaults, provider/model inference, and user overrides with `merge_profile(...)`. The model resolves tool visibility and discovery from `ModelRequestParameters`; `tool_visibility` distinguishes visible from deferred tools, deferred capability IDs track capability loading, and `ToolAvailabilityDeltaPart` records newly available names in history. Profile-native-tool support is intersected with the model class's implemented native tools before exposure.

Provider adapters should focus on wire mapping:

- normalized messages to provider request shape
- normalized tools/output/native tools to provider-specific params
- provider responses/streams back to `ModelResponse` and stream events
- provider-specific settings classes and metadata

Provider compatibility is a central abstraction responsibility:

- Keep normalized core APIs ergonomic while allowing provider-specific capabilities to surface through typed settings, native tools, profiles, provider metadata, and model-specific classes.
- Preserve functionality across providers where possible; when impossible, expose capability facts and fail clearly rather than silently dropping behavior.
- Avoid scattering provider checks through graph/tool/output logic. Prefer profile flags, provider/model overrides, native tool support sets, and typed settings.

Adapter rules (streaming/non-streaming parity, settings, error handling, cache-stable injections): `models/AGENTS.md`.

`DecisionModel` (`models/decision.py`) is a `Model` that maps a run onto typed questions. A backend subclasses it and implements `decide`; `TypeSafeModel` (`models/typesafe.py`) is the built-in backend.

## Provider and Profile Layer

Provider classes own auth, clients, base URLs, and model profile inference (`Provider.model_profile`).

Profiles are where model-family capability facts belong:

- ordinary tool, text output, image output, and audio input support
- inline system prompt support
- structured output support/default mode
- tool support and forced tool choice (`supports_forced_tool_choice`, `supports_forced_tool_choice_with_thinking`, `forced_tool_choice_disables_thinking`)
- native tool support
- thinking support and defaults (`supports_thinking`, `thinking_always_enabled`, `thinking_enabled_by_default`)
- JSON schema transformer
- return schema support
- prompted-output template
- context window
- tool deferral and addition modes
- provider/model quirks that are intrinsic to the model family

The effective native-tool set is the intersection of the profile declaration and the model adapter's supported set.

A generic `ModelProfile` key applies on every provider route that serves the model. A provider-prefixed key applies only where that provider's model class reads it. Prefer the generic key when the fact belongs to the model family.

Provider-specific API behavior belongs on provider/model classes, not in generic graph code. Rules: `providers/AGENTS.md`, `profiles/AGENTS.md`, and the "API Design" rules in `pydantic_ai_slim/pydantic_ai/AGENTS.md`.

## Durable Execution Layer

Durable execution integrations are first-class compatibility targets, not peripheral adapters. The rules, the backend-builder contract, the run-held toolset lifecycle, and workspace routing are in `pydantic_ai_slim/pydantic_ai/durable_exec/AGENTS.md`. Read it before touching durable code or anything it wraps. The `RunContext` fields that cross a Temporal activity boundary are listed in `capabilities/AGENTS.md`.

Layout:

- Shared (`durable_exec/`): `_base.py` (`BaseDurabilityCapability`), `_operation.py` (operation IDs), `_operation_backend.py` (`DurableOperationBackend`, `CallableOperationBackend`, `JournalCallableOperationBackend`, `RegisteredOperationBackend`), `_operation_names.py`, `_capability_operation.py`, `_run_context.py` (`DurableRunContext`), `_toolset.py`, `_runtime_toolsets.py`, `_workspace.py`, `_spec.py`, `_codec.py`.
- Every backend (`temporal/`, `dbos/`, `prefect/`) has `_agent.py` (deprecated wrapper agent), `_durability.py` (the capability), `_model.py`, `_mcp_toolset.py`, `_operation_backend.py`, `_operation_names.py`.
- Temporal and Prefect add `_function_toolset.py`, `_dynamic_toolset.py`, `_toolset.py`. Temporal adds `_run_context.py` (`TemporalRunContext`), `_event_stream.py`, `_workflow.py`. DBOS has no function-toolset wrapper.
- Temporal also has `_transports.py` and `_activity_execution.py`.
- Harness engines live under `src/pydantic_ai_harness/pydantic_ai_harness/`: `absurd/`, `aws_lambda/`, `azure_durable_task/`, `pgtask/`, `render/`. Each has `_capability.py` (the capability) and `_operation_backend.py`, built on the shared backend classes.

Integration shapes:

1. `*Durability` capabilities (`TemporalDurability`, `DBOSDurability`, `PrefectDurability`; harness `AbsurdDurability`, `AWSLambdaDurability`, `AzureDurableTaskDurability`, `PGTaskDurability`, `RenderWorkflows`) are the supported path. All subclass `BaseDurabilityCapability`. `for_agent` binds models/toolsets and registers activities/steps/tasks; ordering is `position='innermost'`.
2. `TemporalAgent` / `DBOSAgent` / `PrefectAgent` are `@deprecated` and scheduled for removal in v3. New features can be capability-only. Workspaces are the example: each wrapper attaches a `RejectWorkspaceInContainer` capability (`_reject_workspace_in_workflow`, or `_reject_workspace_in_flow` on Prefect) and refuses a workspace inside the durable container.

Design implications:

- Changes to graph node semantics, tool call ordering, output finalization, deferred tools, streaming (`CompletedStreamedResponse`, `state='suspended'`, `cancel_suspended_response`), model selection, `RunContext`, toolset lifecycle, and capability ordering can break durable integrations.
- Live `Model` instances do not serialize. Requests carry a `model_id` and rebuild the model through the durability capability's `models=` registry or the agent's `resolve_model_id` inside the durable unit.
- Stable unique toolset/capability `id`s are required for operation names and registries. Persisted operation names are compatibility data; `tests/durable_exec/test_durable_exec_compat.py` pins them.
- Capabilities that contribute toolsets or durable operations attach at construction time. Temporal raises on `agent.run(capabilities=[...])` inside a workflow; observe-only capabilities such as `Instrumentation` are exempt.
- Toolset lifetime follows the engine. Temporal opens and closes attached MCP sessions inside each tool-listing or call activity, and re-resolves `DynamicToolset` factories in activities even when `per_run_step=False`. DBOS and Prefect retain the MCP session and reuse `per_run_step=False` toolsets through the workflow or flow run. See the engine docs; do not infer one lifetime across engines.
- `@durable_operation` methods dispatch through the engine from `before_run` and from per-request hooks. The live-value hooks (`wrap_*`, `get_toolset`, `get_wrapper_toolset`) cannot be decorated.
- `RunContext.in_durable_context` tells a hook it runs in replayed workflow code, where it must stay deterministic.
- Footgun: `*Durability` does not make `agent.run` durable by itself. The user calls it inside their own workflow or flow; the wrappers wrapped the run automatically.
- Type variables for deps/state default to `object`, which matters for durable wrappers and type-safe context propagation.

## Capabilities

Capabilities are middleware plus configuration contributors.

They can provide:

- instructions (`get_instructions`) and a catalog description (`get_description`, for on-demand loading)
- model settings (`get_model_settings`)
- a model or per-step model selection (`get_model`), and model ID resolution (`resolve_model_id`)
- toolsets and wrapper toolsets (`get_toolset`, `get_wrapper_toolset`)
- native tools (`get_native_tools`)
- a workspace backend (`get_workspace`)
- durable operations (`@durable_operation` methods)
- event listeners (`@on_event`, `on_event`, filtered by `listens_to`)
- run/node/model/tool/output/event/deferred hooks (`before_*`, `after_*`, `wrap_*`)
- ordering constraints (`get_ordering` → `CapabilityOrdering`: `position` `'outermost'`/`'innermost'`, `wraps`, `wrapped_by`, `requires`)

`CombinedCapability` composes them in middleware order after `sort_capabilities` resolves ordering constraints. Forward hooks run outer-to-inner; after/wrap unwinding runs inner-to-outer where appropriate.

Use capabilities for cross-cutting behavior that should compose across models, tools, and runs. Capabilities bind through `for_agent()` and `for_run()`. Hook surfaces include `prepare_tools` / `prepare_output_tools`, error recovery, model/tool/output/run/event wrappers, and registered event listeners. Public functional callback adapters live in `capabilities/hooks.py`. `Agent.__init__` has no instrumentation, event-stream, or tool-preparation kwargs, but instrumentation is configurable through `agent.instrument` / `Agent.instrument_all()`, `event_stream_handler` is available per run, and composable implementations include `Instrumentation`, `ProcessEventStream`, `PrepareTools`, and `PrepareOutputTools`. Rules: `capabilities/AGENTS.md`.

`RunContext.emit()` appends registered `CustomEvent` and `CapabilityEvent` values to `GraphAgentState.event_stream_buffer`. Graph-node streams drain that buffer for consumers. See `agent_docs/pydantic-ai-slim.md` → "Run Event Stream Internals" before changing delivery or registration semantics.

## Workspaces

`workspaces/` defines `WorkspaceBackend` and its capability protocols (`SupportsCommands`, `SupportsFilesystem`, `SupportsRealpath`), the `Workspace` facade tools reach through `ctx.workspace`, `LocalWorkspaceBackend`, read-only, authority-capping and unavailable wrappers, and a conformance suite for backends. Workspace selection is resolved in `_prepare_run`: an explicit `workspace=` backend/ref/`'new'` or an override can direct selection; otherwise a recent `ModelResponse.workspace_ref` may be offered to a capability's `get_workspace`, with capability-provided workspaces as the fallback. `sanitize_messages()` strips stored refs by default. Durable routing lives in `durable_exec/_workspace.py`; see `durable_exec/AGENTS.md` → workspaces paragraph.

## Realtime

`realtime/` holds `RealtimeModel` (`AbstractModel`), codec/session machinery, and the provider modules (`openai`, `openai_live`, `google`, `xai`, `azure`, `elevenlabs`, `elevenlabs_agents`). `RealtimeSession` translates provider events to normalized messages and uses the shared `ToolManager` tool loop; it does not run graph nodes or `before_model_request`, and resolves deferred results inline. `Agent.realtime()` binds agent config to a session. Rules: `realtime/AGENTS.md`; architecture: `docs/api/realtime.md`.

## UI Adapters

`ui/` holds `UIAdapter` / `UIEventStream` (`_adapter.py`, `_event_stream.py`) and one package per protocol (`ag_ui/`, `vercel_ai/`). Adapters sanitize client-submitted history with `sanitize_messages` and translate normalized events to the protocol. Rules, including the AG-UI version-floor policy and when an adapter earns a property: `ui/AGENTS.md`.

## Maintainer Layering Heuristics

- If it changes user-facing agent construction/run API: start in `agent/`.
- If it changes loop semantics: start in `_agent_graph.py`.
- If it changes tool-call ordering, barriers, or `end_strategy`: start in `_tool_execution.py`.
- If it changes provider wire shape: start in `models/{provider}.py`.
- If it changes model-family support facts: start in `profiles/{provider}.py`, or a generic `ModelProfile` key when the fact applies on every provider route.
- If it changes cross-cutting behavior: prefer `capabilities/`.
- If it changes tool collection behavior: prefer `toolsets/` or `ToolManager`.
- If it changes normalized data shape: start in `messages.py` or `_output.py`.
- If it requires provider-specific data retention: use `provider_details`/metadata fields, not overloaded strings or args.
- If it needs environment access (commands, files): use a `Workspace` backend or `get_workspace`, not ad hoc filesystem or subprocess calls in tools.
- If it picks the model per step: use `SelectModel` / `get_model`, not graph changes.
- If it integrates a new durable engine: build on the public `durable_exec` backend surface (`durable_exec/AGENTS.md` → "Building a new durable execution engine").
- If it is a batteries-included composition of existing hooks, toolsets and messages: it belongs in the harness (`src/pydantic_ai_harness/`). Change core only for new semantics. Decision rule: `src/pydantic_ai_harness/agent_docs/core-boundary.md`.

## Integrated Harness and CLAI 2

Core owns agent-loop semantics, hooks, normalized messages and events, toolsets, workspace primitives, and durable-execution primitives. Harness packages reusable capabilities on those APIs. CLAI 2 owns the terminal prompt loop, plugin host, rendering, commands, and conversation/session orchestration. Start with the boundary guides before moving behavior between layers: `src/pydantic_ai_harness/agent_docs/core-boundary.md`, `src/pydantic_ai_harness/AGENTS.md`, and `src/pydantic_clai2/AGENTS.md`.

**Harness.** Check current public imports in `pydantic_ai_harness/__init__.py` and each capability package's `__init__.py`; the root resolves exports lazily. Implementations usually start in `<capability>/_capability.py`, with toolsets, events, and stores in neighboring modules when present. Capabilities subclass core's `AbstractCapability`; `Coder` is a `CombinedCapability` composition of workspace requirement, file and shell tools, repo context, subagents, compaction, output limits, and argument repair. File and command access goes through core's `ctx.workspace`. See `src/pydantic_ai_harness/pydantic_ai_harness/__init__.py`, `coder/_capability.py`, `filesystem/_capability.py`, and `shell/_capability.py`.

**State, events, and persistence.** Use `for_run` for capability state scoped to an agent run; application dependencies or shared stores own cross-run state. Core `CustomEvent` and `CapabilityEvent` use `RunContext.emit`; Harness defines typed capability events and consumers can use `@on_event`. CLAI's session/turn plugin events are a separate host API. For queue, store, persistence, and delegate-continuation details, see [coordination primitives](coordination-primitives.md). See `src/pydantic_ai_harness/agent_docs/capability-authoring.md`, `pydantic_ai_slim/pydantic_ai/messages.py`, `src/pydantic_ai_harness/pydantic_ai_harness/filesystem/_events.py`, and `src/pydantic_clai2/pydantic_clai2/plugins/__init__.py`.

**Skills and delegation.** Harness `Skills` reads workspace skill packages and exposes their instructions as deferred capabilities. `SubAgents` provides `delegate_task`; `DelegationTasks` is the opt-in owner for longer-lived child-task lifecycle. CLAI `runtime/tasks.py` presents those tasks in its UI; Harness owns execution and histories. See `skills/_capability.py`, `subagents/_capability.py`, `subagents/_tasks.py`, and `src/pydantic_clai2/pydantic_clai2/runtime/tasks.py`.

**CLAI 2.** `pydantic_clai2.plugins` is the plugin author API: `Plugin.get_capabilities()` returns core capabilities; `collect()` captures the plugin's `get_*` contributions in `LoadedPlugin`; `SessionPlugins` owns one session's loading and dispatch, and `ProcessPlugins` holds the plugins every live session of the process shares. CLAI host events and `Plugin.render()` are client hooks, not substitutes for core capability hooks or event listeners. `SessionRuntime.run_turn()` fires host turn events and sets the run's plugin snapshot; `Session.prompt()` calls `Agent.run()`. A CLAI-owned `StockAgent` is rebuilt when that snapshot changes; a supplied agent receives plugin capabilities per run. `Session._stream()` sends core agent-stream events to CLAI's rendering path. See `src/pydantic_clai2/pydantic_clai2/plugins/__init__.py`, `plugins/loader.py`, `plugins/process_plugins.py`, `app.py`, `runtime/session_runtime.py`, `runtime/session.py`, and `ui/rendering/rendering.py`.
