# Pydantic AI Slim Architecture

Use this guide for non-trivial changes to `pydantic-ai-slim`: public APIs, provider behavior, models/profiles, capabilities, toolsets, tools/output, message history, streaming, UI adapters, or durable execution.

## Ownership

- `Agent` owns user-facing construction and run APIs. Prefer not to add constructor kwargs for behavior that can be modeled as a capability, toolset, model setting, or profile fact.
- `_agent_graph.py` owns loop orchestration: prompt assembly, model requests, tool/output processing, retries, usage checks, and finalization.
- `tool_manager.py`, `tools.py`, and `toolsets/` own tool discovery, validation, execution, retries, approval/deferral, wrapper composition, and stable tool identity.
- `output.py` is the public output API; `_output.py` owns internal output schemas, processors, output tools, and output validation/processing.
- `messages.py` owns the normalized protocol. Provider adapters, UI adapters, durable wrappers, and persisted histories should round-trip through this shape instead of encoding provider facts in strings or ad hoc fields.
- `models/` maps normalized requests/responses to provider wire formats. Put provider-specific request/response translation here rather than in graph/tool/output code.
- `providers/` owns authentication, clients, base URLs, HTTP lifecycle, and provider-level model/profile inference.
- `profiles/` owns model-family facts: structured output defaults, schema quirks, native tool support, thinking support, return-schema support, prompted-output templates, and intrinsic model-family behavior.
- `capabilities/` owns composable cross-cutting behavior: instructions, settings, toolsets, native tools, wrapper toolsets, and run/model/tool/output/event/history hooks.
- `durable_exec/` adapts agents, models, and toolsets to durable runtimes. Treat these integrations as compatibility tests for core semantics, not peripheral adapters.
- `ui/` adapters translate normalized messages/events for frontend protocols. Preserve message history and event semantics across round-trips.

## Compatibility Checks

Before editing, identify which contracts can change:

- Public API: constructor kwargs, decorators, settings, output types, tool/toolset APIs, imports, and documented names.
- Provider API compatibility: request parameters, response parts, streaming chunks, native tools, structured output, thinking/reasoning, prompt caching, token counting, usage, and provider metadata.
- Message/event protocol: persisted message history, partial responses, tool-call parts, output events, native tool parts, retry prompts, and UI event streams.
- Durable execution: context propagation, dependency serialization, tool ordering, retries, message replay, toolset lifecycle, activity/task boundaries, and deterministic behavior.
- Agent specs/config: whether new state is serializable, safe to load, and representable without runtime-only closures.

## Design Rules

- Keep provider facts in providers/profiles/model adapters. Do not scatter provider-name checks through graph, tool, output, or UI code.
- Preserve provider-specific data in structured metadata/provider-details fields. Do not overload normalized IDs, text content, or tool args.
- Prefer general primitives over one-off flags: capabilities for cross-cutting behavior, wrapper toolsets for tool collection behavior, profiles for capability facts, and typed settings for provider API knobs.
- Keep local tools and provider-native tools conceptually separate. If a feature can use either, make the fallback/selection behavior explicit and test the message history it produces.
- When changing tool/output execution, check ordering, retry semantics, deferred calls, output finalization, streaming events, and durable wrappers together.
- When removing deprecated APIs, distinguish public surface cleanup from persisted-data compatibility. Old constructors/imports may be removed in a major version; old serialized histories may still need to deserialize.

## Test Shape

- Use public agent/model/toolset behavior for most tests. Prefer snapshots for message history, event streams, provider request payloads, and protocol shapes.
- Use provider cassettes or real integration tests when behavior depends on provider APIs or SDK behavior.
- Use workflow-level tests when durable runtime behavior depends on serialization, replay, or activity/task boundaries.
- Update docs and examples where users discover the feature, not only docstrings.
