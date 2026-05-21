# Upgrade Guide

In September 2025, Pydantic AI reached V1, which means we're committed to API stability: we will not introduce changes that break your code until V2. For more information, review our [Version Policy](version-policy.md).

## Breaking Changes

Here's a filtered list of the breaking changes for each version to help you upgrade Pydantic AI.

### v2.0.0b1 (2026-05-20)

<!-- TODO(v2-launch): at stable V2.0 release, add a `v2.0.0 (<date>)` heading for any changes since the last beta, resolve the #5339 NOT-YET-MERGED markers below, and drop "beta" framing from the intro. -->

The first V2 beta. V2 removes functionality deprecated during V1 and makes behavior changes that V1's stability guarantee didn't allow. Many of these changes share a throughline: replacing building blocks designed before [capabilities](capabilities.md) with patterns native to how capabilities and [hooks](hooks.md) work, so it's easier to assemble the powerful agents of 2026. See the [Version Policy](version-policy.md#v2-beta) for the recommended upgrade path; in short: upgrade to the latest V1, resolve every deprecation warning, then upgrade to V2.

The breaking changes below are split into two groups:

- [**Changes not covered by deprecation warnings**](#changes-not-covered-by-deprecation-warnings) — removals and behavior changes that couldn't be announced via a V1 deprecation warning. Review these even if you're already on the latest V1 with no warnings.
- [**Changes covered by deprecation warnings**](#changes-covered-by-deprecation-warnings) — if you upgraded to the latest V1 and resolved every deprecation warning, you've already made these. They're listed with full before → after for reference.

Message history serialized with V1 (via [`ModelMessagesTypeAdapter`][pydantic_ai.messages.ModelMessagesTypeAdapter]) continues to deserialize in V2.

#### Changes not covered by deprecation warnings

These removals and behavior changes could not be announced via a V1 deprecation warning, so review them even if you've resolved every deprecation warning on the latest V1.

**Code changes:**

- Generic type parameter defaults changed from `None` to `object`: an un-parameterized `Agent(...)` now infers `Agent[object, str]` instead of `Agent[None, str]`, and the `pydantic_graph` `StateT`/`RunEndT`/`DepsT` defaults changed to match. Update explicit `Agent[None, ...]`, `RunContext[None]`, and `Tool[None]` annotations that don't actually require `None` dependencies to use `object`. This is a type-checking-only change; runtime behavior is unchanged. See [#5307](https://github.com/pydantic/pydantic-ai/pull/5307).
- The `pydantic_graph.persistence` package and the `pydantic_graph.mermaid` module are removed, with no V2 equivalent for graph state persistence or standalone Mermaid generation (render diagrams with `Graph.render()`). The move of the [`GraphBuilder`][pydantic_graph.GraphBuilder] API out of `pydantic_graph.beta` to the top-level `pydantic_graph` *was* deprecation-announced; see [below](#changes-covered-by-deprecation-warnings). See [#5470](https://github.com/pydantic/pydantic-ai/pull/5470).
- [`ModelProfile`][pydantic_ai.profiles.ModelProfile] and its subclasses are now `TypedDict`s instead of dataclasses. Passing `profile=OpenAIModelProfile(field=value)` into a model still works unchanged; the migration only matters if you read or mutate profile fields, or call `.update()`/`.from_profile()`. See [`ModelProfile` is now a `TypedDict`](#modelprofile-is-now-a-typeddict) below. ([#5481](https://github.com/pydantic/pydantic-ai/pull/5481))

**Default behavior changes** — same API, different runtime behavior (roughly ordered by how many users they affect):

- A bare `uv add pydantic-ai` / `pip install pydantic-ai` now installs a slimmer set of extras (frontier providers plus minimal integrations); providers like `bedrock`, `groq`, and `mistral` are no longer included by default, so you'll need to add the extras you use. See [Slimmer default extras](#slimmer-default-pydantic-ai-extras) below. ([#5467](https://github.com/pydantic/pydantic-ai/pull/5467))
- The default `end_strategy` changed from `'early'` to `'graceful'`: when a model calls function tools in the same response as a successful output tool, those function tools now run (and their side effects happen) instead of being skipped, and tool calls run in the order the model emitted them. See [Parallel tool-call execution order](#parallel-tool-call-execution-runs-in-emission-order) below. ([#5339](https://github.com/pydantic/pydantic-ai/pull/5339)) **⚠️ NOT YET MERGED — PENDING [#5339]. UPDATE OR REMOVE THIS ENTRY (AND THE SUBSECTION BELOW) IF IT DOESN'T LAND IN THE BETA.**
- The default instrumentation format is now version 5, and agent run spans report token usage under `gen_ai.aggregated_usage.*`. See [Instrumentation defaults](#instrumentation-defaults-to-version-5-with-aggregated-usage-attributes) below. ([#5523](https://github.com/pydantic/pydantic-ai/pull/5523))
- [`capture_run_messages()`][pydantic_ai.capture_run_messages] now also captures the partial `ModelRequest`/`ModelResponse` from an interrupted run, marked with `state='interrupted'` (a new `ModelRequest.state` field is added). Code that asserts on exact captured-message counts on error paths may need updating. See [#5364](https://github.com/pydantic/pydantic-ai/pull/5364).
- Output tool calls and returns now emit dedicated `OutputToolCallEvent`/`OutputToolResultEvent` instead of `FunctionToolCallEvent`/`FunctionToolResultEvent`. Separately, native tool calls and returns no longer emit dedicated events at all — the `BuiltinToolCallEvent`/`BuiltinToolResultEvent` classes are removed and they surface only via the standard `PartStartEvent`/`PartDeltaEvent`. See [#5332](https://github.com/pydantic/pydantic-ai/pull/5332) and [#5476](https://github.com/pydantic/pydantic-ai/pull/5476).

##### [`ModelProfile`][pydantic_ai.profiles.ModelProfile] is now a `TypedDict`

See the [Model Profile guide](models/openai.md#model-profile) for an overview of what a model profile is and how to configure one.

[`ModelProfile`][pydantic_ai.profiles.ModelProfile] and all its subclasses ([`OpenAIModelProfile`][pydantic_ai.profiles.openai.OpenAIModelProfile], [`AnthropicModelProfile`][pydantic_ai.profiles.anthropic.AnthropicModelProfile], [`GoogleModelProfile`][pydantic_ai.profiles.google.GoogleModelProfile], `BedrockModelProfile`, etc.) are now `TypedDict(total=False)` instead of `@dataclass`. This unifies the mental model with [`ModelSettings`][pydantic_ai.settings.ModelSettings] (also a `TypedDict`) and enables direct dict-spread for cross-class merging.

`ModelProfile.update()` and `ModelProfile.from_profile()` are removed; use the module-level [`merge_profile`][pydantic_ai.profiles.merge_profile] (later argument wins per key).

Migration recipes:

| v1 (dataclass) | v2 (TypedDict) |
|---|---|
| `OpenAIModelProfile(field=value)` | Same syntax; returns a partial `dict` instead of a fully-defaulted instance. |
| `profile.field` (attribute read) | `profile.get('field', <default>)` — non-trivial defaults are exported from [`pydantic_ai.profiles`][pydantic_ai.profiles] (e.g. [`DEFAULT_THINKING_TAGS`][pydantic_ai.profiles.DEFAULT_THINKING_TAGS], [`DEFAULT_PROMPTED_OUTPUT_TEMPLATE`][pydantic_ai.profiles.DEFAULT_PROMPTED_OUTPUT_TEMPLATE]); the fully-merged base is [`DEFAULT_PROFILE`][pydantic_ai.profiles.DEFAULT_PROFILE]. |
| `profile.field = value` (attribute write) | `profile['field'] = value` |
| `dataclasses.replace(profile, field=value)` | `{**profile, 'field': value}` or `merge_profile(profile, ModelProfile(field=value))` |
| `profile.update(other)` | `merge_profile(profile, other)` |
| `OpenAIModelProfile.from_profile(p)` | Just `p` — no upcasting needed |
| `Model(name, profile=full_profile)` (full replace) | Now merges on top of the provider's default profile — usually what you want. For a hard replace use `Model(name, profile=lambda _default: full_profile)`. |
| `Model(name, profile=fn)` where `fn: Callable[[str], ModelProfile \| None]` | Removed — the user-passed callable is now `Callable[[ModelProfile], ModelProfile]`, receiving the resolved default and returning the final profile. The `(model_name: str) -> ModelProfile \| None` shape is still accepted internally by `Provider.model_profile`. |
| `isinstance(profile, OpenAIModelProfile)` | Not supported by `TypedDict` at runtime — raises `TypeError`. Use `isinstance(profile, dict)` or check key presence (`'openai_chat_supports_web_search' in profile`). Pyright still narrows correctly via the TypedDict subclass annotation. |

`Model.profile` is now the single source of truth for the **resolved** profile. It is composed by [`merge_profile`][pydantic_ai.profiles.merge_profile] in this order (later wins):

1. [`DEFAULT_PROFILE`][pydantic_ai.profiles.DEFAULT_PROFILE] — base defaults for every documented key.
2. `Provider.model_profile(model_name)` — provider/model-specific resolution.
3. The user's `profile=` argument — either a partial dict (merged on top) or a `Callable[[ModelProfile], ModelProfile]` (full control: receives the resolved default, returns the final profile).

##### Resolved profiles now carry cross-class fields

In v1, `ModelProfile.update()` silently filtered out fields not declared on the target class. In v2, dict-spread preserves every key.

This means e.g. a Bedrock-hosted Anthropic model's resolved profile now carries the upstream `anthropic_*` fields alongside the `bedrock_*` fields, where v1 dropped them. No in-tree model class reads cross-class fields, so behavior is unchanged in the standard providers; but custom model classes that do `profile.get('anthropic_supports_adaptive_thinking', False)` on a non-Anthropic route will now see the value the upstream Anthropic profile set, where v1 always returned the default.

See the [Model Profile guide](models/openai.md#model-profile) for how to configure a profile, and [PR #5481](https://github.com/pydantic/pydantic-ai/pull/5481) for the full `ModelProfile` redesign.

##### Parallel tool-call execution runs in emission order

> **⚠️ NOT YET MERGED — this subsection documents [#5339](https://github.com/pydantic/pydantic-ai/pull/5339), which is still open. Update or remove it if the PR doesn't land in the beta.**

The default [`end_strategy`][pydantic_ai.agent.EndStrategy] changed from `'early'` to `'graceful'`. This only affects responses where a model calls function tools in the *same* response as an [output tool](output.md#tool-output) (the call that ends the run). When that output tool **succeeds**, the function tools requested alongside it now **run** by default instead of being skipped, so their side effects happen and their results reach the model if the run continues; and a function tool's [`ModelRetry`][pydantic_ai.exceptions.ModelRetry] now suppresses the output result so the model can correct itself on the next round. The case where *every* output tool fails is unchanged: function tools run and the run continues either way. Most agents don't need any change. If you relied on the run ending the instant an output tool succeeds — skipping any function tools requested in the same response — set `end_strategy='early'` explicitly.

The [`sequential=True`](tools-advanced.md#parallel-tool-calls-concurrency) flag on a tool is now a per-tool **barrier** rather than a batch-wide serial switch: a sequential tool runs alone, but other tools in the same response still run in parallel around it. To run *all* of a run's tools serially, wrap the run in [`agent.parallel_tool_call_execution_mode('sequential')`][pydantic_ai.agent.AbstractAgent.parallel_tool_call_execution_mode] or set `parallel_tool_calls=False` on the [model settings][pydantic_ai.settings.ModelSettings].

See [Parallel Output Tool Calls](output.md#parallel-output-tool-calls) for the full behavior of all three strategies, and [#5339](https://github.com/pydantic/pydantic-ai/pull/5339).

##### Slimmer default `pydantic-ai` extras

A bare `uv add pydantic-ai` / `pip install pydantic-ai` now installs `pydantic-ai-slim[openai,anthropic,google,cli,mcp,evals,web,retries,logfire]` — frontier providers plus minimal integrations. Providers and integrations that were previously bundled are no longer installed by default; add the ones you use explicitly, e.g. `uv add 'pydantic-ai[bedrock,groq]'`: `bedrock`, `groq`, `mistral`, `cohere`, `xai`, `huggingface`, `temporal`, `ag-ui`, `ui`, and `spec`. See the [installation guide](install.md) for the full list of extras.

Some `pydantic-ai-slim` extras were also removed outright (not just dropped from the default bundle): the `outlines-*` extras (the Outlines integration is removed), `vertexai` (Vertex AI is now served by the `google` extra), `fastmcp` (the FastMCP back-compat shim is removed), and `a2a` (A2A now lives in the upstream `fasta2a` package). See [#5467](https://github.com/pydantic/pydantic-ai/pull/5467).

##### Instrumentation defaults to version 5 with aggregated usage attributes

The default [instrumentation format](logfire.md#configuring-data-format) is now version 5 (versions 2–4 still work but emit a deprecation warning; version 1 and its `event_mode=`/`logger_provider=` arguments are removed). In version 5, deferred tool calls (`CallDeferred`/`ApprovalRequired`) are no longer recorded as span errors.

Separately, [`InstrumentationSettings`][pydantic_ai.models.instrumented.InstrumentationSettings]'s [`use_aggregated_usage_attribute_names`](logfire.md#aggregated-usage-attribute-names) now defaults to `True`: agent run spans report token usage under `gen_ai.aggregated_usage.*` while model request spans keep `gen_ai.usage.*`, which avoids double-counting in backends that sum parent and child usage. Dashboards and alerts that read token usage from run spans must be updated, or set `use_aggregated_usage_attribute_names=False` to keep the V1 attribute names.

See [#5523](https://github.com/pydantic/pydantic-ai/pull/5523).

#### Changes covered by deprecation warnings

These changes were announced in the latest V1 releases via deprecation warnings that name the replacement API. If you upgraded to the latest V1 and resolved every warning, you've already made them; they're listed here with full before → after for reference.

**Behavior changes that flip silently if the V1 deprecation warning was not addressed** — even though these were announced, an unaddressed warning means the behavior changes without raising an error, so confirm you've handled them:

- The bare `openai:` model prefix now uses the OpenAI Responses API ([`OpenAIResponsesModel`][pydantic_ai.models.openai.OpenAIResponsesModel]) instead of the Chat Completions API ([`OpenAIChatModel`][pydantic_ai.models.openai.OpenAIChatModel]). Use `openai-chat:` to keep Chat Completions, or `openai-responses:` to opt into the new default explicitly. Announced via [#5334](https://github.com/pydantic/pydantic-ai/pull/5334); flipped in [#5469](https://github.com/pydantic/pydantic-ai/pull/5469).
- Provider-adaptive `WebSearch` and `WebFetch` capabilities are now native-only and raise on models that don't support them, and `MCP(url=...)` runs the server locally by default. Restore the V1 fallbacks with `WebSearch(local='duckduckgo')`, `WebFetch(local=True)`, and `MCP(url=..., native=True)`. Announced via [#5331](https://github.com/pydantic/pydantic-ai/pull/5331); changed in [#5333](https://github.com/pydantic/pydantic-ai/pull/5333).

**API removals and renames:**

- `pydantic_ai.providers.grok.GrokProvider` and `pydantic_ai.providers.grok.GrokModelName` are removed; use `pydantic_ai.providers.xai.XaiProvider` with `pydantic_ai.models.xai.XaiModel` (and `pydantic_ai.models.xai.XaiModelName`). The `grok:` model prefix is removed; use `xai:`. See [#5460](https://github.com/pydantic/pydantic-ai/pull/5460).
- `GoogleGLAProvider`, `GoogleVertexProvider`, and `GeminiModel` (the whole `pydantic_ai.models.gemini` module) are removed; use `pydantic_ai.providers.google.GoogleProvider` (Gemini API) or `pydantic_ai.providers.google_cloud.GoogleCloudProvider` (Vertex) with `pydantic_ai.models.google.GoogleModel`. Provider prefixes: `google-gla:`→`google:`, `google-vertex:`→`google-cloud:`, `vertexai:`→`google-cloud:`, and `gateway/gemini:`/`gateway/google-vertex:`→`gateway/google-cloud:`. `GoogleProvider(vertexai=, location=, project=, credentials=)` → `GoogleCloudProvider(...)`. `GoogleModelSettings` keys `google_vertex_service_tier`/`google_service_tier` → `google_cloud_service_tier`. Announced via [#5336](https://github.com/pydantic/pydantic-ai/pull/5336) and [#5543](https://github.com/pydantic/pydantic-ai/pull/5543); removed in [#5479](https://github.com/pydantic/pydantic-ai/pull/5479).
- `OpenAIModel` → `OpenAIChatModel`, `OpenAIModelSettings` → `OpenAIChatModelSettings`; the `OpenAIChatModel(system_prompt_role=...)` kwarg → `OpenAIModelProfile(openai_system_prompt_role=...)`; `OpenAICompaction(instructions=...)` removed; `OpenAIModelProfile.openai_supports_sampling_settings` → `openai_unsupported_model_settings`. See [#5468](https://github.com/pydantic/pydantic-ai/pull/5468).
- Built-in tools are renamed to "native" tools: `pydantic_ai.builtin_tools` → `pydantic_ai.native_tools`; `BuiltinToolCallPart`/`BuiltinToolReturnPart`/`AgentBuiltinTool` → `NativeToolCallPart`/`NativeToolReturnPart`/`AgentNativeTool`; `Agent(builtin_tools=[...])` → `capabilities=[NativeTool(...)]`; `builtin=` → `native=`; `OpenAIModelProfile.openai_builtin_tools` → `openai_native_tools`. The serialized `part_kind` wire values are unchanged, so message history still deserializes. Announced via [#5338](https://github.com/pydantic/pydantic-ai/pull/5338); removed in [#5396](https://github.com/pydantic/pydantic-ai/pull/5396).
- MCP: `MCPServerStdio`/`MCPServerSSE`/`MCPServerStreamableHTTP`/`MCPServerHTTP`, `FastMCPToolset`, `load_mcp_servers`, `Agent.run_mcp_servers()`, and `Agent.set_mcp_sampling_model()` are removed; use `pydantic_ai.mcp.MCPToolset`, `pydantic_ai.mcp.load_mcp_toolsets`, `async with agent:`, and `MCPToolset(sampling_model=...)`. Note that the new `MCPToolset` defaults differ (e.g. `max_retries`, `read_timeout`, `init_timeout`, `elicitation_handler`). Announced via [#5325](https://github.com/pydantic/pydantic-ai/pull/5325); removed in [#5337](https://github.com/pydantic/pydantic-ai/pull/5337).
- `Agent(instrument=...)`, `Agent.from_spec(instrument=...)`, `Agent.from_file(instrument=...)`, and `AgentSpec.instrument` are removed; use `capabilities=[Instrumentation(...)]`. (The `Agent.instrument` property, `Agent.instrument_all()`, and `InstrumentedModel` are unchanged.) See [#5434](https://github.com/pydantic/pydantic-ai/pull/5434).
- `Agent(event_stream_handler=...)` → `capabilities=[ProcessEventStream(...)]`; `Agent(prepare_tools=...)` → `capabilities=[PrepareTools(...)]`. The `event_stream_handler=` argument on `run()`/`run_sync()`/`run_stream()`/`iter()` is unchanged. Announced via [#5335](https://github.com/pydantic/pydantic-ai/pull/5335); removed in [#5475](https://github.com/pydantic/pydantic-ai/pull/5475).
- `Agent(history_processors=...)` → `capabilities=[ProcessHistory(...)]`. See [#5425](https://github.com/pydantic/pydantic-ai/pull/5425).
- `Agent(mcp_servers=[...])` → `Agent(toolsets=[...])`; `Agent.sequential_tool_calls()` → `agent.parallel_tool_call_execution_mode('sequential')`. See [#5466](https://github.com/pydantic/pydantic-ai/pull/5466).
- `Agent.to_a2a()` and the bundled `fasta2a` integration (and the `[a2a]` extra) are removed; install `fasta2a[pydantic-ai]>=0.6.1` and use `from fasta2a.pydantic_ai import agent_to_a2a`. Announced via [#5426](https://github.com/pydantic/pydantic-ai/pull/5426); removed in [#5502](https://github.com/pydantic/pydantic-ai/pull/5502).
- `Agent.to_ag_ui()`, `AGUIApp`, and the `pydantic_ai.ag_ui` shim are removed; use `pydantic_ai.ui.ag_ui.AGUIAdapter`. `pydantic_ai.models.cached_async_http_client` is removed; use `pydantic_ai.models.create_async_http_client()` or your own `httpx.AsyncClient`. Announced via [#5345](https://github.com/pydantic/pydantic-ai/pull/5345); removed in [#5464](https://github.com/pydantic/pydantic-ai/pull/5464).
- `pydantic_ai.ext.aci` (`tool_from_aci`, `ACIToolset`) is removed with no upstream replacement; wrap ACI tools with `Tool.from_schema`. Announced via [#5510](https://github.com/pydantic/pydantic-ai/pull/5510); removed in [#5467](https://github.com/pydantic/pydantic-ai/pull/5467).
- `pydantic_ai.output.DeferredToolCalls` → `DeferredToolRequests`; `pydantic_ai.toolsets.external.DeferredToolset` → `ExternalToolset`. See [#5459](https://github.com/pydantic/pydantic-ai/pull/5459).
- `FunctionToolset.tool()` now raises if the decorated callable's first parameter is not a `RunContext`; use `FunctionToolset.tool_plain()` for context-free tools. See [#5462](https://github.com/pydantic/pydantic-ai/pull/5462).
- Usage/token renames: `request_tokens` → `input_tokens`, `response_tokens` → `output_tokens`, `Usage` → `RunUsage`, `UsageLimits(request_tokens_limit=)` → `input_tokens_limit=`, `UsageLimits(response_tokens_limit=)` → `output_tokens_limit=`. Response field renames: `ModelResponse.vendor_details` → `provider_details`, `vendor_id`/`provider_request_id` → `provider_response_id`. Removed event-class shims `BuiltinToolCallEvent`/`BuiltinToolResultEvent`, and `FunctionToolCallEvent.call_id` → `.tool_call_id`. Message history serialized with the old field names still deserializes via retained validation aliases. See [#5476](https://github.com/pydantic/pydantic-ai/pull/5476).
- Output tool calls now emit dedicated `OutputToolCallEvent`/`OutputToolResultEvent` rather than `FunctionToolCallEvent`/`FunctionToolResultEvent`; `FunctionToolResultEvent(result=...)`/`.result` → `(part=...)`/`.part`. See [#5332](https://github.com/pydantic/pydantic-ai/pull/5332).
- `StreamedRunResult.stream` → `stream_output`, `StreamedRunResult.stream_structured` → `stream_response`, `StreamedRunResult.validate_structured_output` → `validate_response_output`; the plural `stream_responses()` → singular `stream_response()` (which yields a bare `ModelResponse`; read the old `is_last` flag as `response.state != 'incomplete'`). Announced via [#5296](https://github.com/pydantic/pydantic-ai/pull/5296); removed in [#5463](https://github.com/pydantic/pydantic-ai/pull/5463).
- `result.usage()` → `result.usage`, `result.timestamp()` → `result.timestamp`, and `stream.get()` → `stream.response` (method-style accessors become properties). See [#5263](https://github.com/pydantic/pydantic-ai/pull/5263).
- `StreamedResponse.usage()` → `StreamedResponse.usage`: the model-adapter streaming base class (`pydantic_ai.models.StreamedResponse`) now exposes `usage` as a property rather than a method. Relevant if you've subclassed `Model` and call `.usage()` on a streamed response. See [#5546](https://github.com/pydantic/pydantic-ai/pull/5546).
- `pydantic_graph.beta` imports move to the top-level `pydantic_graph` (e.g. `from pydantic_graph import GraphBuilder`). Announced via [#5306](https://github.com/pydantic/pydantic-ai/pull/5306); removed in [#5470](https://github.com/pydantic/pydantic-ai/pull/5470).
- Instrumentation format `version=1` and its version-1-only `InstrumentationSettings(event_mode=...)` and `InstrumentationSettings(logger_provider=...)` arguments are removed (deprecated in V1); `version=2`/`3`/`4` still work but now emit a deprecation warning. The default is `version=5` — see [Instrumentation defaults](#instrumentation-defaults-to-version-5-with-aggregated-usage-attributes) above for the default-behavior changes that ship with it. See [#5523](https://github.com/pydantic/pydantic-ai/pull/5523).
- The Outlines integration (`pydantic_ai.models.outlines.OutlinesModel`, `pydantic_ai.providers.outlines.OutlinesProvider`, and the `outlines-*` extras) is removed. If you'd like to keep using Outlines with Pydantic AI, please file an issue at [dottxt-ai/outlines](https://github.com/dottxt-ai/outlines/issues). See [#5444](https://github.com/pydantic/pydantic-ai/pull/5444).
- `pydantic_ai.native_tools.UrlContextTool` is removed; use `pydantic_ai.native_tools.WebFetchTool` instead. See [#5458](https://github.com/pydantic/pydantic-ai/pull/5458).
- Iterating [`Agent.run_stream_events()`][pydantic_ai.agent.AbstractAgent.run_stream_events] directly is no longer supported; it is now an async context manager only: `async with agent.run_stream_events(...) as events: async for event in events: ...`. See [#5440](https://github.com/pydantic/pydantic-ai/pull/5440).
- The bare (provider-prefix-less) model-name fallback is removed: `Agent('gpt-5')` now raises a `UserError` instead of inferring the provider; pass a provider-prefixed model name like `Agent('openai:gpt-5')`. (V1 emitted a deprecation warning for prefix-less legacy model names.) See [#5464](https://github.com/pydantic/pydantic-ai/pull/5464).
- Pydantic Evals: `EvaluationResult` and `EvaluatorFailure` are now keyword-only; `Dataset.evaluate()`/`evaluate_sync()` make `name`/`max_concurrency`/`progress`/`retry_task`/`retry_evaluators` keyword-only; `Dataset(name=...)` is now required; the `Evaluator.name` classmethod → `Evaluator.get_serialization_name()`. Announced via [#5547](https://github.com/pydantic/pydantic-ai/pull/5547); changed in [#5548](https://github.com/pydantic/pydantic-ai/pull/5548).
- Pydantic Evals: custom `Evaluator`s that advertised a default name or version by setting an `evaluation_name` / `evaluator_version` class attribute should override `Evaluator.get_default_evaluation_name()` / `Evaluator.get_evaluator_version()` instead; the attribute fallback is removed. Announced via [#5554](https://github.com/pydantic/pydantic-ai/pull/5554); removed in [#5556](https://github.com/pydantic/pydantic-ai/pull/5556).

### v1.0.1 (2025-09-05)

The following breaking change was accidentally left out of v1.0.0:

- See [#2808](https://github.com/pydantic/pydantic-ai/pull/2808) - Remove `Python` evaluator from `pydantic_evals` for security reasons

### v1.0.0 (2025-09-04)

- See [#2725](https://github.com/pydantic/pydantic-ai/pull/2725) - Drop support for Python 3.9
- See [#2738](https://github.com/pydantic/pydantic-ai/pull/2738) - Make many dataclasses require keyword arguments
- See [#2715](https://github.com/pydantic/pydantic-ai/pull/2715) - Remove `cases` and `averages` attributes from `pydantic_evals` spans
- See [#2798](https://github.com/pydantic/pydantic-ai/pull/2798) - Change `ModelRequest.parts` and `ModelResponse.parts` types from `list` to `Sequence`
- See [#2726](https://github.com/pydantic/pydantic-ai/pull/2726) - Default `InstrumentationSettings` version to 2
- See [#2717](https://github.com/pydantic/pydantic-ai/pull/2717) - Remove errors when passing `AsyncRetrying` or `Retrying` object to `AsyncTenacityTransport` or `TenacityTransport` instead of `RetryConfig`

### v0.x.x

Before V1, minor versions were used to introduce breaking changes:

**v0.8.0 (2025-08-26)**

See [#2689](https://github.com/pydantic/pydantic-ai/pull/2689) - `AgentStreamEvent` was expanded to be a union of `ModelResponseStreamEvent` and `HandleResponseEvent`, simplifying the `event_stream_handler` function signature. Existing code accepting `AgentStreamEvent | HandleResponseEvent` will continue to work.

**v0.7.6 (2025-08-26)**

The following breaking change was inadvertently released in a patch version rather than a minor version:

See [#2670](https://github.com/pydantic/pydantic-ai/pull/2670) - `TenacityTransport` and `AsyncTenacityTransport` now require the use of `pydantic_ai.retries.RetryConfig` (which is just a `TypedDict` containing the kwargs to `tenacity.retry`) instead of `tenacity.Retrying` or `tenacity.AsyncRetrying`.

**v0.7.0 (2025-08-12)**

See [#2458](https://github.com/pydantic/pydantic-ai/pull/2458) - `pydantic_ai.models.StreamedResponse` now yields a `FinalResultEvent` along with the existing `PartStartEvent` and `PartDeltaEvent`. If you're using `pydantic_ai.direct.model_request_stream` or `pydantic_ai.direct.model_request_stream_sync`, you may need to update your code to account for this.

See [#2458](https://github.com/pydantic/pydantic-ai/pull/2458) - `pydantic_ai.models.Model.request_stream` now receives a `run_context` argument. If you've implemented a custom `Model` subclass, you will need to account for this.

See [#2458](https://github.com/pydantic/pydantic-ai/pull/2458) - `pydantic_ai.models.StreamedResponse` now requires a `model_request_parameters` field and constructor argument. If you've implemented a custom `Model` subclass and implemented `request_stream`, you will need to account for this.

**v0.6.0 (2025-08-06)**

This release was meant to clean some old deprecated code, so we can get a step closer to V1.

See [#2440](https://github.com/pydantic/pydantic-ai/pull/2440) - The `next` method was removed from the `Graph` class. Use `async with graph.iter(...) as run:  run.next()` instead.

See [#2441](https://github.com/pydantic/pydantic-ai/pull/2441) - The `result_type`, `result_tool_name` and `result_tool_description` arguments were removed from the `Agent` class. Use `output_type` instead.

See [#2441](https://github.com/pydantic/pydantic-ai/pull/2441) - The `result_retries` argument was also removed from the `Agent` class. Use `output_retries` instead.

See [#2443](https://github.com/pydantic/pydantic-ai/pull/2443) - The `data` property was removed from the `FinalResult` class. Use `output` instead.

See [#2445](https://github.com/pydantic/pydantic-ai/pull/2445) - The `get_data` and `validate_structured_result` methods were removed from the
`StreamedRunResult` class. Use `get_output` and `validate_response_output` instead.

See [#2446](https://github.com/pydantic/pydantic-ai/pull/2446) - The `format_as_xml` function was moved to the `pydantic_ai.format_as_xml` module.
Import it via `from pydantic_ai import format_as_xml` instead.

See [#2451](https://github.com/pydantic/pydantic-ai/pull/2451) - Removed deprecated `Agent.result_validator` method, `Agent.last_run_messages` property, `AgentRunResult.data` property, and `result_tool_return_content` parameters from result classes.

**v0.5.0 (2025-08-04)**

See [#2388](https://github.com/pydantic/pydantic-ai/pull/2388) - The `source` field of an `EvaluationResult` is now of type `EvaluatorSpec` rather than the actual source `Evaluator` instance, to help with serialization/deserialization.

See [#2163](https://github.com/pydantic/pydantic-ai/pull/2163) - The `EvaluationReport.print` and `EvaluationReport.console_table` methods now require most arguments be passed by keyword.

**v0.4.0 (2025-07-08)**

See [#1799](https://github.com/pydantic/pydantic-ai/pull/1799) - Pydantic Evals `EvaluationReport` and `ReportCase` are now generic dataclasses instead of Pydantic models. If you were serializing them using `model_dump()`, you will now need to use the `EvaluationReportAdapter` and `ReportCaseAdapter` type adapters instead.

See [#1507](https://github.com/pydantic/pydantic-ai/pull/1507) - The `ToolDefinition` `description` argument is now optional and the order of positional arguments has changed from `name, description, parameters_json_schema, ...` to `name, parameters_json_schema, description, ...` to account for this.

**v0.3.0 (2025-06-18)**

See [#1142](https://github.com/pydantic/pydantic-ai/pull/1142) — Adds support for thinking parts.

We now convert the thinking blocks (`"<think>..."</think>"`) in provider specific text parts to
Pydantic AI `ThinkingPart`s. Also, as part of this release, we made the choice to not send back the
`ThinkingPart`s to the provider - the idea is to save costs on behalf of the user. In the future, we
intend to add a setting to customize this behavior.

**v0.2.0 (2025-05-12)**

See [#1647](https://github.com/pydantic/pydantic-ai/pull/1647) — usage makes sense as part of `ModelResponse`, and could be really useful in "messages" (really a sequence of requests and response). In this PR:

- Adds `usage` to `ModelResponse` (field has a default factory of `Usage()` so it'll work to load data that doesn't have usage)
- changes the return type of `Model.request` to just `ModelResponse` instead of `tuple[ModelResponse, Usage]`

**v0.1.0 (2025-04-15)**

See [#1248](https://github.com/pydantic/pydantic-ai/pull/1248) — the attribute/parameter name `result` was renamed to `output` in many places. Hopefully all changes keep a deprecated attribute or parameter with the old name, so you should get many deprecation warnings.

See [#1484](https://github.com/pydantic/pydantic-ai/pull/1484) — `format_as_xml` was moved and made available to import from the package root, e.g. `from pydantic_ai import format_as_xml`.

## Full Changelog

<div id="display-changelog">
  For the full changelog, see <a href="https://github.com/pydantic/pydantic-ai/releases">GitHub Releases</a>.
</div>

<script>
  fetch('/changelog.html').then(r => {
    if (r.ok) {
      r.text().then(t => {
        document.getElementById('display-changelog').innerHTML = t;
      });
    }
  });
</script>
