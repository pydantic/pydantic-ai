# V1 → V2 Migration Map

A lookup index for upgrading from Pydantic AI V1 to V2: find the V1 name you have in your code, read off the V2 name to replace it with.

The [Upgrade Guide](changelog.md) is the canonical source for *why* each change was made, the behavior changes that come with it, and the recommended upgrade path. This page is the fast path for the one question the guide answers in prose: **what replaced what.**

!!! tip "Upgrade through the latest V1 first"
    Most of what V2 removes is deprecated as of v1.100.0, and each deprecation warning names its replacement. Upgrading to the latest V1 and resolving every warning applies the bulk of this page mechanically, and leaves you only the [default behavior changes](changelog.md#changes-not-covered-by-deprecation-warnings) to reason about. Message history serialized with V1 still deserializes in V2.

## Agent configuration

Most V1 `Agent(...)` arguments that configured behavior moved onto [capabilities](capabilities/overview.md), a single composable primitive that bundles an agent's tools, [hooks](hooks.md), instructions, and model settings.

| V1 | V2 |
| --- | --- |
| `Agent(builtin_tools=[...])` | `Agent(capabilities=[NativeTool(...)])` |
| `Agent(event_stream_handler=...)` | `Agent(capabilities=[ProcessEventStream(...)])` (the `event_stream_handler=` argument on `run()`/`run_sync()`/`run_stream()`/`iter()` is unchanged) |
| `Agent(history_processors=...)` | `Agent(capabilities=[ProcessHistory(...)])` |
| `Agent(instrument=...)`, `Agent.from_spec(instrument=...)`, `Agent.from_file(instrument=...)`, `AgentSpec.instrument` | `Agent(capabilities=[Instrumentation(...)])` |
| `Agent(mcp_servers=[...])` | `Agent(toolsets=[...])` |
| `Agent(prepare_tools=...)` | `Agent(capabilities=[PrepareTools(...)])` |
| `Agent.run_mcp_servers()` | `async with agent:` |
| `Agent.sequential_tool_calls()` | `Agent.parallel_tool_call_execution_mode('sequential')` |
| `Agent.to_a2a()` | `fasta2a.pydantic_ai.agent_to_a2a` (install `fasta2a[pydantic-ai]>=0.6.1`) |
| `Agent.to_ag_ui()`, `AGUIApp`, `pydantic_ai.ag_ui` | `pydantic_ai.ui.ag_ui.AGUIAdapter` |
| `Agent('gpt-5')` (no provider prefix) | `Agent('openai:gpt-5')` — the prefix-less fallback now raises `UserError` |
| `Agent[None, ...]`, `RunContext[None]`, `Tool[None]` where deps aren't actually `None` | `Agent[object, ...]`, `RunContext[object]`, `Tool[object]` — the generic defaults changed from `None` to `object` |

## Models and providers

| V1 | V2 |
| --- | --- |
| `pydantic_ai.models.gemini.GeminiModel` | `pydantic_ai.models.google.GoogleModel` |
| `pydantic_ai.models.openai.OpenAIModel` | `pydantic_ai.models.openai.OpenAIChatModel` |
| `pydantic_ai.models.openai.OpenAIModelSettings` | `pydantic_ai.models.openai.OpenAIChatModelSettings` |
| `OpenAIChatModel(system_prompt_role=...)` | `OpenAIChatModel(profile=OpenAIModelProfile(openai_system_prompt_role=...))` — see the [note below](#not-a-straight-rename) if the model already resolves a profile |
| `OpenAICompaction(instructions=...)` | Removed |
| `pydantic_ai.models.outlines.OutlinesModel`, `pydantic_ai.providers.outlines.OutlinesProvider` | Removed, no replacement |
| `pydantic_ai.models.cached_async_http_client` | `pydantic_ai.models.create_async_http_client()` |
| `pydantic_ai.providers.google.GoogleProvider(vertexai=, location=, project=, credentials=)` | `pydantic_ai.providers.google_cloud.GoogleCloudProvider(...)` |
| `pydantic_ai.providers.google.GoogleGLAProvider` | `pydantic_ai.providers.google.GoogleProvider` |
| `pydantic_ai.providers.google.GoogleVertexProvider` | `pydantic_ai.providers.google_cloud.GoogleCloudProvider` |
| `pydantic_ai.providers.grok.GrokProvider`, `GrokModelName` | `pydantic_ai.providers.xai.XaiProvider` with `pydantic_ai.models.xai.XaiModel` / `XaiModelName` |
| `GoogleModelSettings['google_vertex_service_tier']`, `['google_service_tier']` | `GoogleModelSettings['google_cloud_service_tier']` |
| `StreamedResponse.usage()` (custom `Model` subclasses) | `StreamedResponse.usage` property |

### Model name prefixes

| V1 prefix | V2 prefix |
| --- | --- |
| `openai:` (Chat Completions) | `openai:` now means the Responses API; use `openai-chat:` for Chat Completions, `openai-responses:` to be explicit |
| `google-gla:` | `google:` |
| `google-vertex:`, `vertexai:` | `google-cloud:` |
| `gateway/gemini:`, `gateway/google-vertex:` | `gateway/google-cloud:` |
| `grok:` | `xai:` |

## Model profiles

[`ModelProfile`][pydantic_ai.profiles.ModelProfile] and its subclasses are now `TypedDict`s rather than dataclasses. Constructing one (`OpenAIModelProfile(field=value)`) is unchanged; reading, mutating, or merging one is not. The full recipe table is in the Upgrade Guide under [`ModelProfile` is now a `TypedDict`](changelog.md#modelprofile-is-now-a-typeddict).

| V1 | V2 |
| --- | --- |
| `profile.field` | `profile.get('field', <default>)` — defaults are exported from [`pydantic_ai.profiles`][pydantic_ai.profiles] |
| `profile.field = value` | `profile['field'] = value` |
| `dataclasses.replace(profile, field=value)` | `{**profile, 'field': value}` |
| `profile.update(other)` | [`merge_profile(profile, other)`][pydantic_ai.profiles.merge_profile] |
| `OpenAIModelProfile.from_profile(p)` | `p` |
| `isinstance(profile, OpenAIModelProfile)` | Not supported on a `TypedDict` — check key presence instead |
| `OpenAIModelProfile.openai_supports_sampling_settings` | `OpenAIModelProfile.openai_unsupported_model_settings` — **not a rename**, see [below](#not-a-straight-rename) |
| `OpenAIModelProfile.openai_builtin_tools` | `OpenAIModelProfile.openai_native_tools` |

### Not a straight rename

Two of the OpenAI rows above need more than a find-and-replace:

- **`openai_supports_sampling_settings` → `openai_unsupported_model_settings` changes shape, not just name.** The V1 field was a `bool` covering the sampling settings as a group; the V2 field is a sequence of the specific setting names to drop. `openai_supports_sampling_settings=False` becomes an explicit list of what the model doesn't accept, e.g. `openai_unsupported_model_settings=('temperature', 'top_p')`. `True` was the default, so it simply goes away.
- **`system_prompt_role` moves from a model argument into a profile.** If you were already passing `profile=` to the model, merge the setting into that profile rather than replacing it — a second `OpenAIModelProfile(...)` overrides the first wholesale. Profiles are `TypedDict`s in V2, so merging is `{**existing_profile, 'openai_system_prompt_role': 'user'}` or [`merge_profile()`][pydantic_ai.profiles.merge_profile].

## MCP

The per-transport server classes collapsed into a single [`MCPToolset`][pydantic_ai.mcp.MCPToolset] whose transport is inferred from the arguments you pass. Its defaults differ from the V1 classes' — notably `max_retries`, `read_timeout`, `init_timeout`, and `elicitation_handler` — so re-read [MCP Client](mcp/client.md) rather than assuming your V1 timeouts carried over.

| V1 | V2 |
| --- | --- |
| `MCPServerStdio`, `MCPServerSSE`, `MCPServerStreamableHTTP`, `MCPServerHTTP` | [`pydantic_ai.mcp.MCPToolset`][pydantic_ai.mcp.MCPToolset] |
| `FastMCPToolset` (and the `fastmcp` extra) | `MCPToolset` |
| `load_mcp_servers` | [`pydantic_ai.mcp.load_mcp_toolsets`][pydantic_ai.mcp.load_mcp_toolsets] |
| `Agent.run_mcp_servers()` | `async with agent:` |
| `MCP(url=...)` running remotely by default | `MCP(url=..., native=True)` to keep the V1 behavior; `MCP(url=...)` now runs the server locally |

## Tools and toolsets

| V1 | V2 |
| --- | --- |
| `pydantic_ai.builtin_tools` | `pydantic_ai.native_tools` |
| `AgentBuiltinTool` | `AgentNativeTool` |
| `pydantic_ai.native_tools.UrlContextTool` | [`pydantic_ai.native_tools.WebFetchTool`][pydantic_ai.native_tools.WebFetchTool] |
| `builtin=` argument | `native=` |
| `pydantic_ai.output.DeferredToolCalls` | [`DeferredToolRequests`][pydantic_ai.tools.DeferredToolRequests] |
| `DeferredToolCalls.tool_calls` | [`DeferredToolRequests.calls`][pydantic_ai.tools.DeferredToolRequests.calls] |
| `DeferredToolCalls.tool_defs` | Removed — it always returned an empty dict in V1 |
| `pydantic_ai.toolsets.external.DeferredToolset` | [`ExternalToolset`][pydantic_ai.toolsets.ExternalToolset] |
| `FunctionToolset.tool()` on a context-free callable | [`FunctionToolset.tool_plain()`][pydantic_ai.toolsets.FunctionToolset.tool_plain] — `tool()` now raises if the first parameter isn't a `RunContext` |
| `pydantic_ai.ext.aci.tool_from_aci`, `ACIToolset` | Removed; wrap the tool schemas with [`Tool.from_schema`][pydantic_ai.tools.Tool.from_schema] |
| A `prepare` callback returning `None` | Return `[]` — returning `None` now raises `TypeError` instead of stripping all tools |
| `WebSearch()` / `WebFetch()` falling back to a local implementation | `WebSearch(local='duckduckgo')` / `WebFetch(local=True)` — both are native-only by default and now raise on models that don't support them |

## Messages, events and usage

The serialized `part_kind` wire values and the old field names' validation aliases are retained, so message history written by V1 still deserializes in V2.

| V1 | V2 |
| --- | --- |
| `BuiltinToolCallPart`, `BuiltinToolReturnPart` | `NativeToolCallPart`, `NativeToolReturnPart` |
| `BuiltinToolCallEvent`, `BuiltinToolResultEvent` | Removed — native tool calls surface via `PartStartEvent`/`PartDeltaEvent` only |
| `FunctionToolCallEvent`/`FunctionToolResultEvent` for output tools | `OutputToolCallEvent`/`OutputToolResultEvent` |
| `FunctionToolCallEvent.call_id` | `FunctionToolCallEvent.tool_call_id` |
| `FunctionToolResultEvent(result=...)`, `.result` | `FunctionToolResultEvent(part=...)`, `.part` |
| `ModelResponse.vendor_details` | `ModelResponse.provider_details` |
| `ModelResponse.vendor_id`, `ModelResponse.provider_request_id` | `ModelResponse.provider_response_id` |
| `ModelResponse.builtin_tool_calls` | [`ModelResponse.native_tool_calls`][pydantic_ai.messages.ModelResponse.native_tool_calls] |
| `ModelResponse.price()` | [`ModelResponse.cost()`][pydantic_ai.messages.ModelResponse.cost] |
| `Usage` | [`RunUsage`][pydantic_ai.usage.RunUsage] |
| `usage.request_tokens`, `usage.response_tokens` | `usage.input_tokens`, `usage.output_tokens` |
| `UsageLimits(request_tokens_limit=)`, `(response_tokens_limit=)` | `UsageLimits(input_tokens_limit=)`, `(output_tokens_limit=)` |

## Results and streaming

| V1 | V2 |
| --- | --- |
| `result.usage()`, `result.timestamp()` | `result.usage`, `result.timestamp` (properties) |
| `stream.get()` | `stream.response` |
| `StreamedRunResult.stream` | [`stream_output`][pydantic_ai.result.StreamedRunResult.stream_output] |
| `StreamedRunResult.stream_structured` | [`stream_response`][pydantic_ai.result.StreamedRunResult.stream_response] |
| `StreamedRunResult.stream_responses()` (plural, yielding `(response, is_last)`) | `stream_response()` (singular, yielding a bare `ModelResponse`; read the old `is_last` as `response.state != 'incomplete'`) |
| `StreamedRunResult.validate_structured_output` | [`validate_response_output`][pydantic_ai.result.StreamedRunResult.validate_response_output] |
| `async for event in agent.run_stream_events(...)` | `async with agent.run_stream_events(...) as events:` then iterate — it is an async context manager only |

## Pydantic Graph

| V1 | V2 |
| --- | --- |
| `from pydantic_graph.beta import GraphBuilder` | `from pydantic_graph import GraphBuilder` |
| `pydantic_graph.persistence` | No `pydantic_graph` equivalent — the builder API doesn't snapshot graph state. To save, resume, and fork **agent run** state, [Pydantic AI Harness](https://pydantic.dev/docs/ai/harness/) ships [`StepPersistence`](https://pydantic.dev/docs/ai/harness/step-persistence/) |
| `pydantic_graph.mermaid` | Removed — render diagrams with `Graph.render()` |

## Pydantic Evals

| V1 | V2 |
| --- | --- |
| `Evaluator.name` (classmethod) | `Evaluator.get_serialization_name()` |
| `evaluation_name` class attribute | [`Evaluator.get_default_evaluation_name()`][pydantic_evals.evaluators.Evaluator.get_default_evaluation_name] |
| `evaluator_version` class attribute | [`Evaluator.get_evaluator_version()`][pydantic_evals.evaluators.Evaluator.get_evaluator_version] |
| `Dataset(...)` without a name | `Dataset(name=...)` — now required |
| Positional `name`/`max_concurrency`/`progress`/`retry_task`/`retry_evaluators` on `Dataset.evaluate()`/`evaluate_sync()` | Keyword-only |
| Positional construction of `EvaluationResult` / `EvaluatorFailure` | Keyword-only |

## Instrumentation

| V1 | V2 |
| --- | --- |
| `InstrumentationSettings(version=1)`, `event_mode=`, `logger_provider=` | Removed; versions 2–4 still work but warn. The default is version 5 |
| Reading run-span token usage from `gen_ai.usage.*` | Run spans report `gen_ai.aggregated_usage.*`; set `use_aggregated_usage_attribute_names=False` to keep the V1 names |

## Packaging

A bare `uv add pydantic-ai` / `pip install pydantic-ai` now installs a slimmer set of extras. `bedrock`, `groq`, `mistral`, `cohere`, `xai`, `huggingface`, `temporal`, `ag-ui`, `ui`, and `spec` are no longer included by default — add the ones you use, e.g. `uv add 'pydantic-ai[bedrock,groq]'`. The `outlines-*`, `vertexai`, `fastmcp`, and `a2a` extras are removed outright. See the [installation guide](install.md) for the full list.

## Behavior changes with no code change

These flip without any symbol changing name, so they can't be found by grepping for an old name. Each is explained in full in the Upgrade Guide under [changes not covered by deprecation warnings](changelog.md#changes-not-covered-by-deprecation-warnings).

- The default `end_strategy` changed from `'early'` to `'graceful'`, so function tools requested alongside a successful output tool now run instead of being skipped. See [Parallel Output Tool Calls](output.md#parallel-output-tool-calls).
- `sequential=True` on a tool is now a per-tool barrier rather than a batch-wide serial switch, and applies to output tools too.
- [`capture_run_messages()`][pydantic_ai.capture_run_messages] also captures the partial request/response of an interrupted run, marked `state='interrupted'`.
- A resolved model profile now carries fields from other profile classes, where V1 filtered them out.
