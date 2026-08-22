# Upgrade Guide

In September 2025, Pydantic AI reached V1 and committed to API stability: no changes that break your code until V2. V2 is now available, collecting the breaking and behavior changes that stability guarantee didn't allow. This guide is the canonical place to learn what's in V2, how to install it, and how to upgrade; for the guarantees behind these version numbers, see the [Version Policy](version-policy.md).

## Breaking Changes

Here's a filtered list of the breaking changes for each version to help you upgrade Pydantic AI.

### v2.0.0 (2026-06-23)

The stable V2.0 release. There are no new breaking or behavior changes since the betas; the full breaking-change list and recommended upgrade path are in the [v2.0.0b1](#v200b1-2026-05-20) entry below. Install it with:

```bash
uv add pydantic-ai
```

### v2.0.0b7 (2026-06-10)

The seventh V2 beta, forked from **v1.107.0**. There are no new V2 breaking or behavior changes since [v2.0.0b6](#v200b6-2026-06-04) below — everything in that entry applies unchanged — but this beta picks up the latest V1 release on top, which adds Claude Fable 5 / Mythos 5 model support and OpenRouter prompt caching (`CachePoint`), plus `known_model_names()` and Anthropic fixes; see the [v1.107.0 release notes](https://github.com/pydantic/pydantic-ai/releases/tag/v1.107.0) for the full list.

Install it the same way, pinning the exact pre-release version:

```bash
pip/uv-add "pydantic-ai==2.0.0b7"
```

For the full breaking-change list and the recommended upgrade path, see the [v2.0.0b1](#v200b1-2026-05-20) entry below; the only difference is that the latest V1 to upgrade through first is now **v1.107.0**.

### v2.0.0b6 (2026-06-04)

The sixth V2 beta, forked from **v1.106.0**. There are no new V2 breaking or behavior changes since [v2.0.0b5](#v200b5-2026-06-02) below — everything in that entry applies unchanged — but this beta picks up the latest V1 release on top, which adds `api_host`/`timeout` configuration and base `seed` mapping for the xAI provider, plus streaming and data-URI handling fixes; see the [v1.106.0 release notes](https://github.com/pydantic/pydantic-ai/releases/tag/v1.106.0) for the full list.

Install it the same way, pinning the exact pre-release version:

```bash
pip/uv-add "pydantic-ai==2.0.0b6"
```

For the full breaking-change list and the recommended upgrade path, see the [v2.0.0b1](#v200b1-2026-05-20) entry below; the only difference is that the latest V1 to upgrade through first is now **v1.106.0**.

### v2.0.0b5 (2026-06-02)

The fifth V2 beta, forked from **v1.105.0**. There are no new V2 breaking or behavior changes since [v2.0.0b4](#v200b4-2026-05-28) below — everything in that entry (including the prepare-callbacks change) still applies — but this beta picks up the latest V1 release on top, which adds [on-demand (deferred-loading) capabilities](https://github.com/pydantic/pydantic-ai/pull/5230) and [Grok 4.3 `reasoning_effort` support](https://github.com/pydantic/pydantic-ai/pull/5454), plus `GoogleModelSettings.google_cached_content` and Temporal `gateway/` fixes; see the [v1.105.0 release notes](https://github.com/pydantic/pydantic-ai/releases/tag/v1.105.0) for the full list.

Install it the same way, pinning the exact pre-release version:

```bash
pip/uv-add "pydantic-ai==2.0.0b5"
```

For the full breaking-change list and the recommended upgrade path, see the [v2.0.0b1](#v200b1-2026-05-20) entry below; the only difference is that the latest V1 to upgrade through first is now **v1.105.0**.

### v2.0.0b4 (2026-05-28)

The fourth V2 beta, forked from **v1.104.0**. One new V2 behavior change since [v2.0.0b3](#v200b3-2026-05-22):

- Prepare callbacks (`prepare_tools=` / `PrepareTools` capability) that return `None` now raise `TypeError` instead of silently stripping all tools. V1.103.0 announces this change via `PydanticAIDeprecationWarning` (see [#5188](https://github.com/pydantic/pydantic-ai/pull/5188)); V2 turns the warning into a hard error (see [#5668](https://github.com/pydantic/pydantic-ai/pull/5668)). Return an empty list (`[]`) when you mean "no tools for this turn."

This beta also picks up two V1 releases on top — [v1.103.0](https://github.com/pydantic/pydantic-ai/releases/tag/v1.103.0) and [v1.104.0](https://github.com/pydantic/pydantic-ai/releases/tag/v1.104.0) — together adding Claude Opus 4.8 support, `McpServer.list_prompts` / `get_prompt`, message-timestamp roundtripping through `VercelAIAdapter`'s `UIMessage.metadata`, OpenRouter eager input streaming, and several Bedrock and UI fixes. See those release notes for the full list.

Install it the same way, pinning the exact pre-release version:

```bash
pip/uv-add "pydantic-ai==2.0.0b4"
```

For the full breaking-change list and the recommended upgrade path, see the [v2.0.0b1](#v200b1-2026-05-20) entry below; the only difference is that the latest V1 to upgrade through first is now **v1.104.0**.

### v2.0.0b3 (2026-05-22)

The third V2 beta, forked from **v1.102.0**. There are no new V2 breaking changes since [v2.0.0b1](#v200b1-2026-05-20) below — everything in that entry applies unchanged — but this beta picks up the latest V1 release on top, which is a bug-fix release; see the [v1.102.0 release notes](https://github.com/pydantic/pydantic-ai/releases/tag/v1.102.0) for the full list.

Install it the same way, pinning the exact pre-release version:

```bash
pip/uv-add "pydantic-ai==2.0.0b3"
```

For the full breaking-change list and the recommended upgrade path, see the [v2.0.0b1](#v200b1-2026-05-20) entry below; the only difference is that the latest V1 to upgrade through first is now **v1.102.0**.

### v2.0.0b2 (2026-05-21)

The second V2 beta, forked from **v1.101.0**. There are no new V2 breaking changes since [v2.0.0b1](#v200b1-2026-05-20) below — everything in that entry applies unchanged — but this beta picks up the latest V1 release on top, which adds the [pending message queue](https://github.com/pydantic/pydantic-ai/pull/4980) (`ctx.enqueue` / `agent_run.enqueue`).

Install it the same way, pinning the exact pre-release version:

```bash
pip/uv-add "pydantic-ai==2.0.0b2"
```

For the full breaking-change list and the recommended upgrade path, see the [v2.0.0b1](#v200b1-2026-05-20) entry below; the only difference is that the latest V1 to upgrade through first is now **v1.101.0**.

### v2.0.0b1 (2026-05-20)

The first V2 beta, forked from **v1.100.0**, which deprecates most of what V2 removes. V2 leans into a harness-first design with [capabilities](capabilities/overview.md) as a core primitive: a single, composable unit that bundles an agent's tools, [hooks](hooks.md), instructions, and model settings, reaching every layer of the agent through one concept. Many of V2's changes move configuration that used to be spread across `Agent` arguments onto that primitive, alongside the behavior changes that V1's stability guarantee didn't allow. Pydantic AI stays a small core: some capabilities ship with it, more come from the first-party [Pydantic AI Harness](https://pydantic.dev/docs/ai/harness/), and others are third-party or your own.

The breaking changes below are split into two groups:

- [**Changes not covered by deprecation warnings**](#changes-not-covered-by-deprecation-warnings) — removals and behavior changes that couldn't be announced via a V1 deprecation warning. Review these even if you're already on the latest V1 with no warnings.
- [**Changes covered by deprecation warnings**](#changes-covered-by-deprecation-warnings) — if you upgraded to the latest V1 and resolved every deprecation warning, you've already made these. They're listed with full before → after for reference.

**Recommended upgrade path.** To make the jump as smooth as possible:

1. **Upgrade to the latest V1 release.** Most of what V2 removes is deprecated as of **v1.100.0** (the release this beta is forked from), so any V1 at or above that version surfaces those warnings.
2. **Resolve every deprecation warning.** The [changes covered by deprecation warnings](#changes-covered-by-deprecation-warnings) were announced in V1 via warnings that name the new API and, where possible, include a migration snippet. Run your test suite (or app) with warnings visible and address each one — by hand or by pointing a coding agent at them — to migrate across the bulk of V2 ahead of time.
3. **Upgrade to V2** and make the [changes not covered by deprecation warnings](#changes-not-covered-by-deprecation-warnings) — primarily default-behavior changes and a handful of removals with no V1 deprecation.

You can also upgrade straight to V2 and work through the list below directly — it's organized so a coding agent can apply the code changes mechanically. Resolving deprecation warnings on the latest V1 first is still the smoother path, since it spreads the work out and leaves you only the behavior changes to reason about consciously at the end.

Message history serialized with V1 (via [`ModelMessagesTypeAdapter`][pydantic_ai.messages.ModelMessagesTypeAdapter]) continues to deserialize in V2.

#### Changes not covered by deprecation warnings

These removals and behavior changes could not be announced via a V1 deprecation warning, so review them even if you've resolved every deprecation warning on the latest V1.

**Code changes:**

- Generic type parameter defaults changed from `None` to `object`: an un-parameterized `Agent(...)` now infers `Agent[object, str]` instead of `Agent[None, str]`, and the `pydantic_graph` `StateT`/`RunEndT`/`DepsT` defaults changed to match. Update explicit `Agent[None, ...]`, `RunContext[None]`, and `Tool[None]` annotations that don't actually require `None` dependencies to use `object`. This is a type-checking-only change; runtime behavior is unchanged. See [#5307](https://github.com/pydantic/pydantic-ai/pull/5307).
- The `pydantic_graph.persistence` package and the `pydantic_graph.mermaid` module are removed, with no V2 equivalent for standalone Mermaid generation (render diagrams with `Graph.render()`). The builder API deliberately does not snapshot graph state, so there is no `pydantic_graph` replacement for the persistence package; to save, resume, and fork **agent run** state, [Pydantic AI Harness](https://pydantic.dev/docs/ai/harness/) ships the [`StepPersistence`](https://pydantic.dev/docs/ai/harness/step-persistence/) capability. The move of the [`GraphBuilder`][pydantic_graph.GraphBuilder] API out of `pydantic_graph.beta` to the top-level `pydantic_graph` *was* deprecation-announced; see [below](#changes-covered-by-deprecation-warnings). See [#5470](https://github.com/pydantic/pydantic-ai/pull/5470).
- [`ModelProfile`][pydantic_ai.profiles.ModelProfile] and its subclasses are now `TypedDict`s instead of dataclasses. Passing `profile=OpenAIModelProfile(field=value)` into a model still works unchanged; the migration only matters if you read or mutate profile fields, or call `.update()`/`.from_profile()`. See [`ModelProfile` is now a `TypedDict`](#modelprofile-is-now-a-typeddict) below. ([#5481](https://github.com/pydantic/pydantic-ai/pull/5481))

**Default behavior changes** — same API, different runtime behavior (roughly ordered by how many users they affect):

- A bare `uv add pydantic-ai` / `pip install pydantic-ai` now installs a slimmer set of extras (frontier providers plus minimal integrations); providers like `bedrock`, `groq`, and `mistral` are no longer included by default, so you'll need to add the extras you use. See [Slimmer default extras](#slimmer-default-pydantic-ai-extras) below. ([#5467](https://github.com/pydantic/pydantic-ai/pull/5467))
- The default `end_strategy` changed from `'early'` to `'graceful'`: when a model calls function tools in the same response as a successful output tool, those function tools now run (and their side effects happen) instead of being skipped, and tool calls run in the order the model emitted them. See [Parallel tool-call execution order](#parallel-tool-call-execution-runs-in-emission-order) below. ([#5339](https://github.com/pydantic/pydantic-ai/pull/5339))
- The default instrumentation format is now version 5, and agent run spans report token usage under `gen_ai.aggregated_usage.*`. See [Instrumentation defaults](#instrumentation-defaults-to-version-5-with-aggregated-usage-attributes) below. ([#5523](https://github.com/pydantic/pydantic-ai/pull/5523))
- [`capture_run_messages()`][pydantic_ai.capture_run_messages] now also captures the partial `ModelRequest`/`ModelResponse` from an interrupted run, marked with `state='interrupted'` (a new `ModelRequest.state` field is added). Code that asserts on exact captured-message counts on error paths may need updating. See [#5364](https://github.com/pydantic/pydantic-ai/pull/5364).
- Output tool calls and returns now emit dedicated `OutputToolCallEvent`/`OutputToolResultEvent` instead of `FunctionToolCallEvent`/`FunctionToolResultEvent`. Separately, native tool calls and returns no longer emit dedicated events at all — the `BuiltinToolCallEvent`/`BuiltinToolResultEvent` classes are removed and they surface only via the standard `PartStartEvent`/`PartDeltaEvent`. See [#5332](https://github.com/pydantic/pydantic-ai/pull/5332) and [#5476](https://github.com/pydantic/pydantic-ai/pull/5476).

##### [`ModelProfile`][pydantic_ai.profiles.ModelProfile] is now a `TypedDict` {#modelprofile-is-now-a-typeddict}

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

The default [`end_strategy`][pydantic_ai.agent.EndStrategy] changed from `'early'` to `'graceful'`. This only affects responses where a model calls function tools in the *same* response as an [output tool](output.md#tool-output) (the call that ends the run). When that output tool **succeeds**, the function tools requested alongside it now **run** by default instead of being skipped, so their side effects happen and their results reach the model if the run continues; and a function tool's [`ModelRetry`][pydantic_ai.exceptions.ModelRetry] now suppresses the output result so the model can correct itself on the next round. The case where *every* output tool fails is unchanged: function tools run and the run continues either way. Most agents don't need any change. If you relied on the run ending the instant an output tool succeeds — skipping any function tools requested in the same response — set `end_strategy='early'` explicitly.

The [`sequential=True`](tools-advanced.md#parallel-tool-calls-concurrency) flag on a tool is now a per-tool **barrier** rather than a batch-wide serial switch: a sequential tool runs alone, but other tools in the same response still run in parallel around it. The barrier now also applies to output tools via [`ToolOutput(sequential=True)`][pydantic_ai.output.ToolOutput], not just function tools. To run *all* of a run's tools serially, wrap the run in [`agent.parallel_tool_call_execution_mode('sequential')`][pydantic_ai.agent.AbstractAgent.parallel_tool_call_execution_mode] or set `parallel_tool_calls=False` on the [model settings][pydantic_ai.settings.ModelSettings].

See [Parallel Output Tool Calls](output.md#parallel-output-tool-calls) for the full behavior of all three strategies, and [#5339](https://github.com/pydantic/pydantic-ai/pull/5339).

##### Slimmer default `pydantic-ai` extras

A bare `uv add pydantic-ai` / `pip install pydantic-ai` now installs `pydantic-ai-slim[openai,anthropic,google,cli,mcp,evals,web,retries,logfire]` — frontier providers plus minimal integrations. Providers and integrations that were previously bundled are no longer installed by default; add the ones you use explicitly, e.g. `uv add 'pydantic-ai[bedrock,groq]'`: `bedrock`, `groq`, `mistral`, `cohere`, `xai`, `huggingface`, `temporal`, `ag-ui`, `ui`, and `spec`. See the [installation guide](install.md) for the full list of extras.

Some `pydantic-ai-slim` extras were also removed outright (not just dropped from the default bundle): the `outlines-*` extras (the Outlines integration is removed), `vertexai` (Vertex AI is now served by the `google` extra), `fastmcp` (the FastMCP back-compat shim is removed), and `a2a` (A2A now lives in the upstream `fasta2a` package). See [#5467](https://github.com/pydantic/pydantic-ai/pull/5467).

##### Instrumentation defaults to version 5 with aggregated usage attributes

The default [instrumentation format](logfire.md#configuring-data-format) is now version 5 (versions 2–4 still work but emit a deprecation warning; version 1 and its `event_mode=`/`logger_provider=` arguments are removed). In version 5, deferred tool calls (`CallDeferred`/`ApprovalRequired`) are no longer recorded as span errors.

Separately, [`InstrumentationSettings`][pydantic_ai.models.instrumented.InstrumentationSettings]'s [`use_aggregated_usage_attribute_names`](logfire.md#aggregated-usage-attribute-names) now defaults to `True`: agent run spans report token usage under `gen_ai.aggregated_usage.*` while model request spans keep `gen_ai.usage.*`, which avoids double-counting in backends that sum parent and child usage. Dashboards and alerts that read token usage from run spans must be updated, or set `use_aggregated_usage_attribute_names=False` to keep the V1 attribute names.

See [#5523](https://github.com/pydantic/pydantic-ai/pull/5523).

#### Changes covered by deprecation warnings

These changes were announced in the latest V1 releases via deprecation warnings that name the replacement API. If you upgraded to the latest V1 and resolved every warning, you've already made them. The [V1 → V2 migration map](migration.md) carries the full before → after for every symbol; this section records the behavior that changes with them and the PRs each one landed in.

**Behavior changes that flip silently if the V1 deprecation warning was not addressed** — even though these were announced, an unaddressed warning means the behavior changes without raising an error, so confirm you've handled them:

- The bare `openai:` model prefix now uses the OpenAI Responses API ([`OpenAIResponsesModel`][pydantic_ai.models.openai.OpenAIResponsesModel]) instead of the Chat Completions API ([`OpenAIChatModel`][pydantic_ai.models.openai.OpenAIChatModel]). Use `openai-chat:` to keep Chat Completions, or `openai-responses:` to opt into the new default explicitly. Announced via [#5334](https://github.com/pydantic/pydantic-ai/pull/5334); flipped in [#5469](https://github.com/pydantic/pydantic-ai/pull/5469).
- Provider-adaptive `WebSearch` and `WebFetch` capabilities are now native-only and raise on models that don't support them, and `MCP(url=...)` runs the server locally by default. Restore the V1 fallbacks with `WebSearch(local='duckduckgo')`, `WebFetch(local=True)`, and `MCP(url=..., native=True)`. Announced via [#5331](https://github.com/pydantic/pydantic-ai/pull/5331); changed in [#5333](https://github.com/pydantic/pydantic-ai/pull/5333).

**API removals and renames** — look each old name up in the [migration map](migration.md); the PRs that announced and landed each group are:

| Group | PRs |
| --- | --- |
| Providers: Grok → xAI (`grok:` prefix → `xai:`) | [#5460](https://github.com/pydantic/pydantic-ai/pull/5460) |
| Providers: Google GLA/Vertex → `GoogleProvider`/`GoogleCloudProvider`, `GeminiModel` → `GoogleModel` | [#5336](https://github.com/pydantic/pydantic-ai/pull/5336), [#5543](https://github.com/pydantic/pydantic-ai/pull/5543), [#5479](https://github.com/pydantic/pydantic-ai/pull/5479) |
| Models: `OpenAIModel` → `OpenAIChatModel`, `system_prompt_role` and sampling-settings move into the profile | [#5468](https://github.com/pydantic/pydantic-ai/pull/5468) |
| Models: `StreamedResponse.usage()` becomes a property (affects custom `Model` subclasses) | [#5546](https://github.com/pydantic/pydantic-ai/pull/5546) |
| Models: bare provider-prefix-less names (`Agent('gpt-5')`) now raise `UserError` | [#5464](https://github.com/pydantic/pydantic-ai/pull/5464) |
| Native tools: `builtin_tools` → `native_tools` throughout | [#5338](https://github.com/pydantic/pydantic-ai/pull/5338), [#5396](https://github.com/pydantic/pydantic-ai/pull/5396) |
| Native tools: `UrlContextTool` → `WebFetchTool` | [#5458](https://github.com/pydantic/pydantic-ai/pull/5458) |
| MCP: per-transport server classes → `MCPToolset` | [#5325](https://github.com/pydantic/pydantic-ai/pull/5325), [#5337](https://github.com/pydantic/pydantic-ai/pull/5337) |
| Agent config → capabilities: `instrument=` | [#5434](https://github.com/pydantic/pydantic-ai/pull/5434) |
| Agent config → capabilities: `event_stream_handler=`, `prepare_tools=` | [#5335](https://github.com/pydantic/pydantic-ai/pull/5335), [#5475](https://github.com/pydantic/pydantic-ai/pull/5475) |
| Agent config → capabilities: `history_processors=` | [#5425](https://github.com/pydantic/pydantic-ai/pull/5425) |
| Agent config: `mcp_servers=` → `toolsets=`, `sequential_tool_calls()` → `parallel_tool_call_execution_mode()` | [#5466](https://github.com/pydantic/pydantic-ai/pull/5466) |
| Tools: `DeferredToolCalls` → `DeferredToolRequests`, `DeferredToolset` → `ExternalToolset` | [#5459](https://github.com/pydantic/pydantic-ai/pull/5459) |
| Tools: `FunctionToolset.tool()` now requires a `RunContext` first parameter | [#5462](https://github.com/pydantic/pydantic-ai/pull/5462) |
| Tools: `pydantic_ai.ext.aci` removed (wrap with `Tool.from_schema`) | [#5510](https://github.com/pydantic/pydantic-ai/pull/5510), [#5467](https://github.com/pydantic/pydantic-ai/pull/5467) |
| Usage and response-field renames (`request_tokens` → `input_tokens`, `vendor_details` → `provider_details`, …) | [#5476](https://github.com/pydantic/pydantic-ai/pull/5476) |
| Events: dedicated `OutputToolCallEvent`/`OutputToolResultEvent`, `FunctionToolResultEvent.result` → `.part` | [#5332](https://github.com/pydantic/pydantic-ai/pull/5332) |
| Streaming: `StreamedRunResult` accessor renames, `stream_responses()` → `stream_response()` | [#5296](https://github.com/pydantic/pydantic-ai/pull/5296), [#5463](https://github.com/pydantic/pydantic-ai/pull/5463) |
| Streaming: `Agent.run_stream_events()` is an async context manager only | [#5440](https://github.com/pydantic/pydantic-ai/pull/5440) |
| Results: `result.usage()`/`result.timestamp()`/`stream.get()` become properties | [#5263](https://github.com/pydantic/pydantic-ai/pull/5263) |
| Integrations: `Agent.to_a2a()` and the bundled `fasta2a` move upstream | [#5426](https://github.com/pydantic/pydantic-ai/pull/5426), [#5502](https://github.com/pydantic/pydantic-ai/pull/5502) |
| Integrations: `Agent.to_ag_ui()`/`AGUIApp` → `AGUIAdapter`, `cached_async_http_client` → `create_async_http_client()` | [#5345](https://github.com/pydantic/pydantic-ai/pull/5345), [#5464](https://github.com/pydantic/pydantic-ai/pull/5464) |
| Graph: `pydantic_graph.beta` imports move to top-level `pydantic_graph` | [#5306](https://github.com/pydantic/pydantic-ai/pull/5306), [#5470](https://github.com/pydantic/pydantic-ai/pull/5470) |
| Instrumentation: `version=1` and its `event_mode=`/`logger_provider=` arguments removed | [#5523](https://github.com/pydantic/pydantic-ai/pull/5523) |
| Pydantic Evals: keyword-only arguments, required `Dataset(name=...)`, `Evaluator.name` → `get_serialization_name()` | [#5547](https://github.com/pydantic/pydantic-ai/pull/5547), [#5548](https://github.com/pydantic/pydantic-ai/pull/5548) |
| Pydantic Evals: `evaluation_name`/`evaluator_version` class attributes → `get_default_evaluation_name()`/`get_evaluator_version()` | [#5554](https://github.com/pydantic/pydantic-ai/pull/5554), [#5556](https://github.com/pydantic/pydantic-ai/pull/5556) |

Four of these carry a caveat the name mapping alone doesn't convey:

- **MCP:** `Agent.set_mcp_sampling_model()` is *not* removed — it still sets the sampling model on every registered `MCPToolset`, while `MCPToolset(sampling_model=...)` sets it on one toolset at construction.
- **Instrumentation:** `version=2`/`3`/`4` still work but now emit a deprecation warning, and the default is `version=5` — see [Instrumentation defaults](#instrumentation-defaults-to-version-5-with-aggregated-usage-attributes) above for the behavior changes that ship with it.
- **`Agent(instrument=...)`:** only the constructor arguments are removed. The `Agent.instrument` property, `Agent.instrument_all()`, and `InstrumentedModel` are unchanged.
- **Outlines:** the integration is removed outright. If you'd like to keep using Outlines with Pydantic AI, please file an issue at [dottxt-ai/outlines](https://github.com/dottxt-ai/outlines/issues). See [#5444](https://github.com/pydantic/pydantic-ai/pull/5444).

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
