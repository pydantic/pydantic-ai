# Maintainer Mindset From Douwe Comments

Purpose: summarize how Douwe appears to evaluate Pydantic AI issues, PRs, feature requests, and internals work, based on recent GitHub comments and reviews.

## Methodology

- Repository: `pydantic/pydantic-ai`.
- Author/login: searched `DouweM` after confirming `douwem` resolves canonically in GitHub URLs/comments as `DouweM`.
- Window: 2025-12-18 through 2026-05-18.
- Commands/API shape:
  - `gh search issues --repo pydantic/pydantic-ai --commenter DouweM --updated ">=2025-12-18" --limit 200 --json ...`
  - `gh search prs --repo pydantic/pydantic-ai --reviewed-by DouweM --updated ">=2025-12-18" --limit 200 --json ...`
  - `gh api /repos/pydantic/pydantic-ai/issues/{number}/comments`
  - `gh api /repos/pydantic/pydantic-ai/pulls/{number}/reviews`
  - `gh api /repos/pydantic/pydantic-ai/pulls/{number}/comments`
- I sampled across the feature and internals groups in `feature-map.md` and `internals-model.md`, with extra passes over provider/profile work, durable execution, capabilities, toolsets/MCP, output/deferred tools, UI adapters, and public API migrations.
- Some comments are explicitly marked as written by Claude and approved by Douwe. I treated those as lower-weight supporting evidence unless they were paired with Douwe's own inline review comments.

## Reading Frame

Direct evidence: Douwe evaluates Pydantic AI primarily as a public abstraction layer, not as a pile of adapters. Provider API compatibility, normalized message/tool/output semantics, and durable execution compatibility all matter because they determine whether users can trust the framework boundary.

Inference: his review style optimizes for future maintainability under heavy AI-assisted contribution volume. He repeatedly pushes changes toward stable names, single sources of truth, provider capability facts in profiles/providers, clear deprecations, integration tests, and docs that encode the real contract.

## Provider Compatibility Is Core Product Work

Direct evidence:

- On Temporal model compatibility, Douwe rejected a separate provider-to-profile mapping because profiles are model-name-specific and should come from `Provider.model_profile`; he said he could not accept a parallel mapping outside that source of truth. Source: [PR #4100 review comment](https://github.com/pydantic/pydantic-ai/pull/4100#discussion_r2733704228).
- In the same PR, he kept profile/model inference close to `infer_model`, asked for `parse_model_id` naming consistent with prior work, and later checked backward compatibility for custom providers whose `model_profile` remained an instance method. Sources: [PR #4100](https://github.com/pydantic/pydantic-ai/pull/4100#discussion_r2733692860), [PR #4100](https://github.com/pydantic/pydantic-ai/pull/4100#discussion_r2747366038), [PR #4100](https://github.com/pydantic/pydantic-ai/pull/4100#discussion_r2879931117).
- On Anthropic Bedrock/Vertex normalization, he wanted provider/client facts checked against the upstream docs instead of guessed, and corrected docs that implied `GoogleProvider(vertexai=True)` could serve Anthropic models. Sources: [PR #5395](https://github.com/pydantic/pydantic-ai/pull/5395#discussion_r3230537192), [PR #5395](https://github.com/pydantic/pydantic-ai/pull/5395#discussion_r3230549190).
- On Bedrock native JSON output and strict tools, he immediately asked whether all models share the same JSON Schema restrictions and whether Pydantic AI needs its own transformer, matching OpenAI/Google patterns. Source: [issue #4209](https://github.com/pydantic/pydantic-ai/issues/4209#issuecomment-3855751921).
- On Bedrock prompt cache TTL, he asked for consistency with Anthropic settings and for model-profile flags when support varies by model. Source: [PR #4129](https://github.com/pydantic/pydantic-ai/pull/4129#discussion_r2738260097), [PR #4129](https://github.com/pydantic/pydantic-ai/pull/4129#discussion_r2738261500).
- On Anthropic code execution, he kept normalized `tool_name` provider-agnostic and pushed provider-specific variants into `provider_metadata`; he also wanted supported/default version facts represented clearly, likely through profile flags. Source: [PR #4958](https://github.com/pydantic/pydantic-ai/pull/4958#discussion_r3170814519), [PR #4958](https://github.com/pydantic/pydantic-ai/pull/4958#discussion_r3170822629).
- On Azure OpenAI file/document support, he pushed the docs and profile naming to describe the exact unsupported surface, not a vague "files" claim if the real issue is documents. Source: [PR #4048](https://github.com/pydantic/pydantic-ai/pull/4048#discussion_r2770862258), [PR #4048](https://github.com/pydantic/pydantic-ai/pull/4048#discussion_r2886643225).
- On runtime model override without a provider prefix, he treated the observed behavior as understandable but deprecated, and emphasized the intended contract: pass a complete `Model` instance or complete `provider:model` ID. Source: [issue #5273](https://github.com/pydantic/pydantic-ai/issues/5273#issuecomment-4374567477).

Inference:

- Provider support is not "best effort" string plumbing. The preferred shape is: normalized core API, model/provider/profile facts at the proper layer, typed provider settings/metadata for provider-specific capability, and clear failure when behavior cannot be normalized.
- Douwe tolerates provider-specific branches when they preserve real provider functionality, but resists spreading those facts into graph/tool/output code.

## Durable Execution Is A First-Class Compatibility Target

Direct evidence:

- On Restate docs, Douwe suggested saying Pydantic AI officially supports four durable execution solutions, co-maintained with vendor teams, and was careful about linking users to vendor docs to avoid stale local duplication. Source: [PR #5041](https://github.com/pydantic/pydantic-ai/pull/5041#discussion_r3075559721), [PR #5041](https://github.com/pydantic/pydantic-ai/pull/5041#discussion_r3075579975).
- On Temporal model/profile support, he reviewed the generic model/profile inference path rather than treating Temporal as a separate workaround. Source: [PR #4100](https://github.com/pydantic/pydantic-ai/pull/4100#discussion_r2733704228).
- On DBOS parallel tool calls, he did not want a DBOS-only escape hatch documented for ordinary users unless there was a general use case; instead he steered toward a general execution-mode API and then a DBOS-specific flag using the same ontology. Source: [PR #4077](https://github.com/pydantic/pydantic-ai/pull/4077#discussion_r2729292227), [PR #4077](https://github.com/pydantic/pydantic-ai/pull/4077#discussion_r2729304880), [PR #4077](https://github.com/pydantic/pydantic-ai/pull/4077#discussion_r2729311256).
- On Temporal binary tool returns, he pushed the solution into Pydantic serialization/types instead of manual base64 mapping and required a real Temporal workflow test, not only a unit test based on assumptions about Temporal internals. Source: [PR #4043](https://github.com/pydantic/pydantic-ai/pull/4043#discussion_r2713627523), [PR #4043](https://github.com/pydantic/pydantic-ai/pull/4043#discussion_r2722048787).
- On FastMCP retry inheritance, he flagged `RunContext` access as tricky with Temporal and suggested a generic `ToolsetTool.max_retries = None` default resolved by `ToolManager`, which would work with durable execution without bespoke durable code. Source: [PR #4745](https://github.com/pydantic/pydantic-ai/pull/4745#discussion_r3075907508).
- On durable MCP wrappers, he requested carrying full `ctx` rather than only `max_retries`, indicating wrapper boundaries need the full run context semantics. Source: [PR #5243](https://github.com/pydantic/pydantic-ai/pull/5243#discussion_r3164830655).
- On Temporal prepare callbacks, he explained that not every user callable is automatically wrapped as an activity because of overhead; only likely IO/non-deterministic callables are, and users can wrap others explicitly. Source: [issue #5138](https://github.com/pydantic/pydantic-ai/issues/5138#issuecomment-4288851372).

Inference:

- Durable engines are treated as compatibility tests for core design. If a feature requires ad hoc state, hidden call order, non-serializable context, or extra provider/profile lookup outside the central path, Douwe tends to redirect the design into a generic extension point.
- Durable support should be checked for tool execution ordering, retry semantics, message serialization, model/profile inference, MCP/toolset lifecycle, and context propagation.

## API Design: Names Encode Ontology

Direct evidence:

- On Google provider splitting, Douwe supported renames and privatization where names clarified the public surface, pushed old prefixes out, and argued root-level instructions were better than repeating the same comment across every provider. Source: [PR #5336](https://github.com/pydantic/pydantic-ai/pull/5336#discussion_r3238102677), [PR #5336](https://github.com/pydantic/pydantic-ai/pull/5336#discussion_r3244216236), [PR #5336](https://github.com/pydantic/pydantic-ai/pull/5336#discussion_r3238083147).
- On built-in tools becoming native tools, he caught places where the rename leaked into graph/evals contexts incorrectly, showing the term is semantic, not a global search/replace. Source: [PR #5338](https://github.com/pydantic/pydantic-ai/pull/5338#discussion_r3229798104), [PR #5338](https://github.com/pydantic/pydantic-ai/pull/5338#discussion_r3229825046).
- On the new MCP toolset, he wanted public field types to be public, names like `retry` instead of legacy `model_retry`, and old public types deprecated deliberately. Source: [PR #5325](https://github.com/pydantic/pydantic-ai/pull/5325#discussion_r3211709473), [PR #5325](https://github.com/pydantic/pydantic-ai/pull/5325#discussion_r3249971339), [PR #5325](https://github.com/pydantic/pydantic-ai/pull/5325#discussion_r3250026401).
- On event stream capability naming, he preferred `ProcessEventStream` because "handle" underplayed possible side effects, and mentioned a possible `HistoryProcessor` to `ProcessHistory` rename for v2. Source: [PR #5141](https://github.com/pydantic/pydantic-ai/pull/5141#discussion_r3139436148).
- On output/tool events, he challenged `result` as a field name because it implies final agent output, preferring `part` to match event semantics. Source: [PR #5320](https://github.com/pydantic/pydantic-ai/pull/5320#discussion_r3211688548).
- On retries, he wanted `AgentSpec` fields to match `Agent` kwargs and old names deprecated while still deserializing. Source: [PR #5075](https://github.com/pydantic/pydantic-ai/pull/5075#discussion_r3203036453).
- V2 execution PRs strengthen the same pattern: `builtin_*` terminology is removed in favor of `native_*`, `vendor_*` terminology is removed in favor of `provider_*`, output-tool events are split from function-tool events, and MCP client APIs consolidate around `MCPToolset`. Sources: [PR #5396](https://github.com/pydantic/pydantic-ai/pull/5396), [PR #5476](https://github.com/pydantic/pydantic-ai/pull/5476), [PR #5332](https://github.com/pydantic/pydantic-ai/pull/5332), [PR #5337](https://github.com/pydantic/pydantic-ai/pull/5337).

Inference:

- Names are not bikeshedding in this repo. Douwe uses naming to enforce abstraction boundaries: native vs local/provider-executed, function tool vs output tool, provider vs model profile, instructions vs system prompts, tool result part vs final output.
- V2 is the moment where deprecated vocabulary can be removed, but serialized protocol compatibility still needs deliberate bridges where old names may live in persisted histories or specs.

## Backward Compatibility And Migration Shape Matter

Direct evidence:

- On model IDs, he explained why provider-less strings are deprecated and heading to removal, while acknowledging the user's expectation was reasonable. Source: [issue #5273](https://github.com/pydantic/pydantic-ai/issues/5273#issuecomment-4374567477).
- On `AgentSpec` retry rename, he wanted old fields deprecated but still working and future deserialization to support the new shape. Source: [PR #5075](https://github.com/pydantic/pydantic-ai/pull/5075#discussion_r3203036453).
- On adding an abstract method, he flagged that requiring existing subclasses to implement it would be breaking and preferred a default method that raises at use time. Source: [PR #5141](https://github.com/pydantic/pydantic-ai/pull/5141#discussion_r3139458487).
- On graph beta API changes, he explicitly asked whether consumers now need to think about a new type, even for a beta surface. Source: [PR #5023](https://github.com/pydantic/pydantic-ai/pull/5023#discussion_r3054375043).
- On moving graph beta exports, he wanted docs to clarify which imports can drop `.beta` now and which must wait until v2. Source: [PR #5306](https://github.com/pydantic/pydantic-ai/pull/5306#discussion_r3221391040).
- On global warning ignores, he rejected broad ignores because the team needs to notice accidental uses of deprecated formats. Source: [PR #5325](https://github.com/pydantic/pydantic-ai/pull/5325#discussion_r3211766544), [PR #5325](https://github.com/pydantic/pydantic-ai/pull/5325#discussion_r3250117977).
- V2 cleanup PRs target deprecated public shims such as bare model fallback, old provider aliases, old OpenAI/Google/Grok names, `Agent(mcp_servers=)`, `Agent(prepare_tools=)`, `Agent(event_stream_handler=)`, `builtin_*`, `vendor_*`, and old stream/result helpers. Sources: [PR #5464](https://github.com/pydantic/pydantic-ai/pull/5464), [PR #5466](https://github.com/pydantic/pydantic-ai/pull/5466), [PR #5468](https://github.com/pydantic/pydantic-ai/pull/5468), [PR #5475](https://github.com/pydantic/pydantic-ai/pull/5475), [PR #5476](https://github.com/pydantic/pydantic-ai/pull/5476).
- The important distinction for v2 is public API removal versus persisted data compatibility. Deprecated constructors/imports/kwargs can disappear, while serialized message-history aliases may still need to deserialize for replay and frontend/durable roundtrips. Sources: [PR #5476](https://github.com/pydantic/pydantic-ai/pull/5476), [PR #5479](https://github.com/pydantic/pydantic-ai/pull/5479), [issue #5174](https://github.com/pydantic/pydantic-ai/issues/5174#issuecomment-4308761315).

Inference:

- Compatibility is active design, not only "does CI pass." Douwe looks for deprecation warnings, deserialization bridges, old public import behavior, subclass compatibility, and whether docs teach the migration path.
- For v2, the review question shifts from "is this backward compatible?" to "is the break intentional, documented, and not breaking persisted state that users cannot easily regenerate?"

## Scope Control: Features Need General Primitives

Direct evidence:

- In the AI-assisted contribution meta issue, Douwe said non-trivial work should get maintainer sign-off on approach before implementation, except simple bugfixes with regression tests. Source: [issue #4052](https://github.com/pydantic/pydantic-ai/issues/4052#issuecomment-3791785697).
- In the same discussion, he framed the library as needing abstractions over expanding provider APIs and production-grade building blocks, while acknowledging that "basics" move quickly in agent development. Source: [issue #4052](https://github.com/pydantic/pydantic-ai/issues/4052#issuecomment-3791746309).
- On the functional-agent PR, he rejected a patch-on-top implementation and asked for a native design on `(Abstract)Agent` plus an appropriate `pydantic-graph` change. Source: [PR #5123](https://github.com/pydantic/pydantic-ai/pull/5123#discussion_r3096221294).
- On code mode, he repeatedly pushed toward existing toolset abstractions (`RenamedToolset`, `PrefixedToolset`, tool search) and asked for concrete use cases before adding knobs. Source: [PR #4153](https://github.com/pydantic/pydantic-ai/pull/4153#discussion_r2761532414), [PR #4153](https://github.com/pydantic/pydantic-ai/pull/4153#discussion_r2776511696), [PR #4153](https://github.com/pydantic/pydantic-ai/pull/4153#discussion_r2776506441).
- On deferred loading/tool search, he resisted infecting `Agent` with feature-specific state and pushed the feature into toolset/capability composition. Source: [PR #4090](https://github.com/pydantic/pydantic-ai/pull/4090#discussion_r2734130992), [PR #4090](https://github.com/pydantic/pydantic-ai/pull/4090#discussion_r2737629920).
- On extension philosophy, he pointed users to wrapper agents, wrapper models, custom `AbstractToolset`s, and planned hooks/middleware. Source: [issue #4159](https://github.com/pydantic/pydantic-ai/issues/4159#issuecomment-3836404885), [issue #4078](https://github.com/pydantic/pydantic-ai/issues/4078#issuecomment-3801568994).

Inference:

- Douwe is open to ambitious features when they become general primitives. He resists one-off API flags, patch layers, narrow docs, or feature branches that bypass the established extension ontology.

## Internals: Centralize Logic, Respect Layers

Direct evidence:

- On deferred tool handling, he identified duplicated result conversion between `ToolManager` and `UserPromptNode`, pushed validation into useful helpers, and questioned tuple-heavy return shapes. Source: [PR #5142](https://github.com/pydantic/pydantic-ai/pull/5142#discussion_r3139770687), [PR #5142](https://github.com/pydantic/pydantic-ai/pull/5142#discussion_r3139797402), [PR #5142](https://github.com/pydantic/pydantic-ai/pull/5142#discussion_r3139831172).
- On tool search wrapping, he wanted generic helpers in the right modules and object accessors rather than ugly unwrapping in tests. Source: [PR #5047](https://github.com/pydantic/pydantic-ai/pull/5047#discussion_r3066292044), [PR #5047](https://github.com/pydantic/pydantic-ai/pull/5047#discussion_r3066307342).
- On OpenAI compaction, he wanted repeated conversion from provider items to normalized parts centralized. Source: [PR #5108](https://github.com/pydantic/pydantic-ai/pull/5108#discussion_r3096841109).
- On validate-before-deferring, he called out duplication with `handle_call` and wanted a clearer name matching behavior. Source: [PR #4049](https://github.com/pydantic/pydantic-ai/pull/4049#discussion_r2713833080).
- On capability deferred loading, he repeatedly flagged duplicated instruction normalization, tool discovery, and tool-search logic, and wanted message history as the source of truth for discovered/available tools. Source: [PR #5230](https://github.com/pydantic/pydantic-ai/pull/5230#discussion_r3251583214), [PR #5230](https://github.com/pydantic/pydantic-ai/pull/5230#discussion_r3251567097), [PR #5230](https://github.com/pydantic/pydantic-ai/pull/5230#discussion_r3251644352).
- On toolset instructions, he preferred `get_instructions` consistency with `get_tools`, forwarding through combined/wrapped toolsets, and eventually moving provider-specific placement decisions into the model request path rather than dumping everything into graph-level instructions. Source: [PR #4123](https://github.com/pydantic/pydantic-ai/pull/4123#discussion_r2737964746), [PR #4123](https://github.com/pydantic/pydantic-ai/pull/4123#discussion_r2875234799), [PR #4123](https://github.com/pydantic/pydantic-ai/pull/4123#discussion_r2897732014).
- V2 execution work centralizes tool-call handling around `process_tool_calls`: output, function, external, unapproved, and unknown calls are processed in one batch model with sequential barriers and `end_strategy` deciding final-output behavior. Source: [PR #5339](https://github.com/pydantic/pydantic-ai/pull/5339).
- V2 capability cleanup moves cross-cutting knobs into capabilities, especially instrumentation, event-stream processing, and tool preparation. Sources: [PR #5434](https://github.com/pydantic/pydantic-ai/pull/5434), [PR #5475](https://github.com/pydantic/pydantic-ai/pull/5475).

Inference:

- The internal architecture should keep graph orchestration, model/provider wire mapping, profiles, toolsets, capabilities, and messages distinct. Duplication is tolerated briefly during migration, but Douwe quickly asks for a single helper or a better home once a pattern repeats.
- In v2 reviews, expect extra scrutiny when a change adds another `Agent` kwarg, provider branch in graph code, duplicated MCP wrapper, or tool/output special case outside the shared processing path.

## Docs And Tests Are Part Of The Contract

Direct evidence:

- On Temporal binary serialization, he required a real Temporal workflow test because relying on knowledge of two external packages was not enough. Source: [PR #4043](https://github.com/pydantic/pydantic-ai/pull/4043#discussion_r2722048787).
- On output tool events, he asked for a full history/event-stream snapshot and wanted docs examples exercised instead of skipped. Source: [PR #5320](https://github.com/pydantic/pydantic-ai/pull/5320#discussion_r3211660800), [PR #5320](https://github.com/pydantic/pydantic-ai/pull/5320#discussion_r3211657020).
- On UI system prompt handling, he asked for snapshots that would reveal missing system prompt behavior, and wanted docs to explain system prompts versus instructions early. Source: [PR #4087](https://github.com/pydantic/pydantic-ai/pull/4087#discussion_r2729746746), [PR #4087](https://github.com/pydantic/pydantic-ai/pull/4087#discussion_r3083326284).
- On Restate docs, he valued examples but worried about local docs going stale as vendor SDK behavior changes. Source: [PR #5041](https://github.com/pydantic/pydantic-ai/pull/5041#discussion_r3075579975).
- On retry behavior, he asked to document tool retries because several PRs had touched the area recently. Source: [PR #5075](https://github.com/pydantic/pydantic-ai/pull/5075#discussion_r3083353737).
- On MCP instructions, he asked an MCP maintainer whether server instructions should be included by default, then favored not doing so without guidance. Source: [PR #4123](https://github.com/pydantic/pydantic-ai/pull/4123#issuecomment-3813096321), [PR #4123](https://github.com/pydantic/pydantic-ai/pull/4123#issuecomment-3813435188).

Inference:

- Tests should assert behavior at the surface users and integrations rely on: snapshots of messages/events, real provider/durable integration where external behavior matters, and docs examples that run. Docs should explain the concept boundary, not just the happy path.

## User Ergonomics And Safety

Direct evidence:

- On direct tool-output-as-agent-output, he rejected the request because it would break the `Agent.output_type` guarantee; output functions are the typed path for that use case. Source: [issue #5054](https://github.com/pydantic/pydantic-ai/issues/5054#issuecomment-4240231295).
- On Vercel roundtrips, he wanted every aspect of message history to survive frontend roundtrip, including `ModelResponse` fields like timestamp and provider details. Source: [issue #5174](https://github.com/pydantic/pydantic-ai/issues/5174#issuecomment-4308761315).
- On deferred tool handlers, he wanted the handler method documented as the primary path while preserving stop-the-world `DeferredToolRequests` for UI adapters. Source: [PR #5142](https://github.com/pydantic/pydantic-ai/pull/5142#discussion_r3139841425), [PR #5142](https://github.com/pydantic/pydantic-ai/pull/5142#discussion_r3140293112).
- On UI-originated system prompts, he treated frontend system prompts as a security-sensitive surface and favored rejecting them by default with opt-in behavior. Source: [PR #4087](https://github.com/pydantic/pydantic-ai/pull/4087#discussion_r2813772858).
- On `AgentSpec` loading MCP clients, he flagged possible RCE/SSRF risks from user-provided YAML and said the system will need safe/unsafe capability config concepts. Source: [PR #5325](https://github.com/pydantic/pydantic-ai/pull/5325#discussion_r3250093567).
- On `prepare_tools` returning `None`, he called it a footgun, supported a warning now, and expected dropping support in v2. Source: [issue #5177](https://github.com/pydantic/pydantic-ai/issues/5177#issuecomment-4308781467).
- On v2 stream events, deterministic cleanup matters: `run_stream_events` as an async context manager makes background-run cleanup explicit when consumers exit early. Source: [PR #5440](https://github.com/pydantic/pydantic-ai/pull/5440).
- On interrupted runs, preserving partial message state protects users from losing useful provider output or completed tool returns when streaming/tool execution is cancelled. Source: [PR #5364](https://github.com/pydantic/pydantic-ai/pull/5364).

Inference:

- Ergonomics are not just convenience. Douwe weighs whether an API preserves type guarantees, roundtrips losslessly, makes the safe path obvious, and avoids surprising silent behavior.
- V2 ergonomics should be judged against production failure modes: early stream exit, cancellation, durable replay, UI roundtrips, and provider-specific payload preservation.

## Review Tells

Useful signals to watch for in future reviews:

- "Where does this fact belong?" Provider facts should usually move to providers/profiles; graph logic should stay normalized.
- "Is this the same abstraction under another name?" He often detects duplicate types or helpers across tool/output/deferred paths.
- "Does this work with Temporal/DBOS/Prefect/Restate?" Durable engines expose hidden assumptions about context, serialization, determinism, and activity boundaries.
- "Can a user upgrade?" Expect deprecation bridges, warnings, old import/type behavior, and explicit docs.
- "Can the docs/test prove it?" Snapshot whole messages/events when behavior is about protocol shape; use integration tests for external runtime assumptions.
- "Is this a feature of Pydantic AI or a patch on top?" Native integration should use existing primitives or evolve them.
- "Is this v2 break public-surface cleanup or persisted-protocol breakage?" Removing deprecated APIs is different from losing the ability to replay old histories/specs.
- "Is this cross-cutting behavior a capability now?" V2 pushes instrumentation, event processing, prepare-tools behavior, and likely durable runtime integration toward capabilities rather than ad hoc agent kwargs.

## Open Questions To Ask David/Douwe

- After v2's `MCPToolset` and capability consolidation, what minimum capability/toolset contract should any new durable runtime satisfy before it becomes "officially supported"?
- For provider API features that only some provider SDK clients support, when should Pydantic AI expose a generic capability with provider-local fallback versus only a provider-native setting/tool?
- How aggressive should v2 be about removing deprecated names versus keeping read-only/deserialization aliases for ecosystem compatibility?
- Should provider-specific unsupported capability checks always be eager profile errors, or are provider API errors acceptable for fast-moving/new-model support?
- What is the intended long-term home for safe/unsafe `AgentSpec` capability loading and SSRF/RCE guardrails?
- For docs maintained partly by vendor teams, what is the threshold for keeping runnable examples in Pydantic AI docs versus linking out to vendor-owned docs?
