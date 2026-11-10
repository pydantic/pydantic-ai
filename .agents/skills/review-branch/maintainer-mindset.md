# Maintainer Review Principles

Purpose: state how maintainer review evaluates Pydantic AI issues, PRs, feature requests, and internals work, with the public review comments that show each principle.

## Methodology

- Repository: `pydantic/pydantic-ai`. Every source below is a public issue comment, PR review, or PR comment.
- Windows: 2025-12-18 through 2026-05-18, and a fresh sample from 2026-05-20 through 2026-09-29 (nine merged PR discussions, selected after inventorying recent merged PRs).
- Commands/API shape:
  - `gh search issues --repo pydantic/pydantic-ai --commenter <maintainer> --updated ">=<start>" --limit 200 --json ...`
  - `gh search prs --repo pydantic/pydantic-ai --reviewed-by <maintainer> --updated ">=<start>" --limit 200 --json ...`
  - `gh api /repos/pydantic/pydantic-ai/issues/{number}/comments`
  - `gh api /repos/pydantic/pydantic-ai/pulls/{number}/reviews`
  - `gh api /repos/pydantic/pydantic-ai/pulls/{number}/comments`
- The samples cover Pydantic AI's feature areas and the internals groups in `internals-model.md`, with extra passes over provider/profile work, durable execution, capabilities, toolsets/MCP, output/deferred tools, UI adapters, and public API migrations.
- Comments marked as AI-drafted count only as supporting evidence next to inline review comments.
- The first window ends before V2.0 shipped (2026-06-23) and before the harness moved into this repo (2026-09-26).
- The samples are qualitative calibration, not a census. They include comments on maintainers' own PRs and on other contributors' PRs.
- The first sample contains no harness review evidence. The fresh sample adds limited capability and durable-execution discussion, not comprehensive harness review norms. Current harness guidance remains authoritative: read `src/pydantic_ai_harness/AGENTS.md` and `src/pydantic_ai_harness/agent_docs/`.
- The repo's own CE outranks this file. Where root `AGENTS.md`, `agent_docs/*.md`, or a nested `AGENTS.md` states a rule, apply the rule; use the principles here to calibrate judgment where those files are silent.

## Current Review Evidence (2026-05-20 to 2026-09-29)

Selected examples from the nine-PR sample: observations from a bounded sample, not project policy. Current repository instructions take precedence.

- **Cross-surface lifecycle behavior:** In the workspace discussion, review asked that the lifecycle fix use the shared path used by ordinary and realtime runs. The durability work was later adapted to the declarative capability-operation base. Sources: [PR #6492](https://github.com/pydantic/pydantic-ai/pull/6492#issuecomment-5208200349), [PR #6492](https://github.com/pydantic/pydantic-ai/pull/6492#issuecomment-5458405230).
- **Limit core surface for engine-specific behavior:** In the Temporal stream discussion, a private core hook that served a Temporal-only need was removed in favor of ordered capability composition. The same discussion tied the API to the UI adapter's stream contract and end-to-end tests. Sources: [PR #6639](https://github.com/pydantic/pydantic-ai/pull/6639#issuecomment-5736874305), [PR #6639](https://github.com/pydantic/pydantic-ai/pull/6639#issuecomment-5499256665).
- **Verify claims at the behavior boundary:** For the Google Cloud Storage image path, review checked the request path, corrected an impossible docs example, added regression coverage, and requested recorded request/response evidence for live API behavior. Sources: [PR #8225](https://github.com/pydantic/pydantic-ai/pull/8225#discussion_r4087166515), [PR #8225](https://github.com/pydantic/pydantic-ai/pull/8225#issuecomment-5734962018).
- **Pin public and persisted behavior:** Review preserved `Annotated` schema metadata with tests, and rejected reconstructing a durable model instance from an identifier when doing so would silently lose its configuration. Sources: [PR #8608](https://github.com/pydantic/pydantic-ai/pull/8608#discussion_r4077020883), [PR #6898](https://github.com/pydantic/pydantic-ai/pull/6898#discussion_r3686582700).
- **Keep scope evidence-based:** Review left a verified but pre-existing cache-key recursion issue out of a focused PR. In the streaming error discussion, it checked the existing mapper and used a test that failed when the proposed change was reverted. Sources: [PR #6944](https://github.com/pydantic/pydantic-ai/pull/6944#discussion_r3686148458), [PR #9313](https://github.com/pydantic/pydantic-ai/pull/9313#discussion_r4137719311).

## Reading Frame

Principle: review treats Pydantic AI primarily as a public abstraction layer, not as a pile of adapters. Provider API compatibility, normalized message/tool/output semantics, and durable execution compatibility all matter because they determine whether users can trust the framework boundary.

Principle: review optimizes for future maintainability under heavy AI-assisted contribution volume. It repeatedly pushes changes toward stable names, single sources of truth, provider capability facts in profiles/providers, clear deprecations, integration tests, and docs that encode the real contract.

## Provider Compatibility Is Core Product Work

Principles:

- Provider support is not "best effort" string plumbing. The preferred shape is: normalized core API, model/provider/profile facts at the proper layer, typed provider settings/metadata for provider-specific capability, and clear failure when behavior cannot be normalized.
- Provider-specific branches are acceptable when they preserve real provider functionality. Those facts do not spread into graph/tool/output code.

Evidence:

- On Temporal model compatibility, review rejected a separate provider-to-profile mapping because profiles are model-name-specific and should come from `Provider.model_profile`; a parallel mapping outside that source of truth was not acceptable. Source: [PR #4100 review comment](https://github.com/pydantic/pydantic-ai/pull/4100#discussion_r2733704228).
- In the same PR, review kept profile/model inference close to `infer_model`, asked for `parse_model_id` naming consistent with prior work, and later checked backward compatibility for custom providers whose `model_profile` remained an instance method. Sources: [PR #4100](https://github.com/pydantic/pydantic-ai/pull/4100#discussion_r2733692860), [PR #4100](https://github.com/pydantic/pydantic-ai/pull/4100#discussion_r2747366038), [PR #4100](https://github.com/pydantic/pydantic-ai/pull/4100#discussion_r2879931117).
- On Anthropic Bedrock/Vertex normalization, review wanted provider/client facts checked against the upstream docs instead of guessed, and corrected docs that implied `GoogleProvider(vertexai=True)` could serve Anthropic models. Sources: [PR #5395](https://github.com/pydantic/pydantic-ai/pull/5395#discussion_r3230537192), [PR #5395](https://github.com/pydantic/pydantic-ai/pull/5395#discussion_r3230549190).
- On Bedrock native JSON output and strict tools, review immediately asked whether all models share the same JSON Schema restrictions and whether Pydantic AI needs its own transformer, matching OpenAI/Google patterns. Source: [issue #4209](https://github.com/pydantic/pydantic-ai/issues/4209#issuecomment-3855751921).
- On Bedrock prompt cache TTL, review asked for consistency with Anthropic settings and for model-profile flags when support varies by model. Source: [PR #4129](https://github.com/pydantic/pydantic-ai/pull/4129#discussion_r2738260097), [PR #4129](https://github.com/pydantic/pydantic-ai/pull/4129#discussion_r2738261500).
- On Anthropic code execution, review kept normalized `tool_name` provider-agnostic and pushed provider-specific variants into `provider_metadata`; it also wanted supported/default version facts represented clearly, likely through profile flags. Source: [PR #4958](https://github.com/pydantic/pydantic-ai/pull/4958#discussion_r3170814519), [PR #4958](https://github.com/pydantic/pydantic-ai/pull/4958#discussion_r3170822629).
- On Azure OpenAI file/document support, review pushed the docs and profile naming to describe the exact unsupported surface, not a vague "files" claim if the real issue is documents. Source: [PR #4048](https://github.com/pydantic/pydantic-ai/pull/4048#discussion_r2770862258), [PR #4048](https://github.com/pydantic/pydantic-ai/pull/4048#discussion_r2886643225).
- On runtime model override without a provider prefix, review treated the observed behavior as understandable but deprecated, and emphasized the intended contract: pass a complete `Model` instance or complete `provider:model` ID. Source: [issue #5273](https://github.com/pydantic/pydantic-ai/issues/5273#issuecomment-4374567477).

## Durable Execution Is A First-Class Compatibility Target

Principles:

- Durable engines are compatibility tests for core design. When a feature requires ad hoc state, hidden call order, non-serializable context, or extra provider/profile lookup outside the central path, review redirects the design into a generic extension point.
- Check durable support for tool execution ordering, retry semantics, message serialization, model/profile inference, MCP/`MCPToolset` lifecycle, stable toolset `id`s, and context propagation.
- The durability capabilities (`TemporalDurability` / `DBOSDurability` / `PrefectDurability`, [PR #4977](https://github.com/pydantic/pydantic-ai/pull/4977)) are the supported path. `TemporalAgent` / `DBOSAgent` / `PrefectAgent` are `@deprecated`: they keep working until v3, but new features can be capability-only (workspaces are refused by the wrappers). Design new durable behavior against the capability path. A regression in a wrapper's existing behavior is still a regression until v3.

Evidence:

- On Restate docs, review suggested saying Pydantic AI officially supports four durable execution solutions, co-maintained with vendor teams, and was careful about linking users to vendor docs to avoid stale local duplication. Source: [PR #5041](https://github.com/pydantic/pydantic-ai/pull/5041#discussion_r3075559721), [PR #5041](https://github.com/pydantic/pydantic-ai/pull/5041#discussion_r3075579975).
- On Temporal model/profile support, review examined the generic model/profile inference path rather than treating Temporal as a separate workaround. Source: [PR #4100](https://github.com/pydantic/pydantic-ai/pull/4100#discussion_r2733704228).
- On DBOS parallel tool calls, review did not want a DBOS-only escape hatch documented for ordinary users unless there was a general use case; instead it steered toward a general execution-mode API and then a DBOS-specific flag using the same ontology. Source: [PR #4077](https://github.com/pydantic/pydantic-ai/pull/4077#discussion_r2729292227), [PR #4077](https://github.com/pydantic/pydantic-ai/pull/4077#discussion_r2729304880), [PR #4077](https://github.com/pydantic/pydantic-ai/pull/4077#discussion_r2729311256).
- On Temporal binary tool returns, review pushed the solution into Pydantic serialization/types instead of manual base64 mapping and required a real Temporal workflow test, not only a unit test based on assumptions about Temporal internals. Source: [PR #4043](https://github.com/pydantic/pydantic-ai/pull/4043#discussion_r2713627523), [PR #4043](https://github.com/pydantic/pydantic-ai/pull/4043#discussion_r2722048787).
- On FastMCP retry inheritance, review flagged `RunContext` access as tricky with Temporal and suggested a generic `ToolsetTool.max_retries = None` default resolved by `ToolManager`, which would work with durable execution without bespoke durable code. Source: [PR #4745](https://github.com/pydantic/pydantic-ai/pull/4745#discussion_r3075907508).
- On durable MCP wrappers, review requested carrying full `ctx` rather than only `max_retries`, indicating wrapper boundaries need the full run context semantics. Source: [PR #5243](https://github.com/pydantic/pydantic-ai/pull/5243#discussion_r3164830655).
- On Temporal prepare callbacks, a maintainer explained that not every user callable is automatically wrapped as an activity because of overhead; only likely IO/non-deterministic callables are, and users can wrap others explicitly. Source: [issue #5138](https://github.com/pydantic/pydantic-ai/issues/5138#issuecomment-4288851372).

## API Design: Names Encode Ontology

Principles:

- Names are not bikeshedding in this repo. Naming enforces abstraction boundaries: native vs local/provider-executed, function tool vs output tool, provider vs model profile, instructions vs system prompts, tool result part vs final output.
- Persisted names outlive Python names. Native parts still serialize as `builtin-tool-call` / `builtin-tool-return` so stored histories keep loading, and durable operation names are pinned compatibility data. A rename that touches a persisted tag or operation name needs a deliberate bridge, not a search-and-replace.

Evidence:

- On Google provider splitting, review supported renames and privatization where names clarified the public surface, pushed old prefixes out, and argued root-level instructions were better than repeating the same comment across every provider. Source: [PR #5336](https://github.com/pydantic/pydantic-ai/pull/5336#discussion_r3238102677), [PR #5336](https://github.com/pydantic/pydantic-ai/pull/5336#discussion_r3244216236), [PR #5336](https://github.com/pydantic/pydantic-ai/pull/5336#discussion_r3238083147).
- On built-in tools becoming native tools, review caught places where the rename leaked into graph/evals contexts incorrectly, showing the term is semantic, not a global search/replace. Source: [PR #5338](https://github.com/pydantic/pydantic-ai/pull/5338#discussion_r3229798104), [PR #5338](https://github.com/pydantic/pydantic-ai/pull/5338#discussion_r3229825046).
- On the new MCP toolset, review wanted public field types to be public, names like `retry` instead of legacy `model_retry`, and old public types deprecated deliberately. Source: [PR #5325](https://github.com/pydantic/pydantic-ai/pull/5325#discussion_r3211709473), [PR #5325](https://github.com/pydantic/pydantic-ai/pull/5325#discussion_r3249971339), [PR #5325](https://github.com/pydantic/pydantic-ai/pull/5325#discussion_r3250026401).
- On event stream capability naming, review preferred `ProcessEventStream` because "handle" underplayed possible side effects, and mentioned a possible `HistoryProcessor` to `ProcessHistory` rename for v2. Source: [PR #5141](https://github.com/pydantic/pydantic-ai/pull/5141#discussion_r3139436148).
- On output/tool events, review challenged `result` as a field name because it implies final agent output, preferring `part` to match event semantics. Source: [PR #5320](https://github.com/pydantic/pydantic-ai/pull/5320#discussion_r3211688548).
- On retries, review wanted `AgentSpec` fields to match `Agent` kwargs and old names deprecated while still deserializing. Source: [PR #5075](https://github.com/pydantic/pydantic-ai/pull/5075#discussion_r3203036453).
- The V2 cut applied the same pattern: it dropped the `builtin_*` Python API for `native_*`, dropped the deprecated `vendor_*` aliases on message fields for `provider_*`, split output-tool events from function-tool events, and consolidated MCP clients on `MCPToolset`. Sources: [PR #5396](https://github.com/pydantic/pydantic-ai/pull/5396), [PR #5476](https://github.com/pydantic/pydantic-ai/pull/5476), [PR #5332](https://github.com/pydantic/pydantic-ai/pull/5332), [PR #5337](https://github.com/pydantic/pydantic-ai/pull/5337).

## Backward Compatibility And Migration Shape Matter

The rules are in the repo. `docs/version-policy.md` forbids intentional breaking changes in minor releases and keeps deprecated functionality until the next major (v3, no sooner than 3 months after V2.0). `agent_docs/api-design.md` → "Compatibility impacts" defines a compatibility impact and what a permitted one needs: the `compatibility impact` label, a release note, a `[!WARNING]` PR section naming affected code, the policy exception and the migration, and an exact API-check waiver. Root `AGENTS.md` → "Requirements of all contributions" adds that a new parameter whose default changes existing behavior is a break. Apply those rules. The evidence below calibrates how strictly review reads them.

Principles:

- Compatibility is active design, not only "does CI pass." Review looks for deprecation warnings, deserialization bridges, old public import behavior, subclass compatibility, and whether docs teach the migration path.
- The review question for a minor release is "does previously valid code, or previously stored data, need to change?" If yes, the change needs a deprecation bridge, or an exact `docs/version-policy.md` exception with the `agent_docs/api-design.md` paperwork. A removal without either waits for v3.

Evidence:

- On model IDs, a maintainer explained why provider-less strings are deprecated and heading to removal, while acknowledging the user's expectation was reasonable. Source: [issue #5273](https://github.com/pydantic/pydantic-ai/issues/5273#issuecomment-4374567477).
- On `AgentSpec` retry rename, review wanted old fields deprecated but still working and future deserialization to support the new shape. Source: [PR #5075](https://github.com/pydantic/pydantic-ai/pull/5075#discussion_r3203036453).
- On adding an abstract method, review flagged that requiring existing subclasses to implement it would be breaking and preferred a default method that raises at use time. Source: [PR #5141](https://github.com/pydantic/pydantic-ai/pull/5141#discussion_r3139458487).
- On graph beta API changes, review explicitly asked whether consumers now need to think about a new type, even for a beta surface. Source: [PR #5023](https://github.com/pydantic/pydantic-ai/pull/5023#discussion_r3054375043).
- On moving graph beta exports, review wanted docs to clarify which imports can drop `.beta` now and which must wait until v2. Source: [PR #5306](https://github.com/pydantic/pydantic-ai/pull/5306#discussion_r3221391040).
- On global warning ignores, review rejected broad ignores because the team needs to notice accidental uses of deprecated formats. Source: [PR #5325](https://github.com/pydantic/pydantic-ai/pull/5325#discussion_r3211766544), [PR #5325](https://github.com/pydantic/pydantic-ai/pull/5325#discussion_r3250117977).
- The V2 cut removed deprecated public shims (bare model fallback, old provider aliases, `Agent(prepare_tools=)`, `Agent(event_stream_handler=)`, `builtin_*`) while keeping serialized message-history aliases deserializable for replay and frontend/durable roundtrips. Public API removal and persisted-data compatibility are separate questions. Sources: [PR #5464](https://github.com/pydantic/pydantic-ai/pull/5464), [PR #5475](https://github.com/pydantic/pydantic-ai/pull/5475), [PR #5476](https://github.com/pydantic/pydantic-ai/pull/5476), [PR #5479](https://github.com/pydantic/pydantic-ai/pull/5479), [issue #5174](https://github.com/pydantic/pydantic-ai/issues/5174#issuecomment-4308761315).

## Scope Control: Features Need General Primitives

Principles:

- Ambitious features are welcome when they become general primitives. Review resists one-off API flags, patch layers, narrow docs, or feature branches that bypass the established extension ontology.
- The primitives preference shapes designs and new public surface. It is not a license to widen a bug fix. Root `AGENTS.md` → "Requirements of all contributions": a bug fix is the narrowest change that resolves the reproduced behavior; extend it to sibling providers, fields or models only after reproducing the same defect there; a reviewer's finding does not widen scope by itself.
- Batteries-included compositions belong in the harness, not core. Core changes only for a missing primitive (`src/pydantic_ai_harness/agent_docs/core-boundary.md`).

Evidence:

- In the AI-assisted contribution meta issue, a maintainer said non-trivial work should get maintainer sign-off on approach before implementation, except simple bugfixes with regression tests. Source: [issue #4052](https://github.com/pydantic/pydantic-ai/issues/4052#issuecomment-3791785697).
- In the same discussion, the library was framed as needing abstractions over expanding provider APIs and production-grade building blocks, while acknowledging that "basics" move quickly in agent development. Source: [issue #4052](https://github.com/pydantic/pydantic-ai/issues/4052#issuecomment-3791746309).
- On the functional-agent PR, review rejected a patch-on-top implementation and asked for a native design on `(Abstract)Agent` plus an appropriate `pydantic-graph` change. Source: [PR #5123](https://github.com/pydantic/pydantic-ai/pull/5123#discussion_r3096221294).
- On code mode, review repeatedly pushed toward existing toolset abstractions (`RenamedToolset`, `PrefixedToolset`, tool search) and asked for concrete use cases before adding knobs. Source: [PR #4153](https://github.com/pydantic/pydantic-ai/pull/4153#discussion_r2761532414), [PR #4153](https://github.com/pydantic/pydantic-ai/pull/4153#discussion_r2776511696), [PR #4153](https://github.com/pydantic/pydantic-ai/pull/4153#discussion_r2776506441).
- On deferred loading/tool search, review resisted infecting `Agent` with feature-specific state and pushed the feature into toolset/capability composition. Source: [PR #4090](https://github.com/pydantic/pydantic-ai/pull/4090#discussion_r2734130992), [PR #4090](https://github.com/pydantic/pydantic-ai/pull/4090#discussion_r2737629920).
- On extension philosophy, maintainers pointed users to wrapper agents, wrapper models, custom `AbstractToolset`s, and planned hooks/middleware. Source: [issue #4159](https://github.com/pydantic/pydantic-ai/issues/4159#issuecomment-3836404885), [issue #4078](https://github.com/pydantic/pydantic-ai/issues/4078#issuecomment-3801568994).

## Internals: Centralize Logic, Respect Layers

Principles:

- The internal architecture keeps graph orchestration, model/provider wire mapping, profiles, toolsets, capabilities, and messages distinct. Duplication is tolerated briefly during migration, but once a pattern repeats, review asks for a single helper or a better home.
- Expect extra scrutiny when a change adds another `Agent` kwarg, provider branch in graph code, duplicated MCP wrapper, or tool/output special case outside the shared processing path.

Evidence:

- On deferred tool handling, review identified duplicated result conversion between `ToolManager` and `UserPromptNode`, pushed validation into useful helpers, and questioned tuple-heavy return shapes. Source: [PR #5142](https://github.com/pydantic/pydantic-ai/pull/5142#discussion_r3139770687), [PR #5142](https://github.com/pydantic/pydantic-ai/pull/5142#discussion_r3139797402), [PR #5142](https://github.com/pydantic/pydantic-ai/pull/5142#discussion_r3139831172).
- On tool search wrapping, review wanted generic helpers in the right modules and object accessors rather than ugly unwrapping in tests. Source: [PR #5047](https://github.com/pydantic/pydantic-ai/pull/5047#discussion_r3066292044), [PR #5047](https://github.com/pydantic/pydantic-ai/pull/5047#discussion_r3066307342).
- On OpenAI compaction, review wanted repeated conversion from provider items to normalized parts centralized. Source: [PR #5108](https://github.com/pydantic/pydantic-ai/pull/5108#discussion_r3096841109).
- On validate-before-deferring, review called out duplication with `handle_call` and wanted a clearer name matching behavior. Source: [PR #4049](https://github.com/pydantic/pydantic-ai/pull/4049#discussion_r2713833080).
- On capability deferred loading, review repeatedly flagged duplicated instruction normalization, tool discovery, and tool-search logic, and wanted message history as the source of truth for discovered/available tools. Source: [PR #5230](https://github.com/pydantic/pydantic-ai/pull/5230#discussion_r3251583214), [PR #5230](https://github.com/pydantic/pydantic-ai/pull/5230#discussion_r3251567097), [PR #5230](https://github.com/pydantic/pydantic-ai/pull/5230#discussion_r3251644352).
- On toolset instructions, review preferred `get_instructions` consistency with `get_tools`, forwarding through combined/wrapped toolsets, and eventually moving provider-specific placement decisions into the model request path rather than dumping everything into graph-level instructions. Source: [PR #4123](https://github.com/pydantic/pydantic-ai/pull/4123#discussion_r2737964746), [PR #4123](https://github.com/pydantic/pydantic-ai/pull/4123#discussion_r2875234799), [PR #4123](https://github.com/pydantic/pydantic-ai/pull/4123#discussion_r2897732014).
- The V2 execution work centralized tool-call handling in `process_tool_calls` (`_tool_execution.py`): output, function, external, unapproved, and unknown calls are processed in one batch with sequential barriers and `end_strategy` deciding final-output behavior. Source: [PR #5339](https://github.com/pydantic/pydantic-ai/pull/5339).
- The V2 cut moved cross-cutting constructor knobs into capabilities: instrumentation, event-stream processing, and tool preparation. Sources: [PR #5434](https://github.com/pydantic/pydantic-ai/pull/5434), [PR #5475](https://github.com/pydantic/pydantic-ai/pull/5475).

## Docs And Tests Are Part Of The Contract

Principle:

- Tests assert behavior at the surface users and integrations rely on: snapshots of messages/events, real provider/durable integration where external behavior matters, and docs examples that run. Docs explain the concept boundary, not just the happy path.

Evidence:

- On Temporal binary serialization, review required a real Temporal workflow test because relying on knowledge of two external packages was not enough. Source: [PR #4043](https://github.com/pydantic/pydantic-ai/pull/4043#discussion_r2722048787).
- On output tool events, review asked for a full history/event-stream snapshot and wanted docs examples exercised instead of skipped. Source: [PR #5320](https://github.com/pydantic/pydantic-ai/pull/5320#discussion_r3211660800), [PR #5320](https://github.com/pydantic/pydantic-ai/pull/5320#discussion_r3211657020).
- On UI system prompt handling, review asked for snapshots that would reveal missing system prompt behavior, and wanted docs to explain system prompts versus instructions early. Source: [PR #4087](https://github.com/pydantic/pydantic-ai/pull/4087#discussion_r2729746746), [PR #4087](https://github.com/pydantic/pydantic-ai/pull/4087#discussion_r3083326284).
- On Restate docs, review valued examples but worried about local docs going stale as vendor SDK behavior changes. Source: [PR #5041](https://github.com/pydantic/pydantic-ai/pull/5041#discussion_r3075579975).
- On retry behavior, review asked to document tool retries because several PRs had touched the area recently. Source: [PR #5075](https://github.com/pydantic/pydantic-ai/pull/5075#discussion_r3083353737).
- On MCP instructions, review asked an MCP maintainer whether server instructions should be included by default, then favored not doing so without guidance. Source: [PR #4123](https://github.com/pydantic/pydantic-ai/pull/4123#issuecomment-3813096321), [PR #4123](https://github.com/pydantic/pydantic-ai/pull/4123#issuecomment-3813435188).

## User Ergonomics And Safety

Principles:

- Ergonomics are not just convenience. Review weighs whether an API preserves type guarantees, roundtrips losslessly, makes the safe path obvious, and avoids surprising silent behavior.
- Judge ergonomics against production failure modes: early stream exit, cancellation, durable replay, UI roundtrips, and provider-specific payload preservation.

Evidence:

- On direct tool-output-as-agent-output, a maintainer rejected the request because it would break the `Agent.output_type` guarantee; output functions are the typed path for that use case. Source: [issue #5054](https://github.com/pydantic/pydantic-ai/issues/5054#issuecomment-4240231295).
- On Vercel roundtrips, a maintainer wanted every aspect of message history to survive frontend roundtrip, including `ModelResponse` fields like timestamp and provider details. Source: [issue #5174](https://github.com/pydantic/pydantic-ai/issues/5174#issuecomment-4308761315).
- On deferred tool handlers, review wanted the handler method documented as the primary path while preserving stop-the-world `DeferredToolRequests` for UI adapters. Source: [PR #5142](https://github.com/pydantic/pydantic-ai/pull/5142#discussion_r3139841425), [PR #5142](https://github.com/pydantic/pydantic-ai/pull/5142#discussion_r3140293112).
- On UI-originated system prompts, review treated frontend system prompts as a security-sensitive surface and favored rejecting them by default with opt-in behavior. Source: [PR #4087](https://github.com/pydantic/pydantic-ai/pull/4087#discussion_r2813772858).
- On `AgentSpec` loading MCP clients, review flagged possible RCE/SSRF risks from user-provided YAML and said the system will need safe/unsafe capability config concepts. Source: [PR #5325](https://github.com/pydantic/pydantic-ai/pull/5325#discussion_r3250093567).
- On `prepare_tools` returning `None`, a maintainer called it a footgun, supported a warning, and expected dropping support in v2. `ToolsPrepareFunc` returns only `list[ToolDefinition]`. Source: [issue #5177](https://github.com/pydantic/pydantic-ai/issues/5177#issuecomment-4308781467).
- On stream events, deterministic cleanup matters: `run_stream_events` returns an async context manager so background-run cleanup is explicit when consumers exit early. Source: [PR #5440](https://github.com/pydantic/pydantic-ai/pull/5440).
- On interrupted runs, preserving partial message state protects users from losing useful provider output or completed tool returns when streaming/tool execution is cancelled. Source: [PR #5364](https://github.com/pydantic/pydantic-ai/pull/5364).

## Review Tells

Questions to ask in every review:

- "Where does this fact belong?" Provider facts usually move to providers/profiles; graph logic stays normalized.
- "Is this the same abstraction under another name?" Look for duplicate types or helpers across tool/output/deferred paths.
- "Does this work with Temporal/DBOS/Prefect/Restate?" Durable engines expose hidden assumptions about context, serialization, determinism, and activity boundaries.
- "Can the docs/test prove it?" Snapshot whole messages/events when behavior is about protocol shape; use integration tests for external runtime assumptions.
- "Is this a feature of Pydantic AI or a patch on top?" Native integration should use existing primitives or evolve them.
- "Can a user upgrade within v2?" If previously valid code or stored histories/specs need to change, expect a deprecation bridge: a warning, old import/type behavior kept, migration docs. Without a bridge, expect an exact `docs/version-policy.md` exception plus the `compatibility impact` label, `[!WARNING]` PR section and API-check waiver that `agent_docs/api-design.md` → "Compatibility impacts" requires. Losing the ability to replay old histories or specs is the most serious form.
- "Is this cross-cutting behavior a capability?" Instrumentation, event processing, prepare-tools behavior, workspaces, model selection and durable runtime integration are capabilities (`*Durability` via `BaseDurabilityCapability`), not agent kwargs. The wrapper agents are deprecated; new durable behavior targets the capability path.

## Open Questions

The contract a new durable engine must satisfy is answered by `durable_exec/AGENTS.md` → "Building a new durable execution engine". Removal of deprecated names is answered by `docs/version-policy.md`. These remain unsettled; when a change depends on one, raise it with a maintainer instead of assuming an answer:

- Should provider-specific unsupported capability checks always be eager profile errors, or are provider API errors acceptable for fast-moving/new-model support?
- What is the intended long-term home for safe/unsafe `AgentSpec` capability loading and SSRF/RCE guardrails?
- For docs maintained partly by vendor teams, what is the threshold for keeping runnable examples in Pydantic AI docs versus linking out to vendor-owned docs?
