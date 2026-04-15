# Native Provider Tool Search — Comprehensive Plan

## Context

Native tool search builds on the existing local `ToolSearchToolset` and the minimal internal `ToolSearch` capability (merged in PR #5047) to push search down to provider-side tool search (Anthropic `tool_search_tool_bm25` / `tool_search_tool_regex`, OpenAI `tool_search` with `execution: 'server' | 'client'`). Apr 14 call with Douwe decided to:

- Extend `ToolSearch` from a plain internal capability into a provider-adaptive feature with a pluggable `strategy` field
- Ship in releasable stages rather than one megabranch
- Keep tool search in pydantic-ai core for now (possibly move to harness in V3)
- Don't merge this plan file into main — it stays a PR artifact

Related: #4090 (base `ToolSearchToolset` + raw-SDK reference tests), #5047 (internal `ToolSearch` capability + `CapabilityOrdering`), #4167 (Anthropic native), #4566 (OpenAI native).

## Current State (as of 2026-04-15)

### What already exists in `upstream/main`

- **`capabilities/_tool_search.py`** — `ToolSearch(AbstractCapability)`. Plain capability (not a `BuiltinOrLocalTool` subclass). Marked internal via `get_serialization_name() → None`. `get_wrapper_toolset()` returns `ToolSearchToolset(wrapped=toolset)`. `get_ordering()` returns `CapabilityOrdering(position='outermost')`. Merged in PR #5047 (Apr 10).
- **`capabilities/_ordering.py`** — `CapabilityOrdering` + `sort_capabilities()` (topological sort). Supports `position`, `wraps`, `wrapped_by`, `requires`. Added in PR #5047. This is the ordering system Douwe mentioned in the call.
- **`agent/__init__.py`** — `_inject_auto_capabilities()` appends `ToolSearch` (via `_AUTO_INJECT_CAPABILITY_TYPES`) when not explicitly provided. Uses `has_capability_type()` for detection. **Q4 answered** — backward compat mechanism is already in place.
- **`toolsets/_tool_search.py`** — `ToolSearchToolset(WrapperToolset)`. Substring-based search via `_search_tools()`. Partitions tools by `defer_loading=True`, injects `search_tools` function for the model to discover deferred tools.
- **`tools.py`** — `ToolDefinition` has `defer_loading: bool` and `prefer_builtin: str | None` fields. `prefer_builtin` is the 1:1 routing used by `BuiltinOrLocalTool`.
- **`capabilities/builtin_or_local.py`** — `BuiltinOrLocalTool(AbstractCapability)` base with `WebSearch` / `WebFetch` / `ImageGeneration` subclasses. Hooks: `_default_builtin`, `_default_local`, `_builtin_unique_id`, `_requires_builtin`. Wraps local toolsets with `PreparedToolset.prepare_func` that injects `prefer_builtin=uid`.
- **`models/anthropic.py`** — `_map_server_tool_use_block` has a `NotImplementedError` stub: `elif item.name in ('tool_search_tool_regex', 'tool_search_tool_bm25'): raise`. Entry point for Stage 2.
- **`messages.py`** — `BuiltinToolCallPart` / `BuiltinToolReturnPart` with fields `tool_name`, `args`, `tool_call_id`, `id`, `provider_name`, `provider_details`.
- **`_agent_graph.py`** — `CallToolsNode._run_stream` distinguishes `ToolCallPart` (executes locally) from `BuiltinToolCallPart` (yields event, never executes locally).
- **Reference pattern**: `models/anthropic.py` `_map_server_tool_use_block` + `_map_web_search_tool_result_block`; `models/openai.py` `_map_web_search_tool_call` (returns `(BuiltinToolCallPart, BuiltinToolReturnPart)` tuple — call+result collapsed).
- **CodeMode (harness)** — at `pydantic_ai_harness/code_mode/_capability.py`, CodeMode imports `ToolSearch` from core and uses `CapabilityOrdering(position='outermost', wraps=[_ToolSearch])` to sit outside it. No separate `ToolSearch` in harness — Douwe's 'already a capability' call comment referred to PR #5047 in core.
- **`tests/test_native_tool_search_vcr.py`** — **raw-SDK reference tests**, not pydantic-ai integration tests. Added in PR #4090. Shows exact wire format for both providers:
  - **OpenAI**: `NamespaceToolParam(type='namespace', name='tools', tools=[...])` with `{'defer_loading': True}` per inner tool, plus sibling `ToolSearchToolParam(type='tool_search')`. Uses `gpt-5.4`.
  - **Anthropic**: flat `BetaToolParam` list with `defer_loading=True` on each deferred tool, plus a sibling `BetaToolSearchToolBm25_20251119Param(type='tool_search_tool_bm25_20251119', name='tool_search_tool_bm25')`. Uses `claude-sonnet-4-5`.
  - Note in the test: *'ToolFunction (namespace inner type) doesn't have defer_loading in its TypedDict, but the API accepts it.'* — the SDK type is incomplete; API reality is ahead.

### SDK situation

- `anthropic>=0.80.0` — already ships `ToolSearchToolBm25_20251119Param`, `ToolSearchToolRegex20251119Param`, `ToolSearchToolResultBlock`, `ToolSearchToolSearchResultBlock`, `ToolReferenceBlockParam`. **No bump needed.**
- `openai>=2.29.0` — ships `ToolSearchToolParam` (with `execution: 'server' | 'client'`), `ResponseToolSearchCall`, `ResponseToolSearchOutputItem`, `NamespaceToolParam`. **No bump needed.**

## Open Questions — resolve with Douwe before implementation

Stages 1 and 4 are blocked on these. Q1 must be answered before Stage 1. Q2 and Q5 must be answered before Stage 4.

**Answered during planning research** (not listed below): Q3 (parallel harness work — none, work is in core via #5047), Q4 (auto-injection backward compat — `_inject_auto_capabilities` already does this).

### Q1 — How should `ToolSearch` route deferred tool defs to native providers? (blocks Stage 1)

The existing `ToolSearch` capability is a plain `AbstractCapability` that just wraps with `ToolSearchToolset` (which partitions deferred tools and injects `search_tools` locally). For native tool search we need the *opposite* shape: keep deferred tool defs in the outgoing request so the provider can search over them, and mark them as deferred on the wire format so the model doesn't see them initially.

The 1:N relationship still makes `prefer_builtin` the wrong mechanism: `prefer_builtin` means *'remove this local tool when the builtin is available'*, but tool search needs *'keep this tool def in the request as the corpus for the builtin to search over, but render it deferred on the wire format.'*

**Options**:

- **Q1-A**: Add a new field `corpus_for_builtin: str | None` on `ToolDefinition`. Set by an updated `ToolSearch.get_wrapper_toolset()` via a `PreparedToolset.prepare_func` that mirrors how `BuiltinOrLocalTool._add_prefer_builtin` works. Model adapters read the flag at `prepare_request` time, decide native-vs-local, and render accordingly.
- **Q1-B**: Keep `ToolSearch` as a plain capability, pass strategy + resolved tool-def list through `ModelRequestParameters.capabilities` (adapters inspect the capability instance to see the strategy). Requires exposing capability instances through `ModelRequestParameters` if they aren't already.
- **Q1-C**: Refactor `ToolSearch` to subclass `BuiltinOrLocalTool`. Reuses existing `prefer_builtin` infrastructure but changes the semantics of that field (see the 1:1 vs 1:N mismatch above). Breaks the internal-only contract, but since it's internal today that's probably fine.
- **Q1-D**: Keep the partitioning on the capability side — `ToolSearch.get_wrapper_toolset()` checks the model profile (via a new accessor or a lazy hook) and returns either `ToolSearchToolset` (local) or a new `NativeToolSearchToolset` (corpus mode). Avoids adding a new `ToolDefinition` field. Downside: capability needs access to the model profile at toolset-build time, which may not be available.

**Reasoning to preserve** (1:1 vs 1:N): web_search et al. are 1:1 — one builtin replaces one local fallback. Tool search is 1:N — one `tool_search` builtin operates over N deferred function tools, which are the *input* corpus not alternatives. Reusing `prefer_builtin` would create hidden semantic branching per builtin kind.

**Needed from Douwe**: pick an option; if Q1-A, confirm the field name (`corpus_for_builtin` vs `defers_to_builtin` vs `searched_by_builtin`).

### Q2 — Custom-callable routing on OpenAI `execution='client'`: Route A vs Route B (blocks Stage 4)

When the user provides a custom callable for tool search and runs against OpenAI, the model emits `ResponseToolSearchCall(execution='client')` and we need to run a local function to produce results.

- **Route A** (parse-time conversion): `models/openai.py` converts `ResponseToolSearchCall(execution='client')` → `ToolCallPart(tool_name=<local_search_tool_name>, tool_call_id=<preserved>)`. Zero changes to `_agent_graph.py`. Loses the 'originated from a builtin' signal on history but `tool_call_id` round-trips.
- **Route B** (agent-graph hook): Keep `BuiltinToolCallPart` and add a `locally_executed: bool` flag (or a `provider_details` sentinel). `CallToolsNode._run_stream` gets a new branch that resolves the local standin and dispatches. Preserves builtin-origin for history/replay/eventing. Requires `_agent_graph.py` changes and possibly a standin-marker field on `ToolDefinition`.

Douwe leaned Route B in the call (*'we are losing information by not keeping it as a built-in tool called part'*, *'it's usually the agent graph that's responsible for looking up the tools'*) but flagged uncertainty pending SDK signature research. SDK research is now done.

**Needed from Douwe**: which route, and if Route B, how to register the local standin (new field on `ToolDefinition`, new capability hook, etc.).

### Q3 — OpenAI deferred-corpus wire shape: namespace wrapper vs flat list (narrows Stage 3)

The raw-SDK reference test uses `NamespaceToolParam(type='namespace', name='tools', tools=[...])` wrapping the deferred tools, with `defer_loading: True` set on each inner tool, plus `ToolSearchToolParam(type='tool_search')` as a sibling top-level tool.

pydantic-ai today emits tools as a flat list on the Responses API request. Questions:

- **Q3a**: Does the OpenAI Responses API *require* the namespace wrapper for `tool_search` to work, or is the flat list also accepted with per-tool `defer_loading: True`?
- **Q3b**: If the namespace is required, how does that interact with the existing pydantic-ai toolset model? A new wrapper in `models/openai.py` that groups deferred tools into a namespace before emission?
- **Q3c**: The SDK's `NamespaceToolParam.tools` inner TypedDict (`ToolFunction`) lacks `defer_loading` (reference test has a TODO noting this). We may need to emit the inner tool dict manually with a `pyright: ignore` or wait for an OpenAI SDK update.

**Action**: confirm via a live API call during Stage 3 (instrument with `logfire.instrument_httpx()` per `CLAUDE.local.md`).

### Q4 — `'auto'` precedence on provider switch

When both Anthropic and OpenAI native are available (agent switches providers mid-run), how does `strategy='auto'` route? Likely: resolved fresh per-request in `model.prepare_request()` so each provider picks its own default. Document as a known limitation, no special handling beyond what Stages 2/3 already imply. Non-blocking.

### Q5 — Callable return type for custom strategy (blocks Stage 4)

Douwe in the call: *'the user's custom callable returns a `Sequence[str]` of tool names, description from docstring.'* But the two providers return different shapes:

- **Anthropic** `tool_search_tool_result` → returns `tool_references` (names only). `Sequence[str]` → wrap each in a `ToolReferenceBlockParam` → done.
- **OpenAI** `tool_search_output` → returns full `tools: List[Tool]` definitions. `Sequence[str]` → adapter must look up each name in the local toolset to get the full `Tool`. Requires the adapter to have access to the full toolset at response-construction time. Verify this is possible — the adapter has `ModelRequestParameters` but does it have the full resolved toolset at response-parsing time?

**Needed from Douwe**: is `Sequence[str]` the right signature, or should the callable return richer metadata (`Sequence[Tool]` or `Sequence[ToolDefinition]`)?

## Architecture

### The asymmetry that drives the design

| | Anthropic | OpenAI |
|---|---|---|
| Native server search | ✅ `tool_search_tool_bm25` / `tool_search_tool_regex` | ✅ `tool_search` with `execution='server'` |
| Native custom-callable | ❌ **Not a builtin** — define a regular custom function tool returning `tool_reference` blocks | ✅ First-class via `execution='client'` + `parameters` schema |
| Output shape | `tool_references` (just names) | `tools` (full defs) |
| Wire-format deferred marker | `defer_loading: True` on each `BetaToolParam` in the flat tools list | `defer_loading: True` on each inner `ToolFunction` inside a `NamespaceToolParam` wrapper (Q3) |

**Implication**: The `'callable'` strategy is genuinely asymmetric. On OpenAI it routes through `BuiltinToolCallPart('tool_search')` → local execution (Q2). On Anthropic it's just a regular custom tool — no builtin routing. Stages 2 and 3 handle native server search; Stage 4 is where the asymmetry matters.

### Strategy resolution happens at the model layer

From the call (Douwe): *'if it's something that needs to be decided at the provider mapping, then it means we need to pass it all the way down through more request parameters... we need to only make the decision in `model.prepare_request()` because that's where we know what model we really are.'*

So `_DEFAULT_STRATEGY` lives in each `models/*.py`, not on the capability:

- `models/anthropic.py`: `'auto'` / `'bm25'` → `tool_search_tool_bm25_20251119`; `'regex'` → `tool_search_tool_regex_20251119`; `'substring'` → keep `ToolSearchToolset` local; `Callable` → regular custom function tool (no builtin routing).
- `models/openai.py`: `'auto'` / `'tool_search'` → `ToolSearchToolParam(execution='server')`; `'substring'` → keep `ToolSearchToolset` local; `Callable` → `ToolSearchToolParam(execution='client', description=docstring, parameters=<schema>)` + routing per Q2.
- Other providers: `'auto'` → `'substring'` (local fallback via existing `ToolSearchToolset`).

## Stages

### Stage 1 — Extend the existing `ToolSearch` capability

**Blocked on**: Q1.

The `ToolSearch` capability is already merged (PR #5047) but minimal. Auto-injection and `CapabilityOrdering` already work. Stage 1 adds configurability:

- Promote `ToolSearch` from internal to public: `get_serialization_name()` returns `'tool_search'` (or similar).
- Add `strategy: Literal['auto', 'substring', 'bm25', 'regex', 'tool_search'] | Callable[..., Sequence[str]] = 'auto'` field.
- Route `ToolSearch.get_wrapper_toolset()` based on Q1 answer:
  - If Q1-A: add `corpus_for_builtin` field on `ToolDefinition` (`tools.py`), wrap via a new `PreparedToolset.prepare_func` that injects `corpus_for_builtin='tool_search'` on every deferred tool def when strategy is anything other than `'substring'`. Keep the existing `ToolSearchToolset` wrap for `strategy='substring'`.
  - If Q1-B: no new field, but expose the `ToolSearch` instance through `ModelRequestParameters.capabilities` for adapter inspection.
  - If Q1-C: refactor to `class ToolSearch(BuiltinOrLocalTool)` with `_default_builtin() → ToolSearchTool()`, `_builtin_unique_id() → 'tool_search'`. Revisit `prefer_builtin` semantics.
  - If Q1-D: new `NativeToolSearchToolset` wrapper selected inside `get_wrapper_toolset()` based on profile lookup.
- Define `ToolSearchTool(AbstractBuiltinTool)` in `builtin_tools.py` with `kind='tool_search'` and a `unique_id` property. Provider-detection marker only — config lives on the `ToolSearch` capability.
- **No change to** `agent/__init__.py` auto-injection logic — the existing `_inject_auto_capabilities()` still works (auto-injects a default `ToolSearch()` instance when user doesn't provide one).
- File manifest (Stage 1): `capabilities/_tool_search.py` (extend), `builtin_tools.py` (new class), `tools.py` (new field if Q1-A), possibly `models/__init__.py` (`ModelRequestParameters` surface if Q1-B).
- **No model adapter changes.** Lands as an isolated PR.

### Stage 2 — Anthropic native server search (bm25 + regex)

- Fill in the `NotImplementedError` stub at `models/anthropic.py` `_map_server_tool_use_block` for `tool_search_tool_regex` / `tool_search_tool_bm25` → `BuiltinToolCallPart(tool_name='tool_search', provider_name='anthropic', ...)`.
- Add result mapper for `BetaToolSearchToolResultBlock` → `BuiltinToolReturnPart` paralleling `_map_web_search_tool_result_block`.
- Request mapping: when the model profile supports `tool_search` and strategy resolves to `'bm25'` / `'regex'`, emit `BetaToolSearchToolBm25_20251119Param` / `BetaToolSearchToolRegex20251119Param`. Send deferred tool defs in the flat `BetaToolParam` list with `defer_loading=True` on each (reference test pattern).
- Strategy resolution in `models/anthropic.py`: `_resolve_tool_search_strategy(capability) -> Literal['bm25', 'regex']` with `'auto'` → `'bm25'`.
- Streaming: extend the streaming handler for `tool_search_tool_*` content blocks (mirror web search streaming).
- History replay: round-trip `BuiltinToolCallPart('tool_search', provider_name='anthropic')` → `BetaServerToolUseBlockParam` on replay. Parts with `provider_name != 'anthropic'` silently skipped (existing behavior).
- Profile: add `'tool_search'` to `supported_builtin_tools` for Sonnet 4.0+ / Opus 4.0+ in `profiles/anthropic.py`.

### Stage 3 — OpenAI native server search

**Blocked on partial answer to**: Q3 (namespace wrapper vs flat list — live-test during implementation).

- Add `_map_tool_search_call()` in `models/openai.py` paralleling `_map_web_search_tool_call`. Returns `(BuiltinToolCallPart, BuiltinToolReturnPart)` tuple for `ResponseToolSearchCall(execution='server')` + `ResponseToolSearchOutputItem`.
- Request mapping: emit `ToolSearchToolParam(type='tool_search', execution='server')`. Send deferred tool defs per Q3 findings (namespace wrapper or flat with `defer_loading: True`). Live-test with `logfire.instrument_httpx()` before committing to either shape.
- Strategy resolution in `models/openai.py`: `'auto'` → `'tool_search'`.
- History replay: `BuiltinToolCallPart('tool_search', provider_name='openai')` → `ResponseToolSearchCallParam`. Cross-provider parts silently skipped.
- Profile: `'tool_search'` in `supported_builtin_tools` for `gpt-5.4+` in `profiles/openai.py`.
- Chat Completions API: out of scope (tool search is Responses API only).

### Stage 4 — Custom callable strategy

**Blocked on**: Q2 (Route A vs B), Q5 (callable signature).

- Define the standardized callable signature once Q5 is answered. Working assumption: `Callable[[str], Sequence[str]]` (single query string → list of tool names, description from docstring).
- **Anthropic adapter**: register the callable as a regular custom function tool (not a builtin variant — Anthropic has no custom tool-search param type). Agent graph already routes regular tool calls to local execution. User's function returns names; the adapter wraps each in `ToolReferenceBlockParam` and sends the result via `BetaToolResultBlockParam`. **No new routing on the Anthropic side.**
- **OpenAI adapter**: emit `ToolSearchToolParam(type='tool_search', execution='client', description=docstring, parameters=<schema_for_str_query>)`. The model responds with `ResponseToolSearchCall(execution='client')`. Routing per Q2:
  - **Route A**: parse-time conversion to `ToolCallPart(tool_name=<local_search_tool_name>, tool_call_id=<preserved>)`. Zero `_agent_graph.py` changes.
  - **Route B**: emit `BuiltinToolCallPart(locally_executed=True)`. `CallToolsNode._run_stream` gets a new branch that resolves the standin local tool and dispatches. Requires new fields and a new graph branch.
- Cross-provider test: run with Anthropic (regular tool path), then replay history on OpenAI (custom callable path) — confirm replay semantics.

### Stage 5 — Tests

**Reuse existing** `tests/test_native_tool_search_vcr.py` — don't create a new file. The existing raw-SDK reference tests stay as documentation of the wire format; **add pydantic-ai integration tests to the same file**, prefixed with `test_pai_*` or in a `TestPydanticAI...` class to distinguish from the raw-SDK references.

Case-based parametrized design per `tests/AGENTS.md`:

- `test_pai_anthropic_bm25`, `test_pai_anthropic_regex`
- `test_pai_openai_tool_search_server`
- `test_pai_auto_routes_to_native` (per provider)
- `test_pai_auto_falls_back_to_substring` (unsupported providers)
- `test_pai_cross_provider_history` (Anthropic → OpenAI history replay, builtin parts skipped, `search_tools` available for rediscovery)
- `test_pai_custom_callable_anthropic` (regular tool path)
- `test_pai_custom_callable_openai` (Route A or B per Q2)

Snapshot request/response structures to verify correct API payloads. Existing `tests/test_tool_search.py` continues to pass (regression gate for local `ToolSearchToolset`).

Docs left for a later round per `CLAUDE.local.md` ('don't write docs/strings too early').

## Message History (cross-provider switch)

Decision confirmed from prior session: **no cross-provider discovery bridge**. APIs don't validate historical tool calls against current definitions.

```
ModelResponse (from Anthropic):
  BuiltinToolCallPart(provider='anthropic', name='tool_search')   → Anthropic sees; others skip
  BuiltinToolReturnPart(provider='anthropic', content=...)         → Anthropic sees; others skip
  ToolCallPart(name='find_tickers', args={...})                    → ALL providers see

ModelRequest:
  ToolReturnPart(name='find_tickers', content=[...])               → ALL providers see

On OpenAI fallback:
  - Skips Builtin* parts (provider mismatch)
  - Sees find_tickers call+results in history
  - find_tickers NOT in tool definitions (still deferred)
  - search_tools IS available → can rediscover if needed
```

## File Manifest

Legend: 🆕 new file, ✏️ modify existing, 📖 read-only reference (no changes).

| File | Stage | Status | Action |
|------|-------|--------|--------|
| `pydantic_ai_slim/pydantic_ai/capabilities/_tool_search.py` | 1 | ✏️ | Add `strategy` field, extend `get_wrapper_toolset()`, promote to public |
| `pydantic_ai_slim/pydantic_ai/capabilities/_ordering.py` | — | 📖 | Already supports our needs |
| `pydantic_ai_slim/pydantic_ai/capabilities/__init__.py` | 1 | ✏️ | Export `ToolSearch` publicly |
| `pydantic_ai_slim/pydantic_ai/builtin_tools.py` | 1 | 🆕 class | Add `ToolSearchTool(AbstractBuiltinTool)` |
| `pydantic_ai_slim/pydantic_ai/tools.py` | 1 | ✏️ | Add `corpus_for_builtin` field (if Q1-A) |
| `pydantic_ai_slim/pydantic_ai/agent/__init__.py` | — | 📖 | Auto-injection already works |
| `pydantic_ai_slim/pydantic_ai/toolsets/_tool_search.py` | 1 | ✏️ | Possibly add `NativeToolSearchToolset` variant (if Q1-D) |
| `pydantic_ai_slim/pydantic_ai/models/anthropic.py` | 2 | ✏️ | Fill `NotImplementedError` stub, request mapping, streaming, replay |
| `pydantic_ai_slim/pydantic_ai/profiles/anthropic.py` | 2 | ✏️ | `supported_builtin_tools += 'tool_search'` |
| `pydantic_ai_slim/pydantic_ai/models/openai.py` | 3 | ✏️ | `_map_tool_search_call`, request mapping, replay |
| `pydantic_ai_slim/pydantic_ai/profiles/openai.py` | 3 | ✏️ | `supported_builtin_tools += 'tool_search'` |
| `pydantic_ai_slim/pydantic_ai/messages.py` | 4 | ✏️ | `locally_executed` flag (if Q2 = Route B) |
| `pydantic_ai_slim/pydantic_ai/_agent_graph.py` | 4 | ✏️ | New branch in `CallToolsNode._run_stream` (if Q2 = Route B) |
| `tests/test_native_tool_search_vcr.py` | 5 | ✏️ | Add pydantic-ai integration cases alongside existing raw-SDK references |

## Verification (per `CLAUDE.md` + `CLAUDE.local.md`)

1. `make format && make lint`
2. `make typecheck 2>&1 | tee /tmp/typecheck-output.txt` (pipe — command takes ~45s)
3. `source .env && uv run pytest tests/test_native_tool_search_vcr.py -x --record-mode=rewrite` (live record new tests)
4. `source .env && uv run pytest tests/test_native_tool_search_vcr.py -x` (replay)
5. `uv run pytest tests/test_tool_search.py -x` (regression on local toolset)

## Out of Scope

- **Keyword-algorithm fix (substring → better matching)**: Douwe wants this merged ASAP as a separate urgent bug-fix PR (substring `'PR'` matches unrelated tokens). Upstream PR by MagnusS0 fixes it. Not a blocker for this plan.
- **Evals for tool search effectiveness**: Douwe wants better evals (canonical benchmark: GitHub MCP 'top 5 issues' example). Separate work in the harness repo.
- **Eventual move of tool search to the harness**: V3 conversation, not now.
- **OpenAI Chat Completions support**: Responses API only.

## TODO (post-PR)

- Ask Douwe to apply the `provider-adaptive` label to this PR and #4090, #4167, #4566, #5047 for philosophy review (David's call request).
