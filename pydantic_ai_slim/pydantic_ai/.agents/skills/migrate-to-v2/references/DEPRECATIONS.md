# v1 → v2 Deprecation Codemod Table

Each entry: the **warning-message substring** you'll grep test output for, the **v1 form**, the **v2 form**, and the PR introducing the deprecation. All substrings in this table were captured by running `scripts/probe_warnings.py` against a clean `pydantic-ai==1.100.0` install — they are the exact text emitted, not paraphrases. Use them in `pytest.warns(match=...)` or grep filters.

Codemods are listed in roughly the order they're likely to appear in a typical codebase (imports first, then ctor kwargs, then call-site renames). When in doubt, trust the deprecation warning text over this table.

---

## A. Agent constructor kwargs → `capabilities=[...]`

When migrating multiple of these on one `Agent(...)`, **merge into a single `capabilities=[...]` list**. See `SKILL.md` step 3.

All capabilities live in `pydantic_ai.capabilities`.

### A1. `Agent(instrument=...)` → `Instrumentation` capability — PR #4967

- Warning substring: `` `Agent(instrument=...)` is deprecated, use `capabilities=[Instrumentation(...)]` instead. ``
- Also fires for: `Agent.instrument` getter/setter, `Agent.from_spec(instrument=...)`.

```text
# v1
Agent('openai:gpt-4o', instrument=True)
Agent('openai:gpt-4o', instrument=InstrumentationSettings(...))

# v2
from pydantic_ai.capabilities import Instrumentation
Agent('openai-chat:gpt-4o', capabilities=[Instrumentation()])
# Verified v2 ctor: Instrumentation(settings: InstrumentationSettings = <factory>)
Agent('openai-chat:gpt-4o', capabilities=[Instrumentation(InstrumentationSettings(version=4))])
```

### A2. `Agent(history_processors=...)` → `ProcessHistory` — PR #5425

- Warning substring: `` `Agent(history_processors=[fn, ...])` is deprecated and will be removed in v2.0. Replace with `Agent(capabilities=[ProcessHistory(fn), ...])`, or hook the `before_model_request` lifecycle event directly via `Hooks(before_model_request=fn)`. ``

```text
# v1
Agent('openai:gpt-4o', history_processors=[strip_pii, summarize])
# v2
from pydantic_ai.capabilities import ProcessHistory
Agent('openai-chat:gpt-4o', capabilities=[ProcessHistory(strip_pii), ProcessHistory(summarize)])
```

`ProcessHistory.__init__(processor: HistoryProcessorFunc[AgentDepsT])` — single-callable per instance, no list form.

### A3. `Agent(prepare_tools=...)` → `PrepareTools` — PR #5335

- Warning substring: `` `Agent(prepare_tools=...)` is deprecated and will be removed in v2.0. Use `capabilities=[PrepareTools(prepare_tools)]` instead. Note: `prepare_tools` runs only on function tools — to prepare output tools, also pass `PrepareOutputTools(prepare_output_tools)` in `capabilities=[...]`. ``

```text
from pydantic_ai.capabilities import PrepareTools
Agent('openai-chat:gpt-4o', capabilities=[PrepareTools(my_prepare)])
```

### A4. `Agent(prepare_output_tools=...)` → `PrepareOutputTools` — PR #5335

- Warning substring: `` `Agent(prepare_output_tools=...)` is deprecated and will be removed in v2.0. Use `capabilities=[PrepareOutputTools(prepare_output_tools)]` instead. ``

### A5. `Agent(event_stream_handler=...)` → `ProcessEventStream` — PR #5335

- Warning substring: `` `Agent(event_stream_handler=...)` is deprecated and will be removed in v2.0. Use `capabilities=[ProcessEventStream(handler)]` instead. ``

```text
from pydantic_ai.capabilities import ProcessEventStream
Agent('openai-chat:gpt-4o', capabilities=[ProcessEventStream(my_handler)])
```

### A6. `Agent(tool_retries=)` / `Agent(output_retries=)` → `retries={...}` dict — PR #5518

The constructor kwargs are dropped; the underlying knob is now a dict on `retries=`.

- Warning substring (`tool_retries`): `` `Agent(tool_retries=...)` is deprecated and will be removed in v2.0. Use `retries={'tools': ...}` (or `retries=<int>` to set the same budget for both tool and output retries) instead. ``
- Warning substring (`output_retries`): `` `Agent(output_retries=...)` is deprecated and will be removed in v2.0. Use `retries={'output': ...}` (or `retries=<int>` to set the same budget for both tool and output retries) instead. ``

```text
# v1
Agent('openai:gpt-4o', tool_retries=3, output_retries=2)
# v2
Agent('openai-chat:gpt-4o', retries={'tools': 3, 'output': 2})
# or, if both budgets should be the same:
Agent('openai-chat:gpt-4o', retries=3)
```

Note: `Agent(retries=)` itself is **not** deprecated.

### A7. `Agent(mcp_servers=...)` → `toolsets=`

- Warning substring: `` `mcp_servers` is deprecated, use `toolsets` instead ``

```text
Agent('openai-chat:gpt-4o', toolsets=[my_mcp_toolset])
```

---

## B. Model + provider renames

### B1. `OpenAIModel` → `OpenAIChatModel` — PR #2676

- Warning substring: `` `OpenAIModel` was renamed to `OpenAIChatModel` to clearly distinguish it from `OpenAIResponsesModel` ``

```text
# v1
from pydantic_ai.models.openai import OpenAIModel, OpenAIModelSettings
# v2
from pydantic_ai.models.openai import OpenAIChatModel, OpenAIChatModelSettings
```

### B1b. `openai:` string form behavior flip — runtime warning at construction

- Warning substring (v1.100.0 only, advisory): `` In v2.0, 'openai:' will resolve to the OpenAI Responses API by default. Use 'openai-chat:' to keep current Chat Completions behavior, or 'openai-responses:' to opt in early. ``

The v1 deprecation does **not** force the change — it warns. In v2, `'openai:'` silently flips to Responses. For a low-risk upgrade PR, change all `'openai:...'` strings to `'openai-chat:...'`. See `BEHAVIOR_CHANGES.md` §2.

### B2. `GeminiModel` → `GoogleModel` — PR #2416

- Warning substring: `` Use `GoogleModel` instead. See <https://ai.pydantic.dev/models/google/> for more details. ``

```text
# v1
from pydantic_ai.models.gemini import GeminiModel
# v2
from pydantic_ai.models.google import GoogleModel
```

### B3. `GoogleGLAProvider` / `GoogleVertexProvider` → `GoogleProvider` — PR #5336, #2450

- Warning substring: `` `GoogleGLAProvider` is deprecated, use `GoogleProvider` with `GoogleModel` instead. ``
- Warning substring: `` `GoogleVertexProvider` is deprecated, use `GoogleProvider` with `GoogleModel` instead. ``

```text
# v1
from pydantic_ai.providers.google_gla import GoogleGLAProvider
from pydantic_ai.providers.google_vertex import GoogleVertexProvider
# v2
from pydantic_ai.providers.google import GoogleProvider
```

Provider-prefix renames (model strings):
- `'google-gla:...'` → `'google:...'`
- `'google-vertex:...'` → unverified — see UNVERIFIED §B3c below.
- `'gateway/gemini:...'` → `'gateway/google-cloud:...'` (UNVERIFIED, see below)

### B3c. UNVERIFIED — gateway prefix and `google-vertex:` rename

The `gateway/gemini:` warning is conditional on a valid Pydantic AI Gateway key (`PYDANTIC_AI_GATEWAY_API_KEY` that encodes a region). The probe with a dummy key raises before the deprecation can fire. The warning text in v1.100.0 source is: `` The 'gateway/gemini:' prefix is deprecated and will be removed in v2.0. Use 'gateway/google-cloud:' instead. `` — confirmed by grep, but not runtime-validated. Treat as best-effort.

Similarly, `google-vertex:` → `google-cloud:` is in the v2 source but the v1 string mapping was not exercised by the probe (no `google-cloud:` provider prefix in v1, so the deprecation likely fires only via `GoogleProvider(vertexai=True)` migration). Apply by analogy to `google-gla:` → `google:`.

### B4. Positional model name without provider prefix — PR #2711

- Warning substring: `` Specifying a model name without a provider prefix is deprecated. Instead of 'gpt-4o', use 'openai:gpt-4o'. ``

Always include the `provider:` prefix. Combine with B1b — use `openai-chat:` for a safe migration.

### B5. `GrokProvider` → `XaiProvider` + `XaiModel` — PR #3400

- Warning substring: `` `GrokProvider` is deprecated, use `XaiProvider` with `XaiModel` instead for the native xAI SDK. See <https://ai.pydantic.dev/models/xai/> for more details. ``

```text
# v1
from pydantic_ai.providers.grok import GrokProvider
# v2
from pydantic_ai.providers.xai import XaiProvider
from pydantic_ai.models.xai import XaiModel
```

---

## C. Streaming / result APIs

### C1. `agent.iter(...).stream_responses()` → `.stream_response()` (singular) — PR #5296

Yields `ModelResponse` snapshots, **not `(ModelResponse, is_last)` tuples**. Drop the tuple unpack.

- Warning substring: `` `AgentStream.stream_responses()` is deprecated and will be removed in v2.0. Replace `async for r in stream.stream_responses(...)` with `async for r in stream.stream_response(...)` (singular). Both yield the same `ModelResponse` snapshots ``

```text
# v1
async for msg, is_last in stream.stream_responses():
    if is_last:
        ...

# v2
async for msg in stream.stream_response():
    if msg.state == 'complete':
        ...
```

### C2. Method-style `result.usage()` / `result.timestamp()` / `stream.get()` → properties — PR #5263

In v1.100.0 these are `@deprecated_callable_property` — calling them returns the right type but emits a warning.

- Warning substring: `` `AgentRunResult.usage` is no longer a method; access it as a property (drop the parentheses). `` (and analogous for `timestamp`, `stream.get` → `stream.response`)
- Verified via `TestModel` against `1.100.0`.

```text
# v1
usage = result.usage()
ts = result.timestamp()
final = stream.get()

# v2
usage = result.usage
ts = result.timestamp
final = stream.response
```

These fire only when the method is *called*. Import-only smoke tests will not surface them — run a real (or recorded) agent flow.

### C3. `StreamedRunResult.stream` / `.stream_structured` / `.validate_structured_output` — PR #5463 (dropped)

- `` `StreamedRunResult.stream` is deprecated, use `stream_output` instead. ``
- `` `StreamedRunResult.stream_structured` is deprecated, use `stream_responses` instead. `` → then C1 applies, so go directly to `stream_response`.
- `` `validate_structured_output` is deprecated, use `validate_response_output` instead. ``

These were dropped at the source level on `v2-main`; on v1.100.0 they still emit warnings.

---

## D. MCP

### D1. `MCPServerStdio` → `MCPToolset` — PR #5325

- Warning substring: `` `MCPServerStdio` is deprecated and will be removed in v2. Use `MCPToolset('path/to/script.py')` for Python scripts, `MCPToolset('script.js')` for Node scripts, or `MCPToolset(fastmcp.client.transports.StdioTransport(command='...', args=[...]))` for arbitrary commands. ``

```text
# v1
from pydantic_ai.mcp import MCPServerStdio
server = MCPServerStdio('uv', args=['run', 'my_mcp.py'])
# v2
from pydantic_ai.mcp import MCPToolset
from fastmcp.client.transports import StdioTransport
toolset = MCPToolset(StdioTransport(command='uv', args=['run', 'my_mcp.py']))
# or, for a plain script:
toolset = MCPToolset('my_mcp.py')
```

### D2. `MCPServerSSE` → `MCPToolset(url)` — PR #5325

- Warning substring: `` `MCPServerSSE` is deprecated and will be removed in v2. Use `MCPToolset('http://.../sse')` instead — the SSE transport is automatically inferred from URLs ending in `/sse`. ``

### D3. `MCPServerStreamableHTTP` → `MCPToolset(url)` — PR #5325

- Warning substring: `` `MCPServerStreamableHTTP` is deprecated and will be removed in v2. Use `MCPToolset('http://.../mcp')` instead — Streamable HTTP is the default for HTTP URLs. ``

### D4. `MCPServerHTTP` → `MCPServerSSE` → `MCPToolset`

- Warning substring: `` The `MCPServerHTTP` class is deprecated, use `MCPServerSSE` instead. `` then D2.

### D5. `FastMCPToolset` — removed before 1.100.0

`pydantic_ai.mcp.FastMCPToolset` does **not exist** as an importable symbol in `pydantic-ai-slim==1.100.0`. PR #5325 mentions deprecating it but the symbol was removed before the 1.100.0 release. Verified at runtime:

```
ImportError: cannot import name 'FastMCPToolset' from 'pydantic_ai.mcp'
```

If a user is on an older v1 (<1.100.0) they will hit a deprecation warning; on 1.100.0+ they get the ImportError above. Action: replace with `MCPToolset` directly (see D1).

### D6. `MCPServer*` ctor `sse_read_timeout=` → `read_timeout=`

- Warning substring: `` 'sse_read_timeout' is deprecated, use 'read_timeout' instead. ``

### D7. `Agent.run_mcp_servers()` → `async with agent:`

- Warning substring: `` `run_mcp_servers` is deprecated, use `async with agent:` instead. If you need to set a sampling model on all MCP servers, use `agent.set_mcp_sampling_model()`. ``

### D8. `load_mcp_servers(...)` → `load_mcp_toolsets(...)` — PR #5325

Symbol-rename in `pydantic_ai.mcp`. Old name removed in v2; new name returns `MCPToolset` instances.

### D9. `pydantic_ai.mcp.MCPServerConfig` → inline `MCPToolset` construction

- Warning substring: `` `pydantic_ai.mcp.MCPServerConfig` is deprecated and will be removed in v2. Pass the JSON config to `load_mcp_toolsets(...)` directly, or build `MCPToolset`s inline from `fastmcp.client.transports.StdioTransport` / URLs. ``

---

## E. A2A / AG-UI / Outlines / ACI

### E1. `Agent.to_a2a()` → `fasta2a.pydantic_ai.agent_to_a2a` — PR #5426

- Warning substring: `` `Agent.to_a2a()` is deprecated and will be removed in 2.0. The `fasta2a` package is now maintained at https://github.com/datalayer/fasta2a — install it with the `pydantic-ai` extra (`pip install 'fasta2a[pydantic-ai]>=0.6.1'`) and use `from fasta2a.pydantic_ai import agent_to_a2a` directly. ``

```text
# v1
app = agent.to_a2a()
# v2
from fasta2a.pydantic_ai import agent_to_a2a
app = agent_to_a2a(agent)
```

Also drop `pydantic-ai[fasta2a]` from `pyproject.toml`; depend on `fasta2a[pydantic-ai]>=0.6.1` directly.

### E2. `AGUIApp` / `Agent.to_ag_ui()` / `pydantic_ai.ag_ui` shim → `AGUIAdapter` — PR #5345

- Warning substring (module shim): `` The `pydantic_ai.ag_ui` module is deprecated and will be removed in 2.0. Replace: ``  (the warning then prints the exact `from`/`with` block — full text is multi-line, match on the leading sentence).
- Warning substring (`to_ag_ui`): `` `Agent.to_ag_ui()` is deprecated and will be removed in 2.0. ``
- Warning substring (`AGUIApp`): `` `AGUIApp` is deprecated and will be removed in 2.0. ``

```text
# v1
from pydantic_ai.ag_ui import AGUIAdapter, StateDeps
app = agent.to_ag_ui()

# v2
from pydantic_ai.ui import StateDeps, SSE_CONTENT_TYPE
from pydantic_ai.ui.ag_ui import AGUIAdapter
from starlette.applications import Starlette
from starlette.routing import Route

async def run_agent(request):
    return await AGUIAdapter.dispatch_request(request, agent=agent)

app = Starlette(routes=[Route('/', run_agent, methods=['POST'])])
```

`handle_ag_ui_request` and `run_ag_ui` are **removed** in v2 — call `AGUIAdapter.dispatch_request()` directly.

### E3. `OutlinesModel` / `OutlinesProvider` → removed — PR #5432

- Warning substring (`OutlinesModel`): `` `OutlinesModel` is deprecated and will be removed in v2. If you would like to keep using Outlines with Pydantic AI, please file an issue at https://github.com/dottxt-ai/outlines/issues. ``
- Warning substring (`OutlinesProvider`): `` `OutlinesProvider` is deprecated and will be removed in v2. If you would like to keep using Outlines with Pydantic AI, please file an issue at https://github.com/dottxt-ai/outlines/issues. ``
- Both fire on **instantiation** (the classes are decorated with `@deprecated`), not on bare import. Verified against `outlines==1.3.0` + `pydantic-ai==1.100.0`.

In v2, `pydantic_ai.models.outlines` is removed (ImportError). No in-tree replacement — tell the user to track <https://github.com/dottxt-ai/outlines/issues> or drop Outlines.

### E4. `pydantic_ai.ext.aci` (`tool_from_aci`, `ACIToolset`) → removed — PR #5510

- Warning substring: `` `pydantic_ai.ext.aci` is deprecated and will be removed in 2.0. Wrap ACI.dev tools yourself using `pydantic_ai.tools.Tool.from_schema` against `aci.ACI().functions.get_definition(...)`, or use the upstream `aci-sdk` integration directly. ``
- Fires on **use** (calling `tool_from_aci(...)` or instantiating `ACIToolset(...)`), **not** on bare import. Verified against `aci-sdk==1.0.0b4` + `pydantic-ai==1.100.0`.

In v2, `pydantic_ai.ext.aci` is removed (ImportError). No in-tree replacement. v2 migration: wrap manually with `pydantic_ai.tools.Tool.from_schema`.

```text
# v2 — verified Tool.from_schema signature:
#   Tool.from_schema(function, name, description, json_schema, takes_ctx=False,
#                    sequential=False, args_validator=None)
from pydantic_ai.tools import Tool
import aci

defn = aci.ACI().functions.get_definition('GITHUB__CREATE_ISSUE')
tool = Tool.from_schema(
    function=lambda **kwargs: aci.ACI().handle_function_call('GITHUB__CREATE_ISSUE', kwargs),
    name=defn['name'],
    description=defn.get('description', ''),
    json_schema=defn['parameters'],
)
```

---

## F. Usage / token-field renames

### F1. `Usage` → `RunUsage` — PR #2378

- Warning substring: `` `Usage` is deprecated, use `RunUsage` instead ``

Also renames at the field level:
- `request_tokens` → `input_tokens` (warning: `` `request_tokens` is deprecated, use `input_tokens` instead ``)
- `response_tokens` → `output_tokens`
- `request_tokens_limit` → `input_tokens_limit`
- `response_tokens_limit` → `output_tokens_limit`

Per-request usage (single model call) is now `RequestUsage`; aggregate over a run is `RunUsage`.

```text
# v1
from pydantic_ai.usage import Usage
# v2
from pydantic_ai.usage import RunUsage, RequestUsage
```

### F2. `vendor_*` → `provider_*` on `ModelResponse` — PR #5476

- Warning substring: `` `vendor_details` is deprecated, use `provider_details` instead ``
- Warning substring: `` `vendor_id` is deprecated, use `provider_response_id` instead ``
- Warning substring: `` `provider_request_id` is deprecated, use `provider_response_id` instead ``

### F3. `{FunctionToolCallEvent,FunctionToolResultEvent}.call_id` → `.tool_call_id` — PR #2028

- Warning substring: `` `call_id` is deprecated, use `tool_call_id` instead. ``

### F4. `price` → `cost` on usage

- Warning substring: `` `price` is deprecated, use `cost` instead ``

### F5. `FunctionToolResultEvent.result` → `.part`

- Warning substring: `` `result` is deprecated, use `part` instead. ``

---

## G. Built-in tools → `NativeTool` capability

In `pydantic-ai==1.100.0` the `Agent(builtin_tools=...)` ctor kwarg still exists and emits a deprecation. In v2 it is removed entirely (`TypeError: unexpected keyword argument 'builtin_tools'`) — and notably so is `native_tools=` as a kwarg. The v2 form is `capabilities=[NativeTool(...)]` (or a provider-adaptive capability like `WebSearch()` / `WebFetch()` / `MCP()` / `ImageGeneration()`).

- Warning substring: `` `Agent(builtin_tools=...)` is deprecated, use `capabilities=[NativeTool(...)]` for raw native-tool registration, or a provider-adaptive capability like `WebSearch()`, `WebFetch()`, `MCP()`, or `ImageGeneration()` for native-or-local fallback. ``
- Verified at runtime against 1.100.0.

```text
# v1
Agent('openai:gpt-4o', builtin_tools=[WebSearchTool()])
# v2
from pydantic_ai.capabilities import NativeTool, WebSearch
Agent('openai-chat:gpt-4o', capabilities=[NativeTool(WebSearchTool())])
# or for native-or-local fallback:
Agent('openai-chat:gpt-4o', capabilities=[WebSearch()])
```

`WebSearchTool` is deprecated in favor of the `WebSearch` capability; `WebFetchTool` is deprecated in favor of `WebFetch`. See `BEHAVIOR_CHANGES.md` §3 for the local-fallback default flip.

---

## H. Output / toolsets / misc

### H1. `DeferredToolCalls` → `DeferredToolRequests`

- Warning substring: `` `DeferredToolCalls` is deprecated, use `DeferredToolRequests` instead ``
- Field rename: `.tool_calls` → `.calls`
- v2 import location: `from pydantic_ai.tools import DeferredToolRequests` (also re-exported from top-level `pydantic_ai`). Note: **not** in `pydantic_ai.output` in v2.

### H2. `DeferredToolset` → `ExternalToolset`

- Warning substring: `` `DeferredToolset` is deprecated, use `ExternalToolset` instead ``

### H3. UNVERIFIED — `HistoryProcessor` alias

`from pydantic_ai import HistoryProcessor` does not exist in `pydantic-ai-slim==1.100.0`. The class lives at `pydantic_ai.capabilities.HistoryProcessor` (no deprecation alias). The v2 form for processing history is `pydantic_ai.capabilities.ProcessHistory` (A2). If a user has a stale alias from an older v1, point them at `ProcessHistory`.

### H4. UNVERIFIED — `cached_async_http_client`

`from pydantic_ai.providers import cached_async_http_client` does not exist in 1.100.0. If a user has this from an older v1, the replacement is `pydantic_ai.providers.create_async_http_client`.

### H5. `OpenAIModelProfile.openai_supports_sampling_settings`

- Warning substring: `` Set the `system_prompt_role` in the `OpenAIModelProfile` instead. ``
- Already dropped in `3f7427673` on `v2-main`.

### H6. `OpenAICompaction(instructions=)` — dropped

- Warning substring: `` `OpenAICompaction(instructions=...)` is deprecated and will be removed in a future version. OpenAI's `/compact` endpoint treats `instructions` as a system/developer message inserted into the compaction model's context, not as a directive for how to summarize the conversation, so this field does not match `AnthropicCompaction(instructions=...)` semantics. ``
- Category: plain `DeprecationWarning` (not `PydanticAIDeprecationWarning`).
- In v2 the kwarg is removed (`TypeError`). No 1:1 replacement — drop the `instructions=` argument; if you need to steer the summarization, do it via the surrounding agent prompt instead.
- Verified against 1.100.0 and 2.0.0b1.

### H7. `parallel_execution_mode("sequential")`

- Warning substring: `` Use `parallel_execution_mode("sequential")` instead. ``

Replaces a method on `AbstractAgent` / `ToolManager`. Exact PR TBD; trust the warning at the call site.

---

## I. `pydantic_graph`

### I1. Legacy `BaseNode`-runner imports from top-level — PR #5306

- Warning substring: `` Importing `Graph` from `pydantic_graph` is deprecated. The `BaseNode`-based `Graph` runner and its persistence machinery are deprecated and will be removed (or repurposed) in v2; use the builder-based `GraphBuilder` API instead, or pin to `pydantic_graph<2` to keep using them. ``

Affected: `Graph`, `GraphRun`, `GraphRunResult`, `BaseNode`, `Edge`, `End`, `GraphRunContext`, `EndSnapshot`, `NodeSnapshot`, `Snapshot`, `FullStatePersistence`, `SimpleStatePersistence`. All accessed via `pydantic_graph.__getattr__`.

Migration: rewrite using `GraphBuilder` (see `docs/graph.md`). If a full rewrite is out of scope, pin `pydantic-graph<2`.

### I2. `pydantic_graph.beta.*` → top-level — PR #5306

- Warning substring: `` `pydantic_graph.beta.decision` is deprecated, import from `pydantic_graph.decision` instead. ``

```text
# v1
from pydantic_graph.beta.decision import Decision
# v2
from pydantic_graph.decision import Decision
```

---

## J. `pydantic_evals`

### J1. `Dataset(...)` without `name=` — PR #4862

- Warning substring: `` Omitting the `name` parameter is deprecated. Please provide a name for your `Dataset`. ``

```text
# v1
Dataset(cases=[...], evaluators=[...])
# v2
Dataset(name='my_eval', cases=[...], evaluators=[...])
```

### J2. Positional construction of evals classes / positional `Dataset.evaluate` args — PR #5547

Two distinct positional warnings exist in 1.100.0:

- Positional dataclass init for `EvaluationResult` / `EvaluatorFailure`:
  Warning substring: `` Constructing `EvaluationResult` with positional arguments is deprecated; use keyword arguments instead. Positional construction will be removed in pydantic-evals v2. `` (and analogous for `EvaluatorFailure`).
- Positional kwargs (`name`, `max_concurrency`, `progress`, `retry_task`, `retry_evaluators`) passed to `Dataset.evaluate` / `Dataset.evaluate_sync`:
  Warning substring: `` Passing `name`, `max_concurrency`, `progress`, `retry_task`, or `retry_evaluators` positionally to `Dataset.evaluate` / `Dataset.evaluate_sync` is deprecated; pass them as keyword arguments. Positional support will be removed in pydantic-evals v2. ``

Both verified against `pydantic-evals==1.100.0`. In v2 (2.0.0b1) both raise `TypeError` instead. Fix by switching to kwargs.

Note: `Dataset(...)` itself is `kw_only` in 1.100.0 already (positional args raise `TypeError` directly — no deprecation), so the "positional Dataset" case in older drafts of this table doesn't apply.

### J3. `evaluation_name` / `evaluator_version` class/instance attribute — PR #5554

- Warning substring (`evaluation_name`): `` <mod>.<Cls> relies on the `evaluation_name` attribute to customize the default evaluation name. This is deprecated; override `get_default_evaluation_name` in your evaluator class to retain this behavior in pydantic-evals v2. ``
- Warning substring (`evaluator_version`): `` <mod>.<Cls> relies on the `evaluator_version` attribute to set its version. This is deprecated; override `get_evaluator_version` in your evaluator class to retain this behavior in pydantic-evals v2. ``
- Both fire when `get_default_evaluation_name()` / `get_evaluator_version()` are *called* on an evaluator subclass that sets those attributes, not on subclass creation. Match the static substrings (`relies on the \`evaluation_name\` attribute`, `relies on the \`evaluator_version\` attribute`) and ignore the dynamic class prefix.
- Verified against `pydantic-evals==1.100.0`. v2 form: override `get_default_evaluation_name(self) -> str` and `get_evaluator_version(self) -> str | None` instead of setting class attributes.

---

## Items found in `git log` but not confidently mapped

These commits mention deprecations or removed APIs but didn't yield a clear single warning string in this snapshot. Flag in the migration checklist and look up if the user's test output names them:

- PR #5320 (yield `OutputToolCallEvent`/`OutputToolResultEvent` for output tool calls) — affects users who pattern-match on `FunctionToolCallEvent` for output-tool calls.
- PR #5075 (runtime `output_retries` override + retry-field rename) — interacts with A6.
- PR #4208 (`UsageLimits` preserves explicit 0 token limits) — fields covered in F1, cascade behavior worth noting.
- PR #5547, #5554 — see J2, J3.
- PR #2730 (`OpenAIModelProfile.openai_supports_sampling_settings`) — see H5.
