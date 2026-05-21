# v1 → v2 Deprecation Codemod Table

Each entry: warning-message substring (exact text emitted by `pydantic-ai==1.100.0`), v1 form, v2 form. Use substrings in `pytest.warns(match=...)` or grep. Trust the warning text over this table when they disagree.

---

## A. Agent constructor kwargs → `capabilities=[...]`

Merge all of these on one `Agent(...)` into a single `capabilities=[...]` list. Capabilities live in `pydantic_ai.capabilities`.

### A1. `Agent(instrument=...)` → `Instrumentation`

> "`Agent(instrument=...)` is deprecated, use `capabilities=[Instrumentation(...)]` instead."

Also fires for the `Agent.instrument` getter/setter and `Agent.from_spec(instrument=...)`.

```python
from pydantic_ai.capabilities import Instrumentation
Agent('openai-chat:gpt-4o', capabilities=[Instrumentation()])
Agent('openai-chat:gpt-4o', capabilities=[Instrumentation(InstrumentationSettings(version=4))])
```

### A2. `Agent(history_processors=...)` → `ProcessHistory`

> "`Agent(history_processors=[fn, ...])` is deprecated and will be removed in v2.0. Replace with `Agent(capabilities=[ProcessHistory(fn), ...])`, or hook the `before_model_request` lifecycle event directly via `Hooks(before_model_request=fn)`."

One `ProcessHistory` per callable — no list form.

```python
from pydantic_ai.capabilities import ProcessHistory
Agent(..., capabilities=[ProcessHistory(strip_pii), ProcessHistory(summarize)])
```

### A3. `Agent(prepare_tools=...)` → `PrepareTools`

> "`Agent(prepare_tools=...)` is deprecated and will be removed in v2.0. Use `capabilities=[PrepareTools(prepare_tools)]` instead. Note: `prepare_tools` runs only on function tools — to prepare output tools, also pass `PrepareOutputTools(prepare_output_tools)` in `capabilities=[...]`."

### A4. `Agent(prepare_output_tools=...)` → `PrepareOutputTools`

> "`Agent(prepare_output_tools=...)` is deprecated and will be removed in v2.0. Use `capabilities=[PrepareOutputTools(prepare_output_tools)]` instead."

### A5. `Agent(event_stream_handler=...)` → `ProcessEventStream`

> "`Agent(event_stream_handler=...)` is deprecated and will be removed in v2.0. Use `capabilities=[ProcessEventStream(handler)]` instead."

### A6. `Agent(tool_retries=)` / `Agent(output_retries=)` → `retries={...}` dict

> "`Agent(tool_retries=...)` is deprecated ... Use `retries={'tools': ...}` (or `retries=<int>` to set the same budget for both tool and output retries) instead."

Same for `output_retries=` → `retries={'output': ...}`. `Agent(retries=)` itself is **not** deprecated.

```python
Agent(..., retries={'tools': 3, 'output': 2})
```

### A7. `Agent(mcp_servers=...)` → `toolsets=`

> "`mcp_servers` is deprecated, use `toolsets` instead"

---

## B. Model + provider renames

### B1. `OpenAIModel` → `OpenAIChatModel`

> "`OpenAIModel` was renamed to `OpenAIChatModel` to clearly distinguish it from `OpenAIResponsesModel`"

Applies to `OpenAIModelSettings` → `OpenAIChatModelSettings` too.

### B1b. `'openai:'` string prefix flips to Responses API in v2

> "In v2.0, 'openai:' will resolve to the OpenAI Responses API by default. Use 'openai-chat:' to keep current Chat Completions behavior, or 'openai-responses:' to opt in early."

The v1 warning doesn't force the change. For a low-risk upgrade, change every `'openai:...'` → `'openai-chat:...'`. See `BEHAVIOR_CHANGES.md` §2.

### B2. `GeminiModel` → `GoogleModel`

> "Use `GoogleModel` instead."

`from pydantic_ai.models.gemini import GeminiModel` → `from pydantic_ai.models.google import GoogleModel`.

### B3. `GoogleGLAProvider` / `GoogleVertexProvider` → `GoogleProvider`

> "`GoogleGLAProvider` is deprecated, use `GoogleProvider` with `GoogleModel` instead." (and analogous for `GoogleVertexProvider`)

Model-string prefixes: `'google-gla:...'` → `'google:...'`. The `'google-vertex:'` and `'gateway/gemini:'` prefixes likely rename to `'google-cloud:'` / `'gateway/google-cloud:'`; trust the warning emitted at runtime.

### B4. Bare model name without provider prefix

> "Specifying a model name without a provider prefix is deprecated. Instead of 'gpt-4o', use 'openai:gpt-4o'."

Always include a `provider:` prefix. Combine with B1b — use `openai-chat:` for safety.

### B5. `GrokProvider` → `XaiProvider` + `XaiModel`

> "`GrokProvider` is deprecated, use `XaiProvider` with `XaiModel` instead for the native xAI SDK."

`from pydantic_ai.providers.grok import GrokProvider` → `from pydantic_ai.providers.xai import XaiProvider` + `from pydantic_ai.models.xai import XaiModel`.

---

## C. Streaming / result APIs

### C1. `stream_responses()` → `stream_response()` (singular)

> "`AgentStream.stream_responses()` is deprecated and will be removed in v2.0. Replace `async for r in stream.stream_responses(...)` with `async for r in stream.stream_response(...)` (singular). Both yield the same `ModelResponse` snapshots"

Drop the tuple unpack — v2 yields `ModelResponse`, not `(ModelResponse, is_last)`.

```python
async for msg in stream.stream_response():
    if msg.state == 'complete':
        ...
```

### C2. `result.usage()` / `result.timestamp()` / `stream.get()` → properties

> "`AgentRunResult.usage` is no longer a method; access it as a property (drop the parentheses)."

Analogous for `timestamp` and `stream.get` → `stream.response`. Fires only when called — import-smoke won't surface it. Grep for `\.usage()`, `\.timestamp()`, `\.get()` on stream/result objects.

### C3. `StreamedRunResult.stream*` / `validate_structured_output`

- `StreamedRunResult.stream` → `stream_output`
- `StreamedRunResult.stream_structured` → `stream_response` (singular; via C1)
- `validate_structured_output` → `validate_response_output`

---

## D. MCP

### D1. `MCPServerStdio` → `MCPToolset`

> "`MCPServerStdio` is deprecated and will be removed in v2. Use `MCPToolset('path/to/script.py')` for Python scripts, `MCPToolset('script.js')` for Node scripts, or `MCPToolset(fastmcp.client.transports.StdioTransport(command='...', args=[...]))` for arbitrary commands."

```python
from pydantic_ai.mcp import MCPToolset
from fastmcp.client.transports import StdioTransport
toolset = MCPToolset(StdioTransport(command='uv', args=['run', 'my_mcp.py']))
```

### D2 / D3. `MCPServerSSE` / `MCPServerStreamableHTTP` → `MCPToolset(url)`

> "`MCPServerSSE` is deprecated ... Use `MCPToolset('http://.../sse')` instead — the SSE transport is automatically inferred from URLs ending in `/sse`."
> "`MCPServerStreamableHTTP` is deprecated ... Use `MCPToolset('http://.../mcp')` instead — Streamable HTTP is the default for HTTP URLs."

### D4. `MCPServerHTTP` → `MCPServerSSE` → `MCPToolset`

> "The `MCPServerHTTP` class is deprecated, use `MCPServerSSE` instead."

Then apply D2.

### D5. `sse_read_timeout=` → `read_timeout=`

> "'sse_read_timeout' is deprecated, use 'read_timeout' instead."

### D6. `Agent.run_mcp_servers()` → `async with agent:`

> "`run_mcp_servers` is deprecated, use `async with agent:` instead. If you need to set a sampling model on all MCP servers, use `agent.set_mcp_sampling_model()`."

### D7. `load_mcp_servers(...)` → `load_mcp_toolsets(...)`

Symbol rename in `pydantic_ai.mcp`. v2 returns `MCPToolset` instances.

### D8. `MCPServerConfig` → inline `MCPToolset`

> "`pydantic_ai.mcp.MCPServerConfig` is deprecated and will be removed in v2. Pass the JSON config to `load_mcp_toolsets(...)` directly, or build `MCPToolset`s inline from `fastmcp.client.transports.StdioTransport` / URLs."

---

## E. A2A / AG-UI / Outlines / ACI

### E1. `Agent.to_a2a()` → external `fasta2a`

> "`Agent.to_a2a()` is deprecated and will be removed in 2.0. The `fasta2a` package is now maintained at https://github.com/datalayer/fasta2a — install it with the `pydantic-ai` extra (`pip install 'fasta2a[pydantic-ai]>=0.6.1'`) and use `from fasta2a.pydantic_ai import agent_to_a2a` directly."

Drop `pydantic-ai[fasta2a]`; depend on `fasta2a[pydantic-ai]>=0.6.1` directly.

```python
from fasta2a.pydantic_ai import agent_to_a2a
app = agent_to_a2a(agent)
```

### E2. `AGUIApp` / `Agent.to_ag_ui()` / `pydantic_ai.ag_ui` → `AGUIAdapter`

> "The `pydantic_ai.ag_ui` module is deprecated and will be removed in 2.0. Replace:" (multi-line; match the leading sentence)
> "`Agent.to_ag_ui()` is deprecated and will be removed in 2.0."
> "`AGUIApp` is deprecated and will be removed in 2.0."

`handle_ag_ui_request` and `run_ag_ui` are removed; call `AGUIAdapter.dispatch_request()` directly.

```python
from pydantic_ai.ui import StateDeps
from pydantic_ai.ui.ag_ui import AGUIAdapter
from starlette.applications import Starlette
from starlette.routing import Route

async def run_agent(request):
    return await AGUIAdapter.dispatch_request(request, agent=agent)

app = Starlette(routes=[Route('/', run_agent, methods=['POST'])])
```

### E3. `OutlinesModel` / `OutlinesProvider` → removed, no replacement

> "`OutlinesModel` is deprecated and will be removed in v2. If you would like to keep using Outlines with Pydantic AI, please file an issue at https://github.com/dottxt-ai/outlines/issues."

v2 removes `pydantic_ai.models.outlines` entirely. No in-tree replacement.

### E4. `pydantic_ai.ext.aci` → removed, wrap manually

> "`pydantic_ai.ext.aci` is deprecated and will be removed in 2.0. Wrap ACI.dev tools yourself using `pydantic_ai.tools.Tool.from_schema` against `aci.ACI().functions.get_definition(...)`, or use the upstream `aci-sdk` integration directly."

```python
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

### F1. `Usage` → `RunUsage`

> "`Usage` is deprecated, use `RunUsage` instead"

Per-request usage is now `RequestUsage`; per-run aggregate is `RunUsage`. Field renames:
- `request_tokens` → `input_tokens`
- `response_tokens` → `output_tokens`
- `request_tokens_limit` → `input_tokens_limit`
- `response_tokens_limit` → `output_tokens_limit`

### F2. `vendor_*` → `provider_*` on `ModelResponse`

- `vendor_details` → `provider_details`
- `vendor_id` → `provider_response_id`
- `provider_request_id` → `provider_response_id`

### F3. `FunctionTool{Call,Result}Event.call_id` → `.tool_call_id`

### F4. `price` → `cost` on usage

### F5. `FunctionToolResultEvent.result` → `.part`

---

## G. Built-in tools → `NativeTool` capability

> "`Agent(builtin_tools=...)` is deprecated, use `capabilities=[NativeTool(...)]` for raw native-tool registration, or a provider-adaptive capability like `WebSearch()`, `WebFetch()`, `MCP()`, or `ImageGeneration()` for native-or-local fallback."

v2 removes both `builtin_tools=` and `native_tools=` kwargs.

```python
from pydantic_ai.capabilities import NativeTool, WebSearch
Agent(..., capabilities=[NativeTool(WebSearchTool())])
# or for native-or-local fallback:
Agent(..., capabilities=[WebSearch()])
```

`WebSearchTool` / `WebFetchTool` are deprecated in favor of the `WebSearch` / `WebFetch` capabilities. See `BEHAVIOR_CHANGES.md` §3 for the local-fallback default flip.

---

## H. Output / toolsets / misc

### H1. `DeferredToolCalls` → `DeferredToolRequests`

> "`DeferredToolCalls` is deprecated, use `DeferredToolRequests` instead"

Field rename `.tool_calls` → `.calls`. v2 import: `from pydantic_ai.tools import DeferredToolRequests` (also re-exported from top-level). **Not** in `pydantic_ai.output`.

### H2. `DeferredToolset` → `ExternalToolset`

> "`DeferredToolset` is deprecated, use `ExternalToolset` instead"

### H3. `OpenAICompaction(instructions=)` — dropped, no replacement

> "`OpenAICompaction(instructions=...)` is deprecated and will be removed in a future version. OpenAI's `/compact` endpoint treats `instructions` as a system/developer message inserted into the compaction model's context, not as a directive for how to summarize the conversation, so this field does not match `AnthropicCompaction(instructions=...)` semantics."

Plain `DeprecationWarning`, not `PydanticAIDeprecationWarning`. Drop the kwarg; steer summarization via the surrounding agent prompt.

### H4. `parallel_execution_mode("sequential")`

> "Use `parallel_execution_mode(\"sequential\")` instead."

Trust the warning at the call site.

---

## I. `pydantic_graph`

### I1. Legacy `BaseNode`-runner imports

> "Importing `Graph` from `pydantic_graph` is deprecated. The `BaseNode`-based `Graph` runner and its persistence machinery are deprecated and will be removed (or repurposed) in v2; use the builder-based `GraphBuilder` API instead, or pin to `pydantic_graph<2` to keep using them."

Affected: `Graph`, `GraphRun`, `GraphRunResult`, `BaseNode`, `Edge`, `End`, `GraphRunContext`, `EndSnapshot`, `NodeSnapshot`, `Snapshot`, `FullStatePersistence`, `SimpleStatePersistence`.

Rewrite using `GraphBuilder` (see `docs/graph.md`), or pin `pydantic-graph<2`.

### I2. `pydantic_graph.beta.*` → top-level

> "`pydantic_graph.beta.decision` is deprecated, import from `pydantic_graph.decision` instead."

Drop the `.beta` segment.

---

## J. `pydantic_evals`

### J1. `Dataset(...)` without `name=`

> "Omitting the `name` parameter is deprecated. Please provide a name for your `Dataset`."

### J2. Positional construction / positional `Dataset.evaluate` kwargs

> "Constructing `EvaluationResult` with positional arguments is deprecated; use keyword arguments instead." (and `EvaluatorFailure`)
> "Passing `name`, `max_concurrency`, `progress`, `retry_task`, or `retry_evaluators` positionally to `Dataset.evaluate` / `Dataset.evaluate_sync` is deprecated; pass them as keyword arguments."

v2 raises `TypeError` for both. Switch to kwargs.

### J3. `evaluation_name` / `evaluator_version` class attributes

> "... relies on the `evaluation_name` attribute to customize the default evaluation name. This is deprecated; override `get_default_evaluation_name` in your evaluator class to retain this behavior in pydantic-evals v2."

Same for `evaluator_version` → override `get_evaluator_version`. Fires when the method is called, not on subclass creation — match on `relies on the \`evaluation_name\` attribute` / `relies on the \`evaluator_version\` attribute`.
