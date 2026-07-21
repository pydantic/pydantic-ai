# Capabilities

A capability is a reusable, composable unit of agent behavior. Instead of threading multiple arguments through your `Agent` constructor — [instructions](../agent.md#instructions) here, [model settings](../agent.md#model-run-settings) there, a [toolset](../toolsets.md) somewhere else, a [history processor](../message-history.md#processing-message-history) on yet another parameter — you can bundle related behavior into a single capability and pass it via the [`capabilities`][pydantic_ai.agent.Agent.__init__] parameter.

Capabilities can provide any combination of:

* **Tools** — via [toolsets](../toolsets.md) or [native tools](../native-tools.md)
* **Lifecycle hooks** — intercept and modify model requests, tool calls, and the overall run
* **Instructions** — static or dynamic [instruction](../agent.md#instructions) additions
* **Model settings** — static or per-step [model settings](../agent.md#model-run-settings)
* **Models** — static or adaptive model selection and application-specific model ID resolution

This makes them the primary extension point for Pydantic AI. Whether you're building a memory system, a guardrail, a cost tracker, or an approval workflow, a capability is the right abstraction.

Capabilities can be always-on or [loaded by the model on demand](on-demand.md). Pydantic AI ships the built-in capabilities below, [Pydantic AI Harness](#pydantic-ai-harness) and [third-party packages](third-party.md) provide many more, and [Building Custom Capabilities](custom.md) covers writing your own. To run agents durably across failures, restarts, and long waits, see [Durable Execution](../durable_execution/overview.md).

## Built-in capabilities

Pydantic AI ships with several capabilities that cover common needs:

| Capability | What it provides | Spec |
|---|---|:---:|
| [`Thinking`][pydantic_ai.capabilities.Thinking] | Enables model [thinking/reasoning](thinking.md) at configurable effort | Yes |
| [`Hooks`][pydantic_ai.capabilities.Hooks] | Decorator-based [lifecycle hook](../hooks.md) registration | — |
| [`Instrumentation`][pydantic_ai.capabilities.Instrumentation] | OpenTelemetry/Logfire [tracing](instrumentation.md) of runs, model requests, and tool calls | Yes |
| [`SelectModel`][pydantic_ai.capabilities.SelectModel] | Selects a static or per-step [model](select-model.md) with a callable | — |
| [`ResolveModelId`][pydantic_ai.capabilities.ResolveModelId] | Resolves custom [model IDs](resolve-model-id.md) with a callable | — |
| [`WebSearch`][pydantic_ai.capabilities.WebSearch] | [Web search](web-search.md) — native by default, optional [local fallback](../common-tools.md#duckduckgo-search-tool) via `local='duckduckgo'` | Yes |
| [`WebFetch`][pydantic_ai.capabilities.WebFetch] | [URL fetching](web-fetch.md) — native by default, optional [local fallback](../common-tools.md#web-fetch-tool) via `local=True` | Yes |
| [`ImageGeneration`][pydantic_ai.capabilities.ImageGeneration] | [Image generation](image-generation.md) — native by default, optional subagent fallback via `fallback_model` | Yes |
| [`XSearch`][pydantic_ai.capabilities.XSearch] | [X search](x-search.md) — native on xAI, explicit subagent fallback via `fallback_model` | Yes |
| [`MCP`][pydantic_ai.capabilities.MCP] | [MCP server](mcp.md) — runs locally by default; `native=True` opts into the model provider's native MCP support | Yes |
| [`ToolSearch`][pydantic_ai.capabilities.ToolSearch] | [Discovery](tool-search.md) of [deferred tools](../tools-advanced.md#tool-search) — native when supported, local `search_tools` function tool otherwise | Yes |
| [`PrepareTools`][pydantic_ai.capabilities.PrepareTools] | Filters or modifies function [tool definitions](prepare-tools.md) per step | — |
| [`PrepareOutputTools`][pydantic_ai.capabilities.PrepareOutputTools] | Filters or modifies [output tool][pydantic_ai.output.ToolOutput] [definitions](prepare-tools.md) per step | — |
| [`PrefixTools`][pydantic_ai.capabilities.PrefixTools] | Wraps a capability and [prefixes its tool names](prefix-tools.md) | Yes |
| [`NativeTool`][pydantic_ai.capabilities.NativeTool] | Registers a [native tool](../native-tools.md) with the agent | Yes |
| [`Capability`][pydantic_ai.capabilities.Capability] | Bundles instructions, function tools, and toolsets [without subclassing](on-demand.md#the-capability-convenience-class) | — |
| [`Toolset`][pydantic_ai.capabilities.Toolset] | Wraps an [`AbstractToolset`][pydantic_ai.toolsets.AbstractToolset] | — |
| [`IncludeToolReturnSchemas`][pydantic_ai.capabilities.IncludeToolReturnSchemas] | [Includes return type schemas](include-tool-return-schemas.md) in tool definitions sent to the model | Yes |
| [`SetToolMetadata`][pydantic_ai.capabilities.SetToolMetadata] | [Merges metadata key-value pairs](set-tool-metadata.md) onto selected tools | Yes |
| [`RaiseContentFilterError`][pydantic_ai.capabilities.RaiseContentFilterError] | [Raises](raise-content-filter-error.md) [`ContentFilterError`][pydantic_ai.exceptions.ContentFilterError] whenever a model response has `finish_reason='content_filter'` | Yes |
| [`ReinjectSystemPrompt`][pydantic_ai.capabilities.ReinjectSystemPrompt] | [Reinjects the configured system prompt](reinject-system-prompt.md) when the incoming message history is missing one | Yes |
| [`HandleDeferredToolCalls`][pydantic_ai.capabilities.HandleDeferredToolCalls] | Resolves [deferred tool calls](handle-deferred-tool-calls.md) inline with a handler function | — |
| [`ProcessHistory`][pydantic_ai.capabilities.ProcessHistory] | Wraps a [history processor](process-history.md) | — |
| [`ProcessEventStream`][pydantic_ai.capabilities.ProcessEventStream] | Forwards [agent stream events](process-event-stream.md) to a handler function | — |
| [`ThreadExecutor`][pydantic_ai.capabilities.ThreadExecutor] | Uses a [custom thread executor](thread-executor.md) for [sync functions](../tools-advanced.md#thread-executor-for-long-running-servers) | — |

The **Spec** column indicates whether the capability can be used in [agent specs](../agent-spec.md) (YAML/JSON). Capabilities marked **—** take non-serializable arguments (callables, toolset objects) and can only be used in Python code.

Provider-specific [compaction](compaction.md) capabilities ([`OpenAICompaction`][pydantic_ai.models.openai.OpenAICompaction], [`AnthropicCompaction`][pydantic_ai.models.anthropic.AnthropicCompaction]) live in the corresponding model modules. The [durable execution](../durable_execution/overview.md) integrations also ship as capabilities — [`TemporalDurability`][pydantic_ai.durable_exec.temporal.TemporalDurability], [`DBOSDurability`][pydantic_ai.durable_exec.dbos.DBOSDurability], and [`PrefectDurability`][pydantic_ai.durable_exec.prefect.PrefectDurability] — in the `pydantic_ai.durable_exec` subpackages.

```python {title="native_capabilities.py"}
from pydantic_ai import Agent
from pydantic_ai.capabilities import Thinking, WebSearch

agent = Agent(
    'anthropic:claude-opus-4-6',
    instructions='You are a research assistant. Be thorough and cite sources.',
    capabilities=[
        Thinking(effort='high'),
        WebSearch(local='duckduckgo'),
    ],
)
```

[Instructions](../agent.md#instructions) and [model settings](../agent.md#model-run-settings) are configured directly via the `instructions` and `model_settings` parameters on `Agent` (or [`AgentSpec`][pydantic_ai.agent.AgentSpec]). Capabilities are for behavior that goes beyond simple configuration — tools, lifecycle hooks, and custom extensions. They compose well, especially when you want to reuse the same configuration across multiple agents or load it from a [spec file](../agent-spec.md).

## Provider-adaptive tools

[`WebSearch`][pydantic_ai.capabilities.WebSearch], [`WebFetch`][pydantic_ai.capabilities.WebFetch], [`ImageGeneration`][pydantic_ai.capabilities.ImageGeneration], [`XSearch`][pydantic_ai.capabilities.XSearch], and [`MCP`][pydantic_ai.capabilities.MCP] each cover a single capability (web search, URL fetch, image generation, X search, MCP) across two implementations:

- **Native** — invoked by the model provider when the model supports it. The work happens on the provider's side (e.g. Anthropic's web search runs server-side, returning results inline).
- **Local** — runs in your Python process. Used when the model doesn't support the native tool; your code does the work (e.g. calling DuckDuckGo directly).

| Capability | Local fallback | Notes |
|---|---|---|
| [`WebSearch`][pydantic_ai.capabilities.WebSearch] | `local='duckduckgo'` or `local=True` (DuckDuckGo) | Requires the `duckduckgo` optional group |
| [`WebFetch`][pydantic_ai.capabilities.WebFetch] | `local=True` (markdownify-based fetch) | Requires the `web-fetch` optional group |
| [`ImageGeneration`][pydantic_ai.capabilities.ImageGeneration] | Subagent via `fallback_model=` | Delegates to a model that supports native image generation |
| [`XSearch`][pydantic_ai.capabilities.XSearch] | Subagent via `fallback_model=` | No default non-xAI fallback; set `fallback_model` to an xAI model that supports [`XSearchTool`][pydantic_ai.native_tools.XSearchTool] |
| [`MCP`][pydantic_ai.capabilities.MCP] | Direct connection to the MCP server (the default) | Accepts any [`MCPToolset`][pydantic_ai.mcp.MCPToolset] input; transport is auto-detected from a URL |

Because these capabilities contribute model-facing tools, their `id`, `description`, and `defer_loading` fields are meaningful: set them when that tool should stay hidden until the model loads the matching workflow with the `load_capability` tool. This includes [`ImageGeneration`][pydantic_ai.capabilities.ImageGeneration] when image generation should only be available for an image-specific workflow, whether it resolves to a native image tool or a fallback subagent tool.

Configure each side via the `native=` and `local=` kwargs. `native=` accepts `True` (use the capability's default [native tool](../native-tools.md) instance), `False` (disable native), or an explicit instance like `WebSearchTool(...)` for fine-grained config. `local=` accepts `True` (the bundled local fallback, on capabilities that have one — `WebSearch` and `WebFetch`), `False` (disable local), a named strategy string where supported, or any callable, [`Tool`][pydantic_ai.tools.Tool], or [`AbstractToolset`][pydantic_ai.toolsets.AbstractToolset]. Optional installs needed for the local fallback are opt-in — the capability raises a [`UserError`][pydantic_ai.exceptions.UserError] at construction (with an install hint) when you ask for a local strategy whose extra isn't installed.

```python {title="provider_adaptive_tools.py" test="skip" lint="skip"}
from pydantic_ai import Agent
from pydantic_ai.capabilities import MCP, ImageGeneration, WebFetch, WebSearch, XSearch

agent = Agent(
    'anthropic:claude-sonnet-4-6',
    capabilities=[
        # Native when supported; DuckDuckGo fallback on unsupported models
        WebSearch(local='duckduckgo'),
        # Native when supported; markdownify-based fallback on unsupported models
        WebFetch(local=True),
        # Native when supported; subagent fallback via `fallback_model`
        ImageGeneration(fallback_model='openai-responses:gpt-5.4'),
        # Native on xAI; on other models, explicitly delegate to an xAI model
        XSearch(fallback_model='xai:grok-4.3'),
        # Runs the MCP server locally by default; pass `native=True` to also advertise native MCP
        MCP('https://mcp.example.com/api'),
    ],
)
```

`MCP` defaults the other way from the others: because MCP carries credentials, it runs locally by default and you opt into native MCP with `native=True`. The others default to native and you opt into local with `local=`.

[`XSearch`][pydantic_ai.capabilities.XSearch] is slightly different from [`WebSearch`][pydantic_ai.capabilities.WebSearch] and [`WebFetch`][pydantic_ai.capabilities.WebFetch]: there is no default non-xAI fallback. If your agent is not running on an xAI model, set `fallback_model` explicitly to an xAI model that supports [`XSearchTool`][pydantic_ai.native_tools.XSearchTool].

Some constraint fields require the native tool (the bundled local fallback can't enforce them) — passing them locks the capability to the native path. If the model doesn't support the native tool, the capability raises a [`UserError`][pydantic_ai.exceptions.UserError].

```python {title="constraints.py" test="skip" lint="skip"}
# Limit to 5 searches per run — requires native (the local fallback can't track call count)
WebSearch(max_uses=5)

# Only fetch example.com — enforced locally when native is unavailable
WebFetch(allowed_domains=['example.com'], local=True)
```

### Building your own

All five capabilities are subclasses of [`NativeOrLocalTool`][pydantic_ai.capabilities.NativeOrLocalTool], which you can use directly or subclass to build your own provider-adaptive tools. For example, to pair [`CodeExecutionTool`][pydantic_ai.native_tools.CodeExecutionTool] with a local fallback:

```python {title="custom_native_or_local.py" test="skip" lint="skip"}
from pydantic_ai.native_tools import CodeExecutionTool
from pydantic_ai.capabilities import NativeOrLocalTool

cap = NativeOrLocalTool(native=CodeExecutionTool(), local=my_local_executor)
```

## Pydantic AI Harness

[**Pydantic AI Harness**](https://pydantic.dev/docs/ai/harness/) is the official capability library for Pydantic AI -- standalone capabilities like memory, guardrails, context management, and [code mode](https://github.com/pydantic/pydantic-ai-harness/tree/main/pydantic_ai_harness/code_mode) live there rather than in core. See [What goes where?](https://pydantic.dev/docs/ai/harness/#what-goes-where) for the full breakdown, or jump to the [capability matrix](https://github.com/pydantic/pydantic-ai-harness#capability-matrix).

## Third-party capabilities

Third-party packages publish capabilities of their own — see [Third-Party Capabilities](third-party.md) for the ecosystem, and [Publishing capabilities](custom.md#publishing-capabilities) for making your own capability available to others.
