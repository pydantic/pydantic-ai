# Timeouts

Bounding how long one step inside a run may take, and ending a run from inside a tool, are answered by separate mechanisms with separate failure modes. This page maps them. To stop a run that is already in flight, see [Cancelling a Run](agent.md#cancelling-a-run).

## Bounding how long a step takes

Each knob below bounds a different unit of work. None of them bounds the wall-clock duration of a whole run.

| What you want to bound | How to set it | What happens on expiry |
|---|---|---|
| A single model request attempt — a provider SDK client's retries re-arm it for every attempt | `timeout` on [`ModelSettings`][pydantic_ai.settings.ModelSettings] | The provider client raises; the run fails unless a [`FallbackModel`](models/overview.md#fallback-model) or a [transport retry](models/http-request-retries.md) handles it |
| A function tool call | `Agent(tool_timeout=...)`, or `timeout=` on an individual tool — see [Tool Timeout](tools-advanced.md#tool-timeout) | The model receives a retry prompt `'Timed out after N seconds.'`, consuming that tool's [retry budget](retries.md#tool-retries). A `def` tool is not actually stopped: the deadline is enforced around the await, so the worker thread runs to completion |
| A [hook](hooks.md) function | `timeout=` on the `@hooks.on.*` decorator | [`HookTimeoutError`][pydantic_ai.capabilities.HookTimeoutError], which is an [`AgentRunError`][pydantic_ai.exceptions.AgentRunError] and aborts the run. Like a `def` tool, a `def` hook is not actually stopped: the worker thread runs to completion |
| Connecting to an MCP server | `MCPToolset(init_timeout=...)`, default `5` seconds | The connection and `initialize` handshake fail |
| A single MCP request | `MCPToolset(read_timeout=...)`, default `300` seconds | The request fails; under the default [`tool_error_behavior='retry'`](mcp/client.md#tool-errors) the model sees it as a retryable tool error |
| Opening a [realtime session](realtime/overview.md) | `handshake_timeout` on [`RealtimeModelSettings`][pydantic_ai.realtime.RealtimeModelSettings], default `30` seconds — OpenAI, Azure OpenAI, and xAI | Opening the session raises [`RealtimeError`][pydantic_ai.realtime.RealtimeError]. On a reconnect it consumes a [`ReconnectPolicy`][pydantic_ai.realtime.ReconnectPolicy] attempt instead |
| Total work done by a run | [`UsageLimits`][pydantic_ai.usage.UsageLimits] — requests, tool calls, tokens, or cost — see [Usage Limits](agent.md#usage-limits) | [`UsageLimitExceeded`][pydantic_ai.exceptions.UsageLimitExceeded] |
| Wall-clock duration of a whole run | Nothing built in — wrap `agent.run()` in `asyncio.timeout` (Python 3.11+) or `anyio.fail_after()`, or cancel a [`CancellationToken`][pydantic_ai.CancellationToken] from a timer | The run is [cancelled](agent.md#cancelling-a-run) |

Two of these need qualifying:

- **`ModelSettings['timeout']` is applied per model class, not universally.** The model classes that forward it to their provider client are listed under [`ModelSettings.timeout`][pydantic_ai.settings.ModelSettings.timeout]; the ones built on OpenAI's inherit the forwarding from [`OpenAIChatModel`][pydantic_ai.models.openai.OpenAIChatModel] / [`OpenAIResponsesModel`][pydantic_ai.models.openai.OpenAIResponsesModel]. Other model classes ignore the setting, and the timeout on the HTTP client they were built with applies instead. When Pydantic AI creates that client itself, it defaults to a 600-second total timeout with a 5-second connect timeout. Google and Mistral additionally reject an `httpx.Timeout` object and accept only a number of seconds.

    To bound a request on a model class that ignores the setting, configure the timeout where that provider actually takes one. Most providers accept your own `http_client`, but several don't: [`XaiProvider`][pydantic_ai.providers.xai.XaiProvider] takes a client-level `timeout` (or a preconfigured `xai_client`), [`BedrockProvider`][pydantic_ai.providers.bedrock.BedrockProvider] takes `aws_read_timeout` and `aws_connect_timeout` (or a preconfigured `bedrock_client`), and [`HuggingFaceProvider`][pydantic_ai.providers.huggingface.HuggingFaceProvider] rejects `http_client` outright in favor of `hf_client`.
- **Tool timeouts are enforced by [`FunctionToolset`][pydantic_ai.toolsets.FunctionToolset] only, and each toolset carries its own.** `Agent(tool_timeout=...)` sets the default for tools you register *on the agent* — it does not reach into a `FunctionToolset` you constructed yourself and passed via `toolsets=[...]`. Give that toolset its own `FunctionToolset(timeout=...)`, or set `timeout=` on the individual tools. Tools coming from an [MCP server](mcp/client.md), an [external toolset](deferred-tools.md), or a custom [`AbstractToolset`][pydantic_ai.toolsets.AbstractToolset] read neither; bound those with the server-side or transport-level timeout instead.

If you enforce a deadline inside a tool body yourself, catch the `TimeoutError` and re-raise it as [`ModelRetry`][pydantic_ai.exceptions.ModelRetry] or [`ToolFailed`][pydantic_ai.exceptions.ToolFailed] rather than letting it escape. What happens to a bare `TimeoutError` depends on whether that tool has a timeout of its own:

- **No `timeout` on the tool or its toolset.** It is an ordinary exception and propagates out of the agent run — unless a [capability](capabilities/overview.md) implements `on_tool_execute_error`, which can turn it into a replacement tool result or a `ModelRetry`.
- **A `timeout` is configured.** The call runs inside `anyio.fail_after(timeout)`, which signals expiry with `TimeoutError` too, so a `TimeoutError` you raised yourself is indistinguishable from the deadline expiring and becomes the same `'Timed out after N seconds.'` retry prompt — reporting a deadline that may never have passed.

Re-raising in the tool is the more local choice; the hook is for applying one policy across every tool.

## Ending a run from inside a tool

What a tool raises decides whether the run continues, and what the model gets to see:

| Raise | Run continues? | The model sees |
|---|---|---|
| [`ModelRetry`][pydantic_ai.exceptions.ModelRetry] | Yes | A [retry prompt](retries.md#tool-retries) asking it to correct the call — consumes that tool's retry budget |
| [`ToolFailed`][pydantic_ai.exceptions.ToolFailed] | Yes | A [failed tool result](tools-advanced.md#tool-failed) to adapt to — does not consume the retry budget |
| [`ApprovalRequired`][pydantic_ai.exceptions.ApprovalRequired] / [`CallDeferred`][pydantic_ai.exceptions.CallDeferred] | Ends the run with a [`DeferredToolRequests`][pydantic_ai.tools.DeferredToolRequests] output, unless a [`HandleDeferredToolCalls`][pydantic_ai.capabilities.HandleDeferredToolCalls] handler resolves the call inline | Nothing yet — see [Deferred Tools](deferred-tools.md) |
| Any other exception | No | By default nothing — it propagates out of `agent.run()`. A [capability](capabilities/overview.md) implementing `on_tool_execute_error` sees it first and can return a replacement tool result or raise `ModelRetry`, letting the run continue |

The deferred row reads differently inside a [realtime session](realtime/overview.md), which has no way to pause: a live conversation can't wait for an out-of-band result. A `HandleDeferredToolCalls` handler still gets the chance to resolve the call inline, but where a run would end with a `DeferredToolRequests` output, a session instead answers the model with an explanation that the tool can't complete during the session, and keeps going. See [Deferred and approval-required tools](realtime/tools.md#deferred-and-approval-required-tools).

A tool can also end the run without raising, by calling [`RunContext.cancel()`][pydantic_ai.tools.RunContext.cancel] — the run ends with [`RunCancelled`][pydantic_ai.exceptions.RunCancelled] and the tool's return value is discarded. See [Cancelling the Run from a Tool](tools-advanced.md#cancelling-the-run-from-a-tool).

There is no exception that ends a run early with a *successful* output. To let a tool finish the run with a value, make that value the run's output: give the agent an [output tool](output.md#tool-output) the model can call, or an [output function](output.md#output-functions) that produces the result.
