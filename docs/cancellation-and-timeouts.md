# Cancellation and Timeouts

Three different questions tend to be asked with the same word. Stopping a run that is already in flight, bounding how long one step inside it may take, and ending a run from inside a tool are answered by separate mechanisms with separate failure modes. This page maps them.

## Cancelling a run

There is no `cancel()` method on an agent run. [`agent.run()`][pydantic_ai.agent.AbstractAgent.run] is an ordinary coroutine, so you stop it the way you stop any `asyncio` task: cancel the task, or wrap the call in a deadline.

```python {title="cancel_run.py"}
import asyncio

from pydantic_ai import (
    Agent,
    ModelMessage,
    ModelResponse,
    ToolCallPart,
    capture_run_messages,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel

tool_started = asyncio.Event()


async def call_slow_lookup(
    messages: list[ModelMessage], info: AgentInfo
) -> ModelResponse:
    return ModelResponse(parts=[ToolCallPart('slow_lookup', {})])


agent = Agent(FunctionModel(call_slow_lookup))


@agent.tool_plain
async def slow_lookup() -> str:
    tool_started.set()
    await asyncio.sleep(60)
    return 'unreachable'


async def main():
    with capture_run_messages() as messages:  # (1)!
        task = asyncio.create_task(agent.run('Look up the answer'))
        await tool_started.wait()
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            print('run cancelled')
            #> run cancelled

    print([type(m).__name__ for m in messages])  # (2)!
    #> ['ModelRequest', 'ModelResponse']
```

1. A cancelled run raises instead of returning an [`AgentRunResult`][pydantic_ai.run.AgentRunResult], so [`capture_run_messages()`][pydantic_ai.capture_run_messages] is how you read back what happened.
2. The prompt and the model's tool call were recorded before the cancellation; the tool never returned, so nothing was recorded for it.

_(This example is complete, it can be run "as is" — you'll need to add `asyncio.run(main())` to run `main`)_

To put a deadline on a run rather than cancelling it on a signal, wrap the same `await agent.run(...)` in `asyncio.timeout()` (Python 3.11+) or `anyio.fail_after()`. Both cancel the task when the deadline passes, so everything below applies unchanged.

While the run unwinds, Pydantic AI guarantees that:

- Toolsets entered for the run are exited, so MCP sessions and other resources are closed.
- Tool calls still running in parallel are cancelled and then awaited, so their own `finally` blocks complete before the cancellation propagates to your code. This applies to `async def` tools. A `def` tool already executing runs in a worker thread, and Python cannot interrupt a function running in a thread, so its side effects still happen either way — but whether the cancellation *waits* depends on which executor is in play:
    - By default (`anyio.to_thread.run_sync`) the cancellation waits for the tool to return.
    - Under [`Agent.using_thread_executor()`][pydantic_ai.agent.AbstractAgent.using_thread_executor] the cancellation returns immediately while the tool keeps running to completion in the background. If a `def` tool holds a resource that must be released before your code moves on, don't rely on the executor to wait for it.
- The [capability](capabilities/overview.md) hooks `wrap_run` and `wrap_node_run` receive the `asyncio.CancelledError`, which makes them the place to put cancellation-safe cleanup. `after_node_run` is not called for a node interrupted by cancellation, including a cancellation the node absorbed itself.
- A step that swallows the cancellation does not silently resurrect the run: once that step's messages have been recorded, the cancellation is re-raised so the run still ends. This relies on `asyncio.Task.cancelling()`, so on Python 3.10 it is a no-op and an absorbed cancellation lets the run continue.

!!! warning "`run_sync()` cannot be cancelled"
    [`run_sync()`][pydantic_ai.agent.AbstractAgent.run_sync] blocks the calling thread and drives its own event loop, so there is no task for a caller to cancel — only `KeyboardInterrupt` interrupts it. Use `await agent.run(...)` when the run needs to be cancellable.

### Message history after cancellation

The messages captured by [`capture_run_messages()`][pydantic_ai.capture_run_messages] include the partial work of the interrupted step:

- A model response that was still streaming is recorded with its partial content and a non-`complete` [`state`][pydantic_ai.messages.ModelResponse.state].
- If some tool calls in the step had already returned, their results are recorded in a [`ModelRequest`][pydantic_ai.messages.ModelRequest] marked `state='interrupted'`. That request holds only the tool returns, because it was never sent to the model.

This history can be passed straight into a follow-up run: before the next model request, Pydantic AI [repairs the transcript](message-history.md#making-histories-provider-valid), answering any tool call that never received a result — including one whose arguments were cut off mid-stream.

## Cancelling a stream

Streaming has an explicit cancellation handle that run-level cancellation does not: [`StreamedRunResult.cancel()`][pydantic_ai.result.StreamedRunResult.cancel] and [`AgentStream.cancel()`][pydantic_ai.result.AgentStream.cancel] close the model stream so the provider stops generating tokens. [Cancelling Streams](output.md#cancelling-streams) covers all three streaming APIs with examples, including which model integrations can interrupt an in-flight chunk read.

How you stop consuming determines the [`state`][pydantic_ai.messages.ModelResponse.state] the response is recorded with, which is what downstream code should branch on:

| How you stopped | Recorded `state` |
|---|---|
| Drained the stream to completion | `complete` |
| `await result.cancel()` mid-stream | `interrupted` |
| `cancel()` after the stream had already finished | `complete` — a defensive cancel can't downgrade a finished response |
| Broke out of the iterator, then left the `async with` block normally | `incomplete` |
| An exception or cancellation propagated out of the `async with` block | `interrupted` |

Either way the underlying HTTP response is closed when the block exits; `cancel()` is what stops generation *immediately* rather than only stopping local consumption.

## Bounding how long a step takes

Each knob below bounds a different unit of work. None of them bounds the wall-clock duration of a whole run.

| What you want to bound | How to set it | What happens on expiry |
|---|---|---|
| A single model request | `timeout` on [`ModelSettings`][pydantic_ai.settings.ModelSettings] | The provider client raises; the run fails unless a [`FallbackModel`](models/overview.md#fallback-model) or a [transport retry](models/http-request-retries.md) handles it |
| A function tool call | `Agent(tool_timeout=...)`, or `timeout=` on an individual tool — see [Tool Timeout](tools-advanced.md#tool-timeout) | The model receives a retry prompt `'Timed out after N seconds.'`, consuming that tool's [retry budget](retries.md#tool-retries). A `def` tool is not actually stopped: the deadline is enforced around the await, so the worker thread runs to completion |
| A [hook](hooks.md) function | `timeout=` on the `@hooks.on.*` decorator | [`HookTimeoutError`][pydantic_ai.capabilities.HookTimeoutError], which is an [`AgentRunError`][pydantic_ai.exceptions.AgentRunError] and aborts the run |
| Connecting to an MCP server | `MCPToolset(init_timeout=...)`, default `5` seconds | The connection and `initialize` handshake fail |
| A single MCP request | `MCPToolset(read_timeout=...)`, default `300` seconds | The request fails; under the default [`tool_error_behavior='retry'`](mcp/client.md#tool-errors) the model sees it as a retryable tool error |
| Total work done by a run | [`UsageLimits`][pydantic_ai.usage.UsageLimits] — requests, tool calls, tokens, or cost — see [Usage Limits](agent.md#usage-limits) | [`UsageLimitExceeded`][pydantic_ai.exceptions.UsageLimitExceeded] |
| Wall-clock duration of a whole run | Nothing built in — wrap `agent.run()` in `asyncio.timeout` (Python 3.11+) or `anyio.fail_after()` | The run is cancelled, as described [above](#cancelling-a-run) |

Two of these need qualifying:

- **`ModelSettings['timeout']` is applied per model class, not universally.** The OpenAI, Anthropic, Google, Groq, and Mistral model classes forward it to their provider client, as do the model classes built on OpenAI's — `CerebrasModel`, `OllamaModel`, `OpenRouterModel`, `ZaiModel`, and the Bedrock Mantle models — which inherit the forwarding from [`OpenAIChatModel`][pydantic_ai.models.openai.OpenAIChatModel] / [`OpenAIResponsesModel`][pydantic_ai.models.openai.OpenAIResponsesModel]. Other model classes ignore the setting, and the timeout on the HTTP client they were built with applies instead. When Pydantic AI creates that client itself, it defaults to a 600-second total timeout with a 5-second connect timeout. Google and Mistral additionally reject an `httpx.Timeout` object and accept only a number of seconds.

    To bound a request on a model class that ignores the setting, configure the timeout where that provider actually takes one. Most providers accept your own `http_client`, but two don't: [`BedrockProvider`][pydantic_ai.providers.bedrock.BedrockProvider] takes `aws_read_timeout` and `aws_connect_timeout` (or a preconfigured `bedrock_client`), and [`HuggingFaceProvider`][pydantic_ai.providers.huggingface.HuggingFaceProvider] rejects `http_client` outright in favor of `hf_client`.
- **Tool timeouts are enforced by [`FunctionToolset`][pydantic_ai.toolsets.FunctionToolset] only, and each toolset carries its own.** `Agent(tool_timeout=...)` sets the default for tools you register *on the agent* — it does not reach into a `FunctionToolset` you constructed yourself and passed via `toolsets=[...]`. Give that toolset its own `FunctionToolset(timeout=...)`, or set `timeout=` on the individual tools. Tools coming from an [MCP server](mcp/client.md), an [external toolset](deferred-tools.md), or a custom [`AbstractToolset`][pydantic_ai.toolsets.AbstractToolset] read neither; bound those with the server-side or transport-level timeout instead.

If you enforce a deadline inside a tool body yourself, catch the `TimeoutError` and re-raise it as [`ModelRetry`][pydantic_ai.exceptions.ModelRetry] or [`ToolFailed`][pydantic_ai.exceptions.ToolFailed]. A bare `TimeoutError` is an ordinary exception, so by default it propagates out of the agent run — unless a [capability](capabilities/overview.md) implements `on_tool_execute_error`, which can turn it into a replacement tool result or a `ModelRetry`. Re-raising in the tool is the more local choice; the hook is for applying one policy across every tool.

## Ending a run from inside a tool

What a tool raises decides whether the run continues, and what the model gets to see:

| Raise | Run continues? | The model sees |
|---|---|---|
| [`ModelRetry`][pydantic_ai.exceptions.ModelRetry] | Yes | A [retry prompt](retries.md#tool-retries) asking it to correct the call — consumes that tool's retry budget |
| [`ToolFailed`][pydantic_ai.exceptions.ToolFailed] | Yes | A [failed tool result](tools-advanced.md#tool-failed) to adapt to — does not consume the retry budget |
| [`ApprovalRequired`][pydantic_ai.exceptions.ApprovalRequired] / [`CallDeferred`][pydantic_ai.exceptions.CallDeferred] | Ends the run with a [`DeferredToolRequests`][pydantic_ai.tools.DeferredToolRequests] output, unless a [`HandleDeferredToolCalls`][pydantic_ai.capabilities.HandleDeferredToolCalls] handler resolves the call inline | Nothing yet — see [Deferred Tools](deferred-tools.md) |
| Any other exception | No | By default nothing — it propagates out of `agent.run()`. A [capability](capabilities/overview.md) implementing `on_tool_execute_error` sees it first and can return a replacement tool result or raise `ModelRetry`, letting the run continue |

There is no exception that ends a run early with a successful output. To let a tool finish the run with a value, make that value the run's output: give the agent an [output tool](output.md#tool-output) the model can call, or an [output function](output.md#output-functions) that produces the result.
