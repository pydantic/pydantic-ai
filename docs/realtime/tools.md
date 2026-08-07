# Tools and capabilities

Tools registered on an agent are offered to the realtime model and execute on your backend. The
session validates arguments, applies retries, runs tools concurrently, returns results to the
provider, and records ordinary tool-call messages for later handoff.

## Function tools

When a model calls a tool, the session emits
[`FunctionToolCallEvent`][pydantic_ai.messages.FunctionToolCallEvent], runs the tool, returns the
result, and emits [`FunctionToolResultEvent`][pydantic_ai.messages.FunctionToolResultEvent]. Parse
failures and [`ModelRetry`][pydantic_ai.exceptions.ModelRetry] produce a
[`RetryPromptPart`][pydantic_ai.messages.RetryPromptPart], matching a standard agent run. Other tool
exceptions end the session and propagate from iteration.

Provider tool-output channels accept strings, but local history preserves the structured
`return_value`, `content`, and `metadata` of a
[`ToolReturn`][pydantic_ai.messages.ToolReturn]. If the provider cancels an in-flight call, Pydantic
AI cancels the task and records a synthetic cancellation result locally without sending that result
back to the provider.

### Concurrent tool execution

Every tool runs in the background, so a slow tool does not block session events, other tools, or
turn tracking. [`all_messages()`][pydantic_ai.realtime.RealtimeSession.all_messages] keeps each
result adjacent to its call even when calls finish out of order.

Whether the model continues speaking while it waits is provider-specific. Inspect
[`supports_async_tool_calls`][pydantic_ai.realtime.RealtimeModelProfile.supports_async_tool_calls].
OpenAI and Azure models generally fill the gap; Gemini pauses unless
[`google_async_tool_calls`](gemini.md#asynchronous-tool-calls) is enabled on a supported model.

## Native tools

Provider-native tools execute server-side. Add them through high-level capabilities such as
[`WebSearch`][pydantic_ai.capabilities.WebSearch] and
[`WebFetch`][pydantic_ai.capabilities.WebFetch], or through
[`NativeTool`][pydantic_ai.capabilities.NativeTool]. Each model's
[`supported_native_tools`][pydantic_ai.realtime.RealtimeModelProfile.supported_native_tools] profile
is the source of truth.

```python
from pydantic_ai import Agent
from pydantic_ai.capabilities import WebSearch
from pydantic_ai.messages import NativeToolReturnPart, PartEndEvent
from pydantic_ai.realtime.google import GoogleRealtimeModel

agent = Agent(instructions='Answer questions, searching the web when useful.')


async def main():
    async with agent.realtime(
        GoogleRealtimeModel('gemini-2.5-flash-native-audio-latest'),
        capabilities=[WebSearch()],
    ).session() as session:
        async for event in session:
            if isinstance(event, PartEndEvent) and isinstance(event.part, NativeToolReturnPart):
                print(event.part.content)
```

An unsupported native tool with a configured local fallback is replaced before connection. Without
a fallback, opening the session raises [`UserError`][pydantic_ai.exceptions.UserError]. Provider and
model-specific combinations—including Gemini grounding, URL context, and function-tool
restrictions—are canonical on the [Gemini provider page](gemini.md#native-tools).

## Deferred and approval-required tools

**An approval-gated tool is not executable in a realtime session unless you install a
[`HandleDeferredToolCalls`][pydantic_ai.capabilities.HandleDeferredToolCalls] handler.** A standard
run can end with a `DeferredToolRequests` output and resume once a human answers; a live conversation
has nowhere to pause, so with no handler the call is refused *every time* — the model receives an
explanation that the tool cannot complete during a realtime session, and the tool never runs.
Installing a handler is the expected setup for approval-gated tools in a realtime session, not an
optional extra.

The handler resolves each call inline: approve it (the tool then runs and returns normally), deny it
(recorded with `outcome='denied'`), substitute a result, or request a retry.

!!! warning "The handler answers from policy, not from a person"
    The handler must return a decision promptly — it is a programmatic policy resolver, not an approval
    UI. It runs as a background task like the tool itself, so it never blocks the session's events, but
    what the *conversation* does while it thinks is provider-specific in exactly the way
    [concurrent tool execution](#concurrent-tool-execution) describes: OpenAI and Azure carry on, while
    Gemini holds the model's turn until the result arrives. On Gemini a slow handler therefore reads as
    assistant silence, and if the user speaks into that gap the provider cancels the pending call
    outright (recorded as [a synthetic cancellation](#function-tools)).

    Asking a human mid-call and resuming on their answer is not supported yet; a standard run, which
    can end with a [`DeferredToolRequests`][pydantic_ai.tools.DeferredToolRequests] output and resume
    later, is the place for that today.

    [`DeferredToolRequestsEvent`][pydantic_ai.messages.DeferredToolRequestsEvent] on a session is
    informational for the same reason: it is emitted when the handler *has* resolved the calls, so a
    consumer can observe what was asked and decided. It is not a hook to respond to — unlike the same
    event in a standard run, nothing waits for the consumer, and no event is emitted when no handler is
    installed and the call is refused.

This applies to both ways a call is deferred — raising
[`ApprovalRequired`][pydantic_ai.exceptions.ApprovalRequired] or
[`CallDeferred`][pydantic_ai.exceptions.CallDeferred] from the tool, and declaring it up front with
`requires_approval=True` or an [external toolset](../toolsets.md#external-toolset). An approval-gated
tool is still advertised to the model, exactly as in a standard run; calling it opens the approval
flow rather than running the tool.

A realtime session cannot pause and return a `DeferredToolRequests` output for an out-of-band
result. Resolve the request during the call or move that workflow to a standard agent run.

A session's tools are fixed when the connection opens, so nothing can reveal a tool mid-call. Tools
registered with `defer_loading=True` for [tool search](../tools-advanced.md#tool-search), and
[capabilities](../capabilities/overview.md) with `defer_loading=True` that contribute tools or native
tools, are therefore rejected with [`UserError`][pydantic_ai.exceptions.UserError] when the session
opens — rather than advertising a search that could find the tool but never deliver it. Register
those tools normally for a realtime session, or keep that workflow in a standard agent run.

## Enqueuing prompts from tools

[`RunContext.enqueue()`][pydantic_ai.tools.RunContext.enqueue] accepts one plain-text prompt per
call from a realtime tool. The default `priority='asap'` sends it when no response is active;
`priority='when_idle'` waits until the provider reports its current response complete. Neither
priority interrupts assistant speech. Delivered prompts become ordinary user turns in history.

Multimodal content and prebuilt message/part sequences are rejected because the realtime live-input
channel cannot preserve their standard-run semantics.

## Capabilities and hooks

A [capability][pydantic_ai.capabilities.AbstractCapability] attached to the agent or passed to
`realtime(capabilities=...)` participates where its lifecycle maps onto a persistent session:

| Capability stage | Session behavior |
| --- | --- |
| `for_agent`, `for_run`, `get_instructions` | Runs during setup; dynamic instructions are evaluated once at connect. |
| `get_toolset`, `get_wrapper_toolset`, `prepare_tools` | Contributes, wraps, and prepares local tools before connecting. |
| `get_native_tools` | Contributes concrete native tools before connecting; dynamic selectors do not apply. |
| Tool validation/execution hooks | Runs around each local function-tool call. |
| `handle_deferred_tool_calls` | Resolves deferred requests inline. |
| Graph node, model-request, and output-processing hooks | Do not run; no agent graph or output-processing stage exists. |
| `before_run`, `after_run`, `wrap_run`, `on_run_error` | Run once around the session, with the same close-boundary recovery and result-transformation semantics as `iter()`; a realtime session is a run. |
| `wrap_run_event_stream` | Wraps the consumer-facing session iterator. It can observe or transform shared [`AgentStreamEvent`][pydantic_ai.messages.AgentStreamEvent] members and realtime-only [`RealtimeEvent`][pydantic_ai.realtime.RealtimeEvent] members without changing history or tool execution. `event_stream_handler` works through the same stream. |

`get_model_settings()` may run during capability setup, but regular model settings do not configure
a realtime model. Pass [`RealtimeModelSettings`][pydantic_ai.realtime.RealtimeModelSettings] through
`realtime(model_settings=...)` instead. Inside session hooks and tools,
[`ctx.model_settings`][pydantic_ai.tools.RunContext.model_settings] holds the merged
[`RealtimeModelSettings`][pydantic_ai.realtime.RealtimeModelSettings] the session was connected with,
[`ctx.realtime`][pydantic_ai.tools.RunContext.realtime] is `True` from `before_run` onward, and
[`ctx.realtime_session`][pydantic_ai.tools.RunContext.realtime_session] exposes the live
[`RealtimeSession`][pydantic_ai.realtime.RealtimeSession] once it is connected (it is still `None` in
`before_run`, in instruction functions, and in the pre-handler part of `wrap_run`, which all run
before the connection is established).

Deferred capabilities load in a session the same way they do in a regular run: the capability
catalog is part of the session's instructions, and calling the `load_capability` tool returns the
loaded capability's instructions as its result — which works on every provider. What a session
cannot do is advertise *new tools* mid-conversation (the connection's tools are fixed when it
opens), so opening a session with a `defer_loading=True` capability that contributes tools or
native tools raises [`UserError`][pydantic_ai.exceptions.UserError] before connecting — accepting
it would silently provide less than requested. Realtime per-turn/exchange hooks are expected to
widen this boundary in the future; see
[#7190](https://github.com/pydantic/pydantic-ai/issues/7190) and
[#7191](https://github.com/pydantic/pydantic-ai/issues/7191). History-processing capabilities also
do not transform seeded history.

## Delegating work during a call

Realtime models do not provide structured output and can be weaker at complex reasoning than a
frontier text model. Expose a tool that delegates the hard work to a standard
[`Agent`][pydantic_ai.Agent] with an `output_type`:

```python
from pydantic import BaseModel

from pydantic_ai import Agent
from pydantic_ai.realtime.openai import OpenAIRealtimeModel


class Answer(BaseModel):
    summary: str
    confidence: float


supervisor = Agent('openai:gpt-5', output_type=Answer)
voice = Agent(instructions='Answer using the `consult` tool, then read the summary aloud.')


@voice.tool_plain
async def consult(question: str) -> str:
    result = await supervisor.run(question)
    return result.output.summary


async def main():
    async with voice.realtime(OpenAIRealtimeModel('gpt-realtime')).session() as session:
        await session.send('What is the answer?')
```

The delegated run executes concurrently, so providers with asynchronous tool calls can keep talking
while analysis runs. To continue the entire conversation after the voice session, see
[History and handoff](history.md#handing-off-to-a-text-agent).

## Edge cases

- A tool finishing does not necessarily finish the turn. Wait for
  [`RealtimeTurnCompleteEvent`][pydantic_ai.realtime.RealtimeTurnCompleteEvent].
- Short tools can make asynchronous Gemini tool calling counterproductive: the result may interrupt
  a reply that barely started. Enable it for tools whose latency would otherwise create dead air.
- Native-tool behavior is model-specific. Check the profile and provider page rather than assuming
  every model from a provider supports the same tools.
