# Tools

[Tools](../tools.md) registered on an agent are offered to the realtime model and execute on your
backend. The session validates arguments, applies retries, runs tools concurrently, returns results
to the provider, and records ordinary tool-call messages for later handoff. Capability hooks around
tool calls are covered in [Capabilities and hooks](capabilities.md).

## Function tools

When a model calls a tool, the session emits
[`FunctionToolCallEvent`][pydantic_ai.messages.FunctionToolCallEvent], runs the tool, returns the
result, and emits [`FunctionToolResultEvent`][pydantic_ai.messages.FunctionToolResultEvent]. Parse
failures and [`ModelRetry`][pydantic_ai.exceptions.ModelRetry] produce a
[`RetryPromptPart`][pydantic_ai.messages.RetryPromptPart], matching a standard agent run. Other tool
exceptions end the session and propagate from iteration.

Tool return values reach the model exactly as in a
[standard run](../tools-advanced.md#advanced-tool-returns): the model receives the string rendering
of the return value — plus, where the provider supports it, multimodal content attached via
[`ToolReturn`][pydantic_ai.messages.ToolReturn]'s `content` — while local history keeps the full
structured [`ToolReturnPart`][pydantic_ai.messages.ToolReturnPart] with its `return_value`,
`content`, and `metadata`. Attached content is delivered for real or refused loudly — never
silently degraded: OpenAI and Azure OpenAI deliver text and images as a follow-up user message;
Gemini Live's tool results are JSON-only, so text is folded into the result and any binary
attachment raises [`UserError`][pydantic_ai.exceptions.UserError]
([#7362](https://github.com/pydantic/pydantic-ai/issues/7362)); media a provider can't carry
(audio and documents everywhere; images also on xAI) likewise raises before anything is sent.
If the provider cancels an in-flight call, Pydantic AI cancels the task
and records a synthetic cancellation result locally without sending that result back to the
provider.

### Concurrent tool execution

Every tool runs in the background, so a slow tool does not block session events, other tools, or
turn tracking. [`all_messages()`][pydantic_ai.realtime.RealtimeSession.all_messages] keeps each
result adjacent to its call even when calls finish out of order.

Whether the model continues speaking while it waits is provider-specific. Inspect the
[`supports_async_tool_calls`][pydantic_ai.realtime.RealtimeModelProfile.supports_async_tool_calls]
profile flag. OpenAI and Azure models generally fill the gap; Gemini pauses unless the
[`google_async_tool_calls`](gemini.md#asynchronous-tool-calls) setting — which declares the tools
`NON_BLOCKING` to the Live API — is enabled on a supported model.

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

agent = Agent(instructions='Answer questions, searching the web when useful.')


async def main():
    async with agent.realtime(
        'google:gemini-2.5-flash-native-audio-latest',
        capabilities=[WebSearch()],
    ).session() as session:
        await session.send("What's the latest Pydantic AI release?")
        async for event in session:
            if isinstance(event, PartEndEvent) and isinstance(event.part, NativeToolReturnPart):
                print(event.part.content)
```

An unsupported native tool with a configured local fallback is replaced before connection. Without
a fallback, opening the session raises [`UserError`][pydantic_ai.exceptions.UserError]. Provider and
model-specific combinations—including Gemini grounding, URL context, and function-tool
restrictions—are canonical on the [Gemini provider page](gemini.md#native-tools).

## Deferred and approval-required tools

**Approval-gated tools need a
[`HandleDeferredToolCalls`][pydantic_ai.capabilities.HandleDeferredToolCalls] handler; without one
the call is refused every time.** A standard run can end with a
[`DeferredToolRequests`][pydantic_ai.tools.DeferredToolRequests] output and resume once a human
answers (see [Deferred Tools](../deferred-tools.md)), but a live conversation has nowhere to pause:
with no handler, the model is told the tool cannot complete during a realtime session, and the tool
never runs.

The handler resolves each call inline: approve it (the tool then runs and returns normally), deny it
(recorded with `outcome='denied'`), substitute a result, or request a retry. This handler approves
small refunds from policy and denies the rest:

```python
from pydantic_ai import Agent, DeferredToolRequests, DeferredToolResults, ToolDenied
from pydantic_ai.capabilities import HandleDeferredToolCalls
from pydantic_ai.tools import RunContext

agent = Agent(instructions='You are a customer support voice assistant.')


@agent.tool_plain(requires_approval=True)
def issue_refund(order_id: str, amount: float) -> str:
    return f'Refunded ${amount:.2f} for order {order_id}.'


async def refund_policy(
    ctx: RunContext[None], requests: DeferredToolRequests
) -> DeferredToolResults:
    results = DeferredToolResults()
    for call in requests.approvals:
        if call.args_as_dict().get('amount', 0) <= 100:
            results.approvals[call.tool_call_id] = True
        else:
            results.approvals[call.tool_call_id] = ToolDenied(
                'Refunds over $100 need a human; offer to connect one.'
            )
    return results


async def main():
    async with agent.realtime(
        'openai:gpt-realtime',
        capabilities=[HandleDeferredToolCalls(handler=refund_policy)],
    ).session():
        ...
```

This applies to both ways a call is deferred — raising
[`ApprovalRequired`][pydantic_ai.exceptions.ApprovalRequired] or
[`CallDeferred`][pydantic_ai.exceptions.CallDeferred] from the tool, and declaring it up front with
[`requires_approval=True`](../deferred-tools.md#human-in-the-loop-tool-approval) or an
[external toolset](../toolsets.md#external-toolset). An approval-gated
tool is still advertised to the model, exactly as in a standard run; calling it opens the approval
flow rather than running the tool.

!!! warning "The handler answers from policy, not from a person"
    The handler must return a decision promptly — it is a programmatic policy resolver, not an approval
    UI. It runs as a background task like the tool itself, so it never blocks the session's events, but
    what the *conversation* does while it thinks is provider-specific in exactly the way
    [concurrent tool execution](#concurrent-tool-execution) describes: OpenAI and Azure carry on, while
    Gemini holds the model's turn until the result arrives. On Gemini a slow handler therefore reads as
    assistant silence, and if the user speaks into that gap the provider cancels the pending call
    outright (recorded as [a synthetic cancellation](#function-tools)).

Asking a human mid-call and resuming on their answer is not supported yet: a realtime session cannot
pause and return a `DeferredToolRequests` output for an out-of-band result. Resolve the request
during the call, or move that workflow to a standard agent run.

[`DeferredToolRequestsEvent`][pydantic_ai.messages.DeferredToolRequestsEvent] on a session is
informational for the same reason: it is emitted when the handler *has* resolved the calls, so a
consumer can observe what was asked and decided. It is not a hook to respond to — unlike the same
event in a standard run, nothing waits for the consumer, and no event is emitted when no handler is
installed and the call is refused.

Tools registered with `defer_loading=True` are rejected in a realtime session for a related reason;
see [Deferred capability loading](capabilities.md#deferred-capability-loading).

## Enqueuing prompts from tools

[`RunContext.enqueue()`][pydantic_ai.tools.RunContext.enqueue] — the same mechanism as
[injecting follow-up messages from a tool](../tools.md#injecting-follow-up-messages-from-a-tool) in
a standard run — accepts one plain-text prompt per call from a realtime tool. The default
`priority='asap'` sends it when no response is active; `priority='when_idle'` waits until the
provider reports its current response complete. Neither priority interrupts assistant speech.
Delivered prompts become ordinary user turns in history, as in
[injecting messages mid-run](../message-history.md#injecting-messages-mid-run).

Multimodal content and prebuilt message/part sequences are rejected because the realtime live-input
channel cannot preserve their standard-run semantics.

## Delegating work during a call

Realtime models do not provide structured output and can be weaker at complex reasoning than a
frontier text model. Expose a tool that delegates the hard work to a standard
[`Agent`][pydantic_ai.Agent] with an `output_type`:

```python
from pydantic import BaseModel

from pydantic_ai import Agent
from pydantic_ai.realtime import RealtimeTurnCompleteEvent


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
    async with voice.realtime('openai:gpt-realtime').session() as session:
        await session.send(
            'Which of our three shipping options is cheapest for a 4 kg parcel to Berlin?'
        )
        async for event in session:
            if isinstance(event, RealtimeTurnCompleteEvent):
                break
```

The delegated run executes concurrently, so providers with asynchronous tool calls can keep talking
while analysis runs. To continue the entire conversation after the voice session, see
[History and handoff](history.md#handing-off-to-a-text-agent).

## Edge cases

- A tool finishing does not necessarily finish the turn; see the
  [turn boundary](events.md#the-turn-boundary).
- Short tools can make asynchronous Gemini tool calling counterproductive: the result may interrupt
  a reply that barely started. Enable it for tools whose latency would otherwise create dead air.
- Native-tool behavior is model-specific. Check the profile and provider page rather than assuming
  every model from a provider supports the same tools.
