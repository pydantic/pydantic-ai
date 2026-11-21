---
description: "See which Pydantic AI capabilities and hooks run in a realtime voice session, which do not, and how tools and hooks reach the live session through RunContext."
---

# Capabilities and hooks

A [capability](../capabilities/overview.md) attached to the agent or passed to
`realtime(capabilities=...)` participates in a realtime session where its lifecycle maps onto a
persistent connection. [Third-party capabilities](../capabilities/overview.md#third-party-capabilities)
load exactly the same way as in a regular run; nothing realtime-specific is required of them.

## Capability stages in a session

| Capability stage | Session behavior |
| --- | --- |
| `for_agent`, `for_run`, `get_instructions` | Runs during setup; instructions are resolved at connect and [again before each model turn](#instructions-that-change). |
| `get_toolset`, `get_wrapper_toolset`, `prepare_tools` | Contributes, wraps, and prepares local tools before connecting. |
| `get_native_tools` | Contributes native tools before connecting; a dynamic native-tool function is resolved once against the connect-time context. |
| `before_realtime_connect` | Runs once before connecting and can change the seeded history; see [processing seeded history](#processing-seeded-history). |
| Tool validation/execution hooks | Runs around each local function-tool call. |
| `handle_deferred_tool_calls` | Resolves deferred requests inline; see [deferred and approval-required tools](tools.md#deferred-and-approval-required-tools). |
| Graph node, model-request, `get_wrapper_model`, and output-processing hooks | Do not run; no agent graph, model request, or output-processing stage exists. |

All regular [tool validation](../hooks.md#tool-validation-hooks) and
[tool execution](../hooks.md#tool-execution-hooks) hooks — `before`, `after`, `wrap`, and `on_error`
for both stages — run around every local function-tool call exactly as in a standard run, retries
and all. What does not run is anything tied to the request-response graph:
[node hooks](../hooks.md#node-hooks), [model request hooks](../hooks.md#model-request-hooks) such as
`before_model_request`, and [output validation](../hooks.md#output-validation-hooks) and
[output processing](../hooks.md#output-processing-hooks) hooks — a session has no graph nodes, no
per-request boundary, and no output stage.

Opening a session emits a `UserWarning` for each capability that implements any of those hooks,
naming the capability and its hooks, since what it does there won't happen in the session — for
example, [`ReinjectSystemPrompt`][pydantic_ai.capabilities.ReinjectSystemPrompt] works through
`before_model_request`. A [`Hooks`][pydantic_ai.capabilities.Hooks] capability warns only for the
hooks it registers. Built-ins whose behavior the session provides another way, such as
[`Instrumentation`][pydantic_ai.capabilities.Instrumentation] and
[`ProcessHistory`](#processing-seeded-history), don't warn.

## Run hooks

`before_run`, `after_run`, `wrap_run`, and `on_run_error` [run hooks](../hooks.md#run-hooks) run
once around the session — a realtime session is a run — with the same close-boundary recovery and
result-transformation semantics as [`iter()`][pydantic_ai.agent.AbstractAgent.iter].

## The event stream

`wrap_run_event_stream` wraps the consumer-facing session iterator. It can observe or transform
shared [`AgentStreamEvent`][pydantic_ai.messages.AgentStreamEvent] members and realtime-only
[`RealtimeSessionEvent`][pydantic_ai.messages.RealtimeSessionEvent] members (see the
[event reference](events.md)) without changing history or tool execution. There is no
`event_stream_handler` parameter on `realtime()`; a handler-style consumer is attached with the
[`ProcessEventStream`][pydantic_ai.capabilities.ProcessEventStream] capability, which works through
this same stream.

[Custom events](../agent.md#custom-events) and [capability events](../capabilities/overview.md#capability-events)
emitted with [`ctx.emit()`][pydantic_ai.tools.RunContext.emit] from tools and hooks flow through this
stream, and reach `on_event` listeners, as they do in [`iter()`][pydantic_ai.agent.AbstractAgent.iter].
Events emitted before the session connects, from `before_run`, lead the stream; a tool's events land
between its [`FunctionToolCallEvent`][pydantic_ai.messages.FunctionToolCallEvent] and
[`FunctionToolResultEvent`][pydantic_ai.messages.FunctionToolResultEvent]. Events emitted from
`after_run` come after the stream has ended, so as in `iter()` they aren't delivered, except to the
listeners of an [immediately dispatched](../capabilities/overview.md#reacting-to-events) event.

## Model settings and `RunContext`

`get_model_settings()` may run during capability setup, but regular model settings do not configure
a realtime model. Pass [`RealtimeModelSettings`][pydantic_ai.realtime.RealtimeModelSettings] through
`realtime(model_settings=...)` instead. Inside session hooks and tools, the
[`RunContext`][pydantic_ai.tools.RunContext] reflects the session:

| `RunContext` field | Value in a realtime session |
| --- | --- |
| [`ctx.model_settings`][pydantic_ai.tools.RunContext.model_settings] | The merged [`RealtimeModelSettings`][pydantic_ai.realtime.RealtimeModelSettings] the session was connected with. |
| [`ctx.realtime`][pydantic_ai.tools.RunContext.realtime] | `True` for the whole run, including `for_run` and instruction functions that run before the connection exists. |
| [`ctx.realtime_session`][pydantic_ai.tools.RunContext.realtime_session] | The live [`RealtimeSession`][pydantic_ai.realtime.RealtimeSession] in tool and `on_event` contexts; `None` in run hooks. |

!!! note
    `ctx.realtime_session` is `None` throughout `wrap_run`, both before and after `handler()`, because
    the hook keeps the context copy captured before the session exists. It is also `None` in
    `before_run` and instruction functions. Use tool hooks or `on_event` when a capability needs the
    live session.

## Instructions that change

As a standard run resolves the agent's [instructions](../agent.md#instructions) before each request, a session
resolves them again after each model response, and before a tool's result asks for the next one, so a change a tool
makes reaches the model before it answers. That covers instruction functions, capability and toolset instructions,
the instructions passed to `realtime()`, and dynamic system prompts registered with
[`on_change='append'`](../agent.md#appending-system-prompt-changes). A text turn sent meanwhile waits for them.

A session can't rewrite the instructions it opened with, so every change is appended, as an
[appending instruction](../agent.md#appending-instruction-changes) change is in a standard run, whatever the
block's `on_change`: the session sends the block's new text as a `<pyai:context>` element naming it, on the channel a
[system prompt sent with `respond=False`](turns.md#system-prompts) uses. For a change to name its block, every
instruction that can change and has an id of its own (a named instruction function, or one contributed by a
capability or toolset with an `id`) is sent inside a `<pyai:context>` element from the start. An unnamed instruction
function can't be named by a change, so if one changes, the session warns once and the model keeps the text the
session opened with: give it a `name`.

```python {title="realtime_instruction_changes.py"}
from dataclasses import dataclass

from pydantic_ai import Agent, RunContext


@dataclass
class Call:
    verified: bool = False


agent = Agent(deps_type=Call, instructions='You are a support agent for Acme Telecom.')


@agent.instructions(name='caller')
def caller_status(ctx: RunContext[Call]) -> str:
    if ctx.deps.verified:
        return 'The caller is verified: you may discuss their account.'
    return 'The caller is not verified yet: ask for their PIN before discussing their account.'


@agent.tool
async def verify_pin(ctx: RunContext[Call], pin: str) -> str:
    """Check the caller's PIN."""
    ctx.deps.verified = pin == '1234'
    return 'Verified.' if ctx.deps.verified else 'Wrong PIN.'


async def main():
    async with agent.realtime('{{ openai:realtime }}', deps=Call()).session() as session:
        await session.send('My PIN is 1234. What plan am I on?')
        await session.wait_for_reply()
```

History records each change as the [`InstructionDeltaPart`][pydantic_ai.messages.InstructionDeltaPart] a standard
run records, so a standard run [continuing the conversation](history.md#handing-off-to-a-text-agent) sends its own
current instructions rather than the session's changes. A reconnect that replays the conversation replays each
change where it was sent.

Each provider takes the change on its operator channel (see [system prompts](turns.md#system-prompts)), with these
differences:

- On [GPT-Live](openai.md#how-delegation-works), where the agent's instructions are the backend's, the change is
  appended to the backend's instructions, which only apply to the work it's delegated from then on.
- On [Gemini Live](gemini.md), text sent with a tool's result is invisible to the reply that result asks for, so a
  change a tool makes reaches the model from the turn after.
- On [ElevenLabs Agents](elevenlabs.md#elevenlabs-agents), the change is a contextual update, which the agent
  takes as background information rather than as a change to its prompt.

## Processing seeded history

A session has no per-request boundary, so instead of `before_model_request`, it calls
[`before_realtime_connect`][pydantic_ai.capabilities.AbstractCapability.before_realtime_connect] once,
before it connects. The hook receives a [`RealtimeConnectContext`][pydantic_ai.realtime.RealtimeConnectContext],
the realtime counterpart of [`ModelRequestContext`][pydantic_ai.models.ModelRequestContext], and can change the
`messages` the session is [seeded](history.md#seeding-a-session) with. Its `model`, `model_settings` and
`model_request_parameters` are read-only. [`ProcessHistory`][pydantic_ai.capabilities.ProcessHistory] implements
it, so a filter or redaction that a standard run applies before each model request also applies to what the
provider is seeded with. Register a hook of your own with
[`hooks.on.before_realtime_connect`](../hooks.md#realtime-session-hooks) for a change that only a session needs:

```python {title="realtime_seeded_history.py"}
import re
from dataclasses import replace

from pydantic_ai import Agent, ModelMessage, ModelRequest, ModelResponse, RunContext, TextPart, UserPromptPart
from pydantic_ai.capabilities import Hooks
from pydantic_ai.realtime import RealtimeConnectContext

CARD_NUMBER = re.compile(r'\b(?:\d[ -]?){13,16}\b')

hooks = Hooks()


def redact(message: ModelMessage) -> ModelMessage:
    if not isinstance(message, ModelRequest):
        return message
    parts = [
        replace(part, content=CARD_NUMBER.sub('[card number]', part.content))
        if isinstance(part, UserPromptPart) and isinstance(part.content, str)
        else part
        for part in message.parts
    ]
    return replace(message, parts=parts)


@hooks.on.before_realtime_connect
async def redact_card_numbers(ctx: RunContext[None], connect_context: RealtimeConnectContext) -> RealtimeConnectContext:
    connect_context.messages = [redact(message) for message in connect_context.messages]
    return connect_context


agent = Agent(instructions='You are a helpful voice assistant.', capabilities=[hooks])


async def main():
    history = [
        ModelRequest.user_text_prompt('My card number is 4242 4242 4242 4242.'),
        ModelResponse(parts=[TextPart('Thanks, I have it.')]),
    ]
    async with agent.realtime('{{ openai:realtime }}', message_history=history).session() as session:
        await session.send('Continue where we left off.')
        await session.wait_for_reply()
```

The hooks run in capability order, inside `wrap_run`, after `before_run` and after the history's dynamic system
prompts are re-evaluated, so they see and can redact the current content. As in a standard run, what `for_run` and `before_run`
hooks change in [`ctx.messages`][pydantic_ai.tools.RunContext.messages] is what the session is seeded with. They run even when there's no history to
seed, but not on a [reconnect](lifecycle.md#reconnecting), which replays the history they left. If one raises, the session
doesn't connect, and the error goes through `on_run_error`. [WebRTC signaling](deployment.md#browser-webrtc-server-sideband)
resolves the same configuration a session connects with, so the hooks also run, outside any run, when
[`answer_webrtc_offer()`][pydantic_ai.agent.AgentRealtime.answer_webrtc_offer] seeds the call with the history
they leave, and again when the sideband session opens. Keep them deterministic: on
[GPT-Live](openai.md), which takes its history when the offer starts the call, a sideband whose seed differs from the
offer's raises [`UserError`][pydantic_ai.exceptions.UserError], so a processor that summarizes with a model or depends
on the time can't be used there. What they leave is what the session records: it's the
start of [`all_messages()`][pydantic_ai.realtime.RealtimeSession.all_messages], and the instructions of the
capabilities it [loaded](#deferred-capability-loading) are resolved from it. A session can't rewrite the provider's
conversation before each response, so the messages it records after connecting aren't processed.

## Deferred capability loading

Deferred capabilities load in a session much as they do in a regular run: the capability
catalog is part of the session's instructions, and the model calls the `load_capability` tool. A
session can't state the loaded capability's instructions in the system voice the way a regular run does,
so the tool returns them as its result, which works on every provider. When a session is
[seeded](history.md#seeding-a-session) with the history of a regular run that loaded a capability, the
session's instructions include that capability's instructions instead. What a session
cannot do yet is advertise *new tools* mid-conversation: Pydantic AI fixes the connection's tools
when it opens (OpenAI and xAI accept a mid-session tool update;
[#7288](https://github.com/pydantic/pydantic-ai/issues/7288)), so opening a session with a
`defer_loading=True` capability that contributes tools or native tools raises
[`UserError`][pydantic_ai.exceptions.UserError] before connecting — accepting it would silently
provide less than requested. Loading those tools on providers that accept the update is tracked in
[#7191](https://github.com/pydantic/pydantic-ai/issues/7191).

Loading a capability from code with
[`ctx.load_capability()`][pydantic_ai.tools.RunContext.load_capability] isn't supported in a
session: it raises [`UserError`][pydantic_ai.exceptions.UserError] (see
[Loading a capability from code](../capabilities/on-demand.md#loading-from-code)).
