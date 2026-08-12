# Capabilities and hooks

A [capability](../capabilities/overview.md) attached to the agent or passed to
`realtime(capabilities=...)` participates in a realtime session where its lifecycle maps onto a
persistent connection. [Third-party capabilities](../capabilities/overview.md#third-party-capabilities)
load exactly the same way as in a regular run; nothing realtime-specific is required of them.

## Capability stages in a session

| Capability stage | Session behavior |
| --- | --- |
| `for_agent`, `for_run`, `get_instructions` | Runs during setup; dynamic instructions are evaluated once at connect. |
| `get_toolset`, `get_wrapper_toolset`, `prepare_tools` | Contributes, wraps, and prepares local tools before connecting. |
| `get_native_tools` | Contributes native tools before connecting; a dynamic native-tool function is resolved once against the connect-time context, like dynamic instructions. |
| Tool validation/execution hooks | Runs around each local function-tool call. |
| `handle_deferred_tool_calls` | Resolves deferred requests inline; see [deferred and approval-required tools](tools.md#deferred-and-approval-required-tools). |
| Graph node, model-request, and output-processing hooks | Do not run; no agent graph or output-processing stage exists. |

All regular [tool validation](../hooks.md#tool-validation-hooks) and
[tool execution](../hooks.md#tool-execution-hooks) hooks — `before`, `after`, `wrap`, and `on_error`
for both stages — run around every local function-tool call exactly as in a standard run, retries
and all. What does not run is anything tied to the request-response graph:
[node hooks](../hooks.md#node-hooks), [model request hooks](../hooks.md#model-request-hooks) such as
`before_model_request`, and [output validation](../hooks.md#output-validation-hooks) and
[output processing](../hooks.md#output-processing-hooks) hooks — a session has no graph nodes, no
per-request boundary, and no output stage.

## Run hooks

`before_run`, `after_run`, `wrap_run`, and `on_run_error` [run hooks](../hooks.md#run-hooks) run
once around the session — a realtime session is a run — with the same close-boundary recovery and
result-transformation semantics as [`iter()`][pydantic_ai.agent.AbstractAgent.iter].

## The event stream

`wrap_run_event_stream` wraps the consumer-facing session iterator. It can observe or transform
shared [`AgentStreamEvent`][pydantic_ai.messages.AgentStreamEvent] members and realtime-only
[`RealtimeEvent`][pydantic_ai.realtime.RealtimeEvent] members (see the
[event reference](events.md)) without changing history or tool execution. There is no
`event_stream_handler` parameter on `realtime()`; a handler-style consumer is attached with the
[`ProcessEventStream`][pydantic_ai.capabilities.ProcessEventStream] capability, which works through
this same stream.

## Model settings and `RunContext`

`get_model_settings()` may run during capability setup, but regular model settings do not configure
a realtime model. Pass [`RealtimeModelSettings`][pydantic_ai.realtime.RealtimeModelSettings] through
`realtime(model_settings=...)` instead. Inside session hooks and tools, the
[`RunContext`][pydantic_ai.tools.RunContext] reflects the session:

| `RunContext` field | Value in a realtime session |
| --- | --- |
| [`ctx.model_settings`][pydantic_ai.tools.RunContext.model_settings] | The merged [`RealtimeModelSettings`][pydantic_ai.realtime.RealtimeModelSettings] the session was connected with. |
| [`ctx.realtime`][pydantic_ai.tools.RunContext.realtime] | `True` from `before_run` onward. |
| [`ctx.realtime_session`][pydantic_ai.tools.RunContext.realtime_session] | The live [`RealtimeSession`][pydantic_ai.realtime.RealtimeSession] once it is connected. |

!!! note
    `ctx.realtime_session` is still `None` in `before_run`, in instruction functions, and in the
    pre-handler part of `wrap_run`, which all run before the connection is established.

## Seeded history is not processed

History-processing capabilities do not transform `message_history` before it is
[seeded into a session](history.md#seeding-a-session); preprocess the history before opening the
session when filtering or redaction is required.

## Deferred capability loading

Deferred capabilities load in a session the same way they do in a regular run: the capability
catalog is part of the session's instructions, and calling the `load_capability` tool returns the
loaded capability's instructions as its result — which works on every provider. What a session
cannot do is advertise *new tools* mid-conversation (the connection's tools are fixed when it
opens; see [#7288](https://github.com/pydantic/pydantic-ai/issues/7288)), so opening a session with
a `defer_loading=True` capability that contributes tools or native tools raises
[`UserError`][pydantic_ai.exceptions.UserError] before connecting — accepting it would silently
provide less than requested. Realtime per-turn/exchange hooks are expected to widen this boundary
in the future; see [#7190](https://github.com/pydantic/pydantic-ai/issues/7190) and
[#7191](https://github.com/pydantic/pydantic-ai/issues/7191).
