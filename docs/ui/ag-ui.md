# Agent-User Interaction (AG-UI) Protocol

The [Agent-User Interaction (AG-UI) Protocol](https://docs.ag-ui.com/introduction) is an open standard introduced by the
[CopilotKit](https://webflow.copilotkit.ai/blog/introducing-ag-ui-the-protocol-where-agents-meet-users)
team that standardises how frontend applications communicate with AI agents, with support for streaming, frontend tools, shared state, and custom events.

!!! note
    The AG-UI integration was originally built by the team at [Rocket Science](https://www.rocketscience.gg/) and contributed in collaboration with the Pydantic AI and CopilotKit teams. Thanks Rocket Science!

## Installation

The only dependencies are:

- [ag-ui-protocol](https://docs.ag-ui.com/introduction): to provide the AG-UI types and encoder.
- [starlette](https://www.starlette.io): to handle [ASGI](https://asgi.readthedocs.io/en/latest/) requests from a framework like FastAPI.

You can install Pydantic AI with the `ag-ui` extra to ensure you have all the
required AG-UI dependencies:

```bash
pip/uv-add 'pydantic-ai-slim[ag-ui]'
```

To run the examples you'll also need:

- [uvicorn](https://uvicorn.dev) or another ASGI compatible server

```bash
pip/uv-add uvicorn
```

## Usage

There are three ways to run a Pydantic AI agent based on AG-UI run input with streamed AG-UI events as output, from most to least flexible. If you're using a Starlette-based web framework like FastAPI, you'll typically want to use the second method.

1. The [`AGUIAdapter.run_stream()`][pydantic_ai.ui.ag_ui.AGUIAdapter.run_stream] method, when called on an [`AGUIAdapter`][pydantic_ai.ui.ag_ui.AGUIAdapter] instantiated with an agent and an AG-UI [`RunAgentInput`](https://docs.ag-ui.com/sdk/python/core/types#runagentinput) object, will run the agent and return a stream of AG-UI events. It also takes optional [`Agent.iter()`][pydantic_ai.agent.Agent.iter] arguments including `deps`. Use this if you're using a web framework not based on Starlette (e.g. Django or Flask) or want to modify the input or output some way.
2. The [`AGUIAdapter.dispatch_request()`][pydantic_ai.ui.ag_ui.AGUIAdapter.dispatch_request] class method takes an agent and a Starlette request (e.g. from FastAPI) coming from an AG-UI frontend, and returns a streaming Starlette response of AG-UI events that you can return directly from your endpoint. It also takes optional [`Agent.iter()`][pydantic_ai.agent.Agent.iter] arguments including `deps`, that you can vary for each request (e.g. based on the authenticated user). This is a convenience method that combines [`AGUIAdapter.from_request()`][pydantic_ai.ui.ag_ui.AGUIAdapter.from_request], [`AGUIAdapter.run_stream()`][pydantic_ai.ui.ag_ui.AGUIAdapter.run_stream], and [`AGUIAdapter.streaming_response()`][pydantic_ai.ui.ag_ui.AGUIAdapter.streaming_response].
3. Build a stand-alone [`Starlette`](https://www.starlette.io/applications/) app with a single `/` route that calls [`AGUIAdapter.dispatch_request()`][pydantic_ai.ui.ag_ui.AGUIAdapter.dispatch_request]. The same Starlette app can be [mounted](https://fastapi.tiangolo.com/advanced/sub-applications/) at a path in an existing FastAPI app.

When a run ends in [first-party cancellation](../agent.md#cancelling-a-run) — `ctx.cancel()`, `AgentRun.cancel()`, or a [`CancellationToken`][pydantic_ai.CancellationToken] your server wires to a cancel endpoint — the adapter closes any open text or tool events and emits a bare `RUN_FINISHED`. AG-UI currently has no cancelled outcome, so cancellation is not reported as `RUN_ERROR`. Pass an `on_cancel` callback (see the `run_stream()` example below) to persist the resumable message history from [`RunCancelled.all_messages()`][pydantic_ai.exceptions.RunCancelled.all_messages].

!!! note "Client disconnects are external cancellation"
    A client that disconnects (or aborts its request) is seen by the server as an external `asyncio.CancelledError` rather than a first-party cancellation (see [the two kinds of cancellation](../agent.md#cancelling-a-run)), so the bare `RUN_FINISHED` and `on_cancel` do not fire on a disconnect. To observe a stop gesture this way, keep the stream connected and cancel the run first-party via a [`CancellationToken`][pydantic_ai.CancellationToken] triggered from a separate cancel endpoint.

### Handle run input and output directly

This example uses [`AGUIAdapter.run_stream()`][pydantic_ai.ui.ag_ui.AGUIAdapter.run_stream] and performs its own request parsing and response generation.
This can be modified to work with any web framework.

```py {title="run_ag_ui.py"}
import json
from http import HTTPStatus

from fastapi import FastAPI
from fastapi.requests import Request
from fastapi.responses import Response, StreamingResponse
from pydantic import ValidationError

from pydantic_ai import Agent, RunCancelled
from pydantic_ai.ui import SSE_CONTENT_TYPE
from pydantic_ai.ui.ag_ui import AGUIAdapter

agent = Agent('openai:gpt-5.2', instructions='Be fun!')

app = FastAPI()


async def on_cancel(cancelled: RunCancelled) -> None:
    messages = cancelled.all_messages()  # the resumable history to persist
    print(f'cancelled after {len(messages)} messages')


@app.post('/')
async def run_agent(request: Request) -> Response:
    accept = request.headers.get('accept', SSE_CONTENT_TYPE)
    try:
        run_input = AGUIAdapter.build_run_input(await request.body())  # (1)
    except ValidationError as e:
        return Response(
            content=json.dumps(e.json()),
            media_type='application/json',
            status_code=HTTPStatus.UNPROCESSABLE_ENTITY,
        )

    adapter = AGUIAdapter(agent=agent, run_input=run_input, accept=accept)
    event_stream = adapter.run_stream(on_cancel=on_cancel)  # (2)

    sse_event_stream = adapter.encode_stream(event_stream)
    return StreamingResponse(sse_event_stream, media_type=accept) # (3)
```

1. [`AGUIAdapter.build_run_input()`][pydantic_ai.ui.ag_ui.AGUIAdapter.build_run_input] takes the request body as bytes and returns an AG-UI [`RunAgentInput`](https://docs.ag-ui.com/sdk/python/core/types#runagentinput) object. You can also use the [`AGUIAdapter.from_request()`][pydantic_ai.ui.ag_ui.AGUIAdapter.from_request] class method to build an adapter directly from a request.
2. [`AGUIAdapter.run_stream()`][pydantic_ai.ui.ag_ui.AGUIAdapter.run_stream] runs the agent and returns a stream of AG-UI events. It supports the same optional arguments as [`Agent.run_stream_events()`](../agent.md#running-agents), including `deps`. You can also use [`AGUIAdapter.run_stream_native()`][pydantic_ai.ui.ag_ui.AGUIAdapter.run_stream_native] to run the agent and return a stream of Pydantic AI events instead, which can then be transformed into AG-UI events using [`AGUIAdapter.transform_stream()`][pydantic_ai.ui.ag_ui.AGUIAdapter.transform_stream].
3. [`AGUIAdapter.encode_stream()`][pydantic_ai.ui.ag_ui.AGUIAdapter.encode_stream] encodes the stream of AG-UI events as strings according to the accept header value. You can also use [`AGUIAdapter.streaming_response()`][pydantic_ai.ui.ag_ui.AGUIAdapter.streaming_response] to generate a streaming response directly from the AG-UI event stream returned by `run_stream()`.

Since `app` is an ASGI application, it can be used with any ASGI server:

```shell
uvicorn run_ag_ui:app
```

This will expose the agent as an AG-UI server, and your frontend can start sending requests to it.

### Handle a Starlette request

This example uses [`AGUIAdapter.dispatch_request()`][pydantic_ai.ui.ag_ui.AGUIAdapter.dispatch_request] to directly handle a FastAPI request and return a response. Something analogous to this will work with any Starlette-based web framework.

```py {title="handle_ag_ui_request.py"}
from fastapi import FastAPI
from starlette.requests import Request
from starlette.responses import Response

from pydantic_ai import Agent
from pydantic_ai.ui.ag_ui import AGUIAdapter

agent = Agent('openai:gpt-5.2', instructions='Be fun!')

app = FastAPI()


@app.post('/')
async def run_agent(request: Request) -> Response:
    return await AGUIAdapter.dispatch_request(request, agent=agent) # (1)
```

1. This method essentially does the same as the previous example, but it's more convenient to use when you're already using a Starlette/FastAPI app.

Since `app` is an ASGI application, it can be used with any ASGI server:

```shell
uvicorn handle_ag_ui_request:app
```

This will expose the agent as an AG-UI server, and your frontend can start sending requests to it.

### Stand-alone ASGI app

When you don't already have a Starlette/FastAPI app to mount the route on, build a minimal [`Starlette`](https://www.starlette.io/applications/) app whose single `/` route calls [`AGUIAdapter.dispatch_request()`][pydantic_ai.ui.ag_ui.AGUIAdapter.dispatch_request]:

```py {title="ag_ui_app.py"}
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import Response
from starlette.routing import Route

from pydantic_ai import Agent
from pydantic_ai.ui.ag_ui import AGUIAdapter

agent = Agent('openai:gpt-5.2', instructions='Be fun!')


async def run_agent(request: Request) -> Response:
    return await AGUIAdapter.dispatch_request(request, agent=agent)


app = Starlette(routes=[Route('/', run_agent, methods=['POST'])])
```

Since `app` is an ASGI application, it can be used with any ASGI server:

```shell
uvicorn ag_ui_app:app
```

This will expose the agent as an AG-UI server, and your frontend can start sending requests to it.

## Design

The Pydantic AI AG-UI integration supports all features of the spec:

- [Events](https://docs.ag-ui.com/concepts/events)
- [Messages](https://docs.ag-ui.com/concepts/messages)
- [State Management](https://docs.ag-ui.com/concepts/state)
- [Tools](https://docs.ag-ui.com/concepts/tools)

The integration receives messages in the form of a
[`RunAgentInput`](https://docs.ag-ui.com/sdk/python/core/types#runagentinput) object
that describes the details of the requested agent run including message history, state, and available tools.

These are converted to Pydantic AI types and passed to the agent's run method. Events from the agent, including tool calls, are converted to AG-UI events and streamed back to the caller as Server-Sent Events (SSE).

A user request may require multiple round trips between client UI and Pydantic AI
server, depending on the tools and events needed.

## Features

### State management

The integration provides full support for
[AG-UI state management](https://docs.ag-ui.com/concepts/state), which enables
real-time synchronization between agents and frontend applications.

In the example below we have document state which is shared between the UI and
server using the [`StateDeps`][pydantic_ai.ui.StateDeps] [dependencies type](../dependencies.md) that can be used to automatically
validate state contained in [`RunAgentInput.state`](https://docs.ag-ui.com/sdk/js/core/types#runagentinput) using a Pydantic `BaseModel` specified as a generic parameter.

!!! note "Custom dependencies type with AG-UI state"
    If you want to use your own dependencies type to hold AG-UI state as well as other things, it needs to implement the
    [`StateHandler`][pydantic_ai.ui.StateHandler] protocol, meaning it needs to be a [dataclass](https://docs.python.org/3/library/dataclasses.html) with a non-optional `state` field. This lets Pydantic AI ensure that state is properly isolated between requests by building a new dependencies object each time.

    If the `state` field's type is a Pydantic `BaseModel` subclass, the raw state dictionary on the request is automatically validated. If not, you can validate the raw value yourself in your dependencies dataclass's `__post_init__` method.

    If AG-UI state is provided but your dependencies do not implement [`StateHandler`][pydantic_ai.ui.StateHandler], Pydantic AI will emit a warning and ignore the state. Use [`StateDeps`][pydantic_ai.ui.StateDeps] or a custom [`StateHandler`][pydantic_ai.ui.StateHandler] implementation to receive and validate the incoming state.


```python {title="ag_ui_state.py"}
from dataclasses import replace

from pydantic import BaseModel
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import Response
from starlette.routing import Route

from pydantic_ai import Agent
from pydantic_ai.ui import StateDeps
from pydantic_ai.ui.ag_ui import AGUIAdapter


class DocumentState(BaseModel):
    """State for the document being written."""

    document: str = ''


agent = Agent(
    'openai:gpt-5.2',
    instructions='Be fun!',
    deps_type=StateDeps[DocumentState],
)
deps = StateDeps(DocumentState())


async def run_agent(request: Request) -> Response:
    # `dispatch_request` mutates `deps.state` from the request, so give each request its own copy.
    return await AGUIAdapter.dispatch_request(request, agent=agent, deps=replace(deps))


app = Starlette(routes=[Route('/', run_agent, methods=['POST'])])
```

Since `app` is an ASGI application, it can be used with any ASGI server:

```bash
uvicorn ag_ui_state:app --host 0.0.0.0 --port 9000
```

### Tools

AG-UI frontend tools are seamlessly provided to the Pydantic AI agent, enabling rich
user experiences with frontend user interfaces.

### Context

Alongside messages, an AG-UI client can send a `context` array of `description`/`value` pairs describing things it considers relevant to the run: the originating platform, the requesting user, or a channel's standing instructions. Every entry is a claim the client made — it can send any `description`/`value` it likes — so they describe a request, they never establish who is making it.

These entries are not passed to the model automatically, and they don't belong in [instructions][pydantic_ai.agent.Agent.instructions]. Instructions carry operator authority — they're treated as *your* instruction to the model — so building them out of text a client sent lets a prompt injection inherit that authority. Delivering them as data denies them that authority but doesn't make them safe: they're still indirect prompt-injection input, so scope and re-authorize side-effecting tools from `deps` your server established, never from an entry's `description` or `value`. See [Mid-conversation system prompts](../message-history.md#mid-conversation-system-prompts) and the [trust model](./overview.md#trust-model-for-client-submitted-messages).

Read the entries off `adapter.run_input.context` and deliver them to the model as **data**. Facts your server established — the authenticated user, the workspace — are what go in instructions:

```py {title="ag_ui_context.py"}
from dataclasses import dataclass

from ag_ui.core import Context
from fastapi import FastAPI
from starlette.requests import Request
from starlette.responses import Response

from pydantic_ai import Agent, RunContext
from pydantic_ai.ui.ag_ui import AGUIAdapter


@dataclass
class ChannelDeps:
    workspace: str  # (1)!
    context: list[Context]  # (2)!


agent = Agent('openai:gpt-5.2', deps_type=ChannelDeps)
app = FastAPI()


@agent.instructions
def workspace(ctx: RunContext[ChannelDeps]) -> str:
    return f'You are answering in the {ctx.deps.workspace} workspace.'


@agent.tool
def frontend_context(ctx: RunContext[ChannelDeps]) -> list[str]:
    """Context the frontend says is relevant to this conversation."""
    return [f'{entry.description}: {entry.value}' for entry in ctx.deps.context]


def authenticated_workspace(request: Request) -> str:
    """Whatever your auth layer already established — a session, a signed token, an API key."""
    ...


@app.post('/')
async def run_agent(request: Request) -> Response:
    adapter = await AGUIAdapter.from_request(request, agent=agent)
    deps = ChannelDeps(workspace=authenticated_workspace(request), context=adapter.run_input.context)
    return adapter.streaming_response(adapter.run_stream(deps=deps))
```

1. Established by your server, so it can shape how the agent behaves.
2. Sent by the client, so it reaches the model as tool output the agent can read — never as an instruction.

To let a client-supplied fact change how the agent behaves, authenticate it first: verify the caller or channel, look up the policy *your* server holds for it, and write the instruction from that. The entry itself stays data.

Anything that isn't meant for the model at all — a Slack channel ID, a locale — is better carried in `forwardedProps`, which the adapter passes through untouched as `adapter.run_input.forwarded_props`. Validating it proves shape, not identity: who the user is, what tenant they're in, and what they're allowed to do come from authenticated server state.

When the agent's events reach you outside the request that serves the frontend, there's no run input to read them off at all — see ["Encoding events without a request"](./overview.md#encoding-events-without-a-request), where [`AGUIEventStream.thread_id`][pydantic_ai.ui.ag_ui.AGUIEventStream.thread_id] and [`run_id`][pydantic_ai.ui.ag_ui.AGUIEventStream.run_id] take over as the source of the identity the protocol requires.

`context`, `forwardedProps` and `parentRunId` are read straight off [`run_input`][pydantic_ai.ui.UIAdapter.run_input] rather than through adapter properties of their own. The adapter's properties — `messages`, `toolset`, `state`, `conversation_id`, `deferred_tool_results` — are the concepts every UI protocol shares and that the adapter itself feeds into the agent run. These three are AG-UI-specific and consumed only by your code, so they stay on the protocol object where their types are the protocol's own.

### Tool approval (interrupts)

Tools declared with `requires_approval=True` map onto AG-UI's [interrupt-aware run lifecycle](https://docs.ag-ui.com/concepts/interrupts). When the model proposes such a call, the run pauses and the adapter ends the SSE stream with a `RUN_FINISHED` event whose `outcome.type` is `"interrupt"` and whose `outcome.interrupts[]` describes each pending approval. The client renders an approval UI from that list and POSTs the next `RunAgentInput` with a `resume[]` array of `ResumeEntry` items addressing each interrupt.

The mapping the adapter applies (matching the AG-UI Python SDK field names):

| AG-UI direction         | Pydantic AI source / sink                                                                                       |
| ----------------------- | --------------------------------------------------------------------------------------------------------------- |
| `Interrupt.reason`      | Always `"tool_call"` for `requires_approval=True` tools                                                         |
| `Interrupt.tool_call_id`| The `ToolCallPart.tool_call_id` of the proposed call                                                            |
| `Interrupt.id`          | `f"int-{tool_call_id}"` (round-trips back to `tool_call_id` on resume)                                          |
| `Interrupt.metadata`    | `DeferredToolRequests.metadata.get(tool_call_id)`                                                               |
| `ResumeEntry.payload`   | `{ "approved": bool, "editedArgs"?: object, "reason"?: string }`, validated against `Interrupt.response_schema`; a payload that fails validation denies, including when the offending field is not `approved` itself — a wrongly-typed `editedArgs` or `reason` denies even alongside `approved=True`, while omitting either optional field or sending it as `null` is accepted |
| `payload.approved=True` | [`ToolApproved`][pydantic_ai.tools.ToolApproved]                                                                |
| `payload.editedArgs`    | [`ToolApproved.override_args`][pydantic_ai.tools.ToolApproved.override_args] (fully replaces the proposed args) |
| `payload.approved=False`| [`ToolDenied`][pydantic_ai.tools.ToolDenied] with `message=payload.reason`. `approved` is required, so a payload that omits it denies on validation and its `reason` is not used — send `approved: false` explicitly to have your `reason` reach the model |
| `status="cancelled"`    | [`ToolDenied`][pydantic_ai.tools.ToolDenied] with `message="Cancelled by user."` regardless of payload          |

The agent must include [`DeferredToolRequests`][pydantic_ai.tools.DeferredToolRequests] in its `output_type` so the run can pause cleanly instead of erroring on the proposed call:

```python {title="ag_ui_tool_approval.py"}
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import Response
from starlette.routing import Route

from pydantic_ai import Agent
from pydantic_ai.tools import DeferredToolRequests
from pydantic_ai.ui.ag_ui import AGUIAdapter

agent = Agent('openai:gpt-5.2', output_type=[str, DeferredToolRequests])


@agent.tool_plain(requires_approval=True)
def delete_file(path: str) -> str:
    """Delete a file. Pauses on a `RUN_FINISHED` interrupt outcome until the user approves."""
    return f'deleted {path}'


async def run_agent(request: Request) -> Response:
    return await AGUIAdapter.dispatch_request(request, agent=agent)


app = Starlette(routes=[Route('/', run_agent, methods=['POST'])])
```

On the resumed turn the agent re-executes the tool against the **original** `tool_call_id`, so only a `TOOL_CALL_RESULT` event is emitted for that id — no fresh `TOOL_CALL_START`. This preserves the audit trail the AG-UI spec requires.

See [Deferred tools and human-in-the-loop tool approval](../deferred-tools.md) for the underlying Pydantic AI primitive that also works outside AG-UI.

!!! note "Version requirement"
    Interrupts require `ag-ui-protocol >= 0.1.19` ([PR #1569](https://github.com/ag-ui-protocol/ag-ui/pull/1569)). On older installs the adapter silently falls back to emitting a bare `RUN_FINISHED` event without an outcome, and `resume[]` is ignored even if a client sends it.

### Events

Pydantic AI tools can send [AG-UI events](https://docs.ag-ui.com/concepts/events) simply by returning a
[`ToolReturn`](../tools-advanced.md#advanced-tool-returns) object with a
[`BaseEvent`](https://docs.ag-ui.com/sdk/python/core/events#baseevent) (or a list of events) as `metadata`,
which allows for custom events and state updates.

```python {title="ag_ui_tool_events.py"}
from dataclasses import replace

from ag_ui.core import CustomEvent, EventType, StateSnapshotEvent
from pydantic import BaseModel
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import Response
from starlette.routing import Route

from pydantic_ai import Agent, RunContext, ToolReturn
from pydantic_ai.ui import StateDeps
from pydantic_ai.ui.ag_ui import AGUIAdapter


class DocumentState(BaseModel):
    """State for the document being written."""

    document: str = ''


agent = Agent(
    'openai:gpt-5.2',
    instructions='Be fun!',
    deps_type=StateDeps[DocumentState],
)
deps = StateDeps(DocumentState())


async def run_agent(request: Request) -> Response:
    return await AGUIAdapter.dispatch_request(request, agent=agent, deps=replace(deps))


app = Starlette(routes=[Route('/', run_agent, methods=['POST'])])


@agent.tool
async def update_state(ctx: RunContext[StateDeps[DocumentState]]) -> ToolReturn:
    return ToolReturn(
        return_value='State updated',
        metadata=[
            StateSnapshotEvent(
                type=EventType.STATE_SNAPSHOT,
                snapshot=ctx.deps.state,
            ),
        ],
    )


@agent.tool_plain
async def custom_events() -> ToolReturn:
    return ToolReturn(
        return_value='Count events sent',
        metadata=[
            CustomEvent(
                type=EventType.CUSTOM,
                name='count',
                value=1,
            ),
            CustomEvent(
                type=EventType.CUSTOM,
                name='count',
                value=2,
            ),
        ]
    )
```

Since `app` is an ASGI application, it can be used with any ASGI server:

```bash
uvicorn ag_ui_tool_events:app --host 0.0.0.0 --port 9000
```

### Protocol version compatibility

Pydantic AI supports every `ag-ui-protocol` release from `0.1.10` on, and features added after that floor are gated on the version you have installed rather than requiring an upgrade.

That gate runs in both directions. On the way out, content an older protocol version can't express is downgraded or omitted — see [`AGUIAdapter.ag_ui_version`][pydantic_ai.ui.ag_ui.AGUIAdapter.ag_ui_version] for the negotiated thresholds. On the way in, a message `role` or input content `type` your installed `ag-ui-protocol` has no class for is skipped with a `UserWarning` naming the tag, and the rest of the request runs — so a frontend on a newer protocol version than your server keeps working, minus the content your install has no type for. For instance, a gateway that forwards image attachments as typed multimodal content (`ag-ui-protocol >= 0.1.15`) still delivers the accompanying text to an agent running on an older install.

What gets skipped is decided by the tag alone: any `role` or `type` string the installed models don't declare qualifies, so a client that misspells `"txet"` is skipped with the same warning as one sending genuinely newer content — the server has no way to tell those apart. The skip is scoped to well-formed items: a message must still carry a string `id`, the field every AG-UI message type requires.

Everything else is still rejected with `422 Unprocessable Entity` — a payload that is malformed under a `role` or `type` the install *does* know, a `role` or `type` that isn't a string at all, and a body that isn't valid JSON. If you see the warning and the content was real, upgrading `ag-ui-protocol` is what makes it reach your agent.

### Trust model

AG-UI's `RunAgentInput.messages` is fully client-controlled. The [`AGUIAdapter`][pydantic_ai.ui.ag_ui.AGUIAdapter] applies defaults to strip untrusted parts before the agent runs — see [Trust model for client-submitted messages](./overview.md#trust-model-for-client-submitted-messages) in the UI adapter overview, which covers system prompts, file URL schemes, uploaded files ([`allow_uploaded_files`][pydantic_ai.ui.UIAdapter.allow_uploaded_files]), and unresolved tool calls. Those defaults don't make client-submitted history authentic — see [Trust boundary for client-supplied history](../message-history.md#trust-boundary-for-client-supplied-history).

### Compaction

[`CompactionPart`][pydantic_ai.messages.CompactionPart]s round-trip through AG-UI activity messages (`pydantic_ai_compaction`), so [compacted](../capabilities/compaction.md) conversations keep working when the frontend holds the message history. A compaction item submitted by the frontend is honored — the conversation stays compacted — with two caveats. First, it is never trusted to stand in for the system prompt: whichever prompt applies per [System prompts and instructions](#system-prompts-and-instructions) still reaches the model on every request. Second, if the run also receives server-side `message_history` (the [server-side persistence pattern](./overview.md#trust-model-for-client-submitted-messages)), frontend compaction items are ignored — everything before a compaction item is hidden from the model, so honoring one from the frontend would let it hide the server's stored history. See [Client-held history](../capabilities/compaction.md#client-held-history) for the trade-offs and the recommended server-side pattern.

### Assistant message identity

Every streamed tool call carries a `parentMessageId` naming the assistant message that owns it, and that message is always announced by a [`TEXT_MESSAGE_START`](https://github.com/ag-ui-protocol/ag-ui/blob/11f03fa65c4fa22a8637d3f6e06e77d8c1b9ae78/docs/sdk/python/core/events.mdx#L170-L187) first — including when the model's response is nothing but tool calls, in which case the message carries no content and is closed immediately. A frontend that rebuilds the conversation from the event stream alone therefore never has to infer that a message exists.

Tool calls attach to whichever assistant message is open when they stream: text appearing before them in the same response shares their message, and text appearing after starts a new one that any later tool calls attach to instead. [`AGUIAdapter.dump_messages`][pydantic_ai.ui.ag_ui.AGUIAdapter.dump_messages] splits text and tool calls the same way. It splits on more than that, though: a compaction part always starts a new assistant message when history is loaded, a reasoning part does so from `ag-ui-protocol` 0.1.13, and a file part does so only under [`AGUIAdapter.preserve_file_data`][pydantic_ai.ui.ag_ui.AGUIAdapter.preserve_file_data]. The stream splits on none of them, so a response interleaving one of those with tool calls yields more messages loaded than streamed.

### Preserving failed tool outcomes

AG-UI's [`ToolCallResultEvent`](https://github.com/ag-ui-protocol/ag-ui/blob/11f03fa65c4fa22a8637d3f6e06e77d8c1b9ae78/docs/sdk/python/core/events.mdx#L284-L304) has no error or outcome field. Although [encrypted reasoning continuity](https://github.com/ag-ui-protocol/ag-ui/blob/11f03fa65c4fa22a8637d3f6e06e77d8c1b9ae78/docs/concepts/reasoning.mdx#L6-L29) is the intended use of [`ReasoningEncryptedValueEvent`](https://github.com/ag-ui-protocol/ag-ui/blob/11f03fa65c4fa22a8637d3f6e06e77d8c1b9ae78/docs/sdk/python/core/events.mdx#L555-L577), it is also AG-UI's standard event for attaching `encrypted_value` to a message or tool call. Pydantic AI uses that attachment mechanism with a namespaced payload to preserve `outcome='failed'` from [`ToolReturnPart`][pydantic_ai.messages.ToolReturnPart] when using `ag-ui-protocol >= 0.1.13`.

If the client sends those messages back on a later run, the adapter restores the failed outcome. This is a history-continuity mechanism: it does not set [`ToolMessage.error`](https://github.com/ag-ui-protocol/ag-ui/blob/11f03fa65c4fa22a8637d3f6e06e77d8c1b9ae78/docs/concepts/messages.mdx#L143-L163) or guarantee that a frontend visually renders the result as an error. Event streams produced with earlier protocol versions have no metadata carrier for the outcome, so reloading them reconstructs the tool result as `outcome='success'`.

### Preserving files across round-trips

AG-UI has no native representation for agent-generated files ([`FilePart`][pydantic_ai.messages.FilePart]) or [`UploadedFile`][pydantic_ai.messages.UploadedFile] references, so they are omitted from `dump_messages` output by default. Set [`AGUIAdapter.preserve_file_data`][pydantic_ai.ui.ag_ui.AGUIAdapter.preserve_file_data] to `True` to round-trip them through reserved `pydantic_ai_*` [activity messages](https://docs.ag-ui.com/concepts/messages), which a frontend completes by echoing those activity messages back on the next request. This is a representation opt-in, not a security one: an `UploadedFile` reconstructed from a round-tripped activity message is still subject to the inbound [`allow_uploaded_files`][pydantic_ai.ui.UIAdapter.allow_uploaded_files] gate before it reaches the agent.

!!! warning "Behavior change"
    `preserve_file_data` used to gate honoring inbound client-submitted `UploadedFile` references. It is now representation-only. If your app set `AGUIAdapter(preserve_file_data=True)` to accept inbound uploaded files, you must now also set [`allow_uploaded_files`][pydantic_ai.ui.UIAdapter.allow_uploaded_files]`=True`, since the two concerns are now separate flags.

### System prompts and instructions

Pydantic AI supports two ways to provide guidance to the model: [`system_prompt`](../agent.md#system-prompts) (stored in the message history as [`SystemPromptPart`][pydantic_ai.messages.SystemPromptPart]s) and [`instructions`](../agent.md#instructions) (injected fresh on every request, never persisted). When you control the server side, `instructions` is the recommended default.

The rest of this section only matters if you use `system_prompt`. If you only use `instructions`, there's nothing to configure — they're always applied regardless of the AG-UI message history.

For `system_prompt`, you choose who owns it with the `manage_system_prompt` parameter on [`AGUIAdapter`][pydantic_ai.ui.ag_ui.AGUIAdapter]:

- `'server'` (default): the agent's configured `system_prompt` is authoritative. Any `SystemMessage` sent by the frontend is stripped with a warning (a malicious client could otherwise inject arbitrary instructions via crafted API requests), and the agent's own system prompt is reinjected at the head of the first request via the [`ReinjectSystemPrompt`][pydantic_ai.capabilities.ReinjectSystemPrompt] capability.
- `'client'`: the frontend owns the system prompt. Frontend `SystemMessage`s are preserved as-is, and the agent's configured `system_prompt` is not injected — the caller is fully responsible for sending it on every turn if desired. To opt into fallback-to-configured behavior, add the [`ReinjectSystemPrompt`][pydantic_ai.capabilities.ReinjectSystemPrompt] capability to your agent.

```python {title="ag_ui_client_managed_system_prompt.py"}
from fastapi import FastAPI
from starlette.requests import Request
from starlette.responses import Response

from pydantic_ai import Agent
from pydantic_ai.ui.ag_ui import AGUIAdapter

agent = Agent('openai:gpt-5.2')

app = FastAPI()


@app.post('/')
async def run_agent(request: Request) -> Response:
    return await AGUIAdapter.dispatch_request(
        request, agent=agent, manage_system_prompt='client'
    )
```

## Channels

The agent you exposed over AG-UI can also power a bot in Slack or another messaging platform. The [CopilotKit Channels SDK](https://docs.copilotkit.ai/slack/pydantic-ai) receives platform events, runs your agent over AG-UI, and renders its response as native platform content.

!!! note
    CopilotKit maintains the platform setup and deployment instructions. This section shows the Pydantic AI integration; use the [CopilotKit Slack guide for Pydantic AI](https://docs.copilotkit.ai/slack/pydantic-ai/connect) for the complete walkthrough.

### How it fits together

For managed Slack, [CopilotKit Intelligence](https://docs.copilotkit.ai/slack/pydantic-ai) holds the Slack credentials and delivers each turn to a long-running Node process built with [`@copilotkit/channels`](https://www.npmjs.com/package/@copilotkit/channels). That process sends the conversation to your Pydantic AI server over AG-UI and returns the streamed response to Slack.

```
Slack  ──►  CopilotKit Intelligence  ──►  channel process (Node)  ──►  Pydantic AI server (AG-UI)
```

Follow the [CopilotKit guide](https://docs.copilotkit.ai/slack/pydantic-ai/connect) to create the channel process and point its AG-UI client at your Pydantic AI server. When the process handles a platform event, it passes the triggering message to the agent and can attach platform and user details as AG-UI `context` entries.

AG-UI `context` is client-provided data, so Pydantic AI deliberately does not put it in the model prompt automatically. Run the channel process alongside the [`ag_ui_context.py`](#context) server above: it reads `adapter.run_input.context`, keeps authenticated workspace data separate, and exposes the channel entries through the `frontend_context` tool rather than treating them as instructions.

```bash
uvicorn ag_ui_context:app
```

Start the channel process as described in the CopilotKit guide. It needs a long-running host because a serverless request handler cannot own its persistent gateway connection.

### Slack

The managed Slack connection is configured in CopilotKit Intelligence, which walks you through creating the Slack app and holds its credentials. Mention the bot in a real workspace and test a direct message to verify the platform connection, gateway listener, AG-UI server, and reply path together.

### Other platforms

Managed and developer-operated connections have different setup and support. See the [Channels SDK reference](https://docs.copilotkit.ai/reference/channels) for the current managed platforms, direct adapters, and provider-specific guides.

!!! note
    CopilotKit Intelligence reconstructs managed conversation history, but SDK workflow state and interactive callback snapshots use an in-memory store by default. Configure a durable store before promising restart-safe state or interactions; see [Persistence and scaling](https://docs.copilotkit.ai/slack/pydantic-ai/persistence-and-scaling).

## Examples

For more examples see
[`pydantic_ai_examples.ag_ui`](https://github.com/pydantic/pydantic-ai/tree/main/examples/pydantic_ai_examples/ag_ui),
which includes a server for use with the
[AG-UI Dojo](https://docs.ag-ui.com/tutorials/debugging#the-ag-ui-dojo).
