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

from pydantic_ai import Agent
from pydantic_ai.ui import SSE_CONTENT_TYPE
from pydantic_ai.ui.ag_ui import AGUIAdapter

agent = Agent('openai:gpt-5.2', instructions='Be fun!')

app = FastAPI()


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
    event_stream = adapter.run_stream() # (2)

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
    If you want to use your own dependencies type to hold AG-UI state as well as other things, it needs to implements the
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

### Tool approval (interrupts)

Tools declared with `requires_approval=True` map onto AG-UI's [interrupt-aware run lifecycle](https://docs.ag-ui.com/concepts/interrupts). When the model proposes such a call, the run pauses and the adapter ends the SSE stream with a `RUN_FINISHED` event whose `outcome.type` is `"interrupt"` and whose `outcome.interrupts[]` describes each pending approval. The client renders an approval UI from that list and POSTs the next `RunAgentInput` with a `resume[]` array of `ResumeEntry` items addressing each interrupt.

The mapping the adapter applies (matching the AG-UI Python SDK field names):

| AG-UI direction         | Pydantic AI source / sink                                                                                       |
| ----------------------- | --------------------------------------------------------------------------------------------------------------- |
| `Interrupt.reason`      | Always `"tool_call"` for `requires_approval=True` tools                                                         |
| `Interrupt.tool_call_id`| The `ToolCallPart.tool_call_id` of the proposed call                                                            |
| `Interrupt.id`          | `f"int-{tool_call_id}"` (round-trips back to `tool_call_id` on resume)                                          |
| `Interrupt.metadata`    | `DeferredToolRequests.metadata.get(tool_call_id)`                                                               |
| `ResumeEntry.payload`   | `{ "approved": bool, "editedArgs"?: object, "reason"?: string }`                                                |
| `payload.approved=True` | [`ToolApproved`][pydantic_ai.tools.ToolApproved]                                                                |
| `payload.editedArgs`    | [`ToolApproved.override_args`][pydantic_ai.tools.ToolApproved.override_args] (fully replaces the proposed args) |
| `payload.approved=False`| [`ToolDenied`][pydantic_ai.tools.ToolDenied] with `message=payload.reason`                                      |
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

### Multimodal tool returns

When you persist and replay conversation history with [`AGUIAdapter.dump_messages`][pydantic_ai.ui.ag_ui.AGUIAdapter.dump_messages] and [`AGUIAdapter.load_messages`][pydantic_ai.ui.ag_ui.AGUIAdapter.load_messages], tool returns containing files — [`BinaryContent`][pydantic_ai.messages.BinaryContent], [`ImageUrl`][pydantic_ai.messages.ImageUrl], and the other [multimodal content types](../input.md#image-audio-video-document-input) — are preserved across the round-trip when you opt in with `preserve_file_data=True`.

AG-UI's `ToolMessage.content` is text-only, so the file payloads can't ride on the tool message itself. Instead they're carried in a reserved sidecar [activity message](https://docs.ag-ui.com/concepts/activities) (`activity_type='pydantic_ai_tool_return_file'`) paired to the tool return by tool call ID, and merged back into the tool return with any text content on reload: a single file with no text reloads as that file alone, otherwise as a list (text-then-files); the original interleaving order between text and files is lost. Activity types prefixed with `pydantic_ai_` are reserved for this round-trip and should be ignored by your frontend's activity handlers.

With the default `preserve_file_data=False`, file content in tool returns — including files nested inside dicts or lists — is dropped on `dump_messages` (any text is kept). This mirrors how client-uploaded files are handled — see [Trust model](#trust-model) below.

!!! note
    This applies to history serialization. During a live streamed run, multimodal tool returns are emitted as text descriptions (e.g. `[File: image/jpeg]`) without the file payload.

### Trust model

AG-UI's `RunAgentInput.messages` is fully client-controlled. The [`AGUIAdapter`][pydantic_ai.ui.ag_ui.AGUIAdapter] applies defaults to strip untrusted parts before the agent runs — see [Trust model for client-submitted messages](./overview.md#trust-model-for-client-submitted-messages) in the UI adapter overview.

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

## Examples

For more examples see
[`pydantic_ai_examples.ag_ui`](https://github.com/pydantic/pydantic-ai/tree/main/examples/pydantic_ai_examples/ag_ui),
which includes a server for use with the
[AG-UI Dojo](https://docs.ag-ui.com/tutorials/debugging#the-ag-ui-dojo).
