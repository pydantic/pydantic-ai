# Agent User Interaction (AG-UI) Protocol

The [Agent User Interaction (AG-UI) Protocol](https://docs.ag-ui.com/introduction) is an open standard introduced by the
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

- [uvicorn](https://www.uvicorn.org/) or another ASGI compatible server

```bash
pip/uv-add uvicorn
```

## Usage

There are three ways to run a Pydantic AI agent based on AG-UI run input with streamed AG-UI events as output, from most to least flexible. If you're using a Starlette-based web framework like FastAPI, you'll typically want to use the second method.

1. [`run_ag_ui()`][pydantic_ai.ag_ui.run_ag_ui] takes an agent and an AG-UI [`RunAgentInput`](https://docs.ag-ui.com/sdk/python/core/types#runagentinput) object, and returns a stream of AG-UI events encoded as strings. It also takes optional [`Agent.iter()`][pydantic_ai.Agent.iter] arguments including `deps`. Use this if you're using a web framework not based on Starlette (e.g. Django or Flask) or want to modify the input or output some way.
2. [`handle_ag_ui_request()`][pydantic_ai.ag_ui.handle_ag_ui_request] takes an agent and a Starlette request (e.g. from FastAPI) coming from an AG-UI frontend, and returns a streaming Starlette response of AG-UI events that you can return directly from your endpoint. It also takes optional [`Agent.iter()`][pydantic_ai.Agent.iter] arguments including `deps`, that you can vary for each request (e.g. based on the authenticated user).
3. [`Agent.to_ag_ui()`][pydantic_ai.agent.AbstractAgent.to_ag_ui] returns an ASGI application that handles every AG-UI request by running the agent. It also takes optional [`Agent.iter()`][pydantic_ai.Agent.iter] arguments including `deps`, but these will be the same for each request, with the exception of the AG-UI state that's injected as described under [state management](#state-management). This ASGI app can be [mounted](https://fastapi.tiangolo.com/advanced/sub-applications/) at a given path in an existing FastAPI app.

### Handle run input and output directly

This example uses [`run_ag_ui()`][pydantic_ai.ag_ui.run_ag_ui] and performs its own request parsing and response generation.
This can be modified to work with any web framework.

```py {title="run_ag_ui.py"}
import json
from http import HTTPStatus

from ag_ui.core import RunAgentInput
from fastapi import FastAPI
from fastapi.requests import Request
from fastapi.responses import Response, StreamingResponse
from pydantic import ValidationError

from pydantic_ai import Agent
from pydantic_ai.ag_ui import SSE_CONTENT_TYPE, run_ag_ui

agent = Agent('openai:gpt-5', instructions='Be fun!')

app = FastAPI()


@app.post('/')
async def run_agent(request: Request) -> Response:
    accept = request.headers.get('accept', SSE_CONTENT_TYPE)
    try:
        run_input = RunAgentInput.model_validate(await request.json())
    except ValidationError as e:  # pragma: no cover
        return Response(
            content=json.dumps(e.json()),
            media_type='application/json',
            status_code=HTTPStatus.UNPROCESSABLE_ENTITY,
        )

    event_stream = run_ag_ui(agent, run_input, accept=accept)

    return StreamingResponse(event_stream, media_type=accept)
```

Since `app` is an ASGI application, it can be used with any ASGI server:

```shell
uvicorn run_ag_ui:app
```

This will expose the agent as an AG-UI server, and your frontend can start sending requests to it.

### Handle a Starlette request

This example uses [`handle_ag_ui_request()`][pydantic_ai.ag_ui.run_ag_ui] to directly handle a FastAPI request and return a response. Something analogous to this will work with any Starlette-based web framework.

```py {title="handle_ag_ui_request.py"}
from fastapi import FastAPI
from starlette.requests import Request
from starlette.responses import Response

from pydantic_ai import Agent
from pydantic_ai.ag_ui import handle_ag_ui_request

agent = Agent('openai:gpt-5', instructions='Be fun!')

app = FastAPI()

@app.post('/')
async def run_agent(request: Request) -> Response:
    return await handle_ag_ui_request(agent, request)
```

Since `app` is an ASGI application, it can be used with any ASGI server:

```shell
uvicorn handle_ag_ui_request:app
```

This will expose the agent as an AG-UI server, and your frontend can start sending requests to it.

### Stand-alone ASGI app

This example uses [`Agent.to_ag_ui()`][pydantic_ai.agent.AbstractAgent.to_ag_ui] to turn the agent into a stand-alone ASGI application:

```py {title="agent_to_ag_ui.py" hl_lines="4"}
from pydantic_ai import Agent

agent = Agent('openai:gpt-5', instructions='Be fun!')
app = agent.to_ag_ui()
```

Since `app` is an ASGI application, it can be used with any ASGI server:

```shell
uvicorn agent_to_ag_ui:app
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
server using the [`StateDeps`][pydantic_ai.ag_ui.StateDeps] [dependencies type](./dependencies.md) that can be used to automatically
validate state contained in [`RunAgentInput.state`](https://docs.ag-ui.com/sdk/js/core/types#runagentinput) using a Pydantic `BaseModel` specified as a generic parameter.

!!! note "Custom dependencies type with AG-UI state"
    If you want to use your own dependencies type to hold AG-UI state as well as other things, it needs to implements the
    [`StateHandler`][pydantic_ai.ag_ui.StateHandler] protocol, meaning it needs to be a [dataclass](https://docs.python.org/3/library/dataclasses.html) with a non-optional `state` field. This lets Pydantic AI ensure that state is properly isolated between requests by building a new dependencies object each time.

    If the `state` field's type is a Pydantic `BaseModel` subclass, the raw state dictionary on the request is automatically validated. If not, you can validate the raw value yourself in your dependencies dataclass's `__post_init__` method.


```python {title="ag_ui_state.py"}
from pydantic import BaseModel

from pydantic_ai import Agent
from pydantic_ai.ag_ui import StateDeps


class DocumentState(BaseModel):
    """State for the document being written."""

    document: str = ''


agent = Agent(
    'openai:gpt-5',
    instructions='Be fun!',
    deps_type=StateDeps[DocumentState],
)
app = agent.to_ag_ui(deps=StateDeps(DocumentState()))
```

Since `app` is an ASGI application, it can be used with any ASGI server:

```bash
uvicorn ag_ui_state:app --host 0.0.0.0 --port 9000
```

### Tools

AG-UI frontend tools are seamlessly provided to the Pydantic AI agent, enabling rich
user experiences with frontend user interfaces.

### Events

Pydantic AI tools can send [AG-UI events](https://docs.ag-ui.com/concepts/events) simply by returning a
[`ToolReturn`](tools-advanced.md#advanced-tool-returns) object with a
[`BaseEvent`](https://docs.ag-ui.com/sdk/python/core/events#baseevent) (or a list of events) as `metadata`,
which allows for custom events and state updates.

```python {title="ag_ui_tool_events.py"}
from ag_ui.core import CustomEvent, EventType, StateSnapshotEvent
from pydantic import BaseModel

from pydantic_ai import Agent, RunContext, ToolReturn
from pydantic_ai.ag_ui import StateDeps


class DocumentState(BaseModel):
    """State for the document being written."""

    document: str = ''


agent = Agent(
    'openai:gpt-5',
    instructions='Be fun!',
    deps_type=StateDeps[DocumentState],
)
app = agent.to_ag_ui(deps=StateDeps(DocumentState()))


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

## Examples

For more examples of how to use [`to_ag_ui()`][pydantic_ai.agent.AbstractAgent.to_ag_ui] see
[`pydantic_ai_examples.ag_ui`](https://github.com/pydantic/pydantic-ai/tree/main/examples/pydantic_ai_examples/ag_ui),
which includes a server for use with the
[AG-UI Dojo](https://docs.ag-ui.com/tutorials/debugging#the-ag-ui-dojo).
