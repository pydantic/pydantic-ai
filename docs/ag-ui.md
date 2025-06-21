# Agent User Interaction (AG-UI) Protocol

The [Agent User Interaction (AG-UI) Protocol](https://docs.ag-ui.com/introduction)
is an open standard introduced by the
[CopilotKit](https://webflow.copilotkit.ai/blog/introducing-ag-ui-the-protocol-where-agents-meet-users)
team that standardises how front-end applications connect to AI agents through
an open protocol. Think of it as a universal translator for AI-driven systems
no matter what language an agent speaks: AG-UI ensures fluent communication.

The team at [Rocket Science](https://www.rocketscience.gg/), contributed the
[adapter-ag-ui](#adapter-ag-ui) library to make it easy to implement the AG-UI
protocol with PydanticAI agents.

This also includes a convenience method that expose PydanticAI agents as AG-UI
servers - let's have a quick look at how to use it:

```py {title="agent_to_ag_ui.py" py="3.10" hl_lines="17-27"}
"""Basic example for AG-UI with FastAPI and Pydantic AI."""
from __future__ import annotations

from typing import TYPE_CHECKING, Annotated

from adapter_ag_ui import SSE_CONTENT_TYPE
from fastapi import FastAPI, Header
from fastapi.responses import StreamingResponse

from pydantic_ai import Agent

if TYPE_CHECKING:
    from ag_ui.core import RunAgentInput

app = FastAPI(title='AG-UI Endpoint')
agent = Agent('openai:gpt-4.1', instructions='Be fun!')
adapter = agent.to_ag_ui()


@app.post('/')
async def root(
    input_data: RunAgentInput, accept: Annotated[str, Header()] = SSE_CONTENT_TYPE
) -> StreamingResponse:
    return StreamingResponse(
        adapter.run(input_data, accept),
        media_type=SSE_CONTENT_TYPE,
    )
```

You can run the example with:

```shell
uvicorn agent_to_ag_ui:app --host 0.0.0.0 --port 8000
```

This will expose the agent as an AG-UI server, and you can start sending
requests to it.

## Adapter AG UI

[AdapterAGUI][adapter_ag_ui.AdapterAGUI]is an adapter between PydanticAI agents
and the AG-UI protocol written in Python.

### Design

The adapter receives messages in the form of a
[`RunAgentInput`](https://docs.ag-ui.com/sdk/js/core/types#runagentinput)
which describes the details of a request being passed to the agent including
messages and state. These are then converted to PydanticAI types, passed to the
provided agent which then process the request. Results from the agent are
converted from PydanticAI types to AG-UI events and streamed back to the caller
as Server-Sent Events (SSE).

A user request may require multiple round trips between client UI and PydanticAI
server, depending on the tools and events needed.

[AdapterAGUI][adapter_ag_ui.AdapterAGUI] can be used with any ASGI server.

### Installation

[AdapterAGUI][adapter_ag_ui.AdapterAGUI] is available on PyPI as
[`adapter-ag-ui`](https://pypi.org/project/adapter-ag-ui/) so installation is as
simple as:

```bash
pip/uv-add adapter-ag-ui
```

The only dependencies are:

- [ag-ui-protocol](https://docs.ag-ui.com/introduction): to provide the AG-UI
  types and encoder.
- [pydantic](https://pydantic.dev): to validate the request/response messages
- [pydantic-ai](https://ai.pydantic.dev/): to provide the agent framework

To run the examples you'll also need:

- [fastapi](https://fastapi.tiangolo.com/): to provide ASGI compatible server

```bash
pip/uv-add 'fastapi'
```

You can install PydanticAI with the `ag-ui` extra to include **AdapterAGUI**:

```bash
pip/uv-add 'pydantic-ai-slim[ag-ui]'
```

### Usage

To expose a PydanticAI agent as an AG-UI server including state support, you can
use the [`to_ag_ui`][pydantic_ai.agent.Agent.to_ag_ui] method in combination
with fastapi.

In the example below we have document state which is shared between the UI and
server using the [`StateDeps`][adapter_ag_ui.StateDeps] which implements the
[`StateHandler`][adapter_ag_ui.StateHandler] that can be used to automatically
decode state contained in [`RunAgentInput.state`](https://docs.ag-ui.com/sdk/js/core/types#runagentinput)
when processing requests.

#### State management

The adapter provides full support for
[AG-UI state management](https://docs.ag-ui.com/concepts/state), which enables
real-time synchronization between agents and frontend applications.

```python {title="ag_ui_state.py" py="3.10" hl_lines="18-21,28,38"}
"""State example for AG-UI with FastAPI and Pydantic AI."""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated

from adapter_ag_ui import SSE_CONTENT_TYPE, StateDeps
from fastapi import FastAPI, Header
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from pydantic_ai import Agent

if TYPE_CHECKING:
    from ag_ui.core import RunAgentInput


class DocumentState(BaseModel):
    """State for the document being written."""

    document: str


app = FastAPI(title='AG-UI Endpoint')
agent = Agent(
    'openai:gpt-4.1',
    instructions='Be fun!',
    deps_type=StateDeps[DocumentState],
)
adapter = agent.to_ag_ui()


@app.post('/')
async def root(
    input_data: RunAgentInput, accept: Annotated[str, Header()] = SSE_CONTENT_TYPE
) -> StreamingResponse:
    return StreamingResponse(
        adapter.run(input_data, accept, deps=StateDeps(state_type=DocumentState)),
        media_type=SSE_CONTENT_TYPE,
    )

```

Since `app` is an ASGI application, it can be used with any ASGI server.

```bash
uvicorn agent_to_ag_ui:app --host 0.0.0.0 --port 8000
```

Since the goal of [`to_ag_ui`][pydantic_ai.agent.Agent.to_ag_ui] is to be a
convenience method, it accepts the same arguments as the
[`AdapterAGUI`][adapter_ag_ui.AdapterAGUI] constructor.

#### Tools

AG-UI tools are seamlessly provided to the PydanticAI agent, enabling rich
use experiences with frontend user interfaces.

#### Events

The adapter provides the ability for PydanticAI tools to send
[AG-UI events](https://docs.ag-ui.com/concepts/events) simply by defining a tool
which returns a type based off
[`BaseEvent`](https://docs.ag-ui.com/sdk/js/core/events#baseevent) this allows
for custom events and state updates.

```python {title="ag_ui_tool_events.py" py="3.10" hl_lines="35-56"}
"""Tool events example for AG-UI with FastAPI and Pydantic AI."""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated

from adapter_ag_ui import SSE_CONTENT_TYPE, StateDeps
from ag_ui.core import CustomEvent, EventType, StateSnapshotEvent
from fastapi import FastAPI, Header
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from pydantic_ai import Agent, RunContext

if TYPE_CHECKING:
    from ag_ui.core import RunAgentInput


class DocumentState(BaseModel):
    """State for the document being written."""

    document: str


app = FastAPI(title='AG-UI Endpoint')

agent = Agent(
    'openai:gpt-4.1',
    instructions='Be fun!',
    deps_type=StateDeps[DocumentState],
)
adapter = agent.to_ag_ui()


@agent.tool
def update_state(ctx: RunContext[StateDeps[DocumentState]]) -> StateSnapshotEvent:
    return StateSnapshotEvent(
        type=EventType.STATE_SNAPSHOT,
        snapshot=ctx.deps.state,
    )


@agent.tool_plain
def custom_events() -> list[CustomEvent]:
    return [
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


@app.post('/')
async def root(
    input_data: RunAgentInput, accept: Annotated[str, Header()] = SSE_CONTENT_TYPE
) -> StreamingResponse:
    return StreamingResponse(
        adapter.run(input_data, accept, deps=StateDeps(state_type=DocumentState)),
        media_type=SSE_CONTENT_TYPE,
    )

```

### Examples

For more examples of how to use [`AdapterAGUI`][adapter_ag_ui.AdapterAGUI] see
[`adapter_ag_ui_examples`](https://github.com/pydantic/pydantic-ai/tree/main/examples/adapter_ag_ui_examples),
which includes working server for the with the
[AG-UI Dojo](https://docs.ag-ui.com/tutorials/debugging#the-ag-ui-dojo) which
can be run from a clone of the repo or with the `adapter_ag_ui_examples` package
installed with either of the following:

```bash
pip/uv-add 'adapter_ag_ui_examples'
```

Direct:

```shell
python -m adapter_ag_ui_examples.dojo_server
```

Using uvicorn:

```shell
uvicorn adapter_ag_ui_examples.dojo_server:app --host 0.0.0.0 --port 8000
```
