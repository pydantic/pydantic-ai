# Agent User Interaction (AG-UI) Protocol

The [Agent User Interaction (AG-UI) Protocol](https://docs.ag-ui.com/introduction)
is an open standard introduced by the
[CopilotKit](https://webflow.copilotkit.ai/blog/introducing-ag-ui-the-protocol-where-agents-meet-users)
team that standardises how frontend applications connect to AI agents through
an open protocol. Think of it as a universal translator for AI-driven systems
no matter what language an agent speaks: AG-UI ensures fluent communication.

The team at [Rocket Science](https://www.rocketscience.gg/), contributed the
[AG-UI integration](#ag-ui-adapter) to make it easy to implement the AG-UI
protocol with PydanticAI agents.

This also includes an [`Agent.to_ag_ui`][pydantic_ai.Agent.to_ag_ui] convenience
method which simplifies the creation of [`FastAGUI`][pydantic_ai.ag_ui.FastAGUI]
for PydanticAI agents, which is built on top of [Starlette](https://www.starlette.io/),
meaning it's fully compatible with any ASGI server.

## AG-UI Adapter

The [Adapter][pydantic_ai.ag_ui.Adapter] class is an adapter between
PydanticAI agents and the AG-UI protocol written in Python. It provides support
for all aspects of spec including:

- [Events](https://docs.ag-ui.com/concepts/events)
- [Messages](https://docs.ag-ui.com/concepts/messages)
- [State Management](https://docs.ag-ui.com/concepts/state)
- [Tools](https://docs.ag-ui.com/concepts/tools)

### Installation

The only dependencies are:

- [ag-ui-protocol](https://docs.ag-ui.com/introduction): to provide the AG-UI
  types and encoder.
- [pydantic](https://pydantic.dev): to validate the request/response messages
- [pydantic-ai](https://ai.pydantic.dev/): to provide the agent framework

To run the examples you'll also need:

- [uvicorn](https://www.uvicorn.org/) or another ASGI compatible server

```bash
pip/uv-add 'uvicorn'
```

You can install PydanticAI with the `ag-ui` extra to ensure you have all the
required AG-UI dependencies:

```bash
pip/uv-add 'pydantic-ai-slim[ag-ui]'
```

### Quick start

```py {title="agent_to_ag_ui.py" py="3.10" hl_lines="17-28"}
"""Basic example for AG-UI with FastAPI and Pydantic AI."""

from __future__ import annotations

from pydantic_ai import Agent

agent = Agent('openai:gpt-4.1', instructions='Be fun!')
app = agent.to_ag_ui()
```

You can run the example with:

```shell
uvicorn agent_to_ag_ui:app --host 0.0.0.0 --port 8000
```

This will expose the agent as an AG-UI server, and you can start sending
requests to it.

### Design

The adapter receives messages in the form of a
[`RunAgentInput`](https://docs.ag-ui.com/sdk/js/core/types#runagentinput)
which describes the details of a request being passed to the agent including
messages and state. These are then converted to PydanticAI types, passed to the
agent which then process the request.

Results from the agent are converted from PydanticAI types to AG-UI events and
streamed back to the caller as Server-Sent Events (SSE).

A user request may require multiple round trips between client UI and PydanticAI
server, depending on the tools and events needed.

In addition to the [Adapter][pydantic_ai.ag_ui.Adapter] there is also
[FastAGUI][pydantic_ai.ag_ui.FastAGUI] which is slim wrapper around
[Starlette](https://www.starlette.io/) providing easy access to run a PydanticAI
server with AG-UI support with any ASGI server.

### Features

To expose a PydanticAI agent as an AG-UI server including state support, you can
use the [`to_ag_ui`][pydantic_ai.agent.Agent.to_ag_ui] method create an ASGI
compatible server.

In the example below we have document state which is shared between the UI and
server using the [`StateDeps`][pydantic_ai.ag_ui.StateDeps] which implements the
[`StateHandler`][pydantic_ai.ag_ui.StateHandler] that can be used to automatically
decode state contained in [`RunAgentInput.state`](https://docs.ag-ui.com/sdk/js/core/types#runagentinput)
when processing requests.

#### State management

The adapter provides full support for
[AG-UI state management](https://docs.ag-ui.com/concepts/state), which enables
real-time synchronization between agents and frontend applications.

```python {title="ag_ui_state.py" py="3.10" hl_lines="18-40"}
"""State example for AG-UI with FastAPI and Pydantic AI."""

from __future__ import annotations

from pydantic import BaseModel

from pydantic_ai import Agent
from pydantic_ai.ag_ui import StateDeps


class DocumentState(BaseModel):
    """State for the document being written."""

    document: str


agent = Agent(
    'openai:gpt-4.1',
    instructions='Be fun!',
    deps_type=StateDeps[DocumentState],
)
app = agent.to_ag_ui(deps=StateDeps(state_type=DocumentState))
```

Since `app` is an ASGI application, it can be used with any ASGI server e.g.

```bash
uvicorn agent_to_ag_ui:app --host 0.0.0.0 --port 8000
```

Since the goal of [`to_ag_ui`][pydantic_ai.agent.Agent.to_ag_ui] is to be a
convenience method, it accepts the same a combination of the arguments require
for:

- [`Adapter`][pydantic_ai.ag_ui.Adapter] constructor
- [`Agent.iter`][pydantic_ai.agent.Agent.iter] method

If you want more control you can either use
[`agent_to_ag_ui`][pydantic_ai.ag_ui.agent_to_ag_ui] helper method or create
and [`Agent`][pydantic_ai.ag_ui.Agent] directly which also provide
the ability to customise [`Starlette`](https://www.starlette.io/applications/#starlette.applications.Starlette)
options.

#### Tools

AG-UI tools are seamlessly provided to the PydanticAI agent, enabling rich
use experiences with frontend user interfaces.

#### Events

The adapter provides the ability for PydanticAI tools to send
[AG-UI events](https://docs.ag-ui.com/concepts/events) simply by defining a tool
which returns a type based off
[`BaseEvent`](https://docs.ag-ui.com/sdk/js/core/events#baseevent) this allows
for custom events and state updates.

```python {title="ag_ui_tool_events.py" py="3.10" hl_lines="34-55"}
"""Tool events example for AG-UI with FastAPI and Pydantic AI."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ag_ui.core import CustomEvent, EventType, StateSnapshotEvent
from pydantic import BaseModel

from pydantic_ai import Agent, RunContext
from pydantic_ai.ag_ui import StateDeps

if TYPE_CHECKING:
    pass


class DocumentState(BaseModel):
    """State for the document being written."""

    document: str


agent = Agent(
    'openai:gpt-4.1',
    instructions='Be fun!',
    deps_type=StateDeps[DocumentState],
)
app = agent.to_ag_ui(deps=StateDeps(state_type=DocumentState))


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
```

### Examples

For more examples of how to use [`Adapter`][pydantic_ai.ag_ui.Adapter] see
[`pydantic_ai_ag_ui_examples`](https://github.com/pydantic/pydantic-ai/tree/main/examples/pydantic_ai_ag_ui_examples),
which includes working server for the with the
[AG-UI Dojo](https://docs.ag-ui.com/tutorials/debugging#the-ag-ui-dojo) which
can be run from a clone of the repo or with the `pydantic-ai-examples` package
installed with either of the following:

```bash
pip/uv-add pydantic-ai-examples
```

Direct, which supports command line flags:

```shell
python -m pydantic_ai_ag_ui_examples.dojo_server --help
usage: dojo_server.py [-h] [--port PORT] [--reload] [--no-reload] [--log-level {critical,error,warning,info,debug,trace}]

PydanticAI AG-UI Dojo server

options:
  -h, --help            show this help message and exit
  --port PORT, -p PORT  Port to run the server on (default: 9000)
  --reload              Enable auto-reload (default: True)
  --no-reload           Disable auto-reload
  --log-level {critical,error,warning,info,debug,trace}
                        Agent log level (default: info)
```

Run with adapter debug logging:

```shell
python -m pydantic_ai_ag_ui_examples.dojo_server --log-level debug
```

Using uvicorn:

```shell
uvicorn pydantic_ai_ag_ui_examples.dojo_server:app --port 9000
```
