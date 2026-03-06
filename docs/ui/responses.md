# OpenAI Responses Protocol

Pydantic AI supports the [OpenAI Responses API](https://platform.openai.com/docs/api-reference/responses) as a UI event stream integration. This lets you receive Responses API requests from a frontend or OpenAI-compatible client, run a Pydantic AI agent, and stream back Responses-compatible events.

## Installation

The only dependencies are:

- [openai](https://github.com/openai/openai-python): to provide Responses API request and event types.
- [starlette](https://www.starlette.io): to handle [ASGI](https://asgi.readthedocs.io/en/latest/) requests from a framework like FastAPI.

You can install Pydantic AI with the `responses` extra:

```bash
pip/uv-add 'pydantic-ai-slim[responses]'
```

To run the examples you'll also need:

- [uvicorn](https://www.uvicorn.org/) or another ASGI-compatible server

```bash
pip/uv-add uvicorn
```

## Usage

The [`ResponsesAdapter`][pydantic_ai.ui.responses.ResponsesAdapter] converts Responses API input into arguments for [`Agent.run_stream_events()`](../agent.md#running-agents), runs the agent, and transforms native events into Responses API stream events.

If you're using Starlette/FastAPI, the simplest approach is [`ResponsesAdapter.dispatch_request()`][pydantic_ai.ui.responses.ResponsesAdapter.dispatch_request]. If you need more control, you can build and run the adapter directly.

### Usage with Starlette/FastAPI

```py {title="dispatch_responses_request.py"}
from fastapi import FastAPI
from starlette.requests import Request
from starlette.responses import Response

from pydantic_ai import Agent
from pydantic_ai.ui.responses import ResponsesAdapter

agent = Agent('openai:gpt-5.2')

app = FastAPI()


@app.post('/v1/responses')
async def responses(request: Request) -> Response:
    return await ResponsesAdapter.dispatch_request(request, agent=agent)
```

### Advanced usage

If you're not using Starlette/FastAPI, or you need control over parsing/response generation, use the adapter methods directly:

1. [`ResponsesAdapter.build_run_input()`][pydantic_ai.ui.responses.ResponsesAdapter.build_run_input] parses and validates request bytes as Responses API input.
2. [`ResponsesAdapter.run_stream()`][pydantic_ai.ui.UIAdapter.run_stream] runs the agent and returns a Responses event stream.
3. [`ResponsesAdapter.encode_stream()`][pydantic_ai.ui.UIAdapter.encode_stream] encodes that stream as SSE text.

```py {title="run_responses_stream.py"}
import json
from http import HTTPStatus

from fastapi import FastAPI
from fastapi.requests import Request
from fastapi.responses import Response, StreamingResponse
from pydantic import ValidationError

from pydantic_ai import Agent
from pydantic_ai.ui import SSE_CONTENT_TYPE
from pydantic_ai.ui.responses import ResponsesAdapter

agent = Agent('openai:gpt-5.2')

app = FastAPI()


@app.post('/v1/responses')
async def responses(request: Request) -> Response:
    accept = request.headers.get('accept', SSE_CONTENT_TYPE)
    try:
        run_input = ResponsesAdapter.build_run_input(await request.body())
    except ValidationError as e:
        return Response(
            content=json.dumps(e.json()),
            media_type='application/json',
            status_code=HTTPStatus.UNPROCESSABLE_ENTITY,
        )

    adapter = ResponsesAdapter(agent=agent, run_input=run_input, accept=accept)
    event_stream = adapter.run_stream()
    sse_stream = adapter.encode_stream(event_stream)

    return StreamingResponse(sse_stream, media_type=accept)
```

### Stand-alone ASGI app

[`ResponsesApp`][pydantic_ai.ui.responses.ResponsesApp] creates a Starlette app and mounts a `POST /v1/responses` endpoint for you.

```py {title="responses_app.py"}
from pydantic_ai import Agent
from pydantic_ai.ui.responses import ResponsesApp

agent = Agent('openai:gpt-5.2')
app = ResponsesApp(agent)
```

Since `app` is an ASGI application, it can be served with:

```shell
uvicorn responses_app:app
```

## Features

### Input formats

The Responses adapter supports the standard Responses API input patterns:

- string input (mapped to a user message)
- list input with message/function output items
- `instructions` (mapped to system prompt content)

### Tool definitions

Function tools in `tools` are exposed to the agent as frontend tools through an internal [`ExternalToolset`][pydantic_ai.toolsets.ExternalToolset].

### State from metadata

Responses request `metadata` is exposed as frontend state. If your deps type implements [`StateHandler`][pydantic_ai.ui.StateHandler] (or uses [`StateDeps`][pydantic_ai.ui.StateDeps]), that state is injected into `deps.state` for each request.

### Completion callback

Like other UI adapters, [`ResponsesAdapter.dispatch_request()`][pydantic_ai.ui.responses.ResponsesAdapter.dispatch_request] and [`ResponsesApp`][pydantic_ai.ui.responses.ResponsesApp] support `on_complete`, which receives the final [`AgentRunResult`][pydantic_ai.agent.AgentRunResult].
