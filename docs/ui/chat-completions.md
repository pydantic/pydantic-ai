# OpenAI Chat Completions Protocol

Pydantic AI supports the [OpenAI Chat Completions API](https://platform.openai.com/docs/api-reference/chat/create) as a UI event stream integration. This lets you receive Chat Completions requests, run a Pydantic AI agent, and return OpenAI-compatible streaming chunks.

## Installation

The only dependencies are:

- [openai](https://github.com/openai/openai-python): to provide Chat Completions request and chunk types.
- [starlette](https://www.starlette.io): to handle [ASGI](https://asgi.readthedocs.io/en/latest/) requests from a framework like FastAPI.

You can install Pydantic AI with the `chat-completions` extra:

```bash
pip/uv-add 'pydantic-ai-slim[chat-completions]'
```

To run the examples you'll also need:

- [uvicorn](https://www.uvicorn.org/) or another ASGI-compatible server

```bash
pip/uv-add uvicorn
```

## Usage

The [`ChatCompletionsAdapter`][pydantic_ai.ui.chat_completions.ChatCompletionsAdapter] converts Chat Completions messages into arguments for [`Agent.run_stream_events()`](../agent.md#running-agents), runs the agent, and transforms native events into `chat.completion.chunk` SSE events.

If you're using Starlette/FastAPI, use [`ChatCompletionsAdapter.dispatch_request()`][pydantic_ai.ui.chat_completions.ChatCompletionsAdapter.dispatch_request]. For lower-level control, use the adapter directly.

### Usage with Starlette/FastAPI

```py {title="dispatch_chat_completions_request.py"}
from fastapi import FastAPI
from starlette.requests import Request
from starlette.responses import Response

from pydantic_ai import Agent
from pydantic_ai.ui.chat_completions import ChatCompletionsAdapter

agent = Agent('openai:gpt-5.2')

app = FastAPI()


@app.post('/v1/chat/completions')
async def chat_completions(request: Request) -> Response:
    return await ChatCompletionsAdapter.dispatch_request(request, agent=agent)
```

### Advanced usage

For non-Starlette frameworks or custom control:

1. [`ChatCompletionsAdapter.build_run_input()`][pydantic_ai.ui.chat_completions.ChatCompletionsAdapter.build_run_input] validates request bytes as Chat Completions input.
2. [`ChatCompletionsAdapter.run_stream()`][pydantic_ai.ui.UIAdapter.run_stream] runs the agent and returns Chat Completions chunks.
3. [`ChatCompletionsAdapter.encode_stream()`][pydantic_ai.ui.UIAdapter.encode_stream] encodes chunks as SSE strings.

```py {title="run_chat_completions_stream.py"}
import json
from http import HTTPStatus

from fastapi import FastAPI
from fastapi.requests import Request
from fastapi.responses import Response, StreamingResponse
from pydantic import ValidationError

from pydantic_ai import Agent
from pydantic_ai.ui import SSE_CONTENT_TYPE
from pydantic_ai.ui.chat_completions import ChatCompletionsAdapter

agent = Agent('openai:gpt-5.2')

app = FastAPI()


@app.post('/v1/chat/completions')
async def chat_completions(request: Request) -> Response:
    accept = request.headers.get('accept', SSE_CONTENT_TYPE)
    try:
        run_input = ChatCompletionsAdapter.build_run_input(await request.body())
    except ValidationError as e:
        return Response(
            content=json.dumps(e.json()),
            media_type='application/json',
            status_code=HTTPStatus.UNPROCESSABLE_ENTITY,
        )

    adapter = ChatCompletionsAdapter(agent=agent, run_input=run_input, accept=accept)
    event_stream = adapter.run_stream()
    sse_stream = adapter.encode_stream(event_stream)

    return StreamingResponse(sse_stream, media_type=accept)
```

### Stand-alone ASGI app

[`ChatCompletionsApp`][pydantic_ai.ui.chat_completions.ChatCompletionsApp] creates a Starlette app and mounts a `POST /v1/chat/completions` endpoint for you.

```py {title="chat_completions_app.py"}
from pydantic_ai import Agent
from pydantic_ai.ui.chat_completions import ChatCompletionsApp

agent = Agent('openai:gpt-5.2')
app = ChatCompletionsApp(agent)
```

Since `app` is an ASGI application, it can be served with:

```shell
uvicorn chat_completions_app:app
```

## Features

### Message conversion

The adapter supports:

- standard `system`, `user`, `assistant`, and `tool` role messages
- assistant tool calls (including function tool calls)
- multimodal user input content parts, including images/audio/files

### Frontend tools

Function tools passed in the request `tools` field are exposed to the agent as frontend tools through an internal [`ExternalToolset`][pydantic_ai.toolsets.ExternalToolset].

### State

Chat Completions requests do not carry frontend state in this integration, so adapter state is always `None`.

### Completion callback

Like other UI adapters, [`ChatCompletionsAdapter.dispatch_request()`][pydantic_ai.ui.chat_completions.ChatCompletionsAdapter.dispatch_request] and [`ChatCompletionsApp`][pydantic_ai.ui.chat_completions.ChatCompletionsApp] support `on_complete`, which receives the final [`AgentRunResult`][pydantic_ai.agent.AgentRunResult].
