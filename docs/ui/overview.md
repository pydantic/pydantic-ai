# UI Event Streams

If you're building a chat app or other interactive frontend for an AI agent, your backend will need to receive agent run input (like a chat message or full message history) from the frontend, and will need to stream the [agent's events](../agents.md#streaming-all-events) (like text, thinking, and tool calls) to the frontend so that the user knows what's happening in real time.

While your frontend could use Pydantic AI's [`ModelRequest`s](../message-history.md) and [`AgentStreamEvent`s][pydantic_ai.messages.AgentStreamEvent] directly, you'll typically want to use a UI event stream protocol that's natively supported by your frontend framework.

Pydantic AI natively supports two UI event stream protocols:

- [Agent User Interaction (AG-UI) Protocol](./ag-ui.md)
- [Vercel AI Data Stream Protocol](./vercel-ai.md)

These integrations are implemented as subclasses of the abstract [`UIAdapter`][pydantic_ai.ui.UIAdapter] class, so they also serve as a reference for integrating with other UI event stream protocols.

## Usage

The protocol-specific [`UIAdapter`][pydantic_ai.ui.UIAdapter] subclass (i.e. [`AGUIAdapter`][pydantic_ai.ui.ag_ui.AGUIAdapter] or [`VercelAIAdapter`][pydantic_ai.ui.vercel_ai.VercelAIAdapter]) is responsible for transforming agent run input received from the frontend into arguments for [`Agent.run_stream_events()`](../agents.md#running-agents), running the agent, and then transforming Pydantic AI events into protocol-specific events. The event stream transformation is handled by a protocol-specific [`UIEventStream`][pydantic_ai.ui.UIEventStream] subclass, but you typically won't use this directly.

If you're using a Starlette-based web framework like FastAPI, you can use the [`UIAdapter.dispatch_request()`][pydantic_ai.ui.UIAdapter.dispatch_request] class method from an endpoint function to directly handle a request and return a streaming response of protocol-specific events. Besides the request, this method takes the agent and supports the same optional arguments as [`Agent.run_stream_events()`](../agents.md#running-agents), including `deps`.

!!! note
    These examples use the `VercelAIAdapter`, but the same patterns apply to all `UIAdapter` subclasses.

```py {title="dispatch_request.py"}
from fastapi import FastAPI
from starlette.requests import Request
from starlette.responses import Response

from pydantic_ai import Agent
from pydantic_ai.ui.vercel_ai import VercelAIAdapter

agent = Agent('openai:gpt-5')

app = FastAPI()

@app.post('/chat')
async def chat(request: Request) -> Response:
    return await VercelAIAdapter.dispatch_request(request, agent=agent)
```

If you're using a web framework not based on Starlette (e.g. Django or Flask) or want fine-grained control over the input or output, you can create and use a `UIAdapter` instance directly.

!!! note
    This example uses FastAPI, but can be modified to work with any web framework.

```py {title="run_stream.py"}
import json
from http import HTTPStatus

from fastapi import FastAPI
from fastapi.requests import Request
from fastapi.responses import Response, StreamingResponse
from pydantic import ValidationError

from pydantic_ai import Agent
from pydantic_ai.ui import SSE_CONTENT_TYPE
from pydantic_ai.ui.vercel_ai import VercelAIAdapter

agent = Agent('openai:gpt-5')

app = FastAPI()


@app.post('/chat')
async def chat(request: Request) -> Response:
    accept = request.headers.get('accept', SSE_CONTENT_TYPE)
    try:
        run_input = VercelAIAdapter.build_run_input(await request.body())  # (1)
    except ValidationError as e:
        return Response(
            content=json.dumps(e.json()),
            media_type='application/json',
            status_code=HTTPStatus.UNPROCESSABLE_ENTITY,
        )

    adapter = VercelAIAdapter(agent=agent, run_input=run_input, accept=accept)
    event_stream = adapter.run_stream() # (2)

    sse_event_stream = adapter.encode_stream(event_stream)
    return StreamingResponse(sse_event_stream, media_type=accept) # (3)
```

1. [`UIAdapter.build_run_input()`][pydantic_ai.ui.UIAdapter.build_run_input] takes the request body as bytes and returns a protocol-specific run input object. You can also use the [`UIAdapter.from_request()`][pydantic_ai.ui.UIAdapter.from_request] class method to build an adapter directly from a request.
2. [`UIAdapter.run_stream()`][pydantic_ai.ui.UIAdapter.run_stream] runs the agent and returns a stream of protocol-specific events. It supports the same optional arguments as [`Agent.run_stream_events()`](../agents.md#running-agents), including `deps`. You can also use [`UIAdapter.run_stream_native()`][pydantic_ai.ui.UIAdapter.run_stream_native] to run the agent and return a stream of Pydantic AI events instead, which can then be transformed into protocol-specific events using [`UIAdapter.transform_stream()`][pydantic_ai.ui.UIAdapter.transform_stream].
3. [`UIAdapter.encode_stream()`][pydantic_ai.ui.UIAdapter.encode_stream] encodes the stream of protocol-specific events as strings according to the accept header value. You can also use [`UIAdapter.streaming_response()`][pydantic_ai.ui.UIAdapter.streaming_response] to generate a streaming response directly from the protocol-specific event stream returned by `run_stream()`.
