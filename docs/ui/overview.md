# UI Event Streams

If you're building a chat app or other interactive frontend for an AI agent, your backend will need to receive agent run input (like a chat message or complete [message history](../message-history.md)) from the frontend, and will need to stream the [agent's events](../agent.md#streaming-all-events) (like text, thinking, and tool calls) to the frontend so that the user knows what's happening in real time.

While your frontend could use Pydantic AI's [`ModelRequest`][pydantic_ai.messages.ModelRequest] and [`AgentStreamEvent`][pydantic_ai.messages.AgentStreamEvent] directly, you'll typically want to use a UI event stream protocol that's natively supported by your frontend framework.

Pydantic AI natively supports two UI event stream protocols:

- [Agent-User Interaction (AG-UI) Protocol](./ag-ui.md)
- [Vercel AI Data Stream Protocol](./vercel-ai.md)

These integrations are implemented as subclasses of the abstract [`UIAdapter`][pydantic_ai.ui.UIAdapter] class, so they also serve as a reference for integrating with other UI event stream protocols.

## Usage

The protocol-specific [`UIAdapter`][pydantic_ai.ui.UIAdapter] subclass (i.e. [`AGUIAdapter`][pydantic_ai.ui.ag_ui.AGUIAdapter] or [`VercelAIAdapter`][pydantic_ai.ui.vercel_ai.VercelAIAdapter]) is responsible for transforming agent run input received from the frontend into arguments for [`Agent.run_stream_events()`](../agent.md#running-agents), running the agent, and then transforming Pydantic AI events into protocol-specific events. The event stream transformation is handled by a protocol-specific [`UIEventStream`][pydantic_ai.ui.UIEventStream] subclass, but you typically won't use this directly.

If you're using a Starlette-based web framework like FastAPI, you can use the [`UIAdapter.dispatch_request()`][pydantic_ai.ui.UIAdapter.dispatch_request] class method from an endpoint function to directly handle a request and return a streaming response of protocol-specific events. This is demonstrated in the next section.

If you're using a web framework not based on Starlette (e.g. Django or Flask) or need fine-grained control over the input or output, you can create a `UIAdapter` instance and directly use its methods. This is demonstrated in "Advanced Usage" section below.

### Usage with Starlette/FastAPI

Besides the request, [`UIAdapter.dispatch_request()`][pydantic_ai.ui.UIAdapter.dispatch_request] takes the agent, the same optional arguments as [`Agent.run_stream_events()`](../agent.md#running-agents), and an optional `on_complete` callback function that receives the completed [`AgentRunResult`][pydantic_ai.agent.AgentRunResult] and can optionally yield additional protocol-specific events.

!!! note
    These examples use the `VercelAIAdapter`, but the same patterns apply to all `UIAdapter` subclasses.

```py {title="dispatch_request.py"}
from fastapi import FastAPI
from starlette.requests import Request
from starlette.responses import Response

from pydantic_ai import Agent
from pydantic_ai.ui.vercel_ai import VercelAIAdapter

agent = Agent('openai:gpt-5.2')

app = FastAPI()

@app.post('/chat')
async def chat(request: Request) -> Response:
    return await VercelAIAdapter.dispatch_request(request, agent=agent)
```

### Advanced Usage

If you're using a web framework not based on Starlette (e.g. Django or Flask) or need fine-grained control over the input or output, you can create a `UIAdapter` instance and directly use its methods, which can be chained to accomplish the same thing as the `UIAdapter.dispatch_request()` class method shown above:

1. The [`UIAdapter.build_run_input()`][pydantic_ai.ui.UIAdapter.build_run_input] class method takes the request body as bytes and returns a protocol-specific run input object, which you can then pass to the [`UIAdapter()`][pydantic_ai.ui.UIAdapter] constructor along with the agent.
    - You can also use the [`UIAdapter.from_request()`][pydantic_ai.ui.UIAdapter.from_request] class method to build an adapter directly from a Starlette/FastAPI request.
2. The [`UIAdapter.run_stream()`][pydantic_ai.ui.UIAdapter.run_stream] method runs the agent and returns a stream of protocol-specific events. It supports the same optional arguments as [`Agent.run_stream_events()`](../agent.md#running-agents) and an optional `on_complete` callback function that receives the completed [`AgentRunResult`][pydantic_ai.agent.AgentRunResult] and can optionally yield additional protocol-specific events.
    - You can also use [`UIAdapter.run_stream_native()`][pydantic_ai.ui.UIAdapter.run_stream_native] to run the agent and return a stream of Pydantic AI events instead, which can then be transformed into protocol-specific events using [`UIAdapter.transform_stream()`][pydantic_ai.ui.UIAdapter.transform_stream].
3. The [`UIAdapter.encode_stream()`][pydantic_ai.ui.UIAdapter.encode_stream] method encodes the stream of protocol-specific events as SSE (HTTP Server-Sent Events) strings, which you can then return as a streaming response.
    - You can also use [`UIAdapter.streaming_response()`][pydantic_ai.ui.UIAdapter.streaming_response] to generate a Starlette/FastAPI streaming response directly from the protocol-specific event stream returned by `run_stream()`.

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

agent = Agent('openai:gpt-5.2')

app = FastAPI()


@app.post('/chat')
async def chat(request: Request) -> Response:
    accept = request.headers.get('accept', SSE_CONTENT_TYPE)
    try:
        run_input = VercelAIAdapter.build_run_input(await request.body())
    except ValidationError as e:
        return Response(
            content=json.dumps(e.json()),
            media_type='application/json',
            status_code=HTTPStatus.UNPROCESSABLE_ENTITY,
        )

    adapter = VercelAIAdapter(agent=agent, run_input=run_input, accept=accept)
    event_stream = adapter.run_stream()

    sse_event_stream = adapter.encode_stream(event_stream)
    return StreamingResponse(sse_event_stream, media_type=accept)
```

## Trust model for client-submitted messages

UI adapter endpoints aren't authentication boundaries. Both the AG-UI and Vercel AI protocols are designed around the client transmitting the full conversation history on each request, so anything in `message_history` from the protocol — assistant messages, tool calls, file URLs, tool results — is under the caller's control. Treat the adapter endpoint as an internal backend service, running it inside your own authenticated route handler. See the [AG-UI security considerations](https://learn.microsoft.com/en-us/agent-framework/integrations/ag-ui/security-considerations) page for more on the deployment model both protocols assume.

The adapters apply a few defaults so that the authoritative state stays on your side:

- **System prompts** — client-submitted [`SystemPromptPart`][pydantic_ai.messages.SystemPromptPart]s are stripped by default and replaced with the agent's configured prompt. Control with [`UIAdapter.manage_system_prompt`][pydantic_ai.ui.UIAdapter.manage_system_prompt]; see each adapter's docs for details.
- **Dangling tool calls** — if the client-submitted history ends in a [`ModelResponse`][pydantic_ai.messages.ModelResponse] with unresolved [`ToolCallPart`][pydantic_ai.messages.ToolCallPart]s and no matching `deferred_tool_results`, the tool calls are dropped with a warning so the agent doesn't execute tool calls that the model never emitted. For human-in-the-loop resumption, pass explicit `deferred_tool_results` to the run method — tool calls resolved by those results are kept.
- **File URL schemes** — only `http` and `https` are accepted by default for [`FileUrl`][pydantic_ai.messages.FileUrl] parts in client-submitted messages. Non-HTTP schemes like `s3://` or `gs://` are dropped, since they cause the provider to fetch the object using your server's IAM role or service account. See [`UIAdapter.allowed_file_url_schemes`][pydantic_ai.ui.UIAdapter.allowed_file_url_schemes].

For stricter conversation integrity (e.g. ensuring prior assistant turns and tool returns match what the server actually produced), persist the history server-side keyed by the thread/session ID and pass it to the adapter via `message_history` — caller-supplied history is trusted as coming from server-side persistence and isn't subject to this sanitization.
