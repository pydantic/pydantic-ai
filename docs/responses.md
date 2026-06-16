# OpenAI Responses Endpoint

Pydantic AI can serve an agent as an [OpenAI Responses API](https://platform.openai.com/docs/api-reference/responses) endpoint, so any OpenAI-compatible client - the [`openai`](https://github.com/openai/openai-python) SDK pointed at a custom `base_url`, [OpenWebUI](https://openwebui.com/), an LLM gateway, or a framework configured against a custom endpoint - can talk to your agent as if it were an OpenAI model.

Unlike the [UI event stream protocols](ui/overview.md) (AG-UI and Vercel AI), which are designed for interactive frontends and stream the agent's tool calls and reasoning as distinct UI elements, the Responses endpoint exposes the agent as a single model: the agent runs its own [tool loop](tools.md) server-side and only its final text is returned. Internal tool calls and reasoning are not surfaced, since in the Responses protocol a tool call is a request for the *client* to act, which would not make sense for tools the agent has already executed itself.

!!! note
    Both streaming (`stream=True`) and non-streaming requests are supported. The Responses API defaults to non-streaming.

## Usage

The quickest way to serve an agent is [`Agent.to_responses()`][pydantic_ai.agent.Agent.to_responses], which returns a [Starlette](https://www.starlette.io/) application you can run with any ASGI server (such as [Uvicorn](https://www.uvicorn.org/)) or mount into a larger app. The endpoint is served at `/v1/responses` by default, so an OpenAI client configured with `base_url='.../v1'` works unchanged.

```py {title="to_responses.py"}
from pydantic_ai import Agent

agent = Agent('openai:gpt-5.2', instructions='Be concise.')

app = agent.to_responses()
```

Run it with `uvicorn to_responses:app`, then point any OpenAI client at it:

```py {title="client.py" test="skip"}
from openai import OpenAI

client = OpenAI(base_url='http://localhost:8000/v1', api_key='<not used by the agent>')

response = client.responses.create(model='gpt-5.2', input='Where does the sun rise?')
print(response.output_text)
```

The `model` field in the request is informational - the agent's own model is always used - so clients can send any value. Client-provided `system`/`developer` messages and the `instructions` field are honored, in line with the Responses API.

### Usage with Starlette/FastAPI

[`Agent.to_responses()`][pydantic_ai.agent.Agent.to_responses] uses the same `deps` and `model_settings` for every request. To vary them per request - for example to set [dependencies](dependencies.md) based on the authenticated user - use [`handle_responses_request()`][pydantic_ai._responses.handle_responses_request] from your own endpoint instead. It runs the agent for the request and returns either a streaming or a JSON response depending on the request's `stream` field.

```py {title="handle_request.py"}
from fastapi import FastAPI
from starlette.requests import Request
from starlette.responses import Response

from pydantic_ai import Agent
from pydantic_ai._responses import handle_responses_request

agent = Agent('openai:gpt-5.2')

app = FastAPI()


@app.post('/v1/responses')
async def responses(request: Request) -> Response:
    return await handle_responses_request(request, agent)
```
