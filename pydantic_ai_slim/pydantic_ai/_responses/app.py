"""Starlette request handling for serving an agent as an OpenAI Responses API endpoint."""

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from http import HTTPStatus
from typing import TYPE_CHECKING, TypeVar
from uuid import uuid4

from openai.types.responses import Response as OpenAIResponse, ResponseCompletedEvent, ResponseFailedEvent
from pydantic import ValidationError

from ..agent import AbstractAgent
from ..capabilities import ReinjectSystemPrompt
from ..messages import AgentStreamEvent
from ..models import KnownModelName, Model
from ..run import AgentRunResultEvent
from ..settings import ModelSettings
from ..usage import UsageLimits
from .events import encode_sse, response_event_stream
from .messages import load_messages
from .types import responses_request_ta

if TYPE_CHECKING:
    from starlette.applications import Starlette
    from starlette.requests import Request
    from starlette.responses import Response as StarletteResponse

AgentDepsT = TypeVar('AgentDepsT')
OutputDataT = TypeVar('OutputDataT')

__all__ = ['create_responses_app', 'handle_responses_request']

DEFAULT_PATH = '/v1/responses'


async def handle_responses_request(
    request: Request,
    agent: AbstractAgent[AgentDepsT, OutputDataT],
    *,
    model: Model | KnownModelName | str | None = None,
    deps: AgentDepsT = None,
    model_settings: ModelSettings | None = None,
    instructions: str | None = None,
    usage_limits: UsageLimits | None = None,
) -> StarletteResponse:
    """Run the agent for an incoming Responses API request and return the appropriate response.

    Returns a streaming SSE response when the request sets `stream: true`, otherwise a single
    JSON `Response`. Use this from your own Starlette/FastAPI route when you need to vary `deps`
    or other arguments per request; otherwise use [`create_responses_app`][pydantic_ai._responses.create_responses_app].
    """
    from starlette.responses import JSONResponse, StreamingResponse

    try:
        data = responses_request_ta.validate_json(await request.body())
    except ValidationError as e:
        return JSONResponse(content=e.errors(include_url=False), status_code=HTTPStatus.UNPROCESSABLE_ENTITY)

    message_history = load_messages(data)
    run_instructions = '\n\n'.join(part for part in (instructions, data.instructions) if part) or None

    async def agent_events() -> AsyncIterator[AgentStreamEvent | AgentRunResultEvent[OutputDataT]]:
        async with agent.run_stream_events(
            message_history=message_history,
            model=model,
            deps=deps,
            model_settings=model_settings,
            instructions=run_instructions,
            usage_limits=usage_limits,
            # History is reconstructed from the request, so reinject the agent's configured
            # `system_prompt` when the client didn't supply one (client `system`/`developer`
            # messages stay authoritative).
            capabilities=[ReinjectSystemPrompt()],
        ) as stream:
            async for event in stream:
                yield event

    events = response_event_stream(
        agent_events(),
        model=data.model or '',
        response_id=f'resp_{uuid4().hex}',
        message_id=f'msg_{uuid4().hex}',
        created_at=time.time(),
    )

    if data.stream:
        return StreamingResponse((encode_sse(event) async for event in events), media_type='text/event-stream')

    response: OpenAIResponse | None = None
    async for event in events:
        if isinstance(event, (ResponseCompletedEvent, ResponseFailedEvent)):
            response = event.response
    assert response is not None
    return JSONResponse(content=response.model_dump(mode='json'))


def create_responses_app(
    agent: AbstractAgent[AgentDepsT, OutputDataT],
    *,
    path: str = DEFAULT_PATH,
    model: Model | KnownModelName | str | None = None,
    deps: AgentDepsT = None,
    model_settings: ModelSettings | None = None,
    instructions: str | None = None,
) -> Starlette:
    """Create a Starlette app exposing the agent as an OpenAI Responses API endpoint.

    Args:
        agent: The Pydantic AI agent to serve.
        path: The route the endpoint is mounted at. Defaults to `/v1/responses` so an OpenAI client
            configured with `base_url='.../v1'` works unchanged.
        model: Optional model to use for all requests, required if the agent has no model set.
        deps: Optional dependencies to use for all requests.
        model_settings: Optional settings to use for all model requests.
        instructions: Optional extra instructions to pass to each agent run.

    Returns:
        A configured Starlette application ready to be served.
    """
    try:
        from starlette.applications import Starlette
        from starlette.responses import Response
        from starlette.routing import Route
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            'Please install the `starlette` package to use `Agent.to_responses()`, '
            'you can use the `ui` optional group — `pip install "pydantic-ai-slim[ui]"`'
        ) from e

    async def post_responses(request: Request) -> Response:
        return await handle_responses_request(
            request, agent, model=model, deps=deps, model_settings=model_settings, instructions=instructions
        )

    async def options_responses(request: Request) -> Response:
        return Response()

    return Starlette(
        routes=[
            Route(path, post_responses, methods=['POST']),
            Route(path, options_responses, methods=['OPTIONS']),
        ]
    )
