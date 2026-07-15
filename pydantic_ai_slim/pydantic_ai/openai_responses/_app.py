"""Starlette request handling for serving an agent as an OpenAI Responses API endpoint."""

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from http import HTTPStatus
from typing import TYPE_CHECKING, Literal, TypeVar
from uuid import uuid4

from openai.types.responses import Response as OpenAIResponse, ResponseCompletedEvent
from pydantic import ValidationError

from .. import exceptions
from ..agent import AbstractAgent
from ..capabilities import ReinjectSystemPrompt
from ..messages import AgentStreamEvent
from ..models import KnownModelName, Model
from ..run import AgentRunResultEvent
from ..settings import ModelSettings
from ..usage import UsageLimits
from ._events import encode_sse, response_event_stream
from ._messages import OrphanedFunctionCallOutputError, load_messages
from .types import responses_request_ta

if TYPE_CHECKING:
    from starlette.applications import Starlette
    from starlette.requests import Request
    from starlette.responses import Response as StarletteResponse

AgentDepsT = TypeVar('AgentDepsT')
OutputDataT = TypeVar('OutputDataT')
OpenAIErrorType = Literal['invalid_request_error', 'server_error']

__all__ = ['handle_openai_responses_request']

DEFAULT_PATH = '/v1/responses'
_MISSING_DEPS_MESSAGE = (
    'Please install the `starlette` and `openai` packages to use `Agent.to_openai_responses()`, '
    'you can use the `ui` and `openai` optional groups — `pip install "pydantic-ai-slim[ui,openai]"`'
)


def _openai_error_response(
    message: str,
    *,
    error_type: OpenAIErrorType,
    status_code: int,
    param: str | None = None,
    code: str | None = None,
) -> StarletteResponse:
    from starlette.responses import JSONResponse

    return JSONResponse(
        content={'error': {'message': message, 'type': error_type, 'param': param, 'code': code}},
        status_code=status_code,
    )


def _validation_error_response(error: ValidationError) -> StarletteResponse:
    first_error = error.errors(include_url=False)[0]
    param = '.'.join(str(part) for part in first_error['loc']) or None
    message = f'Invalid request: {param}: {first_error["msg"]}' if param else f'Invalid request: {first_error["msg"]}'
    return _openai_error_response(
        message,
        error_type='invalid_request_error',
        status_code=HTTPStatus.BAD_REQUEST,
        param=param,
    )


def _resolved_model_name(
    agent: AbstractAgent[AgentDepsT, OutputDataT],
    model: Model | KnownModelName | str | None,
    requested_model: str | None,
) -> str:
    served_model = model if model is not None else agent.model
    if served_model is None:
        return requested_model or ''
    return served_model.model_name if isinstance(served_model, Model) else served_model


async def handle_openai_responses_request(
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
    or other arguments per request; otherwise use [`Agent.to_openai_responses()`][pydantic_ai.agent.AbstractAgent.to_openai_responses].
    """
    from starlette.responses import JSONResponse, StreamingResponse

    try:
        data = responses_request_ta.validate_json(await request.body())
    except ValidationError as e:
        return _validation_error_response(e)

    if data.previous_response_id is not None:
        return _openai_error_response(
            '`previous_response_id` is not supported: server-side conversation state is not stored, replay the conversation via `input` instead.',
            error_type='invalid_request_error',
            status_code=HTTPStatus.BAD_REQUEST,
            param='previous_response_id',
        )

    try:
        message_history = load_messages(data)
    except OrphanedFunctionCallOutputError as e:
        return _openai_error_response(
            str(e),
            error_type='invalid_request_error',
            status_code=HTTPStatus.BAD_REQUEST,
            param='input',
        )
    run_instructions = '\n\n'.join(part for part in (instructions, data.instructions) if part) or None
    response_model = _resolved_model_name(agent, model, data.model)

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
        model=response_model,
        response_id=f'resp_{uuid4().hex}',
        message_id=f'msg_{uuid4().hex}',
        created_at=time.time(),
        catch_run_errors=data.stream,
    )

    if data.stream:
        return StreamingResponse((encode_sse(event) async for event in events), media_type='text/event-stream')

    response: OpenAIResponse | None = None
    try:
        async for event in events:
            if isinstance(event, ResponseCompletedEvent):
                response = event.response
    except exceptions.UserError as e:
        return _openai_error_response(
            str(e),
            error_type='invalid_request_error',
            status_code=HTTPStatus.BAD_REQUEST,
        )
    except Exception as e:
        return _openai_error_response(str(e), error_type='server_error', status_code=HTTPStatus.INTERNAL_SERVER_ERROR)
    assert response is not None
    return JSONResponse(content=response.model_dump(mode='json'))


def create_app(
    agent: AbstractAgent[AgentDepsT, OutputDataT],
    *,
    path: str = DEFAULT_PATH,
    model: Model | KnownModelName | str | None = None,
    deps: AgentDepsT = None,
    model_settings: ModelSettings | None = None,
    instructions: str | None = None,
    usage_limits: UsageLimits | None = None,
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
        usage_limits: Optional limits on model request count or token usage.

    Returns:
        A configured Starlette application ready to be served.
    """
    try:
        from starlette.applications import Starlette
        from starlette.responses import Response
        from starlette.routing import Route
    except ImportError as e:  # pragma: no cover
        raise ImportError(_MISSING_DEPS_MESSAGE) from e

    async def post_responses(request: Request) -> Response:
        return await handle_openai_responses_request(
            request,
            agent,
            model=model,
            deps=deps,
            model_settings=model_settings,
            instructions=instructions,
            usage_limits=usage_limits,
        )

    async def options_responses(request: Request) -> Response:
        return Response()

    return Starlette(
        routes=[
            Route(path, post_responses, methods=['POST']),
            Route(path, options_responses, methods=['OPTIONS']),
        ]
    )
