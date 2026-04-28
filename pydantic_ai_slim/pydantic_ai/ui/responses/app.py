"""OpenAI Responses protocol integration for Pydantic AI agents."""

from __future__ import annotations

import json
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import replace
from typing import Any, Generic

from typing_extensions import Self

from pydantic_ai import DeferredToolResults
from pydantic_ai._utils import is_str_dict
from pydantic_ai.agent import AbstractAgent
from pydantic_ai.builtin_tools import AbstractBuiltinTool
from pydantic_ai.messages import ModelMessage
from pydantic_ai.models import KnownModelName, Model
from pydantic_ai.output import OutputDataT, OutputSpec
from pydantic_ai.settings import ModelSettings
from pydantic_ai.tools import AgentDepsT
from pydantic_ai.toolsets import AbstractToolset
from pydantic_ai.usage import RunUsage, UsageLimits

from .. import OnCompleteFunc, StateHandler
from ._adapter import ResponsesAdapter

try:
    from starlette.applications import Starlette
    from starlette.middleware import Middleware
    from starlette.requests import Request
    from starlette.responses import JSONResponse, Response
    from starlette.routing import BaseRoute
    from starlette.types import ExceptionHandler, Lifespan
except ImportError as e:  # pragma: no cover
    raise ImportError(
        'Please install the `starlette` package to use `ResponsesApp`, '
        'you can use the `responses` optional group — `pip install "pydantic-ai-slim[responses]"`'
    ) from e


__all__ = ['ResponsesApp', 'gateway']


class ResponsesApp(Generic[AgentDepsT, OutputDataT], Starlette):
    """ASGI application for running Pydantic AI agents with OpenAI Responses protocol support."""

    def __init__(
        self,
        agent: AbstractAgent[AgentDepsT, OutputDataT],
        *,
        # ResponsesAdapter.dispatch_request parameters
        output_type: OutputSpec[Any] | None = None,
        message_history: Sequence[ModelMessage] | None = None,
        deferred_tool_results: DeferredToolResults | None = None,
        model: Model | KnownModelName | str | None = None,
        deps: AgentDepsT = None,
        deps_factory: Callable[[Request], AgentDepsT | Awaitable[AgentDepsT]] | None = None,
        model_settings: ModelSettings | None = None,
        usage_limits: UsageLimits | None = None,
        usage: RunUsage | None = None,
        infer_name: bool = True,
        toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
        builtin_tools: Sequence[AbstractBuiltinTool] | None = None,
        on_complete: OnCompleteFunc[Any] | None = None,
        # Starlette parameters
        debug: bool = False,
        routes: Sequence[BaseRoute] | None = None,
        middleware: Sequence[Middleware] | None = None,
        exception_handlers: Mapping[Any, ExceptionHandler] | None = None,
        on_startup: Sequence[Callable[[], Any]] | None = None,
        on_shutdown: Sequence[Callable[[], Any]] | None = None,
        lifespan: Lifespan[Self] | None = None,
    ) -> None:
        """An ASGI application that handles Responses requests by running the agent and streaming the response.

        Note that the `deps` will be the same for each request, with the exception of the frontend state that's
        injected into the `state` field of a `deps` object that implements the [`StateHandler`][pydantic_ai.ui.StateHandler] protocol.
        To provide different `deps` for each request (e.g. based on the authenticated user),
        use [`ResponsesAdapter.run_stream()`][pydantic_ai.ui.responses.ResponsesAdapter.run_stream] or
        [`ResponsesAdapter.dispatch_request()`][pydantic_ai.ui.responses.ResponsesAdapter.dispatch_request] instead.

        Args:
            agent: The agent to run.

            output_type: Custom output type to use for this run, `output_type` may only be used if the agent has
                no output validators since output validators would expect an argument that matches the agent's
                output type.
            message_history: History of the conversation so far.
            deferred_tool_results: Optional results for deferred tool calls in the message history.
            model: Optional model to use for this run, required if `model` was not set when creating the agent.
            deps: Optional dependencies to use for this run.
            deps_factory: Optional callback that produces per-request `deps` from the incoming Starlette
                `Request`. Sync or async. When provided, takes precedence over `deps` — useful for gateway
                scenarios where each request's deps are derived from headers (e.g. tenant/auth).
            model_settings: Optional settings to use for this model's request.
            usage_limits: Optional limits on model request count or token usage.
            usage: Optional usage to start with, useful for resuming a conversation or agents used in tools.
            infer_name: Whether to try to infer the agent name from the call frame if it's not set.
            toolsets: Optional additional toolsets for this run.
            builtin_tools: Optional additional builtin tools for this run.
            on_complete: Optional callback function called when the agent run completes successfully.
                The callback receives the completed [`AgentRunResult`][pydantic_ai.agent.AgentRunResult] and can access `all_messages()` and other result data.

            debug: Boolean indicating if debug tracebacks should be returned on errors.
            routes: A list of routes to serve incoming HTTP and WebSocket requests.
            middleware: A list of middleware to run for every request. A starlette application will always
                automatically include two middleware classes. `ServerErrorMiddleware` is added as the very
                outermost middleware, to handle any uncaught errors occurring anywhere in the entire stack.
                `ExceptionMiddleware` is added as the very innermost middleware, to deal with handled
                exception cases occurring in the routing or endpoints.
            exception_handlers: A mapping of either integer status codes, or exception class types onto
                callables which handle the exceptions. Exception handler callables should be of the form
                `handler(request, exc) -> response` and may be either standard functions, or async functions.
            on_startup: A list of callables to run on application startup. Startup handler callables do not
                take any arguments, and may be either standard functions, or async functions.
            on_shutdown: A list of callables to run on application shutdown. Shutdown handler callables do
                not take any arguments, and may be either standard functions, or async functions.
            lifespan: A lifespan context function, which can be used to perform startup and shutdown tasks.
                This is a newer style that replaces the `on_startup` and `on_shutdown` handlers. Use one or
                the other, not both.
        """
        super().__init__(
            debug=debug,
            routes=routes,
            middleware=middleware,
            exception_handlers=exception_handlers,
            on_startup=on_startup,
            on_shutdown=on_shutdown,
            lifespan=lifespan,
        )

        async def run_responses(request: Request) -> Response:
            """Endpoint to run the agent with the provided Responses request."""
            # `dispatch_request` will store the frontend state from the request on `deps.state` (if it implements the `StateHandler` protocol),
            # so when no `deps_factory` is provided we copy the shared deps to avoid different requests mutating the same object.
            nonlocal deps
            if deps_factory is None and isinstance(deps, StateHandler):  # pragma: no branch
                deps = replace(deps)

            return await ResponsesAdapter[AgentDepsT, OutputDataT].dispatch_request(
                request,
                agent=agent,
                output_type=output_type,
                message_history=message_history,
                deferred_tool_results=deferred_tool_results,
                model=model,
                deps=deps,
                deps_factory=deps_factory,
                model_settings=model_settings,
                usage_limits=usage_limits,
                usage=usage,
                infer_name=infer_name,
                toolsets=toolsets,
                builtin_tools=builtin_tools,
                on_complete=on_complete,
            )

        self.router.add_route('/v1/responses', run_responses, methods=['POST'])


def gateway(
    agents: Mapping[str, AbstractAgent[Any, Any]],
    *,
    deps_factory: Callable[[Request], Any | Awaitable[Any]] | None = None,
    owned_by: str = 'pydantic-ai',
    debug: bool = False,
    routes: Sequence[BaseRoute] | None = None,
    middleware: Sequence[Middleware] | None = None,
    exception_handlers: Mapping[Any, ExceptionHandler] | None = None,
    on_startup: Sequence[Callable[[], Any]] | None = None,
    on_shutdown: Sequence[Callable[[], Any]] | None = None,
    lifespan: Lifespan[Starlette] | None = None,
) -> Starlette:
    """Build an ASGI app that exposes multiple agents over the OpenAI Responses API.

    The returned Starlette app mounts:

    - `POST /v1/responses` — dispatches to the agent named by the request `model:` field;
      returns a 404 with an OpenAI-shaped error envelope when no agent matches.
    - `GET /v1/models` — lists the configured agents in OpenAI's models-list shape so SDKs
      that introspect available models (e.g. `client.models.list()`) work against the gateway.

    Vanilla `OpenAI` SDK clients can target the gateway and select agents via
    `client.responses.create(model='<agent-name>', ...)`.

    Args:
        agents: Mapping of model id (used by clients in the request `model:` field) to agent.
            Each agent's own configured model is used at runtime; the key here is just the
            routing label.
        deps_factory: Optional callback that produces per-request `deps` from the incoming
            Starlette `Request`. Sync or async. Applied uniformly to every dispatched agent —
            for per-agent deps shaping, mount separate `to_responses()` apps.
        owned_by: String returned in the `owned_by` field of `/v1/models` entries.
        debug: Boolean indicating if debug tracebacks should be returned on errors.
        routes: A list of routes to serve incoming HTTP and WebSocket requests.
        middleware: A list of middleware to run for every request.
        exception_handlers: A mapping of either integer status codes, or exception class types
            onto callables which handle the exceptions.
        on_startup: A list of callables to run on application startup.
        on_shutdown: A list of callables to run on application shutdown.
        lifespan: A lifespan context function.

    Returns:
        A Starlette ASGI application routing OpenAI Responses traffic to the configured agents.
    """
    app = Starlette(
        debug=debug,
        routes=routes,
        middleware=middleware,
        exception_handlers=exception_handlers,
        on_startup=on_startup,
        on_shutdown=on_shutdown,
        lifespan=lifespan,
    )

    async def list_models(_: Request) -> Response:
        created = int(time.time())
        return JSONResponse(
            {
                'object': 'list',
                'data': [{'id': name, 'object': 'model', 'created': created, 'owned_by': owned_by} for name in agents],
            }
        )

    async def run_responses(request: Request) -> Response:
        body = await request.body()
        try:
            payload = json.loads(body)
        except json.JSONDecodeError:
            return _error_response('Request body must be valid JSON.', code='invalid_json', status_code=400)
        if not is_str_dict(payload):
            return _error_response('Request body must be a JSON object.', code='invalid_request', status_code=400)
        model_name = payload.get('model')
        if not isinstance(model_name, str) or model_name not in agents:
            return _error_response(
                f'The model {model_name!r} does not exist or you do not have access to it.',
                code='model_not_found',
                status_code=404,
            )
        return await ResponsesAdapter[Any, Any].dispatch_request(
            request,
            agent=agents[model_name],
            deps_factory=deps_factory,
        )

    app.router.add_route('/v1/responses', run_responses, methods=['POST'])
    app.router.add_route('/v1/models', list_models, methods=['GET'])
    return app


def _error_response(message: str, *, code: str, status_code: int) -> JSONResponse:
    return JSONResponse(
        {'error': {'message': message, 'type': 'invalid_request_error', 'code': code}},
        status_code=status_code,
    )
