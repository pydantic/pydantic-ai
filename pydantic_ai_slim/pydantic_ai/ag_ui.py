"""Provides an AG-UI protocol adapter for the Pydantic AI agent.

This package provides seamless integration between pydantic-ai agents and ag-ui
for building interactive AI applications with streaming event-based communication.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from typing import Any, Generic

from . import DeferredToolResults
from .agent import AbstractAgent
from .messages import ModelMessage
from .models import KnownModelName, Model
from .output import OutputDataT, OutputSpec
from .settings import ModelSettings
from .tools import AgentDepsT
from .toolsets import AbstractToolset
from .usage import RunUsage, UsageLimits

try:
    from ag_ui.core.types import RunAgentInput

    from .ui import OnCompleteFunc, StateDeps, StateHandler
    from .ui.ag_ui import (
        SSE_CONTENT_TYPE,
        AGUIAdapter,
    )
except ImportError as e:  # pragma: no cover
    raise ImportError(
        'Please install the `ag-ui-protocol` package to use `Agent.to_ag_ui()` method, '
        'you can use the `ag-ui` optional group — `pip install "pydantic-ai-slim[ag-ui]"`'
    ) from e

try:
    from starlette.applications import Starlette
    from starlette.middleware import Middleware
    from starlette.requests import Request
    from starlette.responses import Response
    from starlette.routing import BaseRoute
    from starlette.types import ExceptionHandler, Lifespan
except ImportError as e:  # pragma: no cover
    raise ImportError(
        'Please install the `starlette` package to use `Agent.to_ag_ui()` method, '
        'you can use the `ag-ui` optional group — `pip install "pydantic-ai-slim[ag-ui]"`'
    ) from e


__all__ = [
    'SSE_CONTENT_TYPE',
    'StateDeps',
    'StateHandler',
    'AGUIApp',
    'OnCompleteFunc',
    'handle_ag_ui_request',
    'run_ag_ui',
]


class AGUIApp(Generic[AgentDepsT, OutputDataT], Starlette):
    """ASGI application for running Pydantic AI agents with AG-UI protocol support."""

    def __init__(
        self,
        agent: AbstractAgent[AgentDepsT, OutputDataT],
        *,
        # Agent.iter parameters.
        output_type: OutputSpec[Any] | None = None,
        model: Model | KnownModelName | str | None = None,
        deps: AgentDepsT = None,
        model_settings: ModelSettings | None = None,
        usage_limits: UsageLimits | None = None,
        usage: RunUsage | None = None,
        infer_name: bool = True,
        toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
        # Starlette parameters.
        debug: bool = False,
        routes: Sequence[BaseRoute] | None = None,
        middleware: Sequence[Middleware] | None = None,
        exception_handlers: Mapping[Any, ExceptionHandler] | None = None,
        on_startup: Sequence[Callable[[], Any]] | None = None,
        on_shutdown: Sequence[Callable[[], Any]] | None = None,
        lifespan: Lifespan[AGUIApp[AgentDepsT, OutputDataT]] | None = None,
    ) -> None:
        """An ASGI application that handles every AG-UI request by running the agent.

        Note that the `deps` will be the same for each request, with the exception of the AG-UI state that's
        injected into the `state` field of a `deps` object that implements the [`StateHandler`][pydantic_ai.ag_ui.StateHandler] protocol.
        To provide different `deps` for each request (e.g. based on the authenticated user),
        use [`pydantic_ai.ag_ui.run_ag_ui`][pydantic_ai.ag_ui.run_ag_ui] or
        [`pydantic_ai.ag_ui.handle_ag_ui_request`][pydantic_ai.ag_ui.handle_ag_ui_request] instead.

        Args:
            agent: The agent to run.

            output_type: Custom output type to use for this run, `output_type` may only be used if the agent has
                no output validators since output validators would expect an argument that matches the agent's
                output type.
            model: Optional model to use for this run, required if `model` was not set when creating the agent.
            deps: Optional dependencies to use for this run.
            model_settings: Optional settings to use for this model's request.
            usage_limits: Optional limits on model request count or token usage.
            usage: Optional usage to start with, useful for resuming a conversation or agents used in tools.
            infer_name: Whether to try to infer the agent name from the call frame if it's not set.
            toolsets: Optional additional toolsets for this run.

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

        async def endpoint(request: Request) -> Response:
            """Endpoint to run the agent with the provided input data."""
            return await handle_ag_ui_request(
                agent,
                request,
                output_type=output_type,
                model=model,
                deps=deps,
                model_settings=model_settings,
                usage_limits=usage_limits,
                usage=usage,
                infer_name=infer_name,
                toolsets=toolsets,
            )

        self.router.add_route('/', endpoint, methods=['POST'], name='run_agent')


async def handle_ag_ui_request(
    agent: AbstractAgent[AgentDepsT, Any],
    request: Request,
    *,
    output_type: OutputSpec[Any] | None = None,
    model: Model | KnownModelName | str | None = None,
    deps: AgentDepsT = None,
    model_settings: ModelSettings | None = None,
    usage_limits: UsageLimits | None = None,
    usage: RunUsage | None = None,
    infer_name: bool = True,
    toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
    on_complete: OnCompleteFunc | None = None,
) -> Response:
    """Handle an AG-UI request by running the agent and returning a streaming response.

    Args:
        agent: The agent to run.
        request: The Starlette request (e.g. from FastAPI) containing the AG-UI run input.

        output_type: Custom output type to use for this run, `output_type` may only be used if the agent has no
            output validators since output validators would expect an argument that matches the agent's output type.
        model: Optional model to use for this run, required if `model` was not set when creating the agent.
        deps: Optional dependencies to use for this run.
        model_settings: Optional settings to use for this model's request.
        usage_limits: Optional limits on model request count or token usage.
        usage: Optional usage to start with, useful for resuming a conversation or agents used in tools.
        infer_name: Whether to try to infer the agent name from the call frame if it's not set.
        toolsets: Optional additional toolsets for this run.
        on_complete: Optional callback function called when the agent run completes successfully.
            The callback receives the completed [`AgentRunResult`][pydantic_ai.agent.AgentRunResult] and can access `all_messages()` and other result data.

    Returns:
        A streaming Starlette response with AG-UI protocol events.
    """
    return await AGUIAdapter.dispatch_request(
        agent,
        request,
        deps=deps,
        output_type=output_type,
        model=model,
        model_settings=model_settings,
        usage_limits=usage_limits,
        usage=usage,
        infer_name=infer_name,
        toolsets=toolsets,
        on_complete=on_complete,
    )


async def run_ag_ui(
    agent: AbstractAgent[AgentDepsT, Any],
    run_input: RunAgentInput,
    accept: str = SSE_CONTENT_TYPE,
    *,
    output_type: OutputSpec[Any] | None = None,
    message_history: Sequence[ModelMessage] | None = None,
    deferred_tool_results: DeferredToolResults | None = None,
    model: Model | KnownModelName | str | None = None,
    deps: AgentDepsT = None,
    model_settings: ModelSettings | None = None,
    usage_limits: UsageLimits | None = None,
    usage: RunUsage | None = None,
    infer_name: bool = True,
    toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
    on_complete: OnCompleteFunc | None = None,
) -> AsyncIterator[str]:
    """Run the agent with the AG-UI run input and stream AG-UI protocol events.

    Args:
        agent: The agent to run.
        run_input: The AG-UI run input containing thread_id, run_id, messages, etc.
        accept: The accept header value for the run.

        output_type: Custom output type to use for this run, `output_type` may only be used if the agent has no
            output validators since output validators would expect an argument that matches the agent's output type.
        model: Optional model to use for this run, required if `model` was not set when creating the agent.
        deps: Optional dependencies to use for this run.
        model_settings: Optional settings to use for this model's request.
        usage_limits: Optional limits on model request count or token usage.
        usage: Optional usage to start with, useful for resuming a conversation or agents used in tools.
        infer_name: Whether to try to infer the agent name from the call frame if it's not set.
        toolsets: Optional additional toolsets for this run.
        on_complete: Optional callback function called when the agent run completes successfully.
            The callback receives the completed [`AgentRunResult`][pydantic_ai.agent.AgentRunResult] and can access `all_messages()` and other result data.

    Yields:
        Streaming event chunks encoded as strings according to the accept header value.
    """
    adapter = AGUIAdapter(agent=agent, request=run_input)
    async for event in adapter.encode_stream(
        adapter.run_stream(
            output_type=output_type,
            message_history=message_history,
            deferred_tool_results=deferred_tool_results,
            model=model,
            deps=deps,
            model_settings=model_settings,
            usage_limits=usage_limits,
            usage=usage,
            infer_name=infer_name,
            toolsets=toolsets,
            on_complete=on_complete,
        ),
        accept=accept,
    ):
        yield event
