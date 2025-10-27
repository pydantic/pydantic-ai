from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any, Generic

from typing_extensions import Self

from .. import DeferredToolResults
from ..agent import AbstractAgent
from ..builtin_tools import AbstractBuiltinTool
from ..messages import ModelMessage
from ..models import KnownModelName, Model
from ..output import OutputDataT, OutputSpec
from ..settings import ModelSettings
from ..tools import AgentDepsT
from ..toolsets import AbstractToolset
from ..usage import RunUsage, UsageLimits
from .adapter import OnCompleteFunc, UIAdapter

try:
    from starlette.applications import Starlette
    from starlette.middleware import Middleware
    from starlette.requests import Request
    from starlette.responses import Response
    from starlette.routing import BaseRoute
    from starlette.types import ExceptionHandler, Lifespan
except ImportError as e:  # pragma: no cover
    raise ImportError(
        'Please install the `starlette` package to use `UIApp`, '
        'you can use the `ui` optional group — `pip install "pydantic-ai-slim[ui]"`'
    ) from e


class UIApp(Generic[AgentDepsT, OutputDataT], Starlette):
    """ASGI application for running Pydantic AI agents with UI protocol support."""

    def __init__(
        self,
        adapter_type: type[UIAdapter[Any, Any, Any, AgentDepsT, OutputDataT]],
        agent: AbstractAgent[AgentDepsT, OutputDataT],
        *,
        # UIAdapter.dispatch_request parameters
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
        """An ASGI application that handles every request by running the agent and streaming the response.

        Note that the `deps` will be the same for each request, with the exception of the frontend state that's
        injected into the `state` field of a `deps` object that implements the [`StateHandler`][pydantic_ai.ui.StateHandler] protocol.
        To provide different `deps` for each request (e.g. based on the authenticated user),
        use [`UIAdapter.run_stream()`][pydantic_ai.ui.UIAdapter.run_stream] or
        [`UIAdapter.dispatch_request()`][pydantic_ai.ui.UIAdapter.dispatch_request] instead.

        Args:
            adapter_type: The type of the UI adapter to use.
            agent: The agent to run.

            output_type: Custom output type to use for this run, `output_type` may only be used if the agent has
                no output validators since output validators would expect an argument that matches the agent's
                output type.
            message_history: History of the conversation so far.
            deferred_tool_results: Optional results for deferred tool calls in the message history.
            model: Optional model to use for this run, required if `model` was not set when creating the agent.
            deps: Optional dependencies to use for this run.
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

        async def run_agent(request: Request) -> Response:
            """Endpoint to run the agent with the provided input data."""
            return await adapter_type.dispatch_request(
                request,
                agent=agent,
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
                builtin_tools=builtin_tools,
                on_complete=on_complete,
            )

        self.router.add_route('/', run_agent, methods=['POST'])
