"""Beta-track features for `Agent` — gated under `Agent.beta` until graduated.

APIs that are not yet stable enough to graduate live under `agent.beta.<feature>` so
users see the explicit beta gate at every call site.
"""

from __future__ import annotations as _annotations

from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, Generic

from .. import messages as _messages, models
from ..output import OutputDataT, OutputSpec
from ..settings import ModelSettings
from ..tools import AgentDepsT, DeferredToolResults
from ..toolsets import AbstractToolset
from ..usage import RunUsage, UsageLimits

if TYPE_CHECKING:
    from starlette.applications import Starlette
    from starlette.middleware import Middleware
    from starlette.requests import Request
    from starlette.routing import BaseRoute
    from starlette.types import ExceptionHandler, Lifespan

    from ..ui.responses._adapter import ResponsesMode
    from .abstract import AbstractAgent


class AgentBeta(Generic[AgentDepsT, OutputDataT]):
    """Beta-track features for an [`Agent`][pydantic_ai.agent.Agent].

    Access via `agent.beta`. APIs here are subject to change in any release until
    they graduate to the main `Agent` surface.
    """

    def __init__(self, agent: AbstractAgent[AgentDepsT, OutputDataT]) -> None:
        self._agent = agent

    def to_responses(
        self,
        *,
        # Agent.iter parameters
        output_type: OutputSpec[OutputDataT] | None = None,
        message_history: Sequence[_messages.ModelMessage] | None = None,
        deferred_tool_results: DeferredToolResults | None = None,
        model: models.Model | models.KnownModelName | str | None = None,
        deps: AgentDepsT = None,
        deps_factory: Callable[[Request], AgentDepsT | Awaitable[AgentDepsT]] | None = None,
        model_settings: ModelSettings | None = None,
        usage_limits: UsageLimits | None = None,
        usage: RunUsage | None = None,
        infer_name: bool = True,
        toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
        history_loader: Callable[[str], Awaitable[Sequence[_messages.ModelMessage]]] | None = None,
        mode: ResponsesMode = 'auto',
        # Starlette
        debug: bool = False,
        routes: Sequence[BaseRoute] | None = None,
        middleware: Sequence[Middleware] | None = None,
        exception_handlers: Mapping[Any, ExceptionHandler] | None = None,
        on_startup: Sequence[Callable[[], Any]] | None = None,
        on_shutdown: Sequence[Callable[[], Any]] | None = None,
        lifespan: Lifespan[Starlette] | None = None,
    ) -> Starlette:
        """Returns an ASGI application that exposes the agent as an OpenAI Responses API endpoint.

        The returned app mounts a `POST /v1/responses` route, so an OpenAI SDK pointed at the app's
        base URL can talk to your agent unchanged.

        Note that the `deps` will be the same for each request, with the exception of the request
        `metadata` that's injected into the `state` field of a `deps` object that implements the
        [`StateHandler`][pydantic_ai.ui.StateHandler] protocol. To provide different `deps` per
        request (e.g. based on the authenticated user), use
        [`ResponsesAdapter.dispatch_request()`][pydantic_ai.ui.responses.ResponsesAdapter.dispatch_request]
        directly.

        Example:
        ```python
        from pydantic_ai import Agent

        agent = Agent('openai:gpt-5.2')
        app = agent.beta.to_responses()
        ```

        Run with:

        ```bash
        uvicorn app:app --host 0.0.0.0 --port 8000
        ```

        See [Responses docs](../ui/responses.md) for more information.

        Args:
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
            history_loader: Optional async callable that loads message history from a user-managed
                store. Awaited for each request whose body carries a `conversation` /
                `previous_response_id` field; the resolved id is passed as the loader argument.
                When absent, the agent runs fresh — no error is raised. See
                [`ResponsesAdapter.dispatch_request`][pydantic_ai.ui.responses.ResponsesAdapter.dispatch_request].
            mode: How the wire emits backend tool calls (and how the adapter parses extension
                input items). See [`ResponsesMode`][pydantic_ai.ui.responses.ResponsesMode].
                Defaults to `'auto'` (sniff input items for a `<slug>:` extension prefix).

            debug: Boolean indicating if debug tracebacks should be returned on errors.
            routes: A list of routes to serve incoming HTTP and WebSocket requests.
            middleware: A list of middleware to run for every request.
            exception_handlers: A mapping of either integer status codes, or exception class types onto
                callables which handle the exceptions.
            on_startup: A list of callables to run on application startup.
            on_shutdown: A list of callables to run on application shutdown.
            lifespan: A lifespan context function, which can be used to perform startup and shutdown tasks.

        Returns:
            An ASGI application that handles OpenAI Responses API requests by running the agent.
        """
        from ..ui.responses.app import responses_app

        return responses_app(
            agent=self._agent,
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
            history_loader=history_loader,
            mode=mode,
            debug=debug,
            routes=routes,
            middleware=middleware,
            exception_handlers=exception_handlers,
            on_startup=on_startup,
            on_shutdown=on_shutdown,
            lifespan=lifespan,
        )
