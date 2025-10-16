"""Base classes for UI event stream protocols.

This module provides abstract base classes for implementing UI event stream adapters
that transform Pydantic AI agent events into protocol-specific events (e.g., AG-UI, Vercel AI).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence
from dataclasses import Field, dataclass, replace
from functools import cached_property
from http import HTTPStatus
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Generic,
    Protocol,
    TypeAlias,
    TypeVar,
    runtime_checkable,
)

from pydantic import BaseModel, ValidationError

from .. import DeferredToolRequests, DeferredToolResults, _utils
from ..agent import AbstractAgent, AgentDepsT, AgentRunResult
from ..builtin_tools import AbstractBuiltinTool
from ..exceptions import UserError
from ..messages import ModelMessage
from ..models import KnownModelName, Model
from ..output import OutputSpec
from ..settings import ModelSettings
from ..toolsets import AbstractToolset
from ..usage import RunUsage, UsageLimits
from .event_stream import BaseEventStream, SourceEvent

if TYPE_CHECKING:
    from starlette.requests import Request
    from starlette.responses import Response


__all__ = [
    'BaseAdapter',
]


RunRequestT = TypeVar('RunRequestT')
"""Type variable for protocol-specific request types."""

MessageT = TypeVar('MessageT')
"""Type variable for protocol-specific message types."""

EventT = TypeVar('EventT')
"""Type variable for protocol-specific event types."""

OnCompleteFunc: TypeAlias = Callable[[AgentRunResult[Any]], None] | Callable[[AgentRunResult[Any]], Awaitable[None]]
"""Callback function type that receives the `AgentRunResult` of the completed run. Can be sync or async."""


# State management types

StateT = TypeVar('StateT', bound=BaseModel)
"""Type variable for the state type, which must be a subclass of `BaseModel`."""


@runtime_checkable
class StateHandler(Protocol):
    """Protocol for state handlers in agent runs. Requires the class to be a dataclass with a `state` field."""

    # Has to be a dataclass so we can use `replace` to update the state.
    # From https://github.com/python/typeshed/blob/9ab7fde0a0cd24ed7a72837fcb21093b811b80d8/stdlib/_typeshed/__init__.pyi#L352
    __dataclass_fields__: ClassVar[dict[str, Field[Any]]]

    @property
    def state(self) -> Any:
        """Get the current state of the agent run."""
        ...

    @state.setter
    def state(self, state: Any) -> None:
        """Set the state of the agent run.

        This method is called to update the state of the agent run with the
        provided state.

        Args:
            state: The run state.

        Raises:
            InvalidStateError: If `state` does not match the expected model.
        """
        ...


@dataclass
class StateDeps(Generic[StateT]):
    """Provides AG-UI state management.

    This class is used to manage the state of an agent run. It allows setting
    the state of the agent run with a specific type of state model, which must
    be a subclass of `BaseModel`.

    The state is set using the `state` setter by the `Adapter` when the run starts.

    Implements the `StateHandler` protocol.
    """

    state: StateT


@dataclass
class BaseAdapter(ABC, Generic[RunRequestT, MessageT, EventT, AgentDepsT]):
    """TODO (DouwM): Docstring."""

    agent: AbstractAgent[AgentDepsT]
    """The Pydantic AI agent to run."""

    request: RunRequestT
    """The protocol-specific request object."""

    @classmethod
    @abstractmethod
    async def validate_request(cls, request: Request) -> RunRequestT:
        """Validate the request and return the validated request."""
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def load_messages(cls, messages: Sequence[MessageT]) -> list[ModelMessage]:
        """Load messages from the request and return the loaded messages."""
        raise NotImplementedError

    @property
    @abstractmethod
    def event_stream(self) -> BaseEventStream[RunRequestT, EventT, AgentDepsT]:
        """Create an event stream for the adapter."""
        raise NotImplementedError

    @cached_property
    @abstractmethod
    def messages(self) -> list[ModelMessage]:
        """Convert protocol messages to Pydantic AI messages.

        Args:
            messages: List of protocol-specific messages.

        Returns:
            List of Pydantic AI ModelMessage objects.
        """
        raise NotImplementedError

    @cached_property
    def toolset(self) -> AbstractToolset[AgentDepsT] | None:
        """Get the toolset for the adapter."""
        return None

    @cached_property
    def state(self) -> dict[str, Any] | None:
        """Get the state of the agent run."""
        return None

    @property
    def response_headers(self) -> Mapping[str, str] | None:
        """Get the response headers for the adapter."""
        return None

    def encode_stream(self, stream: AsyncIterator[EventT], accept: str | None = None) -> AsyncIterator[str]:
        """Encode a stream of events as SSE strings.

        Args:
            stream: The stream of events to encode.
            accept: The accept header value for encoding format.
        """
        return self.event_stream.encode_stream(stream, accept)

    async def process_stream(
        self,
        stream: AsyncIterator[SourceEvent],
        on_complete: OnCompleteFunc | None = None,
    ) -> AsyncIterator[EventT]:
        """Process a stream of events and return a stream of events.

        Args:
            stream: The stream of events to process.
            on_complete: Optional callback function called when the agent run completes successfully.
        """
        event_stream = self.event_stream
        async for event in event_stream.handle_stream(stream):
            yield event

        try:
            result = event_stream.result
            if on_complete is not None and result is not None:
                if _utils.is_async_callable(on_complete):
                    await on_complete(result)
                else:
                    await _utils.run_in_executor(on_complete, result)
        except Exception as e:  # TODO (DouweM): coverage
            async for event in event_stream.on_error(e):
                yield event

    async def run_stream(
        self,
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
        builtin_tools: Sequence[AbstractBuiltinTool] | None = None,
        on_complete: OnCompleteFunc | None = None,
    ) -> AsyncIterator[EventT]:
        """Run the agent with the AG-UI run input and stream AG-UI protocol events.

        Args:
            output_type: Custom output type to use for this run, `output_type` may only be used if the agent has no
                output validators since output validators would expect an argument that matches the agent's output type.
            message_history: History of the conversation so far.
            deferred_tool_results: Optional results for deferred tool calls in the message history.
            model: Optional model to use for this run, required if `model` was not set when creating the agent.
            deps: Optional dependencies to use for this run.
            model_settings: Optional settings to use for this model's request.
            usage_limits: Optional limits on model request count or token usage.
            usage: Optional usage to start with, useful for resuming a conversation or agents used in tools.
            infer_name: Whether to try to infer the agent name from the call frame if it's not set.
            toolsets: Optional additional toolsets for this run.
            builtin_tools: Optional additional builtin tools to use for this run.
            on_complete: Optional callback function called when the agent run completes successfully.
                The callback receives the completed [`AgentRunResult`][pydantic_ai.agent.AgentRunResult] and can access `all_messages()` and other result data.

        Yields:
            Streaming event chunks encoded as strings according to the accept header value.
        """
        message_history = [*(message_history or []), *self.messages]

        toolset = self.toolset
        if toolset:
            output_type = [output_type or self.agent.output_type, DeferredToolRequests]
            toolsets = [*toolsets, toolset] if toolsets else [toolset]

        if isinstance(deps, StateHandler):
            raw_state = self.state or {}
            if isinstance(deps.state, BaseModel):
                state = type(deps.state).model_validate(raw_state)
            else:
                state = raw_state

            deps = replace(deps, state=state)
        elif self.state:
            raise UserError(
                f'State is provided but `deps` of type `{type(deps).__name__}` does not implement the `StateHandler` protocol: it needs to be a dataclass with a non-optional `state` field.'
            )

        async for event in self.process_stream(
            self.agent.run_stream_events(
                user_prompt=None,
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
            ),
            on_complete=on_complete,
        ):
            yield event

    async def stream_response(self, stream: AsyncIterator[EventT], accept: str | None = None) -> Response:
        """Stream a response to the client.

        Args:
            stream: The stream of events to encode.
            accept: The accept header value for encoding format.
        """
        try:
            from starlette.responses import StreamingResponse
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                'Please install the `starlette` package to use `BaseAdapter.stream_response()` method, '
                'you can use the `ui` optional group — `pip install "pydantic-ai-slim[ui]"`'
            ) from e

        return StreamingResponse(
            self.encode_stream(
                stream,
                accept=accept,
            ),
            headers=self.response_headers,
        )

    @classmethod
    async def dispatch_request(
        cls,
        agent: AbstractAgent[AgentDepsT, Any],
        request: Request,
        *,
        message_history: Sequence[ModelMessage] | None = None,
        deferred_tool_results: DeferredToolResults | None = None,
        model: Model | KnownModelName | str | None = None,
        deps: AgentDepsT = None,
        output_type: OutputSpec[Any] | None = None,
        model_settings: ModelSettings | None = None,
        usage_limits: UsageLimits | None = None,
        usage: RunUsage | None = None,
        infer_name: bool = True,
        toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
        builtin_tools: Sequence[AbstractBuiltinTool] | None = None,
        on_complete: OnCompleteFunc | None = None,
    ) -> Response:
        """Handle an AG-UI request and return a streaming response.

        Args:
            agent: The agent to run.
            request: The incoming Starlette/FastAPI request.
            output_type: Custom output type to use for this run, `output_type` may only be used if the agent has no
                output validators since output validators would expect an argument that matches the agent's output type.
            message_history: History of the conversation so far.
            deferred_tool_results: Optional results for deferred tool calls in the message history.
            model: Optional model to use for this run, required if `model` was not set when creating the agent.
            deps: Optional dependencies to use for this run.
            model_settings: Optional settings to use for this model's request.
            usage_limits: Optional limits on model request count or token usage.
            usage: Optional usage to start with, useful for resuming a conversation or agents used in tools.
            infer_name: Whether to try to infer the agent name from the call frame if it's not set.
            toolsets: Optional additional toolsets for this run.
            builtin_tools: Optional additional builtin tools to use for this run.
            on_complete: Optional callback function called when the agent run completes successfully.
                The callback receives the completed [`AgentRunResult`][pydantic_ai.agent.AgentRunResult] and can access `all_messages()` and other result data.

        Returns:
            A streaming Starlette response with AG-UI protocol events.
        """
        try:
            from starlette.responses import Response
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                'Please install the `starlette` package to use `BaseAdapter.dispatch_request()` method, '
                'you can use the `ui` optional group — `pip install "pydantic-ai-slim[ui]"`'
            ) from e

        try:
            request_data = await cls.validate_request(request)
        except ValidationError as e:  # pragma: no cover
            return Response(
                content=e.json(),
                media_type='application/json',
                status_code=HTTPStatus.UNPROCESSABLE_ENTITY,
            )

        adapter = cls(agent=agent, request=request_data)

        run_stream = adapter.run_stream(
            message_history=message_history,
            deferred_tool_results=deferred_tool_results,
            deps=deps,
            output_type=output_type,
            model=model,
            model_settings=model_settings,
            usage_limits=usage_limits,
            usage=usage,
            infer_name=infer_name,
            toolsets=toolsets,
            builtin_tools=builtin_tools,
            on_complete=on_complete,
        )

        return await adapter.stream_response(run_stream, accept=request.headers.get('accept'))
