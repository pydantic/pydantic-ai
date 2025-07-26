"""Provides an AG-UI protocol adapter for the Pydantic AI agent.

This package provides seamless integration between pydantic-ai agents and ag-ui
for building interactive AI applications with streaming event-based communication.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from http import HTTPStatus
from typing import (
    Any,
    Callable,
    Final,
    Generic,
    Protocol,
    TypeVar,
    runtime_checkable,
)

try:
    from ag_ui.core import (
        AssistantMessage,
        BaseEvent,
        DeveloperMessage,
        EventType,
        Message,
        RunAgentInput,
        RunErrorEvent,
        RunFinishedEvent,
        RunStartedEvent,
        State,
        SystemMessage,
        TextMessageContentEvent,
        TextMessageEndEvent,
        TextMessageStartEvent,
        ThinkingTextMessageContentEvent,
        ThinkingTextMessageEndEvent,
        ThinkingTextMessageStartEvent,
        ToolCallArgsEvent,
        ToolCallEndEvent,
        ToolCallResultEvent,
        ToolCallStartEvent,
        ToolMessage,
        UserMessage,
    )
    from ag_ui.encoder import EventEncoder
except ImportError as e:  # pragma: no cover
    raise ImportError(
        'Please install the `ag-ui-protocol` package to use `Agent.to_ag_ui()` method, '
        'you can use the `ag-ui` optional group — `pip install "pydantic-ai-slim[ag-ui]"`'
    ) from e

try:
    from starlette.applications import Starlette
    from starlette.middleware import Middleware
    from starlette.requests import Request
    from starlette.responses import Response, StreamingResponse
    from starlette.routing import BaseRoute
    from starlette.types import ExceptionHandler, Lifespan
except ImportError as e:  # pragma: no cover
    raise ImportError(
        'Please install the `starlette` package to use `Agent.to_ag_ui()` method, '
        'you can use the `ag-ui` optional group — `pip install "pydantic-ai-slim[ag-ui]"`'
    ) from e

from collections.abc import AsyncGenerator

from pydantic import BaseModel, ValidationError

from ._agent_graph import CallToolsNode, ModelRequestNode
from .agent import Agent, AgentRun, RunOutputDataT
from .messages import (
    AgentStreamEvent,
    FunctionToolResultEvent,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    PartDeltaEvent,
    PartStartEvent,
    SystemPromptPart,
    TextPart,
    TextPartDelta,
    ThinkingPart,
    ThinkingPartDelta,
    ToolCallPart,
    ToolCallPartDelta,
    ToolReturnPart,
    UserPromptPart,
)
from .models import KnownModelName, Model
from .output import DeferredToolCalls, OutputDataT, OutputSpec
from .settings import ModelSettings
from .tools import AgentDepsT, ToolDefinition
from .toolsets import AbstractToolset
from .toolsets.deferred import DeferredToolset
from .usage import Usage, UsageLimits

__all__ = [
    'SSE_CONTENT_TYPE',
    'StateDeps',
    'StateHandler',
    'AGUIApp',
]

SSE_CONTENT_TYPE: Final[str] = 'text/event-stream'
"""Content type header value for Server-Sent Events (SSE)."""


class AGUIApp(Generic[AgentDepsT, OutputDataT], Starlette):
    """ASGI application for running Pydantic AI agents with AG-UI protocol support."""

    def __init__(
        self,
        agent: Agent[AgentDepsT, OutputDataT],
        *,
        # Agent.iter parameters.
        output_type: OutputSpec[OutputDataT] | None = None,
        model: Model | KnownModelName | str | None = None,
        deps: AgentDepsT = None,
        model_settings: ModelSettings | None = None,
        usage_limits: UsageLimits | None = None,
        usage: Usage | None = None,
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
        """Initialise the AG-UI application.

        Args:
            agent: The Pydantic AI `Agent` to adapt.

            output_type: Custom output type to use for this run, `output_type` may only be used if the agent has
                no output validators since output validators would expect an argument that matches the agent's
                output type.
            model: Optional model to use for this run, required if `model` was not set when creating the agent.
            deps: Optional dependencies to use for this run.
            model_settings: Optional settings to use for this model's request.
            usage_limits: Optional limits on model request count or token usage.
            usage: Optional usage to start with, useful for resuming a conversation or agents used in tools.
            infer_name: Whether to try to infer the agent name from the call frame if it's not set.
            toolsets: Optional list of toolsets to use for this agent, defaults to the agent's toolset.

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
        adapter = _Adapter(agent=agent)

        async def endpoint(request: Request) -> Response | StreamingResponse:
            """Endpoint to run the agent with the provided input data."""
            accept = request.headers.get('accept', SSE_CONTENT_TYPE)
            try:
                input_data = RunAgentInput.model_validate(await request.json())
            except ValidationError as e:  # pragma: no cover
                return Response(
                    content=json.dumps(e.json()),
                    media_type='application/json',
                    status_code=HTTPStatus.UNPROCESSABLE_ENTITY,
                )

            return StreamingResponse(
                adapter.run(
                    input_data,
                    accept,
                    output_type=output_type,
                    model=model,
                    deps=deps,
                    model_settings=model_settings,
                    usage_limits=usage_limits,
                    usage=usage,
                    infer_name=infer_name,
                    toolsets=toolsets,
                ),
                media_type=SSE_CONTENT_TYPE,
            )

        self.router.add_route('/', endpoint, methods=['POST'], name='run_agent')


@dataclass(repr=False)
class _Adapter(Generic[AgentDepsT, OutputDataT]):
    """An agent adapter providing AG-UI protocol support for Pydantic AI agents.

    This class manages the agent runs, tool calls, state storage and providing
    an adapter for running agents with Server-Sent Event (SSE) streaming
    responses using the AG-UI protocol.

    Args:
        agent: The Pydantic AI `Agent` to adapt.
    """

    agent: Agent[AgentDepsT, OutputDataT] = field(repr=False)

    async def run(
        self,
        run_input: RunAgentInput,
        accept: str = SSE_CONTENT_TYPE,
        *,
        output_type: OutputSpec[RunOutputDataT] | None = None,
        model: Model | KnownModelName | str | None = None,
        deps: AgentDepsT = None,
        model_settings: ModelSettings | None = None,
        usage_limits: UsageLimits | None = None,
        usage: Usage | None = None,
        infer_name: bool = True,
        toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
    ) -> AsyncGenerator[str, None]:
        """Run the agent with streaming response using AG-UI protocol events.

        The first two arguments are specific to `Adapter` the rest map directly to the `Agent.iter` method.

        Args:
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
            toolsets: Optional list of toolsets to use for this agent, defaults to the agent's toolset.

        Yields:
            Streaming SSE-formatted event chunks.
        """
        encoder = EventEncoder(accept=accept)
        if run_input.tools:
            # AG-UI tools can't be prefixed as that would result in a mismatch between the tool names in the
            # Pydantic AI events and actual AG-UI tool names, preventing the tool from being called. If any
            # conflicts arise, the AG-UI tool should be renamed or a `PrefixedToolset` used for local toolsets.
            toolset = DeferredToolset[AgentDepsT](
                [
                    ToolDefinition(
                        name=tool.name,
                        description=tool.description,
                        parameters_json_schema=tool.parameters,
                    )
                    for tool in run_input.tools
                ]
            )
            toolsets = [*toolsets, toolset] if toolsets else [toolset]

        try:
            yield encoder.encode(
                RunStartedEvent(
                    thread_id=run_input.thread_id,
                    run_id=run_input.run_id,
                ),
            )

            if not run_input.messages:
                raise _NoMessagesError

            if isinstance(deps, StateHandler):
                deps.state = run_input.state

            messages = _messages_from_ag_ui(run_input.messages)

            async with self.agent.iter(
                user_prompt=None,
                output_type=[output_type or self.agent.output_type, DeferredToolCalls],
                message_history=messages,
                model=model,
                deps=deps,
                model_settings=model_settings,
                usage_limits=usage_limits,
                usage=usage,
                infer_name=infer_name,
                toolsets=toolsets,
            ) as run:
                async for event in self._agent_stream(run):
                    yield encoder.encode(event)
        except _RunError as e:
            yield encoder.encode(
                RunErrorEvent(message=e.message, code=e.code),
            )
        except Exception as e:  # pragma: no cover
            yield encoder.encode(
                RunErrorEvent(message=str(e)),
            )
            raise e
        else:
            yield encoder.encode(
                RunFinishedEvent(
                    thread_id=run_input.thread_id,
                    run_id=run_input.run_id,
                ),
            )

    async def _agent_stream(
        self,
        run: AgentRun[AgentDepsT, Any],
    ) -> AsyncGenerator[BaseEvent, None]:
        """Run the agent streaming responses using AG-UI protocol events.

        Args:
            run: The agent run to process.

        Yields:
            AG-UI Server-Sent Events (SSE).
        """
        async for node in run:
            stream_ctx = _RequestStreamContext()
            if isinstance(node, ModelRequestNode):
                async with node.stream(run.ctx) as request_stream:
                    async for agent_event in request_stream:
                        async for msg in self._handle_model_request_event(stream_ctx, agent_event):
                            yield msg

                    if stream_ctx.part_end:  # pragma: no branch
                        yield stream_ctx.part_end
                        stream_ctx.part_end = None
            elif isinstance(node, CallToolsNode):
                async with node.stream(run.ctx) as handle_stream:
                    async for event in handle_stream:
                        if isinstance(event, FunctionToolResultEvent):
                            async for msg in self._handle_tool_result_event(stream_ctx, event):
                                yield msg

    async def _handle_model_request_event(
        self,
        stream_ctx: _RequestStreamContext,
        agent_event: AgentStreamEvent,
    ) -> AsyncGenerator[BaseEvent, None]:
        """Handle an agent event and yield AG-UI protocol events.

        Args:
            stream_ctx: The request stream context to manage state.
            agent_event: The agent event to process.

        Yields:
            AG-UI Server-Sent Events (SSE) based on the agent event.
        """
        if isinstance(agent_event, PartStartEvent):
            if stream_ctx.part_end:
                # End the previous part.
                yield stream_ctx.part_end
                stream_ctx.part_end = None

            part = agent_event.part
            if isinstance(part, TextPart):
                message_id = stream_ctx.new_message_id()
                yield TextMessageStartEvent(
                    message_id=message_id,
                )
                if part.content:  # pragma: no branch
                    yield TextMessageContentEvent(
                        message_id=message_id,
                        delta=part.content,
                    )
                stream_ctx.part_end = TextMessageEndEvent(
                    message_id=message_id,
                )
            elif isinstance(part, ToolCallPart):  # pragma: no branch
                message_id = stream_ctx.message_id or stream_ctx.new_message_id()
                yield ToolCallStartEvent(
                    tool_call_id=part.tool_call_id,
                    tool_call_name=part.tool_name,
                    parent_message_id=message_id,
                )
                if part.args:
                    yield ToolCallArgsEvent(
                        tool_call_id=part.tool_call_id,
                        delta=part.args if isinstance(part.args, str) else json.dumps(part.args),
                    )
                stream_ctx.part_end = ToolCallEndEvent(
                    tool_call_id=part.tool_call_id,
                )

            elif isinstance(part, ThinkingPart):  # pragma: no branch
                yield ThinkingTextMessageStartEvent(
                    type=EventType.THINKING_TEXT_MESSAGE_START,
                )
                # Always send the content even if it's empty, as it may be
                # used to indicate the start of thinking.
                yield ThinkingTextMessageContentEvent(
                    type=EventType.THINKING_TEXT_MESSAGE_CONTENT,
                    delta=part.content,
                )
                stream_ctx.part_end = ThinkingTextMessageEndEvent(
                    type=EventType.THINKING_TEXT_MESSAGE_END,
                )

        elif isinstance(agent_event, PartDeltaEvent):
            delta = agent_event.delta
            if isinstance(delta, TextPartDelta):
                yield TextMessageContentEvent(
                    message_id=stream_ctx.message_id,
                    delta=delta.content_delta,
                )
            elif isinstance(delta, ToolCallPartDelta):  # pragma: no branch
                assert delta.tool_call_id, '`ToolCallPartDelta.tool_call_id` must be set'
                yield ToolCallArgsEvent(
                    tool_call_id=delta.tool_call_id,
                    delta=delta.args_delta if isinstance(delta.args_delta, str) else json.dumps(delta.args_delta),
                )
            elif isinstance(delta, ThinkingPartDelta):  # pragma: no branch
                if delta.content_delta:  # pragma: no branch
                    yield ThinkingTextMessageContentEvent(
                        type=EventType.THINKING_TEXT_MESSAGE_CONTENT,
                        delta=delta.content_delta,
                    )

    async def _handle_tool_result_event(
        self,
        stream_ctx: _RequestStreamContext,
        event: FunctionToolResultEvent,
    ) -> AsyncGenerator[BaseEvent, None]:
        """Convert a tool call result to AG-UI events.

        Args:
            stream_ctx: The request stream context to manage state.
            event: The tool call result event to process.

        Yields:
            AG-UI Server-Sent Events (SSE).
        """
        result = event.result
        if not isinstance(result, ToolReturnPart):
            return

        message_id = stream_ctx.new_message_id()
        yield ToolCallResultEvent(
            message_id=message_id,
            type=EventType.TOOL_CALL_RESULT,
            role='tool',
            tool_call_id=result.tool_call_id,
            content=result.model_response_str(),
        )

        # Now check for  AG-UI events returned by the tool calls.
        content = result.content
        if isinstance(content, BaseEvent):
            yield content
        elif isinstance(content, (str, bytes)):  # pragma: no branch
            # Avoid iterable check for strings and bytes.
            pass
        elif isinstance(content, Iterable):  # pragma: no branch
            for item in content:  # type: ignore[reportUnknownMemberType]
                if isinstance(item, BaseEvent):  # pragma: no branch
                    yield item


def _messages_from_ag_ui(messages: list[Message]) -> list[ModelMessage]:
    """Convert a AG-UI history to a Pydantic AI one."""
    result: list[ModelMessage] = []
    tool_calls: dict[str, str] = {}  # Tool call ID to tool name mapping.
    for msg in messages:
        if isinstance(msg, UserMessage):
            result.append(ModelRequest(parts=[UserPromptPart(content=msg.content)]))
        elif isinstance(msg, AssistantMessage):
            if msg.tool_calls:
                for tool_call in msg.tool_calls:
                    tool_calls[tool_call.id] = tool_call.function.name

                result.append(
                    ModelResponse(
                        parts=[
                            ToolCallPart(
                                tool_name=tool_call.function.name,
                                tool_call_id=tool_call.id,
                                args=tool_call.function.arguments,
                            )
                            for tool_call in msg.tool_calls
                        ]
                    )
                )

            if msg.content:
                result.append(ModelResponse(parts=[TextPart(content=msg.content)]))
        elif isinstance(msg, SystemMessage):
            result.append(ModelRequest(parts=[SystemPromptPart(content=msg.content)]))
        elif isinstance(msg, ToolMessage):
            tool_name = tool_calls.get(msg.tool_call_id)
            if tool_name is None:  # pragma: no cover
                raise _ToolCallNotFoundError(tool_call_id=msg.tool_call_id)

            result.append(
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            tool_name=tool_name,
                            content=msg.content,
                            tool_call_id=msg.tool_call_id,
                        )
                    ]
                )
            )
        elif isinstance(msg, DeveloperMessage):  # pragma: no branch
            result.append(ModelRequest(parts=[SystemPromptPart(content=msg.content)]))

    return result


@runtime_checkable
class StateHandler(Protocol):
    """Protocol for state handlers in agent runs."""

    @property
    def state(self) -> State:
        """Get the current state of the agent run."""
        ...

    @state.setter
    def state(self, state: State) -> None:
        """Set the state of the agent run.

        This method is called to update the state of the agent run with the
        provided state.

        Args:
            state: The run state.

        Raises:
            InvalidStateError: If `state` does not match the expected model.
        """
        ...


StateT = TypeVar('StateT', bound=BaseModel)
"""Type variable for the state type, which must be a subclass of `BaseModel`."""


class StateDeps(Generic[StateT]):
    """Provides AG-UI state management.

    This class is used to manage the state of an agent run. It allows setting
    the state of the agent run with a specific type of state model, which must
    be a subclass of `BaseModel`.

    The state is set using the `state` setter by the `Adapter` when the run starts.

    Implements the `StateHandler` protocol.
    """

    def __init__(self, default: StateT) -> None:
        """Initialize the state with the provided state type."""
        self._state = default

    @property
    def state(self) -> StateT:
        """Get the current state of the agent run.

        Returns:
            The current run state.
        """
        return self._state

    @state.setter
    def state(self, state: State) -> None:
        """Set the state of the agent run.

        This method is called to update the state of the agent run with the
        provided state.

        Implements the `StateHandler` protocol.

        Args:
            state: The run state, which must be `None` or model validate for the state type.

        Raises:
            InvalidStateError: If `state` does not validate.
        """
        if state is None:
            # If state is None, we keep the current state, which will be the default state.
            return

        try:
            self._state = type(self._state).model_validate(state)
        except ValidationError as e:  # pragma: no cover
            raise _InvalidStateError from e


@dataclass(repr=False)
class _RequestStreamContext:
    """Data class to hold request stream context."""

    message_id: str = ''
    part_end: BaseEvent | None = None

    def new_message_id(self) -> str:
        """Generate a new message ID for the request stream.

        Assigns a new UUID to the `message_id` and returns it.

        Returns:
            A new message ID.
        """
        self.message_id = str(uuid.uuid4())
        return self.message_id


@dataclass
class _RunError(Exception):
    """Exception raised for errors during agent runs."""

    message: str
    code: str

    def __str__(self) -> str:  # pragma: no cover
        return self.message


@dataclass
class _NoMessagesError(_RunError):
    """Exception raised when no messages are found in the input."""

    message: str = 'no messages found in the input'
    code: str = 'no_messages'


@dataclass
class _InvalidStateError(_RunError, ValidationError):
    """Exception raised when an invalid state is provided."""

    message: str = 'invalid state provided'
    code: str = 'invalid_state'


class _ToolCallNotFoundError(_RunError, ValueError):
    """Exception raised when an tool result is present without a matching call."""

    def __init__(self, tool_call_id: str) -> None:
        """Initialize the exception with the tool call ID."""
        super().__init__(  # pragma: no cover
            message=f'Tool call with ID {tool_call_id} not found in the history.',
            code='tool_call_not_found',
        )
