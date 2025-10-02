"""Provides an AG-UI protocol adapter for the Pydantic AI agent.

This package provides seamless integration between pydantic-ai agents and ag-ui
for building interactive AI applications with streaming event-based communication.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable, Iterable, Mapping, Sequence
from dataclasses import Field, dataclass, field, replace
from http import HTTPStatus
from typing import (
    Any,
    ClassVar,
    Final,
    Generic,
    Protocol,
    TypeAlias,
    TypeVar,
    runtime_checkable,
)

from pydantic import BaseModel, ValidationError

from . import _utils
from ._agent_graph import CallToolsNode, ModelRequestNode
from .agent import AbstractAgent, AgentRun, AgentRunResult
from .exceptions import UserError
from .messages import (
    BaseToolCallPart,
    BuiltinToolCallPart,
    BuiltinToolReturnPart,
    FunctionToolResultEvent,
    ModelMessage,
    ModelRequest,
    ModelRequestPart,
    ModelResponse,
    ModelResponsePart,
    ModelResponseStreamEvent,
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
from .output import OutputDataT, OutputSpec
from .settings import ModelSettings
from .tools import AgentDepsT, DeferredToolRequests, ToolDefinition
from .toolsets import AbstractToolset
from .toolsets.external import ExternalToolset
from .usage import RunUsage, UsageLimits

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
        ThinkingEndEvent,
        ThinkingStartEvent,
        ThinkingTextMessageContentEvent,
        ThinkingTextMessageEndEvent,
        ThinkingTextMessageStartEvent,
        Tool as AGUITool,
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


__all__ = [
    'SSE_CONTENT_TYPE',
    'StateDeps',
    'StateHandler',
    'AGUIApp',
    'OnCompleteFunc',
    'handle_ag_ui_request',
    'run_ag_ui',
]

SSE_CONTENT_TYPE: Final[str] = 'text/event-stream'
"""Content type header value for Server-Sent Events (SSE)."""

OnCompleteFunc: TypeAlias = Callable[[AgentRunResult[Any]], None] | Callable[[AgentRunResult[Any]], Awaitable[None]]
"""Callback function type that receives the `AgentRunResult` of the completed run. Can be sync or async."""

_BUILTIN_TOOL_CALL_ID_PREFIX: Final[str] = 'pyd_ai_builtin'


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
        run_ag_ui(
            agent,
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
            on_complete=on_complete,
        ),
        media_type=accept,
    )


async def run_ag_ui(
    agent: AbstractAgent[AgentDepsT, Any],
    run_input: RunAgentInput,
    accept: str = SSE_CONTENT_TYPE,
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
    encoder = EventEncoder(accept=accept)
    if run_input.tools:
        # AG-UI tools can't be prefixed as that would result in a mismatch between the tool names in the
        # Pydantic AI events and actual AG-UI tool names, preventing the tool from being called. If any
        # conflicts arise, the AG-UI tool should be renamed or a `PrefixedToolset` used for local toolsets.
        toolset = _AGUIFrontendToolset[AgentDepsT](run_input.tools)
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

        raw_state: dict[str, Any] = run_input.state or {}
        if isinstance(deps, StateHandler):
            if isinstance(deps.state, BaseModel):
                try:
                    state = type(deps.state).model_validate(raw_state)
                except ValidationError as e:  # pragma: no cover
                    raise _InvalidStateError from e
            else:
                state = raw_state

            deps = replace(deps, state=state)
        elif raw_state:
            raise UserError(
                f'AG-UI state is provided but `deps` of type `{type(deps).__name__}` does not implement the `StateHandler` protocol: it needs to be a dataclass with a non-optional `state` field.'
            )
        else:
            # `deps` not being a `StateHandler` is OK if there is no state.
            pass

        messages = _messages_from_ag_ui(run_input.messages)

        async with agent.iter(
            user_prompt=None,
            output_type=[output_type or agent.output_type, DeferredToolRequests],
            message_history=messages,
            model=model,
            deps=deps,
            model_settings=model_settings,
            usage_limits=usage_limits,
            usage=usage,
            infer_name=infer_name,
            toolsets=toolsets,
        ) as run:
            async for event in _agent_stream(run):
                yield encoder.encode(event)

        if on_complete is not None and run.result is not None:
            if _utils.is_async_callable(on_complete):
                await on_complete(run.result)
            else:
                await _utils.run_in_executor(on_complete, run.result)
    except _RunError as e:
        yield encoder.encode(
            RunErrorEvent(message=e.message, code=e.code),
        )
    except Exception as e:
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


async def _agent_stream(run: AgentRun[AgentDepsT, Any]) -> AsyncIterator[BaseEvent]:
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
                    async for msg in _handle_model_request_event(stream_ctx, agent_event):
                        yield msg

                if stream_ctx.part_end:  # pragma: no branch
                    yield stream_ctx.part_end
                    stream_ctx.part_end = None
                if stream_ctx.thinking:
                    yield ThinkingEndEvent(
                        type=EventType.THINKING_END,
                    )
                    stream_ctx.thinking = False
        elif isinstance(node, CallToolsNode):
            async with node.stream(run.ctx) as handle_stream:
                async for event in handle_stream:
                    if isinstance(event, FunctionToolResultEvent):
                        async for msg in _handle_tool_result_event(stream_ctx, event):
                            yield msg


async def _handle_model_request_event(  # noqa: C901
    stream_ctx: _RequestStreamContext,
    agent_event: ModelResponseStreamEvent,
) -> AsyncIterator[BaseEvent]:
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
        if isinstance(part, ThinkingPart):  # pragma: no branch
            if not stream_ctx.thinking:
                yield ThinkingStartEvent(
                    type=EventType.THINKING_START,
                )
                stream_ctx.thinking = True

            if part.content:
                yield ThinkingTextMessageStartEvent(
                    type=EventType.THINKING_TEXT_MESSAGE_START,
                )
                yield ThinkingTextMessageContentEvent(
                    type=EventType.THINKING_TEXT_MESSAGE_CONTENT,
                    delta=part.content,
                )
                stream_ctx.part_end = ThinkingTextMessageEndEvent(
                    type=EventType.THINKING_TEXT_MESSAGE_END,
                )
        else:
            if stream_ctx.thinking:
                yield ThinkingEndEvent(
                    type=EventType.THINKING_END,
                )
                stream_ctx.thinking = False

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
            elif isinstance(part, BaseToolCallPart):
                tool_call_id = part.tool_call_id
                if isinstance(part, BuiltinToolCallPart):
                    builtin_tool_call_id = '|'.join(
                        [_BUILTIN_TOOL_CALL_ID_PREFIX, part.provider_name or '', tool_call_id]
                    )
                    stream_ctx.builtin_tool_call_ids[tool_call_id] = builtin_tool_call_id
                    tool_call_id = builtin_tool_call_id

                message_id = stream_ctx.message_id or stream_ctx.new_message_id()
                yield ToolCallStartEvent(
                    tool_call_id=tool_call_id,
                    tool_call_name=part.tool_name,
                    parent_message_id=message_id,
                )
                if part.args:
                    yield ToolCallArgsEvent(
                        tool_call_id=tool_call_id,
                        delta=part.args_as_json_str(),
                    )
                stream_ctx.part_end = ToolCallEndEvent(
                    tool_call_id=tool_call_id,
                )
            elif isinstance(part, BuiltinToolReturnPart):  # pragma: no branch
                tool_call_id = stream_ctx.builtin_tool_call_ids[part.tool_call_id]
                yield ToolCallResultEvent(
                    message_id=stream_ctx.new_message_id(),
                    type=EventType.TOOL_CALL_RESULT,
                    role='tool',
                    tool_call_id=tool_call_id,
                    content=part.model_response_str(),
                )

    elif isinstance(agent_event, PartDeltaEvent):
        delta = agent_event.delta
        if isinstance(delta, TextPartDelta):
            if delta.content_delta:  # pragma: no branch
                yield TextMessageContentEvent(
                    message_id=stream_ctx.message_id,
                    delta=delta.content_delta,
                )
        elif isinstance(delta, ToolCallPartDelta):  # pragma: no branch
            tool_call_id = delta.tool_call_id
            assert tool_call_id, '`ToolCallPartDelta.tool_call_id` must be set'
            if tool_call_id in stream_ctx.builtin_tool_call_ids:
                tool_call_id = stream_ctx.builtin_tool_call_ids[tool_call_id]
            yield ToolCallArgsEvent(
                tool_call_id=tool_call_id,
                delta=delta.args_delta if isinstance(delta.args_delta, str) else json.dumps(delta.args_delta),
            )
        elif isinstance(delta, ThinkingPartDelta):  # pragma: no branch
            if delta.content_delta:  # pragma: no branch
                if not isinstance(stream_ctx.part_end, ThinkingTextMessageEndEvent):
                    yield ThinkingTextMessageStartEvent(
                        type=EventType.THINKING_TEXT_MESSAGE_START,
                    )
                    stream_ctx.part_end = ThinkingTextMessageEndEvent(
                        type=EventType.THINKING_TEXT_MESSAGE_END,
                    )

                yield ThinkingTextMessageContentEvent(
                    type=EventType.THINKING_TEXT_MESSAGE_CONTENT,
                    delta=delta.content_delta,
                )


async def _handle_tool_result_event(
    stream_ctx: _RequestStreamContext,
    event: FunctionToolResultEvent,
) -> AsyncIterator[BaseEvent]:
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

    yield ToolCallResultEvent(
        message_id=stream_ctx.new_message_id(),
        type=EventType.TOOL_CALL_RESULT,
        role='tool',
        tool_call_id=result.tool_call_id,
        content=result.model_response_str(),
    )

    # Now check for AG-UI events returned by the tool calls.
    possible_event = result.metadata or result.content
    if isinstance(possible_event, BaseEvent):
        yield possible_event
    elif isinstance(possible_event, str | bytes):  # pragma: no branch
        # Avoid iterable check for strings and bytes.
        pass
    elif isinstance(possible_event, Iterable):  # pragma: no branch
        for item in possible_event:  # type: ignore[reportUnknownMemberType]
            if isinstance(item, BaseEvent):  # pragma: no branch
                yield item


def _messages_from_ag_ui(messages: list[Message]) -> list[ModelMessage]:
    """Convert a AG-UI history to a Pydantic AI one."""
    result: list[ModelMessage] = []
    tool_calls: dict[str, str] = {}  # Tool call ID to tool name mapping.
    request_parts: list[ModelRequestPart] | None = None
    response_parts: list[ModelResponsePart] | None = None
    for msg in messages:
        if isinstance(msg, UserMessage | SystemMessage | DeveloperMessage) or (
            isinstance(msg, ToolMessage) and not msg.tool_call_id.startswith(_BUILTIN_TOOL_CALL_ID_PREFIX)
        ):
            if request_parts is None:
                request_parts = []
                result.append(ModelRequest(parts=request_parts))
                response_parts = None

            if isinstance(msg, UserMessage):
                request_parts.append(UserPromptPart(content=msg.content))
            elif isinstance(msg, SystemMessage | DeveloperMessage):
                request_parts.append(SystemPromptPart(content=msg.content))
            else:
                tool_call_id = msg.tool_call_id
                tool_name = tool_calls.get(tool_call_id)
                if tool_name is None:  # pragma: no cover
                    raise _ToolCallNotFoundError(tool_call_id=tool_call_id)

                request_parts.append(
                    ToolReturnPart(
                        tool_name=tool_name,
                        content=msg.content,
                        tool_call_id=tool_call_id,
                    )
                )

        elif isinstance(msg, AssistantMessage) or (  # pragma: no branch
            isinstance(msg, ToolMessage) and msg.tool_call_id.startswith(_BUILTIN_TOOL_CALL_ID_PREFIX)
        ):
            if response_parts is None:
                response_parts = []
                result.append(ModelResponse(parts=response_parts))
                request_parts = None

            if isinstance(msg, AssistantMessage):
                if msg.content:
                    response_parts.append(TextPart(content=msg.content))

                if msg.tool_calls:
                    for tool_call in msg.tool_calls:
                        tool_call_id = tool_call.id
                        tool_name = tool_call.function.name
                        tool_calls[tool_call_id] = tool_name

                        if tool_call_id.startswith(_BUILTIN_TOOL_CALL_ID_PREFIX):
                            _, provider_name, tool_call_id = tool_call_id.split('|', 2)
                            response_parts.append(
                                BuiltinToolCallPart(
                                    tool_name=tool_name,
                                    args=tool_call.function.arguments,
                                    tool_call_id=tool_call_id,
                                    provider_name=provider_name,
                                )
                            )
                        else:
                            response_parts.append(
                                ToolCallPart(
                                    tool_name=tool_name,
                                    tool_call_id=tool_call_id,
                                    args=tool_call.function.arguments,
                                )
                            )
            else:
                tool_call_id = msg.tool_call_id
                tool_name = tool_calls.get(tool_call_id)
                if tool_name is None:  # pragma: no cover
                    raise _ToolCallNotFoundError(tool_call_id=tool_call_id)
                _, provider_name, tool_call_id = tool_call_id.split('|', 2)

                response_parts.append(
                    BuiltinToolReturnPart(
                        tool_name=tool_name,
                        content=msg.content,
                        tool_call_id=tool_call_id,
                        provider_name=provider_name,
                    )
                )

    return result


@runtime_checkable
class StateHandler(Protocol):
    """Protocol for state handlers in agent runs. Requires the class to be a dataclass with a `state` field."""

    # Has to be a dataclass so we can use `replace` to update the state.
    # From https://github.com/python/typeshed/blob/9ab7fde0a0cd24ed7a72837fcb21093b811b80d8/stdlib/_typeshed/__init__.pyi#L352
    __dataclass_fields__: ClassVar[dict[str, Field[Any]]]

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


@dataclass(repr=False)
class _RequestStreamContext:
    """Data class to hold request stream context."""

    message_id: str = ''
    part_end: BaseEvent | None = None
    thinking: bool = False
    builtin_tool_call_ids: dict[str, str] = field(default_factory=dict)

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


class _AGUIFrontendToolset(ExternalToolset[AgentDepsT]):
    def __init__(self, tools: list[AGUITool]):
        super().__init__(
            [
                ToolDefinition(
                    name=tool.name,
                    description=tool.description,
                    parameters_json_schema=tool.parameters,
                )
                for tool in tools
            ]
        )

    @property
    def label(self) -> str:
        return 'the AG-UI frontend tools'  # pragma: no cover
