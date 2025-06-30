"""Provides an AG-UI protocol adapter for the PydanticAI agent.

This package provides seamless integration between pydantic-ai agents and ag-ui
for building interactive AI applications with streaming event-based communication.
"""

from __future__ import annotations

import json
import logging
import uuid
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    TYPE_CHECKING,
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
        Tool as ToolAGUI,
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
        'Please install the `fasta2a` package to use `Agent.to_ag_ui()` method, '
        'you can use the `ag-ui` optional group — `pip install "pydantic-ai-slim[ag-ui]"`'
    ) from e

from pydantic import BaseModel, ValidationError

from pydantic_ai.output import DeferredToolCalls
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.toolsets import AbstractToolset
from pydantic_ai.toolsets.deferred import DeferredToolset

from . import Agent, models
from ._agent_graph import ModelRequestNode
from .agent import RunOutputDataT
from .messages import (
    AgentStreamEvent,
    FinalResultEvent,
    ModelMessage,
    ModelRequest,
    ModelRequestPart,
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
from .output import OutputDataT, OutputSpec
from .result import AgentStream
from .settings import ModelSettings
from .tools import AgentDepsT
from .usage import Usage, UsageLimits

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from ag_ui.encoder import EventEncoder

    from pydantic_graph.nodes import End

    from ._agent_graph import AgentNode
    from .agent import AgentRun
    from .result import FinalResult

# Variables.
_LOGGER: logging.Logger = logging.getLogger(__name__)

# Constants.
SSE_CONTENT_TYPE: Final[str] = 'text/event-stream'
"""Content type header value for Server-Sent Events (SSE)."""


class FastAGUI(Generic[AgentDepsT, OutputDataT], Starlette):
    """A FastAPI-like application for running PydanticAI agents with AG-UI protocol support."""

    def __init__(
        self,
        *,
        # Adapter for the agent.
        adapter: Adapter[AgentDepsT, OutputDataT],
        path: str = '/',
        # Agent.iter parameters.
        output_type: OutputSpec[OutputDataT] = str,
        model: models.Model | models.KnownModelName | str | None = None,
        deps: AgentDepsT = None,
        model_settings: ModelSettings | None = None,
        usage_limits: UsageLimits | None = None,
        usage: Usage | None = None,
        infer_name: bool = True,
        toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
        # Starlette
        debug: bool = False,
        routes: Sequence[BaseRoute] | None = None,
        middleware: Sequence[Middleware] | None = None,
        exception_handlers: Mapping[Any, ExceptionHandler] | None = None,
        on_startup: Sequence[Callable[[], Any]] | None = None,
        on_shutdown: Sequence[Callable[[], Any]] | None = None,
        lifespan: Lifespan[FastAGUI[AgentDepsT, OutputDataT]] | None = None,
    ) -> None:
        """Initialize the FastAGUI application.

        Args:
            adapter: The adapter to use for running the agent.
            path: The path to serve the agent run endpoint.

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

        async def endpoint(request: Request) -> Response | StreamingResponse:
            """Endpoint to run the agent with the provided input data."""
            accept: str = request.headers.get('accept', SSE_CONTENT_TYPE)
            try:
                input_data: RunAgentInput = RunAgentInput.model_validate_json(await request.body())
            except ValidationError as e:  # pragma: no cover
                _LOGGER.error('invalid request: %s', e)
                return Response(
                    content=json.dumps(e.json()),
                    media_type='application/json',
                    status_code=400,
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

        self.router.add_route(path, endpoint, methods=['POST'], name='run_agent')


def agent_to_ag_ui(
    *,
    # Adapter parameters.
    agent: Agent[AgentDepsT, OutputDataT],
    path: str = '/',
    tool_prefix: str = '',
    logger: logging.Logger | None = None,
    # Agent.iter parameters.
    output_type: OutputSpec[OutputDataT] = str,
    model: models.Model | models.KnownModelName | str | None = None,
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
    lifespan: Lifespan[FastAGUI[AgentDepsT, OutputDataT]] | None = None,
) -> FastAGUI[AgentDepsT, OutputDataT]:
    """Create a FastAGUI server from an agent.

    Args:
        agent: The PydanticAI agent to adapt for AG-UI protocol.
        path: The path to serve the agent run endpoint.
        tool_prefix: Optional prefix to add to tool names.
        logger: Optional logger to use for the adapter, defaults to the module's logger.

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
    if logger is None:  # pragma: no branch
        logger = _LOGGER

    adapter: Adapter[AgentDepsT, OutputDataT] = Adapter(
        agent=agent,
        tool_prefix=tool_prefix,
        logger=logger,
    )

    return FastAGUI(
        adapter=adapter,
        path=path,
        # Agent.iter parameter
        output_type=output_type,
        model=model,
        deps=deps,
        model_settings=model_settings,
        usage_limits=usage_limits,
        usage=usage,
        infer_name=infer_name,
        toolsets=toolsets,
        # Starlette
        debug=debug,
        routes=routes,
        middleware=middleware,
        exception_handlers=exception_handlers,
        on_startup=on_startup,
        on_shutdown=on_shutdown,
        lifespan=lifespan,
    )


@dataclass(repr=False)
class Adapter(Generic[AgentDepsT, OutputDataT]):
    """An agent adapter providing AG-UI protocol support for PydanticAI agents.

    This class manages the agent runs, tool calls, state storage and providing
    an adapter for running agents with Server-Sent Event (SSE) streaming
    responses using the AG-UI protocol.

    Examples:
    This is an example of base usage with FastAPI.
    ```python
    from pydantic_ai import Agent

    agent = Agent('openai:gpt-4.1', instructions='Be fun!')
    app = agent.to_ag_ui()
    ```

    PydanticAI tools which return AG-UI events will be sent to the client
    as part of the event stream, single events and event iterables are
    supported.
    ```python
    from ag_ui.core import CustomEvent, EventType, StateSnapshotEvent
    from pydantic import BaseModel

    from pydantic_ai import Agent, RunContext
    from pydantic_ai.ag_ui import StateDeps


    class DocumentState(BaseModel):
        document: str


    agent = Agent(
        'openai:gpt-4.1', instructions='Be fun!', deps_type=StateDeps[DocumentState]
    )


    @agent.tool
    def update_state(ctx: RunContext[StateDeps[DocumentState]]) -> StateSnapshotEvent:
        return StateSnapshotEvent(
            type=EventType.STATE_SNAPSHOT,
            snapshot=ctx.deps.state,
        )


    @agent.tool_plain
    def custom_events() -> list[CustomEvent]:
        return [
            CustomEvent(
                type=EventType.CUSTOM,
                name='count',
                value=1,
            ),
            CustomEvent(
                type=EventType.CUSTOM,
                name='count',
                value=2,
            ),
        ]
    ```
    Args:
        agent: The PydanticAI `Agent` to adapt.
        tool_prefix: Optional prefix to add to tool names.
        logger: The logger to use for logging.
    """

    agent: Agent[AgentDepsT, OutputDataT] = field(repr=False)
    tool_prefix: str = field(default='', repr=False)
    logger: logging.Logger = field(default=_LOGGER, repr=False)

    async def run(
        self,
        run_input: RunAgentInput,
        accept: str = SSE_CONTENT_TYPE,
        *,
        output_type: OutputSpec[RunOutputDataT] | None = None,
        model: models.Model | models.KnownModelName | str | None = None,
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
        self.logger.debug('starting run: %s', json.dumps(run_input.model_dump(), indent=2))

        tool_names: dict[str, str] = {self.tool_prefix + tool.name: tool.name for tool in run_input.tools}
        encoder: EventEncoder = EventEncoder(accept=accept)
        run_toolset: list[AbstractToolset[AgentDepsT]] = list(toolsets) if toolsets else []
        if run_input.tools:
            run_toolset.append(_AGUIToolset[AgentDepsT](run_input.tools))

        try:
            yield encoder.encode(
                RunStartedEvent(
                    type=EventType.RUN_STARTED,
                    thread_id=run_input.thread_id,
                    run_id=run_input.run_id,
                ),
            )

            if not run_input.messages:
                raise _NoMessagesError

            if isinstance(deps, StateHandler):
                deps.set_state(run_input.state)

            history: _History = _convert_history(run_input.messages)

            output_type_: OutputSpec[OutputDataT | DeferredToolCalls | RunOutputDataT]
            if output_type is None:
                # Use the agent's output type if not specified.
                output_type_ = [self.agent.output_type, DeferredToolCalls]
            else:
                output_type_ = [output_type, DeferredToolCalls]

            run: AgentRun[AgentDepsT, Any]
            async with self.agent.iter(
                user_prompt=None,
                # TODO(steve): Could or should it just accept: [output_type, DeferredToolCalls]
                output_type=output_type_,
                message_history=history.messages,
                model=model,
                deps=deps,
                model_settings=model_settings,
                usage_limits=usage_limits,
                usage=usage,
                infer_name=infer_name,
                toolsets=run_toolset,
            ) as run:
                async for event in self._agent_stream(tool_names, run, history):
                    if event is None:
                        # Tool call signals early return, so we stop processing.
                        self.logger.debug('tool call early return')
                        break

                    yield encoder.encode(event)
        except _RunError as e:
            self.logger.exception('agent run')
            yield encoder.encode(
                RunErrorEvent(type=EventType.RUN_ERROR, message=e.message, code=e.code),
            )
        except Exception as e:  # pragma: no cover
            self.logger.exception('unexpected error in agent run')
            yield encoder.encode(
                RunErrorEvent(type=EventType.RUN_ERROR, message=str(e), code='run_error'),
            )
        else:
            yield encoder.encode(
                RunFinishedEvent(
                    type=EventType.RUN_FINISHED,
                    thread_id=run_input.thread_id,
                    run_id=run_input.run_id,
                ),
            )

        self.logger.debug('done thread_id=%s run_id=%s', run_input.thread_id, run_input.run_id)

    async def _tool_events(
        self,
        parts: list[ModelRequestPart],
        history: _History,
    ) -> AsyncGenerator[BaseEvent | None, None]:
        """Check for tool call results that are AG-UI events.

        Args:
            encoder: The event encoder to use for encoding events.
            parts: The list of request parts to check for tool event returns.
            history: The history of messages and tool calls to use for the run.

        Yields:
            AG-UI Server-Sent Events (SSE).
        """
        part: ModelRequestPart
        for part in parts:
            if not isinstance(part, ToolReturnPart):
                continue

            if part.tool_call_id in history.tool_calls:
                # Tool call was passed in the history, so we skip it.
                continue

            yield ToolCallResultEvent(
                message_id=history.prompt_message_id,
                type=EventType.TOOL_CALL_RESULT,
                role=Role.TOOL.value,
                tool_call_id=part.tool_call_id,
                content=part.model_response_str(),
            )

            # Now check for  AG-UI events returned by the tool calls.
            iter: Iterable[Any]
            if isinstance(part.content, BaseEvent):
                self.logger.debug('ag-ui event: %s', part.content)
                yield part.content
            elif isinstance(part.content, (str, bytes)):  # pragma: no branch
                # Avoid iterable check for strings and bytes.
                pass
            elif isinstance(part.content, Iterable):  # pragma: no branch
                # Type: ignore to handle partially unknown type
                iter = part.content  # type: ignore[assignment]
                for item in iter:
                    if isinstance(item, BaseEvent):  # pragma: no branch
                        self.logger.debug('ag-ui event: %s', item)
                        yield item

    async def _agent_stream(
        self,
        tool_names: dict[str, str],
        run: AgentRun[AgentDepsT, Any],
        history: _History,
    ) -> AsyncGenerator[BaseEvent | None, None]:
        """Run the agent streaming responses using AG-UI protocol events.

        Args:
            tool_names: A mapping of tool names to their AG-UI names.
            run: The agent run to process.
            history: The history of messages and tool calls to use for the run.

        Yields:
            AG-UI Server-Sent Events (SSE).
        """
        node: AgentNode[AgentDepsT, Any] | End[FinalResult[Any]]
        msg: BaseEvent | None
        async for node in run:
            self.logger.debug('processing node=%r', node)
            if not isinstance(node, ModelRequestNode):
                # Not interested UserPromptNode, CallToolsNode or End.
                continue

            # Check for tool results.
            async for msg in self._tool_events(node.request.parts, history):
                yield msg

            stream_ctx: _RequestStreamContext = _RequestStreamContext()
            request_stream: AgentStream[AgentDepsT]
            async with node.stream(run.ctx) as request_stream:
                agent_event: AgentStreamEvent
                async for agent_event in request_stream:
                    async for msg in self._handle_agent_event(tool_names, stream_ctx, agent_event):
                        yield msg

                if stream_ctx.part_end:
                    yield stream_ctx.part_end
                    stream_ctx.part_end = None

    async def _handle_agent_event(
        self,
        tool_names: dict[str, str],
        stream_ctx: _RequestStreamContext,
        agent_event: AgentStreamEvent,
    ) -> AsyncGenerator[BaseEvent | None, None]:
        """Handle an agent event and yield AG-UI protocol events.

        Args:
            encoder: The event encoder to use for encoding events.
            tool_names: A mapping of tool names to their AG-UI names.
            stream_ctx: The request stream context to manage state.
            agent_event: The agent event to process.

        Yields:
            AG-UI Server-Sent Events (SSE) based on the agent event.
        """
        self.logger.debug('agent_event: %s', agent_event)
        if isinstance(agent_event, PartStartEvent):
            if stream_ctx.part_end:
                # End the previous part.
                yield stream_ctx.part_end
                stream_ctx.part_end = None

            if isinstance(agent_event.part, TextPart):
                message_id: str = stream_ctx.new_message_id()
                yield TextMessageStartEvent(
                    type=EventType.TEXT_MESSAGE_START,
                    message_id=message_id,
                    role=Role.ASSISTANT.value,
                )
                stream_ctx.part_end = TextMessageEndEvent(
                    type=EventType.TEXT_MESSAGE_END,
                    message_id=message_id,
                )
                if agent_event.part.content:
                    yield TextMessageContentEvent(  # pragma: no cover
                        type=EventType.TEXT_MESSAGE_CONTENT,
                        message_id=message_id,
                        delta=agent_event.part.content,
                    )
            elif isinstance(agent_event.part, ToolCallPart):  # pragma: no branch
                tool_name: str | None = tool_names.get(agent_event.part.tool_name)
                stream_ctx.last_tool_call_id = agent_event.part.tool_call_id
                yield ToolCallStartEvent(
                    type=EventType.TOOL_CALL_START,
                    tool_call_id=agent_event.part.tool_call_id,
                    tool_call_name=tool_name or agent_event.part.tool_name,
                )
                stream_ctx.part_end = ToolCallEndEvent(
                    type=EventType.TOOL_CALL_END,
                    tool_call_id=agent_event.part.tool_call_id,
                )

            elif isinstance(agent_event.part, ThinkingPart):  # pragma: no branch
                yield ThinkingTextMessageStartEvent(
                    type=EventType.THINKING_TEXT_MESSAGE_START,
                )
                if agent_event.part.content:  # pragma: no branch
                    yield ThinkingTextMessageContentEvent(
                        type=EventType.THINKING_TEXT_MESSAGE_CONTENT,
                        delta=agent_event.part.content,
                    )
                stream_ctx.part_end = ThinkingTextMessageEndEvent(
                    type=EventType.THINKING_TEXT_MESSAGE_END,
                )

        elif isinstance(agent_event, PartDeltaEvent):
            if isinstance(agent_event.delta, TextPartDelta):
                yield TextMessageContentEvent(
                    type=EventType.TEXT_MESSAGE_CONTENT,
                    message_id=stream_ctx.message_id,
                    delta=agent_event.delta.content_delta,
                )
            elif isinstance(agent_event.delta, ToolCallPartDelta):  # pragma: no branch
                yield ToolCallArgsEvent(
                    type=EventType.TOOL_CALL_ARGS,
                    tool_call_id=agent_event.delta.tool_call_id
                    or stream_ctx.last_tool_call_id
                    or 'unknown',  # Should never be unknown, but just in case.
                    delta=agent_event.delta.args_delta
                    if isinstance(agent_event.delta.args_delta, str)
                    else json.dumps(agent_event.delta.args_delta),
                )
            elif isinstance(agent_event.delta, ThinkingPartDelta):  # pragma: no cover
                yield ThinkingTextMessageContentEvent(
                    type=EventType.THINKING_TEXT_MESSAGE_CONTENT,
                    delta=agent_event.delta.content_delta or '',
                )
        elif isinstance(agent_event, FinalResultEvent):
            # No equivalent AG-UI event yet.
            pass


@dataclass
class _History:
    """A simple history representation for AG-UI protocol."""

    prompt_message_id: str  # The ID of the last user message.
    messages: list[ModelMessage]
    tool_calls: set[str] = field(default_factory=set)


def _convert_history(messages: list[Message]) -> _History:
    """Convert a AG-UI history to a PydanticAI one.

    Args:
        messages: List of AG-UI messages to convert.

    Returns:
        List of PydanticAI model messages.
    """
    msg: Message
    prompt_message_id: str = ''
    result: list[ModelMessage] = []
    tool_calls: dict[str, str] = {}  # Tool call ID to tool name mapping.
    for msg in messages:
        if isinstance(msg, UserMessage):
            prompt_message_id = msg.id
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
            # TODO(steve): Should we handle as instructions instead of system prompt?
            result.append(ModelRequest(parts=[SystemPromptPart(content=msg.content)]))
        elif isinstance(msg, ToolMessage):
            result.append(
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            tool_name=tool_calls.get(msg.tool_call_id, 'unknown'),
                            content=msg.content,
                            tool_call_id=msg.tool_call_id,
                        )
                    ]
                )
            )
        elif isinstance(msg, DeveloperMessage):  # pragma: no branch
            # TODO(steve): Should these be handled differently?
            result.append(ModelRequest(parts=[SystemPromptPart(content=msg.content)]))

    return _History(
        prompt_message_id=prompt_message_id,
        messages=result,
        tool_calls=set(tool_calls.keys()),
    )


__all__ = [
    'Adapter',
    'SSE_CONTENT_TYPE',
    'StateDeps',
    'StateHandler',
    'FastAGUI',
    'agent_to_ag_ui',
]


# Enums.
# TODO(steve): Remove this and all uses once https://github.com/ag-ui-protocol/ag-ui/pull/49 is merged.
class Role(str, Enum):
    """Enum for message roles in AG-UI protocol."""

    ASSISTANT = 'assistant'
    USER = 'user'
    DEVELOPER = 'developer'
    SYSTEM = 'system'
    TOOL = 'tool'


# Exceptions.
@dataclass
class _RunError(Exception):
    """Exception raised for errors during agent runs."""

    message: str
    code: str

    def __str__(self) -> str:
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


# Protocols.
@runtime_checkable
class StateHandler(Protocol):
    """Protocol for state handlers in agent runs."""

    def set_state(self, state: State) -> None:
        """Set the state of the agent run.

        This method is called to update the state of the agent run with the
        provided state.

        Args:
            state: The run state.

        Raises:
            ValidationError: If `state` does not match the expected model.
        """
        ...


StateT = TypeVar('StateT', bound=BaseModel, contravariant=True)
"""Type variable for the state type, which must be a subclass of `BaseModel`."""


@dataclass
class StateDeps(Generic[StateT]):
    """Provides AG-UI state management.

    This class is used to manage the state of an agent run. It allows setting
    the state of the agent run with a specific type of state model, which must
    be a subclass of `BaseModel`.

    The state is set using the `set_state` when the run starts by the `Adapter`.

    Implements the `StateHandler` protocol.
    """

    state_type: type[StateT]
    state: StateT = field(init=False)

    def set_state(self, state: State) -> None:
        """Set the state of the agent run.

        This method is called to update the state of the agent run with the
        provided state.

        Implements the `StateHandler` protocol.

        Args:
            state: The run state, which should match the expected model type or be `None`.

        Raises:
            InvalidStateError: If `state` does not match the expected model and is not `None`.
        """
        if state is None:
            return

        try:
            self.state = self.state_type.model_validate(state)
        except ValidationError as e:  # pragma: no cover
            raise _InvalidStateError from e


@dataclass(repr=False)
class _RequestStreamContext:
    """Data class to hold request stream context."""

    message_id: str = ''
    last_tool_call_id: str | None = None
    part_end: BaseEvent | None = None

    def new_message_id(self) -> str:
        """Generate a new message ID for the request stream.

        Assigns a new UUID to the `message_id` and returns it.

        Returns:
            A new message ID.
        """
        self.message_id = str(uuid.uuid4())
        return self.message_id


class _AGUIToolset(DeferredToolset[AgentDepsT]):
    """A toolset that is used for AG-UI."""

    def __init__(self, tools: list[ToolAGUI]) -> None:
        super().__init__(
            [
                ToolDefinition(name=tool.name, description=tool.description, parameters_json_schema=tool.parameters)
                for tool in tools
            ]
        )
