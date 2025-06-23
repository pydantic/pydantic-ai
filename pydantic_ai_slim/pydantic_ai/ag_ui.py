"""Provides an AG-UI protocol adapter for the PydanticAI agent.

This package provides seamless integration between pydantic-ai agents and ag-ui
for building interactive AI applications with streaming event-based communication.
"""

from __future__ import annotations

import json
import logging
import uuid
from collections.abc import Iterable, Sequence
from dataclasses import InitVar, dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Final, Generic, Protocol, TypeVar, cast, runtime_checkable

try:
    from ag_ui.core import (
        AssistantMessage,
        BaseEvent,
        DeveloperMessage,
        EventType,
        FunctionCall,
        Message,
        MessagesSnapshotEvent,
        RunAgentInput,
        RunErrorEvent,
        RunFinishedEvent,
        RunStartedEvent,
        State,
        SystemMessage,
        TextMessageContentEvent,
        TextMessageEndEvent,
        TextMessageStartEvent,
        Tool as ToolAGUI,
        ToolCall,
        ToolCallArgsEvent,
        ToolCallEndEvent,
        ToolCallStartEvent,
        ToolMessage,
        UserMessage,
    )
    from ag_ui.encoder import EventEncoder
except ImportError as e:
    raise ImportError(
        'Please install the `ag-ui-protocol` package to use `Agent.to_ag_ui()` method, '
        'you can use the `ag-ui` optional group — `pip install "pydantic-ai-slim[ag-ui]"`'
    ) from e

from pydantic import BaseModel, ValidationError

from . import Agent, models
from ._agent_graph import ModelRequestNode
from ._output import OutputType
from ._parts_manager import ModelResponsePartsManager
from .agent import RunOutputDataT
from .mcp import ToolResult
from .messages import (
    AgentStreamEvent,
    FinalResultEvent,
    ModelMessage,
    ModelRequest,
    ModelRequestPart,
    ModelResponse,
    ModelResponsePart,
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
from .result import AgentStream, OutputDataT
from .settings import ModelSettings
from .tools import AgentDepsT, Tool
from .usage import Usage, UsageLimits

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from ag_ui.encoder import EventEncoder

    from pydantic_graph.nodes import End

    from ._agent_graph import AgentNode
    from .agent import AgentRun
    from .result import FinalResult


_LOGGER: logging.Logger = logging.getLogger(__name__)


# Constants.
SSE_CONTENT_TYPE: Final[str] = 'text/event-stream'
"""Content type header value for Server-Sent Events (SSE)."""


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
class RunError(Exception):
    """Exception raised for errors during agent runs."""

    message: str
    code: str

    def __str__(self) -> str:
        return self.message


@dataclass(kw_only=True)
class UnexpectedToolCallError(RunError):
    """Exception raised when an unexpected tool call is encountered."""

    tool_name: InitVar[str]
    message: str = ''
    code: str = 'unexpected_tool_call'

    def __post_init__(self, tool_name: str) -> None:
        """Set the message for the unexpected tool call.

        Args:
            tool_name: The name of the tool that was unexpectedly called.
        """
        self.message = f'unexpected tool call name={tool_name}'  # pragma: no cover


@dataclass
class NoMessagesError(RunError):
    """Exception raised when no messages are found in the input."""

    message: str = 'no messages found in the input'
    code: str = 'no_messages'


@dataclass
class InvalidStateError(RunError, ValidationError):
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


@dataclass(kw_only=True)
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
            raise InvalidStateError from e


@dataclass(repr=False)
class _RequestStreamContext:
    """Data class to hold request stream context."""

    message_id: str = ''
    last_tool_call_id: str | None = None
    part_ends: list[BaseEvent | None] = field(default_factory=lambda: list[BaseEvent | None]())
    local_tool_calls: set[str] = field(default_factory=set)

    def new_message_id(self) -> str:
        """Generate a new message ID for the request stream.

        Assigns a new UUID to the `message_id` and returns it.

        Returns:
            A new message ID.
        """
        self.message_id = str(uuid.uuid4())
        return self.message_id


@dataclass(kw_only=True, repr=False)
class Adapter(Generic[AgentDepsT, OutputDataT]):
    """An agent adapter providing AG-UI protocol support for PydanticAI agents.

    This class manages the agent runs, tool calls, state storage and providing
    an adapter for running agents with Server-Sent Event (SSE) streaming
    responses using the AG-UI protocol.

    Examples:
    This is an example of base usage with FastAPI.
    ```python
    from __future__ import annotations

    from typing import TYPE_CHECKING, Annotated

    from fastapi import FastAPI, Header
    from fastapi.responses import StreamingResponse
    from pydantic_ai import Agent

    from pydantic_ai.ag_ui import SSE_CONTENT_TYPE, Adapter

    if TYPE_CHECKING:
        from ag_ui.core import RunAgentInput

    app = FastAPI(title="AG-UI Endpoint")
    agent = Agent(
        "openai:gpt-4o-mini",
        deps_type=int,
        instructions="You are a helpful assistant.",
    )
    adapter = agent.to_ag_ui()

    @app.post("/")
    async def root(input_data: RunAgentInput, accept: Annotated[str, Header()] = SSE_CONTENT_TYPE) -> StreamingResponse:
        return StreamingResponse(
            adapter.run(input_data, accept, deps=42),
            media_type=SSE_CONTENT_TYPE,
        )
    ```

    PydanticAI tools which return AG-UI events will be sent to the client
    as part of the event stream, single events and event iterables are
    supported.
    ```python
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
                name="count",
                value=1,
            ),
            CustomEvent(
                type=EventType.CUSTOM,
                name="count",
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
        output_type: OutputType[RunOutputDataT] | None = None,
        model: models.Model | models.KnownModelName | str | None = None,
        deps: AgentDepsT = None,
        model_settings: ModelSettings | None = None,
        usage_limits: UsageLimits | None = None,
        usage: Usage | None = None,
        infer_name: bool = True,
        additional_tools: Sequence[Tool[AgentDepsT]] | None = None,
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
            additional_tools: Additional tools to use for this run.

        Yields:
            Streaming SSE-formatted event chunks.
        """
        self.logger.debug('starting run: %s', json.dumps(run_input.model_dump(), indent=2))

        tool_names: dict[str, str] = {self.tool_prefix + tool.name: tool.name for tool in run_input.tools}
        encoder: EventEncoder = EventEncoder(accept=accept)
        run_tools: list[Tool[AgentDepsT]] = list(additional_tools) if additional_tools else []
        run_tools.extend(self._convert_tools(run_input.tools))

        try:
            yield encoder.encode(
                RunStartedEvent(
                    type=EventType.RUN_STARTED,
                    thread_id=run_input.thread_id,
                    run_id=run_input.run_id,
                ),
            )

            if not run_input.messages:
                raise NoMessagesError

            if isinstance(deps, StateHandler):
                deps.set_state(run_input.state)

            run: AgentRun[AgentDepsT, Any]
            async with self.agent.iter(
                user_prompt=None,
                output_type=output_type,
                message_history=_convert_history(run_input.messages),
                model=model,
                deps=deps,
                model_settings=model_settings,
                usage_limits=usage_limits,
                usage=usage,
                infer_name=infer_name,
                additional_tools=run_tools,
            ) as run:
                parts_manager: ModelResponsePartsManager = ModelResponsePartsManager()
                async for event in self._agent_stream(tool_names, run, parts_manager):
                    if event is None:
                        # Tool call signals early return, so we stop processing.
                        self.logger.debug('tool call early return')

                        # TODO(steve): Remove this workaround, it's only needed as AG-UI doesn't
                        # currently have a way to add server side tool calls to the message history
                        # via events. To workaround this we create a full snapshot of the messages
                        # and send that.
                        snapshot: MessagesSnapshotEvent | None = self._message_snapshot(
                            run, run_input.messages, parts_manager
                        )
                        if snapshot is not None:
                            yield encoder.encode(snapshot)
                        break

                    yield encoder.encode(event)
        except RunError as e:
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

        self.logger.info('done thread_id=%s run_id=%s', run_input.thread_id, run_input.run_id)

    def _message_snapshot(
        self, run: AgentRun[AgentDepsT, Any], messages: list[Message], parts_manager: ModelResponsePartsManager
    ) -> MessagesSnapshotEvent | None:
        """Create a message snapshot to replicate the current state of the run.

        This method collects all messages from the run's state and the parts
        manager, converting them into AG-UI messages.

        Args:
            run: The agent run instance.
            messages: The initial messages from the run input.
            parts_manager: The parts manager containing the response parts.

        Returns:
            A full snapshot of the messages so far in the run if local tool
            calls were made, otherwise `None`.
        """
        new_messages: list[ModelMessage] = run.ctx.state.message_history[len(messages) :]
        if not any(
            isinstance(request_part, ToolReturnPart)
            for msg in new_messages
            if isinstance(msg, ModelRequest)
            for request_part in msg.parts
        ):
            # No tool calls were made, so we don't need a snapshot.
            return None

        # Tool calls were made, so we need to create a snapshot.
        for msg in new_messages:
            match msg:
                case ModelRequest():
                    for request_part in msg.parts:
                        if isinstance(request_part, ToolReturnPart):
                            messages.append(
                                ToolMessage(
                                    id='result-' + request_part.tool_call_id,
                                    role=Role.TOOL,
                                    content=request_part.content,
                                    tool_call_id=request_part.tool_call_id,
                                )
                            )
                case ModelResponse():
                    self._convert_response_parts(msg.parts, messages)

        self._convert_response_parts(parts_manager.get_parts(), messages)

        return MessagesSnapshotEvent(
            type=EventType.MESSAGES_SNAPSHOT,
            messages=messages,
        )

    def _convert_response_parts(self, parts: list[ModelResponsePart], messages: list[Message]) -> None:
        """Convert model response parts to AG-UI messages.

        Args:
            parts: The list of model response parts to convert.
            messages: The list of messages to append the converted parts to.
        """
        response_part: ModelResponsePart
        for response_part in parts:
            match response_part:
                case TextPart():  # pragma: no cover
                    # This is not expected, but we handle it gracefully.
                    messages.append(
                        AssistantMessage(
                            id=uuid.uuid4().hex,
                            role=Role.ASSISTANT,
                            content=response_part.content,
                        )
                    )
                case ToolCallPart():
                    args: str = (
                        json.dumps(response_part.args)
                        if isinstance(response_part.args, dict)
                        else response_part.args or '{}'
                    )
                    messages.append(
                        AssistantMessage(
                            id=uuid.uuid4().hex,
                            role=Role.ASSISTANT,
                            tool_calls=[
                                ToolCall(
                                    id=response_part.tool_call_id,
                                    type='function',
                                    function=FunctionCall(
                                        name=response_part.tool_name,
                                        arguments=args,
                                    ),
                                )
                            ],
                        ),
                    )
                case ThinkingPart():  # pragma: no cover
                    # No AG-UI equivalent for thinking parts, so we skip them.
                    pass

    async def _tool_events(self, parts: list[ModelRequestPart]) -> AsyncGenerator[BaseEvent | None, None]:
        """Check for tool call results that are AG-UI events.

        Args:
            encoder: The event encoder to use for encoding events.
            parts: The list of request parts to check for tool event returns.

        Yields:
            AG-UI Server-Sent Events (SSE).
        """
        # TODO(steve): Determine how to handle multiple parts. Currently
        # AG-UI only supports a single tool call per request, but that
        # may change in the future.
        part: ModelRequestPart
        for part in parts:
            if not isinstance(part, ToolReturnPart):
                continue

            iter: Iterable[Any]
            match part.content:
                case BaseEvent():
                    self.logger.debug('ag-ui event: %s', part.content)
                    yield part.content
                case str() | bytes():
                    # Avoid strings and bytes being checked as iterable.
                    pass
                case Iterable() as iter:
                    for item in iter:
                        if isinstance(item, BaseEvent):  # pragma: no branch
                            self.logger.debug('ag-ui event: %s', item)
                            yield item
                case _:  # pragma: no cover
                    # Not currently interested in other types.
                    pass

    def _convert_tools(self, run_tools: list[ToolAGUI]) -> list[Tool[AgentDepsT]]:
        """Convert AG-UI tools to PydanticAI tools.

        Creates `Tool` objects from AG-UI tool definitions. These tools don't
        actually execute anything, that is done by AG-UI client - they just
        provide the necessary tool definitions to PydanticAI agent.

        Args:
            run_tools: List of AG-UI tool definitions to convert.

        Returns:
            List of PydanticAI Tool objects that call the AG-UI tools.
        """
        return [self._tool_call(tool) for tool in run_tools]

    def _tool_call(self, tool: ToolAGUI) -> Tool[AgentDepsT]:
        """Create a PydanticAI tool from an AG-UI tool definition.

        Args:
            tool: The AG-UI tool definition to convert.

        Returns:
            A PydanticAI `Tool` object that calls the AG-UI tool.
        """

        def _tool_stub(*args: Any, **kwargs: Any) -> ToolResult:
            """Stub function which is never called.

            Returns:
                Never returns as it always raises an exception.

            Raises:
                UnexpectedToolCallError: Always raised since this should never be called.
            """
            raise UnexpectedToolCallError(tool_name=tool.name)  # pragma: no cover

        # TODO(steve): See it we can avoid the cast here.
        return cast(
            'Tool[AgentDepsT]',
            Tool.from_schema(
                function=_tool_stub,
                name=tool.name,
                description=tool.description,
                json_schema=tool.parameters,
            ),
        )

    async def _agent_stream(
        self,
        tool_names: dict[str, str],
        run: AgentRun[AgentDepsT, Any],
        parts_manager: ModelResponsePartsManager,
    ) -> AsyncGenerator[BaseEvent | None, None]:
        """Run the agent streaming responses using AG-UI protocol events.

        Args:
            tool_names: A mapping of tool names to their AG-UI names.
            run: The agent run to process.
            parts_manager: The parts manager to handle tool call parts.

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

            # Check for state updates.
            snapshot: BaseEvent | None
            async for snapshot in self._tool_events(node.request.parts):
                yield snapshot

            stream_ctx: _RequestStreamContext = _RequestStreamContext()
            request_stream: AgentStream[AgentDepsT]
            async with node.stream(run.ctx) as request_stream:
                agent_event: AgentStreamEvent
                async for agent_event in request_stream:
                    async for msg in self._handle_agent_event(tool_names, stream_ctx, agent_event, parts_manager):
                        yield msg

                for part_end in stream_ctx.part_ends:
                    yield part_end

    async def _handle_agent_event(
        self,
        tool_names: dict[str, str],
        stream_ctx: _RequestStreamContext,
        agent_event: AgentStreamEvent,
        parts_manager: ModelResponsePartsManager,
    ) -> AsyncGenerator[BaseEvent | None, None]:
        """Handle an agent event and yield AG-UI protocol events.

        Args:
            encoder: The event encoder to use for encoding events.
            tool_names: A mapping of tool names to their AG-UI names.
            stream_ctx: The request stream context to manage state.
            agent_event: The agent event to process.
            parts_manager: The parts manager to handle tool call parts.

        Yields:
            AG-UI Server-Sent Events (SSE) based on the agent event.
        """
        self.logger.debug('agent_event: %s', agent_event)
        match agent_event:
            case PartStartEvent():
                # If we have a previous part end it.
                part_end: BaseEvent | None
                for part_end in stream_ctx.part_ends:
                    yield part_end
                stream_ctx.part_ends.clear()

                match agent_event.part:
                    case TextPart():
                        message_id: str = stream_ctx.new_message_id()
                        yield TextMessageStartEvent(
                            type=EventType.TEXT_MESSAGE_START,
                            message_id=message_id,
                            role=Role.ASSISTANT.value,
                        )
                        stream_ctx.part_ends = [
                            TextMessageEndEvent(
                                type=EventType.TEXT_MESSAGE_END,
                                message_id=message_id,
                            ),
                        ]
                        if agent_event.part.content:
                            yield TextMessageContentEvent(  # pragma: no cover
                                type=EventType.TEXT_MESSAGE_CONTENT,
                                message_id=message_id,
                                delta=agent_event.part.content,
                            )
                    case ToolCallPart():  # pragma: no branch
                        tool_name: str | None = tool_names.get(agent_event.part.tool_name)
                        if not tool_name:
                            # Local tool calls are not sent as events to the UI.
                            stream_ctx.local_tool_calls.add(agent_event.part.tool_call_id)
                            return

                        parts_manager.handle_tool_call_part(
                            vendor_part_id=None,
                            tool_name=agent_event.part.tool_name,
                            args=agent_event.part.args,
                            tool_call_id=agent_event.part.tool_call_id,
                        )
                        stream_ctx.last_tool_call_id = agent_event.part.tool_call_id
                        yield ToolCallStartEvent(
                            type=EventType.TOOL_CALL_START,
                            tool_call_id=agent_event.part.tool_call_id,
                            tool_call_name=tool_name or agent_event.part.tool_name,
                        )
                        stream_ctx.part_ends = [
                            ToolCallEndEvent(
                                type=EventType.TOOL_CALL_END,
                                tool_call_id=agent_event.part.tool_call_id,
                            ),
                            None,  # Signal continuation of the stream.
                        ]
                    case ThinkingPart():  # pragma: no branch
                        # No equivalent AG-UI event yet.
                        pass
            case PartDeltaEvent():
                match agent_event.delta:
                    case TextPartDelta():
                        yield TextMessageContentEvent(
                            type=EventType.TEXT_MESSAGE_CONTENT,
                            message_id=stream_ctx.message_id,
                            delta=agent_event.delta.content_delta,
                        )
                    case ToolCallPartDelta():  # pragma: no branch
                        if agent_event.delta.tool_call_id in stream_ctx.local_tool_calls:
                            # Local tool calls are not sent as events to the UI.
                            return

                        parts_manager.handle_tool_call_delta(
                            vendor_part_id=None,
                            tool_name=None,
                            args=agent_event.delta.args_delta,
                            tool_call_id=agent_event.delta.tool_call_id,
                        )
                        yield ToolCallArgsEvent(
                            type=EventType.TOOL_CALL_ARGS,
                            tool_call_id=agent_event.delta.tool_call_id
                            or stream_ctx.last_tool_call_id
                            or 'unknown',  # Should never be unknown, but just in case.
                            delta=agent_event.delta.args_delta
                            if isinstance(agent_event.delta.args_delta, str)
                            else json.dumps(agent_event.delta.args_delta),
                        )
                    case ThinkingPartDelta():  # pragma: no branch
                        # No equivalent AG-UI event yet.
                        pass
            case FinalResultEvent():
                # No equivalent AG-UI event yet.
                pass


def _convert_history(messages: list[Message]) -> list[ModelMessage]:
    """Convert a AG-UI history to a PydanticAI one.

    Args:
        messages: List of AG-UI messages to convert.

    Returns:
        List of PydanticAI model messages.
    """
    msg: Message
    result: list[ModelMessage] = []
    tool_calls: dict[str, str] = {}
    for msg in messages:
        match msg:
            case UserMessage():
                result.append(ModelRequest(parts=[UserPromptPart(content=msg.content)]))
            case AssistantMessage():
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
            case SystemMessage():
                # TODO(steve): Should we handle as instructions instead of system prompt?
                result.append(ModelRequest(parts=[SystemPromptPart(content=msg.content)]))
            case ToolMessage():
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
            case DeveloperMessage():  # pragma: no branch
                # TODO(steve): Should these be handled differently?
                result.append(ModelRequest(parts=[SystemPromptPart(content=msg.content)]))

    return result


# =====================================================================================
# Exports
# =====================================================================================

__all__ = [
    'Adapter',
    'SSE_CONTENT_TYPE',
    'StateDeps',
    'StateHandler',
    'Role',
    'RunError',
    'UnexpectedToolCallError',
    'NoMessagesError',
    'InvalidStateError',
]
