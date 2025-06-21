"""Provides an AG-UI protocol adapter for the PydanticAI agent."""

from __future__ import annotations

import json
import logging
import uuid
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Generic, cast

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
    SystemMessage,
    TextMessageContentEvent,
    TextMessageEndEvent,
    TextMessageStartEvent,
    Tool as ToolAGUI,
    ToolCallArgsEvent,
    ToolCallEndEvent,
    ToolCallStartEvent,
    ToolMessage,
    UserMessage,
)
from ag_ui.encoder import EventEncoder

from pydantic_ai import Agent, ModelRequestNode, models
from pydantic_ai._output import OutputType
from pydantic_ai.agent import RunOutputDataT
from pydantic_ai.mcp import ToolResult
from pydantic_ai.messages import (
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
from pydantic_ai.result import AgentStream, OutputDataT
from pydantic_ai.settings import ModelSettings
from pydantic_ai.tools import AgentDepsT, Tool
from pydantic_ai.usage import Usage, UsageLimits

from ._enums import Role
from ._exceptions import NoMessagesError, RunError, UnexpectedToolCallError
from .consts import SSE_CONTENT_TYPE
from .protocols import StateHandler

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from ag_ui.encoder import EventEncoder

    from pydantic_ai._agent_graph import AgentNode
    from pydantic_ai.agent import AgentRun
    from pydantic_ai.result import FinalResult
    from pydantic_graph.nodes import End


_LOGGER: logging.Logger = logging.getLogger(__name__)
_LOGGER.setLevel(logging.DEBUG)


@dataclass(repr=False)
class _RequestStreamContext:
    """Data class to hold request stream context."""

    message_id: str = ''
    last_tool_call_id: str | None = None
    part_ends: list[str | None] = field(default_factory=lambda: list[str | None]())
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
class AdapterAGUI(Generic[AgentDepsT, OutputDataT]):
    """An agent adapter providing AG-UI protocol support for PydanticAI agents.

    This class manages the agent runs, tool calls, and provides an adapter
    for running agents with Server-Sent Event (SSE) streaming responses using
    the AG-UI protocol.

    It supports multiple threads and runs, allowing for concurrent agent
    execution and tool calls.

    Example:

    .. code-block:: python
        from __future__ import annotations

        from typing import TYPE_CHECKING, Annotated

        from fastapi import FastAPI, Header
        from fastapi.responses import StreamingResponse
        from pydantic_ai import Agent

        from adapter_ag_ui import SSE_CONTENT_TYPE, AdapterAGUI

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

    PydanticAI Tools which return AG-UI events will be sent to the client
    as part of the event stream, single events and iterables of events are
    supported.

    Examples:
    .. code-block:: python
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

        Args:
            run_input: The AG-UI run input containing thread_id, run_id, messages, etc.
            accept: The accept header value for the run.

            These arguments map directly to the `Agent.iter` method:

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

            prompt: str | None = None
            if isinstance(run_input.messages[-1], UserMessage):
                prompt = run_input.messages[-1].content
                run_input.messages.pop()

            run: AgentRun[AgentDepsT, Any]
            async with self.agent.iter(
                user_prompt=prompt,
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
                async for event in self._agent_stream(encoder, tool_names, run):
                    if event is None:
                        # Tool call signals early return, so we stop processing.
                        self.logger.debug('tool call early return')
                        break

                    yield event
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

    async def _tool_events(
        self, encoder: EventEncoder, parts: list[ModelRequestPart]
    ) -> AsyncGenerator[str | None, None]:
        """Check for tool call results that are AG-UI events.

        Args:
            encoder: The event encoder to use for encoding events.
            parts: The list of request parts to check for tool event returns.

        Yields:
            AG-UI Server-Sent Events (SSE).
        """
        # TODO(steve): Determine how to handle when we have multiple parts. Currently
        # AG-UI only seems to support a single tool call per request.
        part: ModelRequestPart
        for part in parts:
            if not isinstance(part, ToolReturnPart):
                continue

            iter: Iterable[Any]
            match part.content:
                case BaseEvent():
                    self.logger.debug('ag-ui event: %s', part.content)
                    yield encoder.encode(part.content)
                case str() | bytes():
                    # Avoid strings and bytes being checked as iterable.
                    pass
                case Iterable() as iter:
                    for item in iter:
                        if isinstance(item, BaseEvent):  # pragma: no branch
                            self.logger.debug('ag-ui event: %s', item)
                            yield encoder.encode(item)
                case _:  # pragma: no cover
                    # Not currently interested in other types.
                    pass

    def _convert_tools(self, run_tools: list[ToolAGUI]) -> list[Tool[AgentDepsT]]:
        """Convert AG-UI tools to PydanticAI tools.

        Creates placeholder Tool objects from AG-UI tool definitions. These tools
        don't actually execute anything - they just provide the necessary tool
        definitions and names for the PydanticAI agent.

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
            A PydanticAI Tool object that calls the AG-UI tool.
        """

        def _tool_stub(*args: Any, **kwargs: Any) -> ToolResult:
            """Stub function which is never called.

            Returns:
                Never returns as it always raises an exception.

            Raises:
                UnexpectedToolCallError: Always raised since this is a stub.
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
        encoder: EventEncoder,
        tool_names: dict[str, str],
        run: AgentRun[AgentDepsT, Any],
    ) -> AsyncGenerator[str | None, None]:
        """Run the agent streaming responses using AG-UI protocol events.

        Args:
            encoder: The event encoder to use for encoding events.
            tool_names: A mapping of tool names to their AG-UI names.
            run: The agent run to process.

        Yields:
            AG-UI Server-Sent Events (SSE).
        """
        node: AgentNode[AgentDepsT, Any] | End[FinalResult[Any]]
        msg: str | None
        async for node in run:
            self.logger.debug('processing node=%r', node)
            if not isinstance(node, ModelRequestNode):
                # Not interested UserPromptNode, CallToolsNode or End.
                continue

            # Check for state updates.
            snapshot: str | None
            async for snapshot in self._tool_events(encoder, node.request.parts):
                yield snapshot

            stream_ctx: _RequestStreamContext = _RequestStreamContext()
            request_stream: AgentStream[AgentDepsT]
            async with node.stream(run.ctx) as request_stream:
                agent_event: AgentStreamEvent
                async for agent_event in request_stream:
                    async for msg in self._handle_agent_event(encoder, tool_names, stream_ctx, agent_event):
                        yield msg

                for part_end in stream_ctx.part_ends:
                    yield part_end

    async def _handle_agent_event(
        self,
        encoder: EventEncoder,
        tool_names: dict[str, str],
        stream_ctx: _RequestStreamContext,
        agent_event: AgentStreamEvent,
    ) -> AsyncGenerator[str | None, None]:
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
        match agent_event:
            case PartStartEvent():
                # If we have a previous part end it.
                part_end: str | None
                for part_end in stream_ctx.part_ends:
                    yield part_end
                stream_ctx.part_ends.clear()

                match agent_event.part:
                    case TextPart():
                        message_id: str = stream_ctx.new_message_id()
                        yield encoder.encode(
                            TextMessageStartEvent(
                                type=EventType.TEXT_MESSAGE_START,
                                message_id=message_id,
                                role=Role.ASSISTANT.value,
                            ),
                        )
                        stream_ctx.part_ends = [
                            encoder.encode(
                                TextMessageEndEvent(
                                    type=EventType.TEXT_MESSAGE_END,
                                    message_id=message_id,
                                ),
                            ),
                        ]
                        if agent_event.part.content:
                            yield encoder.encode(  # pragma: no cover
                                TextMessageContentEvent(
                                    type=EventType.TEXT_MESSAGE_CONTENT,
                                    message_id=message_id,
                                    delta=agent_event.part.content,
                                ),
                            )
                    case ToolCallPart():
                        tool_name: str | None = tool_names.get(agent_event.part.tool_name)
                        if not tool_name:
                            stream_ctx.local_tool_calls.add(agent_event.part.tool_call_id)
                            return

                        stream_ctx.last_tool_call_id = agent_event.part.tool_call_id
                        yield encoder.encode(
                            ToolCallStartEvent(
                                type=EventType.TOOL_CALL_START,
                                tool_call_id=agent_event.part.tool_call_id,
                                tool_call_name=tool_name or agent_event.part.tool_name,
                            ),
                        )
                        stream_ctx.part_ends = [
                            encoder.encode(
                                ToolCallEndEvent(
                                    type=EventType.TOOL_CALL_END,
                                    tool_call_id=agent_event.part.tool_call_id,
                                ),
                            ),
                            None,  # Signal continuation of the stream.
                        ]
                    case ThinkingPart():  # pragma: no cover
                        # No equivalent AG-UI event yet.
                        pass
            case PartDeltaEvent():
                match agent_event.delta:
                    case TextPartDelta():
                        yield encoder.encode(
                            TextMessageContentEvent(
                                type=EventType.TEXT_MESSAGE_CONTENT,
                                message_id=stream_ctx.message_id,
                                delta=agent_event.delta.content_delta,
                            ),
                        )
                    case ToolCallPartDelta():
                        if agent_event.delta.tool_call_id in stream_ctx.local_tool_calls:
                            # Local tool calls are not sent to the UI.
                            return

                        yield encoder.encode(
                            ToolCallArgsEvent(
                                type=EventType.TOOL_CALL_ARGS,
                                tool_call_id=agent_event.delta.tool_call_id
                                or stream_ctx.last_tool_call_id
                                or 'unknown',  # Should never be unknown, but just in case.
                                delta=agent_event.delta.args_delta
                                if isinstance(agent_event.delta.args_delta, str)
                                else json.dumps(agent_event.delta.args_delta),
                            ),
                        )
                    case ThinkingPartDelta():  # pragma: no cover
                        # No equivalent AG-UI event yet.
                        pass
            case FinalResultEvent():
                # No equivalent AG-UI event yet.
                pass


def _convert_history(messages: list[Message]) -> list[ModelMessage]:
    """Convert a list of AG-UI history to a PydanticAI one.

    Args:
        messages: List of `ag_ui.Message` to convert.

    Returns:
        List of `ModelMessage` compatible with PydanticAI.
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
