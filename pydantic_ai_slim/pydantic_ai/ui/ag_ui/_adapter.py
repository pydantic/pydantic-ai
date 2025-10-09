"""AG-UI adapter for handling requests."""

from __future__ import annotations

from functools import cached_property
from typing import (
    TYPE_CHECKING,
    Any,
    Final,
)

from ... import ExternalToolset, ToolDefinition
from ...agent import AgentDepsT
from ...messages import (
    BuiltinToolCallPart,
    BuiltinToolReturnPart,
    ModelMessage,
    ModelRequest,
    ModelRequestPart,
    ModelResponse,
    ModelResponsePart,
    SystemPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from ...toolsets import AbstractToolset

try:
    from ag_ui.core import (
        AssistantMessage,
        BaseEvent,
        DeveloperMessage,
        Message,
        RunAgentInput,
        SystemMessage,
        Tool as AGUITool,
        ToolMessage,
        UserMessage,
    )
    from ag_ui.encoder import EventEncoder

    from ..adapter import BaseAdapter
    from ..event_stream import BaseEventStream
    from ._event_stream import BUILTIN_TOOL_CALL_ID_PREFIX, AGUIEventStream
except ImportError as e:  # pragma: no cover
    raise ImportError(
        'Please install the `ag-ui-protocol` package to use AG-UI integration, '
        'you can use the `ag-ui` optional group — `pip install "pydantic-ai-slim[ag-ui]"`'
    ) from e

try:
    from starlette.requests import Request
except ImportError as e:  # pragma: no cover
    raise ImportError(
        'Please install the `starlette` package to use `Agent.to_ag_ui()` method, '
        'you can use the `ag-ui` optional group — `pip install "pydantic-ai-slim[ag-ui]"`'
    ) from e

if TYPE_CHECKING:
    pass

__all__ = ['AGUIAdapter']


SSE_CONTENT_TYPE: Final[str] = 'text/event-stream'
"""Content type header value for Server-Sent Events (SSE)."""

# Frontend toolset


class _AGUIFrontendToolset(ExternalToolset[AgentDepsT]):
    """Toolset for AG-UI frontend tools."""

    def __init__(self, tools: list[AGUITool]):
        """Initialize the toolset with AG-UI tools.

        Args:
            tools: List of AG-UI tool definitions.
        """
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
        """Return the label for this toolset."""
        return 'the AG-UI frontend tools'  # pragma: no cover


class AGUIAdapter(BaseAdapter[RunAgentInput, Message, BaseEvent, AgentDepsT]):
    """Adapter for handling AG-UI protocol requests with Pydantic AI agents.

    This adapter provides an interface for integrating Pydantic AI agents
    with the AG-UI protocol, handling request parsing, message conversion,
    and event streaming.

    Example:
        ```python
        from pydantic_ai import Agent
        from pydantic_ai.ui.ag_ui import AGUIAdapter

        agent = Agent('openai:gpt-4')
        adapter = AGUIAdapter(agent)

        async def handle_request(request: RunAgentInput, deps=None):
            async for event_str in adapter.run_stream_encoded(request, deps):
                yield event_str
        ```
    """

    def create_event_stream(self) -> BaseEventStream[RunAgentInput, BaseEvent, AgentDepsT]:
        """Create an event stream for the adapter."""
        return AGUIEventStream(self.request)

    @classmethod
    async def validate_request(cls, request: Request) -> RunAgentInput:
        """Validate the request and return the validated request."""
        return RunAgentInput.model_validate(await request.json())

    @cached_property
    def toolset(self) -> AbstractToolset[AgentDepsT] | None:
        """Get the toolset for the adapter."""
        if self.request.tools:
            return _AGUIFrontendToolset[AgentDepsT](self.request.tools)
        return None

    @cached_property
    def raw_state(self) -> dict[str, Any]:
        """Get the state of the agent run."""
        return self.request.state or {}

    def encode_event(self, event: BaseEvent, accept: str | None = None) -> str:
        """Encode an AG-UI event as SSE.

        Args:
            event: The AG-UI event to encode.
            accept: The accept header value for encoding format.

        Returns:
            The SSE-formatted string.
        """
        encoder = EventEncoder(accept=accept or SSE_CONTENT_TYPE)
        return encoder.encode(event)

    @cached_property
    def messages(self) -> list[ModelMessage]:
        """Convert AG-UI messages to Pydantic AI messages.

        Args:
            messages: List of AG-UI messages.

        Returns:
            List of Pydantic AI ModelMessage objects.
        """
        result: list[ModelMessage] = []
        tool_calls: dict[str, str] = {}  # Tool call ID to tool name mapping.
        request_parts: list[ModelRequestPart] | None = None
        response_parts: list[ModelResponsePart] | None = None

        for msg in self.request.messages:
            if isinstance(msg, UserMessage | SystemMessage | DeveloperMessage) or (
                isinstance(msg, ToolMessage) and not msg.tool_call_id.startswith(BUILTIN_TOOL_CALL_ID_PREFIX)
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
                        raise ValueError(f'Tool call with ID {tool_call_id} not found in the history.')

                    request_parts.append(
                        ToolReturnPart(
                            tool_name=tool_name,
                            content=msg.content,
                            tool_call_id=tool_call_id,
                        )
                    )

            elif isinstance(msg, AssistantMessage) or (  # pragma: no branch
                isinstance(msg, ToolMessage) and msg.tool_call_id.startswith(BUILTIN_TOOL_CALL_ID_PREFIX)
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

                            if tool_call_id.startswith(BUILTIN_TOOL_CALL_ID_PREFIX):
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
                        raise ValueError(f'Tool call with ID {tool_call_id} not found in the history.')
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
