"""AG-UI adapter for handling requests."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from functools import cached_property
from itertools import groupby
from typing import (
    TYPE_CHECKING,
    Any,
    cast,
)

from ... import ExternalToolset, ToolDefinition
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
from ...output import OutputDataT
from ...tools import AgentDepsT
from ...toolsets import AbstractToolset
from .. import MessagesBuilder

try:
    from ag_ui.core import (
        AssistantMessage,
        BaseEvent,
        DeveloperMessage,
        FunctionCall,
        Message,
        RunAgentInput,
        SystemMessage,
        Tool as AGUITool,
        ToolCall,
        ToolMessage,
        UserMessage,
    )

    from .. import UIAdapter, UIEventStream
    from ._event_stream import BUILTIN_TOOL_CALL_ID_PREFIX, AGUIEventStream
except ImportError as e:  # pragma: no cover
    raise ImportError(
        'Please install the `ag-ui-protocol` package to use AG-UI integration, '
        'you can use the `ag-ui` optional group — `pip install "pydantic-ai-slim[ag-ui]"`'
    ) from e

if TYPE_CHECKING:
    pass

__all__ = ['AGUIAdapter']


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


class AGUIAdapter(UIAdapter[RunAgentInput, Message, BaseEvent, AgentDepsT, OutputDataT]):
    """UI adapter for the Agent-User Interaction (AG-UI) protocol."""

    @classmethod
    def build_run_input(cls, body: bytes) -> RunAgentInput:
        """Build an AG-UI run input object from the request body."""
        return RunAgentInput.model_validate_json(body)

    def build_event_stream(self) -> UIEventStream[RunAgentInput, BaseEvent, AgentDepsT, OutputDataT]:
        """Build an AG-UI event stream transformer."""
        return AGUIEventStream(self.run_input, accept=self.accept)

    @cached_property
    def messages(self) -> list[ModelMessage]:
        """Pydantic AI messages from the AG-UI run input."""
        return self.load_messages(self.run_input.messages)

    @cached_property
    def toolset(self) -> AbstractToolset[AgentDepsT] | None:
        """Toolset representing frontend tools from the AG-UI run input."""
        if self.run_input.tools:
            return _AGUIFrontendToolset[AgentDepsT](self.run_input.tools)
        return None

    @cached_property
    def state(self) -> dict[str, Any] | None:
        """Frontend state from the AG-UI run input."""
        state = self.run_input.state
        if state is None:
            return None

        if isinstance(state, Mapping) and not state:
            return None

        return cast('dict[str, Any]', state)

    @classmethod
    def load_messages(cls, messages: Sequence[Message]) -> list[ModelMessage]:
        """Transform AG-UI messages into Pydantic AI messages."""
        builder = MessagesBuilder()
        tool_calls: dict[str, str] = {}  # Tool call ID to tool name mapping.

        for msg in messages:
            if isinstance(msg, UserMessage | SystemMessage | DeveloperMessage) or (
                isinstance(msg, ToolMessage) and not msg.tool_call_id.startswith(BUILTIN_TOOL_CALL_ID_PREFIX)
            ):
                if isinstance(msg, UserMessage):
                    builder.add(UserPromptPart(content=msg.content))
                elif isinstance(msg, SystemMessage | DeveloperMessage):
                    builder.add(SystemPromptPart(content=msg.content))
                else:
                    tool_call_id = msg.tool_call_id
                    tool_name = tool_calls.get(tool_call_id)
                    if tool_name is None:  # pragma: no cover
                        raise ValueError(f'Tool call with ID {tool_call_id} not found in the history.')

                    builder.add(
                        ToolReturnPart(
                            tool_name=tool_name,
                            content=msg.content,
                            tool_call_id=tool_call_id,
                        )
                    )

            elif isinstance(msg, AssistantMessage) or (  # pragma: no branch
                isinstance(msg, ToolMessage) and msg.tool_call_id.startswith(BUILTIN_TOOL_CALL_ID_PREFIX)
            ):
                if isinstance(msg, AssistantMessage):
                    if msg.content:
                        builder.add(TextPart(content=msg.content))

                    if msg.tool_calls:
                        for tool_call in msg.tool_calls:
                            tool_call_id = tool_call.id
                            tool_name = tool_call.function.name
                            tool_calls[tool_call_id] = tool_name

                            if tool_call_id.startswith(BUILTIN_TOOL_CALL_ID_PREFIX):
                                _, provider_name, tool_call_id = tool_call_id.split('|', 2)
                                builder.add(
                                    BuiltinToolCallPart(
                                        tool_name=tool_name,
                                        args=tool_call.function.arguments,
                                        tool_call_id=tool_call_id,
                                        provider_name=provider_name,
                                    )
                                )
                            else:
                                builder.add(
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

                    builder.add(
                        BuiltinToolReturnPart(
                            tool_name=tool_name,
                            content=msg.content,
                            tool_call_id=tool_call_id,
                            provider_name=provider_name,
                        )
                    )

        return builder.messages

    @classmethod
    def dump_messages(cls, messages: Sequence[ModelMessage]) -> list[Message]:
        """Transform Pydantic AI messages into AG-UI messages.

        Note: AG-UI message IDs are not preserved from load_messages().

        Args:
            messages: Sequence of Pydantic AI [`ModelMessage`][pydantic_ai.messages.ModelMessage] objects.

        Returns:
            List of AG-UI protocol messages.
        """
        ag_ui_messages: list[Message] = []
        message_id_counter = 1

        def get_next_id() -> str:
            nonlocal message_id_counter
            result = f'msg_{message_id_counter}'
            message_id_counter += 1
            return result

        for model_msg in messages:
            if isinstance(model_msg, ModelRequest):
                cls._convert_request_parts(model_msg.parts, ag_ui_messages, get_next_id)

            elif isinstance(model_msg, ModelResponse):
                cls._convert_response_parts(model_msg.parts, ag_ui_messages, get_next_id)

        return ag_ui_messages

    @staticmethod
    def _convert_request_parts(
        parts: Sequence[ModelRequestPart],
        ag_ui_messages: list[Message],
        get_next_id: Callable[[], str],
    ) -> None:
        """Convert ModelRequest parts to AG-UI messages."""
        for part in parts:
            msg_id = get_next_id()

            if isinstance(part, SystemPromptPart):
                ag_ui_messages.append(SystemMessage(id=msg_id, content=part.content))

            elif isinstance(part, UserPromptPart):
                content = part.content if isinstance(part.content, str) else str(part.content)
                ag_ui_messages.append(UserMessage(id=msg_id, content=content))

            elif isinstance(part, ToolReturnPart):
                ag_ui_messages.append(
                    ToolMessage(
                        id=msg_id,
                        content=AGUIAdapter._serialize_content(part.content),
                        tool_call_id=part.tool_call_id,
                    )
                )

    @staticmethod
    def _convert_response_parts(
        parts: Sequence[ModelResponsePart],
        ag_ui_messages: list[Message],
        get_next_id: Callable[[], str],
    ) -> None:
        """Convert ModelResponse parts to AG-UI messages."""

        # Group consecutive assistant parts (text, tool calls) together
        def is_assistant_part(part: ModelResponsePart) -> bool:
            return isinstance(part, TextPart | ToolCallPart | BuiltinToolCallPart)

        for is_assistant, group in groupby(parts, key=is_assistant_part):
            parts_list = list(group)

            if is_assistant:
                # Combine all parts into a single AssistantMessage
                content: str | None = None
                tool_calls: list[ToolCall] = []

                for part in parts_list:
                    if isinstance(part, TextPart):
                        content = part.content
                    elif isinstance(part, ToolCallPart):
                        tool_calls.append(AGUIAdapter._convert_tool_call(part))
                    elif isinstance(part, BuiltinToolCallPart):
                        tool_calls.append(AGUIAdapter._convert_builtin_tool_call(part))

                ag_ui_messages.append(
                    AssistantMessage(
                        id=get_next_id(),
                        content=content,
                        tool_calls=tool_calls if tool_calls else None,
                    )
                )
            else:
                # Each non-assistant part becomes its own message
                for part in parts_list:
                    if isinstance(part, BuiltinToolReturnPart):
                        ag_ui_messages.append(
                            ToolMessage(
                                id=get_next_id(),
                                content=AGUIAdapter._serialize_content(part.content),
                                tool_call_id=AGUIAdapter._make_builtin_tool_call_id(
                                    part.provider_name, part.tool_call_id
                                ),
                            )
                        )

    @staticmethod
    def _make_builtin_tool_call_id(provider_name: str | None, tool_call_id: str) -> str:
        """Create a full builtin tool call ID from provider name and tool call ID."""
        return f'{BUILTIN_TOOL_CALL_ID_PREFIX}|{provider_name}|{tool_call_id}'

    @staticmethod
    def _convert_tool_call(part: ToolCallPart) -> ToolCall:
        """Convert a ToolCallPart to an AG-UI ToolCall."""
        args_str = part.args if isinstance(part.args, str) else json.dumps(part.args)
        return ToolCall(
            id=part.tool_call_id,
            type='function',
            function=FunctionCall(
                name=part.tool_name,
                arguments=args_str,
            ),
        )

    @staticmethod
    def _convert_builtin_tool_call(part: BuiltinToolCallPart) -> ToolCall:
        """Convert a BuiltinToolCallPart to an AG-UI ToolCall."""
        args_str = part.args if isinstance(part.args, str) else json.dumps(part.args)
        return ToolCall(
            id=AGUIAdapter._make_builtin_tool_call_id(part.provider_name, part.tool_call_id),
            type='function',
            function=FunctionCall(
                name=part.tool_name,
                arguments=args_str,
            ),
        )

    @staticmethod
    def _serialize_content(content: Any) -> str:
        """Serialize content to a JSON string."""
        if isinstance(content, str):
            return content
        try:
            return json.dumps(content)
        except (TypeError, ValueError):
            # Fall back to str() if JSON serialization fails
            return str(content)
