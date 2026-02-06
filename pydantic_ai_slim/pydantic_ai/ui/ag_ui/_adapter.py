"""AG-UI adapter for handling requests."""

from __future__ import annotations

import uuid
from base64 import b64decode
from collections.abc import Mapping, Sequence
from functools import cached_property
from typing import (
    TYPE_CHECKING,
    Any,
    cast,
)

from typing_extensions import assert_never

from ... import ExternalToolset, ToolDefinition
from ...messages import (
    AudioUrl,
    BinaryContent,
    BuiltinToolCallPart,
    BuiltinToolReturnPart,
    CachePoint,
    DocumentUrl,
    FilePart,
    ImageUrl,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
    SystemPromptPart,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
    VideoUrl,
)
from ...output import OutputDataT
from ...tools import AgentDepsT
from ...toolsets import AbstractToolset

try:
    from ag_ui.core import (
        ActivityMessage,
        AssistantMessage,
        BaseEvent,
        BinaryInputContent,
        DeveloperMessage,
        FunctionCall,
        Message,
        RunAgentInput,
        SystemMessage,
        TextInputContent,
        Tool as AGUITool,
        ToolCall,
        ToolMessage,
        UserMessage,
    )

    from .. import MessagesBuilder, UIAdapter, UIEventStream
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


def _new_message_id() -> str:
    """Generate a new unique message ID."""
    return str(uuid.uuid4())


def _user_content_to_input(
    item: str | ImageUrl | VideoUrl | AudioUrl | DocumentUrl | BinaryContent | CachePoint,
) -> TextInputContent | BinaryInputContent | None:
    """Convert a user content item to AG-UI input content."""
    if isinstance(item, str):
        return TextInputContent(type='text', text=item)
    elif isinstance(item, (ImageUrl, VideoUrl, AudioUrl, DocumentUrl)):
        return BinaryInputContent(type='binary', url=item.url, mime_type=item.media_type or '')
    elif isinstance(item, BinaryContent):
        return BinaryInputContent(type='binary', data=item.base64, mime_type=item.media_type)
    elif isinstance(item, CachePoint):
        return None
    else:
        assert_never(item)


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
    def load_messages(cls, messages: Sequence[Message]) -> list[ModelMessage]:  # noqa: C901
        """Transform AG-UI messages into Pydantic AI messages."""
        builder = MessagesBuilder()
        tool_calls: dict[str, str] = {}  # Tool call ID to tool name mapping.
        for msg in messages:
            match msg:
                case UserMessage(content=content):
                    if isinstance(content, str):
                        builder.add(UserPromptPart(content=content))
                    else:
                        user_prompt_content: list[Any] = []
                        for part in content:
                            match part:
                                case TextInputContent(text=text):
                                    user_prompt_content.append(text)
                                case BinaryInputContent():
                                    if part.url:
                                        try:
                                            binary_part = BinaryContent.from_data_uri(part.url)
                                        except ValueError:
                                            media_type_constructors = {
                                                'image': ImageUrl,
                                                'video': VideoUrl,
                                                'audio': AudioUrl,
                                            }
                                            media_type_prefix = part.mime_type.split('/', 1)[0]
                                            constructor = media_type_constructors.get(media_type_prefix, DocumentUrl)
                                            binary_part = constructor(url=part.url, media_type=part.mime_type)
                                    elif part.data:
                                        binary_part = BinaryContent(
                                            data=b64decode(part.data), media_type=part.mime_type
                                        )
                                    else:  # pragma: no cover
                                        raise ValueError('BinaryInputContent must have either a `url` or `data` field.')
                                    user_prompt_content.append(binary_part)
                                case _:
                                    assert_never(part)

                        if user_prompt_content:  # pragma: no branch
                            content_to_add = (
                                user_prompt_content[0]
                                if len(user_prompt_content) == 1 and isinstance(user_prompt_content[0], str)
                                else user_prompt_content
                            )
                            builder.add(UserPromptPart(content=content_to_add))

                case SystemMessage(content=content) | DeveloperMessage(content=content):
                    builder.add(SystemPromptPart(content=content))

                case AssistantMessage(content=content, tool_calls=tool_calls_list):
                    if content:
                        builder.add(TextPart(content=content))
                    if tool_calls_list:
                        for tool_call in tool_calls_list:
                            tool_call_id = tool_call.id
                            tool_name = tool_call.function.name
                            tool_calls[tool_call_id] = tool_name

                            if tool_call_id.startswith(BUILTIN_TOOL_CALL_ID_PREFIX):
                                _, provider_name, original_id = tool_call_id.split('|', 2)
                                builder.add(
                                    BuiltinToolCallPart(
                                        tool_name=tool_name,
                                        args=tool_call.function.arguments,
                                        tool_call_id=original_id,
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
                case ToolMessage() as tool_msg:
                    tool_call_id = tool_msg.tool_call_id
                    tool_name = tool_calls.get(tool_call_id)
                    if tool_name is None:  # pragma: no cover
                        raise ValueError(f'Tool call with ID {tool_call_id} not found in the history.')

                    if tool_call_id.startswith(BUILTIN_TOOL_CALL_ID_PREFIX):
                        _, provider_name, original_id = tool_call_id.split('|', 2)
                        builder.add(
                            BuiltinToolReturnPart(
                                tool_name=tool_name,
                                content=tool_msg.content,
                                tool_call_id=original_id,
                                provider_name=provider_name,
                            )
                        )
                    else:
                        builder.add(
                            ToolReturnPart(
                                tool_name=tool_name,
                                content=tool_msg.content,
                                tool_call_id=tool_call_id,
                            )
                        )

                case ActivityMessage() as activity_msg:
                    # Round-trip from ActivitySnapshotEvent emitted by _event_stream.py.
                    # See: https://docs.ag-ui.com/concepts/messages#activitymessage
                    content = activity_msg.content
                    if activity_msg.activity_type == 'pydantic_ai_thinking':
                        builder.add(
                            ThinkingPart(
                                content=content.get('content', ''),
                                id=content.get('id'),
                                signature=content.get('signature'),
                                provider_name=content.get('provider_name'),
                                provider_details=content.get('provider_details'),
                            )
                        )
                    elif activity_msg.activity_type == 'pydantic_ai_file':
                        builder.add(
                            FilePart(
                                content=BinaryContent.from_data_uri(content.get('url', '')),
                                id=content.get('id'),
                                provider_name=content.get('provider_name'),
                                provider_details=content.get('provider_details'),
                            )
                        )

                case _:  # pragma: no cover
                    raise ValueError(f'Unsupported message type: {type(msg)}')

        return builder.messages

    @staticmethod
    def _dump_request_parts(msg: ModelRequest) -> tuple[list[Message], dict[str, str]]:
        """Convert a `ModelRequest` into AG-UI messages.

        Returns:
            A tuple of (messages, tool_call_id_to_name mapping).
        """
        result: list[Message] = []
        tool_call_names: dict[str, str] = {}
        system_content: list[str] = []
        user_content: list[TextInputContent | BinaryInputContent] = []

        for part in msg.parts:
            if isinstance(part, SystemPromptPart):
                system_content.append(part.content)
            elif isinstance(part, UserPromptPart):
                if isinstance(part.content, str):
                    user_content.append(TextInputContent(type='text', text=part.content))
                else:
                    for item in part.content:
                        converted = _user_content_to_input(item)
                        if converted is not None:
                            user_content.append(converted)
            elif isinstance(part, ToolReturnPart):
                tool_call_names[part.tool_call_id] = part.tool_name
                result.append(
                    ToolMessage(
                        id=_new_message_id(),
                        content=part.model_response_str(),
                        tool_call_id=part.tool_call_id,
                    )
                )
            elif isinstance(part, RetryPromptPart):
                if part.tool_name:
                    tool_call_names[part.tool_call_id] = part.tool_name
                    result.append(
                        ToolMessage(
                            id=_new_message_id(),
                            content=part.model_response(),
                            tool_call_id=part.tool_call_id,
                            error=part.model_response(),
                        )
                    )
                else:
                    user_content.append(TextInputContent(type='text', text=part.model_response()))
            else:
                assert_never(part)

        messages: list[Message] = []
        if system_content:
            messages.append(SystemMessage(id=_new_message_id(), content='\n'.join(system_content)))
        if user_content:
            # Simplify to plain string if only single text item
            if len(user_content) == 1 and isinstance(user_content[0], TextInputContent):
                messages.append(UserMessage(id=_new_message_id(), content=user_content[0].text))
            else:
                messages.append(UserMessage(id=_new_message_id(), content=user_content))
        messages.extend(result)
        return messages, tool_call_names

    @staticmethod
    def _dump_response_parts(msg: ModelResponse) -> list[Message]:
        """Convert a `ModelResponse` into AG-UI messages."""
        result: list[Message] = []
        text_content: list[str] = []
        tool_calls_list: list[ToolCall] = []
        builtin_tool_returns: list[BuiltinToolReturnPart] = []

        for part in msg.parts:
            if isinstance(part, TextPart):
                text_content.append(part.content)
            elif isinstance(part, ThinkingPart):
                thinking_content: dict[str, Any] = {'content': part.content}
                for attr in ['id', 'signature', 'provider_name', 'provider_details']:
                    if getattr(part, attr) is not None:
                        thinking_content[attr] = getattr(part, attr)
                result.append(
                    ActivityMessage(
                        id=_new_message_id(),
                        activity_type='pydantic_ai_thinking',
                        content=thinking_content,
                    )
                )
            elif isinstance(part, ToolCallPart):
                tool_calls_list.append(
                    ToolCall(
                        id=part.tool_call_id,
                        function=FunctionCall(name=part.tool_name, arguments=part.args_as_json_str()),
                    )
                )
            elif isinstance(part, BuiltinToolCallPart):
                prefixed_id = '|'.join([BUILTIN_TOOL_CALL_ID_PREFIX, part.provider_name or '', part.tool_call_id])
                tool_calls_list.append(
                    ToolCall(
                        id=prefixed_id,
                        function=FunctionCall(name=part.tool_name, arguments=part.args_as_json_str()),
                    )
                )
            elif isinstance(part, BuiltinToolReturnPart):
                builtin_tool_returns.append(part)
            elif isinstance(part, FilePart):
                file_content: dict[str, Any] = {
                    'url': part.content.data_uri,
                    'media_type': part.content.media_type,
                }
                for attr in ['id', 'provider_name', 'provider_details']:
                    if getattr(part, attr) is not None:
                        file_content[attr] = getattr(part, attr)
                result.append(
                    ActivityMessage(
                        id=_new_message_id(),
                        activity_type='pydantic_ai_file',
                        content=file_content,
                    )
                )
            else:
                assert_never(part)

        if text_content or tool_calls_list:
            result.append(
                AssistantMessage(
                    id=_new_message_id(),
                    content='\n'.join(text_content) if text_content else None,
                    tool_calls=tool_calls_list if tool_calls_list else None,
                )
            )

        for part in builtin_tool_returns:
            prefixed_id = '|'.join([BUILTIN_TOOL_CALL_ID_PREFIX, part.provider_name or '', part.tool_call_id])
            result.append(
                ToolMessage(
                    id=_new_message_id(),
                    content=part.model_response_str(),
                    tool_call_id=prefixed_id,
                )
            )

        return result

    @classmethod
    def dump_messages(cls, messages: Sequence[ModelMessage]) -> list[Message]:
        """Transform Pydantic AI messages into AG-UI messages.

        Args:
            messages: A sequence of ModelMessage objects to convert.

        Returns:
            A list of AG-UI Message objects.
        """
        result: list[Message] = []

        for msg in messages:
            if isinstance(msg, ModelRequest):
                request_messages, _ = cls._dump_request_parts(msg)
                result.extend(request_messages)
            elif isinstance(msg, ModelResponse):
                result.extend(cls._dump_response_parts(msg))
            else:
                assert_never(msg)

        return result
