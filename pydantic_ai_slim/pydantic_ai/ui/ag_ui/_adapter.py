"""AG-UI adapter for handling requests."""

from __future__ import annotations

import json
import uuid
from base64 import b64decode
from collections.abc import Mapping, Sequence
from dataclasses import KW_ONLY, dataclass
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
    UploadedFile,
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
        ReasoningMessage,
        RunAgentInput,
        SystemMessage,
        TextInputContent,
        Tool as AGUITool,
        ToolCall,
        ToolMessage,
        UserMessage,
    )

    from .. import MessagesBuilder, UIAdapter, UIEventStream
    from ._event_stream import BUILTIN_TOOL_CALL_ID_PREFIX, AGUIEventStream, thinking_encrypted_metadata
except ImportError as e:  # pragma: no cover
    raise ImportError(
        'Please install the `ag-ui-protocol` package to use AG-UI integration, '
        'you can use the `ag-ui` optional group — `pip install "pydantic-ai-slim[ag-ui]"`'
    ) from e

if TYPE_CHECKING:
    from starlette.requests import Request

    from ...agent import AbstractAgent

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
    item: str | ImageUrl | VideoUrl | AudioUrl | DocumentUrl | BinaryContent | UploadedFile | CachePoint,
) -> TextInputContent | BinaryInputContent | None:
    """Convert a user content item to AG-UI input content."""
    if isinstance(item, str):
        return TextInputContent(type='text', text=item)
    elif isinstance(item, (ImageUrl, VideoUrl, AudioUrl, DocumentUrl)):
        return BinaryInputContent(type='binary', url=item.url, mime_type=item.media_type or '')
    elif isinstance(item, BinaryContent):
        return BinaryInputContent(type='binary', data=item.base64, mime_type=item.media_type)
    elif isinstance(item, UploadedFile):
        # UploadedFile holds an opaque provider file_id (e.g. 'file-abc123'), not a URL or
        # binary data, so it can't be mapped to AG-UI's BinaryInputContent. Skipped like CachePoint.
        return None
    elif isinstance(item, CachePoint):
        return None
    else:
        assert_never(item)


@dataclass
class AGUIAdapter(UIAdapter[RunAgentInput, Message, BaseEvent, AgentDepsT, OutputDataT]):
    """UI adapter for the Agent-User Interaction (AG-UI) protocol."""

    _: KW_ONLY
    include_file_parts: bool = False
    """Whether to include ``FilePart`` data in message conversion.

    When ``True``, ``FilePart`` round-trips as ``ActivityMessage(activity_type='pydantic_ai_file')``.
    When ``False`` (default), ``FilePart`` is silently dropped from ``dump_messages`` output
    and ``ActivityMessage`` with ``activity_type='pydantic_ai_file'`` is ignored by ``load_messages``.
    """

    @classmethod
    def build_run_input(cls, body: bytes) -> RunAgentInput:
        """Build an AG-UI run input object from the request body."""
        return RunAgentInput.model_validate_json(body)

    def build_event_stream(self) -> UIEventStream[RunAgentInput, BaseEvent, AgentDepsT, OutputDataT]:
        """Build an AG-UI event stream transformer."""
        return AGUIEventStream(self.run_input, accept=self.accept)

    @classmethod
    async def from_request(
        cls,
        request: Request,
        *,
        agent: AbstractAgent[AgentDepsT, OutputDataT],
        include_file_parts: bool = False,
        **kwargs: Any,
    ) -> AGUIAdapter[AgentDepsT, OutputDataT]:
        """Extends [`from_request`][pydantic_ai.ui.UIAdapter.from_request] with the `include_file_parts` parameter."""
        return await super().from_request(request, agent=agent, include_file_parts=include_file_parts, **kwargs)

    @cached_property
    def messages(self) -> list[ModelMessage]:
        """Pydantic AI messages from the AG-UI run input."""
        return self.load_messages(self.run_input.messages, include_file_parts=self.include_file_parts)

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
    def load_messages(cls, messages: Sequence[Message], *, include_file_parts: bool = False) -> list[ModelMessage]:  # noqa: C901
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

                case ReasoningMessage() as reasoning_msg:
                    try:
                        metadata: dict[str, Any] = (
                            json.loads(reasoning_msg.encrypted_value) if reasoning_msg.encrypted_value else {}
                        )
                    except json.JSONDecodeError:
                        metadata = {}
                    builder.add(
                        ThinkingPart(
                            content=reasoning_msg.content,
                            id=metadata.get('id'),
                            signature=metadata.get('signature'),
                            provider_name=metadata.get('provider_name'),
                            provider_details=metadata.get('provider_details'),
                        )
                    )

                case ActivityMessage() as activity_msg:
                    if activity_msg.activity_type == 'pydantic_ai_file' and include_file_parts:
                        activity_content = activity_msg.content
                        url = activity_content.get('url', '')
                        if not url:
                            raise ValueError(
                                'ActivityMessage with activity_type=pydantic_ai_file must have a non-empty url.'
                            )
                        builder.add(
                            FilePart(
                                content=BinaryContent.from_data_uri(url),
                                id=activity_content.get('id'),
                                provider_name=activity_content.get('provider_name'),
                                provider_details=activity_content.get('provider_details'),
                            )
                        )

                case _:
                    assert_never(msg)

        return builder.messages

    @staticmethod
    def _dump_request_parts(msg: ModelRequest) -> list[Message]:
        """Convert a `ModelRequest` into AG-UI messages."""
        result: list[Message] = []
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
                result.append(
                    ToolMessage(
                        id=_new_message_id(),
                        content=part.model_response_str(),
                        tool_call_id=part.tool_call_id,
                    )
                )
            elif isinstance(part, RetryPromptPart):
                if part.tool_name:
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
        return messages

    @staticmethod
    def _dump_response_parts(msg: ModelResponse, *, include_file_parts: bool = False) -> list[Message]:  # noqa: C901
        """Convert a `ModelResponse` into AG-UI messages.

        Uses a flush pattern to preserve part ordering: text that appears after tool calls
        gets its own AssistantMessage, and ThinkingPart/FilePart boundaries trigger a flush
        so content on either side doesn't get merged.
        """
        result: list[Message] = []
        text_content: list[str] = []
        tool_calls_list: list[ToolCall] = []
        tool_messages: list[ToolMessage] = []

        builtin_returns = {part.tool_call_id: part for part in msg.parts if isinstance(part, BuiltinToolReturnPart)}

        def flush() -> None:
            nonlocal text_content, tool_calls_list, tool_messages
            if not text_content and not tool_calls_list:
                return
            result.append(
                AssistantMessage(
                    id=_new_message_id(),
                    content='\n'.join(text_content) if text_content else None,
                    tool_calls=tool_calls_list if tool_calls_list else None,
                )
            )
            result.extend(tool_messages)
            text_content = []
            tool_calls_list = []
            tool_messages = []

        for part in msg.parts:
            if isinstance(part, TextPart):
                if tool_calls_list:
                    flush()
                text_content.append(part.content)
            elif isinstance(part, ThinkingPart):
                flush()
                encrypted = thinking_encrypted_metadata(part)
                result.append(
                    ReasoningMessage(
                        id=_new_message_id(),
                        content=part.content,
                        encrypted_value=json.dumps(encrypted) if encrypted else None,
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
                if builtin_return := builtin_returns.get(part.tool_call_id):
                    tool_messages.append(
                        ToolMessage(
                            id=_new_message_id(),
                            content=builtin_return.model_response_str(),
                            tool_call_id=prefixed_id,
                        )
                    )
            elif isinstance(part, BuiltinToolReturnPart):
                # Emitted when matching BuiltinToolCallPart is processed above.
                pass
            elif isinstance(part, FilePart):
                if include_file_parts:
                    flush()
                    file_content: dict[str, Any] = {
                        'url': part.content.data_uri,
                        'media_type': part.content.media_type,
                    }
                    if part.id is not None:
                        file_content['id'] = part.id
                    if part.provider_name is not None:
                        file_content['provider_name'] = part.provider_name
                    if part.provider_details is not None:
                        file_content['provider_details'] = part.provider_details
                    result.append(
                        ActivityMessage(
                            id=_new_message_id(),
                            activity_type='pydantic_ai_file',
                            content=file_content,
                        )
                    )
            else:
                assert_never(part)

        flush()
        return result

    @classmethod
    def dump_messages(cls, messages: Sequence[ModelMessage], *, include_file_parts: bool = False) -> list[Message]:
        """Transform Pydantic AI messages into AG-UI messages.

        Note: The round-trip ``dump_messages`` -> ``load_messages`` is not fully lossless:

        - ``TextPart.id``, ``.provider_name``, ``.provider_details`` are lost.
        - ``ToolCallPart.id``, ``.provider_name``, ``.provider_details`` are lost.
        - ``RetryPromptPart`` becomes ``ToolReturnPart`` (or ``UserPromptPart``) on reload.
        - ``CachePoint`` and ``UploadedFile`` content items are dropped.
        - ``FilePart`` is silently dropped unless ``include_file_parts=True``.
        - Part ordering within a ``ModelResponse`` may change when text follows tool calls.

        Args:
            messages: A sequence of ModelMessage objects to convert.
            include_file_parts: Whether to include ``FilePart`` as ``ActivityMessage``.

        Returns:
            A list of AG-UI Message objects.
        """
        result: list[Message] = []

        for msg in messages:
            if isinstance(msg, ModelRequest):
                request_messages = cls._dump_request_parts(msg)
                result.extend(request_messages)
            elif isinstance(msg, ModelResponse):
                result.extend(cls._dump_response_parts(msg, include_file_parts=include_file_parts))
            else:
                assert_never(msg)

        return result
