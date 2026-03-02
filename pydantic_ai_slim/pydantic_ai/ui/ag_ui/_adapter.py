"""AG-UI adapter for handling requests."""

from __future__ import annotations

from base64 import b64decode
from collections.abc import Mapping, Sequence
from functools import cached_property
from typing import (
    TYPE_CHECKING,
    Any,
    cast,
)
from uuid import uuid4

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
        # `ToolMessage` only gives us the `tool_call_id`, so remember the tool name from the
        # earlier assistant message in order to reconstruct the matching return part later.
        tool_calls: dict[str, str] = {}
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
                                case _:  # pragma: no cover
                                    raise ValueError(f'Unsupported user message part type: {type(part)}')

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
                                # AG-UI persists built-in and regular tool calls with the same schema.
                                # We encode builtin provenance into a prefixed ID when dumping and decode it here.
                                provider_name, original_id = _load_builtin_tool_call_id(tool_call_id)
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
                        provider_name, original_id = _load_builtin_tool_call_id(tool_call_id)
                        if tool_msg.error is not None:
                            # Persisted AG-UI history has no dedicated retry message type, so retries roundtrip
                            # through `ToolMessage.error` while the full formatted text stays in `content`.
                            builder.add(
                                RetryPromptPart(
                                    tool_name=tool_name,
                                    content=tool_msg.error,
                                    tool_call_id=original_id,
                                )
                            )
                        else:
                            builder.add(
                                BuiltinToolReturnPart(
                                    tool_name=tool_name,
                                    content=tool_msg.content,
                                    tool_call_id=original_id,
                                    provider_name=provider_name,
                                )
                            )
                    else:
                        if tool_msg.error is not None:
                            # See note above: `ToolMessage.error` is our persisted-history carrier for retries.
                            builder.add(
                                RetryPromptPart(
                                    tool_name=tool_name,
                                    content=tool_msg.error,
                                    tool_call_id=tool_call_id,
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

                case ActivityMessage():
                    pass

        return builder.messages

    @staticmethod
    def _dump_tool_message(
        result: ToolReturnPart | BuiltinToolReturnPart | RetryPromptPart,
        *,
        tool_call_id: str,
    ) -> ToolMessage:
        if isinstance(result, (ToolReturnPart, BuiltinToolReturnPart)):
            return ToolMessage(
                id=_message_id(),
                content=result.model_response_str(),
                tool_call_id=tool_call_id,
            )
        elif isinstance(result, RetryPromptPart):
            error = result.content if isinstance(result.content, str) else result.model_response()
            return ToolMessage(
                id=_message_id(),
                content=result.model_response(),
                tool_call_id=tool_call_id,
                error=error,
            )
        else:
            assert_never(result)

    @classmethod
    def _dump_response_message(
        cls,
        msg: ModelResponse,
        tool_results: dict[str, ToolReturnPart | RetryPromptPart],
    ) -> list[Message]:
        messages: list[Message] = []
        assistant_content = ''
        assistant_tool_calls: list[ToolCall] = []
        assistant_tool_messages: list[ToolMessage] = []

        builtin_returns = {part.tool_call_id: part for part in msg.parts if isinstance(part, BuiltinToolReturnPart)}
        used_builtin_return_ids: set[str] = set()

        def flush_assistant_message() -> None:
            nonlocal assistant_content, assistant_tool_calls, assistant_tool_messages
            if not assistant_content and not assistant_tool_calls:
                return

            messages.append(
                AssistantMessage(
                    id=_message_id(),
                    content=assistant_content or None,
                    tool_calls=assistant_tool_calls or None,
                )
            )
            messages.extend(assistant_tool_messages)
            assistant_content = ''
            assistant_tool_calls = []
            assistant_tool_messages = []

        for part in msg.parts:
            if isinstance(part, TextPart):
                if assistant_tool_calls:
                    # AG-UI can store text and tool calls together on one assistant message, but text that appears
                    # after a tool interruption needs a new assistant message to preserve ordering on roundtrip.
                    flush_assistant_message()
                assistant_content += part.content
            elif isinstance(part, ToolCallPart):
                assistant_tool_calls.append(
                    ToolCall(
                        id=part.tool_call_id,
                        function=FunctionCall(name=part.tool_name, arguments=part.args_as_json_str()),
                    )
                )
                if tool_result := tool_results.get(part.tool_call_id):
                    assistant_tool_messages.append(cls._dump_tool_message(tool_result, tool_call_id=part.tool_call_id))
            elif isinstance(part, BuiltinToolCallPart):
                dumped_tool_call_id = _dump_builtin_tool_call_id(part.tool_call_id, part.provider_name)
                assistant_tool_calls.append(
                    ToolCall(
                        id=dumped_tool_call_id,
                        function=FunctionCall(name=part.tool_name, arguments=part.args_as_json_str()),
                    )
                )
                if builtin_return := builtin_returns.get(part.tool_call_id):
                    used_builtin_return_ids.add(part.tool_call_id)
                    assistant_tool_messages.append(
                        cls._dump_tool_message(builtin_return, tool_call_id=dumped_tool_call_id)
                    )
            elif isinstance(part, BuiltinToolReturnPart):
                # Built-in tool returns are emitted when the matching BuiltinToolCallPart is processed.
                pass
            elif isinstance(part, ThinkingPart):
                # `ag-ui-protocol==0.1.10` does not expose a persisted assistant reasoning message type.
                # We flush before skipping so text on either side does not get merged across the omission.
                flush_assistant_message()
            elif isinstance(part, FilePart):
                raise ValueError('AG-UI integration cannot persist assistant file parts in message history.')
            else:
                assert_never(part)

        orphaned_builtin_returns = builtin_returns.keys() - used_builtin_return_ids
        if orphaned_builtin_returns:
            raise ValueError('Built-in tool return parts must be paired with matching BuiltinToolCallParts.')

        flush_assistant_message()
        return messages

    @classmethod
    def _dump_request_message(cls, msg: ModelRequest) -> list[Message]:
        messages: list[Message] = []

        for part in msg.parts:
            if isinstance(part, SystemPromptPart):
                messages.append(SystemMessage(id=_message_id(), content=part.content))
            elif isinstance(part, UserPromptPart):
                if user_message := _convert_user_prompt_part(part):
                    messages.append(user_message)
            elif isinstance(part, ToolReturnPart):
                # Tool results are emitted immediately after their assistant tool-call message.
                pass
            elif isinstance(part, RetryPromptPart):
                if part.tool_name is None:
                    messages.append(UserMessage(id=_message_id(), content=part.model_response()))
                # Tool retries are emitted after their assistant tool-call message.
            else:
                assert_never(part)

        return messages

    @staticmethod
    def _collect_tool_results(messages: Sequence[ModelMessage]) -> dict[str, ToolReturnPart | RetryPromptPart]:
        tool_results: dict[str, ToolReturnPart | RetryPromptPart] = {}

        for msg in messages:
            if isinstance(msg, ModelRequest):
                for part in msg.parts:
                    # Pydantic AI stores tool results on subsequent requests, while AG-UI persists them as
                    # separate `tool` messages immediately after the assistant tool call that triggered them.
                    if isinstance(part, ToolReturnPart):
                        tool_results[part.tool_call_id] = part
                    elif isinstance(part, RetryPromptPart) and part.tool_name:
                        tool_results[part.tool_call_id] = part

        return tool_results

    @classmethod
    def dump_messages(cls, messages: Sequence[ModelMessage]) -> list[Message]:
        """Transform Pydantic AI messages into AG-UI messages."""
        tool_results = cls._collect_tool_results(messages)
        dumped_messages: list[Message] = []
        for msg in messages:
            if isinstance(msg, ModelRequest):
                dumped_messages.extend(cls._dump_request_message(msg))
            elif isinstance(msg, ModelResponse):
                dumped_messages.extend(cls._dump_response_message(msg, tool_results))
            else:
                assert_never(msg)

        return dumped_messages


def _message_id() -> str:
    return uuid4().hex


def _dump_builtin_tool_call_id(tool_call_id: str, provider_name: str | None) -> str:
    return '|'.join([BUILTIN_TOOL_CALL_ID_PREFIX, provider_name or '', tool_call_id])


def _load_builtin_tool_call_id(tool_call_id: str) -> tuple[str | None, str]:
    _, provider_name, original_id = tool_call_id.split('|', 2)
    return provider_name or None, original_id


def _convert_user_prompt_part(part: UserPromptPart) -> UserMessage | None:
    if isinstance(part.content, str):
        return UserMessage(id=_message_id(), content=part.content)

    content: list[TextInputContent | BinaryInputContent] = []
    for item in part.content:
        if isinstance(item, str):
            content.append(TextInputContent(text=item))
        elif isinstance(item, BinaryContent):
            content.append(BinaryInputContent(data=item.base64, mime_type=item.media_type))
        elif isinstance(item, (ImageUrl, AudioUrl, VideoUrl, DocumentUrl)):
            content.append(BinaryInputContent(url=item.url, mime_type=item.media_type))
        elif isinstance(item, CachePoint):
            # Cache points affect model prompt caching only and do not belong in persisted UI history.
            pass
        else:
            assert_never(item)

    if not content:
        return None

    if len(content) == 1 and isinstance(content[0], TextInputContent):
        return UserMessage(id=_message_id(), content=content[0].text)

    return UserMessage(id=_message_id(), content=content)
