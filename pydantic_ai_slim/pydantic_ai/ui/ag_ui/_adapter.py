"""AG-UI adapter for handling requests."""

from __future__ import annotations

import json
import uuid
import warnings
from base64 import b64decode
from collections.abc import Mapping, Sequence
from dataclasses import KW_ONLY, dataclass
from functools import cached_property
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
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
    CompactionPart,
    DocumentUrl,
    FilePart,
    ImageUrl,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
    SystemPromptPart,
    TextContent,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolReturnPart,
    UploadedFile,
    UserContent,
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
    from ._event_stream import AGUIEventStream
    from ._utils import (
        BUILTIN_TOOL_CALL_ID_PREFIX,
        DEFAULT_AG_UI_VERSION,
        FILE_ACTIVITY_TYPE,
        MULTIMODAL_VERSION,
        REASONING_VERSION,
        UPLOADED_FILE_ACTIVITY_TYPE,
        parse_ag_ui_version,
        thinking_encrypted_metadata,
    )
except ImportError as e:  # pragma: no cover
    raise ImportError(
        'Please install the `ag-ui-protocol` package to use AG-UI integration, '
        'you can use the `ag-ui` optional group — `pip install "pydantic-ai-slim[ag-ui]"`'
    ) from e

if TYPE_CHECKING:
    from ag_ui.core import (
        AudioInputContent,
        DocumentInputContent,
        ImageInputContent,
        ReasoningMessage,
        VideoInputContent,
    )
    from starlette.requests import Request

    from ...agent import AbstractAgent
else:
    try:
        from ag_ui.core import ReasoningMessage
    except ImportError:  # pragma: no cover

        class ReasoningMessage:
            """Stub for ag-ui-protocol < 0.1.13 — no instances exist, so pattern matching is a no-op."""

    try:
        from ag_ui.core import AudioInputContent, DocumentInputContent, ImageInputContent, VideoInputContent
    except ImportError:  # pragma: no cover

        class ImageInputContent:
            """Stub for ag-ui-protocol < 0.1.15."""

        class AudioInputContent:
            """Stub for ag-ui-protocol < 0.1.15."""

        class VideoInputContent:
            """Stub for ag-ui-protocol < 0.1.15."""

        class DocumentInputContent:
            """Stub for ag-ui-protocol < 0.1.15."""


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
                    parameters_json_schema=tool.parameters or {},
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
    item: str | TextContent | ImageUrl | VideoUrl | AudioUrl | DocumentUrl | BinaryContent | UploadedFile | CachePoint,
    *,
    use_multimodal: bool = False,
) -> (
    TextInputContent
    | BinaryInputContent
    | ImageInputContent
    | AudioInputContent
    | VideoInputContent
    | DocumentInputContent
    | None
):
    """Convert a user content item to AG-UI input content.

    When `use_multimodal` is True (ag-ui >= 0.1.15), media URLs are emitted as typed
    multimodal input content (e.g. `ImageInputContent`) instead of generic `BinaryInputContent`.
    """
    if isinstance(item, str):
        return TextInputContent(type='text', text=item)
    elif isinstance(item, TextContent):
        return TextInputContent(type='text', text=item.content)
    elif isinstance(item, (ImageUrl, VideoUrl, AudioUrl, DocumentUrl)):
        if use_multimodal:
            from ._multimodal import media_url_to_multimodal

            return media_url_to_multimodal(item)
        return BinaryInputContent(type='binary', url=item.url, mime_type=item.media_type or '')
    elif isinstance(item, BinaryContent):
        if use_multimodal:
            from ._multimodal import binary_to_multimodal

            return binary_to_multimodal(item)
        return BinaryInputContent(type='binary', data=item.base64, mime_type=item.media_type)
    elif isinstance(item, UploadedFile):
        # UploadedFile holds an opaque provider file_id (e.g. 'file-abc123'), not a URL or
        # binary data, so it can't be mapped to AG-UI input content. Skipped like CachePoint.
        return None
    elif isinstance(item, CachePoint):
        return None
    else:
        assert_never(item)


@dataclass
class AGUIAdapter(UIAdapter[RunAgentInput, Message, BaseEvent, AgentDepsT, OutputDataT]):
    """UI adapter for the Agent-User Interaction (AG-UI) protocol."""

    _: KW_ONLY
    ag_ui_version: str = DEFAULT_AG_UI_VERSION
    """AG-UI protocol version controlling behavior thresholds.

    Accepts any version string (e.g. `'0.1.13'`). Defaults to the version detected from
    the installed `ag-ui-protocol` package.

    Known thresholds:

    - `< 0.1.13`: emits `THINKING_*` events during streaming, drops `ThinkingPart`
      from `dump_messages` output.
    - `>= 0.1.13`: emits `REASONING_*` events with encrypted metadata during streaming, and
      includes `ThinkingPart` as `ReasoningMessage` in `dump_messages` output for full round-trip
      fidelity of thinking signatures and provider metadata.
    - `>= 0.1.15`: emits typed multimodal input content (`ImageInputContent`, `AudioInputContent`,
      `VideoInputContent`, `DocumentInputContent`) instead of generic `BinaryInputContent`.

    `load_messages` always accepts `ReasoningMessage` and multimodal content types regardless
    of this setting.
    """

    preserve_file_data: bool = False
    """Whether to preserve agent-generated files and uploaded files in AG-UI message conversion.

    When `True`, agent-generated files and uploaded files are stored as
    [activity messages](https://docs.ag-ui.com/concepts/activities) during `dump_messages`
    and restored during `load_messages`, enabling full round-trip fidelity.
    When `False` (default), they are silently dropped.

    If your AG-UI frontend uses activities, be aware that `pydantic_ai_*` activity types
    are reserved for internal round-trip use and should be ignored by frontend activity handlers.
    """

    @classmethod
    def build_run_input(cls, body: bytes) -> RunAgentInput:
        """Build an AG-UI run input object from the request body."""
        return RunAgentInput.model_validate_json(body)

    def build_event_stream(self) -> UIEventStream[RunAgentInput, BaseEvent, AgentDepsT, OutputDataT]:
        """Build an AG-UI event stream transformer."""
        return AGUIEventStream(self.run_input, accept=self.accept, ag_ui_version=self.ag_ui_version)

    @classmethod
    async def from_request(
        cls,
        request: Request,
        *,
        agent: AbstractAgent[AgentDepsT, OutputDataT],
        ag_ui_version: str = DEFAULT_AG_UI_VERSION,
        preserve_file_data: bool = False,
        manage_system_prompt: Literal['server', 'client'] = 'server',
        allowed_file_url_schemes: frozenset[str] = frozenset({'http', 'https'}),
        **kwargs: Any,
    ) -> AGUIAdapter[AgentDepsT, OutputDataT]:
        """Extends [`from_request`][pydantic_ai.ui.UIAdapter.from_request] with AG-UI-specific parameters."""
        return await super().from_request(
            request,
            agent=agent,
            ag_ui_version=ag_ui_version,
            preserve_file_data=preserve_file_data,
            manage_system_prompt=manage_system_prompt,
            allowed_file_url_schemes=allowed_file_url_schemes,
            **kwargs,
        )

    @cached_property
    def messages(self) -> list[ModelMessage]:
        """Pydantic AI messages from the AG-UI run input."""
        return self.load_messages(self.run_input.messages, preserve_file_data=self.preserve_file_data)

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

    @cached_property
    def conversation_id(self) -> str | None:
        """Conversation ID from the AG-UI `RunAgentInput.threadId`."""
        return self.run_input.thread_id

    @classmethod
    def load_messages(cls, messages: Sequence[Message], *, preserve_file_data: bool = False) -> list[ModelMessage]:  # noqa: C901
        """Transform AG-UI messages into Pydantic AI messages."""
        builder = MessagesBuilder()
        tool_calls: dict[str, str] = {}  # Tool call ID to tool name mapping.
        for msg in messages:
            match msg:
                case UserMessage(content=content):
                    if isinstance(content, str):
                        builder.add(UserPromptPart(content=content))
                    else:
                        user_prompt_content: list[UserContent] = []
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
                                case (
                                    ImageInputContent()
                                    | AudioInputContent()
                                    | VideoInputContent()
                                    | DocumentInputContent()
                                ):
                                    from ._multimodal import (
                                        multimodal_input_to_content,
                                    )

                                    user_prompt_content.append(multimodal_input_to_content(part))
                                case _:
                                    assert_never(part)

                        if user_prompt_content:
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
                        content: Any = tool_msg.content
                        if isinstance(content, str):
                            try:
                                content = json.loads(content)
                            except json.JSONDecodeError:
                                pass
                        builder.add(
                            BuiltinToolReturnPart(
                                tool_name=tool_name,
                                content=content,
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
                        if not isinstance(metadata, dict):
                            metadata = {}
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
                    if activity_msg.activity_type == FILE_ACTIVITY_TYPE and preserve_file_data:
                        activity_content = activity_msg.content
                        url = activity_content.get('url', '')
                        if not url:
                            raise ValueError(
                                f'ActivityMessage with activity_type={FILE_ACTIVITY_TYPE!r} must have a non-empty url.'
                            )
                        builder.add(
                            FilePart(
                                content=BinaryContent.from_data_uri(url),
                                id=activity_content.get('id'),
                                provider_name=activity_content.get('provider_name'),
                                provider_details=activity_content.get('provider_details'),
                            )
                        )
                    elif activity_msg.activity_type == UPLOADED_FILE_ACTIVITY_TYPE and preserve_file_data:
                        activity_content = activity_msg.content
                        file_id = activity_content.get('file_id', '')
                        provider_name = activity_content.get('provider_name', '')
                        if not file_id or not provider_name:
                            raise ValueError(
                                f'ActivityMessage with activity_type={UPLOADED_FILE_ACTIVITY_TYPE!r}'
                                ' must have non-empty file_id and provider_name.'
                            )
                        builder.add(
                            UserPromptPart(
                                content=[
                                    UploadedFile(
                                        file_id=file_id,
                                        provider_name=provider_name,
                                        vendor_metadata=activity_content.get('vendor_metadata'),
                                        media_type=activity_content.get('media_type'),
                                        identifier=activity_content.get('identifier'),
                                    )
                                ]
                            )
                        )

                case _:
                    if TYPE_CHECKING:
                        assert_never(msg)
                    warnings.warn(
                        f'AG-UI message type {type(msg).__name__} is not yet implemented; skipping.',
                        UserWarning,
                        stacklevel=2,
                    )

        return builder.messages

    @staticmethod
    def _dump_request_parts(
        msg: ModelRequest,
        *,
        ag_ui_version: str = DEFAULT_AG_UI_VERSION,
        preserve_file_data: bool = False,
    ) -> list[Message]:
        """Convert a `ModelRequest` into AG-UI messages."""
        use_multimodal = parse_ag_ui_version(ag_ui_version) >= MULTIMODAL_VERSION
        result: list[Message] = []
        system_content: list[str] = []
        user_content: list[
            TextInputContent
            | BinaryInputContent
            | ImageInputContent
            | AudioInputContent
            | VideoInputContent
            | DocumentInputContent
        ] = []

        for part in msg.parts:
            if isinstance(part, SystemPromptPart):
                system_content.append(part.content)
            elif isinstance(part, UserPromptPart):
                if isinstance(part.content, str):
                    user_content.append(TextInputContent(type='text', text=part.content))
                else:
                    for item in part.content:
                        if isinstance(item, UploadedFile) and preserve_file_data:
                            # AG-UI has no native uploaded-file message type. We repurpose
                            # ActivityMessage with a reserved `pydantic_ai_*` activity_type
                            # for round-trip fidelity. See UploadedFileActivityContent.
                            uploaded_content: dict[str, Any] = {
                                'file_id': item.file_id,
                                'provider_name': item.provider_name,
                                'media_type': item.media_type,
                                'identifier': item.identifier,
                            }
                            if item.vendor_metadata is not None:
                                uploaded_content['vendor_metadata'] = item.vendor_metadata
                            result.append(
                                ActivityMessage(
                                    id=_new_message_id(),
                                    activity_type=UPLOADED_FILE_ACTIVITY_TYPE,
                                    content=uploaded_content,
                                )
                            )
                        else:
                            converted = _user_content_to_input(item, use_multimodal=use_multimodal)
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
    def _dump_response_parts(  # noqa: C901
        msg: ModelResponse, *, ag_ui_version: str = DEFAULT_AG_UI_VERSION, preserve_file_data: bool = False
    ) -> list[Message]:
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
                if parse_ag_ui_version(ag_ui_version) >= REASONING_VERSION:
                    from ag_ui.core import ReasoningMessage

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
                if preserve_file_data:
                    # AG-UI has no native file message type. We repurpose ActivityMessage
                    # with a reserved `pydantic_ai_*` activity_type for round-trip fidelity.
                    # See FileActivityContent.
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
                            activity_type=FILE_ACTIVITY_TYPE,
                            content=file_content,
                        )
                    )
            elif isinstance(part, CompactionPart):  # pragma: no cover
                pass  # Compaction parts are not rendered in AG-UI
            else:
                assert_never(part)

        flush()
        return result

    @classmethod
    def dump_messages(
        cls,
        messages: Sequence[ModelMessage],
        *,
        ag_ui_version: str = DEFAULT_AG_UI_VERSION,
        preserve_file_data: bool = False,
    ) -> list[Message]:
        """Transform Pydantic AI messages into AG-UI messages.

        Note: The round-trip `dump_messages` -> `load_messages` is not fully lossless:

        - `TextPart.id`, `.provider_name`, `.provider_details` are lost.
        - `ToolCallPart.id`, `.provider_name`, `.provider_details` are lost.
        - `BuiltinToolCallPart.id`, `.provider_details` are lost (only `.provider_name` survives
          via the prefixed tool call ID).
        - `BuiltinToolReturnPart.provider_details` is lost.
        - `RetryPromptPart` becomes `ToolReturnPart` (or `UserPromptPart`) on reload.
        - `CachePoint` and `UploadedFile` content items are dropped (unless `preserve_file_data=True`).
        - `ThinkingPart` is dropped when `ag_ui_version='0.1.10'`.
        - `FilePart` is silently dropped unless `preserve_file_data=True`.
        - `UploadedFile` in a multi-item `UserPromptPart` is split into a separate activity message
          when `preserve_file_data=True`, which reloads as a separate `UserPromptPart`.
        - Part ordering within a `ModelResponse` may change when text follows tool calls.

        Args:
            messages: A sequence of ModelMessage objects to convert.
            ag_ui_version: AG-UI protocol version controlling `ThinkingPart` emission.
            preserve_file_data: Whether to include `FilePart` and `UploadedFile` as `ActivityMessage`.

        Returns:
            A list of AG-UI Message objects.
        """
        result: list[Message] = []

        for msg in messages:
            if isinstance(msg, ModelRequest):
                request_messages = cls._dump_request_parts(
                    msg, ag_ui_version=ag_ui_version, preserve_file_data=preserve_file_data
                )
                result.extend(request_messages)
            elif isinstance(msg, ModelResponse):
                result.extend(
                    cls._dump_response_parts(msg, ag_ui_version=ag_ui_version, preserve_file_data=preserve_file_data)
                )
            else:
                assert_never(msg)

        return result
