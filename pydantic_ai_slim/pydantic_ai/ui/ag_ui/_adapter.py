"""AG-UI adapter for handling requests."""

from __future__ import annotations

import json
import uuid
import warnings
from base64 import b64decode
from collections.abc import Sequence
from dataclasses import KW_ONLY, dataclass
from functools import cached_property
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
)

from typing_extensions import assert_never

from ... import ExternalToolset, ToolDefinition
from ..._utils import is_str_dict
from ...messages import (
    AudioUrl,
    BinaryContent,
    CachePoint,
    CompactionPart,
    DocumentUrl,
    FilePart,
    ForceDownloadMode,
    ImageUrl,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    NativeToolCallPart,
    NativeToolReturnPart,
    RetryPromptPart,
    SystemPromptPart,
    TextContent,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolPartKind,
    ToolReturnPart,
    UploadedFile,
    UserContent,
    UserPromptPart,
    VideoUrl,
    narrow_message_parts,
)
from ...output import OutputDataT
from ...tools import (
    AgentDepsT,
    DeferredToolApprovalResult,
    DeferredToolResults,
)
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
    from ._interrupt import (
        HAS_INTERRUPTS,
        ResumeEntry,
        interrupt_id_to_tool_call_id,
        resume_entry_to_approval,
    )
    from ._utils import (
        BUILTIN_TOOL_CALL_ID_PREFIX,
        DEFAULT_AG_UI_VERSION,
        ENCRYPTED_VALUE_VERSION,
        FILE_ACTIVITY_TYPE,
        MULTIMODAL_VERSION,
        REASONING_VERSION,
        UPLOADED_FILE_ACTIVITY_TYPE,
        dump_tool_return_content,
        parse_ag_ui_version,
        parse_builtin_tool_call_id,
        parse_encrypted_tool_kind,
        rehydrate_tool_return_content,
        thinking_encrypted_metadata,
        tool_kind_encrypted_value_kwargs,
        warn_tool_kind_not_persisted,
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
    except ImportError:

        class ReasoningMessage:
            """Stub for ag-ui-protocol < 0.1.11 — no instances exist, so pattern matching is a no-op."""

    try:
        from ag_ui.core import AudioInputContent, DocumentInputContent, ImageInputContent, VideoInputContent
    except ImportError:

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
    """Whether to round-trip `FilePart` and `UploadedFile` through reserved `pydantic_ai_*`
    [activity messages](https://docs.ag-ui.com/concepts/messages).

    Defaults to `False`. AG-UI has no native representation for agent-generated files
    ([`FilePart`][pydantic_ai.messages.FilePart]) or uploaded-file references
    ([`UploadedFile`][pydantic_ai.messages.UploadedFile]), so when this is `True` they are
    serialized as sidecar activity messages on `dump_messages` and reconstructed on
    `load_messages`. A frontend only completes the round-trip if it echoes these activity
    messages back on the next request.

    This is a representation setting, not a security one: honoring a reconstructed inbound
    `UploadedFile` still requires
    [`allow_uploaded_files`][pydantic_ai.ui.UIAdapter.allow_uploaded_files], which the shared
    `sanitize_messages` step enforces regardless of this flag. Multimodal tool-return files are
    unaffected — they ride inline in `ToolMessage.content`.
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
        allowed_file_url_force_download: frozenset[ForceDownloadMode] = frozenset(),
        allow_uploaded_files: bool = False,
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
            allowed_file_url_force_download=allowed_file_url_force_download,
            allow_uploaded_files=allow_uploaded_files,
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
        if is_str_dict(state) and state:
            return state

        return None

    @cached_property
    def conversation_id(self) -> str | None:
        """Conversation ID from the AG-UI `RunAgentInput.threadId`."""
        return self.run_input.thread_id

    @cached_property
    def deferred_tool_results(self) -> DeferredToolResults | None:
        """Translate AG-UI `RunAgentInput.resume[]` into Pydantic AI `DeferredToolResults`.

        See [docs.ag-ui.com/concepts/interrupts](https://docs.ag-ui.com/concepts/interrupts).

        Each `ResumeEntry` is mapped to an approval keyed by the original `tool_call_id`.
        The mapping is **deny-by-default**: approval requires an explicit
        `payload.approved == True`. Any other shape is treated as a denial so a malformed
        or hostile client cannot accidentally execute a tool that requires human approval.

        - `status == 'cancelled'` → `ToolDenied('Cancelled by user.')`
        - `payload.approved is True` with `payload.editedArgs` → `ToolApproved(override_args=...)`
        - `payload.approved is True` without edits → `ToolApproved()`
        - Anything else (`False`, missing, `null`, non-bool, non-dict payload) →
          `ToolDenied(payload.get('reason'))` if `reason` is a non-empty string, else
          `ToolDenied()` (which carries the default `"The tool call was denied."` message).

        Returns `None` when `resume` is missing or empty, or when the installed
        ag-ui-protocol predates the interrupt lifecycle.
        """
        if not HAS_INTERRUPTS:
            return None
        resume: list[ResumeEntry] | None = getattr(self.run_input, 'resume', None)
        if not resume:
            return None
        approvals: dict[str, DeferredToolApprovalResult | bool] = {
            interrupt_id_to_tool_call_id(entry.interrupt_id): resume_entry_to_approval(entry) for entry in resume
        }
        return DeferredToolResults(approvals=approvals)

    @classmethod
    def load_messages(cls, messages: Sequence[Message], *, preserve_file_data: bool = False) -> list[ModelMessage]:  # noqa: C901
        """Transform AG-UI messages into Pydantic AI messages."""
        builder = MessagesBuilder()
        tool_calls: dict[str, str] = {}  # Tool call ID to tool name mapping.
        tool_kinds: dict[str, ToolPartKind] = {}  # Tool call ID to `tool_kind` claim mapping.
        # `ToolCall`/`ToolMessage.encrypted_value` only exists on the installed model from 0.1.11
        # onward; older versions drop the client's claim, so the field is only read when present.
        use_encrypted_value = parse_ag_ui_version(DEFAULT_AG_UI_VERSION) >= ENCRYPTED_VALUE_VERSION
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

                            # The claim is client-supplied, so it's set on the base part and promoted
                            # best-effort by the final `narrow_message_parts` pass (which strips it if
                            # it doesn't validate against the typed subclass).
                            tool_kind = (
                                parse_encrypted_tool_kind(tool_call.encrypted_value) if use_encrypted_value else None
                            )
                            if tool_kind is not None:
                                tool_kinds[tool_call_id] = tool_kind

                            builtin_id = parse_builtin_tool_call_id(tool_call_id)
                            if builtin_id is not None:
                                provider_name, original_id = builtin_id
                                builder.add(
                                    NativeToolCallPart(
                                        tool_name=tool_name,
                                        args=tool_call.function.arguments,
                                        tool_call_id=original_id,
                                        provider_name=provider_name,
                                        tool_kind=tool_kind,
                                    )
                                )
                            else:
                                builder.add(
                                    ToolCallPart(
                                        tool_name=tool_name,
                                        tool_call_id=tool_call_id,
                                        args=tool_call.function.arguments,
                                        tool_kind=tool_kind,
                                    )
                                )
                case ToolMessage() as tool_msg:
                    tool_call_id = tool_msg.tool_call_id
                    tool_name = tool_calls.get(tool_call_id)
                    if tool_name is None:  # pragma: no cover
                        raise ValueError(f'Tool call with ID {tool_call_id} not found in the history.')

                    # Rehydrate here (not in a later `ModelMessagesTypeAdapter` pass) so structured and
                    # multimodal content comes back as real types; see `rehydrate_tool_return_content`.
                    content = rehydrate_tool_return_content(tool_msg.content)

                    # Fall back to the paired call's claim: `ToolCallResultEvent` has no metadata
                    # slot, so client-built ToolMessages usually carry no `encrypted_value`. Error
                    # results stay untyped — typed return parts imply success to their readers.
                    tool_kind = None
                    if tool_msg.error is None:
                        encrypted_tool_kind = (
                            parse_encrypted_tool_kind(tool_msg.encrypted_value) if use_encrypted_value else None
                        )
                        tool_kind = encrypted_tool_kind or tool_kinds.get(tool_call_id)

                    builtin_id = parse_builtin_tool_call_id(tool_call_id)
                    if builtin_id is not None:
                        provider_name, original_id = builtin_id
                        builder.add(
                            NativeToolReturnPart(
                                tool_name=tool_name,
                                content=content,
                                tool_call_id=original_id,
                                provider_name=provider_name,
                                tool_kind=tool_kind,
                            )
                        )
                    else:
                        # The final `narrow_message_parts` pass parses the rehydrated content into a typed
                        # return subclass when the `tool_kind` claim validates, and leaves the base
                        # `ToolReturnPart` (dropping the claim) when it doesn't.
                        builder.add(
                            ToolReturnPart(
                                tool_name=tool_name,
                                content=content,
                                tool_call_id=tool_call_id,
                                tool_kind=tool_kind,
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
                        binary_content = BinaryContent.from_data_uri(url)
                        vendor_metadata = activity_content.get('vendor_metadata')
                        # `vendor_metadata` is client-supplied and typed `Any`; assignment on the
                        # (non-`validate_assignment`) `BinaryContent` dataclass bypasses validation,
                        # so ignore anything that isn't a dict rather than let it reach the provider.
                        if is_str_dict(vendor_metadata):
                            binary_content.vendor_metadata = vendor_metadata
                        builder.add(
                            FilePart(
                                content=binary_content,
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

        # Parts above are built as base `ToolCallPart`/`ToolReturnPart`/`NativeTool*Part` carrying a
        # `tool_kind` claim; promote them to their typed subclasses in one best-effort pass.
        return narrow_message_parts(builder.messages)

    @staticmethod
    def _dump_request_parts(  # noqa: C901
        msg: ModelRequest,
        *,
        ag_ui_version: str = DEFAULT_AG_UI_VERSION,
        preserve_file_data: bool = False,
    ) -> list[Message]:
        """Convert a `ModelRequest` into AG-UI messages.

        Uses a flush pattern to preserve part ordering: buffered user content is flushed before
        each tool message, so a `ToolReturnPart` that precedes a `UserPromptPart` in the original
        request keeps its position instead of being reordered after the user prompt.
        """
        use_multimodal = parse_ag_ui_version(ag_ui_version) >= MULTIMODAL_VERSION
        # `ToolMessage.encrypted_value` (the `tool_kind` carrier here) landed in 0.1.11 — see
        # `tool_kind_encrypted_value`.
        use_encrypted_value = parse_ag_ui_version(ag_ui_version) >= ENCRYPTED_VALUE_VERSION
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

        def flush_user_content() -> None:
            nonlocal user_content
            if not user_content:
                return
            # Simplify to plain string if only a single text item.
            if len(user_content) == 1 and isinstance(user_content[0], TextInputContent):
                result.append(UserMessage(id=_new_message_id(), content=user_content[0].text))
            else:
                result.append(UserMessage(id=_new_message_id(), content=user_content))
            user_content = []

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
                            flush_user_content()
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
                flush_user_content()
                # Tool-return files ride inline in `ToolMessage.content` (see `dump_tool_return_content`).
                result.append(
                    ToolMessage(
                        id=_new_message_id(),
                        content=dump_tool_return_content(part.content),
                        tool_call_id=part.tool_call_id,
                        **tool_kind_encrypted_value_kwargs(part.tool_kind, supported=use_encrypted_value),
                    )
                )
            elif isinstance(part, RetryPromptPart):
                if part.tool_name:
                    flush_user_content()
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
        flush_user_content()
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

        version = parse_ag_ui_version(ag_ui_version)
        # `ReasoningMessage` is a REASONING_* type (0.1.13+); the `tool_kind` carrier
        # `ToolCall`/`ToolMessage.encrypted_value` landed earlier in 0.1.11 — see
        # `tool_kind_encrypted_value`.
        use_reasoning = version >= REASONING_VERSION
        use_encrypted_value = version >= ENCRYPTED_VALUE_VERSION

        builtin_returns = {part.tool_call_id: part for part in msg.parts if isinstance(part, NativeToolReturnPart)}

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
                if use_reasoning:
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
                        **tool_kind_encrypted_value_kwargs(part.tool_kind, supported=use_encrypted_value),
                    )
                )
            elif isinstance(part, NativeToolCallPart):
                prefixed_id = '|'.join([BUILTIN_TOOL_CALL_ID_PREFIX, part.provider_name or '', part.tool_call_id])
                tool_calls_list.append(
                    ToolCall(
                        id=prefixed_id,
                        function=FunctionCall(name=part.tool_name, arguments=part.args_as_json_str()),
                        **tool_kind_encrypted_value_kwargs(part.tool_kind, supported=use_encrypted_value),
                    )
                )
                if builtin_return := builtin_returns.get(part.tool_call_id):
                    # Built-in tool-return files also ride inline in `ToolMessage.content` (see above).
                    tool_messages.append(
                        ToolMessage(
                            id=_new_message_id(),
                            content=dump_tool_return_content(builtin_return.content),
                            tool_call_id=prefixed_id,
                            **tool_kind_encrypted_value_kwargs(builtin_return.tool_kind, supported=use_encrypted_value),
                        )
                    )
            elif isinstance(part, NativeToolReturnPart):
                # Emitted when matching NativeToolCallPart is processed above.
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
                    if part.content.vendor_metadata is not None:
                        file_content['vendor_metadata'] = part.content.vendor_metadata
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
        - `NativeToolCallPart.id`, `.provider_details` are lost (only `.provider_name` survives
          via the prefixed tool call ID).
        - `NativeToolReturnPart.provider_details` is lost.
        - `tool_kind` is lost when `ag_ui_version < '0.1.11'` (before its `encrypted_value` carrier
          existed), so typed tool parts reload as their base classes.
        - `tool_kind` is not restored on error/denied tool returns (a typed return implies
          success to its readers), so those reload as plain `ToolReturnPart`.
        - `RetryPromptPart` becomes `ToolReturnPart` (or `UserPromptPart`) on reload.
        - `CachePoint` and `UploadedFile` content items are dropped (unless `preserve_file_data=True`).
        - `FileUrl.force_download` is dropped when `ag_ui_version < '0.1.15'` (before typed
          multimodal content gained a metadata carrier).
        - `ThinkingPart` is dropped when `ag_ui_version='0.1.10'`.
        - `FilePart` is silently dropped unless `preserve_file_data=True`.
        - `UploadedFile` in a multi-item `UserPromptPart` is split into a separate activity message
          when `preserve_file_data=True`, which reloads as a separate `UserPromptPart`.
        - `MultiModalContent` items in `ToolReturnPart`/`NativeToolReturnPart.content` always round-trip,
          regardless of `preserve_file_data`: the full content (files as base64/URL dicts) is serialized
          inline into the JSON `ToolMessage.content` and rehydrated on reload via the `ToolReturnContent`
          discriminator. The same serialization is used for both history (`dump_messages`) and the live
          event stream (`ToolCallResultEvent.content`), so files survive either round-trip.
        - Part ordering within a `ModelResponse` may change when text follows tool calls.

        Args:
            messages: A sequence of ModelMessage objects to convert.
            ag_ui_version: AG-UI protocol version controlling `ThinkingPart` emission.
            preserve_file_data: Whether to include `FilePart` and `UploadedFile` items as `ActivityMessage`s.
                (Multimodal tool-return files always ride inline in `ToolMessage.content` and are unaffected.)

        Returns:
            A list of AG-UI Message objects.
        """
        result: list[Message] = []

        if parse_ag_ui_version(ag_ui_version) < ENCRYPTED_VALUE_VERSION and any(
            isinstance(part, (ToolCallPart, ToolReturnPart, NativeToolCallPart, NativeToolReturnPart))
            and part.tool_kind is not None
            for msg in messages
            for part in msg.parts
        ):
            warn_tool_kind_not_persisted(ag_ui_version)

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
