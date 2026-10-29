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
    assert_never,
    cast,
)

from pydantic import TypeAdapter, ValidationError

from ... import ExternalToolset, ToolDefinition
from ..._mcp_apps import get_mcp_app_tool_result
from ..._utils import is_str_dict
from ...messages import (
    ERROR_OUTCOMES,
    AudioUrl,
    BinaryContent,
    CachePoint,
    CompactionPart,
    DocumentUrl,
    FilePart,
    ForceDownloadMode,
    ImageUrl,
    InstructionBaselinePart,
    InstructionDeltaPart,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ModelSettingsDeltaPart,
    NativeToolCallPart,
    NativeToolReturnPart,
    RetryFeedbackPart,
    RetryPromptPart,  # pyright: ignore[reportDeprecated]  # TODO(v3): remove RetryPromptPart
    SpeechPart,
    StructuredContent,
    SystemPromptPart,
    TextContent,
    TextPart,
    ThinkingPart,
    ToolAvailabilityDeltaPart,
    ToolCallPart,
    ToolPartKind,
    ToolReturnPart,
    UploadedFile,
    UserContent,
    UserPromptPart,
    VideoUrl,
    _retry_feedback_speaks_for_the_harness,  # pyright: ignore[reportPrivateUsage]
    _translate_legacy_retry_part,  # pyright: ignore[reportPrivateUsage]
    narrow_message_parts,
    tool_return_content_ta,
)
from ...output import OutputDataT
from ...tools import (
    AgentDepsT,
    DeferredToolApprovalResult,
    DeferredToolResults,
)
from ...toolsets import AbstractToolset
from .._adapter import (
    DEFAULT_ALLOWED_CONTENT_TYPES,
    compaction_part_from_payload,
    compaction_payload,
    model_settings_delta_from_payload,
    model_settings_delta_payload,
    pending_tool_calls,
    structured_content_from_payload,
    structured_content_payload,
    tool_availability_delta_from_payload,
    warn_dropped_tool_results,
)
from .._utils import MCP_APPS_UI_TYPE, get_ui_message_id, mcp_app_ui_payload, set_ui_message_id

try:
    from ag_ui.core import (
        ActivityMessage,
        AssistantMessage,
        BaseEvent,
        DeveloperMessage,
        FunctionCall,
        InputContent,
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
    from ._forward_compat import HAS_BINARY_INPUT_PART, adapt_unsupported_items
    from ._interrupt import (
        HAS_INTERRUPTS,
        ResumeEntry,
        interrupt_id_to_tool_call_id,
        resume_entry_to_approval,
    )
    from ._utils import (
        BUILTIN_TOOL_CALL_ID_PREFIX,
        COMPACTION_ACTIVITY_TYPE,
        DEFAULT_AG_UI_VERSION,
        ENCRYPTED_VALUE_VERSION,
        ENQUEUED_MESSAGES_ACTIVITY_TYPE,
        FILE_ACTIVITY_TYPE,
        MODEL_SETTINGS_DELTA_ACTIVITY_TYPE,
        MULTIMODAL_VERSION,
        REASONING_VERSION,
        TOOL_AVAILABILITY_DELTA_ACTIVITY_TYPE,
        UPLOADED_FILE_ACTIVITY_TYPE,
        dump_tool_return_content,
        parse_ag_ui_version,
        parse_builtin_tool_call_id,
        parse_encrypted_outcome,
        parse_encrypted_retry_feedback,
        parse_encrypted_structured_content,
        parse_encrypted_tool_kind,
        parse_encrypted_tool_retry,
        rehydrate_tool_return_content,
        retry_feedback_encrypted_value_kwargs,
        structured_content_encrypted_value_kwargs,
        thinking_encrypted_metadata,
        tool_kind_encrypted_value_kwargs,
        warn_encrypted_value_not_persisted,
    )
except ImportError as e:
    raise ImportError(
        'Please install the `ag-ui-protocol` package to use AG-UI integration, '
        'you can use the `ag-ui` optional group — `pip install "pydantic-ai-slim[ag-ui]"`'
    ) from e

if TYPE_CHECKING:
    from ag_ui.core import (
        AudioInputContent,
        BinaryInputContent,
        DocumentInputContent,
        ImageInputContent,
        ReasoningMessage,
        VideoInputContent,
    )
    from starlette.requests import Request

    from ...agent import AbstractAgent
else:
    try:
        from ag_ui.core import BinaryInputContent
    except ImportError:  # pragma: lax no cover

        class BinaryInputContent:
            """Stub for an ag-ui-protocol that no longer ships the `binary` part 1.0 retired."""

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


def _input_content_to_user_content(
    content: Sequence[InputContent], structured: Mapping[str, Any] | None = None
) -> list[UserContent]:
    """Convert a message's typed content parts to Pydantic AI content.

    `structured` holds `StructuredContent` payloads keyed by the position of their text part, as
    `parse_encrypted_structured_content` reads them. A part the installed `ag-ui-protocol` accepts but
    Pydantic AI cannot use, such as a `file` source with no known provider, is dropped with a warning.
    """
    structured = structured or {}
    converted: list[UserContent] = []
    for index, part in enumerate(content):
        match part:
            case TextInputContent(text=text):
                payload = structured.get(str(index))
                converted.append(text if payload is None else structured_content_from_payload(payload, text))
            case _ if isinstance(part, BinaryInputContent):
                # A guard, not a class pattern: AG-UI 1.0 retired the part from the content union,
                # but an install below 1.0 still sends it.
                if part.url:
                    try:
                        converted.append(BinaryContent.from_data_uri(part.url))
                    except ValueError:
                        media_type_constructors = {'image': ImageUrl, 'video': VideoUrl, 'audio': AudioUrl}
                        constructor = media_type_constructors.get(part.mime_type.split('/', 1)[0], DocumentUrl)
                        converted.append(constructor(url=part.url, media_type=part.mime_type))
                elif part.data:
                    converted.append(BinaryContent(data=b64decode(part.data), media_type=part.mime_type))
                else:
                    raise ValueError('BinaryInputContent must have either a `url` or `data` field.')
            case ImageInputContent() | AudioInputContent() | VideoInputContent() | DocumentInputContent():
                from ._multimodal import multimodal_input_to_content

                if (media := multimodal_input_to_content(part)) is not None:
                    converted.append(media)
            case _:
                assert_never(part)
    return converted


_messages_ta: TypeAdapter[list[Message]] = TypeAdapter(list[Message])


def _enqueued_messages(content: dict[str, Any]) -> list[Message]:
    """The AG-UI messages in a `pydantic_ai_enqueued_messages` activity, or none when the client sent something malformed.

    Enqueued-message activities nested inside are dropped: the content is the client's to send, and
    honoring them would let it recurse `load_messages` as deeply as it liked.
    """
    try:
        messages = _messages_ta.validate_python(content.get('messages'))
    except ValidationError:
        return []
    return [
        message
        for message in messages
        if not (isinstance(message, ActivityMessage) and message.activity_type == ENQUEUED_MESSAGES_ACTIVITY_TYPE)
    ]


def _user_content_to_input(
    item: str | TextContent | ImageUrl | VideoUrl | AudioUrl | DocumentUrl | BinaryContent | UploadedFile | CachePoint,
    *,
    use_multimodal: bool = False,
) -> InputContent | None:
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
        try:
            mime_type = item.media_type
        except ValueError:
            mime_type = ''  # reloads as no media type
        # Cast for 1.0, where the retired class is outside the `InputContent` union; only reached below 1.0.
        return cast(InputContent, BinaryInputContent(type='binary', url=item.url, mime_type=mime_type))
    elif isinstance(item, BinaryContent):
        if use_multimodal:
            from ._multimodal import binary_to_multimodal

            return binary_to_multimodal(item)
        return cast(InputContent, BinaryInputContent(type='binary', data=item.base64, mime_type=item.media_type))
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

    Accepts any version string (e.g. `'0.1.11'`). Defaults to the version detected from
    the installed `ag-ui-protocol` package.

    Known thresholds:

    - `< 0.1.11`: emits `THINKING_*` events during streaming, drops `ThinkingPart`
      from `dump_messages` output.
    - `>= 0.1.11`: emits `REASONING_*` events with encrypted metadata during streaming, and
      includes `ThinkingPart` as `ReasoningMessage` in `dump_messages` output for full round-trip
      fidelity of thinking signatures and provider metadata.
    - `>= 0.1.15`: emits `ImageInputContent`, `AudioInputContent`, `VideoInputContent` and
      `DocumentInputContent` instead of `BinaryInputContent`. With `ag-ui-protocol >= 1.0` installed this
      applies at any setting, since 1.0 removed `BinaryInputContent`.
    - `>= 0.1.19`: emits `ACTIVITY_SNAPSHOT` events for tool availability, compaction, enqueued
      messages and [MCP Apps](../mcp/client.md#mcp-apps) during streaming.
    - `>= 1.0`: sends `protocolVersion` on `RUN_STARTED`, token usage on `RUN_FINISHED` with
      `include_usage`, and the `cancelled` outcome and `pendingToolCallIds` to clients that declare 1.0
      (see the [AG-UI docs](../ui/ag-ui.md#protocol-version-compatibility)).

    `load_messages` accepts reasoning and multimodal content regardless of this setting. `build_run_input`
    skips inbound content your installed `ag-ui-protocol` doesn't support, with a warning, and on 1.0
    converts the removed `binary` part to typed content.
    """

    preserve_file_data: bool = False
    """Whether to round-trip `FilePart` and `UploadedFile` through reserved `pydantic_ai_*`
    [activity messages](https://docs.ag-ui.com/concepts/messages).

    Defaults to `False`. AG-UI has no native representation for agent-generated files
    ([`FilePart`][pydantic_ai.messages.FilePart]), and `dump_messages` doesn't write
    [`UploadedFile`][pydantic_ai.messages.UploadedFile] references as AG-UI file parts,
    so when this is `True` they are serialized as sidecar activity messages on `dump_messages` and
    reconstructed on `load_messages`. A frontend only completes the round-trip if it sends these activity
    messages back on the next request, which `@ag-ui/client` doesn't do; see
    [Records carried as activity messages](../ui/ag-ui.md#activity-records).

    This is a representation setting, not a security one: honoring a reconstructed inbound
    `UploadedFile` still requires
    [`allow_uploaded_files`][pydantic_ai.ui.UIAdapter.allow_uploaded_files], which the shared
    `sanitize_messages` step enforces regardless of this flag. Multimodal tool-return files are
    unaffected — they ride inline in `ToolMessage.content`.
    """

    include_usage: bool = False
    """Whether `RUN_FINISHED` reports the run's token usage per provider and model (`ag-ui-protocol >= 1.0`).

    Off by default because it tells the client which provider and model your server uses. Has no
    effect with `ag-ui-protocol` below 1.0.
    """

    @classmethod
    def build_run_input(cls, body: bytes) -> RunAgentInput:
        """Build an AG-UI run input object from the request body.

        A message `role`, input content `type` or media `source` type introduced by a protocol version
        newer than the installed `ag-ui-protocol` is skipped with a warning rather than failing the whole request,
        per the backwards-compatibility policy in `pydantic_ai/ui/AGENTS.md`. Only items the
        installed models cannot dispatch at all are skipped: a body that is invalid for any other
        reason still raises, so a client bug isn't converted into silent misbehavior.

        With `ag-ui-protocol >= 1.0`, a `binary` input part (removed in 1.0) is converted to image, audio,
        video or document content; one with no MIME type or payload still raises.
        """
        try:
            return RunAgentInput.model_validate_json(body)
        except ValidationError:
            if (adapted := adapt_unsupported_items(body)) is None:
                raise
            payload, skipped = adapted

        # Validated outside the `except` block so a body that is *also* malformed reports the
        # remaining errors on their own rather than chained behind the unknown-tag failure.
        run_input = RunAgentInput.model_validate(payload)
        if skipped:
            warnings.warn(
                f'AG-UI content the installed ag-ui-protocol {DEFAULT_AG_UI_VERSION} does not support '
                f'({", ".join(sorted(skipped))}) was skipped; upgrade `ag-ui-protocol` to accept it.',
                UserWarning,
                stacklevel=2,
            )
        return run_input

    def build_event_stream(self) -> UIEventStream[RunAgentInput, BaseEvent, AgentDepsT, OutputDataT]:
        """Build an AG-UI event stream transformer."""
        return AGUIEventStream(
            self.run_input,
            accept=self.accept,
            ag_ui_version=self.ag_ui_version,
            include_usage=self.include_usage,
            manage_system_prompt=self.manage_system_prompt,
            preserve_file_data=self.preserve_file_data,
        )

    @classmethod
    async def from_request(
        cls,
        request: Request,
        *,
        agent: AbstractAgent[AgentDepsT, OutputDataT],
        ag_ui_version: str = DEFAULT_AG_UI_VERSION,
        preserve_file_data: bool = False,
        manage_system_prompt: Literal['server', 'client'] = 'server',
        manage_message_history: Literal['client', 'server'] = 'client',
        allowed_file_url_schemes: frozenset[str] = frozenset({'http', 'https'}),
        allowed_file_url_force_download: frozenset[ForceDownloadMode] = frozenset(),
        allow_uploaded_files: bool = False,
        allowed_content_types: frozenset[str] | None = DEFAULT_ALLOWED_CONTENT_TYPES,
        include_usage: bool = False,
        **kwargs: Any,
    ) -> AGUIAdapter[AgentDepsT, OutputDataT]:
        """Extends [`from_request`][pydantic_ai.ui.UIAdapter.from_request] with AG-UI-specific parameters."""
        return await super().from_request(
            request,
            agent=agent,
            ag_ui_version=ag_ui_version,
            preserve_file_data=preserve_file_data,
            manage_system_prompt=manage_system_prompt,
            manage_message_history=manage_message_history,
            allowed_file_url_schemes=allowed_file_url_schemes,
            allowed_file_url_force_download=allowed_file_url_force_download,
            allow_uploaded_files=allow_uploaded_files,
            allowed_content_types=allowed_content_types,
            include_usage=include_usage,
            **kwargs,
        )

    @cached_property
    def messages(self) -> list[ModelMessage]:
        """Pydantic AI messages from the AG-UI run input."""
        return self.load_messages(self.run_input.messages, preserve_file_data=self.preserve_file_data)

    def _server_held_client_messages(self, message_history: Sequence[ModelMessage]) -> list[ModelMessage]:
        """Load a tool result the client sends without its call against the call `message_history` is waiting on.

        A `ToolMessage` carries no tool name, so it only loads after the `AssistantMessage` that made the
        call, which a client sending only its new messages leaves out. The stored history's pending call
        stands in for it, so nothing the client sends decides which call it answers. A result for a call
        that isn't pending is dropped with a warning, as `manage_message_history='server'` drops it anyway.
        """
        messages = self.run_input.messages
        called = {call.id for m in messages if isinstance(m, AssistantMessage) for call in m.tool_calls or ()}
        orphaned = {m.tool_call_id for m in messages if isinstance(m, ToolMessage) and m.tool_call_id not in called}
        if not orphaned:
            return self.messages
        pending = [call for call in pending_tool_calls(message_history) if call.tool_call_id in orphaned]
        if dropped := len(orphaned) - len(pending):
            warn_dropped_tool_results(dropped, stacklevel=5)
        answered = {call.tool_call_id for call in pending}
        dropped_ids = orphaned - answered
        kept = [m for m in messages if not isinstance(m, ToolMessage) or m.tool_call_id not in dropped_ids]
        if pending:
            calls = [
                ToolCall(
                    id=call.tool_call_id, function=FunctionCall(name=call.tool_name, arguments=call.args_as_json_str())
                )
                for call in pending
            ]
            kept = [AssistantMessage(id='pydantic_ai-pending-tool-calls', tool_calls=calls), *kept]
        return self.load_messages(kept, preserve_file_data=self.preserve_file_data)

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
        The payload is validated against the same Pydantic model whose JSON schema is
        advertised on `Interrupt.response_schema`, and the mapping is **deny-by-default**:
        approval requires a payload that validates with `approved=True`, so a malformed
        client response cannot accidentally execute a tool that requires human approval.
        (Approval guards against the *model* acting without human sign-off; it is not an
        authorization boundary against the client — see
        [Trust model for client-submitted messages](../../ui/overview.md#trust-model-for-client-submitted-messages).)

        The full mapping, including which payloads raise instead of denying, is documented under
        [Tool approval (interrupts)](../../ui/ag-ui.md#tool-approval-interrupts).

        Raises `UserError` for a payload that fails validation, and for a `ResumeEntry.interrupt_id`
        that carries no recognizable prefix. Both are resolved inside the run stream (see
        [`run_stream_native`][pydantic_ai.ui.UIAdapter.run_stream_native]), so they reach the client
        as a `RunErrorEvent` rather than escaping the streaming response.

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
            checkpoint = builder.checkpoint()
            match msg:
                case UserMessage() as user_msg:
                    # A user message only reloads as harness retry feedback when it carries the
                    # `encrypted_value` marker this adapter dumped it with; one a person wrote has no
                    # such claim and stays a `UserPromptPart`. See `retry_feedback_from_payload`.
                    user_feedback = (
                        parse_encrypted_retry_feedback(user_msg.encrypted_value) if use_encrypted_value else None
                    )
                    # Otherwise the same carrier may hold the values of its `StructuredContent` text items.
                    structured = (
                        parse_encrypted_structured_content(user_msg.encrypted_value) if use_encrypted_value else {}
                    )
                    content = user_msg.content
                    if user_feedback is not None:
                        builder.add(user_feedback)
                    elif isinstance(content, str):
                        loaded = (
                            structured_content_from_payload(structured['0'], content) if '0' in structured else content
                        )
                        builder.add(UserPromptPart(content=loaded if isinstance(loaded, str) else [loaded]))
                    elif user_prompt_content := _input_content_to_user_content(content, structured):
                        content_to_add = (
                            user_prompt_content[0]
                            if len(user_prompt_content) == 1 and isinstance(user_prompt_content[0], str)
                            else user_prompt_content
                        )
                        builder.add(UserPromptPart(content=content_to_add))

                case SystemMessage() | DeveloperMessage() as system_msg:
                    # A system message only reloads as harness retry feedback when it carries the
                    # `encrypted_value` marker this adapter dumped it with; one the client wrote has
                    # no such claim and stays a plain `SystemPromptPart`, so it can never gain
                    # harness provenance. See `retry_feedback_from_payload`.
                    feedback = (
                        parse_encrypted_retry_feedback(system_msg.encrypted_value) if use_encrypted_value else None
                    )
                    builder.add(feedback if feedback is not None else SystemPromptPart(content=system_msg.content))

                case AssistantMessage(content=content, tool_calls=tool_calls_list):
                    # An empty answer only marks a response boundary, so content that follows it in the same
                    # response replaces it rather than joining it.
                    last = builder.messages[-1] if builder.messages else None
                    if (
                        (content or tool_calls_list)
                        and isinstance(last, ModelResponse)
                        and last.parts == [TextPart(content='')]
                    ):
                        builder.messages.pop()
                    # An empty answer still keeps its place as a response, so the request after it doesn't
                    # merge into the one before it.
                    if content or (content == '' and not tool_calls_list):
                        builder.add(TextPart(content=content))
                    if tool_calls_list:
                        for tool_call in tool_calls_list:
                            tool_call_id = tool_call.id
                            tool_name = tool_call.function.name
                            tool_calls[tool_call_id] = tool_name

                            # The claim is client-supplied, so it's set on the base part and promoted
                            # best-effort by the final `narrow_message_parts` pass. A claim whose data
                            # doesn't validate leaves a base part; only core's kinds are then stripped.
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

                    if isinstance(tool_msg.content, list):
                        # 1.0 tool content parts: a lone text part is the string form, so it rehydrates like one;
                        # anything else validates as a tool return. No parts left keeps the result, with no content.
                        parts = _input_content_to_user_content(tool_msg.content)
                        if len(parts) == 1 and isinstance(parts[0], str):
                            content = rehydrate_tool_return_content(parts[0])
                        elif parts:
                            content = tool_return_content_ta.validate_python(parts[0] if len(parts) == 1 else parts)
                        else:
                            content = None
                    else:
                        # Rehydrate here (not in a later `ModelMessagesTypeAdapter` pass) so structured and
                        # multimodal content comes back as real types; see `rehydrate_tool_return_content`.
                        content = rehydrate_tool_return_content(tool_msg.content)

                    # Fall back to the paired call's claim: `ToolCallResultEvent` has no metadata
                    # slot, so client-built ToolMessages usually carry no `encrypted_value`. A
                    # non-success return keeps its kind too: it is never promoted to a typed return
                    # part, so the kind can't pass it off as a success. A non-success outcome claim
                    # keeps the return from reloading as `'success'`, which would change how it
                    # serializes to the provider.
                    encrypted_tool_kind = (
                        parse_encrypted_tool_kind(tool_msg.encrypted_value) if use_encrypted_value else None
                    )
                    tool_kind = encrypted_tool_kind or tool_kinds.get(tool_call_id)
                    outcome: Literal['success', 'failed', 'denied', 'interrupted', 'retried'] = 'success'
                    encrypted_outcome = (
                        parse_encrypted_outcome(tool_msg.encrypted_value) if use_encrypted_value else None
                    )
                    retry = None
                    if encrypted_outcome is not None:
                        outcome = encrypted_outcome
                        if outcome == 'retried':
                            retry = parse_encrypted_tool_retry(tool_msg.encrypted_value)
                    elif tool_msg.error is not None:
                        outcome = 'failed'

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
                                outcome=outcome,
                            )
                        )
                    else:
                        # The final `narrow_message_parts` pass parses the rehydrated content into a typed
                        # return subclass when the return succeeded and the `tool_kind` claim validates,
                        # and otherwise leaves the base `ToolReturnPart` (which keeps the claim, unless
                        # it is one of core's kinds).
                        builder.add(
                            ToolReturnPart(
                                tool_name=tool_name,
                                content=content,
                                tool_call_id=tool_call_id,
                                tool_kind=tool_kind,
                                outcome=outcome,
                                retry=retry,
                            )
                        )

                case ReasoningMessage() as reasoning_msg:
                    try:
                        metadata: dict[str, Any] = (
                            json.loads(reasoning_msg.encrypted_value) if reasoning_msg.encrypted_value else {}
                        )
                        if not isinstance(metadata, dict):
                            metadata = {}
                    except (json.JSONDecodeError, RecursionError):  # `RecursionError` on deeply nested JSON
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
                    if activity_msg.activity_type == TOOL_AVAILABILITY_DELTA_ACTIVITY_TYPE:
                        builder.add(tool_availability_delta_from_payload(activity_msg.content))
                    elif activity_msg.activity_type == MODEL_SETTINGS_DELTA_ACTIVITY_TYPE:
                        if (effort_part := model_settings_delta_from_payload(activity_msg.content)) is not None:
                            builder.add(effort_part)
                    elif activity_msg.activity_type == COMPACTION_ACTIVITY_TYPE:
                        if (compaction_part := compaction_part_from_payload(activity_msg.content)) is not None:
                            builder.add(compaction_part)
                    elif activity_msg.activity_type == ENQUEUED_MESSAGES_ACTIVITY_TYPE:
                        # Whole messages, not parts: each delivery is its own `ModelRequest` in history.
                        # They carry their own message ids, so the activity's id isn't kept on them.
                        # `_append_whole` is private on purpose, so custom adapters don't depend on it yet.
                        builder._append_whole(  # pyright: ignore[reportPrivateUsage]
                            cls.load_messages(
                                _enqueued_messages(activity_msg.content), preserve_file_data=preserve_file_data
                            )
                        )
                        continue
                    elif activity_msg.activity_type == FILE_ACTIVITY_TYPE and preserve_file_data:
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

            # Keep the AG-UI message id for `dump_messages`. `ModelRequest`/`ModelResponse` have no id
            # field, so it goes in the metadata of the message the parts above landed in. That may be a new
            # message or the previous one extended, and some AG-UI messages add no parts at all, so ask the
            # builder rather than assume the tail. An AG-UI message lands in a request or a response, never
            # both (an `ActivityMessage` can be either), so whichever lookup answers is the one.
            target = builder.last_modified(checkpoint, of_type=ModelRequest) or builder.last_modified(
                checkpoint, of_type=ModelResponse
            )
            if target is not None:
                set_ui_message_id(target, msg.id)

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
        request keeps its position instead of being reordered after the user prompt. A run of
        consecutive system prompts becomes one `SystemMessage` in the same way, so a system prompt
        enqueued after a tool result reloads after it, not at the head of the request.
        """
        # An SDK without the retired `binary` part can't build a message holding it, so it writes typed media.
        use_multimodal = parse_ag_ui_version(ag_ui_version) >= MULTIMODAL_VERSION or not HAS_BINARY_INPUT_PART
        # `ToolMessage.encrypted_value` (the `tool_kind` carrier here) and `UserMessage.encrypted_value`
        # (the `StructuredContent` carrier) landed in 0.1.11 — see `tool_kind_encrypted_value`.
        use_encrypted_value = parse_ag_ui_version(ag_ui_version) >= ENCRYPTED_VALUE_VERSION
        result: list[Message] = []
        system_content: list[str] = []
        user_content: list[InputContent] = []
        # Positions in `user_content` of the text items that are `StructuredContent`, with what text can't carry.
        structured_payloads: dict[int, dict[str, Any]] = {}

        def flush_user_content() -> None:
            nonlocal user_content, structured_payloads
            if not user_content:
                return
            result.append(
                UserMessage(
                    id=_new_message_id(),
                    # Simplify to plain string if only a single text item.
                    content=user_content[0].text
                    if len(user_content) == 1 and isinstance(user_content[0], TextInputContent)
                    else user_content,
                    **structured_content_encrypted_value_kwargs(structured_payloads),
                )
            )
            user_content = []
            structured_payloads = {}

        def flush_system_content() -> None:
            nonlocal system_content
            if system_content:
                result.append(SystemMessage(id=_new_message_id(), content='\n'.join(system_content)))
                system_content = []

        for part in msg.parts:
            if isinstance(part, SystemPromptPart):
                # A request-scoped prompt is for the model's current request only; loaded back, it would
                # come back as a lasting system prompt.
                if part.scope != 'request':
                    flush_user_content()
                    system_content.append(part.content)
                continue
            if not isinstance(part, (InstructionDeltaPart, InstructionBaselinePart, SpeechPart)):
                # Parts that dump to nothing don't end the run of system prompts.
                flush_system_content()
            if isinstance(part, UserPromptPart):
                # Each prompt gets its own `UserMessage`, which loads back as its own `UserPromptPart`. Merged
                # into one, a request's prompts would come back as one, which providers like OpenAI Chat send
                # as one user message instead of one per prompt.
                flush_user_content()
                if isinstance(part.content, str):
                    user_content.append(TextInputContent(type='text', text=part.content))
                else:
                    for item in part.content:
                        if isinstance(item, StructuredContent):
                            # AG-UI has no structured user input, so the item is text the frontend shows,
                            # and its value rides the message's `encrypted_value`. Before 0.1.11 it's text only, with
                            # no warning (unlike a lost `tool_kind`): the text the user and model see is unchanged.
                            if use_encrypted_value:
                                structured_payloads[len(user_content)] = structured_content_payload(item)
                            user_content.append(TextInputContent(type='text', text=item.content))
                        elif isinstance(item, UploadedFile) and preserve_file_data:
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
                # A non-success outcome rides the `encrypted_value` carrier alongside `tool_kind`,
                # since a `ToolMessage` has no outcome slot. `error` is the return's own content, so a
                # retry's feedback reaches the frontend as the feedback itself, without the
                # instruction framing a `RetryPromptPart` renders
                # (https://github.com/pydantic/pydantic-ai/pull/4869).
                result.append(
                    ToolMessage(
                        id=_new_message_id(),
                        content=dump_tool_return_content(part.content),
                        tool_call_id=part.tool_call_id,
                        error=part.model_response_str(wrap_if_error=False)
                        if part.outcome in ERROR_OUTCOMES or part.outcome == 'denied'
                        else None,
                        **tool_kind_encrypted_value_kwargs(
                            part.tool_kind, outcome=part.outcome, retry=part.retry, supported=use_encrypted_value
                        ),
                    )
                )
                if (app := get_mcp_app_tool_result(part.metadata)) is not None:
                    result.append(
                        ActivityMessage(
                            id=_new_message_id(), activity_type=MCP_APPS_UI_TYPE, content=mcp_app_ui_payload(app)
                        )
                    )
            elif isinstance(part, (InstructionDeltaPart, InstructionBaselinePart)):
                pass  # Operator instruction state stays in trusted server-side history.
            elif isinstance(part, ModelSettingsDeltaPart):
                flush_user_content()
                result.append(
                    ActivityMessage(
                        id=_new_message_id(),
                        activity_type=MODEL_SETTINGS_DELTA_ACTIVITY_TYPE,
                        content=model_settings_delta_payload(part),
                    )
                )
            elif isinstance(part, ToolAvailabilityDeltaPart):
                flush_user_content()
                result.append(
                    ActivityMessage(
                        id=_new_message_id(),
                        activity_type=TOOL_AVAILABILITY_DELTA_ACTIVITY_TYPE,
                        content={
                            'added': part.tools_added,
                            'tool_call_id': part.tool_call_id,
                        },
                    )
                )
            # TODO(v3): remove `RetryPromptPart`
            elif isinstance(part, RetryPromptPart | RetryFeedbackPart):  # pyright: ignore[reportDeprecated]
                # Harness feedback dumps in the voice the model is shown it in, not as text a person
                # appears to have written (https://github.com/pydantic/pydantic-ai/issues/6404), and
                # carries the part itself so the reload can rebuild it. Flushing first keeps it in
                # the position it was authored in, instead of joining the standing prompt at the head
                # of the request; `VercelAIAdapter` keeps the authored order the same way.
                flush_user_content()
                translated = (
                    _translate_legacy_retry_part(part)
                    if isinstance(part, RetryPromptPart)  # pyright: ignore[reportDeprecated]
                    else part
                )
                if isinstance(translated, ToolReturnPart):
                    result.append(
                        ToolMessage(
                            id=_new_message_id(),
                            content=dump_tool_return_content(translated.content),
                            tool_call_id=translated.tool_call_id,
                            error=translated.model_response_str(wrap_if_error=False),
                            **tool_kind_encrypted_value_kwargs(
                                translated.tool_kind,
                                outcome=translated.outcome,
                                retry=translated.retry,
                                supported=use_encrypted_value,
                            ),
                        )
                    )
                elif _retry_feedback_speaks_for_the_harness(translated):
                    result.append(
                        SystemMessage(
                            id=_new_message_id(),
                            content=translated.model_response(),
                            **retry_feedback_encrypted_value_kwargs(translated, supported=use_encrypted_value),
                        )
                    )
                else:
                    result.append(
                        UserMessage(
                            id=_new_message_id(),
                            content=translated.model_response(),
                            **retry_feedback_encrypted_value_kwargs(translated, supported=use_encrypted_value),
                        )
                    )
            elif isinstance(part, CompactionPart):
                flush_user_content()
                result.append(
                    ActivityMessage(
                        id=_new_message_id(),
                        activity_type=COMPACTION_ACTIVITY_TYPE,
                        content=compaction_payload(part),
                    )
                )
            elif isinstance(part, SpeechPart):  # pragma: no cover
                pass  # Realtime audio parts are not rendered in AG-UI
            else:
                assert_never(part)

        flush_system_content()
        flush_user_content()
        return result

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
        # `ReasoningMessage` and the `tool_kind` carrier `ToolCall`/`ToolMessage.encrypted_value`
        # both landed in 0.1.11, so these two thresholds coincide — see `tool_kind_encrypted_value`.
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

        for index, part in enumerate(msg.parts):
            if isinstance(part, TextPart):
                if tool_calls_list:
                    flush()
                # Check adjacency in the original parts, not the buffer: a part dropped from the dump still separates text.
                if index > 0 and isinstance(msg.parts[index - 1], TextPart):
                    text_content[-1] += part.content
                else:
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
                            error=builtin_return.model_response_str(wrap_if_error=False)
                            if builtin_return.outcome in ERROR_OUTCOMES or builtin_return.outcome == 'denied'
                            else None,
                            **tool_kind_encrypted_value_kwargs(
                                builtin_return.tool_kind, outcome=builtin_return.outcome, supported=use_encrypted_value
                            ),
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
            elif isinstance(part, CompactionPart):
                flush()
                result.append(
                    ActivityMessage(
                        id=_new_message_id(),
                        activity_type=COMPACTION_ACTIVITY_TYPE,
                        content=compaction_payload(part),
                    )
                )
            elif isinstance(part, SpeechPart):  # pragma: no cover
                pass  # Realtime audio parts are not rendered in AG-UI
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

        - `ModelRequest.metadata` and top-level `ModelResponse.provider_details` are lost. AG-UI has
          no trusted message-level carrier for framework or provider state; general client-controlled
          metadata must not be restored as server-side state. The one thing `load_messages` keeps in
          the reserved `__pydantic_ai__` namespace is the AG-UI message id, which comes back on the
          last message dumped from each `ModelMessage`.
        - `TextPart.id`, `.provider_name`, `.provider_details`, `.citations` are lost.
        - `ToolCallPart.id`, `.provider_name`, `.provider_details` are lost.
        - `ToolCallPart.args` and `NativeToolCallPart.args` that don't parse as a JSON object are
          rewritten to `'{"INVALID_JSON":"<raw args>"}'` (see
          [`args_as_json_str`][pydantic_ai.messages.BaseToolCallPart.args_as_json_str]), so the raw
          string is no longer recoverable as args on reload. Unlike the live event stream, which emits
          them verbatim so streamed fragments stay concatenable, history has to hold a sendable value.
          Args that do parse are dumped as compact JSON, the form the model adapters send, so whitespace
          in the model's raw string doesn't survive.
        - `NativeToolCallPart.id`, `.provider_details` are lost (only `.provider_name` survives
          via the prefixed tool call ID).
        - `NativeToolReturnPart.provider_details` is lost.
        - `tool_kind` is lost when `ag_ui_version < '0.1.11'` (before its `encrypted_value` carrier
          existed), so typed tool parts reload as their base classes.
        - A failed, denied or interrupted (native) tool return keeps its `tool_kind`, but reloads as a
          plain `ToolReturnPart` / `NativeToolReturnPart`: only successful returns are promoted to typed parts.
        - A non-`'success'` `outcome` on a (native) tool return survives via the `encrypted_value`
          carrier from 0.1.11 (`ToolMessage` has no outcome slot). Below that, `'failed'` survives
          via `ToolMessage.error`, `'denied'` and `'retried'` reload as `'failed'`, and
          `'interrupted'` reloads as `'success'`.
        - `RetryFeedbackPart` dumps as a message in the voice the model is shown it in — a
          `UserMessage` for a `'validation_error'` or `'model_retry'`, a `SystemMessage` for
          `'no_output'` feedback in the framework's own words — with the part
          itself in that message's `encrypted_value`; it reloads as a `RetryFeedbackPart` only from
          there, so a message the client wrote stays a plain prompt part. Below 0.1.11 the carrier
          doesn't exist, so it reloads as that plain part holding the rendered text, and the dump
          warns.
        - `RetryPromptPart` is dumped as the part it always meant, so a tool-bound one becomes a
          `ToolMessage` and reloads as a `ToolReturnPart` with `outcome='retried'`, and a tool-less
          one takes the `RetryFeedbackPart` path above.
        - Each `UserPromptPart` is dumped as its own `UserMessage`, so a request's prompts reload as the
          same separate parts rather than merged into one.
        - A `NativeToolReturnPart` is always emitted directly after its `NativeToolCallPart`, so any
          part that originally sat between them — e.g. a `CompactionPart` — reloads after the pair
          instead. Provider adapters emit compaction parts outside call/return pairs, so this only
          affects hand-constructed histories.
        - `CachePoint` and `UploadedFile` content items are dropped (unless `preserve_file_data=True`).
        - `FileUrl.force_download` is dropped when `ag_ui_version < '0.1.15'` (before typed
          multimodal content gained a metadata carrier) on an installed `ag-ui-protocol < 1.0`.
        - `ThinkingPart` is dropped when `ag_ui_version='0.1.10'`.
        - `FilePart` is silently dropped unless `preserve_file_data=True`.
        - `UploadedFile` in a multi-item `UserPromptPart` is split into a separate activity message
          when `preserve_file_data=True`, which reloads as a separate `UserPromptPart`.
        - `StructuredContent` is dumped as a text item, with its value in the `UserMessage`'s
          `encrypted_value` from `ag_ui_version='0.1.11'`, and reloads without its `metadata`. Below
          that it's dumped, and reloads, as plain text.
        - `MultiModalContent` items in `ToolReturnPart`/`NativeToolReturnPart.content` always round-trip,
          regardless of `preserve_file_data`: the full content (files as base64/URL dicts) is serialized
          inline into the JSON `ToolMessage.content` and rehydrated on reload through the `ToolReturnContent`
          union. The same serialization is used for both history (`dump_messages`) and the live
          event stream (`ToolCallResultEvent.content`), so files survive either round-trip.
        - An MCP Apps record in `ToolReturnPart.metadata` is dumped as an `mcp-apps` activity message
          after the tool result, which `load_messages` ignores, so the record is not restored.
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
            isinstance(part, RetryFeedbackPart)
            # TODO(v3): remove `RetryPromptPart`. A tool-less one is translated into the feedback
            # part it always meant and rides the same marker channel, so it is lost below the floor
            # the same way. A tool-bound one becomes a `tool_kind`-less return, which falls into the
            # outcome gap described below.
            or (isinstance(part, RetryPromptPart) and part.tool_name is None)  # pyright: ignore[reportDeprecated]
            or (
                isinstance(part, (ToolCallPart, ToolReturnPart, NativeToolCallPart, NativeToolReturnPart))
                # A non-`'success'` `outcome` on a `tool_kind`-less return also degrades below the
                # floor and also goes unwarned. That gap predates this part and stays as it was —
                # arming the warning for it would start warning users who never hit retry feedback.
                and part.tool_kind is not None
            )
            for msg in messages
            for part in msg.parts
        ):
            warn_encrypted_value_not_persisted(ag_ui_version)

        for msg in messages:
            if isinstance(msg, ModelRequest):
                dumped = cls._dump_request_parts(
                    msg, ag_ui_version=ag_ui_version, preserve_file_data=preserve_file_data
                )
            elif isinstance(msg, ModelResponse):
                dumped = cls._dump_response_parts(
                    msg, ag_ui_version=ag_ui_version, preserve_file_data=preserve_file_data
                )
            else:
                assert_never(msg)
            # `load_messages` kept the id of the last AG-UI message merged into `msg`, so it goes back on the
            # last AG-UI message dumped from it. The others keep their fresh ids: several tool results in one
            # request become several `ToolMessage`s, and those need distinct ids.
            if dumped and (ui_message_id := get_ui_message_id(msg)) is not None:
                dumped[-1].id = ui_message_id
            result.extend(dumped)

        return result
