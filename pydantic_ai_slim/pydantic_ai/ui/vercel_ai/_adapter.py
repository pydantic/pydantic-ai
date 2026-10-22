"""Vercel AI adapter for handling requests."""

from __future__ import annotations

import json
import uuid
from collections.abc import Callable, Sequence
from dataclasses import KW_ONLY, InitVar, dataclass
from functools import cached_property
from typing import TYPE_CHECKING, Any, Literal, assert_never, cast

from pydantic import TypeAdapter, ValidationError

from pydantic_ai._utils import is_str_dict as _is_str_dict

from ... import _instructions
from ...messages import (
    _FILE_URL_KINDS,  # pyright: ignore[reportPrivateUsage]
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
    MultiModalContent,
    NativeToolCallPart,
    NativeToolReturnPart,
    RetryFeedbackPart,
    RetryPromptPart,  # pyright: ignore[reportDeprecated]  # TODO(v3): remove RetryPromptPart
    SpeechPart,
    SystemPromptPart,
    TextContent,
    TextPart,
    ThinkingPart,
    ToolAvailabilityDeltaPart,
    ToolCallPart,
    ToolReturnPart,
    UploadedFile,
    UploadedFileProviderName,
    UserContent,
    UserPromptPart,
    VideoUrl,
    _retry_feedback_speaks_for_the_harness,  # pyright: ignore[reportPrivateUsage]
    _translate_legacy_retry_part,  # pyright: ignore[reportPrivateUsage]
    narrow_message_parts,
    parse_tool_kind,
    tool_return_content_ta,
    tool_return_ta,
)
from ...output import OutputDataT
from ...tools import AgentDepsT, DeferredToolResults, ToolDenied
from .. import BuilderCheckpoint, MessagesBuilder, UIAdapter
from .._adapter import (
    DEFAULT_ALLOWED_CONTENT_TYPES,
    compaction_part_from_payload,
    compaction_payload,
    parse_tool_retry,
    resolve_allow_uploaded_files,
    retry_feedback_from_payload,
    retry_feedback_payload,
    tool_availability_delta_from_payload,
    tool_retry_payload,
)
from .._utils import get_ui_message_id, set_ui_message_id
from ._event_stream import VercelAIEventStream
from ._utils import (
    COMPACTION_DATA_TYPE,
    ENQUEUED_MESSAGES_DATA_TYPE,
    PROVIDER_METADATA_KEY,
    TOOL_AVAILABILITY_DELTA_DATA_TYPE,
    apply_message_metadata,
    dump_message_metadata,
    dump_provider_metadata,
    iter_metadata_chunks,
    iter_tool_approval_responses,
    load_provider_metadata,
    tool_return_output,
)
from .request_types import (
    DataUIPart,
    DynamicToolOutputAvailablePart,
    DynamicToolOutputDeniedPart,
    DynamicToolOutputErrorPart,
    DynamicToolUIPart,
    FileUIPart,
    ProviderMetadata,
    ReasoningUIPart,
    RequestData,
    SourceDocumentUIPart,
    SourceUrlUIPart,
    StepStartUIPart,
    TextUIPart,
    ToolApprovalRequested,
    ToolApprovalRequestedPart,
    ToolApprovalResponded,
    ToolInputAvailablePart,
    ToolOutputAvailablePart,
    ToolOutputDeniedPart,
    ToolOutputErrorPart,
    ToolUIPart,
    UIMessage,
    UIMessagePart,
)
from .response_types import BaseChunk, DataChunk, FileChunk, SourceDocumentChunk, SourceUrlChunk

if TYPE_CHECKING:
    from starlette.requests import Request
    from starlette.responses import Response

    from ...agent import AbstractAgent
    from ...agent.abstract import AgentMetadata
    from ...capabilities import AbstractCapability
    from ...models import KnownModelName, Model
    from ...output import OutputSpec
    from ...settings import ModelSettings
    from ...tools import DeferredToolApprovalResult
    from ...toolsets import AbstractToolset
    from ...usage import RunUsage, UsageLimits
    from .. import UIEventStream
    from .._adapter import DispatchDepsT, DispatchOutputDataT
    from .._event_stream import OnCancelFunc, OnCompleteFunc

__all__ = ['VercelAIAdapter']

request_data_ta: TypeAdapter[RequestData] = TypeAdapter(RequestData)
_ui_messages_ta: TypeAdapter[list[UIMessage]] = TypeAdapter(list[UIMessage])

_MEDIA_PREFIX_TO_URL_TYPE: dict[str, type[ImageUrl | AudioUrl | VideoUrl]] = {
    'image': ImageUrl,
    'video': VideoUrl,
    'audio': AudioUrl,
}

_USER_PROMPT_START_KEY = 'user_prompt_start'
"""`provider_metadata` key on the user UI part where a request's second or later `UserPromptPart` begins."""

_KIND_TO_URL_TYPE: dict[str, type[ImageUrl | AudioUrl | VideoUrl | DocumentUrl]] = {
    ImageUrl.kind: ImageUrl,
    VideoUrl.kind: VideoUrl,
    AudioUrl.kind: AudioUrl,
    DocumentUrl.kind: DocumentUrl,
}


def _generate_message_id(
    msg: ModelRequest | ModelResponse, role: Literal['system', 'user', 'assistant'], message_index: int
) -> str:
    """Return the `UIMessage.id` the message was loaded from, else a deterministic ID from content and position.

    Priority order:
    1. The `UIMessage.id` kept by `load_messages`, so a history the client sent comes back with its own ids.
    2. For `ModelResponse` with `provider_response_id` set, use '{provider_response_id}-{message_index}'.
    3. For any message with run_id set, use '{run_id}-{message_index}'.
    4. Fallback: UUID5 from 'timestamp-kind-role-message_index'.
    """
    if (ui_message_id := get_ui_message_id(msg)) is not None:
        return ui_message_id
    if isinstance(msg, ModelResponse) and msg.provider_response_id:
        return f'{msg.provider_response_id}-{message_index}'
    if msg.run_id:
        return f'{msg.run_id}-{message_index}'
    ts_str = msg.timestamp.isoformat() if msg.timestamp else ''
    return str(uuid.uuid5(uuid.NAMESPACE_OID, f'{ts_str}-{msg.kind}-{role}-{message_index}'))


@dataclass
class VercelAIAdapter(UIAdapter[RequestData, UIMessage, BaseChunk, AgentDepsT, OutputDataT]):
    """UI adapter for the Vercel AI protocol."""

    _: KW_ONLY
    sdk_version: Literal[5, 6, 7] = 5
    """Vercel AI SDK version to target. Default is 5 for backwards compatibility.

    Setting `sdk_version=6` enables tool approval streaming for human-in-the-loop workflows.
    `sdk_version=7` emits the same wire as 6 (v7's data-stream protocol equals v6's); it is
    accepted so the value reflects the client's real SDK major and reserves it for future
    v7-only chunks.
    """
    server_message_id: str | None = None
    """Optional server-generated message ID to include in the `StartChunk`."""

    preserve_file_data: InitVar[bool | None] = None  # TODO(v3): remove preserve_file_data
    """Deprecated alias for [`allow_uploaded_files`][pydantic_ai.ui.UIAdapter.allow_uploaded_files]."""

    def __post_init__(self, preserve_file_data: bool | None) -> None:
        # `stacklevel=4` points the warning at the user's `VercelAIAdapter(...)` call:
        # user → generated `__init__` → `__post_init__` → helper → `warn`.
        self.allow_uploaded_files = resolve_allow_uploaded_files(
            self.allow_uploaded_files, preserve_file_data, stacklevel=4
        )

    @classmethod
    def build_run_input(cls, body: bytes) -> RequestData:
        """Build a Vercel AI run input object from the request body."""
        return request_data_ta.validate_json(body)

    @classmethod
    async def from_request(
        cls,
        request: Request,
        *,
        agent: AbstractAgent[AgentDepsT, OutputDataT],
        sdk_version: Literal[5, 6, 7] = 5,
        server_message_id: str | None = None,
        manage_system_prompt: Literal['server', 'client'] = 'server',
        allowed_file_url_schemes: frozenset[str] = frozenset({'http', 'https'}),
        allowed_file_url_force_download: frozenset[ForceDownloadMode] = frozenset(),
        allow_uploaded_files: bool = False,
        preserve_file_data: bool | None = None,
        allowed_content_types: frozenset[str] | None = DEFAULT_ALLOWED_CONTENT_TYPES,
        **kwargs: Any,
    ) -> VercelAIAdapter[AgentDepsT, OutputDataT]:
        """Extends [`from_request`][pydantic_ai.ui.UIAdapter.from_request] with Vercel AI-specific parameters.

        `preserve_file_data` is a deprecated alias for `allow_uploaded_files`.
        """
        allow_uploaded_files = resolve_allow_uploaded_files(allow_uploaded_files, preserve_file_data)
        return await super().from_request(
            request,
            agent=agent,
            sdk_version=sdk_version,
            server_message_id=server_message_id,
            manage_system_prompt=manage_system_prompt,
            allowed_file_url_schemes=allowed_file_url_schemes,
            allowed_file_url_force_download=allowed_file_url_force_download,
            allow_uploaded_files=allow_uploaded_files,
            allowed_content_types=allowed_content_types,
            **kwargs,
        )

    @classmethod
    async def dispatch_request(
        cls,
        request: Request,
        *,
        agent: AbstractAgent[DispatchDepsT, DispatchOutputDataT],
        sdk_version: Literal[5, 6, 7] = 5,
        server_message_id: str | None = None,
        message_history: Sequence[ModelMessage] | None = None,
        deferred_tool_results: DeferredToolResults | None = None,
        conversation_id: str | None = None,
        run_id: str | None = None,
        model: Model | KnownModelName | str | None = None,
        instructions: _instructions.AgentInstructions[DispatchDepsT] = None,
        deps: DispatchDepsT = None,
        output_type: OutputSpec[Any] | None = None,
        model_settings: ModelSettings | None = None,
        usage_limits: UsageLimits | None = None,
        usage: RunUsage | None = None,
        metadata: AgentMetadata[DispatchDepsT] | None = None,
        infer_name: bool = True,
        toolsets: Sequence[AbstractToolset[DispatchDepsT]] | None = None,
        capabilities: Sequence[AbstractCapability[DispatchDepsT]] | None = None,
        on_complete: OnCompleteFunc[BaseChunk] | None = None,
        on_cancel: OnCancelFunc[BaseChunk] | None = None,
        manage_system_prompt: Literal['server', 'client'] = 'server',
        allowed_file_url_schemes: frozenset[str] = frozenset({'http', 'https'}),
        allowed_file_url_force_download: frozenset[ForceDownloadMode] = frozenset(),
        allow_uploaded_files: bool = False,
        preserve_file_data: bool | None = None,
        allowed_content_types: frozenset[str] | None = DEFAULT_ALLOWED_CONTENT_TYPES,
        **kwargs: Any,
    ) -> Response:
        """Extends [`dispatch_request`][pydantic_ai.ui.UIAdapter.dispatch_request] with Vercel AI-specific parameters.

        `preserve_file_data` is a deprecated alias for `allow_uploaded_files`.
        """
        allow_uploaded_files = resolve_allow_uploaded_files(allow_uploaded_files, preserve_file_data)
        return await super().dispatch_request(
            request,
            agent=agent,
            sdk_version=sdk_version,
            server_message_id=server_message_id,
            message_history=message_history,
            deferred_tool_results=deferred_tool_results,
            conversation_id=conversation_id,
            run_id=run_id,
            model=model,
            instructions=instructions,
            deps=deps,
            output_type=output_type,
            model_settings=model_settings,
            usage_limits=usage_limits,
            usage=usage,
            metadata=metadata,
            infer_name=infer_name,
            toolsets=toolsets,
            capabilities=capabilities,
            on_complete=on_complete,
            on_cancel=on_cancel,
            manage_system_prompt=manage_system_prompt,
            allowed_file_url_schemes=allowed_file_url_schemes,
            allowed_file_url_force_download=allowed_file_url_force_download,
            allow_uploaded_files=allow_uploaded_files,
            allowed_content_types=allowed_content_types,
            **kwargs,
        )

    def build_event_stream(self) -> UIEventStream[RequestData, BaseChunk, AgentDepsT, OutputDataT]:
        """Build a Vercel AI event stream transformer."""
        return VercelAIEventStream(
            self.run_input,
            accept=self.accept,
            sdk_version=self.sdk_version,
            server_message_id=self.server_message_id,
            manage_system_prompt=self.manage_system_prompt,
        )

    @cached_property
    def deferred_tool_results(self) -> DeferredToolResults | None:
        """Extract deferred tool results from Vercel AI messages with approval responses."""
        if self.sdk_version < 6:
            return None
        approvals: dict[str, bool | DeferredToolApprovalResult] = {}
        for tool_call_id, approval in iter_tool_approval_responses(self.run_input.messages):
            if approval.approved:
                approvals[tool_call_id] = True
            elif approval.reason:
                approvals[tool_call_id] = ToolDenied(message=approval.reason)
            else:
                approvals[tool_call_id] = False
        return DeferredToolResults(approvals=approvals) if approvals else None

    @cached_property
    def messages(self) -> list[ModelMessage]:
        """Pydantic AI messages from the Vercel AI run input."""
        return self.load_messages(self.run_input.messages)

    @cached_property
    def conversation_id(self) -> str | None:
        """Conversation ID from the top-level `id` field of the Vercel AI request body (the chat ID)."""
        return self.run_input.id

    @classmethod
    def load_messages(cls, messages: Sequence[UIMessage]) -> list[ModelMessage]:  # noqa: C901
        """Transform Vercel AI messages into Pydantic AI messages."""
        builder = MessagesBuilder()

        for msg in messages:
            checkpoint = builder.checkpoint()
            enqueued: list[ModelMessage] = []

            if msg.role == 'system':
                for part in msg.parts:
                    if isinstance(part, TextUIPart):
                        # A system part only reloads as harness retry feedback when it carries the
                        # marker this adapter dumped it with; a client-authored system message has
                        # no such payload and stays a plain `SystemPromptPart`, so it can never gain
                        # harness provenance. See `retry_feedback_from_payload`.
                        feedback = retry_feedback_from_payload(
                            load_provider_metadata(part.provider_metadata).get('retry_feedback')
                        )
                        builder.add(feedback if feedback is not None else SystemPromptPart(content=part.text))
                    else:  # pragma: no cover
                        raise ValueError(f'Unsupported system message part type: {type(part)}')
            elif msg.role == 'user':
                user_prompt_content: list[UserContent] = []

                def flush_user_prompt(content: list[UserContent] = user_prompt_content) -> None:
                    """Close off the user content accumulated so far, ahead of whatever follows it.

                    `content` defaults to this message's list rather than closing over the name,
                    which the next message rebinds; callers pass nothing.
                    """
                    if not content:
                        return
                    builder.add(_user_prompt_part([*content]))
                    content.clear()

                for part in msg.parts:
                    if (
                        isinstance(part, TextUIPart | FileUIPart)
                        and user_prompt_content
                        and load_provider_metadata(part.provider_metadata).get(_USER_PROMPT_START_KEY)
                    ):
                        # `dump_messages` marks where each of a request's later prompts begins.
                        flush_user_prompt()

                    if isinstance(part, TextUIPart):
                        # A user part reloads as harness retry feedback only when it carries the
                        # marker this adapter dumped it with; text a person wrote has no such payload
                        # and stays a `UserPromptPart`. See `retry_feedback_from_payload`.
                        feedback = retry_feedback_from_payload(
                            load_provider_metadata(part.provider_metadata).get('retry_feedback')
                        )
                        if feedback is not None:
                            flush_user_prompt()
                            builder.add(feedback)
                        else:
                            user_prompt_content.append(part.text)
                    elif isinstance(part, FileUIPart):
                        provider_meta = load_provider_metadata(part.provider_metadata)
                        # Restoring client-supplied `vendor_metadata` is intentional (as the `UploadedFile` branch
                        # already does — see https://github.com/pydantic/pydantic-ai/issues/5571 and
                        # https://github.com/pydantic/pydantic-ai/issues/5772): it carries only the requester's own
                        # request params and is dict-validated by the constructors below.
                        vendor_metadata = provider_meta.get('vendor_metadata')
                        force_download = provider_meta.get('force_download', False)
                        try:
                            file = BinaryContent.from_data_uri(part.url)
                        except ValueError:
                            # Check provider_metadata for UploadedFile data
                            uploaded_file_id = provider_meta.get('file_id')
                            uploaded_file_provider = provider_meta.get('provider_name')
                            if uploaded_file_id and uploaded_file_provider:
                                file = UploadedFile(
                                    file_id=uploaded_file_id,
                                    provider_name=cast(UploadedFileProviderName, uploaded_file_provider),
                                    media_type=part.media_type,
                                    vendor_metadata=vendor_metadata,
                                    identifier=provider_meta.get('identifier'),
                                )
                            else:
                                if part.media_type:
                                    url_type = _MEDIA_PREFIX_TO_URL_TYPE.get(
                                        part.media_type.split('/', 1)[0], DocumentUrl
                                    )
                                else:
                                    # A URL Pydantic AI could not read a media type out of, dumped with an
                                    # empty one: recover the kind from the metadata written alongside it,
                                    # rather than letting the empty media prefix make everything a document.
                                    # `provider_metadata` is the client's to send, so normalize before the
                                    # lookup, which falls back to a document on anything we didn't write.
                                    url_type = _KIND_TO_URL_TYPE.get(str(provider_meta.get('kind')), DocumentUrl)
                                file = url_type(
                                    url=part.url,
                                    media_type=part.media_type,
                                    force_download=force_download,
                                    vendor_metadata=vendor_metadata,
                                )
                        else:
                            # `from_data_uri` succeeded: restore vendor_metadata onto the BinaryContent.
                            # Reconstruct through the constructor so a malformed client value is rejected
                            # here (matching the URL constructor path) instead of being stored unvalidated
                            # and crashing a provider model later. Re-narrow afterwards so an image
                            # round-trips back to `BinaryImage` (as `from_data_uri` returned it), not
                            # plain `BinaryContent`.
                            if vendor_metadata is not None:
                                file = BinaryContent.narrow_type(
                                    BinaryContent(
                                        data=file.data,
                                        media_type=file.media_type,
                                        identifier=file.identifier,
                                        vendor_metadata=vendor_metadata,
                                    )
                                )
                        user_prompt_content.append(file)
                    elif isinstance(part, DataUIPart):
                        if part.type == TOOL_AVAILABILITY_DELTA_DATA_TYPE and _is_str_dict(part.data):
                            builder.add(tool_availability_delta_from_payload(part.data))
                    else:  # pragma: no cover
                        raise ValueError(f'Unsupported user message part type: {type(part)}')

                flush_user_prompt()

            elif msg.role == 'assistant':
                # A `ModelResponse` with parallel tool calls is followed by one `ModelRequest` with all
                # their returns, but a Vercel tool part carries a call and its output together. Hold the
                # returns back until the step (one model response) ends, so the calls stay in one
                # response. Streamed messages mark steps with `step-start` parts; without them (as
                # `dump_messages` writes, one response per message) content after a completed tool
                # call is taken to start the next step.
                pending_returns: list[ToolReturnPart | ToolAvailabilityDeltaPart] = []
                # Enqueued messages were delivered after the step's tool returns, so they wait with them.
                # `_append_whole` is private on purpose, so custom adapters don't depend on it yet.
                pending_enqueued: list[ModelMessage] = []
                has_steps = any(isinstance(part, StepStartUIPart) for part in msg.parts)
                for part in msg.parts:
                    if pending_returns and (
                        isinstance(part, StepStartUIPart)
                        or (not has_steps and isinstance(part, TextUIPart | ReasoningUIPart | FileUIPart))
                    ):
                        for return_part in pending_returns:
                            builder.add(return_part)
                        builder._append_whole(pending_enqueued)  # pyright: ignore[reportPrivateUsage]
                        pending_returns = []
                        pending_enqueued = []

                    if isinstance(part, TextUIPart):
                        provider_meta = load_provider_metadata(part.provider_metadata)
                        builder.add(
                            TextPart(
                                content=part.text,
                                id=provider_meta.get('id'),
                                provider_name=provider_meta.get('provider_name'),
                                provider_details=provider_meta.get('provider_details'),
                            )
                        )
                    elif isinstance(part, ReasoningUIPart):
                        provider_meta = load_provider_metadata(part.provider_metadata)
                        builder.add(
                            ThinkingPart(
                                content=part.text,
                                id=provider_meta.get('id'),
                                signature=None if part.state == 'streaming' else provider_meta.get('signature'),
                                provider_name=provider_meta.get('provider_name'),
                                provider_details=provider_meta.get('provider_details'),
                            )
                        )
                    elif isinstance(part, FileUIPart):
                        try:
                            file = BinaryContent.from_data_uri(part.url)
                        except ValueError as e:  # pragma: no cover
                            # We don't yet handle non-data-URI file URLs returned by assistants, as no Pydantic AI models do this.
                            raise ValueError(
                                'Vercel AI integration can currently only handle assistant file parts with data URIs.'
                            ) from e
                        provider_meta = load_provider_metadata(part.provider_metadata)
                        vendor_metadata = provider_meta.get('vendor_metadata')
                        # `vendor_metadata` is client-supplied and unconstrained; assignment on the
                        # (non-`validate_assignment`) `BinaryContent` dataclass bypasses validation,
                        # so ignore anything that isn't a dict rather than let it reach the provider.
                        if _is_str_dict(vendor_metadata):
                            file.vendor_metadata = vendor_metadata
                        builder.add(
                            FilePart(
                                content=file,
                                id=provider_meta.get('id'),
                                provider_name=provider_meta.get('provider_name'),
                                provider_details=provider_meta.get('provider_details'),
                            )
                        )
                    elif isinstance(part, DataUIPart):
                        if (
                            part.type == COMPACTION_DATA_TYPE
                            and _is_str_dict(part.data)
                            and (compaction_part := compaction_part_from_payload(part.data)) is not None
                        ):
                            builder.add(compaction_part)
                        elif part.type == TOOL_AVAILABILITY_DELTA_DATA_TYPE and _is_str_dict(part.data):
                            # The event stream emits the delta after the tool result that caused it, so
                            # the AI SDK attaches it to the assistant message. Adding it here lands it on
                            # the request carrying that tool return, matching the server-side history. It waits
                            # with the step's held-back tool returns so it follows the return that caused it.
                            pending_returns.append(tool_availability_delta_from_payload(part.data))
                        elif part.type == ENQUEUED_MESSAGES_DATA_TYPE:
                            # Whole messages, not parts: each delivery is its own `ModelRequest` in history.
                            loaded = cls.load_messages(_enqueued_ui_messages(part.data))
                            if pending_returns:
                                pending_enqueued.extend(loaded)
                            else:
                                builder._append_whole(loaded)  # pyright: ignore[reportPrivateUsage]
                            enqueued.extend(loaded)
                    elif isinstance(part, ToolUIPart | DynamicToolUIPart):
                        if isinstance(part, DynamicToolUIPart):
                            tool_name = part.tool_name
                            builtin_tool = part.provider_executed
                        else:
                            tool_name = part.type.removeprefix('tool-')
                            builtin_tool = part.provider_executed

                        tool_call_id = part.tool_call_id

                        args: str | dict[str, Any] | None = part.input

                        if isinstance(args, str):
                            try:
                                parsed = json.loads(args)
                                if _is_str_dict(parsed):
                                    args = parsed
                            except json.JSONDecodeError:
                                pass
                        elif isinstance(args, dict) or args is None:
                            pass
                        else:
                            assert_never(args)

                        provider_meta = load_provider_metadata(part.call_provider_metadata)
                        part_id = provider_meta.get('id')
                        provider_name = provider_meta.get('provider_name')
                        provider_details = provider_meta.get('provider_details')
                        raw_tool_kind = provider_meta.get('tool_kind')
                        tool_kind = parse_tool_kind(raw_tool_kind) if isinstance(raw_tool_kind, str) else None

                        if builtin_tool:
                            # For builtin tools, we need to create 2 parts (BuiltinToolCall & BuiltinToolReturn) for a single Vercel ToolOutput
                            # The call and return metadata are combined in the output part.
                            # So we extract and return them to the respective parts
                            call_meta = return_meta = {}
                            has_tool_output = isinstance(
                                part,
                                (
                                    ToolOutputAvailablePart,
                                    ToolOutputErrorPart,
                                    ToolOutputDeniedPart,
                                    DynamicToolOutputAvailablePart,
                                    DynamicToolOutputErrorPart,
                                    DynamicToolOutputDeniedPart,
                                ),
                            )

                            if has_tool_output:
                                call_meta, return_meta = cls._load_builtin_tool_meta(provider_meta)

                            # `tool_kind` comes from client-supplied metadata, so each claim is validated
                            # to a known `ToolPartKind` (else dropped) before being set on the base part;
                            # the final `narrow_message_parts` pass then promotes it best-effort. A claim
                            # whose data doesn't validate leaves a base part; only core's kinds are stripped.
                            raw_call_tool_kind = call_meta.get('tool_kind')
                            call_tool_kind = (
                                parse_tool_kind(raw_call_tool_kind) if isinstance(raw_call_tool_kind, str) else None
                            )
                            raw_return_tool_kind = return_meta.get('tool_kind')
                            return_tool_kind = (
                                parse_tool_kind(raw_return_tool_kind) if isinstance(raw_return_tool_kind, str) else None
                            )
                            builder.add(
                                NativeToolCallPart(
                                    tool_name=tool_name,
                                    tool_call_id=tool_call_id,
                                    args=args,
                                    id=call_meta.get('id') or part_id,
                                    provider_name=call_meta.get('provider_name') or provider_name,
                                    provider_details=call_meta.get('provider_details') or provider_details,
                                    tool_kind=call_tool_kind or tool_kind,
                                )
                            )

                            if has_tool_output:
                                if isinstance(part, ToolOutputErrorPart | DynamicToolOutputErrorPart):
                                    output: Any = part.error_text
                                    outcome: Literal['success', 'failed', 'denied'] = 'failed'
                                elif isinstance(part, ToolOutputDeniedPart | DynamicToolOutputDeniedPart):
                                    output = _denial_reason(part)
                                    outcome = 'denied'
                                else:
                                    raw_output = (
                                        part.output
                                        if isinstance(part, ToolOutputAvailablePart | DynamicToolOutputAvailablePart)
                                        else None
                                    )
                                    output = _validate_tool_output(raw_output)
                                    outcome = 'success'
                                builder.add(
                                    NativeToolReturnPart(
                                        tool_name=tool_name,
                                        tool_call_id=tool_call_id,
                                        content=output,
                                        provider_name=return_meta.get('provider_name') or provider_name,
                                        provider_details=return_meta.get('provider_details') or provider_details,
                                        outcome=outcome,
                                        # Kept on every outcome: a return that isn't a success is never
                                        # promoted to its typed subclass, so the kind can't pass for success.
                                        tool_kind=return_tool_kind or tool_kind,
                                    )
                                )
                        else:
                            builder.add(
                                ToolCallPart(
                                    tool_name=tool_name,
                                    tool_call_id=tool_call_id,
                                    args=args,
                                    id=part_id,
                                    provider_name=provider_name,
                                    provider_details=provider_details,
                                    tool_kind=tool_kind,
                                )
                            )

                            if part.state == 'output-available':
                                # A synthesized interrupted return dumps as neutral `output-available`
                                # with an `'interrupted'` outcome claim in the metadata channel; restore
                                # it so a round-trip doesn't upgrade the outcome to `'success'`. Every
                                # return keeps the call's `tool_kind`: one that isn't a success is never
                                # promoted to its typed subclass, so the kind can't pass for success.
                                interrupted = provider_meta.get('outcome') == 'interrupted'
                                pending_returns.append(
                                    ToolReturnPart(
                                        tool_name=tool_name,
                                        tool_call_id=tool_call_id,
                                        content=_validate_tool_output(part.output),
                                        outcome='interrupted' if interrupted else 'success',
                                        tool_kind=tool_kind,
                                    )
                                )
                            elif part.state == 'output-error':
                                # `'failed'` and `'retried'` share `output-error`, so the outcome
                                # claim in the metadata channel is what tells a retry apart from a
                                # definitive failure; without it the return reloads as `'failed'`.
                                retried = provider_meta.get('outcome') == 'retried'
                                structured = provider_meta.get('content') if retried else None
                                pending_returns.append(
                                    ToolReturnPart(
                                        tool_name=tool_name,
                                        tool_call_id=tool_call_id,
                                        # `error_text` is the rendered string; a retry's validation errors
                                        # ride the metadata channel as the list they are, so the
                                        # provider is sent the same structure after a round-trip.
                                        content=_validate_tool_output(structured)
                                        if isinstance(structured, list)
                                        else part.error_text,
                                        outcome='retried' if retried else 'failed',
                                        retry=parse_tool_retry(provider_meta.get('retry')) if retried else None,
                                        tool_kind=tool_kind,
                                    )
                                )
                            elif part.state == 'output-denied':
                                pending_returns.append(
                                    ToolReturnPart(
                                        tool_name=tool_name,
                                        tool_call_id=tool_call_id,
                                        content=_denial_reason(part),
                                        outcome='denied',
                                        tool_kind=tool_kind,
                                    )
                                )
                    elif isinstance(part, DataUIPart):  # pragma: no cover
                        # Contains custom data that shouldn't be sent to the model
                        pass
                    elif isinstance(part, SourceUrlUIPart):  # pragma: no cover
                        # TODO: Once we support citations: https://github.com/pydantic/pydantic-ai/issues/3126
                        pass
                    elif isinstance(part, SourceDocumentUIPart):  # pragma: no cover
                        # TODO: Once we support citations: https://github.com/pydantic/pydantic-ai/issues/3126
                        pass
                    elif isinstance(part, StepStartUIPart):
                        # Step boundaries are handled above, by flushing the held-back tool returns
                        pass
                    else:
                        assert_never(part)
                for return_part in pending_returns:
                    builder.add(return_part)
                builder._append_whole(pending_enqueued)  # pyright: ignore[reportPrivateUsage]
            else:
                assert_never(msg.role)

            # Apply metadata and the id to the role-corresponding `ModelMessage`: assistant UIMessages
            # may also append a synthetic `ModelRequest` carrying tool-return parts, which we skip
            # via the type filter so they land on the response, not the tool returns.
            target_type = ModelResponse if msg.role == 'assistant' else ModelRequest
            if (target := _metadata_target(builder, checkpoint, target_type, enqueued)) is not None:
                apply_message_metadata(target, msg.metadata)
                set_ui_message_id(target, msg.id)

        # Parts above are built as base `ToolCallPart`/`ToolReturnPart`/`NativeTool*Part` carrying a
        # `tool_kind` claim; promote them to their typed subclasses in one best-effort pass.
        return narrow_message_parts(builder.messages)

    @staticmethod
    def _dump_builtin_tool_meta(
        call_provider_metadata: ProviderMetadata | None, return_provider_metadata: ProviderMetadata | None
    ) -> ProviderMetadata | None:
        """Use special keys (call_meta and return_meta) to dump combined provider metadata."""
        return dump_provider_metadata(call_meta=call_provider_metadata, return_meta=return_provider_metadata)

    @staticmethod
    def _load_builtin_tool_meta(
        provider_metadata: ProviderMetadata,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Use special keys (call_meta and return_meta) to load combined provider metadata.

        `call_provider_metadata` is client-controlled, so a forged non-dict `call_meta`/`return_meta`
        reads as an empty dict rather than crashing the downstream `.get(...)` lookups.
        """
        call_meta = provider_metadata.get('call_meta')
        return_meta = provider_metadata.get('return_meta')
        return (
            call_meta if isinstance(call_meta, dict) else {},
            return_meta if isinstance(return_meta, dict) else {},
        )

    @staticmethod
    def _dump_request_message(
        msg: ModelRequest,
    ) -> list[tuple[Literal['system', 'user'], list[UIMessagePart]]]:
        """Convert a ModelRequest into its UI message segments, one per run of same-role parts, in authored order.

        Keeping the order, rather than putting every system part first, is what lets a request whose user
        prompt precedes system-voice retry feedback load back the same way, so it re-sends the same bytes.
        """
        segments: list[tuple[Literal['system', 'user'], list[UIMessagePart]]] = []

        def add(role: Literal['system', 'user'], *ui_parts: UIMessagePart) -> None:
            if segments and segments[-1][0] == role:
                segments[-1][1].extend(ui_parts)
            elif ui_parts:
                # A part that dumps to nothing, like an empty user prompt, opens no segment of its own.
                segments.append((role, list(ui_parts)))

        seen_user_prompt = False

        for part in msg.parts:
            if isinstance(part, SystemPromptPart):
                # A request-scoped prompt is for the model's current request only; loaded back, it would
                # come back as a lasting system prompt.
                if part.scope != 'request':
                    add('system', TextUIPart(text=part.content, state='done'))
            elif isinstance(part, UserPromptPart):
                add('user', *_convert_user_prompt_part(part, starts_new_prompt=seen_user_prompt))
                seen_user_prompt = True
            elif isinstance(part, ToolReturnPart):
                # Tool returns are merged into the tool call in the assistant message
                pass
            elif isinstance(part, (InstructionDeltaPart, InstructionBaselinePart)):
                pass  # Operator instruction state stays in trusted server-side history.
            elif isinstance(part, ToolAvailabilityDeltaPart):
                add(
                    'user',
                    DataUIPart(
                        type=TOOL_AVAILABILITY_DELTA_DATA_TYPE,
                        data={
                            'added': part.tools_added,
                            'tool_call_id': part.tool_call_id,
                        },
                    ),
                )
            # TODO(v3): remove `RetryPromptPart`
            elif isinstance(part, RetryPromptPart | RetryFeedbackPart):  # pyright: ignore[reportDeprecated]
                translated = (
                    _translate_legacy_retry_part(part)
                    if isinstance(part, RetryPromptPart)  # pyright: ignore[reportDeprecated]
                    else part
                )
                if isinstance(translated, ToolReturnPart):
                    # Tool-related retries are handled when processing ToolCallPart in ModelResponse
                    pass
                else:
                    # Harness feedback dumps in the voice the model is shown it in, so a transcript
                    # can't read it as something a person wrote
                    # (https://github.com/pydantic/pydantic-ai/issues/6404). The part itself rides in
                    # `provider_metadata` either way, so the reload reconstructs it instead of the
                    # plain part the rendered text alone would produce.
                    feedback_ui_part = TextUIPart(
                        text=translated.model_response(),
                        state='done',
                        provider_metadata=dump_provider_metadata(retry_feedback=retry_feedback_payload(translated)),
                    )
                    add('system' if _retry_feedback_speaks_for_the_harness(translated) else 'user', feedback_ui_part)
            elif isinstance(part, SpeechPart):  # pragma: no cover
                pass  # Realtime audio parts are not rendered in the UI
            else:
                assert_never(part)

        return segments

    @classmethod
    def _dump_response_message(
        cls,
        msg: ModelResponse,
        tool_results: dict[str, ToolReturnPart],
        sdk_version: Literal[5, 6, 7] = 5,
    ) -> list[UIMessagePart]:
        """Convert a ModelResponse into a UIMessage."""
        ui_parts: list[UIMessagePart] = []

        # For builtin tools, returns can be in the same ModelResponse as calls
        local_builtin_returns: dict[str, NativeToolReturnPart] = {
            part.tool_call_id: part for part in msg.parts if isinstance(part, NativeToolReturnPart)
        }

        for part in msg.parts:
            if isinstance(part, NativeToolReturnPart):
                continue
            elif isinstance(part, TextPart):
                # Combine consecutive text parts
                if ui_parts and isinstance(ui_parts[-1], TextUIPart):
                    ui_parts[-1].text += part.content
                else:
                    provider_metadata = dump_provider_metadata(
                        id=part.id,
                        provider_name=part.provider_name,
                        provider_details=part.provider_details,
                    )
                    ui_parts.append(TextUIPart(text=part.content, state='done', provider_metadata=provider_metadata))
            elif isinstance(part, ThinkingPart):
                provider_metadata = dump_provider_metadata(
                    id=part.id,
                    signature=part.signature,
                    provider_name=part.provider_name,
                    provider_details=part.provider_details,
                )
                ui_parts.append(ReasoningUIPart(text=part.content, state='done', provider_metadata=provider_metadata))
            elif isinstance(part, FilePart):
                ui_parts.append(
                    FileUIPart(
                        url=part.content.data_uri,
                        media_type=part.content.media_type,
                        provider_metadata=dump_provider_metadata(
                            id=part.id,
                            provider_name=part.provider_name,
                            provider_details=part.provider_details,
                            vendor_metadata=part.content.vendor_metadata,
                        ),
                    )
                )
            elif isinstance(part, NativeToolCallPart):
                tool_name = f'tool-{part.tool_name}'
                if builtin_return := local_builtin_returns.get(part.tool_call_id):
                    # Builtin tool calls are represented by two parts in pydantic_ai:
                    #   1. NativeToolCallPart (the tool request) -> part
                    #   2. NativeToolReturnPart (the tool's output) -> builtin_return
                    # The Vercel AI SDK only has a single ToolOutputPart (ToolOutputAvailablePart or ToolOutputErrorPart).
                    # So, we need to combine the metadata so that when we later convert back from Vercel AI to pydantic_ai,
                    # we can properly reconstruct both the call and return parts with their respective metadata.
                    # Note: This extra metadata handling is only needed for built-in tools, since normal tool returns
                    # (ToolReturnPart) do not include provider metadata.

                    call_meta = dump_provider_metadata(
                        wrapper_key=None,
                        id=part.id,
                        provider_name=part.provider_name,
                        provider_details=part.provider_details,
                        tool_kind=part.tool_kind,
                    )
                    return_meta = dump_provider_metadata(
                        wrapper_key=None,
                        provider_name=builtin_return.provider_name,
                        provider_details=builtin_return.provider_details,
                        tool_kind=builtin_return.tool_kind,
                    )
                    combined_provider_meta = cls._dump_builtin_tool_meta(call_meta, return_meta)

                    if builtin_return.outcome == 'denied':
                        ui_parts.append(
                            ToolOutputDeniedPart(
                                type=tool_name,
                                tool_call_id=part.tool_call_id,
                                input=part.args_as_dict(),
                                provider_executed=True,
                                call_provider_metadata=combined_provider_meta,
                                approval=ToolApprovalResponded(
                                    id=str(uuid.uuid4()),
                                    approved=False,
                                    reason=builtin_return.model_response_str(wrap_if_error=False),
                                ),
                            )
                        )
                    elif (
                        builtin_return.outcome in ERROR_OUTCOMES
                        or builtin_return.model_response_object(wrap_if_error=False).get('is_error') is True
                    ):
                        response_obj = builtin_return.model_response_object(wrap_if_error=False)
                        error_text = response_obj.get(
                            'error_text', builtin_return.model_response_str(wrap_if_error=False)
                        )
                        ui_parts.append(
                            ToolOutputErrorPart(
                                type=tool_name,
                                tool_call_id=part.tool_call_id,
                                input=part.args_as_dict(),
                                error_text=error_text,
                                provider_executed=True,
                                call_provider_metadata=combined_provider_meta,
                            )
                        )
                    else:
                        # `'success'` and `'interrupted'` both render as neutral tool output; only
                        # the error outcomes are errors, so `'interrupted'` is never surfaced as one.
                        ui_parts.append(
                            ToolOutputAvailablePart(
                                type=tool_name,
                                tool_call_id=part.tool_call_id,
                                input=part.args_as_dict(),
                                output=tool_return_output(builtin_return),
                                provider_executed=True,
                                call_provider_metadata=combined_provider_meta,
                            )
                        )
                else:
                    call_provider_metadata = dump_provider_metadata(
                        id=part.id,
                        provider_name=part.provider_name,
                        provider_details=part.provider_details,
                        tool_kind=part.tool_kind,
                    )
                    # No result found → the tool call is deferred (awaiting approval or external result).
                    # On v6, emit `approval-requested` so the frontend can render approve/reject buttons on reload.
                    # On v5, fall back to `input-available` since approval states are v6-only.
                    # `approval.id` is not used for matching (tool_call_id is the match key),
                    # so we use tool_call_id for a stable, deterministic value in dump output.
                    if sdk_version >= 6:
                        ui_parts.append(
                            ToolApprovalRequestedPart(
                                type=tool_name,
                                tool_call_id=part.tool_call_id,
                                input=part.args_as_dict(),
                                provider_executed=True,
                                call_provider_metadata=call_provider_metadata,
                                approval=ToolApprovalRequested(id=part.tool_call_id),
                            )
                        )
                    else:
                        ui_parts.append(
                            ToolInputAvailablePart(
                                type=tool_name,
                                tool_call_id=part.tool_call_id,
                                input=part.args_as_dict(),
                                provider_executed=True,
                                call_provider_metadata=call_provider_metadata,
                            )
                        )
            elif isinstance(part, ToolCallPart):
                ui_parts.extend(cls._dump_tool_call_part(part, tool_results, sdk_version))
            elif isinstance(part, CompactionPart):
                ui_parts.append(
                    DataUIPart(
                        type=COMPACTION_DATA_TYPE,
                        data=compaction_payload(part),
                    )
                )
            elif isinstance(part, SpeechPart):  # pragma: no cover
                pass  # Realtime audio parts are not rendered in the UI
            else:
                assert_never(part)

        return ui_parts

    @staticmethod
    def _dump_tool_call_part(
        part: ToolCallPart,
        tool_results: dict[str, ToolReturnPart],
        sdk_version: Literal[5, 6, 7] = 5,
    ) -> list[UIMessagePart]:
        """Convert a ToolCallPart (with optional result) into UIMessageParts."""
        tool_result = tool_results.get(part.tool_call_id)
        # The two outcomes the UI part state can't represent on its own: `'interrupted'` dumps as
        # neutral `output-available` below, and `'retried'` shares `output-error` with `'failed'`.
        # Both ride the metadata channel so a dump/load round-trip doesn't degrade them to
        # `'success'` / `'failed'`, which would change how the return serializes to the provider.
        carried_outcome = (
            tool_result.outcome
            if isinstance(tool_result, ToolReturnPart) and tool_result.outcome in ('interrupted', 'retried')
            else None
        )
        call_provider_metadata = dump_provider_metadata(
            id=part.id,
            provider_name=part.provider_name,
            provider_details=part.provider_details,
            tool_kind=part.tool_kind,
            outcome=carried_outcome,
            retry=tool_retry_payload(tool_result.retry)
            if isinstance(tool_result, ToolReturnPart) and tool_result.retry is not None
            else None,
            content=_retried_structured_content(tool_result),
        )
        tool_type = f'tool-{part.tool_name}'
        ui_parts: list[UIMessagePart] = []

        if isinstance(tool_result, ToolReturnPart):
            if tool_result.outcome == 'denied':
                ui_parts.append(
                    ToolOutputDeniedPart(
                        type=tool_type,
                        tool_call_id=part.tool_call_id,
                        input=part.args_as_dict(),
                        provider_executed=False,
                        call_provider_metadata=call_provider_metadata,
                        approval=ToolApprovalResponded(
                            id=str(uuid.uuid4()),
                            approved=False,
                            reason=tool_result.model_response_str(wrap_if_error=False),
                        ),
                    )
                )
            elif tool_result.outcome in ERROR_OUTCOMES:
                # `error_text` is the return's own content, with no wrapper and no instruction
                # framing: a retry's feedback reaches the frontend as the feedback itself
                # (https://github.com/pydantic/pydantic-ai/pull/4869).
                ui_parts.append(
                    ToolOutputErrorPart(
                        type=tool_type,
                        tool_call_id=part.tool_call_id,
                        input=part.args_as_dict(),
                        error_text=tool_result.model_response_str(wrap_if_error=False),
                        provider_executed=False,
                        call_provider_metadata=call_provider_metadata,
                    )
                )
            else:
                # `'success'` and `'interrupted'` both render as neutral tool output; only the
                # error outcomes are errors, so a synthesized `'interrupted'` return (from
                # message-history repair) shows its interruption message as the output rather
                # than an error.
                ui_parts.append(
                    ToolOutputAvailablePart(
                        type=tool_type,
                        tool_call_id=part.tool_call_id,
                        input=part.args_as_dict(),
                        output=tool_return_output(tool_result),
                        provider_executed=False,
                        call_provider_metadata=call_provider_metadata,
                    )
                )
            # Check for Vercel AI chunks returned by tool calls via metadata.
            ui_parts.extend(_extract_metadata_ui_parts(tool_result))
        else:
            # No result found → the tool call is deferred (awaiting approval or external result).
            # On v6, emit `approval-requested` so the frontend can render approve/reject buttons on reload.
            # On v5, fall back to `input-available` since approval states are v6-only.
            # `approval.id` is not used for matching (tool_call_id is the match key),
            # so we use tool_call_id for a stable, deterministic value in dump output.
            if sdk_version >= 6:
                ui_parts.append(
                    ToolApprovalRequestedPart(
                        type=tool_type,
                        tool_call_id=part.tool_call_id,
                        input=part.args_as_dict(),
                        provider_executed=False,
                        call_provider_metadata=call_provider_metadata,
                        approval=ToolApprovalRequested(id=part.tool_call_id),
                    )
                )
            else:
                ui_parts.append(
                    ToolInputAvailablePart(
                        type=tool_type,
                        tool_call_id=part.tool_call_id,
                        input=part.args_as_dict(),
                        provider_executed=False,
                        call_provider_metadata=call_provider_metadata,
                    )
                )

        return ui_parts

    @classmethod
    def dump_messages(
        cls,
        messages: Sequence[ModelMessage],
        *,
        generate_message_id: Callable[[ModelRequest | ModelResponse, Literal['system', 'user', 'assistant'], int], str]
        | None = None,
        sdk_version: Literal[5, 6, 7] = 5,
    ) -> list[UIMessage]:
        """Transform Pydantic AI messages into Vercel AI messages.

        Note: The round-trip `dump_messages` -> `load_messages` is not fully lossless for tool
        results. Successful, failed, and denied results each round-trip via their own part type
        (`ToolOutputAvailablePart` / `ToolOutputErrorPart` / `ToolOutputDeniedPart`). A retried
        result (`outcome='retried'`) shares `ToolOutputErrorPart` with a failed one — the protocol
        has no separate retry concept — so its outcome rides the `providerMetadata` channel that
        already carries `'interrupted'`, and is restored from there on reload, along with its
        validation errors as the list they are, since `errorText` can only hold their rendering. A legacy
        `RetryPromptPart` is dumped as the part it always meant, so a tool-bound one takes that same
        path and reloads as a `ToolReturnPart` with `outcome='retried'`.

        A `RetryFeedbackPart` dumps as a message in the voice the model is shown it in — `role='user'`
        for a `'validation_error'` or `'model_retry'`, `role='system'` for `'no_output'` feedback in
        the framework's own words — with the part itself in that part's
        `providerMetadata`; it reloads as a `RetryFeedbackPart` only from there, so a message the
        client wrote stays a plain prompt part.

        A `ModelRequest` mixing system-voice parts (`SystemPromptPart`, system-voice
        `RetryFeedbackPart`) with user content goes out as one `UIMessage` per run of same-role parts,
        in the order they were authored, so it reloads as the same parts in the same order and
        re-sends the same request bytes. The request's metadata rides on the last `role='user'` one, or
        on the last one when there is none, so a frontend that drops system messages keeps it.

        Tool calls lose one thing too: `ToolCallPart.args` that don't parse as a JSON object are
        rewritten to `{'INVALID_JSON': '<raw args>'}` (see
        [`args_as_dict`][pydantic_ai.messages.BaseToolCallPart.args_as_dict]), so the raw string is
        no longer recoverable as args on reload.

        Application keys in `ModelRequest.metadata` round-trip, but the reserved `__pydantic_ai__`
        namespace and top-level `ModelResponse.provider_details` do not. `UIMessage.metadata` is
        client-controlled, so framework and provider response state is not exposed or restored through
        it; see `_PydanticAIMessageMetadata`. The `UIMessage.id` that `load_messages` keeps in that
        namespace is the exception: the default `generate_message_id` restores it.

        When `sdk_version=6`, tool calls that have no corresponding result in the message history
        are automatically detected as deferred and emitted with `state='approval-requested'`, so the
        frontend can render approve/reject buttons on reload. On v5, such tool calls are emitted
        with `state='input-available'` (approval states are v6-only).

        Args:
            messages: A sequence of ModelMessage objects to convert
            generate_message_id: Optional custom function to generate message IDs. If provided,
                it receives the message, the role ('system', 'user', or 'assistant'), and the
                message index (incremented per UIMessage appended), and should return a unique
                string ID. If not provided, uses `provider_response_id` for responses,
                run_id-based IDs for messages with run_id, or a deterministic UUID5 fallback.
            sdk_version: Vercel AI SDK version to target: 5, 6, or 7. Defaults to 5 for backwards
                compatibility. Set to 6 to emit tool approval parts for deferred tool calls; 7 emits
                identically to 6 (v7's data-stream protocol equals v6's).

        Returns:
            A list of UIMessage objects in Vercel AI format
        """
        tool_results: dict[str, ToolReturnPart] = {}

        for msg in messages:
            if isinstance(msg, ModelRequest):
                for part in msg.parts:
                    if isinstance(part, ToolReturnPart):
                        tool_results[part.tool_call_id] = part
                    # TODO(v3): remove `RetryPromptPart`
                    elif isinstance(part, RetryPromptPart) and isinstance(  # pyright: ignore[reportDeprecated]
                        translated := _translate_legacy_retry_part(part), ToolReturnPart
                    ):
                        tool_results[part.tool_call_id] = translated

        id_generator = generate_message_id or _generate_message_id
        result: list[UIMessage] = []
        message_index = 0

        for msg in messages:
            if isinstance(msg, ModelRequest):
                segments = cls._dump_request_message(msg)
                # Metadata goes on one UIMessage of a split request, so reload applies it once to the
                # merged ModelRequest: the last user one, which a frontend that drops system messages
                # still keeps, or the last one when there is no user message.
                request_metadata = dump_message_metadata(msg)
                user_positions = [position for position, (role, _) in enumerate(segments) if role == 'user']
                metadata_position = user_positions[-1] if user_positions else len(segments) - 1
                for position, (role, ui_parts) in enumerate(segments):
                    ui_message = UIMessage(id=id_generator(msg, role, message_index), role=role, parts=ui_parts)
                    if request_metadata and position == metadata_position:
                        ui_message.metadata = request_metadata
                    result.append(ui_message)
                    message_index += 1

            elif isinstance(  # pragma: no branch
                msg, ModelResponse
            ):
                ui_parts: list[UIMessagePart] = cls._dump_response_message(msg, tool_results, sdk_version)
                if ui_parts:  # pragma: no branch
                    result.append(
                        UIMessage(
                            id=id_generator(msg, 'assistant', message_index),
                            role='assistant',
                            metadata=dump_message_metadata(msg),
                            parts=ui_parts,
                        )
                    )
                    message_index += 1
            else:
                assert_never(msg)

        return result


def _convert_user_prompt_part(
    part: UserPromptPart, *, starts_new_prompt: bool = False
) -> list[TextUIPart | FileUIPart]:
    """Convert a UserPromptPart to a list of UI message parts.

    All of a request's prompts go into one user `UIMessage`, so with `starts_new_prompt` the first UI part is
    marked as where this prompt begins, for `load_messages` to rebuild the same `UserPromptPart`s.
    """
    ui_parts: list[TextUIPart | FileUIPart] = []

    if isinstance(part.content, str):
        ui_parts.append(TextUIPart(text=part.content, state='done'))
    else:
        for item in part.content:
            if isinstance(item, str):
                ui_parts.append(TextUIPart(text=item, state='done'))
            elif isinstance(item, TextContent):
                ui_parts.append(TextUIPart(text=item.content, state='done'))
            elif isinstance(item, BinaryContent):
                ui_parts.append(
                    FileUIPart(
                        url=item.data_uri,
                        media_type=item.media_type,
                        # Round-trip vendor_metadata (e.g. OpenAI/xAI image `detail`,
                        # Google `video_metadata`); see `BinaryContent.vendor_metadata`.
                        provider_metadata=dump_provider_metadata(vendor_metadata=item.vendor_metadata),
                    )
                )
            elif isinstance(item, ImageUrl | AudioUrl | VideoUrl | DocumentUrl):
                try:
                    media_type = item.media_type
                except ValueError:
                    media_type = ''
                ui_parts.append(
                    FileUIPart(
                        url=item.url,
                        media_type=media_type,
                        # Round-trip vendor_metadata (e.g. OpenAI/xAI image `detail`,
                        # Google `video_metadata`) and non-default `force_download`; see `FileUrl`.
                        # `kind` only for a URL we could not read a media type out of: the media type
                        # is what the kind is normally recovered from, and `''` recovers nothing.
                        provider_metadata=dump_provider_metadata(
                            force_download=item.force_download or None,
                            vendor_metadata=item.vendor_metadata,
                            kind=None if media_type else item.kind,
                        ),
                    )
                )
            elif isinstance(item, UploadedFile):
                # Store uploaded file info in provider_metadata for round-trip support
                provider_metadata = dump_provider_metadata(
                    file_id=item.file_id,
                    provider_name=item.provider_name,
                    vendor_metadata=item.vendor_metadata,
                    identifier=item._identifier,  # pyright: ignore[reportPrivateUsage]
                )
                ui_parts.append(
                    FileUIPart(url=item.file_id, media_type=item.media_type, provider_metadata=provider_metadata)
                )
            elif isinstance(item, CachePoint):
                # CachePoint is metadata for prompt caching, skip for UI conversion
                pass
            else:
                assert_never(item)

    if starts_new_prompt and ui_parts:
        first = ui_parts[0]
        provider_metadata = first.provider_metadata or {}
        first.provider_metadata = {
            **provider_metadata,
            PROVIDER_METADATA_KEY: {**provider_metadata.get(PROVIDER_METADATA_KEY, {}), _USER_PROMPT_START_KEY: True},
        }

    return ui_parts


def _user_prompt_part(content: list[UserContent]) -> UserPromptPart:
    """Build the `UserPromptPart` for the content of a user `UIMessage`, a lone text as a plain string."""
    if len(content) == 1 and isinstance(content[0], str):
        return UserPromptPart(content=content[0])
    return UserPromptPart(content=content)


def _denial_reason(part: ToolUIPart | DynamicToolUIPart) -> str:
    """Extract the denial reason from a tool part's approval, or return a default message."""
    if isinstance(part.approval, ToolApprovalResponded) and part.approval.reason:
        return part.approval.reason
    return ToolDenied().message


def _retried_structured_content(tool_result: ToolReturnPart | None) -> list[Any] | None:
    """A retried return's list content, as JSON data for the metadata channel, or `None`.

    `ToolOutputErrorPart.error_text` is a string, and rendering a retry's validation errors into it
    loses the structure a provider without a native error channel is sent: `{"error": [...]}` would
    become `{"error": "[...]"}` after a round-trip. Only list content rides along; a string error is
    already exactly what `error_text` holds.
    """
    content: object = None if tool_result is None or tool_result.outcome != 'retried' else tool_result.content
    if not isinstance(content, list):
        return None
    return tool_return_ta.dump_python(content, mode='json', by_alias=True)


def _validate_tool_output(output: Any) -> Any:
    """Rehydrate `ToolOutputAvailablePart.output` (typed `Any` on the wire) into `ToolReturnContent`.

    `tool_return_content_ta` resolves the `ToolReturnContent` union, so multimodal items
    (`BinaryContent`, `ImageUrl`, etc.) come back as their subclasses instead of raw dicts.
    `BinaryContent` instances with image media types are narrowed to `BinaryImage`. The file shapes a
    browser tool is documented to return are completed first (see `_normalize_client_file_shapes`).
    """
    return tool_return_content_ta.validate_python(_normalize_client_file_shapes(output))


_multi_modal_content_ta: TypeAdapter[MultiModalContent] = TypeAdapter(MultiModalContent)
"""Builds a URL item from the client's own mapping, so `_normalize_client_file_shapes` can read back the
media type the type itself infers rather than repeating that inference here."""


def _normalize_client_file_shapes(value: Any) -> Any:
    """Complete the file shapes a Vercel AI client-side tool is documented to return.

    A [client-side tool](https://ai-sdk.dev/docs/ai-sdk-ui/chatbot-tool-usage) (resolved server-side as
    an external/deferred tool call) returns a file by putting a shape matching one of our multi-modal
    types in its output, and `docs/ui/vercel-ai.md` documents two of those shapes in forms the
    `ToolReturnContent` union cannot take as they stand:

    - `{kind: 'binary', media_type: ..., data: ...}` with the raw bytes a JavaScript frontend produces:
      `JSON.stringify` serializes a `Uint8Array` as `{'0': N, '1': N, ...}` and a Node `Buffer` as
      `{'type': 'Buffer', 'data': [N, ...]}`, both of which pydantic's bytes validator rejects. Base64
      strings pass through untouched, which is what a file the agent itself produced round-trips as.
    - `{kind: 'image-url', url: ...}` and its three siblings with no `media_type`: the union requires
      one of a URL item, so we infer it here the way the type itself would, by building the item and
      reading back the media type it derived from the URL. A URL the type cannot derive one from is
      left as the client sent it: with `media_type` absent or empty it reaches the agent as the
      ordinary mapping it is, and with `null`, the value our own dump writes for such a URL, as a file
      with no media type.

    Everything else is passed through, and a plain user mapping that merely reuses one of our `kind`
    values keeps the values its tool put in it: the binary branch is gated on the `media_type` a real
    `BinaryContent` carries, and the URL branch validates the mapping as it stands, so it writes a
    media type only into a mapping that is already the file it claims to be.
    """
    if isinstance(value, list):
        return [_normalize_client_file_shapes(v) for v in value]  # pyright: ignore[reportUnknownVariableType]
    if not isinstance(value, dict):
        return value
    normalized: dict[str, Any] = {k: _normalize_client_file_shapes(v) for k, v in value.items()}  # pyright: ignore[reportUnknownVariableType]
    kind = normalized.get('kind')
    if kind == 'binary' and 'media_type' in normalized:
        normalized['data'] = _js_binary_to_bytes(normalized.get('data'))
    # Absent, `null` or `''` are the three ways a client leaves the media type open — `File.type` is
    # `''` whenever the browser cannot tell — and are what `FileUrl` itself treats as "infer one". A
    # `media_type` of any other shape is the client's own value and is left for validation to judge.
    elif kind in _FILE_URL_KINDS and normalized.get('media_type') in (None, ''):
        try:
            normalized['media_type'] = _multi_modal_content_ta.validate_python(normalized).media_type
        except ValueError:
            # `Could not infer media type`, or a mapping the type rejects — a missing `url` included:
            # leave it as the client sent it, rather than adding a key to something that stays a plain
            # mapping anyway.
            pass
    return normalized


def _js_binary_to_bytes(data: Any) -> Any:
    """Map a JS-serialized `Uint8Array`/`Buffer` shape to `bytes`; pass through other values.

    Any shape that isn't a canonical, in-range byte sequence is passed through unchanged so that
    `tool_return_content_ta` surfaces a clean `ValidationError`, rather than this helper raising
    `KeyError`/`ValueError` on malformed client input.
    """
    if not isinstance(data, dict):
        return data
    mapping: dict[str, Any] = data  # pyright: ignore[reportUnknownVariableType]
    # Node Buffer: `{'type': 'Buffer', 'data': [N, ...]}`
    if mapping.get('type') == 'Buffer':
        buf_data: Any = mapping.get('data')
        if isinstance(buf_data, list) and all(isinstance(b, int) and 0 <= b <= 255 for b in buf_data):  # pyright: ignore[reportUnknownVariableType]
            return bytes(buf_data)  # pyright: ignore[reportUnknownArgumentType]
    # Uint8Array via `JSON.stringify`: `{'0': N, '1': N, ...}`. Require canonical contiguous keys
    # (`'0'..'n-1'`) so non-canonical keys like `'00'` pass through instead of raising `KeyError`.
    if mapping and all(str(i) in mapping for i in range(len(mapping))):
        values: list[Any] = [mapping[str(i)] for i in range(len(mapping))]
        if all(isinstance(v, int) and 0 <= v <= 255 for v in values):
            return bytes(values)
    return data  # pyright: ignore[reportUnknownVariableType]


def _extract_metadata_ui_parts(tool_result: ToolReturnPart) -> list[UIMessagePart]:
    """Convert data-carrying chunks from tool metadata into UIMessageParts.

    Both this dump path and the streaming path use `iter_metadata_chunks`,
    but the streaming path yields raw chunk objects (preserving `transient`
    and other chunk-specific fields) while this path converts to persisted
    `UIMessagePart` equivalents — matching Vercel AI SDK semantics where
    transient data is streamed but not persisted.
    """
    parts: list[UIMessagePart] = []
    for chunk in iter_metadata_chunks(tool_result):
        if isinstance(chunk, DataChunk):
            parts.append(DataUIPart(type=chunk.type, id=chunk.id, data=chunk.data))
        elif isinstance(chunk, SourceUrlChunk):
            parts.append(
                SourceUrlUIPart(
                    source_id=chunk.source_id,
                    url=chunk.url,
                    title=chunk.title,
                    provider_metadata=chunk.provider_metadata,
                )
            )
        elif isinstance(chunk, SourceDocumentChunk):
            parts.append(
                SourceDocumentUIPart(
                    source_id=chunk.source_id,
                    media_type=chunk.media_type,
                    title=chunk.title,
                    filename=chunk.filename,
                    provider_metadata=chunk.provider_metadata,
                )
            )
        elif isinstance(chunk, FileChunk):
            parts.append(FileUIPart(url=chunk.url, media_type=chunk.media_type))
        else:
            assert_never(chunk)
    return parts


def _enqueued_ui_messages(data: Any) -> list[UIMessage]:
    """The `UIMessage`s in a `data-enqueued-messages` payload, or none when the client sent something malformed.

    Enqueued-message parts nested inside are dropped: the payload is the client's to send, and honoring
    them would let it recurse `load_messages` as deeply as it liked.
    """
    raw = data.get('messages') if _is_str_dict(data) else None
    try:
        messages = _ui_messages_ta.validate_python(raw)
    except ValidationError:
        return []
    return [
        message.model_copy(
            update={
                'parts': [
                    part
                    for part in message.parts
                    if not (isinstance(part, DataUIPart) and part.type == ENQUEUED_MESSAGES_DATA_TYPE)
                ]
            }
        )
        for message in messages
    ]


def _metadata_target(
    builder: MessagesBuilder,
    checkpoint: BuilderCheckpoint,
    of_type: type[ModelRequest] | type[ModelResponse],
    enqueued: Sequence[ModelMessage],
) -> ModelMessage | None:
    """The message a `UIMessage`'s id and metadata belong on: `MessagesBuilder.last_modified`, minus enqueued messages.

    Enqueued messages loaded from a `data-enqueued-messages` part are appended whole and carry their own id
    and metadata, so they're never the target, even when one lands before the message's own parts.

    `_modified_since` is private on purpose, so custom adapters don't depend on it yet.
    """
    return next(
        (
            message
            for message in builder._modified_since(checkpoint)  # pyright: ignore[reportPrivateUsage]
            if isinstance(message, of_type) and not any(message is loaded for loaded in enqueued)
        ),
        None,
    )
