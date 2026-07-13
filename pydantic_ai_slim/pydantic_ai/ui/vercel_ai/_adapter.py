"""Vercel AI adapter for handling requests."""

from __future__ import annotations

import json
import uuid
from collections.abc import Callable, Sequence
from dataclasses import KW_ONLY, InitVar, dataclass
from functools import cached_property
from typing import TYPE_CHECKING, Any, Literal, cast

from pydantic import TypeAdapter
from typing_extensions import assert_never

from pydantic_ai._utils import is_str_dict as _is_str_dict

from ... import _instructions
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
    ToolReturnPart,
    UploadedFile,
    UploadedFileProviderName,
    UserContent,
    UserPromptPart,
    VideoUrl,
    narrow_message_parts,
    parse_tool_kind,
    tool_return_content_ta,
)
from ...output import OutputDataT
from ...tools import AgentDepsT, DeferredToolResults, ToolDenied
from .. import MessagesBuilder, UIAdapter
from .._adapter import resolve_allow_uploaded_files
from ._event_stream import VercelAIEventStream
from ._utils import (
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
    from .._event_stream import OnCompleteFunc

__all__ = ['VercelAIAdapter']

request_data_ta: TypeAdapter[RequestData] = TypeAdapter(RequestData)

_MEDIA_PREFIX_TO_URL_TYPE: dict[str, type[ImageUrl | AudioUrl | VideoUrl]] = {
    'image': ImageUrl,
    'video': VideoUrl,
    'audio': AudioUrl,
}


def _generate_message_id(
    msg: ModelRequest | ModelResponse, role: Literal['system', 'user', 'assistant'], message_index: int
) -> str:
    """Generate a deterministic message ID based on message content and position.

    Priority order:
    1. For `ModelResponse` with `provider_response_id` set, use '{provider_response_id}-{message_index}'.
    2. For any message with run_id set, use '{run_id}-{message_index}'.
    3. Fallback: UUID5 from 'timestamp-kind-role-message_index'.
    """
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
        manage_system_prompt: Literal['server', 'client'] = 'server',
        allowed_file_url_schemes: frozenset[str] = frozenset({'http', 'https'}),
        allowed_file_url_force_download: frozenset[ForceDownloadMode] = frozenset(),
        allow_uploaded_files: bool = False,
        preserve_file_data: bool | None = None,
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
            manage_system_prompt=manage_system_prompt,
            allowed_file_url_schemes=allowed_file_url_schemes,
            allowed_file_url_force_download=allowed_file_url_force_download,
            allow_uploaded_files=allow_uploaded_files,
            **kwargs,
        )

    def build_event_stream(self) -> UIEventStream[RequestData, BaseChunk, AgentDepsT, OutputDataT]:
        """Build a Vercel AI event stream transformer."""
        return VercelAIEventStream(
            self.run_input,
            accept=self.accept,
            sdk_version=self.sdk_version,
            server_message_id=self.server_message_id,
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

            if msg.role == 'system':
                for part in msg.parts:
                    if isinstance(part, TextUIPart):
                        builder.add(SystemPromptPart(content=part.text))
                    else:  # pragma: no cover
                        raise ValueError(f'Unsupported system message part type: {type(part)}')
            elif msg.role == 'user':
                user_prompt_content: str | list[UserContent] = []
                for part in msg.parts:
                    if isinstance(part, TextUIPart):
                        user_prompt_content.append(part.text)
                    elif isinstance(part, FileUIPart):
                        provider_meta = load_provider_metadata(part.provider_metadata)
                        # Restoring client-supplied `vendor_metadata` is intentional (as the `UploadedFile` branch
                        # already does, #5571/#5772): it carries only the requester's own request params and is
                        # dict-validated by the constructors below.
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
                                url_type = _MEDIA_PREFIX_TO_URL_TYPE.get(part.media_type.split('/', 1)[0], DocumentUrl)
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
                        # Contains custom data that shouldn't be sent to the model
                        pass
                    else:  # pragma: no cover
                        raise ValueError(f'Unsupported user message part type: {type(part)}')

                if user_prompt_content:  # pragma: no branch
                    if len(user_prompt_content) == 1 and isinstance(user_prompt_content[0], str):
                        user_prompt_content = user_prompt_content[0]
                    builder.add(UserPromptPart(content=user_prompt_content))

            elif msg.role == 'assistant':
                for part in msg.parts:
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
                            # the final `narrow_message_parts` pass then promotes it best-effort and
                            # strips any claim whose data doesn't validate against the typed subclass.
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
                                        # As in the non-builtin branch below, error/denied returns carry
                                        # no `tool_kind`: a typed return subclass signals shape-valid
                                        # success to readers like `parse_discovered_tools`.
                                        tool_kind=(return_tool_kind or tool_kind) if outcome == 'success' else None,
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
                                builder.add(
                                    ToolReturnPart(
                                        tool_name=tool_name,
                                        tool_call_id=tool_call_id,
                                        content=_validate_tool_output(part.output),
                                        tool_kind=tool_kind,
                                    )
                                )
                            # Error/denied returns deliberately carry no `tool_kind`: typed return
                            # subclasses only ever wrap successful, shape-valid content, and readers
                            # like `parse_loaded_capabilities` treat their presence as proof of success.
                            elif part.state == 'output-error':
                                builder.add(
                                    ToolReturnPart(
                                        tool_name=tool_name,
                                        tool_call_id=tool_call_id,
                                        content=part.error_text,
                                        outcome='failed',
                                    )
                                )
                            elif part.state == 'output-denied':
                                builder.add(
                                    ToolReturnPart(
                                        tool_name=tool_name,
                                        tool_call_id=tool_call_id,
                                        content=_denial_reason(part),
                                        outcome='denied',
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
                    elif isinstance(part, StepStartUIPart):  # pragma: no cover
                        # Nothing to do here
                        pass
                    else:
                        assert_never(part)
            else:
                assert_never(msg.role)

            # Apply metadata to the role-corresponding `ModelMessage`: assistant UIMessages
            # may also append a synthetic `ModelRequest` carrying tool-return parts, which we
            # skip via the type filter so metadata lands on the response, not the tool returns.
            target_type = ModelResponse if msg.role == 'assistant' else ModelRequest
            if (target := builder.last_modified(checkpoint, of_type=target_type)) is not None:
                apply_message_metadata(target, msg.metadata)

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
    def _dump_request_message(msg: ModelRequest) -> tuple[list[UIMessagePart], list[UIMessagePart]]:
        """Convert a ModelRequest into a UIMessage."""
        system_ui_parts: list[UIMessagePart] = []
        user_ui_parts: list[UIMessagePart] = []

        for part in msg.parts:
            if isinstance(part, SystemPromptPart):
                system_ui_parts.append(TextUIPart(text=part.content, state='done'))
            elif isinstance(part, UserPromptPart):
                user_ui_parts.extend(_convert_user_prompt_part(part))
            elif isinstance(part, ToolReturnPart):
                # Tool returns are merged into the tool call in the assistant message
                pass
            elif isinstance(part, RetryPromptPart):
                if part.tool_name:
                    # Tool-related retries are handled when processing ToolCallPart in ModelResponse
                    pass
                else:
                    # Non-tool retries (e.g., output validation errors) become user text
                    user_ui_parts.append(TextUIPart(text=part.model_response(), state='done'))
            else:
                assert_never(part)

        return system_ui_parts, user_ui_parts

    @classmethod
    def _dump_response_message(
        cls,
        msg: ModelResponse,
        tool_results: dict[str, ToolReturnPart | RetryPromptPart],
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
                                    reason=builtin_return.model_response_str(),
                                ),
                            )
                        )
                    elif (
                        builtin_return.outcome == 'failed'
                        or builtin_return.model_response_object().get('is_error') is True
                    ):
                        response_obj = builtin_return.model_response_object()
                        error_text = response_obj.get('error_text', builtin_return.model_response_str())
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
            elif isinstance(part, CompactionPart):  # pragma: no cover
                pass  # Compaction parts are not rendered in the UI
            else:
                assert_never(part)

        return ui_parts

    @staticmethod
    def _dump_tool_call_part(
        part: ToolCallPart,
        tool_results: dict[str, ToolReturnPart | RetryPromptPart],
        sdk_version: Literal[5, 6, 7] = 5,
    ) -> list[UIMessagePart]:
        """Convert a ToolCallPart (with optional result) into UIMessageParts."""
        tool_result = tool_results.get(part.tool_call_id)
        call_provider_metadata = dump_provider_metadata(
            id=part.id,
            provider_name=part.provider_name,
            provider_details=part.provider_details,
            tool_kind=part.tool_kind,
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
                            reason=tool_result.model_response_str(),
                        ),
                    )
                )
            elif tool_result.outcome == 'failed':
                ui_parts.append(
                    ToolOutputErrorPart(
                        type=tool_type,
                        tool_call_id=part.tool_call_id,
                        input=part.args_as_dict(),
                        error_text=tool_result.model_response_str(),
                        provider_executed=False,
                        call_provider_metadata=call_provider_metadata,
                    )
                )
            else:
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
        elif isinstance(tool_result, RetryPromptPart):
            ui_parts.append(
                ToolOutputErrorPart(
                    type=tool_type,
                    tool_call_id=part.tool_call_id,
                    input=part.args_as_dict(),
                    error_text=tool_result.model_response(),
                    provider_executed=False,
                    call_provider_metadata=call_provider_metadata,
                )
            )
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
        tool_results: dict[str, ToolReturnPart | RetryPromptPart] = {}

        for msg in messages:
            if isinstance(msg, ModelRequest):
                for part in msg.parts:
                    if isinstance(part, ToolReturnPart):
                        tool_results[part.tool_call_id] = part
                    elif isinstance(part, RetryPromptPart) and part.tool_name:
                        tool_results[part.tool_call_id] = part

        id_generator = generate_message_id or _generate_message_id
        result: list[UIMessage] = []
        message_index = 0

        for msg in messages:
            if isinstance(msg, ModelRequest):
                system_ui_parts, user_ui_parts = cls._dump_request_message(msg)
                # Metadata only goes on the trailing UIMessage of a split request so reload
                # applies it once to the merged ModelRequest.
                request_metadata = dump_message_metadata(msg) or None
                if system_ui_parts:
                    result.append(
                        UIMessage(
                            id=id_generator(msg, 'system', message_index),
                            role='system',
                            metadata=None if user_ui_parts else request_metadata,
                            parts=system_ui_parts,
                        )
                    )
                    message_index += 1

                if user_ui_parts:
                    result.append(
                        UIMessage(
                            id=id_generator(msg, 'user', message_index),
                            role='user',
                            metadata=request_metadata,
                            parts=user_ui_parts,
                        )
                    )
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


def _convert_user_prompt_part(part: UserPromptPart) -> list[UIMessagePart]:
    """Convert a UserPromptPart to a list of UI message parts."""
    ui_parts: list[UIMessagePart] = []

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
                ui_parts.append(
                    FileUIPart(
                        url=item.url,
                        media_type=item.media_type,
                        # Round-trip vendor_metadata (e.g. OpenAI/xAI image `detail`,
                        # Google `video_metadata`) and non-default `force_download`; see `FileUrl`.
                        provider_metadata=dump_provider_metadata(
                            force_download=item.force_download or None,
                            vendor_metadata=item.vendor_metadata,
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

    return ui_parts


def _denial_reason(part: ToolUIPart | DynamicToolUIPart) -> str:
    """Extract the denial reason from a tool part's approval, or return a default message."""
    if isinstance(part.approval, ToolApprovalResponded) and part.approval.reason:
        return part.approval.reason
    return ToolDenied().message


def _validate_tool_output(output: Any) -> Any:
    """Rehydrate `ToolOutputAvailablePart.output` (typed `Any` on the wire) into `ToolReturnContent`.

    `tool_return_content_ta` runs the lifted `Discriminator` on the union, so multimodal items
    (`BinaryContent`, `ImageUrl`, etc.) come back as their subclasses instead of raw dicts.
    `BinaryContent` instances with image media types are narrowed to `BinaryImage`. JS-serialized
    binary shapes are coerced to `bytes` first (see `_coerce_js_binary_data`).
    """
    return tool_return_content_ta.validate_python(_coerce_js_binary_data(output))


def _coerce_js_binary_data(value: Any) -> Any:
    """Convert `BinaryContent.data` shapes that JavaScript frontends commonly emit into `bytes`.

    This is what lets a Vercel AI [client-side tool](https://ai-sdk.dev/docs/ai-sdk-ui/chatbot-tool-usage)
    (resolved server-side as an external/deferred tool call) return a file — an image, say — by putting a
    `{kind: 'binary', media_type: ..., data: ...}` shape in its output, without base64-encoding the bytes
    by hand. `JSON.stringify` serializes a `Uint8Array` as `{'0': N, '1': N, ...}` and a Node `Buffer` as
    `{'type': 'Buffer', 'data': [N, ...]}`; pydantic's bytes validator rejects both, so we normalize them
    (and pass base64 strings through untouched) at the wire boundary before validation. A file the agent
    itself produced round-trips as base64 and never hits these shapes.
    """
    if isinstance(value, list):
        return [_coerce_js_binary_data(v) for v in value]  # pyright: ignore[reportUnknownVariableType]
    if not isinstance(value, dict):
        return value
    coerced: dict[str, Any] = {k: _coerce_js_binary_data(v) for k, v in value.items()}  # pyright: ignore[reportUnknownVariableType]
    # Gate on `media_type` (the type-specific field a real `BinaryContent` carries) so this matches
    # the core `ToolReturnContent` discriminator: a plain user mapping that merely reuses
    # `kind: 'binary'` stays untouched instead of having its `data` rewritten to bytes.
    if coerced.get('kind') == 'binary' and 'media_type' in coerced:
        coerced['data'] = _js_binary_to_bytes(coerced.get('data'))
    return coerced


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
