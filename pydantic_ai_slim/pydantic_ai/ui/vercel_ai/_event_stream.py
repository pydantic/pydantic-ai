"""Vercel AI event stream implementation."""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping
from dataclasses import KW_ONLY, dataclass, field
from typing import Any, Literal

from pydantic_core import to_json

from ...messages import (
    BaseToolReturnPart,
    FilePart,
    FinishReason as PydanticFinishReason,
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    NativeToolCallPart,
    NativeToolReturnPart,
    OutputToolCallEvent,
    OutputToolResultEvent,
    RetryPromptPart,
    TextPart,
    TextPartDelta,
    ThinkingPart,
    ThinkingPartDelta,
    ToolCallEvent,
    ToolCallPart,
    ToolCallPartDelta,
    ToolReturnPart,
)
from ...output import OutputDataT
from ...run import AgentRunResultEvent
from ...tools import AgentDepsT, DeferredToolRequests
from .. import UIEventStream
from .._event_stream import describe_file
from ._utils import dump_provider_metadata, iter_metadata_chunks, tool_return_output
from .request_types import RequestData
from .response_types import (
    BaseChunk,
    DoneChunk,
    ErrorChunk,
    FileChunk,
    FinishChunk,
    FinishReason,
    FinishStepChunk,
    ReasoningDeltaChunk,
    ReasoningEndChunk,
    ReasoningStartChunk,
    StartChunk,
    StartStepChunk,
    TextDeltaChunk,
    TextEndChunk,
    TextStartChunk,
    ToolApprovalRequestChunk,
    ToolInputAvailableChunk,
    ToolInputDeltaChunk,
    ToolInputErrorChunk,
    ToolInputStartChunk,
    ToolOutputAvailableChunk,
    ToolOutputDeniedChunk,
    ToolOutputErrorChunk,
)

# Map Pydantic AI finish reasons to Vercel AI format
_FINISH_REASON_MAP: dict[PydanticFinishReason, FinishReason] = {
    'stop': 'stop',
    'length': 'length',
    'content_filter': 'content-filter',
    'tool_call': 'tool-calls',
    'error': 'error',
}

__all__ = ['VercelAIEventStream']

# See https://ai-sdk.dev/docs/ai-sdk-ui/stream-protocol#data-stream-protocol
VERCEL_AI_DSP_HEADERS = {'x-vercel-ai-ui-message-stream': 'v1'}


def _json_dumps(obj: Any) -> str:
    """Dump an object to JSON string."""
    return to_json(obj).decode('utf-8')


def _tool_return_with_files(part: BaseToolReturnPart) -> Any:
    """Wrap tool_return_output with file descriptions for multimodal tool returns."""
    if file_descriptions := [describe_file(f) for f in part.files]:
        return [part.model_response_object(), *file_descriptions]
    return tool_return_output(part)


@dataclass
class VercelAIEventStream(UIEventStream[RequestData, BaseChunk, AgentDepsT, OutputDataT]):
    """UI event stream transformer for the Vercel AI protocol."""

    _: KW_ONLY
    sdk_version: Literal[5, 6] = 5
    """Vercel AI SDK version to target. Setting to 6 enables tool approval streaming."""
    server_message_id: str | None = None
    """Optional server-generated message ID to include in the `StartChunk`."""

    _step_started: bool = False
    _finish_reason: FinishReason = None
    _invalidated_tool_calls: dict[str, ToolCallPart] = field(default_factory=dict[str, ToolCallPart])
    """Calls whose `tool-input-available` chunk was suppressed because validation failed.

    Keyed by tool call ID; the part carries the raw args and provider metadata that the
    later `tool-input-error` chunk needs to mirror what the suppressed `tool-input-available`
    would have shown. The entry is popped by `_handle_tool_result` when the matching
    `RetryPromptPart` arrives, so `tool-input-error` is emitted there instead of
    `tool-output-error`.
    """
    _streamed_call_parts: dict[str, ToolCallPart] = field(default_factory=dict[str, ToolCallPart])
    """Tool call parts seen at `PartEndEvent` time, kept until `_handle_tool_call` takes over.

    Used by `_handle_tool_result` to backfill `tool-input-available` if the agent raises
    before the call event fires (e.g. output-tool `UnexpectedModelBehavior` with no prior
    `final_result`, where `_agent_graph.py` raises without yielding `OutputToolCallEvent`).
    Without the backfill, both v5 and v6 frontends would transition `input-streaming` ->
    `output-error` with no input announcement in between.
    """

    @property
    def response_headers(self) -> Mapping[str, str] | None:
        return VERCEL_AI_DSP_HEADERS

    def encode_event(self, event: BaseChunk) -> str:
        return f'data: {event.encode(self.sdk_version)}\n\n'

    async def before_stream(self) -> AsyncIterator[BaseChunk]:
        yield StartChunk(message_id=self.server_message_id)

    async def before_response(self) -> AsyncIterator[BaseChunk]:
        if self._step_started:
            yield FinishStepChunk()

        self._step_started = True
        yield StartStepChunk()

    async def after_stream(self) -> AsyncIterator[BaseChunk]:
        yield FinishStepChunk()

        yield FinishChunk(finish_reason=self._finish_reason)
        yield DoneChunk()

    async def handle_run_result(self, event: AgentRunResultEvent) -> AsyncIterator[BaseChunk]:
        pydantic_reason = event.result.response.finish_reason
        if pydantic_reason:
            self._finish_reason = _FINISH_REASON_MAP.get(pydantic_reason, 'other')

        # Emit tool approval requests for deferred approvals (only when sdk_version >= 6)
        output = event.result.output
        if self.sdk_version >= 6 and isinstance(output, DeferredToolRequests):
            for tool_call in output.approvals:
                yield ToolApprovalRequestChunk(
                    approval_id=tool_call.tool_call_id,
                    tool_call_id=tool_call.tool_call_id,
                )
            return
        return
        yield

    async def on_error(self, error: Exception) -> AsyncIterator[BaseChunk]:
        self._finish_reason = 'error'
        yield ErrorChunk(error_text=str(error))

    async def handle_text_start(self, part: TextPart, follows_text: bool = False) -> AsyncIterator[BaseChunk]:
        provider_metadata = dump_provider_metadata(
            id=part.id, provider_name=part.provider_name, provider_details=part.provider_details
        )
        if follows_text:
            message_id = self.message_id
        else:
            message_id = self.new_message_id()
            yield TextStartChunk(id=message_id, provider_metadata=provider_metadata)

        if part.content:
            yield TextDeltaChunk(id=message_id, delta=part.content, provider_metadata=provider_metadata)

    async def handle_text_delta(self, delta: TextPartDelta) -> AsyncIterator[BaseChunk]:
        if delta.content_delta:  # pragma: no branch
            provider_metadata = dump_provider_metadata(
                provider_name=delta.provider_name, provider_details=delta.provider_details
            )
            yield TextDeltaChunk(id=self.message_id, delta=delta.content_delta, provider_metadata=provider_metadata)

    async def handle_text_end(self, part: TextPart, followed_by_text: bool = False) -> AsyncIterator[BaseChunk]:
        if not followed_by_text:
            provider_metadata = dump_provider_metadata(
                id=part.id, provider_name=part.provider_name, provider_details=part.provider_details
            )
            yield TextEndChunk(id=self.message_id, provider_metadata=provider_metadata)

    async def handle_thinking_start(
        self, part: ThinkingPart, follows_thinking: bool = False
    ) -> AsyncIterator[BaseChunk]:
        message_id = self.new_message_id()
        provider_metadata = dump_provider_metadata(
            id=part.id,
            signature=part.signature,
            provider_name=part.provider_name,
            provider_details=part.provider_details,
        )
        yield ReasoningStartChunk(id=message_id, provider_metadata=provider_metadata)
        if part.content:
            yield ReasoningDeltaChunk(id=message_id, delta=part.content, provider_metadata=provider_metadata)

    async def handle_thinking_delta(self, delta: ThinkingPartDelta) -> AsyncIterator[BaseChunk]:
        if delta.content_delta:  # pragma: no branch
            provider_metadata = dump_provider_metadata(
                provider_name=delta.provider_name,
                signature=delta.signature_delta,
                provider_details=delta.provider_details,
            )
            yield ReasoningDeltaChunk(
                id=self.message_id, delta=delta.content_delta, provider_metadata=provider_metadata
            )

    async def handle_thinking_end(
        self, part: ThinkingPart, followed_by_thinking: bool = False
    ) -> AsyncIterator[BaseChunk]:
        provider_metadata = dump_provider_metadata(
            id=part.id,
            signature=part.signature,
            provider_name=part.provider_name,
            provider_details=part.provider_details,
        )
        yield ReasoningEndChunk(id=self.message_id, provider_metadata=provider_metadata)

    def handle_tool_call_start(self, part: ToolCallPart | NativeToolCallPart) -> AsyncIterator[BaseChunk]:
        return self._handle_tool_call_start(part)

    def handle_builtin_tool_call_start(self, part: NativeToolCallPart) -> AsyncIterator[BaseChunk]:
        return self._handle_tool_call_start(part, provider_executed=True)

    async def _handle_tool_call_start(
        self,
        part: ToolCallPart | NativeToolCallPart,
        tool_call_id: str | None = None,
        provider_executed: bool | None = None,
    ) -> AsyncIterator[BaseChunk]:
        tool_call_id = tool_call_id or part.tool_call_id
        yield ToolInputStartChunk(
            tool_call_id=tool_call_id,
            tool_name=part.tool_name,
            provider_executed=provider_executed,
            provider_metadata=dump_provider_metadata(
                id=part.id, provider_name=part.provider_name, provider_details=part.provider_details
            ),
        )
        if part.args:
            yield ToolInputDeltaChunk(tool_call_id=tool_call_id, input_text_delta=part.args_as_json_str())

    async def handle_tool_call_delta(self, delta: ToolCallPartDelta) -> AsyncIterator[BaseChunk]:
        tool_call_id = delta.tool_call_id or ''
        assert tool_call_id, '`ToolCallPartDelta.tool_call_id` must be set'
        yield ToolInputDeltaChunk(
            tool_call_id=tool_call_id,
            input_text_delta=delta.args_delta if isinstance(delta.args_delta, str) else _json_dumps(delta.args_delta),
        )

    async def handle_tool_call_end(self, part: ToolCallPart) -> AsyncIterator[BaseChunk]:
        # Stash the streamed part. `_handle_tool_call` (post-validation) takes over emission
        # in the normal flow and pops the stash. If the agent raises before the call event
        # fires (e.g. output-tool `UnexpectedModelBehavior` with no `final_result`), the
        # stash survives and `_handle_tool_result` uses it to backfill `tool-input-available`
        # before the synthesized `tool-output-error`.
        self._streamed_call_parts[part.tool_call_id] = part
        return
        yield  # pragma: no cover  # mark this as an async generator

    async def handle_function_tool_call(self, event: FunctionToolCallEvent) -> AsyncIterator[BaseChunk]:
        async for chunk in self._handle_tool_call(event):
            yield chunk

    async def handle_output_tool_call(self, event: OutputToolCallEvent) -> AsyncIterator[BaseChunk]:
        async for chunk in self._handle_tool_call(event):
            yield chunk

    async def _handle_tool_call(self, event: ToolCallEvent) -> AsyncIterator[BaseChunk]:
        part = event.part
        # The call event arrived; we own the input-chunk lifecycle from here. Drop the
        # stash so `_handle_tool_result` doesn't double-emit a backfill chunk.
        self._streamed_call_parts.pop(part.tool_call_id, None)

        # `args_valid is None` covers resume of non-`ToolApproved` deferred results
        # (`ToolDenied`, `ModelRetry`, direct return) and the output-tool
        # end-strategy-skipped path. The original `tool-input-available` already fired
        # on the first agent run; re-emitting here would be misleading.
        if event.args_valid is None:
            return

        # SDK v6+ supports `tool-input-error`, so we suppress `tool-input-available` on
        # validation failure and let `_handle_tool_result` emit the dedicated error chunk
        # when the matching `RetryPromptPart` arrives. v5 does not have `tool-input-error`;
        # for v5 we keep the pre-PR behavior of emitting `tool-input-available` regardless
        # of validity (with `tool-output-error` later from the result handler) so the tool
        # call lifecycle stays observable for v5 frontends.
        if event.args_valid is False and self.sdk_version >= 6:
            self._invalidated_tool_calls[part.tool_call_id] = part
            return

        yield ToolInputAvailableChunk(
            tool_call_id=part.tool_call_id,
            tool_name=part.tool_name,
            input=part.args_as_dict(),
            provider_metadata=dump_provider_metadata(
                id=part.id, provider_name=part.provider_name, provider_details=part.provider_details
            ),
        )

    async def handle_builtin_tool_call_end(self, part: NativeToolCallPart) -> AsyncIterator[BaseChunk]:
        yield ToolInputAvailableChunk(
            tool_call_id=part.tool_call_id,
            tool_name=part.tool_name,
            input=part.args_as_dict(),
            provider_executed=True,
            provider_metadata=dump_provider_metadata(
                id=part.id, provider_name=part.provider_name, provider_details=part.provider_details
            ),
        )

    async def handle_builtin_tool_return(self, part: NativeToolReturnPart) -> AsyncIterator[BaseChunk]:
        if self.sdk_version >= 6 and part.outcome == 'denied':
            yield ToolOutputDeniedChunk(tool_call_id=part.tool_call_id)
        elif part.outcome == 'failed':
            yield ToolOutputErrorChunk(tool_call_id=part.tool_call_id, error_text=part.model_response_str())
        else:
            yield ToolOutputAvailableChunk(
                tool_call_id=part.tool_call_id,
                output=_tool_return_with_files(part),
                provider_executed=True,
            )

    async def handle_file(self, part: FilePart) -> AsyncIterator[BaseChunk]:
        file = part.content
        yield FileChunk(url=file.data_uri, media_type=file.media_type)

    async def handle_function_tool_result(self, event: FunctionToolResultEvent) -> AsyncIterator[BaseChunk]:
        async for chunk in self._handle_tool_result(event.part):
            yield chunk

    async def handle_output_tool_result(self, event: OutputToolResultEvent) -> AsyncIterator[BaseChunk]:
        async for chunk in self._handle_tool_result(event.part):
            yield chunk

    async def _handle_tool_result(self, part: ToolReturnPart | RetryPromptPart) -> AsyncIterator[BaseChunk]:
        tool_call_id = part.tool_call_id

        invalidated_part = self._invalidated_tool_calls.pop(tool_call_id, None)
        streamed_part = self._streamed_call_parts.pop(tool_call_id, None)

        # Backfill `tool-input-available` if `_handle_tool_call` never fired for this call —
        # happens when the agent raises before yielding the call event (e.g. output-tool
        # `UnexpectedModelBehavior` with no prior `final_result`). The base class then
        # synthesizes a `ToolReturnPart(outcome='failed')` for the pending call, which
        # arrives here; without the backfill, v5/v6 frontends would transition
        # `input-streaming` -> `output-error` with no input announcement in between.
        # `invalidated_part is not None` means `_handle_tool_call` deliberately suppressed
        # the chunk for the v6 invalidated path — don't backfill in that case.
        if streamed_part is not None and invalidated_part is None:
            yield ToolInputAvailableChunk(
                tool_call_id=tool_call_id,
                tool_name=streamed_part.tool_name,
                input=streamed_part.args_as_dict(),
                provider_metadata=dump_provider_metadata(
                    id=streamed_part.id,
                    provider_name=streamed_part.provider_name,
                    provider_details=streamed_part.provider_details,
                ),
            )

        if self.sdk_version >= 6 and isinstance(part, ToolReturnPart) and part.outcome == 'denied':
            yield ToolOutputDeniedChunk(tool_call_id=tool_call_id)
        elif invalidated_part is not None:
            # The original `tool-input-available` was suppressed because `args_valid=False`.
            # Complete the v6 lifecycle by emitting `tool-input-error` instead of letting the
            # result chunk (success/output-error) fire — the call never actually executed.
            # `error_text` comes from `RetryPromptPart.model_response()` on the normal retry
            # path, or `ToolReturnPart.model_response_str()` on the exhaustive output-strategy
            # skip path (where the status part says e.g. "Output tool not used …").
            yield ToolInputErrorChunk(
                tool_call_id=tool_call_id,
                tool_name=invalidated_part.tool_name,
                input=invalidated_part.args_as_dict(),
                provider_metadata=dump_provider_metadata(
                    id=invalidated_part.id,
                    provider_name=invalidated_part.provider_name,
                    provider_details=invalidated_part.provider_details,
                ),
                error_text=part.model_response() if isinstance(part, RetryPromptPart) else part.model_response_str(),
            )
        elif isinstance(part, RetryPromptPart):
            yield ToolOutputErrorChunk(tool_call_id=tool_call_id, error_text=part.model_response())
        elif isinstance(part, ToolReturnPart) and part.outcome == 'failed':
            yield ToolOutputErrorChunk(tool_call_id=tool_call_id, error_text=part.model_response_str())
        else:
            yield ToolOutputAvailableChunk(tool_call_id=tool_call_id, output=_tool_return_with_files(part))

        # ToolOutputAvailableChunk/ToolOutputErrorChunk.output may hold user parts
        # (e.g. text, images) that Vercel AI does not currently have chunk types for.

        # Check for data-carrying Vercel AI chunks returned by tool calls via metadata.
        # Only data-carrying chunks (DataChunk, SourceUrlChunk, etc.) are yielded;
        # protocol-control chunks are filtered out by iter_metadata_chunks.
        if isinstance(part, ToolReturnPart):
            for chunk in iter_metadata_chunks(part):
                yield chunk
