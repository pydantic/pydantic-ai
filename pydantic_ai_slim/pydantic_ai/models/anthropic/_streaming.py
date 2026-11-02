"""`AnthropicStreamedResponse`, which maps Anthropic stream events to Pydantic AI parts and response metadata."""

from __future__ import annotations as _annotations

from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import TYPE_CHECKING, Any, TypeAlias, cast

from anthropic import AsyncStream
from anthropic.types.beta import (
    BetaAdvisorToolResultBlock,
    BetaBashCodeExecutionToolResultBlock,
    BetaCitationsDelta,
    BetaCodeExecutionToolResultBlock,
    BetaCompactionBlock,
    BetaCompactionContentBlockDelta,
    BetaContentBlock,
    BetaInputJSONDelta,
    BetaMCPToolResultBlock,
    BetaMCPToolUseBlock,
    BetaRawContentBlockDeltaEvent,
    BetaRawContentBlockStartEvent,
    BetaRawContentBlockStopEvent,
    BetaRawMessageDeltaEvent,
    BetaRawMessageStartEvent,
    BetaRawMessageStopEvent,
    BetaRawMessageStreamEvent,
    BetaRedactedThinkingBlock,
    BetaServerToolUseBlock,
    BetaSignatureDelta,
    BetaTextBlock,
    BetaTextCitation,
    BetaTextDelta,
    BetaTextEditorCodeExecutionToolResultBlock,
    BetaThinkingBlock,
    BetaThinkingDelta,
    BetaToolSearchToolResultBlock,
    BetaToolUseBlock,
    BetaWebFetchToolResultBlock,
    BetaWebSearchToolResultBlock,
)

from ... import _utils
from ..._context_size import mark_context_input_tokens
from ...messages import CompactionPart, ModelResponseStreamEvent, NativeToolCallPart, NativeToolSearchCallPart, TextPart
from .. import StreamedResponse
from .._decode_errors import MapStreamDecodeErrors
from ._citations import citations_provider_details, fetched_page, map_citations
from ._errors import map_api_errors
from ._native_tools import (
    finalize_streamed_tool_search_call_part,
    map_advisor_tool_result_block,
    map_bash_code_execution_tool_result_block,
    map_code_execution_tool_result_block,
    map_mcp_server_result_block,
    map_mcp_server_use_block,
    map_server_tool_use_block,
    map_text_editor_code_execution_tool_result_block,
    map_tool_search_tool_result_block,
    map_web_fetch_tool_result_block,
    map_web_search_tool_result_block,
)
from ._response import (
    FINISH_REASON_MAP,
    compaction_provider_details,
    last_pass_input_tokens,
    map_usage,
    report_input_transformations,
)

if TYPE_CHECKING:
    from . import AnthropicModelName

AnthropicEventStream: TypeAlias = _utils.PeekableAsyncStream[
    BetaRawMessageStreamEvent, AsyncStream[BetaRawMessageStreamEvent]
]
"""A streamed response whose first event has been read, so an error that stops it is raised before it's processed."""


@dataclass
class AnthropicStreamedResponse(StreamedResponse):
    """Implementation of `StreamedResponse` for Anthropic models."""

    _model_name: AnthropicModelName
    _response: AnthropicEventStream
    _provider_name: str
    _model_id_namespace: str
    _provider_url: str
    _enabled_server_tool_names: frozenset[str]
    _timestamp: datetime = field(default_factory=_utils.now_utc)
    _history_fetched_pages: Sequence[tuple[str, str]] = ()

    async def _get_event_iterator(self) -> AsyncIterator[ModelResponseStreamEvent]:  # noqa: C901
        with map_api_errors(self._model_name, self._model_id_namespace):
            current_block: BetaContentBlock | None = None
            ignored_server_tool_use_indices: set[int] = set()
            pending_citations: dict[int, list[BetaTextCitation]] = {}
            # Web Fetch citations name their page only by its text, so the fetched pages are kept to find their URLs.
            fetched_pages: list[tuple[str, str]] = list(self._history_fetched_pages)

            builtin_tool_calls: dict[str, NativeToolCallPart] = {}
            async for event in MapStreamDecodeErrors(self._response, self._model_name):
                if isinstance(event, BetaRawMessageStartEvent):
                    if event.message is None:  # pyright: ignore[reportUnnecessaryComparison]
                        # See `map_usage`: Bedrock emits type-less chunks the SDK constructs
                        # as `BetaRawMessageStartEvent(message=None)`. Skip them entirely so we
                        # don't dereference `event.message.id` / `.container` below.
                        continue
                    self._usage = map_usage(event, self._provider_name, self._provider_url, self._model_name)
                    self.provider_response_id = event.message.id
                    if event.message.container:
                        self.provider_details = self.provider_details or {}
                        self.provider_details['container_id'] = event.message.container.id
                    # A live stream reports dropped thinking blocks here, on the opening message.
                    # The final `message_delta` carries them only when a server-side model fallback
                    # happened mid-stream, and then it replaces this array rather than extending it.
                    if event.message.input_transformations:
                        self.provider_details = self.provider_details or {}
                        self.provider_details['input_transformations'] = report_input_transformations(
                            event.message.input_transformations
                        )
                    # Diagnostics are only reported on the opening message.
                    if event.message.diagnostics is not None:
                        self.provider_details = self.provider_details or {}
                        self.provider_details['cache_diagnostics'] = event.message.diagnostics.model_dump(mode='json')

                elif isinstance(event, BetaRawContentBlockStartEvent):
                    current_block = event.content_block
                    if isinstance(current_block, BetaTextBlock) and current_block.text:
                        for event_ in self._parts_manager.handle_text_delta(
                            vendor_part_id=event.index, content=current_block.text
                        ):
                            yield event_
                    elif isinstance(current_block, BetaThinkingBlock):
                        for event_ in self._parts_manager.handle_thinking_delta(
                            vendor_part_id=event.index,
                            content=current_block.thinking,
                            signature=current_block.signature,
                            provider_name=self.provider_name,
                        ):
                            yield event_
                    elif isinstance(current_block, BetaRedactedThinkingBlock):
                        for event_ in self._parts_manager.handle_thinking_delta(
                            vendor_part_id=event.index,
                            id='redacted_thinking',
                            signature=current_block.data,
                            provider_name=self.provider_name,
                        ):
                            yield event_
                    elif isinstance(current_block, BetaToolUseBlock):
                        maybe_event = self._parts_manager.handle_tool_call_delta(
                            vendor_part_id=event.index,
                            tool_name=current_block.name,
                            args=cast(dict[str, Any], current_block.input) or None,
                            tool_call_id=current_block.id,
                        )
                        if maybe_event is not None:  # pragma: no branch
                            yield maybe_event
                    elif isinstance(current_block, BetaServerToolUseBlock):
                        if current_block.name not in self._enabled_server_tool_names:
                            # Unlike non-streaming, this cannot pre-scan later result blocks, so a gap in the
                            # enabled-name set would leave their return part orphaned. Result presence is only
                            # advisory: newer web tools can omit paired blocks with `response_inclusion: "excluded"`,
                            # which pydantic-ai does not currently send.
                            ignored_server_tool_use_indices.add(event.index)
                            continue
                        call_part = map_server_tool_use_block(current_block, self.provider_name)
                        builtin_tool_calls[call_part.tool_call_id] = call_part
                        # In streaming, the block's `input` is usually empty at start and arrives via
                        # subsequent `BetaInputJSONDelta` events. Emit with `args=None` so the
                        # accumulating JSON deltas can attach as a string; the
                        # `BetaRawContentBlockStopEvent` handler below normalizes a tool search's final
                        # value back to the canonical part shape (matching non-streaming). A server tool
                        # call made from code execution carries its whole input here, without deltas.
                        yield self._parts_manager.handle_part(
                            vendor_part_id=event.index,
                            part=call_part if current_block.input else replace(call_part, args=None),
                        )
                    elif isinstance(current_block, BetaWebSearchToolResultBlock):
                        yield self._parts_manager.handle_part(
                            vendor_part_id=event.index,
                            part=map_web_search_tool_result_block(current_block, self.provider_name),
                        )
                    elif isinstance(current_block, BetaToolSearchToolResultBlock):
                        yield self._parts_manager.handle_part(
                            vendor_part_id=event.index,
                            part=map_tool_search_tool_result_block(current_block, self.provider_name),
                        )
                    elif isinstance(current_block, BetaCodeExecutionToolResultBlock):
                        # Legacy code execution responses used this bare `code_execution_tool_result` shape.
                        # Current code execution tool versions emit the named bash/text-editor blocks below.
                        yield self._parts_manager.handle_part(
                            vendor_part_id=event.index,
                            part=map_code_execution_tool_result_block(current_block, self.provider_name),
                        )
                    elif isinstance(current_block, BetaBashCodeExecutionToolResultBlock):
                        yield self._parts_manager.handle_part(
                            vendor_part_id=event.index,
                            part=map_bash_code_execution_tool_result_block(current_block, self.provider_name),
                        )
                    elif isinstance(current_block, BetaTextEditorCodeExecutionToolResultBlock):
                        yield self._parts_manager.handle_part(
                            vendor_part_id=event.index,
                            part=map_text_editor_code_execution_tool_result_block(current_block, self.provider_name),
                        )
                    elif isinstance(current_block, BetaWebFetchToolResultBlock):  # pragma: lax no cover
                        if page := fetched_page(current_block):
                            fetched_pages.append(page)
                        yield self._parts_manager.handle_part(
                            vendor_part_id=event.index,
                            part=map_web_fetch_tool_result_block(current_block, self.provider_name),
                        )
                    elif isinstance(current_block, BetaAdvisorToolResultBlock):
                        # The advisor result block arrives fully formed in a single `content_block_start`
                        # (no deltas), same as web search results.
                        yield self._parts_manager.handle_part(
                            vendor_part_id=event.index,
                            part=map_advisor_tool_result_block(current_block, self.provider_name),
                        )
                    elif isinstance(current_block, BetaMCPToolUseBlock):
                        call_part = map_mcp_server_use_block(current_block, self.provider_name)
                        builtin_tool_calls[call_part.tool_call_id] = call_part

                        args_json = call_part.args_as_json_str()
                        # Drop the final `{}}` so that we can add tool args deltas
                        args_json_delta = args_json[:-3]
                        assert args_json_delta.endswith('"tool_args":'), (
                            f'Expected {args_json_delta!r} to end in `"tool_args":`'
                        )

                        yield self._parts_manager.handle_part(
                            vendor_part_id=event.index, part=replace(call_part, args=None)
                        )
                        maybe_event = self._parts_manager.handle_tool_call_delta(
                            vendor_part_id=event.index,
                            args=args_json_delta,
                        )
                        if maybe_event is not None:  # pragma: no branch
                            yield maybe_event
                    elif isinstance(current_block, BetaMCPToolResultBlock):
                        call_part = builtin_tool_calls.get(current_block.tool_use_id)
                        yield self._parts_manager.handle_part(
                            vendor_part_id=event.index,
                            part=map_mcp_server_result_block(current_block, call_part, self.provider_name),
                        )
                    elif isinstance(current_block, BetaCompactionBlock):
                        yield self._parts_manager.handle_part(
                            vendor_part_id=event.index,
                            part=CompactionPart(
                                content=current_block.content,
                                provider_name=self.provider_name,
                                provider_details=compaction_provider_details(
                                    current_block.encrypted_content, current_block.signature
                                ),
                            ),
                        )

                elif isinstance(event, BetaRawContentBlockDeltaEvent):
                    if event.index in ignored_server_tool_use_indices:
                        continue
                    if isinstance(event.delta, BetaTextDelta):
                        for event_ in self._parts_manager.handle_text_delta(
                            vendor_part_id=event.index, content=event.delta.text
                        ):
                            yield event_
                    elif isinstance(event.delta, BetaThinkingDelta):
                        for event_ in self._parts_manager.handle_thinking_delta(
                            vendor_part_id=event.index,
                            content=event.delta.thinking,
                            provider_name=self.provider_name,
                        ):
                            yield event_
                    elif isinstance(event.delta, BetaSignatureDelta):
                        for event_ in self._parts_manager.handle_thinking_delta(
                            vendor_part_id=event.index,
                            signature=event.delta.signature,
                            provider_name=self.provider_name,
                        ):
                            yield event_
                    elif isinstance(event.delta, BetaInputJSONDelta):
                        maybe_event = self._parts_manager.handle_tool_call_delta(
                            vendor_part_id=event.index,
                            args=event.delta.partial_json,
                        )
                        if maybe_event is not None:  # pragma: no branch
                            yield maybe_event
                    elif isinstance(event.delta, BetaCompactionContentBlockDelta):
                        if event.delta.content:  # pragma: no branch
                            # Re-emit part with updated content; replaces the initial block start part
                            yield self._parts_manager.handle_part(
                                vendor_part_id=event.index,
                                part=CompactionPart(
                                    content=event.delta.content,
                                    provider_name=self.provider_name,
                                    provider_details=compaction_provider_details(event.delta.encrypted_content),
                                ),
                            )
                    elif isinstance(event.delta, BetaCitationsDelta):  # pragma: no branch
                        pending_citations.setdefault(event.index, []).append(event.delta.citation)

                elif isinstance(event, BetaRawMessageDeltaEvent):
                    self._usage = map_usage(
                        event, self._provider_name, self._provider_url, self._model_name, self._usage
                    )
                    if (context_input_tokens := last_pass_input_tokens(event.usage)) is not None:
                        mark_context_input_tokens(self, context_input_tokens)
                    if raw_finish_reason := event.delta.stop_reason:  # pragma: no branch
                        self.provider_details = self.provider_details or {}
                        self.provider_details['finish_reason'] = raw_finish_reason
                        self.finish_reason = FINISH_REASON_MAP.get(raw_finish_reason)
                        self.state = 'suspended' if raw_finish_reason == 'pause_turn' else 'complete'
                    if event.delta.stop_details is not None:
                        self.provider_details = self.provider_details or {}
                        if event.delta.stop_details.explanation is not None:
                            self.provider_details['refusal'] = event.delta.stop_details.explanation
                        if event.delta.stop_details.category is not None:
                            self.provider_details['refusal_category'] = event.delta.stop_details.category
                    if event.delta.container:
                        self.provider_details = self.provider_details or {}
                        self.provider_details['container_id'] = event.delta.container.id
                    if event.input_transformations:
                        # Replaces, never extends: a `message_delta` reports this only after a
                        # mid-stream model fallback, and then it holds the serving model's whole
                        # array, which repeats the entries `message_start` already reported.
                        self.provider_details = self.provider_details or {}
                        self.provider_details['input_transformations'] = report_input_transformations(
                            event.input_transformations
                        )

                elif isinstance(event, BetaRawContentBlockStopEvent):  # pragma: no branch
                    block_citations = pending_citations.pop(event.index, [])
                    part = self._parts_manager.get_part_by_vendor_id(event.index)
                    if isinstance(part, TextPart) and (
                        citations := map_citations(block_citations, part.content, fetched_pages)
                    ):
                        for event_ in self._parts_manager.handle_text_delta(
                            vendor_part_id=event.index,
                            content='',
                            provider_name=self.provider_name,
                            provider_details=citations_provider_details(block_citations),
                            citations=citations,
                        ):
                            yield event_
                    if event.index in ignored_server_tool_use_indices:
                        ignored_server_tool_use_indices.remove(event.index)
                    elif isinstance(current_block, BetaMCPToolUseBlock):
                        maybe_event = self._parts_manager.handle_tool_call_delta(
                            vendor_part_id=event.index,
                            args='}',
                        )
                        if maybe_event is not None:  # pragma: no branch
                            yield maybe_event
                    elif isinstance(current_block, BetaServerToolUseBlock) and current_block.name in (
                        'tool_search_tool_regex',
                        'tool_search_tool_bm25',
                    ):
                        # The streaming start emitted the part with `args=None`; JSON deltas
                        # have since accumulated as a string. Re-emit with the normalized
                        # cross-provider `ToolSearchArgs` shape so downstream code (history
                        # replay, typed-part dispatch) sees the same structure as the
                        # non-streaming `_process_response` path produces.
                        existing = self._parts_manager.get_part_by_vendor_id(event.index)
                        if isinstance(existing, NativeToolSearchCallPart):  # pragma: no branch
                            yield self._parts_manager.handle_part(
                                vendor_part_id=event.index,
                                part=finalize_streamed_tool_search_call_part(existing),
                            )
                    current_block = None
                elif isinstance(event, BetaRawMessageStopEvent):  # pragma: no branch
                    current_block = None

    async def close_stream(self) -> None:
        await self._response.source.close()

    @property
    def model_name(self) -> AnthropicModelName:
        """Get the model name of the response."""
        return self._model_name

    @property
    def provider_name(self) -> str:
        """Get the provider name."""
        return self._provider_name

    @property
    def provider_url(self) -> str:
        """Get the provider base URL."""
        return self._provider_url

    @property
    def timestamp(self) -> datetime:
        """Get the timestamp of the response."""
        return self._timestamp
