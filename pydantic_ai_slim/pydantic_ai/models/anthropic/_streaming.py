"""`AnthropicStreamedResponse`, which maps Anthropic stream events to Pydantic AI parts and response metadata."""

from __future__ import annotations as _annotations

from collections.abc import AsyncIterator, Iterator, Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import TYPE_CHECKING, Any, TypeAlias, cast

import pydantic_core
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
from ...exceptions import UnexpectedModelBehavior
from ...messages import CompactionPart, ModelResponseStreamEvent, NativeToolCallPart, NativeToolSearchCallPart, TextPart
from .. import StreamedResponse
from .._decode_errors import MapStreamDecodeErrors
from ._citations import citations_provider_details, fetched_page, map_citations
from ._computer import NativeComputer, takes_no_input
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
    refusal_details,
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
    _native_computer: NativeComputer | None = None
    _current_block: BetaContentBlock | None = field(default=None, init=False)
    _ignored_server_tool_use_indices: set[int] = field(default_factory=set[int], init=False)
    _pending_citations: dict[int, list[BetaTextCitation]] = field(
        default_factory=dict[int, list[BetaTextCitation]], init=False
    )
    # Web Fetch citations name their page only by its text, so the fetched pages are kept to find their URLs.
    _fetched_pages: list[tuple[str, str]] = field(default_factory=list[tuple[str, str]], init=False)
    _builtin_tool_calls: dict[str, NativeToolCallPart] = field(
        default_factory=dict[str, NativeToolCallPart], init=False
    )
    # A response's computer toolset calls are one batch: one `computer` call at the first one's place, started when
    # another block or the response's end follows a member, so no later part's event comes before it.
    _computer_blocks: list[dict[str, Any]] = field(default_factory=list[dict[str, Any]], init=False)
    _computer_started: int = field(default=0, init=False)
    """How many of the batch's calls the `computer` call last started held."""
    _computer_input: list[str] = field(default_factory=list[str], init=False)
    # Calls whose input didn't arrive whole, and the other tools called between the batch's calls.
    _computer_incomplete: set[str] = field(default_factory=set[str], init=False)
    _computer_between: list[str] = field(default_factory=list[str], init=False)
    _computer_split: dict[str, list[str]] = field(default_factory=dict[str, list[str]], init=False)
    # Only a response that reached its end is known to hold every call it made.
    _ended: bool = field(default=False, init=False)

    async def _get_event_iterator(self) -> AsyncIterator[ModelResponseStreamEvent]:
        with map_api_errors(self._model_name, self._model_id_namespace):
            self._fetched_pages = list(self._history_fetched_pages)
            async for event in MapStreamDecodeErrors(self._response, self._model_name):
                for event_ in self._map_event(event):
                    yield event_
            native_computer = self._native_computer
            if (
                native_computer
                and not self._ended
                and (self._computer_blocks or native_computer.answers(self._current_block))
            ):
                # The stream stopped early: a call after the ones that arrived may be missing, and they run as one.
                raise UnexpectedModelBehavior(
                    'The response ended before Claude finished its computer use calls, so none of them ran.'
                )
            for event_ in self._start_computer_batch():
                yield event_

    def _start_computer_batch(self) -> Iterator[ModelResponseStreamEvent]:
        native_computer = self._native_computer
        if native_computer and len(self._computer_blocks) > self._computer_started:
            self._computer_started = len(self._computer_blocks)
            yield self._parts_manager.handle_part(
                vendor_part_id=self._computer_blocks[0]['id'],
                part=native_computer.call_part(
                    self._computer_blocks,
                    self.provider_name,
                    incomplete=self._computer_incomplete,
                    split=self._computer_split,
                ),
            )

    def _map_event(self, event: BetaRawMessageStreamEvent) -> Iterator[ModelResponseStreamEvent]:
        match event:
            case BetaRawMessageStartEvent():
                self._map_message_start(event)
            case BetaRawContentBlockStartEvent():
                yield from self._map_content_block_start(event)
            case BetaRawContentBlockDeltaEvent():
                yield from self._map_content_block_delta(event)
            case BetaRawMessageDeltaEvent():
                self._map_message_delta(event)
            case BetaRawContentBlockStopEvent():
                yield from self._map_content_block_stop(event)
            case BetaRawMessageStopEvent():
                self._current_block = None
                self._ended = True
            case _:  # pragma: no cover
                pass

    def _map_message_start(self, event: BetaRawMessageStartEvent) -> None:
        if event.message is None:  # pyright: ignore[reportUnnecessaryComparison]
            # See `map_usage`: Bedrock emits type-less chunks the SDK constructs
            # as `BetaRawMessageStartEvent(message=None)`. Skip them entirely so we
            # don't dereference `event.message.id` / `.container` below.
            return
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
        # See `response_provider_details`. Only the opening message's usage reports the speed.
        if event.message.usage.speed == 'fast':
            self.provider_details = self.provider_details or {}
            self.provider_details['speed'] = 'fast'

    def _map_content_block_start(self, event: BetaRawContentBlockStartEvent) -> Iterator[ModelResponseStreamEvent]:  # noqa: C901
        self._current_block = current_block = event.content_block
        native_computer = self._native_computer
        if native_computer and native_computer.answers(current_block):
            self._computer_input = []
            if self._computer_between:
                self._computer_split[current_block.id] = self._computer_between
                self._computer_between = []
            return
        yield from self._start_computer_batch()
        if self._computer_blocks and isinstance(current_block, BetaToolUseBlock):
            self._computer_between.append(current_block.name)
        match current_block:
            case BetaTextBlock() if current_block.text:
                yield from self._parts_manager.handle_text_delta(vendor_part_id=event.index, content=current_block.text)
            case BetaThinkingBlock():
                yield from self._parts_manager.handle_thinking_delta(
                    vendor_part_id=event.index,
                    content=current_block.thinking,
                    signature=current_block.signature,
                    provider_name=self.provider_name,
                )
            case BetaRedactedThinkingBlock():
                yield from self._parts_manager.handle_thinking_delta(
                    vendor_part_id=event.index,
                    id='redacted_thinking',
                    signature=current_block.data,
                    provider_name=self.provider_name,
                )
            case BetaToolUseBlock():
                yield from self._parts_manager.handle_tool_call_delta_events(
                    vendor_part_id=event.index,
                    tool_name=current_block.name,
                    args=cast(dict[str, Any], current_block.input) or None,
                    tool_call_id=current_block.id,
                )
            case BetaServerToolUseBlock():
                yield from self._map_server_tool_use_block_start(event.index, current_block)
            case BetaWebSearchToolResultBlock():
                yield self._parts_manager.handle_part(
                    vendor_part_id=event.index,
                    part=map_web_search_tool_result_block(current_block, self.provider_name),
                )
            case BetaToolSearchToolResultBlock():
                yield self._parts_manager.handle_part(
                    vendor_part_id=event.index,
                    part=map_tool_search_tool_result_block(current_block, self.provider_name),
                )
            case BetaCodeExecutionToolResultBlock():
                # Legacy code execution responses used this bare `code_execution_tool_result` shape.
                # Current code execution tool versions emit the named bash/text-editor blocks below.
                yield self._parts_manager.handle_part(
                    vendor_part_id=event.index,
                    part=map_code_execution_tool_result_block(current_block, self.provider_name),
                )
            case BetaBashCodeExecutionToolResultBlock():
                yield self._parts_manager.handle_part(
                    vendor_part_id=event.index,
                    part=map_bash_code_execution_tool_result_block(current_block, self.provider_name),
                )
            case BetaTextEditorCodeExecutionToolResultBlock():
                yield self._parts_manager.handle_part(
                    vendor_part_id=event.index,
                    part=map_text_editor_code_execution_tool_result_block(current_block, self.provider_name),
                )
            case BetaWebFetchToolResultBlock():  # pragma: lax no cover
                if page := fetched_page(current_block):
                    self._fetched_pages.append(page)
                yield self._parts_manager.handle_part(
                    vendor_part_id=event.index,
                    part=map_web_fetch_tool_result_block(current_block, self.provider_name),
                )
            case BetaAdvisorToolResultBlock():
                # The advisor result block arrives fully formed in a single `content_block_start`
                # (no deltas), same as web search results.
                yield self._parts_manager.handle_part(
                    vendor_part_id=event.index,
                    part=map_advisor_tool_result_block(current_block, self.provider_name),
                )
            case BetaMCPToolUseBlock():
                yield from self._map_mcp_tool_use_block_start(event.index, current_block)
            case BetaMCPToolResultBlock():
                yield self._parts_manager.handle_part(
                    vendor_part_id=event.index,
                    part=map_mcp_server_result_block(
                        current_block, self._builtin_tool_calls.get(current_block.tool_use_id), self.provider_name
                    ),
                )
            case BetaCompactionBlock():
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
            case _:
                pass

    def _map_server_tool_use_block_start(
        self, index: int, block: BetaServerToolUseBlock
    ) -> Iterator[ModelResponseStreamEvent]:
        if block.name not in self._enabled_server_tool_names:
            # Unlike non-streaming, this cannot pre-scan later result blocks, so a gap in the
            # enabled-name set would leave their return part orphaned. Result presence is only
            # advisory: newer web tools can omit paired blocks with `response_inclusion: "excluded"`,
            # which pydantic-ai does not currently send.
            self._ignored_server_tool_use_indices.add(index)
            return
        call_part = map_server_tool_use_block(block, self.provider_name)
        self._builtin_tool_calls[call_part.tool_call_id] = call_part
        # In streaming, the block's `input` is usually empty at start and arrives via
        # subsequent `BetaInputJSONDelta` events. Emit with `args=None` so the
        # accumulating JSON deltas can attach as a string; `_map_content_block_stop`
        # normalizes a tool search's final value back to the canonical part shape (matching non-streaming).
        # A server tool call made from code execution carries its whole input here, without deltas.
        yield self._parts_manager.handle_part(
            vendor_part_id=index,
            part=call_part if block.input else replace(call_part, args=None),
        )

    def _map_mcp_tool_use_block_start(
        self, index: int, block: BetaMCPToolUseBlock
    ) -> Iterator[ModelResponseStreamEvent]:
        call_part = map_mcp_server_use_block(block, self.provider_name)
        self._builtin_tool_calls[call_part.tool_call_id] = call_part

        args_json = call_part.args_as_json_str()
        # Drop the final `{}}` so that we can add tool args deltas
        args_json_delta = args_json[:-3]
        assert args_json_delta.endswith('"tool_args":'), f'Expected {args_json_delta!r} to end in `"tool_args":`'

        yield self._parts_manager.handle_part(vendor_part_id=index, part=replace(call_part, args=None))
        yield from self._parts_manager.handle_tool_call_delta_events(vendor_part_id=index, args=args_json_delta)

    def _map_content_block_delta(self, event: BetaRawContentBlockDeltaEvent) -> Iterator[ModelResponseStreamEvent]:
        if event.index in self._ignored_server_tool_use_indices:
            return
        delta = event.delta
        match delta:
            case BetaTextDelta():
                yield from self._parts_manager.handle_text_delta(vendor_part_id=event.index, content=delta.text)
            case BetaThinkingDelta():
                yield from self._parts_manager.handle_thinking_delta(
                    vendor_part_id=event.index,
                    content=delta.thinking,
                    provider_name=self.provider_name,
                )
            case BetaSignatureDelta():
                yield from self._parts_manager.handle_thinking_delta(
                    vendor_part_id=event.index,
                    signature=delta.signature,
                    provider_name=self.provider_name,
                )
            case BetaInputJSONDelta() if self._native_computer and self._native_computer.answers(self._current_block):
                self._computer_input.append(delta.partial_json)
            case BetaInputJSONDelta():
                yield from self._parts_manager.handle_tool_call_delta_events(
                    vendor_part_id=event.index,
                    args=delta.partial_json,
                )
            case BetaCompactionContentBlockDelta():
                if delta.content:  # pragma: no branch
                    # Re-emit part with updated content; replaces the initial block start part
                    yield self._parts_manager.handle_part(
                        vendor_part_id=event.index,
                        part=CompactionPart(
                            content=delta.content,
                            provider_name=self.provider_name,
                            provider_details=compaction_provider_details(delta.encrypted_content),
                        ),
                    )
            case BetaCitationsDelta():
                self._pending_citations.setdefault(event.index, []).append(delta.citation)
            case _:  # pragma: no cover
                pass

    def _map_message_delta(self, event: BetaRawMessageDeltaEvent) -> None:
        self._usage = map_usage(event, self._provider_name, self._provider_url, self._model_name, self._usage)
        if (context_input_tokens := last_pass_input_tokens(event.usage)) is not None:
            mark_context_input_tokens(self, context_input_tokens)
        if raw_finish_reason := event.delta.stop_reason:  # pragma: no branch
            self.provider_details = self.provider_details or {}
            self.provider_details['finish_reason'] = raw_finish_reason
            self.finish_reason = FINISH_REASON_MAP.get(raw_finish_reason)
            self.state = 'suspended' if raw_finish_reason == 'pause_turn' else 'complete'
        if event.delta.stop_details is not None:
            self.provider_details = self.provider_details or {}
            self.provider_details.update(refusal_details(event.delta.stop_details))
        if event.delta.container:
            self.provider_details = self.provider_details or {}
            self.provider_details['container_id'] = event.delta.container.id
        if event.input_transformations:
            # Replaces, never extends: a `message_delta` reports this only after a
            # mid-stream model fallback, and then it holds the serving model's whole
            # array, which repeats the entries `message_start` already reported.
            self.provider_details = self.provider_details or {}
            self.provider_details['input_transformations'] = report_input_transformations(event.input_transformations)

    def _map_content_block_stop(self, event: BetaRawContentBlockStopEvent) -> Iterator[ModelResponseStreamEvent]:
        block_citations = self._pending_citations.pop(event.index, [])
        part = self._parts_manager.get_part_by_vendor_id(event.index)
        if isinstance(part, TextPart) and (
            citations := map_citations(block_citations, part.content, self._fetched_pages)
        ):
            yield from self._parts_manager.handle_text_delta(
                vendor_part_id=event.index,
                content='',
                provider_name=self.provider_name,
                provider_details=citations_provider_details(block_citations),
                citations=citations,
            )
        current_block = self._current_block
        native_computer = self._native_computer
        if native_computer and native_computer.answers(current_block):
            yield from self._map_computer_block_stop(native_computer, current_block)
        elif event.index in self._ignored_server_tool_use_indices:
            self._ignored_server_tool_use_indices.remove(event.index)
        elif isinstance(current_block, BetaMCPToolUseBlock):
            yield from self._parts_manager.handle_tool_call_delta_events(vendor_part_id=event.index, args='}')
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
        self._current_block = None

    def _map_computer_block_stop(
        self, native_computer: NativeComputer, current_block: BetaToolUseBlock
    ) -> Iterator[ModelResponseStreamEvent]:
        # A member that takes input and streamed none was cut off before it.
        computer_input_json = ''.join(self._computer_input) or ('{}' if takes_no_input(current_block.name) else '')
        computer_args: Any
        try:
            computer_args = pydantic_core.from_json(computer_input_json)
        except ValueError:
            # Cut off mid-input, as at `max_tokens`: what parsed is kept to send back, but none of it runs, as a
            # missing field such as a held key would change what the rest does.
            self._computer_incomplete.add(current_block.id)
            try:
                computer_args = pydantic_core.from_json(computer_input_json, allow_partial=True)
            except ValueError:  # not even the start of a value, such as `n` of `null`
                computer_args = {}
        block = {'id': current_block.id, 'name': current_block.name, 'input': computer_args}
        if native_computer.is_toolset:
            self._computer_blocks.append(block)
        else:
            yield self._parts_manager.handle_part(
                vendor_part_id=block['id'],
                part=native_computer.call_part([block], self.provider_name, incomplete=self._computer_incomplete),
            )

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
