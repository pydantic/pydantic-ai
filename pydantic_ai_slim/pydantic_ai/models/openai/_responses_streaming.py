"""`OpenAIResponsesStreamedResponse`, the Responses API stream."""

from __future__ import annotations as _annotations

import base64
import warnings
from collections import deque
from collections.abc import AsyncIterator, Callable, Iterator
from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import TYPE_CHECKING, Any, Literal, cast

from openai import AsyncStream
from openai.types import responses
from openai.types.responses import ResponseCompactionItem
from openai.types.responses.response_status import ResponseStatus
from pydantic import ValidationError
from pydantic_core import to_json

from ... import _utils, usage
from ..._utils import now_utc as _now_utc, number_to_datetime
from ...computer_use import COMPUTER_KIND
from ...exceptions import ModelConnectionError
from ...messages import BinaryImage, FilePart, ModelResponse, ModelResponseStreamEvent, TextPart, ToolCallPart
from .. import StreamedResponse
from .._decode_errors import MapStreamDecodeErrors
from ._citations import (
    jsonable_annotation,
    map_responses_citations,
    responses_citations_provider_details,
    responses_output_text_annotation_ta,
)
from ._computer import map_computer_call
from ._errors import map_api_errors, response_error, response_error_type
from ._history_digest import Hasher, with_history_digest
from ._model_names import OpenAIModelName
from ._native_tools import (
    map_code_interpreter_tool_call,
    map_file_search_tool_call,
    map_image_generation_tool_call,
    map_mcp_call,
    map_mcp_list_tools,
    map_web_search_tool_call,
)
from ._response_fields import (
    RESPONSES_FINISH_REASON_MAP,
    map_compaction_item,
    map_logprobs,
    map_usage,
    response_status_to_state,
    response_tool_call_id,
)
from ._settings import OpenAIResponsesModelSettings, echoed_store
from ._tool_search import (
    build_tool_search_return_part,
    map_client_tool_search_call,
    map_tool_search_call,
    match_tool_search_output_to_call,
)

if TYPE_CHECKING:
    from ._responses_websocket import ResponsesWebSocketStream


@dataclass
class OpenAIResponsesStreamedResponse(StreamedResponse):
    """Implementation of `StreamedResponse` for OpenAI Responses API."""

    _model_name: OpenAIModelName
    _model_settings: OpenAIResponsesModelSettings
    _response: _utils.PeekableAsyncStream[
        responses.ResponseStreamEvent, AsyncStream[responses.ResponseStreamEvent] | ResponsesWebSocketStream
    ]
    _provider_name: str
    _model_id_namespace: str
    _provider_url: str
    _tool_call_ids_are_response_scoped: bool
    _process_provider_details: Callable[[responses.Response], dict[str, Any] | None]
    _provider_timestamp: datetime | None = None
    _timestamp: datetime = field(default_factory=_now_utc)
    _usage_received: bool | None = field(default=False, init=False)
    _has_refusal: bool = field(default=False, init=False)
    _refusal_text: str = field(default='', init=False)
    _last_sequence_number: int | None = field(default=None, init=False)
    # Text annotations by item ID and content index, collected until `output_text.done` maps them.
    _annotations_by_item: dict[tuple[str, int], list[object]] = field(
        default_factory=dict[tuple[str, int], list[object]], init=False
    )
    # `phase` (commentary | final_answer) on assistant message items, captured from the `output_item.added` event
    # and merged into the corresponding `TextPart.provider_details` on the first `output_text.delta` (so consumers
    # can filter commentary from final-answer text as it streams, rather than having to buffer the whole part), or
    # on `output_text.done` if no delta was received.
    _phase_by_item: dict[str, Literal['commentary', 'final_answer']] = field(
        default_factory=dict[str, Literal['commentary', 'final_answer']], init=False
    )
    _phase_started_content: set[tuple[str, int]] = field(default_factory=set[tuple[str, int]], init=False)
    _mcp_list_tools_return_ids: set[str] = field(default_factory=set[str], init=False)
    _pending_tool_search_call_ids: deque[str] = field(default_factory=deque[str], init=False)
    _function_call_args: dict[str, str] = field(default_factory=dict[str, str], init=False)
    _finalized_function_calls: set[str] = field(default_factory=set[str], init=False)
    _history: Hasher | None = None
    """The history this response was created from, digested with its parts as `ModelResponse.history_digest`."""
    # Custom HTTP clients can return streams outside the SDK's AsyncStream class.
    _is_websocket: bool = False

    def get(self) -> ModelResponse:
        response = super().get()
        if self._is_websocket:
            response.provider_details = {**(response.provider_details or {}), 'transport': 'websocket'}
        return with_history_digest(response, self._history)

    def _set_state(self, status: ResponseStatus | None) -> None:
        # `_track_background` stamps the `background` marker from `response.background` on every status
        # event before this runs, so a pending status only suspends for a genuine (resumable) background job.
        background = bool((self.provider_details or {}).get('background'))
        self.state = response_status_to_state(status, background=background)

    def _track_background(self, response: responses.Response) -> None:
        """Mark a background job so `cancel_suspended_response` can cancel it server-side.

        `responses.cancel` only works on background jobs; an explicit `background` marker (stamped
        from the API's own `response.background` field on every status event, so it survives the
        stream ending suspended, resuming via `retrieve`, or completing) keeps the cancel guard
        robust, rather than inferring background mode from the `continuation_delay` poll interval.
        """
        if response.background:
            self.provider_details = {**(self.provider_details or {}), 'background': True}

    def _handle_terminal_response(self, response: responses.Response) -> None:
        """Take what the complete response that ends the stream reports as `_process_response` does."""
        # A stream resumed with `starting_after` has no `response.created`, so this is where it first sees these.
        self._model_name = response.model or self._model_name
        self.provider_response_id = self.provider_response_id or response.id
        if response.created_at and 'timestamp' not in (self.provider_details or {}):
            self.provider_details = {
                **(self.provider_details or {}),
                'timestamp': number_to_datetime(response.created_at),
            }
        # Only terminal events report the tier that served the request; earlier ones echo the requested tier.
        if service_tier := response.service_tier:
            self.provider_details = {**(self.provider_details or {}), 'service_tier': service_tier}
        # Likewise, earlier events can report prompt cache diagnostics as `unavailable` before the comparison has finished.
        if (diagnostics := response.prompt_cache_diagnostics) is not None:
            self.provider_details = {
                **(self.provider_details or {}),
                'prompt_cache_diagnostics': diagnostics.model_dump(mode='json'),
            }
        if (store := echoed_store(response, self._model_settings)) is not None:
            self.provider_details = {**(self.provider_details or {}), 'store': store}
        # Merged last so built-in provider details win on collisions, as in `_process_response`; those set later in
        # the stream, like `finish_reason`, do too.
        if hook_provider_details := self._process_provider_details(response):
            self.provider_details = {**hook_provider_details, **(self.provider_details or {})}

    def get_stream_cancel_errors(self) -> tuple[type[BaseException], ...]:
        # Closing a WebSocket response interrupts its pending read with a connection error at the `read` phase.
        return (ModelConnectionError,) if self._is_websocket else super().get_stream_cancel_errors()

    async def close_stream(self) -> None:
        await self._response.source.close()

    async def _get_event_iterator(self) -> AsyncIterator[ModelResponseStreamEvent]:
        with map_api_errors(self._model_name, self._model_id_namespace):
            if self._provider_timestamp is not None:  # pragma: no branch
                self.provider_details = {'timestamp': self._provider_timestamp}

            async for chunk in MapStreamDecodeErrors(self._response, self._model_name):
                for event in self._map_event(chunk):
                    yield event

            if self._refusal_text:
                self.provider_details = {**(self.provider_details or {}), 'refusal': self._refusal_text}

        # This is used to resume suspended background streams with `starting_after`.
        if self.state == 'suspended' and self._last_sequence_number is not None:
            self.provider_details = {
                **(self.provider_details or {}),
                'last_sequence_number': self._last_sequence_number,
            }

    def _map_event(self, chunk: responses.ResponseStreamEvent) -> Iterator[ModelResponseStreamEvent]:  # noqa: C901
        self._last_sequence_number = chunk.sequence_number
        if isinstance(
            chunk,
            (
                responses.ResponseCreatedEvent,
                responses.ResponseInProgressEvent,
                responses.ResponseQueuedEvent,
                responses.ResponseCompletedEvent,
                responses.ResponseFailedEvent,
                responses.ResponseIncompleteEvent,
            ),
        ):
            # Stamp the background marker from any status event that carries it, so it survives a
            # stream that starts mid-way (a resumed `retrieve(stream=True)` whose first event is
            # `in_progress`/`queued`) or only reaches a terminal event. `cancel_suspended_response`
            # relies on it to cancel the server-side job.
            self._track_background(chunk.response)
            if isinstance(
                chunk,
                (
                    responses.ResponseCompletedEvent,
                    responses.ResponseFailedEvent,
                    responses.ResponseIncompleteEvent,
                ),
            ):
                self._handle_terminal_response(chunk.response)
        match chunk:
            # NOTE: You can inspect the builtin tools used checking the `ResponseCompletedEvent`.
            case responses.ResponseCompletedEvent():
                yield from self._map_completed(chunk)
            case responses.ResponseCreatedEvent():
                self._map_created(chunk)
            case responses.ResponseFailedEvent():
                self._map_failed(chunk)
            case responses.ResponseFunctionCallArgumentsDeltaEvent():
                yield from self._map_function_call_arguments_delta(chunk)
            case responses.ResponseFunctionCallArgumentsDoneEvent():
                yield from self._finalize_function_call(chunk.item_id, chunk.arguments)
            case responses.ResponseInProgressEvent() | responses.ResponseQueuedEvent():
                self._map_in_progress(chunk)
            case responses.ResponseIncompleteEvent():
                self._map_incomplete(chunk)
            case responses.ResponseOutputItemAddedEvent():
                yield from self._map_output_item_added(chunk)
            case responses.ResponseOutputItemDoneEvent():
                yield from self._map_output_item_done(chunk.item)
            case responses.ResponseReasoningSummaryPartAddedEvent():
                yield from self._map_reasoning_summary_delta(chunk.item_id, chunk.summary_index, chunk.part.text)
            case responses.ResponseReasoningSummaryTextDeltaEvent():
                yield from self._map_reasoning_summary_delta(chunk.item_id, chunk.summary_index, chunk.delta)
            case responses.ResponseReasoningTextDeltaEvent():
                # Handle raw CoT from gpt-oss models using callback pattern
                yield from self._parts_manager.handle_thinking_delta(
                    vendor_part_id=chunk.item_id,
                    id=chunk.item_id,
                    provider_name=self.provider_name,
                    provider_details=make_raw_content_updater(chunk.delta, chunk.content_index),
                )
            case responses.ResponseOutputTextAnnotationAddedEvent():
                # The event's `annotation` uses look-alike classes from another SDK module, which
                # `responses_output_text_annotations_ta` rejects, so it's turned into a plain dict first.
                self._annotations_by_item.setdefault((chunk.item_id, chunk.content_index), []).append(
                    jsonable_annotation(chunk.annotation)
                )
            case responses.ResponseTextDeltaEvent():
                yield from self._map_text_delta(chunk)
            case responses.ResponseTextDoneEvent():
                yield from self._map_text_done(chunk)
            case responses.ResponseRefusalDeltaEvent():
                self._map_refusal_delta(chunk)
            case responses.ResponseRefusalDoneEvent():
                self._map_refusal_done(chunk)
            case responses.ResponseCodeInterpreterCallCodeDeltaEvent():
                yield from self._map_code_interpreter_code_delta(chunk)
            case responses.ResponseCodeInterpreterCallCodeDoneEvent():
                yield from self._parts_manager.handle_tool_call_delta_events(
                    vendor_part_id=f'{chunk.item_id}-call',
                    args='"}',
                )
            case responses.ResponseImageGenCallPartialImageEvent():
                yield self._map_partial_image(chunk)
            case responses.ResponseMcpCallArgumentsDoneEvent():
                yield from self._parts_manager.handle_tool_call_delta_events(
                    vendor_part_id=f'{chunk.item_id}-call',
                    args='}',
                )
            case responses.ResponseMcpCallArgumentsDeltaEvent():
                yield from self._parts_manager.handle_tool_call_delta_events(
                    vendor_part_id=f'{chunk.item_id}-call',
                    args=chunk.delta,
                )
            case responses.ResponseErrorEvent():
                raise response_error(self._model_name, chunk.code, chunk.message, in_stream=True)
            case (
                responses.ResponseContentPartAddedEvent()
                | responses.ResponseContentPartDoneEvent()
                | responses.ResponseReasoningSummaryPartDoneEvent()
                | responses.ResponseReasoningSummaryTextDoneEvent()
                # Raw CoT content is already accumulated via delta events.
                | responses.ResponseReasoningTextDoneEvent()
                | responses.ResponseWebSearchCallInProgressEvent()
                | responses.ResponseWebSearchCallSearchingEvent()
                | responses.ResponseWebSearchCallCompletedEvent()
                | responses.ResponseAudioDeltaEvent()
                | responses.ResponseCodeInterpreterCallCompletedEvent()
                | responses.ResponseCodeInterpreterCallInProgressEvent()
                | responses.ResponseCodeInterpreterCallInterpretingEvent()
                | responses.ResponseImageGenCallCompletedEvent()
                | responses.ResponseImageGenCallGeneratingEvent()
                | responses.ResponseImageGenCallInProgressEvent()
                | responses.ResponseMcpListToolsInProgressEvent()
                | responses.ResponseMcpListToolsCompletedEvent()
                | responses.ResponseMcpListToolsFailedEvent()
                | responses.ResponseMcpCallInProgressEvent()
                | responses.ResponseMcpCallFailedEvent()
                | responses.ResponseMcpCallCompletedEvent()
                | responses.ResponseFileSearchCallCompletedEvent()
                | responses.ResponseFileSearchCallSearchingEvent()
                | responses.ResponseFileSearchCallInProgressEvent()
            ):
                pass  # there's nothing we need to do here
            case _:  # pragma: no cover
                warnings.warn(
                    f'Handling of this event type is not yet implemented. Please report on our GitHub: {chunk}',
                    UserWarning,
                )

    def _map_completed(self, chunk: responses.ResponseCompletedEvent) -> Iterator[ModelResponseStreamEvent]:
        # Only the return part is backfilled; the call part is already emitted via `output_item.added`.
        # Backfill mcp_list_tools results missing from streamed output_item.done events (see https://github.com/pydantic/pydantic-ai/issues/5419).
        for item in chunk.response.output:
            if (
                isinstance(item, responses.response_output_item.McpListTools)
                and item.id not in self._mcp_list_tools_return_ids
            ):
                _, return_part = map_mcp_list_tools(item, self.provider_name)
                yield self._parts_manager.handle_part(vendor_part_id=f'{item.id}-return', part=return_part)
                self._mcp_list_tools_return_ids.add(item.id)

        self._record_usage(chunk.response)
        self._store_conversation_id(chunk.response)
        self._set_state(chunk.response.status)

        raw_finish_reason = details.reason if (details := chunk.response.incomplete_details) else chunk.response.status

        if raw_finish_reason:  # pragma: no branch
            if not self._has_refusal:
                self.provider_details = {
                    **(self.provider_details or {}),
                    'finish_reason': raw_finish_reason,
                }
                self.finish_reason = RESPONSES_FINISH_REASON_MAP.get(raw_finish_reason)

        if chunk.response.moderation:
            self.provider_details = {
                **(self.provider_details or {}),
                'moderation': chunk.response.moderation.model_dump(),
            }

    def _map_created(self, chunk: responses.ResponseCreatedEvent) -> None:
        self._set_state(chunk.response.status)
        if chunk.response.id:  # pragma: no branch
            self.provider_response_id = chunk.response.id
        self._store_conversation_id(chunk.response)

    def _map_failed(self, chunk: responses.ResponseFailedEvent) -> None:
        self._record_usage(chunk.response)
        # Record the terminal state first, so a failed background job isn't cancelled after the raise.
        self._set_state(chunk.response.status)
        if error := chunk.response.error:
            raise response_error(
                self._model_name,
                error.code,
                error.message,
                error_type=response_error_type(error),
                in_stream=True,
            )
        # Parity with the non-streaming `_process_response`: a `failed` status maps to 'error'.
        if not self._has_refusal:
            self.provider_details = {**(self.provider_details or {}), 'finish_reason': 'failed'}
            self.finish_reason = RESPONSES_FINISH_REASON_MAP.get('failed')

    def _map_incomplete(self, chunk: responses.ResponseIncompleteEvent) -> None:
        self._record_usage(chunk.response)
        # Parity with the non-streaming `_process_response`: map the incomplete
        # reason when the provider sends one, leave the finish reason unset otherwise.
        raw_finish_reason = details.reason if (details := chunk.response.incomplete_details) else None
        if raw_finish_reason and not self._has_refusal:
            self.provider_details = {**(self.provider_details or {}), 'finish_reason': raw_finish_reason}
            self.finish_reason = RESPONSES_FINISH_REASON_MAP.get(raw_finish_reason)
        self._set_state(chunk.response.status)

    def _map_in_progress(self, chunk: responses.ResponseInProgressEvent | responses.ResponseQueuedEvent) -> None:
        self._record_usage(chunk.response)
        self._set_state(chunk.response.status)

    def _map_function_call_arguments_delta(
        self, chunk: responses.ResponseFunctionCallArgumentsDeltaEvent
    ) -> Iterator[ModelResponseStreamEvent]:
        # Without `output_item.added` (a resumed stream) the part has no tool name yet, so
        # skip its deltas and let `output_item.done` create it; never mutate finalized args.
        if chunk.item_id in self._function_call_args and chunk.item_id not in self._finalized_function_calls:
            self._function_call_args[chunk.item_id] += chunk.delta
            yield from self._parts_manager.handle_tool_call_delta_events(
                vendor_part_id=chunk.item_id,
                args=chunk.delta,
            )

    def _handle_function_call_part(self, item: responses.ResponseFunctionToolCall) -> ModelResponseStreamEvent:
        item_id = item.id
        if item_id is not None:  # pragma: no branch
            self._function_call_args[item_id] = item.arguments
        # Preserve any namespace attached to a discovered deferred tool for replay.
        provider_details: dict[str, Any] | None = None
        if item.namespace:
            provider_details = {'namespace': item.namespace}
        return self._parts_manager.handle_tool_call_part(
            vendor_part_id=item_id,
            tool_name=item.name,
            args=item.arguments,
            tool_call_id=response_tool_call_id(
                item.call_id,
                self.provider_response_id if self._tool_call_ids_are_response_scoped else None,
            ),
            id=item_id,
            provider_name=self.provider_name,
            provider_details=provider_details,
        )

    def _finalize_function_call(self, item_id: str, final_args: str) -> Iterator[ModelResponseStreamEvent]:
        if item_id in self._finalized_function_calls or not final_args:
            return
        existing_part = self._parts_manager.get_part_by_vendor_id(item_id)
        if not isinstance(existing_part, ToolCallPart):
            # A stream resumed after `output_item.added` has no complete part yet;
            # `output_item.done` creates it from the full item.
            return

        received_args = self._function_call_args.get(item_id, '')
        if received_args == final_args:
            self._finalized_function_calls.add(item_id)
            return

        if final_args.startswith(received_args):
            # The Codex backend can send arguments only in the done events.
            yield from self._parts_manager.handle_tool_call_delta_events(
                vendor_part_id=item_id,
                args=final_args[len(received_args) :],
            )
        else:
            # Replacing the part would emit a second `PartStartEvent` for the same call,
            # which UI event streams render as a duplicate tool call, so keep what streamed.
            self._finalized_function_calls.add(item_id)
            warnings.warn(
                'The provider sent a `function_call_arguments.done`/`output_item.done` snapshot whose '
                f'arguments differ from the streamed argument deltas for function call item {item_id!r} '
                f'(tool {existing_part.tool_name!r}); the streamed arguments were kept. '
                'Please open an issue at https://github.com/pydantic/pydantic-ai/issues.',
                UserWarning,
            )
            return

        self._function_call_args[item_id] = final_args
        self._finalized_function_calls.add(item_id)

    def _map_output_item_added(
        self, chunk: responses.ResponseOutputItemAddedEvent
    ) -> Iterator[ModelResponseStreamEvent]:
        item = chunk.item
        match item:
            case responses.ResponseFunctionToolCall():
                yield self._handle_function_call_part(item)
            case responses.ResponseComputerToolCall() | responses.ResponseReasoningItem():
                # A computer call's actions arrive whole in its done event, where it's mapped.
                pass
            case responses.ResponseOutputMessage():
                if item.phase is not None:
                    self._phase_by_item[item.id] = item.phase
            case responses.ResponseFunctionWebSearch():
                yield self._map_web_search_call_added(item)
            case responses.ResponseToolSearchCall():
                yield self._map_tool_search_call_added(item)
            case responses.ResponseFileSearchToolCall():
                yield self._map_file_search_call_added(item)
            case responses.ResponseToolSearchOutputItem():
                # The completed result is emitted from `output_item.done`.
                pass
            case responses.ResponseCodeInterpreterToolCall():
                yield from self._map_code_interpreter_call_added(item)
            case responses.response_output_item.ImageGenerationCall():
                yield self._map_image_generation_call_added(item)
            case responses.response_output_item.McpCall():
                yield from self._map_mcp_call_added(item)
            case responses.response_output_item.McpListTools():
                yield self._map_mcp_list_tools_added(item)
            case ResponseCompactionItem():
                # Emit a PartStartEvent so UIs can render compaction in progress.
                # The "done" event replaces this with the final encrypted_content.
                yield self._parts_manager.handle_part(
                    vendor_part_id=item.id,
                    part=map_compaction_item(item, self.provider_name),
                )
            case _:  # pragma: no cover
                warnings.warn(
                    f'Handling of this item type is not yet implemented. Please report on our GitHub: {chunk}',
                    UserWarning,
                )

    def _map_web_search_call_added(self, item: responses.ResponseFunctionWebSearch) -> ModelResponseStreamEvent:
        call_part, _ = map_web_search_tool_call(item, self.provider_name)
        return self._parts_manager.handle_part(vendor_part_id=f'{item.id}-call', part=replace(call_part, args=None))

    def _map_tool_search_call_added(self, item: responses.ResponseToolSearchCall) -> ModelResponseStreamEvent:
        """Start the tool search call part, with its arguments left for `output_item.done` to fill in."""
        if item.execution == 'client':
            # Emit a regular function-tool call so the standard
            # tool-execution path runs our local `search_tools`.
            client_call_part = map_client_tool_search_call(item, self.provider_name)
            return self._parts_manager.handle_tool_call_part(
                vendor_part_id=item.id,
                tool_name=client_call_part.tool_name,
                args=None,
                tool_call_id=client_call_part.tool_call_id,
                id=client_call_part.id,
                provider_name=self.provider_name,
            )
        call_part = map_tool_search_call(item, self.provider_name)
        if item.call_id is None:
            self._pending_tool_search_call_ids.append(call_part.tool_call_id)
        return self._parts_manager.handle_part(vendor_part_id=f'{item.id}-call', part=replace(call_part, args=None))

    def _map_file_search_call_added(self, item: responses.ResponseFileSearchToolCall) -> ModelResponseStreamEvent:
        call_part, _ = map_file_search_tool_call(item, self.provider_name)
        return self._parts_manager.handle_part(vendor_part_id=f'{item.id}-call', part=replace(call_part, args=None))

    def _map_code_interpreter_call_added(
        self, item: responses.ResponseCodeInterpreterToolCall
    ) -> Iterator[ModelResponseStreamEvent]:
        call_part, _, _ = map_code_interpreter_tool_call(item, self.provider_name)

        args_json = call_part.args_as_json_str()
        # Drop the final `"}` so that we can add code deltas
        args_json_delta = args_json[:-2]

        yield self._parts_manager.handle_part(vendor_part_id=f'{item.id}-call', part=replace(call_part, args=None))
        yield from self._parts_manager.handle_tool_call_delta_events(
            vendor_part_id=f'{item.id}-call',
            args=args_json_delta,
        )

    def _map_image_generation_call_added(
        self, item: responses.response_output_item.ImageGenerationCall
    ) -> ModelResponseStreamEvent:
        call_part, _, _ = map_image_generation_tool_call(item, self.provider_name)
        return self._parts_manager.handle_part(vendor_part_id=f'{item.id}-call', part=call_part)

    def _map_mcp_call_added(self, item: responses.response_output_item.McpCall) -> Iterator[ModelResponseStreamEvent]:
        call_part, _ = map_mcp_call(item, self.provider_name)

        args_json = call_part.args_as_json_str()
        # Drop the final `{}}` so that we can add tool args deltas
        args_json_delta = args_json[:-3]
        assert args_json_delta.endswith('"tool_args":'), f'Expected {args_json_delta!r} to end in `"tool_args":"`'

        yield self._parts_manager.handle_part(vendor_part_id=f'{item.id}-call', part=replace(call_part, args=None))
        yield from self._parts_manager.handle_tool_call_delta_events(
            vendor_part_id=f'{item.id}-call',
            args=args_json_delta,
        )

    def _map_mcp_list_tools_added(self, item: responses.response_output_item.McpListTools) -> ModelResponseStreamEvent:
        call_part, _ = map_mcp_list_tools(item, self.provider_name)
        return self._parts_manager.handle_part(vendor_part_id=f'{item.id}-call', part=call_part)

    def _map_output_item_done(self, item: responses.ResponseOutputItem) -> Iterator[ModelResponseStreamEvent]:
        match item:
            case responses.ResponseFunctionToolCall(id=str(item_id)) if item_id:
                yield from self._map_function_call_done(item, item_id)
            case responses.ResponseComputerToolCall():
                yield from self._map_computer_call_done(item)
            case responses.ResponseReasoningItem():
                if signature := item.encrypted_content:  # pragma: no branch
                    # Add the signature to the part corresponding to the first summary/raw CoT
                    yield from self._parts_manager.handle_thinking_delta(
                        vendor_part_id=item.id,
                        id=item.id,
                        signature=signature,
                        provider_name=self.provider_name,
                    )
            case responses.ResponseCodeInterpreterToolCall():
                yield from self._map_code_interpreter_call_done(item)
            case responses.ResponseFunctionWebSearch():
                yield from self._map_web_search_call_done(item)
            case responses.ResponseToolSearchCall():
                yield from self._handle_tool_search_call_done(item)
            case responses.ResponseToolSearchOutputItem():
                if item.execution == 'server':
                    yield self._map_tool_search_output_done(item)
            case responses.ResponseFileSearchToolCall():
                yield from self._map_file_search_call_done(item)
            case responses.response_output_item.ImageGenerationCall():
                yield from self._map_image_generation_call_done(item)
            case responses.response_output_item.McpCall():
                yield self._map_mcp_call_done(item)
            case responses.response_output_item.McpListTools():
                yield from self._map_mcp_list_tools_done(item)
            case ResponseCompactionItem():
                # Replace the preliminary part from the "added" event with the
                # final encrypted_content for round-tripping.
                yield self._parts_manager.handle_part(
                    vendor_part_id=item.id,
                    part=map_compaction_item(item, self.provider_name),
                )
            case _:
                pass

    def _computer_call_id(self, item: responses.ResponseComputerToolCall) -> str:
        return response_tool_call_id(
            item.call_id, self.provider_response_id if self._tool_call_ids_are_response_scoped else None
        )

    def _map_computer_call_done(self, item: responses.ResponseComputerToolCall) -> Iterator[ModelResponseStreamEvent]:
        if not (computer_name := self.model_request_parameters.tool_name_for_kind(COMPUTER_KIND)):
            return
        # Its actions arrive whole in the done event, and OpenAI finishes each item before adding the next, so the
        # call starts complete here and keeps its place among the other parts.
        call = map_computer_call(item, computer_name, self._computer_call_id(item), self.provider_name)
        yield self._parts_manager.handle_part(vendor_part_id=item.id, part=call)

    def _map_function_call_done(
        self, item: responses.ResponseFunctionToolCall, item_id: str
    ) -> Iterator[ModelResponseStreamEvent]:
        # The Codex backend can send a call's arguments only in the done events, without deltas.
        if not isinstance(self._parts_manager.get_part_by_vendor_id(item_id), ToolCallPart):
            yield self._handle_function_call_part(item)
            self._finalized_function_calls.add(item_id)
        else:
            yield from self._finalize_function_call(item_id, item.arguments)

    def _map_code_interpreter_call_done(
        self, item: responses.ResponseCodeInterpreterToolCall
    ) -> Iterator[ModelResponseStreamEvent]:
        _, return_part, file_parts = map_code_interpreter_tool_call(item, self.provider_name)
        for i, file_part in enumerate(file_parts):
            yield self._parts_manager.handle_part(vendor_part_id=f'{item.id}-file-{i}', part=file_part)
        yield self._parts_manager.handle_part(vendor_part_id=f'{item.id}-return', part=return_part)

    def _map_web_search_call_done(
        self, item: responses.ResponseFunctionWebSearch
    ) -> Iterator[ModelResponseStreamEvent]:
        call_part, return_part = map_web_search_tool_call(item, self.provider_name)

        yield from self._parts_manager.handle_tool_call_delta_events(
            vendor_part_id=f'{item.id}-call',
            args=call_part.args,
        )

        yield self._parts_manager.handle_part(vendor_part_id=f'{item.id}-return', part=return_part)

    def _map_tool_search_output_done(self, item: responses.ResponseToolSearchOutputItem) -> ModelResponseStreamEvent:
        """Return a server-executed tool search's result, paired with the call it answers."""
        call_id = match_tool_search_output_to_call(item.call_id, self._pending_tool_search_call_ids)
        return self._parts_manager.handle_part(
            vendor_part_id=f'{item.id}-return',
            part=build_tool_search_return_part(
                call_id if call_id is not None else item.id,
                item,
                self.provider_name,
            ),
        )

    def _map_file_search_call_done(
        self, item: responses.ResponseFileSearchToolCall
    ) -> Iterator[ModelResponseStreamEvent]:
        call_part, return_part = map_file_search_tool_call(item, self.provider_name)

        yield from self._parts_manager.handle_tool_call_delta_events(
            vendor_part_id=f'{item.id}-call',
            args=call_part.args,
        )

        yield self._parts_manager.handle_part(vendor_part_id=f'{item.id}-return', part=return_part)

    def _map_image_generation_call_done(
        self, item: responses.response_output_item.ImageGenerationCall
    ) -> Iterator[ModelResponseStreamEvent]:
        _, return_part, file_part = map_image_generation_tool_call(item, self.provider_name)
        if file_part:  # pragma: no branch
            yield self._parts_manager.handle_part(vendor_part_id=f'{item.id}-file', part=file_part)
        yield self._parts_manager.handle_part(vendor_part_id=f'{item.id}-return', part=return_part)

    def _map_mcp_call_done(self, item: responses.response_output_item.McpCall) -> ModelResponseStreamEvent:
        _, return_part = map_mcp_call(item, self.provider_name)
        return self._parts_manager.handle_part(vendor_part_id=f'{item.id}-return', part=return_part)

    def _map_mcp_list_tools_done(
        self, item: responses.response_output_item.McpListTools
    ) -> Iterator[ModelResponseStreamEvent]:
        _, return_part = map_mcp_list_tools(item, self.provider_name)
        yield self._parts_manager.handle_part(vendor_part_id=f'{item.id}-return', part=return_part)
        self._mcp_list_tools_return_ids.add(item.id)

    def _map_reasoning_summary_delta(
        self, item_id: str, summary_index: int, content: str
    ) -> Iterator[ModelResponseStreamEvent]:
        # Use same vendor_part_id as raw CoT for first summary (index 0) so they merge into one ThinkingPart
        vendor_id = item_id if summary_index == 0 else f'{item_id}-{summary_index}'
        yield from self._parts_manager.handle_thinking_delta(
            vendor_part_id=vendor_id,
            content=content,
            id=item_id,
            provider_name=self.provider_name,
        )

    def _map_refusal_delta(self, chunk: responses.ResponseRefusalDeltaEvent) -> None:
        # Accumulate refusal text from deltas as a fallback in case the done event is missing.
        self._has_refusal = True
        self.finish_reason = 'content_filter'
        self._refusal_text += chunk.delta

    def _map_refusal_done(self, chunk: responses.ResponseRefusalDoneEvent) -> None:
        # The done event contains the full refusal text, replacing any accumulated deltas.
        self._has_refusal = True
        self.finish_reason = 'content_filter'
        self._refusal_text = chunk.refusal

    def _map_code_interpreter_code_delta(
        self, chunk: responses.ResponseCodeInterpreterCallCodeDeltaEvent
    ) -> Iterator[ModelResponseStreamEvent]:
        json_args_delta = to_json(chunk.delta).decode()[1:-1]  # Drop the surrounding `"`
        yield from self._parts_manager.handle_tool_call_delta_events(
            vendor_part_id=f'{chunk.item_id}-call',
            args=json_args_delta,
        )

    def _map_partial_image(self, chunk: responses.ResponseImageGenCallPartialImageEvent) -> ModelResponseStreamEvent:
        # Not present on the type, but present on the actual object.
        # See https://github.com/openai/openai-python/issues/2649
        output_format = getattr(chunk, 'output_format', 'png')
        file_part = FilePart(
            content=BinaryImage(
                data=base64.b64decode(chunk.partial_image_b64),
                media_type=f'image/{output_format}',
            ),
            id=chunk.item_id,
        )
        return self._parts_manager.handle_part(vendor_part_id=f'{chunk.item_id}-file', part=file_part)

    def _map_text_delta(self, chunk: responses.ResponseTextDeltaEvent) -> Iterator[ModelResponseStreamEvent]:
        # Guard against delta=null from OpenAI-compatible gateways (e.g. Bifrost).
        if chunk.delta is not None:  # pyright: ignore[reportUnnecessaryComparison]
            # Attach the message-level phase once to each content part.
            delta_provider_details: dict[str, Any] | None = None
            content_key = (chunk.item_id, chunk.content_index)
            if (
                content_key not in self._phase_started_content
                and (phase := self._phase_by_item.get(chunk.item_id)) is not None
            ):
                delta_provider_details = {'phase': phase}
                self._phase_started_content.add(content_key)
            yield from self._parts_manager.handle_text_delta(
                vendor_part_id=content_key,
                content=chunk.delta,
                id=chunk.item_id,
                provider_name=self.provider_name,
                provider_details=delta_provider_details,
            )

    def _map_text_done(self, chunk: responses.ResponseTextDoneEvent) -> Iterator[ModelResponseStreamEvent]:
        # Add annotations to provider_details if available
        provider_details: dict[str, Any] = {}
        content_key = (chunk.item_id, chunk.content_index)
        raw_annotations = self._annotations_by_item.pop(content_key, [])
        if raw_annotations and self._model_settings.get('openai_include_raw_annotations'):
            provider_details['annotations'] = raw_annotations
        annotations: list[responses.response_output_text.Annotation] = []
        for raw_annotation in raw_annotations:
            try:
                annotations.append(responses_output_text_annotation_ta.validate_python(raw_annotation))
            except ValidationError:
                pass  # an annotation type the SDK doesn't know
        if chunk.logprobs:
            provider_details['logprobs'] = map_logprobs(chunk.logprobs)
        if (
            content_key not in self._phase_started_content
            and (phase := self._phase_by_item.get(chunk.item_id)) is not None
        ):
            provider_details['phase'] = phase
            self._phase_started_content.add(content_key)
        existing_part = self._parts_manager.get_part_by_vendor_id(content_key)
        # Annotation positions count into the final text, so they can't go on streamed text that differs.
        citations = (
            map_responses_citations(annotations, chunk.text)
            if isinstance(existing_part, TextPart) and existing_part.content == chunk.text
            else None
        )
        if citations:
            provider_details.update(responses_citations_provider_details(annotations))
        # Empty text makes no part, as in `_process_response`.
        if provider_details and chunk.text:
            yield from self._parts_manager.handle_text_delta(
                vendor_part_id=content_key,
                content='',
                provider_name=self.provider_name,
                provider_details=provider_details,
                citations=citations,
            )

    def _handle_tool_search_call_done(
        self, item: responses.ResponseToolSearchCall
    ) -> Iterator[ModelResponseStreamEvent]:
        """Complete the tool search call part `output_item.added` started with the item's final arguments."""
        if item.execution == 'client':
            # Feed the final args through the tool-call delta path so the
            # part manager resolves its pending args; there's no paired
            # `tool_search_output` — the return part is produced by the
            # local tool runner after the stream completes.
            #
            # OpenAI Responses streaming sometimes assigns a different
            # `call_id` between the `output_item.added` and `output_item.done`
            # frames for client-executed `tool_search_call`s; the server's
            # log of record is the latter. Pass it through here so the
            # `tool_search_output` we replay on the next turn references the
            # call_id the API actually expects.
            client_call_part = map_client_tool_search_call(item, self.provider_name)
            client_args_json = client_call_part.args_as_json_str()
            yield from self._parts_manager.handle_tool_call_delta_events(
                vendor_part_id=item.id,
                args=client_args_json,
                tool_call_id=client_call_part.tool_call_id,
            )
        else:
            call_part = map_tool_search_call(item, self.provider_name)
            yield from self._parts_manager.handle_tool_call_delta_events(
                vendor_part_id=f'{item.id}-call',
                args=cast('str | dict[str, Any] | None', call_part.args),
                provider_details=call_part.provider_details,
            )

    def _store_conversation_id(self, response: responses.Response) -> None:
        if response.conversation:
            self.provider_details = {**(self.provider_details or {}), 'conversation_id': response.conversation.id}

    def _map_usage(self, response: responses.Response) -> usage.RequestUsage:
        return map_usage(response, self._provider_name, self._provider_url, self.model_name)

    def _record_usage(self, response: responses.Response) -> None:
        self._add_usage(self._map_usage(response), reported=response.usage is not None)

    @property
    def model_name(self) -> OpenAIModelName:
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


def make_raw_content_updater(delta: str, index: int) -> Callable[[dict[str, Any] | None], dict[str, Any]]:
    """Create a callback that updates `provider_details['raw_content']`.

    This is used for streaming raw CoT from gpt-oss models. The callback pattern keeps
    `raw_content` logic in OpenAI code while the parts manager stays provider-agnostic.
    """

    def update_provider_details(existing: dict[str, Any] | None) -> dict[str, Any]:
        details = {**(existing or {})}
        raw_list: list[str] = list(details.get('raw_content', []))
        while len(raw_list) <= index:
            raw_list.append('')
        raw_list[index] += delta
        details['raw_content'] = raw_list
        return details

    return update_provider_details
