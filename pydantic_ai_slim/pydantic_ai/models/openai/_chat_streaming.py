"""`OpenAIStreamedResponse`, the Chat Completions stream."""

from __future__ import annotations as _annotations

import itertools
import warnings
from collections.abc import AsyncIterable, AsyncIterator, Iterable, Iterator
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal, cast

from openai import AsyncStream
from openai.types import chat
from openai.types.chat import ChatCompletionChunk, chat_completion_chunk
from pydantic import ValidationError

from ... import ModelAPIError, _utils, usage
from ..._utils import now_utc as _now_utc, number_to_datetime
from ...messages import FinishReason, ModelResponseStreamEvent, PartStartEvent, TextPart, ThinkingPart
from ...profiles import DEFAULT_THINKING_TAGS
from ...profiles.openai import OpenAIModelProfile
from .. import StreamedResponse
from .._decode_errors import MapStreamDecodeErrors
from .._tool_call_ids import stored_tool_call_id
from ._chat_completions import (
    CHAT_FINISH_REASON_MAP,
    map_provider_details,
    map_tool_call_provider_details,
    split_content_parts,
    unmapped_choice_data,
)
from ._citations import (
    chat_annotation_ta,
    chat_citations_provider_details,
    jsonable_annotation,
    map_chat_citations,
)
from ._errors import map_api_errors
from ._model_names import OpenAIModelName
from ._response_fields import map_usage
from ._settings import OpenAIChatModelSettings


@dataclass
class OpenAIStreamedResponse(StreamedResponse):
    """Implementation of `StreamedResponse` for OpenAI models."""

    _model_name: OpenAIModelName
    _model_profile: OpenAIModelProfile
    _response: _utils.PeekableAsyncStream[ChatCompletionChunk, AsyncStream[ChatCompletionChunk]]
    _provider_name: str
    _model_id_namespace: str
    _provider_url: str
    _provider_timestamp: datetime | None = None
    _timestamp: datetime = field(default_factory=_now_utc)
    _model_settings: OpenAIChatModelSettings | None = None
    _usage_received: bool | None = field(default=False, init=False)
    # Content segments are anonymous; reasoning segments also carry their field identity.
    _content_generation: int = field(default=0, init=False)
    _reasoning_generation: int = field(default=0, init=False)
    _held_text_whitespace: str = field(default='', init=False)
    _awaiting_text_after_reasoning: bool = field(default=False, init=False)
    _has_refusal: bool = field(default=False, init=False)
    _refusal_text: str = field(default='', init=False)
    _has_finish_reason: bool = field(default=False, init=False)
    _logprobs: list[dict[str, Any]] = field(default_factory=list[dict[str, Any]], init=False)
    _raw_text_content: str = field(default='', init=False)
    _pending_raw_annotations: list[object] = field(default_factory=list[object], init=False)
    _unmapped_chunks: list[dict[str, Any]] = field(default_factory=list[dict[str, Any]], init=False)

    async def close_stream(self) -> None:
        await self._response.source.close()

    async def _get_event_iterator(self) -> AsyncIterator[ModelResponseStreamEvent]:
        with map_api_errors(self._model_name, self._model_id_namespace):
            async for chunk in MapStreamDecodeErrors(self._validate_response(), self._model_name):
                for event in self._map_chunk(chunk):
                    yield event

            if self._refusal_text:
                self.provider_details = {**(self.provider_details or {}), 'refusal': self._refusal_text}
            if self._logprobs:
                self.provider_details = {**(self.provider_details or {}), 'logprobs': self._logprobs}
            self._publish_unmapped_chunks()
            if (
                self._model_profile.get('openai_chat_streaming_requires_finish_reason', False)
                and not self._has_finish_reason
                and not self.cancelled
            ):
                raise ModelAPIError(
                    model_name=self.model_name,
                    message='Streamed response ended without a `finish_reason`',
                )
            if not self._has_finish_reason and not self._has_refusal and not self.cancelled:
                # Some OpenAI-compatible providers never send a finish reason.
                self.finish_reason = self._missing_finish_reason()
            for event in self._flush_pending_chat_annotations():
                yield event

    def _map_chunk(self, chunk: ChatCompletionChunk) -> Iterator[ModelResponseStreamEvent]:
        self._map_top_level_fields(chunk)

        # Empty on the final usage-only chunk; `None` from OpenAI-compatible providers emitting
        # malformed chunks that the openai SDK's loose constructor lets through (https://github.com/pydantic/pydantic-ai/issues/5165).
        if not chunk.choices:
            return
        choice = chunk.choices[0]
        raw_finish_reason = choice.finish_reason
        self._has_finish_reason = self._has_finish_reason or bool(raw_finish_reason)

        # When using Azure OpenAI and an async content filter is enabled, the openai SDK can return None deltas.
        if choice.delta is None:  # pyright: ignore[reportUnnecessaryComparison]
            return

        # Handle refusal responses (structured output safety filter).
        # Note: OpenAI sends refusal instead of content (not alongside it), so in practice
        # text parts won't have been yielded before _has_refusal is set.
        if choice.delta.refusal:
            self._has_refusal = True
            self.finish_reason = 'content_filter'
            self._refusal_text += choice.delta.refusal
            return

        if raw_finish_reason and not self._has_refusal:
            self.finish_reason = self._map_finish_reason(raw_finish_reason)

        if provider_details := self._map_provider_details(chunk):  # pragma: no branch
            if self._has_refusal:
                provider_details.pop('finish_reason', None)
            self.provider_details = {**(self.provider_details or {}), **provider_details}

        yield from self._map_part_delta(choice)

    def _map_top_level_fields(self, chunk: ChatCompletionChunk) -> None:
        """Map the fields of `chunk` outside its choices, which chunks without choices carry too."""
        if self._provider_timestamp is None and chunk.created:
            self._provider_timestamp = number_to_datetime(chunk.created)
            self.provider_details = {
                **(self.provider_details or {}),
                'timestamp': self._provider_timestamp,
            }

        # When continuous_usage_stats is enabled, each chunk contains cumulative usage,
        # so we replace rather than increment to avoid double-counting.
        cumulative = bool(self._model_settings and self._model_settings.get('openai_continuous_usage_stats'))
        if chunk.usage is not None or not cumulative:
            self._add_usage(self._map_usage(chunk), reported=chunk.usage is not None, cumulative=cumulative)

        if chunk.id:  # pragma: no branch
            self.provider_response_id = chunk.id

        if chunk.model:
            self._model_name = chunk.model

        # The moderation chunk carries no choices.
        if chunk.moderation:
            self.provider_details = {
                **(self.provider_details or {}),
                'moderation': chunk.moderation.model_dump(),
            }
        if chunk.service_tier:
            self.provider_details = {**(self.provider_details or {}), 'service_tier': chunk.service_tier}
        if chunk_provider_details := self._map_chunk_provider_details(chunk):
            self.provider_details = {**(self.provider_details or {}), **chunk_provider_details}
        self._record_unmapped_chunk_data(chunk)

    def _flush_pending_chat_annotations(self) -> Iterator[ModelResponseStreamEvent]:
        if not self._pending_raw_annotations:
            return

        text_part_ids = [
            vendor_part_id
            for generation in range(self._content_generation + 1)
            if isinstance(
                self._parts_manager.get_part_by_vendor_id(vendor_part_id := ('content', generation)), TextPart
            )
        ]
        # Annotations belong to text, so a stream that ended without any (such as only thinking) has nowhere to put them.
        if not text_part_ids:
            return
        annotations: list[chat.chat_completion_message.Annotation] = []
        serialized_annotations: list[object] = []
        for raw_annotation in self._pending_raw_annotations:
            try:
                annotation = chat_annotation_ta.validate_python(raw_annotation)
            except ValidationError:
                # OpenRouter's stream parses its own annotation types, such as `file`, into models.
                serialized_annotations.append(jsonable_annotation(raw_annotation))
            else:
                annotations.append(annotation)
                serialized_annotations.append(chat_annotation_ta.dump_python(annotation, warnings=False))

        part = self._parts_manager.get_part_by_vendor_id(text_part_ids[-1])
        assert isinstance(part, TextPart)
        # Annotation positions count into the message's original text, so citations only go on a single text part that
        # still equals it: not split by reasoning, and not changed by removing thinking tags or whitespace.
        citations = (
            map_chat_citations(annotations, part.content)
            if len(text_part_ids) == 1 and part.content == self._raw_text_content
            else None
        )
        provider_details: dict[str, Any] = {}
        if self._model_settings and self._model_settings.get('openai_include_raw_annotations'):
            provider_details['annotations'] = serialized_annotations
        if citations:
            provider_details.update(chat_citations_provider_details(annotations))
        if not provider_details:
            return
        yield from self._parts_manager.handle_text_delta(
            vendor_part_id=text_part_ids[-1],
            content='',
            provider_name=self.provider_name,
            provider_details=provider_details,
            citations=citations,
        )

    def _validate_response(self) -> AsyncIterable[ChatCompletionChunk]:
        """Hook that validates incoming chunks.

        This method may be overridden by subclasses of `OpenAIStreamedResponse` to apply custom chunk validations.

        By default, this is a no-op since `ChatCompletionChunk` is already validated.
        """
        return self._response

    def _map_part_delta(self, choice: chat_completion_chunk.Choice) -> Iterable[ModelResponseStreamEvent]:
        """Hook that determines the sequence of mappings that will be called to produce events.

        This method may be overridden by subclasses of `OpenAIStreamResponse` to customize the mapping.
        """
        return itertools.chain(
            self._map_thinking_delta(choice), self._map_text_delta(choice), self._map_tool_call_delta(choice)
        )

    def _advance_part_boundary(self, source: Literal['reasoning', 'content', 'tool']) -> None:
        if source != 'reasoning':
            self._reasoning_generation += 1
            self._awaiting_text_after_reasoning = False

        # Keep the content key while it names a ThinkingPart so closing tags retain their context.
        if source != 'content' and isinstance(
            self._parts_manager.get_part_by_vendor_id(('content', self._content_generation)), TextPart
        ):
            self._content_generation += 1
            self._awaiting_text_after_reasoning = source == 'reasoning'

    def _map_thinking_delta(self, choice: chat_completion_chunk.Choice) -> Iterable[ModelResponseStreamEvent]:
        """Hook that maps thinking delta content to events.

        This method may be overridden by subclasses of `OpenAIStreamResponse` to customize the mapping.
        """
        profile = self._model_profile
        custom_field = profile.get('openai_chat_thinking_field', None)

        # Prefer the configured custom reasoning field, if present in profile.
        # Fall back to built-in fields if no custom field result was found.

        # The `reasoning_content` field is typically present in DeepSeek and Moonshot models.
        # https://api-docs.deepseek.com/guides/reasoning_model

        # The `reasoning` field is typically present in gpt-oss via Ollama and OpenRouter.
        # - https://cookbook.openai.com/articles/gpt-oss/handle-raw-cot#chat-completions-api
        # - https://openrouter.ai/docs/use-cases/reasoning-tokens#basic-usage-with-reasoning-tokens
        for field_name in (custom_field, 'reasoning', 'reasoning_content'):
            if not field_name:
                continue
            reasoning: object = getattr(choice.delta, field_name, None)
            if not reasoning:
                continue
            if not isinstance(reasoning, str):
                warnings.warn(
                    f'Unexpected non-string value for {field_name!r}: {type(reasoning).__name__}. '
                    'Please open an issue at https://github.com/pydantic/pydantic-ai/issues.',
                    UserWarning,
                )
                continue
            for event in self._parts_manager.handle_thinking_delta(
                vendor_part_id=('reasoning', field_name, self._reasoning_generation),
                id=field_name,
                content=reasoning,
                provider_name=self.provider_name,
            ):
                if isinstance(event, PartStartEvent):
                    self._advance_part_boundary('reasoning')
                yield event
            break

    def _map_text_delta(self, choice: chat_completion_chunk.Choice) -> Iterable[ModelResponseStreamEvent]:
        """Hook that maps text delta content to events.

        This method may be overridden by subclasses of `OpenAIStreamResponse` to customize the mapping.
        """
        # Handle the text part of the response. Some OpenAI-compatible servers stream it as a list of parts.
        content, _ = split_content_parts(choice.delta.content)
        if isinstance(content, str) and content:
            self._raw_text_content += content
            emitted = False
            for event in self._parts_manager.handle_text_delta(
                vendor_part_id=('content', self._content_generation),
                content=content,
                thinking_tags=self._model_profile.get('thinking_tags', DEFAULT_THINKING_TAGS),
                ignore_leading_whitespace=self._model_profile.get('ignore_streamed_leading_whitespace', False),
            ):
                emitted = True
                if isinstance(event, PartStartEvent):
                    self._start_content_part(event)
                yield event
            if not emitted and content.isspace() and self._awaiting_text_after_reasoning:
                # `ignore_streamed_leading_whitespace` dropped the whitespace-only start of text right after reasoning
                # that interrupted a text part. Hold it for the next text part, even across a tool call or `<think>`
                # part, so the joined text keeps the separator; if no text follows, it stays dropped. Leading
                # whitespace anywhere else is dropped as usual.
                self._held_text_whitespace += content

        # Chat chunk deltas don't declare annotations, so they arrive as an extra field.
        raw_annotations = getattr(choice.delta, 'annotations', None)
        if isinstance(raw_annotations, list):
            self._pending_raw_annotations.extend(cast(list[object], raw_annotations))

    def _start_content_part(self, event: PartStartEvent) -> None:
        """Set up a part the `content` field starts, before its `PartStartEvent` is yielded."""
        self._advance_part_boundary('content')
        if isinstance(event.part, ThinkingPart):
            event.part.id = 'content'
            event.part.provider_name = self.provider_name
        elif isinstance(event.part, TextPart) and self._held_text_whitespace:
            event.part.content = self._held_text_whitespace + event.part.content
            self._held_text_whitespace = ''

    def _map_tool_call_delta(self, choice: chat_completion_chunk.Choice) -> Iterable[ModelResponseStreamEvent]:
        """Hook that maps tool call delta content to events.

        This method may be overridden by subclasses of `OpenAIStreamResponse` to customize the mapping.
        """
        for dtc in choice.delta.tool_calls or []:
            provider_details = map_tool_call_provider_details(dtc)
            tool_call_id = dtc.id and stored_tool_call_id(
                dtc.id,
                response_scoped=self._model_profile.get('tool_call_ids_are_response_scoped', False),
                provider_response_id=self.provider_response_id,
                timestamp=self.timestamp,
            )
            for event in self._parts_manager.handle_tool_call_delta_events(
                vendor_part_id=dtc.index,
                tool_name=dtc.function.name if dtc.function is not None else None,
                args=dtc.function.arguments if dtc.function is not None else None,
                tool_call_id=tool_call_id,
                provider_name=self.provider_name if provider_details else None,
                provider_details=provider_details,
            ):
                if isinstance(event, PartStartEvent):
                    self._advance_part_boundary('tool')
                yield event

    def _map_provider_details(self, chunk: ChatCompletionChunk) -> dict[str, Any] | None:
        """Hook that generates the provider details from chunk content.

        This method may be overridden by subclasses of `OpenAIStreamResponse` to customize the provider details.
        Overrides should call `super()` so `logprobs` are collected across chunks.
        """
        provider_details = map_provider_details(chunk.choices[0])
        if provider_details and (logprobs := provider_details.pop('logprobs', None)):
            # Each chunk carries only its own tokens' logprobs, so collect them and publish the full list
            # once the stream ends.
            self._logprobs.extend(logprobs)
        return provider_details or None

    def _missing_finish_reason(self) -> FinishReason:
        """The finish reason of a stream that ended without one, matching `OpenAIChatModel._missing_finish_reason`.

        This method may be overridden by subclasses of `OpenAIStreamResponse` whose model picks another one.
        """
        return 'stop'

    def _publish_unmapped_chunks(self) -> None:
        """Put the collected `_record_unmapped_chunk_data` into `provider_details['unmapped']`, once the stream ends."""
        if self._unmapped_chunks:
            self.provider_details = {**(self.provider_details or {}), 'unmapped': {'chunks': self._unmapped_chunks}}

    def _record_unmapped_chunk_data(self, chunk: ChatCompletionChunk) -> None:
        """Keep the data of `chunk` the mapping doesn't read, when `openai_include_unmapped_response_data` is on."""
        if not (self._model_settings and self._model_settings.get('openai_include_unmapped_response_data')):
            return
        choice = chunk.choices[0] if chunk.choices else None
        delta = getattr(choice, 'delta', None)
        _, content_parts = split_content_parts(getattr(delta, 'content', None))
        data = unmapped_choice_data(
            chunk,
            choice,
            delta,
            content_parts,
            message_key='delta',
            thinking_field=self._model_profile.get('openai_chat_thinking_field', None),
        )
        if data:
            self._unmapped_chunks.append(data)

    def _map_chunk_provider_details(self, chunk: ChatCompletionChunk) -> dict[str, Any] | None:
        """Hook that generates provider details from a chunk's top-level fields, rather than its choice.

        This method may be overridden by subclasses of `OpenAIStreamResponse` to customize the provider details.
        Unlike `_map_provider_details`, it's also called for chunks without choices, like the final usage-only chunk.
        """
        return None

    def _map_usage(self, response: ChatCompletionChunk) -> usage.RequestUsage:
        return map_usage(response, self._provider_name, self._provider_url, self.model_name)

    def _map_finish_reason(
        self, key: Literal['stop', 'length', 'tool_calls', 'content_filter', 'function_call']
    ) -> FinishReason | None:
        """Hooks that maps a finish reason key to a [FinishReason](pydantic_ai.messages.FinishReason).

        This method may be overridden by subclasses of `OpenAIChatModel` to accommodate custom keys.
        """
        return CHAT_FINISH_REASON_MAP.get(key)

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
