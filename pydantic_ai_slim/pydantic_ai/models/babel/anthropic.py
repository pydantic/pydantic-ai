"""`AnthropicModel` with request assembly, response parsing and streaming done by babel's `anthropic-messages` codec."""

from __future__ import annotations as _annotations

from collections.abc import AsyncIterator, Sequence
from typing import Any, Literal, cast

from anthropic.types.beta import (
    BetaContainer,
    BetaInputTransformation,
    BetaMessage,
    BetaMessageParam,
    BetaRawMessageDeltaEvent,
    BetaRawMessageStartEvent,
    BetaRefusalStopDetails,
    BetaStopReason,
    BetaTextBlockParam,
)
from llm_transform.media import MEDIA_URL_OK
from llm_transform.registry import decode_response, encode, stream_step

from ...messages import FinishReason, InstructionPart, ModelMessage, ModelResponse, ModelResponseStreamEvent
from .. import ModelRequestParameters
from ..anthropic import (
    _FINISH_REASON_MAP,  # pyright: ignore[reportPrivateUsage]
    AnthropicModel,
    AnthropicModelSettings,
    AnthropicStreamedResponse,
    _map_api_errors,  # pyright: ignore[reportPrivateUsage]
    _map_usage,  # pyright: ignore[reportPrivateUsage]
    _report_input_transformations,  # pyright: ignore[reportPrivateUsage]
)
from ._adapters import IR, download_url_media, fold_stream_emits, ir_to_model_response, messages_to_ir

__all__ = ('BabelAnthropicModel', 'BabelAnthropicStreamedResponse')


class BabelAnthropicStreamedResponse(AnthropicStreamedResponse):
    """`AnthropicStreamedResponse` whose events are folded by babel's `anthropic-messages` `stream_step`."""

    async def _get_event_iterator(self) -> AsyncIterator[ModelResponseStreamEvent]:
        state: Any = {}
        with _map_api_errors(self._model_name, self._model_id_namespace):
            async for event in self._response:
                if isinstance(event, BetaRawMessageStartEvent):
                    if event.message is None:  # pyright: ignore[reportUnnecessaryComparison]
                        # On Bedrock the SDK drops SSE event types, so a Bedrock-only chunk is constructed
                        # as a `BetaRawMessageStartEvent` with no message; it carries nothing to fold.
                        continue
                    self.provider_response_id = event.message.id
                    self._usage = _map_usage(event, self._provider_name, self._provider_url, self._model_name)
                    self._record_details(
                        container=event.message.container,
                        input_transformations=event.message.input_transformations,
                    )
                elif isinstance(event, BetaRawMessageDeltaEvent):
                    # `message_delta` reports the cumulative usage and the stop reason for the message.
                    self._usage = _map_usage(
                        event, self._provider_name, self._provider_url, self._model_name, self._usage
                    )
                    self._record_details(
                        stop_reason=event.delta.stop_reason,
                        stop_details=event.delta.stop_details,
                        container=event.delta.container,
                        input_transformations=event.input_transformations,
                    )
                    if event.delta.stop_reason:
                        self.finish_reason = _finish_reason(event.delta.stop_reason)
                        self.state = _response_state(event.delta.stop_reason)
                result = stream_step('anthropic-messages', state, event.model_dump())
                state = result['state']
                for stream_event in fold_stream_emits(
                    result['emit'], self._parts_manager, self, provider_name=self._provider_name
                ):
                    yield stream_event

    def _record_details(
        self,
        *,
        stop_reason: str | None = None,
        stop_details: BetaRefusalStopDetails | None = None,
        container: BetaContainer | None = None,
        input_transformations: list[BetaInputTransformation] | None = None,
    ) -> None:
        details = _provider_details(
            stop_reason=stop_reason,
            stop_details=stop_details,
            container=container,
            input_transformations=input_transformations,
        )
        if details:
            self.provider_details = {**(self.provider_details or {}), **details}


class BabelAnthropicModel(AnthropicModel):
    """[`AnthropicModel`][pydantic_ai.models.anthropic.AnthropicModel] mapped by babel's `anthropic-messages` codec.

    Construct it exactly like `AnthropicModel`. The provider, client, settings, native tools and
    beta headers behave as they do there; only the translation between the message history and the
    Messages wire is babel's. `anthropic_cache_instructions` and `CachePoint` breakpoints are placed
    the same way as in the native model, and the 4-breakpoint limit is enforced by it. The response
    details the native model's next request depends on (the container id, a paused turn, a dropped
    thinking block) are recorded the same way.

    Audio and video are not supported by the Messages API; document and image URLs are sent as
    URLs unless `force_download` is set.
    """

    @property
    def _streamed_response_cls(self) -> type[AnthropicStreamedResponse]:
        return BabelAnthropicStreamedResponse

    async def _map_message(
        self,
        messages: list[ModelMessage],
        model_request_parameters: ModelRequestParameters,
        model_settings: AnthropicModelSettings,
    ) -> tuple[str | list[BetaTextBlockParam], list[BetaMessageParam]]:
        messages = self._trim_before_compaction(messages)
        messages = await download_url_media(messages, MEDIA_URL_OK['anthropic-messages'])
        instruction_parts = self._get_instruction_parts(messages, model_request_parameters) or []
        ir = messages_to_ir(
            messages,
            model_name=self.model_name,
            provider_name=self._provider.name,
            instruction_parts=instruction_parts,
        )
        encoded = encode('anthropic-messages', ir)
        system = _pack_system(
            encoded.get('system') or [], instruction_parts, model_settings.get('anthropic_cache_instructions')
        )
        return cast('str | list[BetaTextBlockParam]', system), cast(list[BetaMessageParam], encoded['messages'])

    def _process_response(
        self,
        response: BetaMessage,
        model_request_parameters: ModelRequestParameters,
        model_settings: AnthropicModelSettings,
    ) -> ModelResponse:
        details = _provider_details(
            stop_reason=response.stop_reason,
            stop_details=response.stop_details,
            container=response.container,
            input_transformations=response.input_transformations,
        )
        return ir_to_model_response(
            decode_response('anthropic-messages', response.model_dump()),
            fmt='anthropic-messages',
            provider_name=self._provider.name,
            provider_url=self._provider.base_url,
            usage=_map_usage(response, self._provider.name, self._provider.base_url, self._model_name),
            model_name=self.model_name,
            provider_details=details or None,
            finish_reason=_finish_reason(response.stop_reason),
            state=_response_state(response.stop_reason),
        )


def _finish_reason(stop_reason: BetaStopReason | None) -> FinishReason | None:
    """The finish reason the native model gives a raw stop reason."""
    return _FINISH_REASON_MAP.get(stop_reason) if stop_reason else None


def _response_state(stop_reason: str | None) -> Literal['complete', 'suspended']:
    """A `pause_turn` stop suspends the response so the agent reissues the paused server-side turn."""
    return 'suspended' if stop_reason == 'pause_turn' else 'complete'


def _provider_details(
    *,
    stop_reason: str | None,
    stop_details: BetaRefusalStopDetails | None,
    container: BetaContainer | None,
    input_transformations: list[BetaInputTransformation] | None,
) -> dict[str, Any]:
    """The `provider_details` the native model records from a message or a `message_delta`.

    The raw stop reason and any refusal explanation are kept for inspection. The container id is
    what the next request reuses for code execution, and the input transformations say when the
    API dropped a replayed thinking block, which the next request then leaves out.
    """
    details: dict[str, Any] = {}
    if stop_reason:
        details['finish_reason'] = stop_reason
    if stop_details is not None:
        if stop_details.explanation is not None:
            details['refusal'] = stop_details.explanation
        if stop_details.category is not None:
            details['refusal_category'] = stop_details.category
    if container:
        details['container_id'] = container.id
    if input_transformations:
        details['input_transformations'] = _report_input_transformations(input_transformations)
    return details


def _pack_system(
    system_blocks: list[IR],
    instruction_parts: Sequence[InstructionPart],
    cache_instructions: bool | Literal['5m', '1h'] | None,
) -> str | list[dict[str, Any]]:
    """Package babel's system segments the way `AnthropicModel` does.

    The static system prompts are joined into one block and each instruction part follows as its own
    block, so `anthropic_cache_instructions` can place its breakpoint after the last static
    instruction (dynamic instructions stay out of the cached prefix). Without instructions or caching
    the prompt is a plain string, as the native model sends it.
    """
    prompt_count = len(system_blocks) - len(instruction_parts)
    joined = '\n\n'.join(block.get('text', '') for block in system_blocks[:prompt_count])
    if not instruction_parts and not cache_instructions:
        return joined
    blocks: list[dict[str, Any]] = []
    if joined:
        blocks.append({'type': 'text', 'text': joined})
    blocks.extend({'type': 'text', 'text': block.get('text', '')} for block in system_blocks[prompt_count:])
    if blocks and cache_instructions:
        index = _cache_instructions_index(bool(joined), len(blocks), instruction_parts)
        if index is not None:
            ttl = '5m' if cache_instructions is True else cache_instructions
            blocks[index] = {**blocks[index], 'cache_control': {'type': 'ephemeral', 'ttl': ttl}}
    return blocks


def _cache_instructions_index(
    has_prompt: bool, block_count: int, instruction_parts: Sequence[InstructionPart]
) -> int | None:
    """Which block in `[prompt?, *instructions]` takes the `anthropic_cache_instructions` breakpoint."""
    if not instruction_parts:
        return 0 if has_prompt else None
    static_count = sum(1 for part in instruction_parts if not part.dynamic)
    if static_count == len(instruction_parts):
        return block_count - 1
    if static_count > 0:
        return (1 if has_prompt else 0) + static_count - 1
    return 0 if has_prompt else None
