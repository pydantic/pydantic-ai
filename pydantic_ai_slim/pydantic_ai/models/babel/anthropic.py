"""`AnthropicModel` with request assembly, response parsing and streaming done by babel's `anthropic-messages` codec."""

from __future__ import annotations as _annotations

import dataclasses
from collections.abc import AsyncIterator, Sequence
from typing import Any, Literal, cast

from anthropic.types.beta import (
    BetaMessage,
    BetaMessageParam,
    BetaRawMessageDeltaEvent,
    BetaRawMessageStartEvent,
    BetaTextBlockParam,
)
from llm_transform.media import MEDIA_URL_OK
from llm_transform.registry import decode_response, encode, stream_step

from ...messages import InstructionPart, ModelMessage, ModelResponse, ModelResponseStreamEvent
from ...usage import RequestUsage
from .. import ModelRequestParameters
from ..anthropic import AnthropicModel, AnthropicModelSettings, AnthropicStreamedResponse
from ._adapters import IR, download_url_media, fold_stream_emits, ir_to_model_response, messages_to_ir

__all__ = ('BabelAnthropicModel', 'BabelAnthropicStreamedResponse')


class BabelAnthropicStreamedResponse(AnthropicStreamedResponse):
    """`AnthropicStreamedResponse` whose events are folded by babel's `anthropic-messages` `stream_step`."""

    async def _get_event_iterator(self) -> AsyncIterator[ModelResponseStreamEvent]:
        state: Any = {}
        async for event in self._response:
            if isinstance(event, BetaRawMessageStartEvent):
                if event.message is None:  # pyright: ignore[reportUnnecessaryComparison]
                    # On Bedrock the SDK drops SSE event types, so a Bedrock-only chunk is constructed
                    # as a `BetaRawMessageStartEvent` with no message; it carries nothing to fold.
                    continue
                self.provider_response_id = event.message.id
                self._usage = _map_usage(
                    event.message.model, event.message.usage.model_dump(), self._provider_name, self._provider_url
                )
            elif isinstance(event, BetaRawMessageDeltaEvent):
                # `message_delta` reports the cumulative output tokens for the message.
                self._usage = dataclasses.replace(self._usage, output_tokens=event.usage.output_tokens)
            result = stream_step('anthropic-messages', state, event.model_dump())
            state = result['state']
            for stream_event in fold_stream_emits(
                result['emit'], self._parts_manager, self, provider_name=self._provider_name
            ):
                yield stream_event


class BabelAnthropicModel(AnthropicModel):
    """[`AnthropicModel`][pydantic_ai.models.anthropic.AnthropicModel] mapped by babel's `anthropic-messages` codec.

    Construct it exactly like `AnthropicModel`. The provider, client, settings, native tools and
    beta headers behave as they do there; only the translation between the message history and the
    Messages wire is babel's. `anthropic_cache_instructions` and `CachePoint` breakpoints are placed
    the same way as in the native model, and the 4-breakpoint limit is enforced by it.

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
        return ir_to_model_response(
            decode_response('anthropic-messages', response.model_dump()),
            fmt='anthropic-messages',
            provider_name=self._provider.name,
            provider_url=self._provider.base_url,
            usage=_map_usage(response.model, response.usage.model_dump(), self._provider.name, self._provider.base_url),
            model_name=self.model_name,
        )


def _map_usage(model: str, usage: dict[str, Any], provider: str, provider_url: str) -> RequestUsage:
    return RequestUsage.extract(
        dict(model=model, usage=usage), provider=provider, provider_url=provider_url, provider_fallback='anthropic'
    )


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
