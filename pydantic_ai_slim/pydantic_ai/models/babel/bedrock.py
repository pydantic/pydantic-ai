"""`BedrockConverseModel` with request assembly, response parsing and streaming done by babel's `bedrock-converse` codec."""

from __future__ import annotations as _annotations

from collections.abc import AsyncIterator, Sequence
from typing import TYPE_CHECKING, Any, TypeVar, cast

import anyio.to_thread
from llm_transform.media import MEDIA_URL_OK
from llm_transform.registry import decode_response, encode, stream_step

from ...messages import ModelMessage, ModelResponse, ModelResponseStreamEvent
from ...providers.bedrock import remove_bedrock_geo_prefix
from ...usage import RequestUsage
from .. import ModelRequestParameters
from ..bedrock import BedrockConverseModel, BedrockModelSettings, BedrockStreamedResponse
from ._adapters import download_url_media, fold_stream_emits, ir_to_model_response, messages_to_ir

if TYPE_CHECKING:
    from botocore.eventstream import EventStream
    from mypy_boto3_bedrock_runtime.type_defs import (
        ConverseResponseTypeDef,
        MessageUnionTypeDef,
        SystemContentBlockTypeDef,
        TokenUsageTypeDef,
    )

__all__ = ('BabelBedrockConverseModel', 'BabelBedrockStreamedResponse')

T = TypeVar('T')

_END_OF_STREAM = object()


class BabelBedrockStreamedResponse(BedrockStreamedResponse):
    """`BedrockStreamedResponse` whose events are folded by babel's `bedrock-converse` `stream_step`."""

    async def _get_event_iterator(self) -> AsyncIterator[ModelResponseStreamEvent]:
        if self._provider_response_id is not None:
            self.provider_response_id = self._provider_response_id
        state: Any = {}
        async for chunk in _iterate_in_thread(self._event_stream):
            match chunk:
                case {'metadata': metadata} if 'usage' in metadata:
                    self._usage += _map_usage(
                        self._model_name, metadata['usage'], self._provider_name, self._provider_url
                    )
                case _:
                    pass
            result = stream_step('bedrock-converse', state, chunk)
            state = result['state']
            for event in fold_stream_emits(
                result['emit'], self._parts_manager, self, provider_name=self._provider_name
            ):
                yield event


class BabelBedrockConverseModel(BedrockConverseModel):
    """[`BedrockConverseModel`][pydantic_ai.models.bedrock.BedrockConverseModel] mapped by babel's `bedrock-converse` codec.

    Construct it exactly like `BedrockConverseModel`. The provider, boto3 client, settings and
    guardrail configuration behave as they do there; only the translation between the message
    history and the Converse wire is babel's. Converse takes no media by URL, so every `FileUrl`
    is downloaded and inlined as bytes.
    """

    @property
    def _streamed_response_cls(self) -> type[BedrockStreamedResponse]:
        return BabelBedrockStreamedResponse

    async def _map_messages(
        self,
        messages: Sequence[ModelMessage],
        model_request_parameters: ModelRequestParameters,
        model_settings: BedrockModelSettings | None,
    ) -> tuple[list[SystemContentBlockTypeDef], list[MessageUnionTypeDef]]:
        messages = await download_url_media(messages, MEDIA_URL_OK['bedrock-converse'])
        ir = messages_to_ir(
            messages,
            model_name=self.model_name,
            provider_name=self._provider.name,
            instruction_parts=self._get_instruction_parts(messages, model_request_parameters),
        )
        encoded = encode('bedrock-converse', ir)
        return (
            cast(list['SystemContentBlockTypeDef'], encoded.get('system') or []),
            cast(list['MessageUnionTypeDef'], encoded['messages']),
        )

    async def _process_response(self, response: ConverseResponseTypeDef) -> ModelResponse:
        return ir_to_model_response(
            decode_response('bedrock-converse', cast(dict[str, Any], response)),
            fmt='bedrock-converse',
            provider_name=self._provider.name,
            provider_url=self._provider.base_url,
            usage=_map_usage(self.model_name, response['usage'], self._provider.name, self._provider.base_url),
            model_name=self.model_name,
            provider_response_id=response.get('ResponseMetadata', {}).get('RequestId'),
        )


def _map_usage(model_name: str, usage: TokenUsageTypeDef, provider: str, provider_url: str) -> RequestUsage:
    return RequestUsage.extract(
        dict(model=remove_bedrock_geo_prefix(model_name), usage=usage),
        provider=provider,
        provider_url=provider_url,
        provider_fallback='bedrock',
    )


async def _iterate_in_thread(event_stream: EventStream[T]) -> AsyncIterator[T]:
    """Iterate boto3's blocking event stream from a worker thread."""
    iterator = iter(event_stream)
    while True:
        chunk = await anyio.to_thread.run_sync(next, iterator, _END_OF_STREAM)
        if chunk is _END_OF_STREAM:
            return
        yield cast(T, chunk)
