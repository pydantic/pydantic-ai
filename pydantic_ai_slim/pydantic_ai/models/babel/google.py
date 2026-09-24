"""`GoogleModel` with request assembly, response parsing and streaming done by babel's `gemini` codec."""

from __future__ import annotations as _annotations

from collections.abc import AsyncIterator
from typing import Any, cast

from google.genai.types import ContentDict, ContentUnionDict, GenerateContentResponse
from llm_transform.media import MEDIA_URL_OK
from llm_transform.registry import decode_response, encode, stream_step

from ...messages import ModelMessage, ModelResponse, ModelResponseStreamEvent
from ...usage import RequestUsage
from .. import ModelRequestParameters
from ..google import GeminiStreamedResponse, GoogleModel
from ._adapters import download_url_media, fold_stream_emits, gemini_rest_to_sdk, ir_to_model_response, messages_to_ir

__all__ = ('BabelGeminiStreamedResponse', 'BabelGoogleModel')


class BabelGeminiStreamedResponse(GeminiStreamedResponse):
    """`GeminiStreamedResponse` whose chunks are folded by babel's `gemini` `stream_step`."""

    async def _get_event_iterator(self) -> AsyncIterator[ModelResponseStreamEvent]:
        if self._provider_timestamp is not None:
            self.provider_details = {'timestamp': self._provider_timestamp}
        state: Any = {}
        async for chunk in self._response:
            if chunk.usage_metadata is not None:
                # Each chunk reports the cumulative usage for the response.
                self._usage = _map_usage(chunk, self._provider_name, self._provider_url)
            if chunk.response_id:
                self.provider_response_id = chunk.response_id
            result = stream_step('gemini', state, chunk.model_dump(mode='json', by_alias=True, exclude_none=True))
            state = result['state']
            for event in fold_stream_emits(
                result['emit'], self._parts_manager, self, provider_name=self._provider_name
            ):
                yield event


class BabelGoogleModel(GoogleModel):
    """[`GoogleModel`][pydantic_ai.models.google.GoogleModel] mapped by babel's `gemini` codec.

    Construct it exactly like `GoogleModel`. The provider, client, settings and native tools behave
    as they do there; only the translation between the message history and the `generateContent`
    wire is babel's. Gemini takes no media by plain URL, so every `FileUrl` is downloaded and
    inlined as base64.
    """

    @property
    def _streamed_response_cls(self) -> type[GeminiStreamedResponse]:
        return BabelGeminiStreamedResponse

    async def _map_messages(
        self,
        messages: list[ModelMessage],
        model_request_parameters: ModelRequestParameters,
    ) -> tuple[ContentDict | None, list[ContentUnionDict]]:
        messages = await download_url_media(messages, MEDIA_URL_OK['gemini'])
        ir = messages_to_ir(
            messages,
            model_name=self.model_name,
            provider_name=self._provider.name,
            instruction_parts=self._get_instruction_parts(messages, model_request_parameters),
        )
        encoded = encode('gemini', ir)
        system_instruction = encoded.get('systemInstruction')
        return (
            cast(ContentDict, gemini_rest_to_sdk(system_instruction)) if system_instruction is not None else None,
            cast(list[ContentUnionDict], gemini_rest_to_sdk(encoded['contents'])),
        )

    def _process_response(self, response: GenerateContentResponse) -> ModelResponse:
        return ir_to_model_response(
            decode_response('gemini', response.model_dump(mode='json', by_alias=True, exclude_none=True)),
            fmt='gemini',
            provider_name=self._provider.name,
            provider_url=self._provider.base_url,
            usage=_map_usage(response, self._provider.name, self._provider.base_url),
            model_name=self.model_name,
        )


def _map_usage(response: GenerateContentResponse, provider: str, provider_url: str) -> RequestUsage:
    return RequestUsage.extract(
        response.model_dump(include={'model_version', 'usage_metadata'}, by_alias=True),
        provider=provider,
        provider_url=provider_url,
        provider_fallback='google',
    )
