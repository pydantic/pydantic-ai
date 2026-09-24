"""`GoogleModel` with request assembly, response parsing and streaming done by babel's `gemini` codec."""

from __future__ import annotations as _annotations

from collections.abc import AsyncIterator
from typing import Any, cast

from google.genai import errors
from google.genai.types import (
    BlockedReason,
    ContentDict,
    ContentUnionDict,
    FinishReason as GoogleFinishReason,
    GenerateContentResponse,
    GenerateContentResponsePromptFeedback,
    SafetyRating,
)
from llm_transform.media import MEDIA_URL_OK
from llm_transform.registry import decode_response, encode, stream_step

from ...messages import FinishReason, ModelMessage, ModelResponse, ModelResponseStreamEvent
from .. import ModelRequestParameters
from ..google import (
    _FINISH_REASON_MAP,  # pyright: ignore[reportPrivateUsage]
    GeminiStreamedResponse,
    GoogleModel,
    _map_api_error,  # pyright: ignore[reportPrivateUsage]
    _metadata_as_usage,  # pyright: ignore[reportPrivateUsage]
)
from ._adapters import download_url_media, fold_stream_emits, gemini_rest_to_sdk, ir_to_model_response, messages_to_ir

__all__ = ('BabelGeminiStreamedResponse', 'BabelGoogleModel')


class BabelGeminiStreamedResponse(GeminiStreamedResponse):
    """`GeminiStreamedResponse` whose chunks are folded by babel's `gemini` `stream_step`.

    The finish reason, safety ratings, blocked-prompt feedback, service tier and traffic type are
    read from the raw chunks and recorded as the native stream records them.
    """

    async def _get_event_iterator(self) -> AsyncIterator[ModelResponseStreamEvent]:
        if self._provider_timestamp is not None:
            self.provider_details = {'timestamp': self._provider_timestamp}
        state: Any = {}
        try:
            async for chunk in self._response:
                # Each chunk reports the cumulative usage; merging keeps a count a later chunk omits.
                self._usage = _metadata_as_usage(chunk, self._provider_name, self._provider_url, self._usage)
                self._record_details(_response_details(chunk))
                if chunk.response_id:
                    self.provider_response_id = chunk.response_id
                if not chunk.candidates:
                    feedback = chunk.prompt_feedback
                    if feedback and feedback.block_reason:
                        # The prompt was blocked: there will be no content, and no later finish reason
                        # replaces the content filter.
                        self._has_content_filter = True
                        self._record_details(_prompt_feedback_details(feedback.block_reason, feedback))
                        self.finish_reason = 'content_filter'
                    continue
                candidate = chunk.candidates[0]
                if candidate.finish_reason and not self._has_content_filter:
                    self._record_details(_candidate_details(candidate.finish_reason, candidate.safety_ratings))
                    self.finish_reason = _FINISH_REASON_MAP.get(candidate.finish_reason.value)
                result = stream_step('gemini', state, chunk.model_dump(mode='json', by_alias=True, exclude_none=True))
                state = result['state']
                for event in fold_stream_emits(
                    result['emit'], self._parts_manager, self, provider_name=self._provider_name
                ):
                    yield event
        except errors.APIError as e:
            raise _map_api_error(e, self._model_name, self._model_id_namespace) from e

    def _record_details(self, details: dict[str, Any]) -> None:
        if details:
            self.provider_details = {**(self.provider_details or {}), **details}


class BabelGoogleModel(GoogleModel):
    """[`GoogleModel`][pydantic_ai.models.google.GoogleModel] mapped by babel's `gemini` codec.

    Construct it exactly like `GoogleModel`. The provider, client, settings and native tools behave
    as they do there; only the translation between the message history and the `generateContent`
    wire is babel's. Gemini takes no media by plain URL, so every `FileUrl` is downloaded and
    inlined as base64. The response details the native model records (the raw finish reason, safety
    ratings, a blocked prompt's feedback, logprobs, service tier and traffic type) are recorded the
    same way.
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
        candidate = response.candidates[0] if response.candidates else None
        details = _response_details(response)
        finish_reason: FinishReason | None = None
        feedback = response.prompt_feedback
        if candidate and candidate.finish_reason:
            details.update(_candidate_details(candidate.finish_reason, candidate.safety_ratings))
            finish_reason = _FINISH_REASON_MAP.get(candidate.finish_reason.value)
        elif candidate is None and feedback and feedback.block_reason:
            details.update(_prompt_feedback_details(feedback.block_reason, feedback))
            finish_reason = 'content_filter'
        if response.create_time is not None:
            details['timestamp'] = response.create_time
        if candidate and (logprobs_result := candidate.logprobs_result):
            details['logprobs'] = logprobs_result.model_dump(mode='json')
            details['avg_logprobs'] = candidate.avg_logprobs
        return ir_to_model_response(
            decode_response('gemini', response.model_dump(mode='json', by_alias=True, exclude_none=True)),
            fmt='gemini',
            provider_name=self._provider.name,
            provider_url=self._provider.base_url,
            usage=_metadata_as_usage(response, self._provider.name, self._provider.base_url),
            model_name=self.model_name,
            provider_details=details or None,
            finish_reason=finish_reason,
        )


def _response_details(response: GenerateContentResponse) -> dict[str, Any]:
    """The response-level details: the service tier header and the usage's traffic type."""
    details: dict[str, Any] = {}
    if (
        response.sdk_http_response
        and response.sdk_http_response.headers
        and (service_tier := response.sdk_http_response.headers.get('x-gemini-service-tier'))
    ):
        details['service_tier'] = service_tier.lower()
    if response.usage_metadata and response.usage_metadata.traffic_type:
        details['traffic_type'] = response.usage_metadata.traffic_type.value
    return details


def _candidate_details(finish_reason: GoogleFinishReason, safety_ratings: list[SafetyRating] | None) -> dict[str, Any]:
    """The raw finish reason and the safety ratings that came with it."""
    details: dict[str, Any] = {'finish_reason': finish_reason.value}
    if safety_ratings:
        details['safety_ratings'] = [rating.model_dump(by_alias=True) for rating in safety_ratings]
    return details


def _prompt_feedback_details(
    block_reason: BlockedReason, feedback: GenerateContentResponsePromptFeedback
) -> dict[str, Any]:
    """Why a prompt was blocked, recorded as the native model records it."""
    details: dict[str, Any] = {'block_reason': block_reason.value}
    if feedback.block_reason_message:
        details['block_reason_message'] = feedback.block_reason_message
    if feedback.safety_ratings:
        details['safety_ratings'] = [rating.model_dump(by_alias=True) for rating in feedback.safety_ratings]
    return details
