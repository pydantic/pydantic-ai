"""`BedrockConverseModel` with request assembly, response parsing and streaming done by babel's `bedrock-converse` codec."""

from __future__ import annotations as _annotations

from collections.abc import AsyncIterator, Sequence
from typing import TYPE_CHECKING, Any, cast

from llm_transform.media import MEDIA_URL_OK
from llm_transform.registry import decode_response, encode, stream_step

from ...messages import ModelMessage, ModelResponse, ModelResponseStreamEvent
from .. import ModelRequestParameters
from ..bedrock import (
    _FINISH_REASON_MAP,  # pyright: ignore[reportPrivateUsage]
    BedrockConverseModel,
    BedrockModelSettings,
    BedrockStreamedResponse,
    _AsyncIteratorWrapper,  # pyright: ignore[reportPrivateUsage]
    _map_api_errors,  # pyright: ignore[reportPrivateUsage]
    _map_usage,  # pyright: ignore[reportPrivateUsage]
)
from ._adapters import download_url_media, fold_stream_emits, ir_to_model_response, messages_to_ir

if TYPE_CHECKING:
    from mypy_boto3_bedrock_runtime.type_defs import (
        ConverseResponseTypeDef,
        MessageUnionTypeDef,
        SystemContentBlockTypeDef,
    )

__all__ = ('BabelBedrockConverseModel', 'BabelBedrockStreamedResponse')

# Converse takes S3 objects by reference (`s3Location`), which babel encodes from an `s3://` URL, so
# such a URL is never downloaded.
_PASSTHROUGH_SCHEMES = frozenset({'s3'})


class BabelBedrockStreamedResponse(BedrockStreamedResponse):
    """`BedrockStreamedResponse` whose events are folded by babel's `bedrock-converse` `stream_step`."""

    async def _get_event_iterator(self) -> AsyncIterator[ModelResponseStreamEvent]:
        with _map_api_errors(self._model_name, self._model_id_namespace):
            if self._provider_response_id is not None:
                self.provider_response_id = self._provider_response_id
            ignore_leading_whitespace = self._model_profile.get('ignore_streamed_leading_whitespace', False)
            state: Any = {}
            async for chunk in _AsyncIteratorWrapper(self._event_stream):
                match chunk:
                    case {'messageStop': message_stop}:
                        raw_finish_reason = message_stop['stopReason']
                        self.provider_details = {**(self.provider_details or {}), 'finish_reason': raw_finish_reason}
                        self.finish_reason = _FINISH_REASON_MAP.get(raw_finish_reason)
                    case {'metadata': metadata}:
                        if 'usage' in metadata:
                            self._usage += _map_usage(
                                metadata['usage'], self._provider_name, self._provider_url, self._model_name
                            )
                        if 'trace' in metadata:
                            self.provider_details = {**(self.provider_details or {}), 'trace': metadata['trace']}
                    case _:
                        pass
                result = stream_step('bedrock-converse', state, chunk)
                state = result['state']
                for event in fold_stream_emits(
                    result['emit'],
                    self._parts_manager,
                    self,
                    provider_name=self._provider_name,
                    ignore_leading_whitespace=ignore_leading_whitespace,
                ):
                    yield event


class BabelBedrockConverseModel(BedrockConverseModel):
    """[`BedrockConverseModel`][pydantic_ai.models.bedrock.BedrockConverseModel] mapped by babel's `bedrock-converse` codec.

    Construct it exactly like `BedrockConverseModel`. The provider, boto3 client, settings and
    guardrail configuration behave as they do there; only the translation between the message
    history and the Converse wire is babel's. Converse takes no media by URL except S3 objects, so
    every other `FileUrl` is downloaded and inlined as bytes. The raw stop reason and a guardrail
    trace are recorded in `provider_details` as the native model records them.
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
        messages = await download_url_media(
            messages, MEDIA_URL_OK['bedrock-converse'], passthrough_schemes=_PASSTHROUGH_SCHEMES
        )
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
        raw_finish_reason = response['stopReason']
        details: dict[str, Any] = {'finish_reason': raw_finish_reason}
        if 'trace' in response:
            details['trace'] = response['trace']
        return ir_to_model_response(
            decode_response('bedrock-converse', cast(dict[str, Any], response)),
            fmt='bedrock-converse',
            provider_name=self._provider.name,
            provider_url=self._provider.base_url,
            usage=_map_usage(response['usage'], self._provider.name, self._provider.base_url, self.model_name),
            model_name=self.model_name,
            provider_response_id=response.get('ResponseMetadata', {}).get('RequestId'),
            provider_details=details,
            finish_reason=_FINISH_REASON_MAP.get(raw_finish_reason),
        )
