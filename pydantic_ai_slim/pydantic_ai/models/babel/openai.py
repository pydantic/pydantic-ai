"""`OpenAIChatModel` with request assembly, response parsing and streaming done by babel's `openai-chat` codec."""

from __future__ import annotations as _annotations

from collections.abc import AsyncIterator, Sequence
from typing import Any, cast

from llm_transform.media import MEDIA_URL_OK
from llm_transform.registry import decode_response, encode, stream_step
from openai.types import chat
from openai.types.chat import chat_completion_chunk
from pydantic import ValidationError

from ... import _utils
from ...exceptions import ModelAPIError, UnexpectedModelBehavior
from ...messages import ModelMessage, ModelResponse, ModelResponseStreamEvent
from ...profiles.openai import OpenAIModelProfile
from ...settings import ModelSettings
from .. import ModelRequestParameters
from ..openai import (
    OpenAIChatModel,
    OpenAIStreamedResponse,
    _map_api_errors,  # pyright: ignore[reportPrivateUsage]
)
from ._adapters import download_url_media, fold_stream_emits, ir_to_model_response, messages_to_ir

__all__ = ('BabelOpenAIChatModel', 'BabelOpenAIStreamedResponse')


class BabelOpenAIStreamedResponse(OpenAIStreamedResponse):
    """`OpenAIStreamedResponse` whose chunks are folded by babel's `openai-chat` `stream_step`.

    The finish reason, refusal, logprobs, moderation, service tier and timestamp are read from the
    raw chunks and recorded as the native stream records them; babel folds the content.
    """

    async def _get_event_iterator(self) -> AsyncIterator[ModelResponseStreamEvent]:
        state: Any = {}
        ignore_leading_whitespace = self._model_profile.get('ignore_streamed_leading_whitespace', False)
        with _map_api_errors(self._model_name, self._model_id_namespace):
            async for chunk in self._validate_response():
                self._record_chunk(chunk)
                # Empty on the final usage-only chunk, `None` from some OpenAI-compatible providers.
                if not chunk.choices or not self._record_choice(chunk, chunk.choices[0]):
                    continue
                result = stream_step('openai-chat', state, chunk.model_dump())
                state = result['state']
                for event in fold_stream_emits(
                    result['emit'],
                    self._parts_manager,
                    self,
                    provider_name=self._provider_name,
                    ignore_leading_whitespace=ignore_leading_whitespace,
                ):
                    yield event
            self._record_stream_end()

    def _record_chunk(self, chunk: chat.ChatCompletionChunk) -> None:
        """Record what every chunk may carry: timestamp, usage, id, model, moderation and service tier."""
        if self._provider_timestamp is None and chunk.created:
            self._provider_timestamp = _utils.number_to_datetime(chunk.created)
            self._record_details({'timestamp': self._provider_timestamp})
        chunk_usage = self._map_usage(chunk)
        if self._model_settings and self._model_settings.get('openai_continuous_usage_stats'):
            # Each chunk then carries the cumulative usage, so the latest replaces the total.
            self._usage = chunk_usage
        else:
            self._usage += chunk_usage
        if chunk.id:
            self.provider_response_id = chunk.id
        if chunk.model:
            self._model_name = chunk.model
        if chunk.moderation:
            self._record_details({'moderation': chunk.moderation.model_dump()})
        if chunk.service_tier:
            self._record_details({'service_tier': chunk.service_tier})

    def _record_choice(self, chunk: chat.ChatCompletionChunk, choice: chat_completion_chunk.Choice) -> bool:
        """Record the choice's finish reason, refusal and details; `False` when there is nothing to fold."""
        raw_finish_reason = choice.finish_reason
        self._has_finish_reason = self._has_finish_reason or bool(raw_finish_reason)
        # Azure's asynchronous content filter can send chunks with no delta.
        if choice.delta is None:  # pyright: ignore[reportUnnecessaryComparison]
            return False
        # A refusal (the structured output safety filter) comes instead of content.
        if choice.delta.refusal:
            self._has_refusal = True
            self.finish_reason = 'content_filter'
            self._refusal_text += choice.delta.refusal
            return False
        if raw_finish_reason and not self._has_refusal:
            self.finish_reason = self._map_finish_reason(raw_finish_reason)
        if provider_details := self._map_provider_details(chunk):
            if self._has_refusal:
                provider_details.pop('finish_reason', None)
            self._record_details(provider_details)
        return True

    def _record_stream_end(self) -> None:
        if self._refusal_text:
            self._record_details({'refusal': self._refusal_text})
        if self._logprobs:
            self._record_details({'logprobs': self._logprobs})
        if (
            self._model_profile.get('openai_chat_streaming_requires_finish_reason', False)
            and not self._has_finish_reason
            and not self.cancelled
        ):
            raise ModelAPIError(model_name=self.model_name, message='Streamed response ended without a `finish_reason`')

    def _record_details(self, details: dict[str, Any]) -> None:
        self.provider_details = {**(self.provider_details or {}), **details}


class BabelOpenAIChatModel(OpenAIChatModel):
    """[`OpenAIChatModel`][pydantic_ai.models.openai.OpenAIChatModel] mapped by babel's `openai-chat` codec.

    Construct it exactly like `OpenAIChatModel`. The provider, HTTP client, settings and model
    profile behave as they do there; only the translation between the message history and the
    Chat Completions wire is babel's. The profile's `openai_system_prompt_role` and
    `openai_chat_supports_multiple_system_messages` are still applied on top of the mapping, and
    the response is validated, its refusal, logprobs, moderation and service tier recorded, as the
    native model does.

    Audio and document URLs are downloaded and inlined as base64, as the Chat Completions API
    requires; image URLs are sent as URLs unless `force_download` is set.
    """

    @property
    def _streamed_response_cls(self) -> type[OpenAIStreamedResponse]:
        return BabelOpenAIStreamedResponse

    async def _map_messages(
        self,
        messages: Sequence[ModelMessage],
        model_request_parameters: ModelRequestParameters,
        *,
        model_settings: ModelSettings | None = None,
    ) -> list[chat.ChatCompletionMessageParam]:
        messages = await download_url_media(messages, MEDIA_URL_OK['openai-chat'])
        ir = messages_to_ir(
            messages,
            model_name=self.model_name,
            provider_name=self._provider.name,
            instruction_parts=self._get_instruction_parts(messages, model_request_parameters),
        )
        openai_messages = _apply_system_prompt_profile(self.profile, encode('openai-chat', ir)['messages'])
        return cast(list[chat.ChatCompletionMessageParam], openai_messages)

    def _process_response(self, response: chat.ChatCompletion | str) -> ModelResponse:
        # The SDK does not validate the completion it returns, and a plain-text body arrives as a string.
        if not isinstance(response, chat.ChatCompletion):
            raise UnexpectedModelBehavior(
                f'Invalid response from {self.system} chat completions endpoint, expected JSON data'
            )
        timestamp = _utils.now_utc()
        if not response.created:
            response.created = int(timestamp.timestamp())
        # Local Ollama sometimes returns a `None` finish reason.
        if response.choices and (choice := response.choices[0]) and choice.finish_reason is None:  # pyright: ignore[reportUnnecessaryComparison]
            choice.finish_reason = 'stop'
        try:
            completion = self._validate_completion(response)
        except ValidationError as e:
            raise UnexpectedModelBehavior(f'Invalid response from {self.system} chat completions endpoint: {e}') from e
        choice = completion.choices[0]
        details = self._process_provider_details(completion) or {}
        if completion.moderation:
            details['moderation'] = completion.moderation.model_dump()
        if completion.service_tier:
            details['service_tier'] = completion.service_tier
        details['timestamp'] = _utils.number_to_datetime(completion.created)
        usage = self._map_usage(completion)
        if choice.message.refusal:
            # The structured output safety filter refused: no content, and the refusal is the detail.
            details.pop('finish_reason', None)
            details['refusal'] = choice.message.refusal
            return ModelResponse(
                parts=[],
                usage=usage,
                model_name=completion.model,
                timestamp=timestamp,
                provider_details=details,
                provider_response_id=completion.id,
                provider_name=self._provider.name,
                provider_url=self._provider.base_url,
                finish_reason='content_filter',
            )
        return ir_to_model_response(
            decode_response('openai-chat', completion.model_dump()),
            fmt='openai-chat',
            provider_name=self._provider.name,
            provider_url=self._provider.base_url,
            usage=usage,
            model_name=self.model_name,
            provider_details=details,
            finish_reason=self._map_finish_reason(choice.finish_reason),
        )


def _apply_system_prompt_profile(profile: OpenAIModelProfile, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Apply the model profile's system-prompt facts to encoded messages.

    Some models take instructions under the `developer` or `user` role rather than `system`, and
    some OpenAI-compatible APIs accept a single system message, in which case the leading system
    messages are merged into one.
    """
    role = profile.get('openai_system_prompt_role') or 'system'
    messages = [{**message, 'role': role} if message.get('role') == 'system' else message for message in messages]
    if profile.get('openai_chat_supports_multiple_system_messages', True) or role != 'system':
        return messages
    leading: list[str] = []
    rest = 0
    while rest < len(messages) and messages[rest].get('role') == 'system':
        leading.append(messages[rest]['content'])
        rest += 1
    if len(leading) <= 1:
        return messages
    return [{'role': 'system', 'content': '\n\n'.join(leading)}, *messages[rest:]]
