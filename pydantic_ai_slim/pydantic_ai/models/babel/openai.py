"""`OpenAIChatModel` with request assembly, response parsing and streaming done by babel's `openai-chat` codec."""

from __future__ import annotations as _annotations

from collections.abc import AsyncIterator, Sequence
from typing import Any, cast

from llm_transform.media import MEDIA_URL_OK
from llm_transform.registry import decode_response, encode, stream_step
from openai.types import chat

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
    """`OpenAIStreamedResponse` whose chunks are folded by babel's `openai-chat` `stream_step`."""

    async def _get_event_iterator(self) -> AsyncIterator[ModelResponseStreamEvent]:
        state: Any = {}
        with _map_api_errors(self._model_name, self._model_id_namespace):
            async for chunk in self._validate_response():
                if chunk.id:
                    self.provider_response_id = chunk.id
                if chunk.model:
                    self._model_name = chunk.model
                chunk_usage = self._map_usage(chunk)
                if self._model_settings and self._model_settings.get('openai_continuous_usage_stats'):
                    # Each chunk then carries the cumulative usage, so the latest replaces the total.
                    self._usage = chunk_usage
                else:
                    self._usage += chunk_usage
                result = stream_step('openai-chat', state, chunk.model_dump())
                state = result['state']
                for event in fold_stream_emits(
                    result['emit'], self._parts_manager, self, provider_name=self._provider_name
                ):
                    yield event


class BabelOpenAIChatModel(OpenAIChatModel):
    """[`OpenAIChatModel`][pydantic_ai.models.openai.OpenAIChatModel] mapped by babel's `openai-chat` codec.

    Construct it exactly like `OpenAIChatModel`. The provider, HTTP client, settings and model
    profile behave as they do there; only the translation between the message history and the
    Chat Completions wire is babel's. The profile's `openai_system_prompt_role` and
    `openai_chat_supports_multiple_system_messages` are still applied on top of the mapping.

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
        completion = chat.ChatCompletion.model_validate_json(response) if isinstance(response, str) else response
        return ir_to_model_response(
            decode_response('openai-chat', completion.model_dump()),
            fmt='openai-chat',
            provider_name=self._provider.name,
            provider_url=self._provider.base_url,
            usage=self._map_usage(completion),
            model_name=self.model_name,
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
