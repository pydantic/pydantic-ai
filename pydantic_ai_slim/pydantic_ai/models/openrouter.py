from __future__ import annotations

from collections.abc import AsyncIterable, AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, Literal, cast, overload

from typing_extensions import assert_never

from pydantic_ai.messages import (
    FinishReason,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
    SystemPromptPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models import (
    Model,
    ModelRequestParameters,
    StreamedResponse,
    check_allow_model_requests,
)
from pydantic_ai.models._openai_compat import (
    completions_create,
    map_usage,
    process_response,
    process_streamed_response,
)
from pydantic_ai.profiles import ModelProfileSpec
from pydantic_ai.providers import Provider
from pydantic_ai.providers.openrouter import OpenRouterProvider
from pydantic_ai.settings import ModelSettings

try:
    from openai import AsyncOpenAI, AsyncStream
    from openai.types import chat
    from openai.types.chat import ChatCompletionChunk
except ImportError as _import_error:
    raise ImportError(
        'Please install `openai` to use the OpenRouter model, you can use the `openai` optional group - `pip install "pydantic-ai-slim[openai]"'
    ) from _import_error

OpenRouterModelName = str


__all__ = ['OpenRouterModel']


_OPENROUTER_CHAT_FINISH_REASON_MAP: dict[str, FinishReason] = {
    'stop': 'stop',
    'length': 'length',
    'tool_calls': 'tool_call',
    'content_filter': 'content_filter',
    'function_call': 'tool_call',
    'error': 'error',
}


@dataclass(init=False)
class OpenRouterModel(Model):
    """Model integration for OpenRouter's OpenAI-compatible chat completions API."""

    client: AsyncOpenAI = field(repr=False)
    _model_name: OpenRouterModelName = field(repr=False)
    _system: str = field(default='openrouter', repr=False)

    def __init__(
        self,
        model_name: OpenRouterModelName,
        *,
        provider: Literal['openrouter'] | Provider[AsyncOpenAI] = 'openrouter',
        profile: ModelProfileSpec | None = None,
        settings: ModelSettings | None = None,
    ):
        """Initialize an OpenRouter model.

        Args:
            model_name: The name of the OpenRouter model to use (e.g., 'openai/gpt-4o', 'google/gemini-2.5-flash-lite').
            provider: The provider to use for authentication and API access. Can be either the string
                'openrouter' or an instance of `Provider[AsyncOpenAI]`. If not provided, a new provider will be
                created using environment variables.
            profile: The model profile to use. Defaults to a profile picked by the provider based on the model name.
            settings: Default model settings for this model instance.
        """
        self._model_name = model_name

        if isinstance(provider, str):
            provider = OpenRouterProvider()
        self._provider = provider
        self.client = provider.client

        super().__init__(settings=settings, profile=profile or provider.model_profile(model_name))

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def system(self) -> str:
        return self._system

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        check_allow_model_requests()
        response = await self._completions_create(messages, False, model_settings, model_request_parameters)
        model_response = self._process_response(response)
        return model_response

    @asynccontextmanager
    async def request_stream(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
        run_context: Any | None = None,
    ) -> AsyncIterator[StreamedResponse]:
        check_allow_model_requests()
        response = await self._completions_create(messages, True, model_settings, model_request_parameters)
        async with response:
            yield await self._process_streamed_response(response, model_request_parameters)

    @overload
    async def _completions_create(
        self,
        messages: list[ModelMessage],
        stream: Literal[True],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> AsyncStream[ChatCompletionChunk]: ...

    @overload
    async def _completions_create(
        self,
        messages: list[ModelMessage],
        stream: Literal[False],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> chat.ChatCompletion: ...

    async def _completions_create(
        self,
        messages: list[ModelMessage],
        stream: bool,
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> chat.ChatCompletion | AsyncStream[ChatCompletionChunk]:
        settings_to_use: ModelSettings = model_settings or {}
        reasoning_param = self._build_reasoning_param(settings_to_use)
        if reasoning_param:
            settings_dict = dict(settings_to_use)
            extra_body_raw = settings_dict.get('extra_body')
            extra_body: dict[str, Any] = (
                dict(cast(dict[str, Any], extra_body_raw)) if isinstance(extra_body_raw, dict) else {}
            )
            extra_body['reasoning'] = reasoning_param
            settings_dict['extra_body'] = extra_body
            settings_to_use = cast(ModelSettings, settings_dict)

        return await completions_create(
            self,
            messages,
            stream,
            settings_to_use,
            model_request_parameters,
        )

    def _build_reasoning_param(self, model_settings: ModelSettings) -> dict[str, Any] | None:
        reasoning_config: dict[str, Any] = {}

        if 'openrouter_reasoning_effort' in model_settings:
            reasoning_config['effort'] = model_settings['openrouter_reasoning_effort']
        elif 'openrouter_reasoning_max_tokens' in model_settings:
            reasoning_config['max_tokens'] = model_settings['openrouter_reasoning_max_tokens']
        elif 'openrouter_reasoning_enabled' in model_settings:
            reasoning_config['enabled'] = model_settings['openrouter_reasoning_enabled']

        if 'openrouter_reasoning_exclude' in model_settings:
            reasoning_config['exclude'] = model_settings['openrouter_reasoning_exclude']

        return reasoning_config if reasoning_config else None

    def _process_response(self, response: chat.ChatCompletion) -> ModelResponse:
        return process_response(
            self,
            response,
            map_usage_fn=map_usage,
            finish_reason_map=_OPENROUTER_CHAT_FINISH_REASON_MAP,
        )

    async def _process_streamed_response(
        self, response: AsyncStream[ChatCompletionChunk], model_request_parameters: ModelRequestParameters
    ) -> StreamedResponse:
        return await process_streamed_response(
            self,
            response,
            model_request_parameters,
            map_usage_fn=map_usage,
            finish_reason_map=_OPENROUTER_CHAT_FINISH_REASON_MAP,
        )

    @staticmethod
    def _map_user_prompt(part: UserPromptPart) -> chat.ChatCompletionUserMessageParam:
        if isinstance(part.content, str):
            return chat.ChatCompletionUserMessageParam(role='user', content=part.content)
        else:
            content_parts: list[str] = []
            for item in part.content:
                if isinstance(item, str):
                    content_parts.append(item)
            return chat.ChatCompletionUserMessageParam(role='user', content=' '.join(content_parts))

    @classmethod
    async def _map_user_message(cls, message: ModelRequest) -> AsyncIterable[chat.ChatCompletionMessageParam]:
        for part in message.parts:
            if isinstance(part, SystemPromptPart):
                yield chat.ChatCompletionSystemMessageParam(role='system', content=part.content)
            elif isinstance(part, UserPromptPart):
                yield cls._map_user_prompt(part)
            elif isinstance(part, ToolReturnPart):
                yield chat.ChatCompletionToolMessageParam(
                    role='tool',
                    tool_call_id=part.tool_call_id,
                    content=part.model_response_str(),
                )
            elif isinstance(part, RetryPromptPart):
                if part.tool_name is None:
                    yield chat.ChatCompletionUserMessageParam(role='user', content=part.model_response())
                else:
                    yield chat.ChatCompletionToolMessageParam(
                        role='tool',
                        tool_call_id=part.tool_call_id,
                        content=part.model_response(),
                    )
            else:
                assert_never(part)
