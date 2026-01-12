"""LiteLLM model implementation for pydantic-ai.

Uses litellm.acompletion() directly to support all LiteLLM providers
without requiring a proxy server.

See: https://github.com/pydantic/pydantic-ai/issues/3935
"""
from __future__ import annotations as _annotations

from collections.abc import AsyncIterator
from typing import Any, Literal, cast

from typing_extensions import override

from ..exceptions import ModelHTTPError
from ..messages import ModelMessage
from ..profiles import ModelProfileSpec
from ..settings import ModelSettings
from . import ModelRequestParameters

try:
    from openai import AsyncOpenAI
    from openai.types import chat

    from .openai import OpenAIChatModel, OpenAIChatModelSettings
except ImportError as _import_error:
    raise ImportError(
        'Please install `openai` to use the LiteLLM model, '
        'you can use the `openai` optional group — `pip install "pydantic-ai-slim[openai]"`'
    ) from _import_error

try:
    import litellm
except ImportError as _import_error:
    raise ImportError(
        'Please install `litellm` to use the LiteLLM model, '
        'you can use `pip install litellm`'
    ) from _import_error

from ..providers import Provider
from ..providers.litellm import LiteLLMProvider


class LiteLLMModelSettings(ModelSettings, total=False):
    """Settings for LiteLLM models."""

    litellm_drop_params: bool
    """Drop unsupported params when switching providers."""


class LiteLLMModel(OpenAIChatModel):
    """LiteLLM model that calls litellm.acompletion() directly.

    This bypasses the OpenAI SDK client to support all LiteLLM providers
    without requiring a proxy server.
    """

    def __init__(
        self,
        model_name: str,
        *,
        provider: Literal['litellm'] | Provider[AsyncOpenAI] = 'litellm',
        profile: ModelProfileSpec | None = None,
        settings: ModelSettings | None = None,
    ):
        if provider == 'litellm':
            provider = LiteLLMProvider()
        super().__init__(model_name, provider=provider, profile=profile, settings=settings)

    @override
    async def _completions_create(
        self,
        messages: list[ModelMessage],
        stream: bool,
        model_settings: OpenAIChatModelSettings,
        model_request_parameters: ModelRequestParameters,
    ) -> chat.ChatCompletion | AsyncIterator[chat.ChatCompletionChunk]:
        """Use litellm.acompletion() directly."""
        from ..profiles.openai import OpenAIModelProfile

        tools = self._get_tools(model_request_parameters)
        tool_choice: Literal['none', 'required', 'auto'] | None = None
        if tools:
            if (
                not model_request_parameters.allow_text_output
                and OpenAIModelProfile.from_profile(self.profile).openai_supports_tool_choice_required
            ):
                tool_choice = 'required'
            else:
                tool_choice = 'auto'

        openai_messages = await self._map_messages(messages, model_request_parameters)

        response_format: dict[str, Any] | None = None
        if model_request_parameters.output_mode == 'native':
            output_object = model_request_parameters.output_object
            assert output_object is not None
            response_format = self._map_json_schema(output_object)
        elif model_request_parameters.output_mode == 'prompted' and self.profile.supports_json_object_output:
            response_format = {'type': 'json_object'}

        kwargs: dict[str, Any] = {
            'model': self.model_name,
            'messages': [dict(m) for m in openai_messages],
            'stream': stream,
        }
        if tools:
            kwargs['tools'] = tools
        if tool_choice:
            kwargs['tool_choice'] = tool_choice
        if response_format:
            kwargs['response_format'] = response_format
        if 'temperature' in model_settings:
            kwargs['temperature'] = model_settings['temperature']
        if 'max_tokens' in model_settings:
            kwargs['max_tokens'] = model_settings['max_tokens']
        if 'top_p' in model_settings:
            kwargs['top_p'] = model_settings['top_p']
        if cast(LiteLLMModelSettings, model_settings).get('litellm_drop_params'):
            kwargs['drop_params'] = True

        try:
            response = await litellm.acompletion(**kwargs)
            if stream:
                return self._wrap_stream(response)
            return chat.ChatCompletion.model_validate(response.model_dump())
        except litellm.exceptions.AuthenticationError as e:
            raise ModelHTTPError(status_code=401, model_name=self.model_name, body=str(e)) from e
        except litellm.exceptions.BadRequestError as e:
            raise ModelHTTPError(status_code=400, model_name=self.model_name, body=str(e)) from e
        except litellm.exceptions.RateLimitError as e:
            raise ModelHTTPError(status_code=429, model_name=self.model_name, body=str(e)) from e
        except litellm.exceptions.APIError as e:
            raise ModelHTTPError(status_code=getattr(e, 'status_code', 500), model_name=self.model_name, body=str(e)) from e

    async def _wrap_stream(self, response: Any) -> AsyncIterator[chat.ChatCompletionChunk]:
        """Wrap LiteLLM streaming response."""
        async for chunk in response:
            yield chat.ChatCompletionChunk.model_validate(chunk.model_dump())
