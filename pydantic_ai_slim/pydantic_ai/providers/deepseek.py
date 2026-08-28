from __future__ import annotations as _annotations

import os
from typing import Literal, overload

from pydantic_ai import ModelProfile
from pydantic_ai.exceptions import UserError
from pydantic_ai.profiles import merge_profile
from pydantic_ai.profiles.deepseek import deepseek_model_profile
from pydantic_ai.profiles.openai import OpenAIJsonSchemaTransformer, OpenAIModelProfile

try:
    from openai import AsyncOpenAI
except ImportError as _import_error:
    raise ImportError(
        'Please install the `openai` package to use the DeepSeek provider, '
        'you can use the `openai` optional group — `pip install "pydantic-ai-slim[openai]"`'
    ) from _import_error
else:
    from ._openai_compatible import (
        AsyncHTTPClient as _OpenAIHTTPClient,
        OpenAICompatibleProvider as _OpenAICompatibleProvider,
    )


DeepSeekModelName = Literal['deepseek-chat', 'deepseek-reasoner', 'deepseek-v4-flash', 'deepseek-v4-pro']


class DeepSeekProvider(_OpenAICompatibleProvider):
    """Provider for DeepSeek API."""

    @property
    def name(self) -> str:
        return 'deepseek'

    @property
    def base_url(self) -> str:
        return 'https://api.deepseek.com'

    @property
    def client(self) -> AsyncOpenAI:
        return self._client

    @staticmethod
    def model_profile(model_name: str) -> ModelProfile | None:
        profile = deepseek_model_profile(model_name)

        # As DeepSeekProvider is most often used with OpenAIChatModel, which used to unconditionally use OpenAIJsonSchemaTransformer,
        # we need to maintain that behavior unless json_schema_transformer is set explicitly.
        # This was not the case when using a DeepSeek model with another model class (e.g. BedrockConverseModel or GroqModel),
        # so we won't do this in `deepseek_model_profile` unless we learn it's always needed.
        return merge_profile(
            OpenAIModelProfile(json_schema_transformer=OpenAIJsonSchemaTransformer),
            profile,
            OpenAIModelProfile(
                supports_json_object_output=True,
                openai_chat_thinking_field='reasoning_content',
                # Starting from DeepSeek v3.2, DeepSeek requires sending thinking parts for optimal agentic performance.
                openai_chat_send_back_thinking_parts='field',
                # DeepSeek's Responses endpoint documents merging each function call into the
                # assistant message adjacent to it, unlike the official Responses API, so an
                # assistant item between two calls strands the first one without its output.
                openai_responses_supports_interleaved_function_calls=False,
                # Reasoning-capable models do not support tool_choice=required; use startswith so
                # future deepseek-v4-* SKUs are covered automatically without listing each one.
                openai_supports_tool_choice_required=(
                    model_name != 'deepseek-reasoner' and not model_name.startswith('deepseek-v4-')
                ),
                # `openai_supports_phase` is deliberately left at its `False` default even though
                # DeepSeek labels its Responses API output with `phase`: the field is absent from
                # DeepSeek's documented input items, and its API silently ignores what it doesn't
                # support, so a request carrying `phase` returns 200 without that proving the model
                # reads it. Sending it back would add an undocumented field for no observable gain.
                # See https://api-docs.deepseek.com/guides/responses_api.
            ),
        )

    @overload
    def __init__(self, *, openai_client: AsyncOpenAI) -> None: ...

    @overload
    def __init__(
        self,
        *,
        api_key: str | None = None,
        openai_client: None = None,
        http_client: _OpenAIHTTPClient | None = None,
    ) -> None: ...

    def __init__(
        self,
        *,
        api_key: str | None = None,
        openai_client: AsyncOpenAI | None = None,
        http_client: _OpenAIHTTPClient | None = None,
    ) -> None:
        api_key = api_key or os.getenv('DEEPSEEK_API_KEY')
        if not api_key and openai_client is None:
            raise UserError(
                'Set the `DEEPSEEK_API_KEY` environment variable or pass it via `DeepSeekProvider(api_key=...)`'
                ' to use the DeepSeek provider.'
            )

        if openai_client is not None:
            self._client = openai_client
        else:
            self._client = self._create_openai_client(base_url=self.base_url, api_key=api_key, http_client=http_client)
