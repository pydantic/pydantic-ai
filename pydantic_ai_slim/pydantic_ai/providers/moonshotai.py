from __future__ import annotations as _annotations

import os
from typing import Any, Literal, overload

from pydantic_ai import ModelProfile
from pydantic_ai.exceptions import UserError
from pydantic_ai.profiles import merge_profile
from pydantic_ai.profiles.moonshotai import moonshotai_model_profile
from pydantic_ai.profiles.openai import (
    OpenAIJsonSchemaTransformer,
    OpenAIModelProfile,
)

try:
    from openai import AsyncOpenAI
except ImportError as _import_error:
    raise ImportError(
        'Please install the `openai` package to use the MoonshotAI provider, '
        'you can use the `openai` optional group — `pip install "pydantic-ai-slim[openai]"`'
    ) from _import_error
else:
    from ._openai_compatible import (
        AsyncHTTPClient as _OpenAIHTTPClient,
        _OpenAISDKProvider,  # pyright: ignore[reportPrivateUsage]
    )

MoonshotAIModelName = Literal[
    'moonshot-v1-8k',
    'moonshot-v1-32k',
    'moonshot-v1-128k',
    'moonshot-v1-8k-vision-preview',
    'moonshot-v1-32k-vision-preview',
    'moonshot-v1-128k-vision-preview',
    'moonshot-v1-auto',
    'kimi-latest',
    'kimi-thinking-preview',
    'kimi-k2-0711-preview',
    'kimi-k2.5',
    'kimi-k2.6',
    'kimi-k2.7-code',
    'kimi-k2.7-code-highspeed',
    'kimi-k3',
    'kimi-k3.1-pro',
    'kimi-k3.1-pro-highspeed',
]


_THINKING_FALSE_BODY_BY_PREFIX: dict[str, dict[str, Any]] = {
    # Kimi K3 always reasons and takes `reasoning_effort` `low`, `high` or `max`, so `thinking=False` sends `low`:
    # https://platform.kimi.ai/docs/guide/use-reasoning-effort
    'kimi-k3': {'reasoning_effort': 'low'},
    # Kimi K2.5 and K2.6 turn thinking off with `thinking={'type': 'disabled'}`, and reject `reasoning_effort='none'`
    # (https://github.com/pydantic/pydantic-ai/pull/6208): https://platform.kimi.ai/docs/guide/use-thinking-models
    'kimi-k2.5': {'thinking': {'type': 'disabled'}},
    'kimi-k2.6': {'thinking': {'type': 'disabled'}},
}
"""What unified `thinking=False` sends on `api.moonshot.ai`, per model-name prefix. Kimi K2.7 Code always reasons
and has no effort knob, so it's left out and `thinking=False` sends nothing for it."""


def _thinking_false_body(model_name: str) -> dict[str, Any] | None:
    lower_name = model_name.lower()
    return next(
        (body for prefix, body in _THINKING_FALSE_BODY_BY_PREFIX.items() if lower_name.startswith(prefix)), None
    )


class MoonshotAIProvider(_OpenAISDKProvider):
    """Provider for MoonshotAI platform (Kimi models)."""

    @property
    def name(self) -> str:
        return 'moonshotai'

    @property
    def base_url(self) -> str:
        return self._base_url

    @property
    def client(self) -> AsyncOpenAI:
        return self._client

    @staticmethod
    def model_profile(model_name: str) -> ModelProfile | None:
        profile = moonshotai_model_profile(model_name)

        # `api.moonshot.ai` rejects `reasoning_effort='none'`. Kimi K2.5 and K2.6 turn thinking off with a
        # `thinking={'type': 'disabled'}` body object instead; Kimi K3 and K2.7 Code can't turn it off, so they're
        # marked always-enabled, and `thinking=False` sends K3's lowest effort and nothing on K2.7 Code. This is set
        # here, not in `moonshotai_model_profile`, because that profile is also routed through OpenRouter/Heroku,
        # whose gateways don't share this endpoint's request shape.
        is_reasoning = bool(profile and profile.get('supports_thinking'))
        thinking_false_body = _thinking_false_body(model_name) if is_reasoning else None
        can_disable_thinking = thinking_false_body is not None and 'thinking' in thinking_false_body
        reasoning_profile = OpenAIModelProfile(
            thinking_always_enabled=is_reasoning and not can_disable_thinking,
            # Probed live on `api.moonshot.ai`; the other Kimi reasoning ids weren't reachable to verify.
            openai_chat_supports_max_reasoning_effort=model_name.lower().startswith(
                ('kimi-k3', 'kimi-k2.6', 'kimi-k2.7')
            ),
        )
        if thinking_false_body is not None:
            reasoning_profile['openai_chat_thinking_false_body'] = thinking_false_body
        if can_disable_thinking:
            # K2.5 and K2.6 think unless told not to.
            reasoning_profile['thinking_enabled_by_default'] = True

        # As the MoonshotAI API is OpenAI-compatible, let's assume we also need OpenAIJsonSchemaTransformer,
        # unless json_schema_transformer is set explicitly.
        # Also, MoonshotAI does not support strict tool definitions
        # https://platform.moonshot.ai/docs/guide/migrating-from-openai-to-kimi#about-tool_choice
        # "Please note that the current version of Kimi API does not support the tool_choice=required parameter."
        return merge_profile(
            OpenAIModelProfile(json_schema_transformer=OpenAIJsonSchemaTransformer),
            profile,
            OpenAIModelProfile(
                supports_forced_tool_choice=False,
                supports_json_object_output=True,
                openai_chat_thinking_field='reasoning_content',
                openai_chat_send_back_thinking_parts='field',
            ),
            reasoning_profile,
        )

    @overload
    def __init__(self, *, base_url: str | None = None) -> None: ...

    @overload
    def __init__(self, *, api_key: str, base_url: str | None = None) -> None: ...

    @overload
    def __init__(self, *, api_key: str, http_client: _OpenAIHTTPClient, base_url: str | None = None) -> None: ...

    @overload
    def __init__(self, *, openai_client: AsyncOpenAI | None = None) -> None: ...

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        openai_client: AsyncOpenAI | None = None,
        http_client: _OpenAIHTTPClient | None = None,
    ) -> None:
        """Create a new MoonshotAI provider.

        Args:
            api_key: The API key for authentication. Defaults to the `MOONSHOTAI_API_KEY`
                environment variable.
            base_url: The base URL for requests. Defaults to the `MOONSHOTAI_BASE_URL`
                environment variable, then `https://api.moonshot.ai/v1`.
                Use `https://api.moonshot.cn/v1` for the Chinese endpoint.
            openai_client: An existing `AsyncOpenAI` client to use. Its base URL is used
                instead of `MOONSHOTAI_BASE_URL`. Cannot be combined with `base_url`.
            http_client: An existing `httpx2.AsyncClient` or legacy `httpx.AsyncClient` to use for making HTTP requests.
        """
        api_key = api_key or os.getenv('MOONSHOTAI_API_KEY')
        if not api_key and openai_client is None:
            raise UserError(
                'Set the `MOONSHOTAI_API_KEY` environment variable or pass it via '
                '`MoonshotAIProvider(api_key=...)` to use the MoonshotAI provider.'
            )

        if openai_client is not None and base_url is not None:
            raise UserError('Cannot provide both `openai_client` and `base_url`')

        if openai_client is not None:
            self._client = openai_client
            self._base_url = str(openai_client.base_url)
        else:
            self._base_url = base_url or os.getenv('MOONSHOTAI_BASE_URL') or 'https://api.moonshot.ai/v1'
            self._client = self._create_openai_client(base_url=self.base_url, api_key=api_key, http_client=http_client)
