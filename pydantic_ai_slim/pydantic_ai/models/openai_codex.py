from __future__ import annotations as _annotations

from typing import Literal

from typing_extensions import override

from ..messages import ModelRequest, ModelResponse
from ..profiles import ModelProfileSpec
from ..providers import Provider
from ..settings import ModelSettings

try:
    from openai import AsyncOpenAI

    from .openai import OpenAIModelName, OpenAIResponsesModel, OpenAIResponsesModelSettings
except ImportError as _import_error:  # pragma: no cover
    raise ImportError(
        'Please install the `openai` package to use the OpenAI Codex model, '
        'you can use the `openai` optional group — `pip install "pydantic-ai-slim[openai]"`'
    ) from _import_error

__all__ = ('OpenAICodexModel',)

_SESSION_HEADERS = ('session-id', 'thread-id', 'x-client-request-id')


class OpenAICodexModel(OpenAIResponsesModel):
    """A model that uses the OpenAI Codex backend under a ChatGPT/Codex subscription.

    Beyond the Responses API dialect described by the model profile, the Codex backend keys prompt caching off
    a stable session identity. This model mirrors the official Codex client by sending the `session-id`,
    `thread-id`, and `x-client-request-id` headers and the `prompt_cache_key` field, all derived from the
    `conversation_id` of the message history. Explicit `extra_headers` and `openai_prompt_cache_key` settings win.

    Apart from `__init__`, all methods are private or match those of the base class.
    """

    def __init__(
        self,
        model_name: OpenAIModelName,
        *,
        provider: Literal['openai-codex'] | Provider[AsyncOpenAI] = 'openai-codex',
        profile: ModelProfileSpec | None = None,
        settings: ModelSettings | None = None,
    ):
        """Initialize an OpenAI Codex model.

        Args:
            model_name: The name of the OpenAI model to use.
            provider: The provider to use. Defaults to `'openai-codex'`.
            profile: The model profile to use. Defaults to a profile picked by the provider based on the model name.
            settings: Default model settings for this model instance.
        """
        super().__init__(model_name, provider=provider, profile=profile, settings=settings)

    @override
    def _prepare_responses_settings(
        self,
        messages: list[ModelRequest | ModelResponse],
        model_settings: OpenAIResponsesModelSettings,
    ) -> OpenAIResponsesModelSettings:
        session_id = next((m.conversation_id for m in reversed(messages) if m.conversation_id), None)
        if session_id is None:
            return model_settings

        # HTTP field names are case-insensitive, so a case-variant override counts as supplied.
        extra_headers = dict(model_settings.get('extra_headers', {}))
        supplied_headers = {name.lower() for name in extra_headers}
        for header in _SESSION_HEADERS:
            if header not in supplied_headers:
                extra_headers[header] = session_id

        model_settings['extra_headers'] = extra_headers
        model_settings.setdefault('openai_prompt_cache_key', session_id)
        return model_settings
