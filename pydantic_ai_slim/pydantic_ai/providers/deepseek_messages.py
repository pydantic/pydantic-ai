"""DeepSeek's Anthropic-compatible Messages endpoint."""

import os

import httpx2

from ..native_tools import WebFetchTool, WebSearchTool
from ..profiles.anthropic import AnthropicModelProfile
from . import missing_api_key_error
from .anthropic import AnthropicProvider


class DeepSeekMessagesProvider(AnthropicProvider):
    """Use DeepSeek's Messages API with `AnthropicModel`."""

    _model_id_namespace = 'deepseek'

    @property
    def name(self) -> str:
        return 'deepseek-messages'

    @property
    def canonical_system(self) -> str:
        return 'deepseek-messages'

    @staticmethod
    def model_profile(model_name: str) -> AnthropicModelProfile:
        flash = model_name == 'deepseek-flash'
        return AnthropicModelProfile(
            anthropic_deepseek_web_tools=flash,
            supports_thinking=flash,
            thinking_enabled_by_default=flash,
            supports_forced_tool_choice_with_thinking=False,
            supported_native_tools=frozenset({WebSearchTool, WebFetchTool}) if flash else frozenset(),
        )

    def __init__(self, *, api_key: str | None = None, http_client: httpx2.AsyncClient | None = None) -> None:
        api_key = api_key or os.getenv('DEEPSEEK_API_KEY')
        if not api_key:
            raise missing_api_key_error('Set `DEEPSEEK_API_KEY` or pass `api_key` to `DeepSeekMessagesProvider`.')
        super().__init__(api_key=api_key, base_url='https://api.deepseek.com/anthropic', http_client=http_client)
