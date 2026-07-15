from __future__ import annotations as _annotations

import os
from typing import overload

import httpx

from pydantic_ai import ModelProfile
from pydantic_ai.exceptions import UserError
from pydantic_ai.models import create_async_http_client
from pydantic_ai.profiles import merge_profile
from pydantic_ai.profiles.cohere import cohere_model_profile
from pydantic_ai.profiles.deepseek import deepseek_model_profile
from pydantic_ai.profiles.google import google_model_profile
from pydantic_ai.profiles.harmony import harmony_model_profile
from pydantic_ai.profiles.meta import meta_model_profile
from pydantic_ai.profiles.mistral import mistral_model_profile
from pydantic_ai.profiles.openai import OpenAIJsonSchemaTransformer, OpenAIModelProfile
from pydantic_ai.profiles.qwen import qwen_model_profile
from pydantic_ai.providers import Provider

try:
    from openai import AsyncOpenAI
except ImportError as _import_error:  # pragma: no cover
    raise ImportError(
        'Please install the `openai` package to use the vLLM provider, '
        'you can use the `openai` optional group: `pip install "pydantic-ai-slim[openai]"`'
    ) from _import_error


class VLLMProvider(Provider[AsyncOpenAI]):
    """Provider for local or remote vLLM API."""

    @property
    def name(self) -> str:
        return 'vllm'

    @property
    def base_url(self) -> str:
        return str(self.client.base_url)

    @property
    def client(self) -> AsyncOpenAI:
        return self._client

    @staticmethod
    def model_profile(model_name: str) -> ModelProfile | None:
        prefix_to_profile = {
            'llama': meta_model_profile,
            'gemma': google_model_profile,
            'qwen': qwen_model_profile,
            'qwq': qwen_model_profile,
            'deepseek': deepseek_model_profile,
            'mistral': mistral_model_profile,
            'command': cohere_model_profile,
            'c4ai-command': cohere_model_profile,
            'gpt-oss': harmony_model_profile,
        }

        model_name = model_name.lower()
        # vLLM model names are usually Hugging Face repo IDs like `meta-llama/Llama-3-8B`, where the family
        # appears after the org namespace. Match the prefix against the full id (so org-based families like
        # `mistralai/...` keep matching) and the bare name after the namespace, and pass the bare name to the
        # profile function so its own name-based detection (e.g. `qwen_model_profile`) sees a clean model id.
        bare_name = model_name.rpartition('/')[2]
        profile = None
        for prefix, profile_func in prefix_to_profile.items():
            if model_name.startswith(prefix) or bare_name.startswith(prefix):
                profile = profile_func(bare_name)

        # `json_schema_transformer` is a fallback (the upstream model profile wins if it set one). The other
        # overrides win on top: vLLM's /v1/chat/completions endpoint supports response_format with json_schema
        # and strict function tools. It does not support OpenAI file content parts or native tool return schemas.
        # `openai_chat_supports_multiple_system_messages` is forced off because some chat templates served by
        # vLLM reject more than one leading system message. See #5812.
        #
        # `openai_chat_thinking_field='reasoning'` matches vLLM, which renamed `reasoning_content` to `reasoning`
        # to follow OpenAI's gpt-oss guidance (vllm-project/vllm#27752). Older vLLM servers only emit
        # `reasoning_content`, but reading falls back to it (see `OpenAIChatModel._process_thinking`), and
        # this only sets the field used when sending reasoning back on multi-turn requests.
        return merge_profile(
            OpenAIModelProfile(json_schema_transformer=OpenAIJsonSchemaTransformer),
            profile,
            OpenAIModelProfile(
                openai_chat_thinking_field='reasoning',
                openai_supports_strict_tool_definition=True,
                openai_chat_supports_document_input=False,
                supports_tool_return_schema=False,
                supports_json_schema_output=True,
                supports_json_object_output=True,
                openai_chat_supports_multiple_system_messages=False,
            ),
        )

    @overload
    def __init__(self, *, openai_client: AsyncOpenAI) -> None: ...

    @overload
    def __init__(
        self,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        openai_client: None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None: ...

    def __init__(
        self,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        openai_client: AsyncOpenAI | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        """Create a new vLLM provider.

        Args:
            base_url: The base url for the vLLM requests. If not provided, the `VLLM_BASE_URL` environment variable
                will be used if available.
            api_key: The API key to use for authentication, if not provided, the `VLLM_API_KEY` environment variable
                will be used if available.
            openai_client: An existing
                [`AsyncOpenAI`](https://github.com/openai/openai-python?tab=readme-ov-file#async-usage)
                client to use. If provided, `base_url`, `api_key`, and `http_client` must be `None`.
            http_client: An existing `httpx.AsyncClient` to use for making HTTP requests.
        """
        if openai_client is not None:
            assert base_url is None, 'Cannot provide both `openai_client` and `base_url`'
            assert http_client is None, 'Cannot provide both `openai_client` and `http_client`'
            assert api_key is None, 'Cannot provide both `openai_client` and `api_key`'
            self._client = openai_client
        else:
            base_url = base_url or os.getenv('VLLM_BASE_URL')
            if not base_url:
                raise UserError(
                    'Set the `VLLM_BASE_URL` environment variable or pass it via `VLLMProvider(base_url=...)`'
                    ' to use the vLLM provider.'
                )

            # This is a workaround for the OpenAI client requiring an API key, whilst locally served,
            # openai compatible models do not always need an API key, but a placeholder (non-empty) key is required.
            api_key = api_key or os.getenv('VLLM_API_KEY') or 'api-key-not-set'

            if http_client is not None:
                self._client = AsyncOpenAI(base_url=base_url, api_key=api_key, http_client=http_client)
            else:
                http_client = create_async_http_client()
                self._own_http_client = http_client
                self._http_client_factory = create_async_http_client
                self._client = AsyncOpenAI(base_url=base_url, api_key=api_key, http_client=http_client)

    def _set_http_client(self, http_client: httpx.AsyncClient) -> None:
        self._client._client = http_client  # pyright: ignore[reportPrivateUsage]
