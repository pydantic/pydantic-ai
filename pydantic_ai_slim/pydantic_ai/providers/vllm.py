from __future__ import annotations as _annotations

import os
from typing import overload

from pydantic_ai import ModelProfile
from pydantic_ai.exceptions import UserError
from pydantic_ai.profiles import merge_profile
from pydantic_ai.profiles.cohere import cohere_model_profile
from pydantic_ai.profiles.deepseek import deepseek_model_profile
from pydantic_ai.profiles.google import google_model_profile
from pydantic_ai.profiles.harmony import harmony_model_profile
from pydantic_ai.profiles.meta import meta_model_profile
from pydantic_ai.profiles.mistral import mistral_model_profile
from pydantic_ai.profiles.openai import OpenAIJsonSchemaTransformer, OpenAIModelProfile
from pydantic_ai.profiles.qwen import qwen_model_profile
from pydantic_ai.profiles.zai import zai_model_profile

try:
    from openai import AsyncOpenAI
except ImportError as _import_error:
    raise ImportError(
        'Please install the `openai` package to use the vLLM provider, '
        'you can use the `openai` optional group — `pip install "pydantic-ai-slim[openai]"`'
    ) from _import_error
else:
    from ._openai_compatible import (
        AsyncHTTPClient as _OpenAIHTTPClient,
        OpenAICompatibleProvider as _OpenAICompatibleProvider,
    )


class VLLMProvider(_OpenAICompatibleProvider):
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
            'glm': zai_model_profile,
        }

        model_name = model_name.lower()
        # Match both parts of Hugging Face repo IDs, preserving org-based families such as `mistralai/Mixtral`.
        bare_name = model_name.rpartition('/')[2]
        # The Qwen profile uses the provider spelling `qwen-3-coder`, while Hugging Face uses `qwen3-coder`.
        profile_name = bare_name.replace('qwen3-coder', 'qwen-3-coder', 1)
        profile = None
        for prefix, profile_func in prefix_to_profile.items():
            if model_name.startswith(prefix) or bare_name.startswith(prefix):
                profile = profile_func(profile_name)
                break
        # vLLM maps `reasoning_effort` to the chat-template switch used by these reasoning families.
        # Qwen3-Coder is excluded because its official model card documents non-thinking-only behavior.
        # Explicit Qwen3 `-Thinking` checkpoints are always-on.
        # Qwen3.8 is excluded because its effort ladder does not match the unified OpenAI-style values.
        # See https://docs.vllm.ai/en/stable/features/reasoning_outputs/ and
        # https://huggingface.co/Qwen/Qwen3-Coder-480B-A35B-Instruct and
        # https://huggingface.co/Qwen/Qwen3-235B-A22B-Thinking-2507 and
        # https://huggingface.co/Qwen/Qwen3.8-27B.
        supports_qwen3_thinking = (
            bare_name.startswith('qwen3')
            and not bare_name.startswith(('qwen3-coder', 'qwen3.8'))
            and '-instruct' not in bare_name
        )
        supports_thinking = supports_qwen3_thinking or bare_name.startswith(('gemma-4', 'deepseek-v4-'))
        thinking_profile = None
        if supports_thinking:
            thinking_profile = ModelProfile(
                supports_thinking=True,
                thinking_always_enabled=bare_name.startswith('qwen3') and '-thinking' in bare_name,
            )

        # vLLM supports required tool choice and strict schemas for every supported model, including Qwen3-Coder.
        # See https://docs.vllm.ai/en/stable/features/tool_calling/.
        qwen3_coder_profile = None
        if bare_name.startswith('qwen3-coder'):
            qwen3_coder_profile = OpenAIModelProfile(
                openai_supports_tool_choice_required=True,
                openai_supports_strict_tool_definition=True,
            )

        # `json_schema_transformer` is a fallback (the family profile wins if it set one). The other overrides
        # win on top of the family profile:
        # - The Chat Completions API supports `json_schema`/`json_object` response formats via server-side
        #   guided decoding. That is pure token masking, so the model only sees the schema if it is also
        #   injected into the instructions. See https://github.com/pydantic/pydantic-ai/issues/3490.
        # - File content parts and native tool return schemas are not supported.
        # - Some chat templates served by vLLM reject more than one leading system message.
        #   See https://github.com/pydantic/pydantic-ai/issues/5812.
        # The gpt-oss family opt-out for `openai_supports_tool_choice_required` survives the merge because
        # vLLM ignores `tool_choice='required'` for gpt-oss.
        # See https://github.com/vllm-project/vllm/issues/44216.
        return merge_profile(
            OpenAIModelProfile(json_schema_transformer=OpenAIJsonSchemaTransformer),
            profile,
            thinking_profile,
            qwen3_coder_profile,
            OpenAIModelProfile(
                openai_chat_supports_document_input=False,
                openai_chat_supports_multiple_system_messages=False,
                supports_tool_return_schema=False,
                supports_json_schema_output=True,
                supports_json_object_output=True,
                native_output_requires_schema_in_instructions=True,
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
        http_client: _OpenAIHTTPClient | None = None,
    ) -> None: ...

    def __init__(
        self,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        openai_client: AsyncOpenAI | None = None,
        http_client: _OpenAIHTTPClient | None = None,
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
            http_client: An existing `httpx2.AsyncClient` or legacy `httpx.AsyncClient` to use for making HTTP requests.
        """
        if openai_client is not None:
            if base_url is not None:
                raise UserError('Cannot provide both `openai_client` and `base_url`')
            if http_client is not None:
                raise UserError('Cannot provide both `openai_client` and `http_client`')
            if api_key is not None:
                raise UserError('Cannot provide both `openai_client` and `api_key`')
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

            self._client = self._create_openai_client(base_url=base_url, api_key=api_key, http_client=http_client)
