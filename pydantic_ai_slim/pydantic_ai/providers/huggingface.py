from __future__ import annotations as _annotations

import logging
import os
from functools import lru_cache
from typing import overload

import httpx
from httpx import AsyncClient
from pydantic import TypeAdapter, ValidationError
from typing_extensions import TypedDict

from pydantic_ai import ModelProfile
from pydantic_ai.exceptions import UserError
from pydantic_ai.profiles.deepseek import deepseek_model_profile
from pydantic_ai.profiles.google import google_model_profile
from pydantic_ai.profiles.meta import meta_model_profile
from pydantic_ai.profiles.mistral import mistral_model_profile
from pydantic_ai.profiles.moonshotai import moonshotai_model_profile
from pydantic_ai.profiles.qwen import qwen_model_profile

from . import Provider

try:
    from huggingface_hub import AsyncInferenceClient
except ImportError as _import_error:  # pragma: no cover
    raise ImportError(
        'Please install the `huggingface_hub` package to use the HuggingFace provider, '
        "you can use the `huggingface` optional group — `pip install 'pydantic-ai-slim[huggingface]'`"
    ) from _import_error

_logger = logging.getLogger(__name__)

HF_ROUTER_MODELS_URL = 'https://router.huggingface.co/v1/models'


class HfRouterModel(TypedDict):
    """Hugging Face router model definition."""

    id: str


class HfRouterResponse(TypedDict):
    """Hugging Face router response."""

    data: list[HfRouterModel]


class HfRouterProvider(TypedDict):
    """Hugging Face router provider definition."""

    provider: str
    status: str
    supports_tools: bool
    supports_structured_output: bool


class HfRouterModelInfo(TypedDict):
    """Hugging Face router model info."""

    id: str
    providers: list[HfRouterProvider]


class HfRouterResponseData(TypedDict):
    """Hugging Face router response data."""

    data: HfRouterModelInfo


@lru_cache(maxsize=128)
def get_router_info(model_id: str) -> HfRouterModelInfo | None:
    """Fetch model info from the HuggingFace router API.

    This function is cached to avoid repeated HTTP requests for the same model.

    Args:
        model_id: The model ID to fetch info for (e.g., 'Qwen/Qwen2.5-72B-Instruct').

    Returns:
        The router model info if found, None otherwise.
    """
    try:
        resp = httpx.get(f'{HF_ROUTER_MODELS_URL}/{model_id}', timeout=5.0, follow_redirects=True)
        if resp.status_code != 200:
            return None
        payload = TypeAdapter(HfRouterResponseData).validate_json(resp.content)
        return payload['data']
    except (httpx.HTTPError, ValidationError, Exception):
        return None


def select_provider(providers: list[HfRouterProvider]) -> HfRouterProvider | None:
    """Select the best provider based on capabilities."""
    live_providers = [p for p in providers if p['status'] == 'live']
    if not live_providers:
        live_providers = providers

    # 1 - supports_tools=True AND supports_structured_output=True
    both = [p for p in live_providers if p['supports_tools'] and p['supports_structured_output']]
    if both:
        return both[0]

    # 2 - supports_tools=True OR supports_structured_output=True
    either = [p for p in live_providers if p['supports_tools'] or p['supports_structured_output']]
    if either:
        return either[0]

    # 3 - Any
    return live_providers[0] if live_providers else None


class HuggingFaceProvider(Provider[AsyncInferenceClient]):
    """Provider for Hugging Face."""

    @property
    def name(self) -> str:
        return 'huggingface'

    @property
    def base_url(self) -> str:
        return self.client.model  # type: ignore

    @property
    def client(self) -> AsyncInferenceClient:
        return self._client

    @property
    def provider_name(self) -> str | None:
        """The user-specified provider name, if any."""
        return self._provider

    def model_profile(self, model_name: str) -> ModelProfile | None:
        """Return the base model profile without fetching router info.

        Router info fetching is deferred to HuggingFaceModel.request() to avoid
        making HTTP requests during synchronous initialization.
        """
        provider_to_profile = {
            'deepseek-ai': deepseek_model_profile,
            'google': google_model_profile,
            'qwen': qwen_model_profile,
            'meta-llama': meta_model_profile,
            'mistralai': mistral_model_profile,
            'moonshotai': moonshotai_model_profile,
        }

        if '/' not in model_name:
            return None

        model_name = model_name.lower()
        model_prefix, model_suffix = model_name.split('/', 1)

        if model_prefix in provider_to_profile:
            return provider_to_profile[model_prefix](model_suffix)

        return None

    @overload
    def __init__(self, *, base_url: str, api_key: str | None = None) -> None: ...
    @overload
    def __init__(self, *, provider_name: str, api_key: str | None = None) -> None: ...
    @overload
    def __init__(self, *, hf_client: AsyncInferenceClient, api_key: str | None = None) -> None: ...
    @overload
    def __init__(self, *, hf_client: AsyncInferenceClient, base_url: str, api_key: str | None = None) -> None: ...
    @overload
    def __init__(self, *, hf_client: AsyncInferenceClient, provider_name: str, api_key: str | None = None) -> None: ...
    @overload
    def __init__(self, *, api_key: str | None = None) -> None: ...

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        hf_client: AsyncInferenceClient | None = None,
        http_client: AsyncClient | None = None,
        provider_name: str | None = None,
    ) -> None:
        """Create a new Hugging Face provider.

        Args:
            base_url: The base url for the Hugging Face requests.
            api_key: The API key to use for authentication, if not provided, the `HF_TOKEN` environment variable
                will be used if available.
            hf_client: An existing
                [`AsyncInferenceClient`](https://huggingface.co/docs/huggingface_hub/v0.29.3/en/package_reference/inference_client#huggingface_hub.AsyncInferenceClient)
                client to use. If not provided, a new instance will be created.
            http_client: (currently ignored) An existing `httpx.AsyncClient` to use for making HTTP requests.
            provider_name: Name of the provider to use for inference. available providers can be found in the [HF Inference Providers documentation](https://huggingface.co/docs/inference-providers/index#partners).
                defaults to "auto", which will select the first available provider for the model, the first of the providers available for the model, sorted by the user's order in https://hf.co/settings/inference-providers.
                If `base_url` is passed, then `provider_name` is not used.
        """
        api_key = api_key or os.getenv('HF_TOKEN')
        if api_key is None and hf_client is not None:
            api_key = getattr(hf_client, 'token', None)

        self.api_key = api_key

        if self.api_key is None:
            raise UserError(
                'Set the `HF_TOKEN` environment variable or pass it via `HuggingFaceProvider(api_key=...)`'
                'to use the HuggingFace provider.'
            )

        if http_client is not None:
            raise ValueError('`http_client` is ignored for HuggingFace provider, please use `hf_client` instead.')

        if base_url is not None and provider_name is not None:
            raise ValueError('Cannot provide both `base_url` and `provider_name`.')

        if hf_client is None:
            self._client = AsyncInferenceClient(api_key=self.api_key, provider=provider_name, base_url=base_url)  # type: ignore
        else:
            self._client = hf_client

        self._provider: str | None = provider_name
