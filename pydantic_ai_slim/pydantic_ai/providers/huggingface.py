from __future__ import annotations as _annotations

import os

from mistralai import httpx

try:
    from huggingface_hub import AsyncInferenceClient
except ImportError as _import_error:  # pragma: no cover
    raise ImportError(
        'Please install the `huggingface_hub` package to use the HuggingFace provider, '
        "you can use the `huggingface` optional group — `pip install 'pydantic-ai-slim[huggingface]'`"
    ) from _import_error

from . import Provider


class HuggingFaceProvider(Provider[AsyncInferenceClient]):
    """Provider for HuggingFace API."""

    @property
    def name(self) -> str:
        return 'huggingface'

    @property
    def base_url(self) -> str:
        return self.client.model  # type: ignore

    @property
    def client(self) -> AsyncInferenceClient:
        return self._client

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        hf_client: AsyncInferenceClient | None = None,
        http_client: httpx.AsyncClient | None = None,
        provider: str | None = None,
    ) -> None:
        """Create a new Hugging Face provider.

        Args:
            base_url: The base url for the Hugging Face requests. If not provided, it will default to the HF Inference API base url.
            api_key: The API key to use for authentication, if not provided, the `HF_TOKEN` environment variable
                will be used if available.
            hf_client: An existing
                [`AsyncInferenceClient`](https://huggingface.co/docs/huggingface_hub/v0.29.3/en/package_reference/inference_client#huggingface_hub.AsyncInferenceClient)
                client to use. If not provided, a new instance will be created.
            http_client: (currently ignored) An existing `httpx.AsyncClient` to use for making HTTP requests.
            provider : Name of the provider to use for inference. available providers can be found in the [HF Inference Providers documentation](https://huggingface.co/docs/inference-providers/index#partners).
                defaults to "auto", which will select the first available provider for the model, the first of the providers available for the model, sorted by the user's order in https://hf.co/settings/inference-providers.
                If `base_url` is passed, then `provider` is not used.
        """
        api_key = api_key or os.environ.get('HF_TOKEN')

        if api_key is None:
            raise ValueError(
                'Set the `HF_TOKEN` environment variable or pass it via `HuggingFaceProvider(api_key=...)`'
                'to use the HuggingFace provider.'
            )

        if http_client is not None:
            raise ValueError('`http_client` is ignored for HuggingFace provider, please use `hf_client` instead')

        if base_url is not None and provider is not None:
            raise ValueError('Cannot provide both `base_url` and `provider`')

        if hf_client is None:
            self._client = AsyncInferenceClient(api_key=api_key, provider=provider, base_url=base_url)  # type: ignore
        else:
            self._client = hf_client
