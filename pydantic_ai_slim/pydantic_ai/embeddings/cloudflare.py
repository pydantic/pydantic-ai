"""Embedding models hosted on Cloudflare Workers AI."""

from typing import Literal

from ..providers.cloudflare import CloudflareProvider
from .openai import OpenAIEmbeddingModel
from .settings import EmbeddingSettings

__all__ = ('CloudflareEmbeddingModel', 'CloudflareEmbeddingModelName')

CloudflareEmbeddingModelName = Literal['@cf/baai/bge-small-en-v1.5', '@cf/baai/bge-m3'] | str
"""Workers AI embedding model IDs. Unlisted IDs are accepted."""


class CloudflareEmbeddingModel(OpenAIEmbeddingModel):
    """A Workers AI embedding model using the OpenAI-compatible embeddings endpoint."""

    def __init__(
        self,
        model_name: CloudflareEmbeddingModelName,
        *,
        provider: Literal['cloudflare'] | CloudflareProvider = 'cloudflare',
        settings: EmbeddingSettings | None = None,
    ) -> None:
        """Initialize an embedding model.

        Args:
            model_name: The full Workers AI embedding model ID.
            provider: Account configuration and HTTP client.
            settings: Default embedding settings. Unsupported dimensions are rejected by the API.
        """
        if isinstance(provider, str):
            provider = CloudflareProvider()
        super().__init__(
            model_name,
            provider=provider._openai_provider,  # pyright: ignore[reportPrivateUsage]
            settings=settings,
        )
