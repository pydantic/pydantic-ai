"""Chat Completions models hosted on Cloudflare Workers AI."""

from typing import Literal

from ..profiles import ModelProfileSpec
from ..providers.cloudflare import CloudflareProvider
from ..settings import ModelSettings
from .openai import OpenAIChatModel

__all__ = ('CloudflareChatModel', 'CloudflareChatModelName')

CloudflareChatModelName = (
    Literal[
        '@cf/meta/llama-4-scout-17b-16e-instruct',
        '@cf/qwen/qwen3-30b-a3b-fp8',
        '@cf/openai/gpt-oss-20b',
    ]
    | str
)
"""Workers AI chat model IDs. Unlisted IDs are accepted without catalog lookup."""


class CloudflareChatModel(OpenAIChatModel):
    """A chat model hosted on Cloudflare Workers AI's OpenAI-compatible endpoint.

    Configure account credentials with a
    [`CloudflareProvider`][pydantic_ai.providers.cloudflare.CloudflareProvider].
    See [Cloudflare](../../models/cloudflare.md) for supported capabilities and model discovery.
    """

    def __init__(
        self,
        model_name: CloudflareChatModelName,
        *,
        provider: Literal['cloudflare'] | CloudflareProvider = 'cloudflare',
        profile: ModelProfileSpec | None = None,
        settings: ModelSettings | None = None,
    ) -> None:
        """Initialize a native Workers AI chat model.

        Args:
            model_name: The full Workers AI model ID.
            provider: Account configuration and HTTP client.
            profile: Capability overrides for the selected model.
            settings: Default model settings.
        """
        if isinstance(provider, str):
            provider = CloudflareProvider()
        super().__init__(
            model_name,
            provider=provider._openai_provider,  # pyright: ignore[reportPrivateUsage]
            profile=profile,
            settings=settings,
        )
