"""The identity shared by every kind of model.

Kept apart from [`pydantic_ai.models`][pydantic_ai.models] because it is not specific to the
request-response models defined there: [`RealtimeModel`][pydantic_ai.realtime.RealtimeModel] is a
model too, and lives in [`pydantic_ai.realtime`][pydantic_ai.realtime]. Both need the same identity —
a name, a provider, and a display label — for model resolution and OpenTelemetry attributes.
"""

from __future__ import annotations as _annotations

from abc import ABC, abstractmethod
from types import TracebackType

from typing_extensions import Self

__all__ = ('AbstractModel',)


class AbstractModel(ABC):
    """Shared identity for request-response and realtime models."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """The model name."""
        raise NotImplementedError()

    @property
    @abstractmethod
    def system(self) -> str:
        """The model provider, ex: openai.

        Use to populate the `gen_ai.system` OpenTelemetry semantic convention attribute,
        so should use well-known values listed in
        https://opentelemetry.io/docs/specs/semconv/attributes-registry/gen-ai/#gen-ai-system
        when applicable.
        """
        raise NotImplementedError()

    @property
    def base_url(self) -> str | None:
        """The base URL for the provider API, if available."""
        return None

    @property
    def model_id(self) -> str:
        """The fully qualified model name in `'provider:model_name'` format."""
        return f'{self.system}:{self.model_name}'

    @property
    def context_window(self) -> int | None:
        """The maximum number of tokens the model can handle at once, input and output combined, or `None` when unknown.

        Models with a profile read it from the profile's `context_window`, which is filled from
        [genai-prices](https://github.com/pydantic/genai-prices) when no profile layer sets it. Wrapper
        models report their wrapped model's; a [`FallbackModel`][pydantic_ai.models.fallback.FallbackModel]
        reports the smallest among its candidates.
        """
        return None

    @property
    def label(self) -> str:
        """Human-friendly display label for the model.

        Handles common patterns:
        - gpt-5 -> GPT 5
        - claude-sonnet-4-5 -> Claude Sonnet 4.5
        - gemini-2.5-pro -> Gemini 2.5 Pro
        - meta-llama/llama-3-70b -> Llama 3 70b (OpenRouter style)
        """
        label = self.model_name
        # Handle OpenRouter-style names with / (e.g., meta-llama/llama-3-70b)
        if '/' in label:
            label = label.split('/')[-1]

        parts = label.split('-')
        result: list[str] = []
        for i, part in enumerate(parts):
            if i == 0 and part.lower() == 'gpt':
                result.append(part.upper())
            elif part.replace('.', '').isdigit():
                if result and result[-1].replace('.', '').isdigit():
                    result[-1] = f'{result[-1]}.{part}'
                else:
                    result.append(part)
            else:
                result.append(part.capitalize())
        return ' '.join(result)

    async def __aenter__(self) -> Self:
        """Enter the model context."""
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> bool | None:
        """Exit the model context."""
