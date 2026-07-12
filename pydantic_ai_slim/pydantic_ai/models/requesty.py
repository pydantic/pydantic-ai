"""Requesty model implementation using OpenAI-compatible API."""

from __future__ import annotations as _annotations

from dataclasses import dataclass
from typing import Literal

from ..profiles import ModelProfileSpec
from ..providers import Provider
from ..settings import ModelSettings

try:
    from openai import AsyncOpenAI

    from .openai import OpenAIChatModel
except ImportError as _import_error:
    raise ImportError(
        'Please install the `openai` package to use the Requesty model, '
        'you can use the `requesty` optional group — `pip install "pydantic-ai-slim[requesty]"'
    ) from _import_error

__all__ = ('RequestyModel', 'RequestyModelName')

RequestyModelName = str
"""Possible Requesty model names.

Requesty is a router/gateway that exposes models using `provider/model` naming (e.g. `openai/gpt-4o-mini`,
`anthropic/claude-sonnet-4-5`). Since the catalog is large and changes frequently, any name is allowed.

See <https://app.requesty.ai/router/list> for an up to date list of models.
"""


@dataclass(init=False)
class RequestyModel(OpenAIChatModel):
    """A model that uses Requesty's OpenAI-compatible API.

    Requesty is an LLM gateway/router that provides access to models from many providers through a single
    OpenAI-compatible endpoint.

    Apart from `__init__`, all methods are private or match those of the base class.
    """

    def __init__(
        self,
        model_name: RequestyModelName,
        *,
        provider: Literal['requesty'] | Provider[AsyncOpenAI] = 'requesty',
        profile: ModelProfileSpec | None = None,
        settings: ModelSettings | None = None,
    ):
        """Initialize a Requesty model.

        Args:
            model_name: The name of the Requesty model to use.
            provider: The provider to use. Defaults to 'requesty'.
            profile: The model profile to use. Defaults to a profile based on the model name.
            settings: Model-specific settings that will be used as defaults for this model.
        """
        super().__init__(model_name, provider=provider, profile=profile, settings=settings)
