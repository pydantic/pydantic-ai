"""Cerebras model implementation using OpenAI-compatible API."""

from __future__ import annotations as _annotations

from dataclasses import dataclass
from typing import Literal

from ..profiles import ModelProfileSpec
from ..providers import Provider
from ..settings import ModelSettings
from .openai import OpenAIChatModel

try:
    from openai import AsyncOpenAI
except ImportError as _import_error:  # pragma: no cover
    raise ImportError(
        'Please install the `openai` package to use the Cerebras model, '
        'you can use the `cerebras` optional group — `pip install "pydantic-ai-slim[cerebras]"'
    ) from _import_error

__all__ = ('CerebrasModel', 'CerebrasModelName', 'CerebrasModelSettings')

_KnownCerebrasModelName = Literal[
    'gpt-oss-120b',
    'llama-3.3-70b',
    'llama3.1-8b',
    'qwen-3-235b-a22b-instruct-2507',
    'qwen-3-32b',
    'zai-glm-4.6',
]

CerebrasModelName = str | _KnownCerebrasModelName
"""Possible Cerebras model names.

Since Cerebras supports a variety of models and the list changes frequently, we explicitly list known models
but allow any name in the type hints.

See <https://inference-docs.cerebras.ai/models/overview> for an up to date list of models.
"""


class CerebrasModelSettings(ModelSettings, total=False):
    """Settings used for a Cerebras model request.

    ALL FIELDS MUST BE `cerebras_` PREFIXED SO YOU CAN MERGE THEM WITH OTHER MODELS.
    """

    cerebras_disable_reasoning: bool
    """Disable reasoning for the model.

    See [the Cerebras docs](https://inference-docs.cerebras.ai/resources/openai#passing-non-standard-parameters) for more details.
    """


@dataclass(init=False)
class CerebrasModel(OpenAIChatModel):
    """A model that uses Cerebras's OpenAI-compatible API.

    Cerebras provides ultra-fast inference powered by the Wafer-Scale Engine (WSE).

    Apart from `__init__`, all methods are private or match those of the base class.
    """

    def __init__(
        self,
        model_name: CerebrasModelName,
        *,
        provider: Literal['cerebras'] | Provider[AsyncOpenAI] = 'cerebras',
        profile: ModelProfileSpec | None = None,
        settings: CerebrasModelSettings | None = None,
    ):
        """Initialize a Cerebras model.

        Args:
            model_name: The name of the Cerebras model to use.
            provider: The provider to use. Defaults to 'cerebras'.
            profile: The model profile to use. Defaults to a profile based on the model name.
            settings: Model-specific settings that will be used as defaults for this model.
        """
        super().__init__(model_name, provider=provider, profile=profile, settings=settings)
