"""Crusoe model implementation using OpenAI-compatible API."""

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
        'Please install the `openai` package to use the Crusoe model, '
        'you can use the `crusoe` optional group — `pip install "pydantic-ai-slim[crusoe]"`'
    ) from _import_error

__all__ = ('CrusoeModel', 'CrusoeModelName')

LatestCrusoeModelNames = Literal[
    'Qwen/Qwen3-235B-A22B-Instruct-2507',
    'deepseek-ai/DeepSeek-V3-0324',
    'deepseek-ai/DeepSeek-V4-Pro',
    'deepseek-ai/Deepseek-V4-Flash',
    'google/gemma-4-31b-it',
    'meta-llama/Llama-3.3-70B-Instruct',
    'moonshotai/Kimi-K2.6',
    'nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B',
    'nvidia/NVIDIA-Nemotron-3-Super-120B-A12B',
    'nvidia/Nemotron-3-Nano-Omni-Reasoning-30B-A3B',
    'nvidia/Nemotron-3.5-Lightning-30B-A3B',
    'openai/gpt-oss-120b',
    'yutori/n1.5',
    'zai/GLM-5.1',
    'zai/GLM-5.2',
]

CrusoeModelName = str | LatestCrusoeModelNames
"""Possible Crusoe model names.

Since Crusoe supports a variety of models and the list changes frequently, we explicitly list known models
but allow any name in the type hints.

See <https://docs.crusoecloud.com/serverless-inference/overview> for an up to date list of models.
"""


@dataclass(init=False)
class CrusoeModel(OpenAIChatModel):
    """A model that uses Crusoe's OpenAI-compatible Serverless Inference API.

    Crusoe serves open-weight models from many labs behind one endpoint, so the model family — and with
    it the profile [`CrusoeProvider`][pydantic_ai.providers.crusoe.CrusoeProvider] resolves — is derived
    from the vendor prefix on the model name (`zai/`, `deepseek-ai/`, `meta-llama/`, …).

    Every model is served with guided decoding, so [`NativeOutput`][pydantic_ai.output.NativeOutput] works
    across the catalog, including for families whose own profiles don't claim native structured output
    support. Thinking is returned in a non-standard field (`reasoning`, or `reasoning_content` for
    DeepSeek), both of which `OpenAIChatModel` reads.

    Apart from `__init__`, all methods are inherited from the base class.
    """

    def __init__(
        self,
        model_name: CrusoeModelName,
        *,
        provider: Literal['crusoe'] | Provider[AsyncOpenAI] = 'crusoe',
        profile: ModelProfileSpec | None = None,
        settings: ModelSettings | None = None,
    ):
        """Initialize a Crusoe model.

        Args:
            model_name: The name of the Crusoe model to use, including the vendor prefix (e.g. `'zai/GLM-5.2'`).
            provider: The provider to use. Defaults to `'crusoe'`.
            profile: The model profile to use. Defaults to a profile picked by the provider based on the model name.
            settings: Model-specific settings that will be used as defaults for this model.
        """
        super().__init__(model_name, provider=provider, profile=profile, settings=settings)
