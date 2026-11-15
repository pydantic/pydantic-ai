from __future__ import annotations as _annotations

from typing_extensions import deprecated

from pydantic_ai import ModelProfile
from pydantic_ai._warnings import PydanticAIDeprecationWarning
from pydantic_ai.profiles.decision import decision_model_profile

try:
    from pydantic_ai.providers.openai import OpenAIProvider
except ImportError as _import_error:  # pragma: no cover
    raise ImportError(
        'Please install the `openai` package to use the OpenAI Decisions provider, '
        'you can use the `openai` optional group — `pip install "pydantic-ai-slim[openai]"`'
    ) from _import_error


@deprecated(
    '`OpenAIDecisionsProvider` is deprecated and will be removed in v3. '
    'Use `OpenAIProvider` with `OpenAIDecisionsModel` instead; it takes the same arguments.',
    category=PydanticAIDeprecationWarning,
)
class OpenAIDecisionsProvider(OpenAIProvider):
    """Deprecated provider for OpenAI's Decisions API, to be removed in v3.

    Use [`OpenAIProvider`][pydantic_ai.providers.openai.OpenAIProvider] with
    [`OpenAIDecisionsModel`][pydantic_ai.models.openai_decisions.OpenAIDecisionsModel] instead: it takes the same
    arguments, and the model profiles itself and names its responses `'openai-decisions'`, as `OpenAIChatModel` and
    `OpenAIResponsesModel` share `OpenAIProvider`.
    """

    @property
    def name(self) -> str:
        return 'openai-decisions'

    @staticmethod
    def model_profile(model_name: str) -> ModelProfile | None:
        return decision_model_profile(model_name)
