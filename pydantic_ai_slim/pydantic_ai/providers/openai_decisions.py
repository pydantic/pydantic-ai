from __future__ import annotations as _annotations

from pydantic_ai import ModelProfile
from pydantic_ai.profiles.decision import decision_model_profile

try:
    from pydantic_ai.providers.openai import OpenAIProvider
except ImportError as _import_error:  # pragma: no cover
    raise ImportError(
        'Please install the `openai` package to use the OpenAI Decisions provider, '
        'you can use the `openai` optional group — `pip install "pydantic-ai-slim[openai]"`'
    ) from _import_error


class OpenAIDecisionsProvider(OpenAIProvider):
    """Provider for OpenAI's Decisions API.

    Takes the same arguments as [`OpenAIProvider`][pydantic_ai.providers.openai.OpenAIProvider] and reads the same
    `OPENAI_API_KEY`, but profiles a model ID such as `gpt-6-luna` as a decision model for
    [`OpenAIDecisionsModel`][pydantic_ai.models.openai_decisions.OpenAIDecisionsModel], rather than as the Responses
    API model `OpenAIProvider` takes it for.
    """

    _model_id_namespace = 'openai-decisions'

    @staticmethod
    def model_profile(model_name: str) -> ModelProfile | None:
        return decision_model_profile(model_name)
