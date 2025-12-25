"""OpenAI model implementations.

This package provides support for both the Chat Completions API and the Responses API.
"""

from ...profiles.openai import OpenAISystemPromptRole
from ._shared import (
    OpenAIChatModelSettings,
    OpenAIModelName,
    OpenAIModelSettings,  # pyright: ignore[reportDeprecated]
    OpenAIResponsesModelSettings,
)
from .completions import (
    OpenAIChatModel,
    OpenAIModel,  # pyright: ignore[reportDeprecated]
    OpenAIStreamedResponse,
)
from .responses import (
    OpenAIResponsesModel,
    OpenAIResponsesStreamedResponse,
)

try:
    from openai import NOT_GIVEN, omit

    OMIT = omit
except ImportError:
    pass

__all__ = (
    'OpenAIModel',
    'OpenAIChatModel',
    'OpenAIResponsesModel',
    'OpenAIModelSettings',
    'OpenAIChatModelSettings',
    'OpenAIResponsesModelSettings',
    'OpenAIModelName',
    'OpenAIStreamedResponse',
    'OpenAIResponsesStreamedResponse',
    'OpenAISystemPromptRole',
    'NOT_GIVEN',
    'OMIT',
)
