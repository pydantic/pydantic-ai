from __future__ import annotations as _annotations

from typing import TYPE_CHECKING

from ..._removed_names import removed_names_getattr

try:
    from openai import NOT_GIVEN as NOT_GIVEN, omit

    from ._chat_model import OpenAIChatModel
    from ._chat_streaming import OpenAIStreamedResponse
    from ._citations import (
        chat_annotation_ta,
        chat_annotations_ta,
        responses_output_text_annotation_ta,
        responses_output_text_annotations_ta,
    )
    from ._compaction import OpenAICompaction
    from ._model_names import DEPRECATED_OPENAI_MODELS, OpenAIModelName
    from ._native_tools import MCP_SERVER_TOOL_CONNECTOR_URI_SCHEME
    from ._responses_model import OpenAIResponsesModel
    from ._responses_streaming import OpenAIResponsesStreamedResponse
    from ._settings import OpenAIChatModelSettings, OpenAIPromptCacheOptions, OpenAIResponsesModelSettings

    OMIT = omit
except ImportError as _import_error:
    raise ImportError(
        'Please install `openai` to use the OpenAI model, '
        'you can use the `openai` optional group — `pip install "pydantic-ai-slim[openai]"`'
    ) from _import_error

__all__ = (
    'DEPRECATED_OPENAI_MODELS',
    'OpenAIChatModel',
    'OpenAIResponsesModel',
    'OpenAIChatModelSettings',
    'OpenAIResponsesModelSettings',
    'OpenAIPromptCacheOptions',
    'OpenAIModelName',
    'OpenAIStreamedResponse',
    'OpenAIResponsesStreamedResponse',
    'OpenAICompaction',
    'MCP_SERVER_TOOL_CONNECTOR_URI_SCHEME',
    'responses_output_text_annotations_ta',
    'responses_output_text_annotation_ta',
    'chat_annotations_ta',
    'chat_annotation_ta',
)

_REMOVED_IN_V2 = {
    'OpenAIModel': 'use `OpenAIChatModel` (the same Chat Completions API), or `OpenAIResponsesModel`.',
    'OpenAIModelSettings': 'use `OpenAIChatModelSettings`.',
}

# Kept from type checkers, which would otherwise accept any name on this module.
if not TYPE_CHECKING:  # pragma: no branch
    __getattr__ = removed_names_getattr(__name__, _REMOVED_IN_V2)
