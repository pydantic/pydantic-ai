"""Azure-specific content filter handling for OpenAI models."""

from __future__ import annotations as _annotations

from typing import Any

from pydantic import BaseModel, ValidationError

from ... import _utils
from ...messages import ModelResponse

try:
    from openai import APIStatusError
except ImportError as _import_error:
    raise ImportError(
        'Please install `openai` to use the OpenAI model, '
        'you can use the `openai` optional group — `pip install "pydantic-ai-slim[openai]"`'
    ) from _import_error


class _AzureContentFilterResultDetail(BaseModel):
    filtered: bool
    severity: str | None = None
    detected: bool | None = None


class _AzureContentFilterResult(BaseModel):
    hate: _AzureContentFilterResultDetail | None = None
    self_harm: _AzureContentFilterResultDetail | None = None
    sexual: _AzureContentFilterResultDetail | None = None
    violence: _AzureContentFilterResultDetail | None = None
    jailbreak: _AzureContentFilterResultDetail | None = None
    profanity: _AzureContentFilterResultDetail | None = None


class _AzureInnerError(BaseModel):
    code: str
    content_filter_result: _AzureContentFilterResult


class _AzureError(BaseModel):
    code: str
    message: str
    innererror: _AzureInnerError | None = None


class _AzureErrorResponse(BaseModel):
    error: _AzureError


def check_azure_content_filter(e: APIStatusError, system: str, model_name: str) -> ModelResponse | None:
    """Check if the error is an Azure content filter error.

    Args:
        e: The API status error from OpenAI
        system: The system/provider name (e.g., 'azure')
        model_name: The model name being used

    Returns:
        A ModelResponse with content filter details if this is a content filter error,
        None otherwise
    """
    # Assign to Any to avoid 'dict[Unknown, Unknown]' inference in strict mode
    body_any: Any = e.body

    if system == 'azure' and e.status_code == 400 and isinstance(body_any, dict):
        try:
            error_data = _AzureErrorResponse.model_validate(body_any)

            if error_data.error.code == 'content_filter':
                provider_details: dict[str, Any] = {'finish_reason': 'content_filter'}

                if error_data.error.innererror:
                    provider_details['content_filter_result'] = (
                        error_data.error.innererror.content_filter_result.model_dump(exclude_none=True)
                    )

                return ModelResponse(
                    parts=[],  # Empty parts to trigger content filter error in agent graph
                    model_name=model_name,
                    timestamp=_utils.now_utc(),
                    provider_name=system,
                    finish_reason='content_filter',
                    provider_details=provider_details,
                )
        except ValidationError:
            pass
    return None
