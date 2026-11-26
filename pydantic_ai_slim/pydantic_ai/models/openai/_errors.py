"""API error mapping, including Azure content filter errors."""

from __future__ import annotations as _annotations

from collections.abc import Generator, Mapping
from contextlib import contextmanager
from typing import Any

import httpx2
from openai import APIConnectionError, APIError, APIStatusError, APITimeoutError, AsyncAzureOpenAI, AsyncOpenAI
from openai.types import responses
from pydantic import BaseModel, ValidationError

from ... import ModelAPIError, ModelHTTPError, _model_errors, _utils
from ..._utils import is_str_dict as _is_str_dict
from ...exceptions import ModelConnectionError, ModelContextWindowExceededError, ModelRequestRejectedError
from ...messages import ModelResponse
from .. import (
    _suggest_known_model_id_from_provider_error,  # pyright: ignore[reportPrivateUsage]
)
from .._transport_errors import transport_error_message


@contextmanager
def map_api_errors(model_name: str, model_id_namespace: str = 'openai') -> Generator[None]:
    try:
        yield
    except APIStatusError as e:
        if e.status_code >= 400:
            raise map_status_error(e, model_name, model_id_namespace) from e
        raise ModelAPIError(model_name=model_name, message=e.message) from e  # pragma: lax no cover
    except APIConnectionError as e:
        raise map_connection_error(e, model_name) from e
    except APIError as e:
        # The SDK raises the base `APIError` for an error object inside a stream, after the HTTP 200 has already
        # been received; it gets the status the same error has before a stream opens.
        raise _model_errors.stream_error(model_name, e.message, e.body) from e
    except httpx2.HTTPStatusError as e:
        raise map_transport_status_error(e, model_name, model_id_namespace) from e
    except httpx2.TransportError as e:
        # For a streamed request, `openai` reads the body of an error status outside its own transport error handling,
        # so a connection that breaks off while sending that body surfaces as the raw `httpx2` error.
        timeout = isinstance(e, httpx2.TimeoutException)
        raise _model_errors.connection_error(model_name, transport_error_message(e), e, timeout=timeout) from e


def map_status_error(e: APIStatusError, model_name: str, model_id_namespace: str) -> ModelHTTPError:
    return status_error(
        model_name,
        model_id_namespace,
        status_code=e.status_code,
        body=e.body,
        code=e.code,
        error_type=e.type,
        message=e.message,
        headers=dict(e.response.headers),
    )


def map_transport_status_error(e: httpx2.HTTPStatusError, model_name: str, model_id_namespace: str) -> ModelHTTPError:
    """The error for a status a retrying transport gave up on, which `openai` lets through unwrapped."""
    body, error, text = _model_errors.transport_error_body(e.response)
    code, error_type = _model_errors.openai_error_fields(error)
    return status_error(
        model_name,
        model_id_namespace,
        status_code=e.response.status_code,
        body=body,
        code=code,
        error_type=error_type,
        message=_model_errors.openai_error_message(error, text),
        headers=dict(e.response.headers),
        error_object=error,
    )


def status_error(
    model_name: str,
    model_id_namespace: str,
    *,
    status_code: int,
    body: object | None,
    code: str | None,
    error_type: str | None,
    message: str | None,
    headers: Mapping[str, str] | None = None,
    error_object: Mapping[str, object] | None = None,
) -> ModelHTTPError:
    """The error for an HTTP error status, which a batch also reports for each request that failed.

    `body` is the error object, as `openai` unwraps it from the response body, unless `error_object` is given: the body
    of a status a retrying transport gave up on is the whole response body, which wraps it.
    """
    suggested_model_id = None
    if code == 'model_not_found':
        suggested_model_id = _suggest_known_model_id_from_provider_error(model_id_namespace, model_name)
    category = _model_errors.openai_compatible_category(status_code, code, error_type, message)
    reported = body if error_object is None else error_object
    metadata = reported.get('metadata') if _utils.is_str_dict(reported) else None
    if model_id_namespace == 'zai':
        # Z.ai's numeric codes mean something only from Z.ai, so other OpenAI-compatible APIs don't get this mapping.
        category = _model_errors.zai_code_category(code) or category
    elif model_id_namespace == 'moonshotai':
        # Moonshot sends an exhausted balance as a 429 too; only its error type tells it from a rate limit.
        category = _model_errors.moonshot_type_category(error_type) or category
    elif (
        model_id_namespace == 'openrouter'
        and category is ModelRequestRejectedError
        and metadata_raw_reports_context_window(metadata)
    ):
        # OpenRouter forwards the upstream provider's overflow behind a generic rejection; a specific category wins.
        category = ModelContextWindowExceededError
    if status_code == 403 and flags_input(metadata):
        # OpenRouter sends a 403 for flagged input too: a rejected request, not a credentials problem.
        category = ModelRequestRejectedError
    return ModelHTTPError.for_category(
        category,
        status_code=status_code,
        model_name=model_name,
        body=body,
        headers=headers,
        suggested_model_id=suggested_model_id,
        provider_error_code=code,
        provider_error_type=error_type,
    )


def flags_input(metadata: object) -> bool:
    """Whether an error's `metadata` holds a moderation result, as OpenRouter's 403 for flagged input does.

    See https://openrouter.ai/docs/api-reference/errors#moderation-errors.
    """
    return _utils.is_str_dict(metadata) and ('flagged_input' in metadata or 'reasons' in metadata)


def metadata_raw_reports_context_window(metadata: object) -> bool:
    """Whether the upstream error OpenRouter forwards in an error's `metadata.raw` says the context window was exceeded.

    OpenRouter puts it there behind a generic message like `Provider returned error`, as plain text or as a JSON
    `{"error": {...}}` string. Only its message is read: a nested code or type, like a generic `400`, never counts.
    """
    raw = metadata.get('raw') if _utils.is_str_dict(metadata) else None
    if not isinstance(raw, str):
        return False
    parsed, _, fields = _model_errors.parse_error_body(raw)
    if parsed is None:
        message = raw
    elif _utils.is_str_dict(parsed) and 'error' in parsed:
        message = fields.get('message')
    else:
        return False
    return _model_errors.openai_compatible_category(None, None, None, message) is ModelContextWindowExceededError


def map_connection_error(e: APIConnectionError, model_name: str) -> ModelConnectionError:
    return _model_errors.connection_error(model_name, e.message, e, timeout=isinstance(e, APITimeoutError))


def response_error_type(error: responses.ResponseError) -> str | None:
    """The provider error `type`, like Azure's `too_many_requests`.

    `openai` declares the field from 3.32.0; before that it's an extra, which `getattr` also reads.
    """
    error_type: object = getattr(error, 'type', None)
    return error_type if isinstance(error_type, str) else None


def response_error(
    model_name: str, code: str | None, message: str, *, error_type: str | None = None, in_stream: bool
) -> ModelAPIError:
    """Build the error for a Responses API failure reported in a 200 body or stream.

    It gets the same category either way. In a stream, it also gets the status the same error has before the stream
    opens where that's clear, like any other in-stream error. A failed response body, e.g. a background response
    retrieved later, has no status to report.
    """
    if in_stream:
        body: dict[str, str | None] = {'code': code, 'message': message}
        if error_type is not None:
            body['type'] = error_type
        error = _model_errors.stream_error(model_name, message, body)
        if isinstance(error, ModelHTTPError):
            return error
    category = _model_errors.openai_compatible_category(
        _model_errors.openai_compatible_status(code, error_type), code, error_type, message
    )
    return (category or ModelAPIError)(
        model_name=model_name,
        message=f'{code}: {message}' if code else message,
        provider_error_code=code,
        provider_error_type=error_type,
        in_stream=in_stream,
    )


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
    code: str | None = None
    content_filter_result: _AzureContentFilterResult | None = None


class _AzureContentFilter(BaseModel):
    """An entry of the top-level `content_filters` list the Responses API sends for a blocked prompt."""

    blocked: bool | None = None
    source_type: str | None = None
    content_filter_results: _AzureContentFilterResult | None = None


class _AzureError(BaseModel):
    code: str
    message: str | None = None
    innererror: _AzureInnerError | None = None
    content_filters: list[_AzureContentFilter] | None = None


def is_azure(client: AsyncOpenAI, system: str) -> bool:
    return system == 'azure' or isinstance(client, AsyncAzureOpenAI)


def check_azure_content_filter(
    e: APIStatusError, client: AsyncOpenAI, system: str, model_name: str
) -> ModelResponse | None:
    """Check if the error is an Azure content filter error."""
    # Assign to Any to avoid 'dict[Unknown, Unknown]' inference in strict mode
    body_any: Any = e.body

    if not (is_azure(client, system) and e.status_code == 400 and _is_str_dict(body_any)):
        return None
    try:
        error_data = _AzureError.model_validate(body_any.get('error', body_any))

        if error_data.code != 'content_filter':
            return None
        provider_details: dict[str, Any] = {'finish_reason': 'content_filter'}

        if error_data.innererror and error_data.innererror.content_filter_result:
            provider_details['content_filter_result'] = error_data.innererror.content_filter_result.model_dump(
                exclude_none=True
            )
        if error_data.content_filters:
            provider_details['content_filters'] = [
                content_filter.model_dump(exclude_none=True) for content_filter in error_data.content_filters
            ]

        return ModelResponse(
            parts=[],  # Empty parts to trigger content filter error in agent graph
            model_name=model_name,
            timestamp=_utils.now_utc(),
            provider_name=system,
            finish_reason='content_filter',
            provider_details=provider_details,
        )
    except ValidationError:
        return None
