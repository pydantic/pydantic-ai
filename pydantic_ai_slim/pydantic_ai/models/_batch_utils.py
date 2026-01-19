"""Utilities for implementing batch processing in model providers.

This module provides shared utilities that reduce code duplication across
batch processing implementations. See docs/adding-batch-support.md for
the complete guide on implementing batch support for new providers.

The key components are:

- `extract_batch_error`: Parses error data from various provider formats
- `parse_batch_datetime`: Handles multiple datetime formats (epoch, ISO, datetime)
- `validate_batch_complete`: Ensures batch is complete before result retrieval
- `BatchResultBuilder`: Tracks processed IDs and builds result lists

Example usage in a provider:

    from ._batch_utils import extract_batch_error, parse_batch_datetime, BatchResultBuilder

    def _parse_batch_response(self, response) -> ProviderBatch:
        return ProviderBatch(
            id=response.id,
            status=self._map_status(response.status),
            created_at=parse_batch_datetime(response.created_at) or now_utc(),
            completed_at=parse_batch_datetime(response.completed_at),
            ...
        )

    async def batch_results(self, batch) -> list[BatchResult]:
        validate_batch_complete(batch, 'retrieve results')

        builder = BatchResultBuilder()
        for item in response.results:
            if item.response:
                builder.add_success(item.custom_id, self._process_response(item.response))
            elif item.error:
                builder.add_error_from_dict(item.custom_id, item.error)
        return builder.results
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, cast

from . import BatchError, BatchResult

if TYPE_CHECKING:
    from ..messages import ModelResponse
    from . import Batch


def extract_batch_error(error_data: dict[str, Any] | Any) -> BatchError:
    """Extract error code and message from provider error response.

    Handles common formats from OpenAI, Anthropic, Google, etc.
    Works with nested error structures like `{"error": {"code": ..., "message": ...}}`
    as well as flat structures.

    Args:
        error_data: Error data from provider, typically a dict with error details.
            Can also be any object (will be stringified).

    Returns:
        BatchError with normalized code and message.

    Examples:
        >>> extract_batch_error({'code': 'rate_limit', 'message': 'Too many requests'})
        BatchError(code='rate_limit', message='Too many requests')

        >>> extract_batch_error({'error': {'type': 'validation_error', 'detail': 'Invalid input'}})
        BatchError(code='validation_error', message='Invalid input')

        >>> extract_batch_error({'error': 'Something went wrong'})
        BatchError(code='unknown', message='Something went wrong')
    """
    if not isinstance(error_data, dict):
        return BatchError(code='unknown', message=str(error_data) if error_data else 'Unknown error')

    # Cast to help pyright understand the type after isinstance check
    error_dict = cast(dict[str, Any], error_data)

    # Handle nested error structures: {"error": {...}}
    error = error_dict.get('error', error_dict)
    if isinstance(error, dict):
        # Cast again after isinstance check
        nested_error = cast(dict[str, Any], error)
        # Try common field names for code: code, type
        code = str(nested_error.get('code') or nested_error.get('type') or 'unknown')
        # Try common field names for message: message, detail
        message = str(nested_error.get('message') or nested_error.get('detail') or 'Unknown error')
        return BatchError(code=code, message=message)

    # error is a string or other primitive
    return BatchError(code='unknown', message=str(error) if error else 'Unknown error')


def parse_batch_datetime(value: Any) -> datetime | None:
    """Parse datetime from various provider formats.

    Handles:
    - Unix timestamps (int/float seconds since epoch)
    - ISO 8601 strings (with or without timezone)
    - datetime objects (returned as-is)
    - None (returned as-is)

    Args:
        value: Datetime value in various formats.

    Returns:
        Parsed datetime (timezone-aware, UTC) or None if parsing fails.

    Examples:
        >>> parse_batch_datetime(1705555200)  # Unix timestamp
        datetime.datetime(2024, 1, 18, 4, 0, tzinfo=datetime.timezone.utc)

        >>> parse_batch_datetime('2024-01-18T04:00:00Z')  # ISO string
        datetime.datetime(2024, 1, 18, 4, 0, tzinfo=datetime.timezone.utc)

        >>> parse_batch_datetime(None)
        None
    """
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(value, tz=timezone.utc)
    if isinstance(value, str):
        try:
            # Handle 'Z' suffix and various ISO formats
            return datetime.fromisoformat(value.replace('Z', '+00:00'))
        except ValueError:
            return None
    return None


def validate_batch_complete(batch: Batch, operation: str = 'retrieve results') -> None:
    """Validate that a batch is complete before operating on it.

    Args:
        batch: Batch object to validate.
        operation: Description of the operation being attempted (for error message).

    Raises:
        ValueError: If batch is not complete (status not in terminal state).

    Example:
        >>> validate_batch_complete(batch, 'retrieve results')
        ValueError: Cannot retrieve results: batch batch_123 is not complete (status: IN_PROGRESS)
    """
    if not batch.is_complete:
        raise ValueError(f'Cannot {operation}: batch {batch.id} is not complete (status: {batch.status})')


@dataclass
class BatchResultBuilder:
    """Helper for building BatchResult lists from provider responses.

    Tracks processed custom_ids to avoid duplicates. This is particularly
    important for providers like OpenAI that may return results in separate
    output and error files, where the same custom_id could appear in both.

    Attributes:
        results: List of BatchResult objects built so far.
        processed_ids: Set of custom_ids that have been processed.

    Example:
        >>> builder = BatchResultBuilder()
        >>> builder.add_success('req-1', model_response)
        True
        >>> builder.add_error('req-2', BatchError(code='timeout', message='Request timed out'))
        True
        >>> builder.add_success('req-1', another_response)  # Duplicate, skipped
        False
        >>> len(builder.results)
        2
    """

    results: list[BatchResult] = field(default_factory=list)
    processed_ids: set[str] = field(default_factory=set)

    def add_success(
        self,
        custom_id: str,
        response: ModelResponse,
        *,
        skip_duplicate: bool = True,
    ) -> bool:
        """Add a successful result.

        Args:
            custom_id: Unique identifier from the original request.
            response: The ModelResponse for this request.
            skip_duplicate: If True, skip if custom_id was already processed.

        Returns:
            True if added, False if skipped (duplicate).
        """
        if skip_duplicate and custom_id in self.processed_ids:
            return False
        self.processed_ids.add(custom_id)
        self.results.append(BatchResult(custom_id=custom_id, response=response))
        return True

    def add_error(
        self,
        custom_id: str,
        error: BatchError,
        *,
        skip_duplicate: bool = True,
    ) -> bool:
        """Add an error result.

        Args:
            custom_id: Unique identifier from the original request.
            error: BatchError with code and message.
            skip_duplicate: If True, skip if custom_id was already processed.

        Returns:
            True if added, False if skipped (duplicate).
        """
        if skip_duplicate and custom_id in self.processed_ids:
            return False
        self.processed_ids.add(custom_id)
        self.results.append(BatchResult(custom_id=custom_id, error=error))
        return True

    def add_error_from_dict(
        self,
        custom_id: str,
        error_data: dict[str, Any] | Any,
        *,
        skip_duplicate: bool = True,
    ) -> bool:
        """Add an error result by extracting from API error dict.

        Convenience method that combines extract_batch_error with add_error.

        Args:
            custom_id: Unique identifier from the original request.
            error_data: Raw error data from the provider API.
            skip_duplicate: If True, skip if custom_id was already processed.

        Returns:
            True if added, False if skipped (duplicate).
        """
        return self.add_error(custom_id, extract_batch_error(error_data), skip_duplicate=skip_duplicate)
