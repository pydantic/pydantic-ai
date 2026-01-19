# Adding Batch Support to a Model Provider

This guide explains how to implement batch processing support for a new model provider in PydanticAI.

## Prerequisites

Before implementing batch support, verify that:

1. **The provider has a batch API** - Not all LLM providers support batch processing
2. **You understand the provider's batch lifecycle** - Status codes, result retrieval, error handling
3. **You've read the existing implementations** - See `openai.py`, `anthropic.py`, and `google.py` for reference

## Overview

Batch processing allows submitting multiple requests at once for asynchronous processing, typically at reduced cost (50% for OpenAI/Anthropic/Google). To add batch support, you need to:

1. Create a provider-specific `Batch` subclass (e.g., `ProviderBatch`)
2. Implement the four batch methods in your `Model` class
3. Add tests with appropriate coverage

## Step 1: Create a Batch Subclass

Create a dataclass that extends the base `Batch` class with provider-specific fields:

```python {test="skip" lint="skip"}
from dataclasses import dataclass
from pydantic_ai.models import Batch


@dataclass
class ProviderBatch(Batch):
    """Provider-specific batch job information."""

    # Add provider-specific fields
    job_name: str | None = None
    """Full resource name of the batch job."""

    raw_status: str | None = None
    """Raw provider status value."""

    # Add any other provider-specific metadata
```

## Step 2: Define Status Mapping

Create a mapping from provider-specific status strings to the normalized `BatchStatus` enum:

```python {test="skip" lint="skip"}
from pydantic_ai.models import BatchStatus

_PROVIDER_STATUS_MAP: dict[str, BatchStatus] = {
    'pending': BatchStatus.PENDING,
    'validating': BatchStatus.VALIDATING,
    'running': BatchStatus.IN_PROGRESS,
    'finalizing': BatchStatus.FINALIZING,
    'completed': BatchStatus.COMPLETED,
    'failed': BatchStatus.FAILED,
    'cancelled': BatchStatus.CANCELLED,
    'cancelling': BatchStatus.CANCELLING,
    'expired': BatchStatus.EXPIRED,
}
```

## Step 3: Implement the Four Batch Methods

### 3.1 `batch_create`

Submit a batch of requests to the provider:

```python {test="skip" lint="skip"}
from collections.abc import Sequence
from pydantic_ai.messages import ModelMessage
from pydantic_ai.models import ModelRequestParameters, check_allow_model_requests


async def batch_create(
    self,
    requests: Sequence[tuple[str, list[ModelMessage], ModelRequestParameters]],
    model_settings: ModelSettings | None = None,
) -> ProviderBatch:
    """Submit a batch of requests for async processing."""
    check_allow_model_requests()

    if len(requests) < 2:
        raise ValueError('Batch must contain at least 2 requests')

    # Build provider-specific request format
    batch_requests = []
    for custom_id, messages, params in requests:
        # Use existing request-building logic
        request_params = self._build_request_params(messages, params, model_settings)
        batch_requests.append({
            'custom_id': custom_id,
            'params': request_params,
        })

    # Submit to provider API with error handling
    try:
        response = await self.client.batches.create(requests=batch_requests)
    except ProviderAPIError as e:
        # Handle errors using standard patterns
        raise ModelHTTPError(...) from e

    return self._parse_batch_response(response)
```

### 3.2 `batch_status`

Check the current status of a batch job:

```python {test="skip" lint="skip"}
async def batch_status(self, batch: Batch) -> ProviderBatch:
    """Get current status of a batch job."""
    check_allow_model_requests()

    try:
        response = await self.client.batches.retrieve(batch.id)
    except ProviderAPIError as e:
        raise ModelHTTPError(...) from e

    return self._parse_batch_response(response)
```

### 3.3 `batch_results`

Retrieve results from a completed batch:

```python {test="skip" lint="skip"}
from pydantic_ai.models import BatchResult
from pydantic_ai.models._batch_utils import BatchResultBuilder, extract_batch_error


async def batch_results(self, batch: Batch) -> list[BatchResult]:
    """Retrieve results from a completed batch."""
    check_allow_model_requests()

    if not batch.is_complete:
        raise ValueError(f'Batch {batch.id} is not complete (status: {batch.status})')

    if not isinstance(batch, ProviderBatch):
        raise TypeError(f'Expected ProviderBatch, got {type(batch).__name__}')

    builder = BatchResultBuilder()

    try:
        result_stream = await self.client.batches.results(batch.id)
        async for entry in result_stream:
            custom_id = entry.custom_id

            if entry.result.type == 'succeeded':
                # Use existing response processing
                model_response = self._process_response(entry.result.response)
                builder.add_success(custom_id, model_response)
            elif entry.result.type == 'errored':
                builder.add_error_from_dict(custom_id, entry.result.error)
            else:
                builder.add_error(
                    custom_id,
                    BatchError(code=entry.result.type, message=f'Request {entry.result.type}'),
                )
    except ProviderAPIError as e:
        raise ModelHTTPError(...) from e

    return builder.results
```

### 3.4 `batch_cancel`

Cancel a pending or in-progress batch:

```python {test="skip" lint="skip"}
async def batch_cancel(self, batch: Batch) -> ProviderBatch:
    """Cancel a batch job."""
    check_allow_model_requests()

    try:
        response = await self.client.batches.cancel(batch.id)
    except ProviderAPIError as e:
        raise ModelHTTPError(...) from e

    return self._parse_batch_response(response)
```

## Step 4: Implement the Response Parser

Use the shared utilities for datetime parsing:

```python {test="skip" lint="skip"}
from pydantic_ai.models._batch_utils import parse_batch_datetime
from pydantic_ai import _utils


def _parse_batch_response(self, response: Any) -> ProviderBatch:
    """Convert provider response to ProviderBatch object."""
    status = _PROVIDER_STATUS_MAP.get(response.status, BatchStatus.PENDING)

    # Use parse_batch_datetime for consistent datetime handling
    created_at = parse_batch_datetime(response.created_at) or _utils.now_utc()
    completed_at = parse_batch_datetime(response.completed_at)

    # Extract counts
    request_count = getattr(response.counts, 'total', 0) or 0
    completed_count = getattr(response.counts, 'completed', 0) or 0
    failed_count = getattr(response.counts, 'failed', 0) or 0

    return ProviderBatch(
        id=response.id,
        status=status,
        created_at=created_at,
        completed_at=completed_at,
        request_count=request_count,
        completed_count=completed_count,
        failed_count=failed_count,
        # Provider-specific fields
        job_name=response.name,
        raw_status=response.status,
    )
```

## Step 5: Use Shared Utilities

The `_batch_utils` module provides several utilities to reduce code duplication:

### `parse_batch_datetime`

Handles multiple datetime formats (Unix timestamps, ISO strings, datetime objects):

```python {test="skip" lint="skip"}
from pydantic_ai.models._batch_utils import parse_batch_datetime

# Handles all these formats automatically:
parse_batch_datetime(1705579200)           # Unix timestamp
parse_batch_datetime('2024-01-18T12:00:00Z')  # ISO string
parse_batch_datetime(datetime.now())       # datetime object
parse_batch_datetime(None)                 # Returns None
```

### `extract_batch_error`

Extracts error information from various provider formats:

```python {test="skip" lint="skip"}
from pydantic_ai.models._batch_utils import extract_batch_error

# Handles nested and flat error structures:
extract_batch_error({'code': 'rate_limit', 'message': 'Too many requests'})
extract_batch_error({'error': {'type': 'validation', 'detail': 'Invalid input'}})
extract_batch_error('Plain error message')
```

### `BatchResultBuilder`

Tracks processed IDs to prevent duplicates (important when results come from multiple sources):

```python {test="skip" lint="skip"}
from pydantic_ai.models._batch_utils import BatchResultBuilder

builder = BatchResultBuilder()

# Add results - duplicates are automatically skipped
builder.add_success('req-1', response)
builder.add_error('req-2', BatchError(code='timeout', message='Timed out'))
builder.add_error_from_dict('req-3', {'error': {'code': 'rate_limit'}})

# Access final results
return builder.results
```

### `validate_batch_complete`

Validates batch completion before result retrieval:

```python {test="skip" lint="skip"}
from pydantic_ai.models._batch_utils import validate_batch_complete

# Raises ValueError if batch not complete
validate_batch_complete(batch, 'retrieve results')
```

## Step 6: Add Tests

Create tests in `tests/models/test_<provider>_batch.py`:

```python {test="skip" lint="skip"}
import pytest
from pydantic_ai.models.<provider> import ProviderModel, ProviderBatch
from pydantic_ai.models import BatchStatus, supports_batch


class TestProviderBatch:
    def test_batch_create_minimum_requests(self, model):
        """Test that batch requires at least 2 requests."""
        with pytest.raises(ValueError, match='at least 2 requests'):
            await model.batch_create([single_request])

    def test_batch_status_mapping(self):
        """Test that provider statuses map correctly."""
        # Test each status mapping

    def test_batch_results_requires_complete(self, model, incomplete_batch):
        """Test that results require complete batch."""
        with pytest.raises(ValueError, match='not complete'):
            await model.batch_results(incomplete_batch)

    def test_supports_batch(self, model):
        """Test that supports_batch returns True for this model."""
        assert supports_batch(model) is True
```

## Verification Checklist

Before submitting your PR, verify:

- [ ] All four batch methods are implemented
- [ ] `supports_batch(model)` returns `True` for your model
- [ ] Provider-specific `Batch` subclass is created
- [ ] Status mapping covers all provider statuses
- [ ] Shared utilities are used where applicable
- [ ] Error handling follows existing patterns
- [ ] Tests cover all methods and edge cases
- [ ] 100% test coverage is maintained

## Reference Implementations

- **OpenAI**: `pydantic_ai_slim/pydantic_ai/models/openai.py` - File-based batch with separate output/error files
- **Anthropic**: `pydantic_ai_slim/pydantic_ai/models/anthropic.py` - Streaming results with detailed status tracking
- **Google**: `pydantic_ai_slim/pydantic_ai/models/google.py` - Inline results with job state management

## See Also

- [`BatchCapable` protocol](../api/models.md#pydantic_ai.models.BatchCapable)
- [`supports_batch` function](../api/models.md#pydantic_ai.models.supports_batch)
- [Batch Processing User Guide](../batch-processing.md)
