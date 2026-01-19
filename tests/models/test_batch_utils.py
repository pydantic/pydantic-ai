"""Tests for batch processing utilities."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pytest

from pydantic_ai.messages import ModelResponse, TextPart
from pydantic_ai.models import Batch, BatchCapable, BatchError, BatchResult, BatchStatus, supports_batch
from pydantic_ai.models._batch_utils import (
    BatchResultBuilder,
    extract_batch_error,
    parse_batch_datetime,
    validate_batch_complete,
)
from pydantic_ai.settings import ModelSettings


class TestExtractBatchError:
    """Tests for extract_batch_error function."""

    def test_simple_dict_with_code_and_message(self):
        error_data = {'code': 'rate_limit', 'message': 'Too many requests'}
        result = extract_batch_error(error_data)
        assert result.code == 'rate_limit'
        assert result.message == 'Too many requests'

    def test_nested_error_structure(self):
        error_data = {'error': {'code': 'validation_error', 'message': 'Invalid input'}}
        result = extract_batch_error(error_data)
        assert result.code == 'validation_error'
        assert result.message == 'Invalid input'

    def test_type_field_for_code(self):
        """Anthropic uses 'type' instead of 'code'."""
        error_data = {'type': 'invalid_request', 'message': 'Bad request format'}
        result = extract_batch_error(error_data)
        assert result.code == 'invalid_request'
        assert result.message == 'Bad request format'

    def test_detail_field_for_message(self):
        error_data = {'code': 'server_error', 'detail': 'Internal server error occurred'}
        result = extract_batch_error(error_data)
        assert result.code == 'server_error'
        assert result.message == 'Internal server error occurred'

    def test_nested_with_type_and_detail(self):
        error_data = {'error': {'type': 'validation_error', 'detail': 'Missing required field'}}
        result = extract_batch_error(error_data)
        assert result.code == 'validation_error'
        assert result.message == 'Missing required field'

    def test_string_error_in_nested_structure(self):
        error_data = {'error': 'Something went wrong'}
        result = extract_batch_error(error_data)
        assert result.code == 'unknown'
        assert result.message == 'Something went wrong'

    def test_non_dict_input(self):
        result = extract_batch_error('Plain error message')
        assert result.code == 'unknown'
        assert result.message == 'Plain error message'

    def test_none_input(self):
        result = extract_batch_error(None)
        assert result.code == 'unknown'
        assert result.message == 'Unknown error'

    def test_empty_dict(self):
        result = extract_batch_error({})
        assert result.code == 'unknown'
        assert result.message == 'Unknown error'

    def test_missing_code_field(self):
        error_data = {'message': 'Error without code'}
        result = extract_batch_error(error_data)
        assert result.code == 'unknown'
        assert result.message == 'Error without code'

    def test_missing_message_field(self):
        error_data = {'code': 'some_error'}
        result = extract_batch_error(error_data)
        assert result.code == 'some_error'
        assert result.message == 'Unknown error'


class TestParseBatchDatetime:
    """Tests for parse_batch_datetime function."""

    def test_none_returns_none(self):
        assert parse_batch_datetime(None) is None

    def test_datetime_object_returned_as_is(self):
        dt = datetime(2024, 1, 18, 12, 0, 0, tzinfo=timezone.utc)
        result = parse_batch_datetime(dt)
        assert result is dt

    def test_unix_timestamp_int(self):
        # 2024-01-18 12:00:00 UTC
        timestamp = 1705579200
        result = parse_batch_datetime(timestamp)
        assert result is not None
        assert result.year == 2024
        assert result.month == 1
        assert result.day == 18
        assert result.tzinfo == timezone.utc

    def test_unix_timestamp_float(self):
        timestamp = 1705579200.5
        result = parse_batch_datetime(timestamp)
        assert result is not None
        assert result.year == 2024
        assert result.tzinfo == timezone.utc

    def test_iso_string_with_z_suffix(self):
        result = parse_batch_datetime('2024-01-18T12:00:00Z')
        assert result is not None
        assert result.year == 2024
        assert result.month == 1
        assert result.day == 18
        assert result.hour == 12
        assert result.tzinfo == timezone.utc

    def test_iso_string_with_offset(self):
        result = parse_batch_datetime('2024-01-18T12:00:00+00:00')
        assert result is not None
        assert result.year == 2024
        assert result.tzinfo == timezone.utc

    def test_iso_string_without_timezone(self):
        result = parse_batch_datetime('2024-01-18T12:00:00')
        assert result is not None
        assert result.year == 2024

    def test_invalid_string_returns_none(self):
        result = parse_batch_datetime('not a date')
        assert result is None

    def test_unsupported_type_returns_none(self):
        result = parse_batch_datetime([2024, 1, 18])
        assert result is None


class TestValidateBatchComplete:
    """Tests for validate_batch_complete function."""

    def test_complete_batch_passes(self):
        batch = Batch(
            id='batch_123',
            status=BatchStatus.COMPLETED,
            created_at=datetime.now(tz=timezone.utc),
        )
        # Should not raise
        validate_batch_complete(batch, 'retrieve results')

    def test_failed_batch_passes(self):
        """Failed is also a terminal state."""
        batch = Batch(
            id='batch_123',
            status=BatchStatus.FAILED,
            created_at=datetime.now(tz=timezone.utc),
        )
        # Should not raise
        validate_batch_complete(batch, 'retrieve results')

    def test_in_progress_batch_raises(self):
        batch = Batch(
            id='batch_123',
            status=BatchStatus.IN_PROGRESS,
            created_at=datetime.now(tz=timezone.utc),
        )
        with pytest.raises(ValueError, match='Cannot retrieve results: batch batch_123 is not complete'):
            validate_batch_complete(batch, 'retrieve results')

    def test_pending_batch_raises(self):
        batch = Batch(
            id='batch_456',
            status=BatchStatus.PENDING,
            created_at=datetime.now(tz=timezone.utc),
        )
        with pytest.raises(ValueError, match=r'batch batch_456 is not complete.*(?:pending|PENDING)'):
            validate_batch_complete(batch, 'download results')

    def test_custom_operation_in_error_message(self):
        batch = Batch(
            id='batch_789',
            status=BatchStatus.VALIDATING,
            created_at=datetime.now(tz=timezone.utc),
        )
        with pytest.raises(ValueError, match='Cannot process files:'):
            validate_batch_complete(batch, 'process files')


class TestBatchResultBuilder:
    """Tests for BatchResultBuilder class."""

    def _make_response(self, text: str = 'test') -> ModelResponse:
        return ModelResponse(parts=[TextPart(content=text)])

    def test_add_success(self):
        builder = BatchResultBuilder()
        response = self._make_response('Hello')
        result = builder.add_success('req-1', response)

        assert result is True
        assert len(builder.results) == 1
        assert builder.results[0].custom_id == 'req-1'
        assert builder.results[0].response is response
        assert builder.results[0].error is None

    def test_add_error(self):
        builder = BatchResultBuilder()
        error = BatchError(code='timeout', message='Request timed out')
        result = builder.add_error('req-1', error)

        assert result is True
        assert len(builder.results) == 1
        assert builder.results[0].custom_id == 'req-1'
        assert builder.results[0].response is None
        assert builder.results[0].error is error

    def test_add_error_from_dict(self):
        builder = BatchResultBuilder()
        error_data = {'code': 'rate_limit', 'message': 'Too many requests'}
        result = builder.add_error_from_dict('req-1', error_data)

        assert result is True
        assert len(builder.results) == 1
        assert builder.results[0].error is not None
        assert builder.results[0].error.code == 'rate_limit'
        assert builder.results[0].error.message == 'Too many requests'

    def test_skip_duplicate_success(self):
        builder = BatchResultBuilder()
        response1 = self._make_response('First')
        response2 = self._make_response('Second')

        builder.add_success('req-1', response1)
        result = builder.add_success('req-1', response2, skip_duplicate=True)

        assert result is False
        assert len(builder.results) == 1
        assert builder.results[0].response is response1

    def test_skip_duplicate_error(self):
        builder = BatchResultBuilder()
        error1 = BatchError(code='error1', message='First error')
        error2 = BatchError(code='error2', message='Second error')

        builder.add_error('req-1', error1)
        result = builder.add_error('req-1', error2, skip_duplicate=True)

        assert result is False
        assert len(builder.results) == 1
        assert builder.results[0].error is error1

    def test_skip_duplicate_mixed(self):
        """Error added first, then success skipped."""
        builder = BatchResultBuilder()
        error = BatchError(code='error', message='Error message')
        response = self._make_response('Response')

        builder.add_error('req-1', error)
        result = builder.add_success('req-1', response, skip_duplicate=True)

        assert result is False
        assert len(builder.results) == 1
        assert builder.results[0].error is error

    def test_allow_duplicate_if_disabled(self):
        builder = BatchResultBuilder()
        response1 = self._make_response('First')
        response2 = self._make_response('Second')

        builder.add_success('req-1', response1)
        result = builder.add_success('req-1', response2, skip_duplicate=False)

        assert result is True
        assert len(builder.results) == 2

    def test_multiple_different_ids(self):
        builder = BatchResultBuilder()

        builder.add_success('req-1', self._make_response('Response 1'))
        builder.add_error('req-2', BatchError(code='error', message='Error'))
        builder.add_success('req-3', self._make_response('Response 3'))

        assert len(builder.results) == 3
        assert len(builder.processed_ids) == 3
        assert builder.processed_ids == {'req-1', 'req-2', 'req-3'}

    def test_empty_builder(self):
        builder = BatchResultBuilder()
        assert len(builder.results) == 0
        assert len(builder.processed_ids) == 0


class TestSupportsBatch:
    """Tests for supports_batch function."""

    def test_base_model_does_not_support_batch(self):
        """The base Model class should not support batch (stubs raise NotImplementedError)."""
        from pydantic_ai.models.test import TestModel

        model = TestModel()
        assert supports_batch(model) is False

    def test_batch_capable_model(self):
        """Test that the BatchCapable protocol works."""
        # Create a minimal implementation for testing
        from collections.abc import Sequence

        from pydantic_ai.messages import ModelMessage
        from pydantic_ai.models import Model, ModelRequestParameters

        class BatchCapableModel(Model):
            """A mock model that implements batch methods."""

            async def batch_create(
                self,
                requests: Sequence[tuple[str, list[ModelMessage], ModelRequestParameters]],
                model_settings: ModelSettings | None = None,
            ) -> Batch:
                return Batch(id='test', status=BatchStatus.PENDING, created_at=datetime.now(tz=timezone.utc))

            async def batch_status(self, batch: Batch) -> Batch:
                return batch

            async def batch_results(self, batch: Batch) -> list[BatchResult]:
                return []

            async def batch_cancel(self, batch: Batch) -> Batch:
                return batch

            # Required abstract methods
            async def request(self, *args: Any, **kwargs: Any) -> Any:
                raise NotImplementedError

            @property
            def model_name(self) -> str:
                return 'test'

            @property
            def system(self) -> str:
                return 'test'

        model = BatchCapableModel()
        assert supports_batch(model) is True

        # Check protocol conformance
        assert isinstance(model, BatchCapable)


class TestBatchCapableProtocol:
    """Tests for BatchCapable protocol."""

    def test_protocol_is_runtime_checkable(self):
        """Verify that BatchCapable can be used with isinstance."""
        from pydantic_ai.models.test import TestModel

        model = TestModel()
        # TestModel doesn't implement batch methods, so it shouldn't match
        # Note: runtime_checkable only checks method names exist, not implementations
        # The supports_batch function does the deeper check
        assert not supports_batch(model)
