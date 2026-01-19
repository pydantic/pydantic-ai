"""Tests for OpenAI batch processing functionality."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, cast
from unittest.mock import MagicMock

import pytest

from pydantic_ai import (
    Batch,
    BatchError,
    BatchResult,
    BatchStatus,
    ModelMessage,
    ModelRequest,
    ModelRequestParameters,
    ModelResponse,
    TextPart,
)
from pydantic_ai.direct import (
    batch_cancel,
    batch_cancel_sync,
    batch_create,
    batch_create_sync,
    batch_results,
    batch_status,
    batch_status_sync,
)
from pydantic_ai.usage import RequestUsage

from ..conftest import try_import

with try_import() as imports_successful:
    from openai import APIConnectionError, APIStatusError, AsyncOpenAI
    from openai.types import Batch as OpenAIBatchResponse
    from openai.types.batch_request_counts import BatchRequestCounts

    from pydantic_ai import ModelAPIError, ModelHTTPError
    from pydantic_ai.models.openai import OpenAIBatch, OpenAIChatModel
    from pydantic_ai.providers.openai import OpenAIProvider

pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='openai not installed'),
    pytest.mark.anyio,
]


# --- Unit Tests for Batch Abstractions ---


class TestBatchStatus:
    """Tests for BatchStatus enum."""

    def test_batch_status_values(self):
        """Verify all expected status values exist."""
        assert BatchStatus.PENDING == 'pending'
        assert BatchStatus.VALIDATING == 'validating'
        assert BatchStatus.IN_PROGRESS == 'in_progress'
        assert BatchStatus.FINALIZING == 'finalizing'
        assert BatchStatus.COMPLETED == 'completed'
        assert BatchStatus.FAILED == 'failed'
        assert BatchStatus.EXPIRED == 'expired'
        assert BatchStatus.CANCELLING == 'cancelling'
        assert BatchStatus.CANCELLED == 'cancelled'

    def test_batch_status_is_string_enum(self):
        """BatchStatus should be usable as strings."""
        assert BatchStatus.COMPLETED.value == 'completed'
        assert BatchStatus.PENDING.value == 'pending'
        # Can compare with string values
        assert BatchStatus.COMPLETED == 'completed'


class TestBatch:
    """Tests for Batch dataclass."""

    def test_batch_creation(self):
        """Test basic batch creation."""
        batch = Batch(
            id='batch_123',
            status=BatchStatus.PENDING,
            created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
        )
        assert batch.id == 'batch_123'
        assert batch.status == BatchStatus.PENDING
        assert batch.request_count == 0
        assert batch.completed_count == 0
        assert batch.failed_count == 0
        assert batch.completed_at is None

    def test_batch_is_complete_property(self):
        """Test is_complete property for various states."""
        base_id = 'batch_123'
        base_created_at = datetime(2024, 1, 1, tzinfo=timezone.utc)

        # Not complete states
        assert Batch(id=base_id, created_at=base_created_at, status=BatchStatus.PENDING).is_complete is False
        assert Batch(id=base_id, created_at=base_created_at, status=BatchStatus.VALIDATING).is_complete is False
        assert Batch(id=base_id, created_at=base_created_at, status=BatchStatus.IN_PROGRESS).is_complete is False
        assert Batch(id=base_id, created_at=base_created_at, status=BatchStatus.FINALIZING).is_complete is False
        assert Batch(id=base_id, created_at=base_created_at, status=BatchStatus.CANCELLING).is_complete is False

        # Complete states
        assert Batch(id=base_id, created_at=base_created_at, status=BatchStatus.COMPLETED).is_complete is True
        assert Batch(id=base_id, created_at=base_created_at, status=BatchStatus.FAILED).is_complete is True
        assert Batch(id=base_id, created_at=base_created_at, status=BatchStatus.EXPIRED).is_complete is True
        assert Batch(id=base_id, created_at=base_created_at, status=BatchStatus.CANCELLED).is_complete is True

    def test_batch_is_successful_property(self):
        """Test is_successful property."""
        base_id = 'batch_123'
        base_created_at = datetime(2024, 1, 1, tzinfo=timezone.utc)

        # Only COMPLETED is successful
        assert Batch(id=base_id, created_at=base_created_at, status=BatchStatus.COMPLETED).is_successful is True
        assert Batch(id=base_id, created_at=base_created_at, status=BatchStatus.FAILED).is_successful is False
        assert Batch(id=base_id, created_at=base_created_at, status=BatchStatus.EXPIRED).is_successful is False
        assert Batch(id=base_id, created_at=base_created_at, status=BatchStatus.CANCELLED).is_successful is False


class TestBatchResult:
    """Tests for BatchResult dataclass."""

    def test_successful_result(self):
        """Test successful batch result."""
        response = ModelResponse(
            parts=[TextPart(content='Hello')],
            model_name='gpt-5-mini',
            timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
            usage=RequestUsage(input_tokens=10, output_tokens=5),
        )
        result = BatchResult(custom_id='req-1', response=response)

        assert result.custom_id == 'req-1'
        assert result.is_successful is True
        assert result.response == response
        assert result.error is None

    def test_failed_result(self):
        """Test failed batch result."""
        error = BatchError(code='rate_limit', message='Rate limit exceeded')
        result = BatchResult(custom_id='req-2', error=error)

        assert result.custom_id == 'req-2'
        assert result.is_successful is False
        assert result.response is None
        assert result.error == error


class TestBatchError:
    """Tests for BatchError dataclass."""

    def test_batch_error_creation(self):
        """Test batch error creation."""
        error = BatchError(code='invalid_request', message='Invalid JSON in request body')

        assert error.code == 'invalid_request'
        assert error.message == 'Invalid JSON in request body'


# --- OpenAI Batch Implementation Tests ---


@dataclass
class MockOpenAIBatch:
    """Mock for OpenAI batch-related operations."""

    batches_list: list[OpenAIBatchResponse] = field(default_factory=list)
    files_list: list[dict[str, Any]] = field(default_factory=list)
    batch_create_calls: list[dict[str, Any]] = field(default_factory=list)
    file_create_calls: list[dict[str, Any]] = field(default_factory=list)
    file_content_responses: dict[str, str] = field(default_factory=dict)
    base_url: str = 'https://api.openai.com/v1'

    def create_model(self) -> OpenAIChatModel:
        """Create an OpenAIChatModel using this mock as the client."""
        return OpenAIChatModel('gpt-5-mini', provider=OpenAIProvider(openai_client=cast(AsyncOpenAI, self)))

    @property
    def batches(self) -> Any:
        return self._BatchesNamespace(self)

    @property
    def files(self) -> Any:
        return self._FilesNamespace(self)

    @dataclass
    class _BatchesNamespace:
        parent: MockOpenAIBatch

        async def create(self, **kwargs: Any) -> OpenAIBatchResponse:
            self.parent.batch_create_calls.append(kwargs)
            if self.parent.batches_list:
                return self.parent.batches_list[0]
            return OpenAIBatchResponse(
                id='batch_abc123',
                completion_window='24h',
                created_at=1704067200,  # 2024-01-01
                endpoint='/v1/chat/completions',
                input_file_id=kwargs.get('input_file_id', 'file_xyz'),
                object='batch',
                status='validating',
            )

        async def retrieve(self, batch_id: str) -> OpenAIBatchResponse:
            for batch in self.parent.batches_list:
                if batch.id == batch_id:  # pragma: no branch
                    return batch
            return OpenAIBatchResponse(  # pragma: no cover
                id=batch_id,
                completion_window='24h',
                created_at=1704067200,
                endpoint='/v1/chat/completions',
                input_file_id='file_xyz',
                object='batch',
                status='completed',
                output_file_id='file_output',
            )

        async def cancel(self, batch_id: str) -> OpenAIBatchResponse:
            return OpenAIBatchResponse(
                id=batch_id,
                completion_window='24h',
                created_at=1704067200,
                endpoint='/v1/chat/completions',
                input_file_id='file_xyz',
                object='batch',
                status='cancelling',
            )

    @dataclass
    class _FilesNamespace:
        parent: MockOpenAIBatch

        async def create(self, **kwargs: Any) -> MagicMock:
            self.parent.file_create_calls.append(kwargs)
            mock_file = MagicMock()
            mock_file.id = 'file_uploaded_123'
            return mock_file

        async def content(self, file_id: str) -> MagicMock:
            mock_response = MagicMock()
            content = self.parent.file_content_responses.get(file_id, '')
            mock_response.text = content
            mock_response.read.return_value = content.encode('utf-8')
            return mock_response


class TestOpenAIBatch:
    """Tests for OpenAIBatch dataclass."""

    def test_openai_batch_creation(self):
        """Test OpenAIBatch extends Batch with OpenAI-specific fields."""
        batch = OpenAIBatch(
            id='batch_123',
            status=BatchStatus.COMPLETED,
            created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
            input_file_id='file_input',
            output_file_id='file_output',
            error_file_id=None,
            endpoint='/v1/chat/completions',
            completion_window='24h',
            metadata={'key': 'value'},
        )

        assert batch.id == 'batch_123'
        assert batch.input_file_id == 'file_input'
        assert batch.output_file_id == 'file_output'
        assert batch.endpoint == '/v1/chat/completions'
        assert batch.is_complete is True


class TestOpenAIChatModelBatch:
    """Tests for OpenAIChatModel batch methods."""

    async def test_batch_create_builds_jsonl(self, allow_model_requests: None):
        """Test that batch_create builds proper JSONL and uploads."""
        mock_client = MockOpenAIBatch()

        model = mock_client.create_model()

        messages_1: list[ModelMessage] = [ModelRequest.user_text_prompt('Hello')]
        messages_2: list[ModelMessage] = [ModelRequest.user_text_prompt('World')]
        requests = [
            ('req-1', messages_1, ModelRequestParameters()),
            ('req-2', messages_2, ModelRequestParameters()),
        ]

        batch = await model.batch_create(requests)

        # Verify file was created
        assert len(mock_client.file_create_calls) == 1
        file_call = mock_client.file_create_calls[0]
        assert file_call['purpose'] == 'batch'

        # Verify batch was created with correct params
        assert len(mock_client.batch_create_calls) == 1
        batch_call = mock_client.batch_create_calls[0]
        assert batch_call['endpoint'] == '/v1/chat/completions'
        assert batch_call['completion_window'] == '24h'
        assert batch_call['input_file_id'] == 'file_uploaded_123'

        # Verify returned batch
        assert batch.id == 'batch_abc123'
        assert batch.status == BatchStatus.VALIDATING

    async def test_batch_create_minimum_requests(self, allow_model_requests: None):
        """Test that batch_create requires at least 2 requests."""
        mock_client = MockOpenAIBatch()
        model = mock_client.create_model()

        messages_1: list[ModelMessage] = [ModelRequest.user_text_prompt('Hello')]
        requests = [
            ('req-1', messages_1, ModelRequestParameters()),
        ]

        with pytest.raises(ValueError, match='at least 2 requests'):
            await model.batch_create(requests)

    async def test_batch_status(self, allow_model_requests: None):
        """Test batch_status retrieves updated batch info."""
        mock_client = MockOpenAIBatch()
        mock_client.batches_list = [
            OpenAIBatchResponse(
                id='batch_123',
                completion_window='24h',
                created_at=1704067200,
                endpoint='/v1/chat/completions',
                input_file_id='file_input',
                object='batch',
                status='completed',
                output_file_id='file_output',
                request_counts=BatchRequestCounts(completed=5, failed=0, total=5),
            )
        ]

        model = mock_client.create_model()

        initial_batch = OpenAIBatch(
            id='batch_123',
            status=BatchStatus.IN_PROGRESS,
            created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
            input_file_id='file_input',
        )

        updated = await model.batch_status(initial_batch)

        assert updated.status == BatchStatus.COMPLETED
        assert updated.request_count == 5
        assert updated.completed_count == 5
        assert updated.failed_count == 0

    async def test_batch_results_success(self, allow_model_requests: None):
        """Test batch_results parses output file correctly."""
        # Create JSONL output content
        output_line = {
            'id': 'resp_123',
            'custom_id': 'req-1',
            'response': {
                'status_code': 200,
                'body': {
                    'id': 'chatcmpl-123',
                    'object': 'chat.completion',
                    'created': 1704067200,
                    'model': 'gpt-5-mini',
                    'choices': [
                        {
                            'index': 0,
                            'message': {'role': 'assistant', 'content': 'Hello there!'},
                            'finish_reason': 'stop',
                        }
                    ],
                    'usage': {'prompt_tokens': 10, 'completion_tokens': 5, 'total_tokens': 15},
                },
            },
        }
        output_content = json.dumps(output_line)

        mock_client = MockOpenAIBatch()
        mock_client.file_content_responses['file_output'] = output_content

        model = mock_client.create_model()

        batch = OpenAIBatch(
            id='batch_123',
            status=BatchStatus.COMPLETED,
            created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
            input_file_id='file_input',
            output_file_id='file_output',
        )

        results = await model.batch_results(batch)

        assert len(results) == 1
        assert results[0].custom_id == 'req-1'
        assert results[0].is_successful is True
        assert results[0].response is not None
        assert len(results[0].response.parts) == 1
        assert isinstance(results[0].response.parts[0], TextPart)
        assert results[0].response.parts[0].content == 'Hello there!'

    async def test_batch_results_incomplete_raises(self, allow_model_requests: None):
        """Test batch_results raises if batch not complete."""
        mock_client = MockOpenAIBatch()
        model = mock_client.create_model()

        batch = OpenAIBatch(
            id='batch_123',
            status=BatchStatus.IN_PROGRESS,
            created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
            input_file_id='file_input',
        )

        with pytest.raises(ValueError, match='not complete'):
            await model.batch_results(batch)

    async def test_batch_cancel(self, allow_model_requests: None):
        """Test batch_cancel sends cancel request."""
        mock_client = MockOpenAIBatch()
        model = mock_client.create_model()

        batch = OpenAIBatch(
            id='batch_123',
            status=BatchStatus.IN_PROGRESS,
            created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
            input_file_id='file_input',
        )

        cancelled = await model.batch_cancel(batch)

        assert cancelled.status == BatchStatus.CANCELLING


# --- Direct API Batch Function Tests ---


class TestDirectAPIBatchFunctions:
    """Tests for Direct API batch functions."""

    async def test_batch_create_direct_api(self, allow_model_requests: None):
        """Test batch_create Direct API function."""
        mock_client = MockOpenAIBatch()
        model = mock_client.create_model()

        messages_1: list[ModelMessage] = [ModelRequest.user_text_prompt('Hello')]
        messages_2: list[ModelMessage] = [ModelRequest.user_text_prompt('World')]
        requests = [
            ('req-1', messages_1, ModelRequestParameters()),
            ('req-2', messages_2, ModelRequestParameters()),
        ]

        batch = await batch_create(model, requests)

        assert batch.id == 'batch_abc123'
        assert batch.status == BatchStatus.VALIDATING

    async def test_batch_create_direct_api_simple_tuples(self, allow_model_requests: None):
        """Test batch_create Direct API with simple 2-tuples (no explicit params)."""
        mock_client = MockOpenAIBatch()
        model = mock_client.create_model()

        # Use simple 2-tuples - parameters should default
        requests: list[tuple[str, list[ModelMessage]]] = [
            ('req-1', [ModelRequest.user_text_prompt('Hello')]),
            ('req-2', [ModelRequest.user_text_prompt('World')]),
        ]

        batch = await batch_create(model, requests)

        assert batch.id == 'batch_abc123'
        assert batch.status == BatchStatus.VALIDATING

    async def test_batch_status_direct_api(self, allow_model_requests: None):
        """Test batch_status Direct API function."""
        mock_client = MockOpenAIBatch()
        mock_client.batches_list = [
            OpenAIBatchResponse(
                id='batch_123',
                completion_window='24h',
                created_at=1704067200,
                endpoint='/v1/chat/completions',
                input_file_id='file_input',
                object='batch',
                status='completed',
                output_file_id='file_output',
            )
        ]

        model = mock_client.create_model()

        initial_batch = OpenAIBatch(
            id='batch_123',
            status=BatchStatus.IN_PROGRESS,
            created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
            input_file_id='file_input',
        )

        updated = await batch_status(model, initial_batch)

        assert updated.status == BatchStatus.COMPLETED

    async def test_batch_results_direct_api(self, allow_model_requests: None):
        """Test batch_results Direct API function."""
        output_line = {
            'id': 'resp_123',
            'custom_id': 'req-1',
            'response': {
                'status_code': 200,
                'body': {
                    'id': 'chatcmpl-123',
                    'object': 'chat.completion',
                    'created': 1704067200,
                    'model': 'gpt-5-mini',
                    'choices': [
                        {
                            'index': 0,
                            'message': {'role': 'assistant', 'content': 'Test response'},
                            'finish_reason': 'stop',
                        }
                    ],
                },
            },
        }

        mock_client = MockOpenAIBatch()
        mock_client.file_content_responses['file_output'] = json.dumps(output_line)

        model = mock_client.create_model()

        batch = OpenAIBatch(
            id='batch_123',
            status=BatchStatus.COMPLETED,
            created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
            input_file_id='file_input',
            output_file_id='file_output',
        )

        results = await batch_results(model, batch)

        assert len(results) == 1
        assert results[0].is_successful is True

    async def test_batch_cancel_direct_api(self, allow_model_requests: None):
        """Test batch_cancel Direct API function."""
        mock_client = MockOpenAIBatch()

        model = mock_client.create_model()

        batch = OpenAIBatch(
            id='batch_123',
            status=BatchStatus.IN_PROGRESS,
            created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
            input_file_id='file_input',
        )

        cancelled = await batch_cancel(model, batch)

        assert cancelled.status == BatchStatus.CANCELLING

    def test_batch_create_sync(self, allow_model_requests: None):
        """Test synchronous batch_create."""
        mock_client = MockOpenAIBatch()
        model = mock_client.create_model()

        messages_1: list[ModelMessage] = [ModelRequest.user_text_prompt('Hello')]
        messages_2: list[ModelMessage] = [ModelRequest.user_text_prompt('World')]
        requests = [
            ('req-1', messages_1, ModelRequestParameters()),
            ('req-2', messages_2, ModelRequestParameters()),
        ]

        batch = batch_create_sync(model, requests)

        assert batch.id == 'batch_abc123'

    def test_batch_status_sync(self, allow_model_requests: None):
        """Test synchronous batch_status."""
        mock_client = MockOpenAIBatch()
        mock_client.batches_list = [
            OpenAIBatchResponse(
                id='batch_123',
                completion_window='24h',
                created_at=1704067200,
                endpoint='/v1/chat/completions',
                input_file_id='file_input',
                object='batch',
                status='completed',
            )
        ]
        model = mock_client.create_model()

        initial_batch = OpenAIBatch(
            id='batch_123',
            status=BatchStatus.IN_PROGRESS,
            created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
            input_file_id='file_input',
        )

        updated = batch_status_sync(model, initial_batch)

        assert updated.status == BatchStatus.COMPLETED

    def test_batch_cancel_sync(self, allow_model_requests: None):
        """Test synchronous batch_cancel."""
        mock_client = MockOpenAIBatch()

        model = mock_client.create_model()

        batch = OpenAIBatch(
            id='batch_123',
            status=BatchStatus.IN_PROGRESS,
            created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
            input_file_id='file_input',
        )

        cancelled = batch_cancel_sync(model, batch)

        assert cancelled.status == BatchStatus.CANCELLING

    def test_batch_results_sync(self, allow_model_requests: None):
        """Test synchronous batch_results."""
        from pydantic_ai.direct import batch_results_sync

        # Create response content
        response_line = {
            'id': 'resp_1',
            'custom_id': 'req-1',
            'response': {
                'status_code': 200,
                'body': {
                    'id': 'chatcmpl-1',
                    'object': 'chat.completion',
                    'created': 1704067200,
                    'model': 'gpt-5-mini',
                    'choices': [
                        {
                            'index': 0,
                            'message': {'role': 'assistant', 'content': 'Hello!'},
                            'finish_reason': 'stop',
                        }
                    ],
                },
            },
        }

        mock_client = MockOpenAIBatch()
        mock_client.file_content_responses['file_output'] = json.dumps(response_line)

        model = mock_client.create_model()

        batch = OpenAIBatch(
            id='batch_123',
            status=BatchStatus.COMPLETED,
            created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
            input_file_id='file_input',
            output_file_id='file_output',
        )

        results = batch_results_sync(model, batch)

        assert len(results) == 1
        assert results[0].custom_id == 'req-1'
        assert results[0].is_successful


# --- Model Base Class Tests ---


class TestModelBatchMethodsNotImplemented:
    """Test that Model base class batch methods raise NotImplementedError."""

    async def test_batch_create_not_implemented(self, allow_model_requests: None):
        """Test batch_create raises NotImplementedError for unsupported models."""
        from pydantic_ai.models.test import TestModel

        model = TestModel()
        messages_1: list[ModelMessage] = [ModelRequest.user_text_prompt('Hello')]
        messages_2: list[ModelMessage] = [ModelRequest.user_text_prompt('World')]
        requests = [
            ('req-1', messages_1, ModelRequestParameters()),
            ('req-2', messages_2, ModelRequestParameters()),
        ]

        with pytest.raises(NotImplementedError, match='does not support batch processing'):
            await model.batch_create(requests)

    async def test_batch_status_not_implemented(self, allow_model_requests: None):
        """Test batch_status raises NotImplementedError for unsupported models."""
        from pydantic_ai.models.test import TestModel

        model = TestModel()
        batch = Batch(
            id='batch_123',
            status=BatchStatus.PENDING,
            created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
        )

        with pytest.raises(NotImplementedError, match='does not support batch processing'):
            await model.batch_status(batch)

    async def test_batch_results_not_implemented(self, allow_model_requests: None):
        """Test batch_results raises NotImplementedError for unsupported models."""
        from pydantic_ai.models.test import TestModel

        model = TestModel()
        batch = Batch(
            id='batch_123',
            status=BatchStatus.COMPLETED,
            created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
        )

        with pytest.raises(NotImplementedError, match='does not support batch processing'):
            await model.batch_results(batch)

    async def test_batch_cancel_not_implemented(self, allow_model_requests: None):
        """Test batch_cancel raises NotImplementedError for unsupported models."""
        from pydantic_ai.models.test import TestModel

        model = TestModel()
        batch = Batch(
            id='batch_123',
            status=BatchStatus.IN_PROGRESS,
            created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
        )

        with pytest.raises(NotImplementedError, match='does not support batch processing'):
            await model.batch_cancel(batch)


# --- Integration Tests ---


class TestBatchErrorHandling:
    """Tests for batch error handling scenarios."""

    async def test_batch_results_with_errors(self, allow_model_requests: None):
        """Test batch_results handles error responses."""
        # Success response
        success_line = {
            'id': 'resp_1',
            'custom_id': 'req-1',
            'response': {
                'status_code': 200,
                'body': {
                    'id': 'chatcmpl-1',
                    'object': 'chat.completion',
                    'created': 1704067200,
                    'model': 'gpt-5-mini',
                    'choices': [
                        {
                            'index': 0,
                            'message': {'role': 'assistant', 'content': 'Success!'},
                            'finish_reason': 'stop',
                        }
                    ],
                },
            },
        }

        # Error response
        error_line = {
            'id': 'resp_2',
            'custom_id': 'req-2',
            'error': {
                'code': 'rate_limit_exceeded',
                'message': 'Rate limit exceeded',
            },
        }

        output_content = '\n'.join([json.dumps(success_line), json.dumps(error_line)])

        mock_client = MockOpenAIBatch()
        mock_client.file_content_responses['file_output'] = output_content

        model = mock_client.create_model()

        batch = OpenAIBatch(
            id='batch_123',
            status=BatchStatus.COMPLETED,
            created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
            input_file_id='file_input',
            output_file_id='file_output',
        )

        results = await model.batch_results(batch)

        assert len(results) == 2

        # First result should be successful
        assert results[0].custom_id == 'req-1'
        assert results[0].is_successful is True
        assert results[0].response is not None

        # Second result should have error
        assert results[1].custom_id == 'req-2'
        assert results[1].is_successful is False
        assert results[1].error is not None
        assert results[1].error.code == 'rate_limit_exceeded'
        assert results[1].error.message == 'Rate limit exceeded'

    async def test_batch_results_empty_output(self, allow_model_requests: None):
        """Test batch_results handles empty output file."""
        mock_client = MockOpenAIBatch()
        mock_client.file_content_responses['file_output'] = ''

        model = mock_client.create_model()

        batch = OpenAIBatch(
            id='batch_123',
            status=BatchStatus.COMPLETED,
            created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
            input_file_id='file_input',
            output_file_id='file_output',
        )

        results = await model.batch_results(batch)

        assert results == []


class TestBatchAPIErrorHandling:
    """Tests for batch API error handling (APIStatusError and APIConnectionError)."""

    @dataclass
    class MockOpenAIWithAPIError:
        """Mock OpenAI client that raises API errors."""

        error_type: str = 'status'  # 'status' or 'connection'
        error_on: str = 'batches.create'  # which method should error
        base_url: str = 'https://api.openai.com/v1'

        def create_model(self) -> OpenAIChatModel:
            """Create an OpenAIChatModel using this mock as the client."""
            return OpenAIChatModel('gpt-5-mini', provider=OpenAIProvider(openai_client=cast(AsyncOpenAI, self)))

        @property
        def batches(self) -> Any:
            return self._BatchesNamespace(self)

        @property
        def files(self) -> Any:
            return self._FilesNamespace(self)

        def _raise_error(self) -> None:
            if self.error_type == 'status':
                raise APIStatusError(
                    message='Rate limit exceeded',
                    response=MagicMock(status_code=429),
                    body={'error': {'message': 'Rate limit exceeded'}},
                )
            else:
                raise APIConnectionError(message='Connection failed', request=MagicMock())

        @dataclass
        class _BatchesNamespace:
            parent: TestBatchAPIErrorHandling.MockOpenAIWithAPIError

            async def create(self, **kwargs: Any) -> OpenAIBatchResponse:  # pragma: no cover
                if self.parent.error_on == 'batches.create':
                    self.parent._raise_error()
                return OpenAIBatchResponse(
                    id='batch_abc123',
                    completion_window='24h',
                    created_at=1704067200,
                    endpoint='/v1/chat/completions',
                    input_file_id='file_xyz',
                    object='batch',
                    status='validating',
                )

            async def retrieve(self, batch_id: str) -> OpenAIBatchResponse:
                if self.parent.error_on == 'batches.retrieve':
                    self.parent._raise_error()
                return OpenAIBatchResponse(  # pragma: no cover
                    id=batch_id,
                    completion_window='24h',
                    created_at=1704067200,
                    endpoint='/v1/chat/completions',
                    input_file_id='file_xyz',
                    object='batch',
                    status='completed',
                )

            async def cancel(self, batch_id: str) -> OpenAIBatchResponse:
                if self.parent.error_on == 'batches.cancel':
                    self.parent._raise_error()
                return OpenAIBatchResponse(  # pragma: no cover
                    id=batch_id,
                    completion_window='24h',
                    created_at=1704067200,
                    endpoint='/v1/chat/completions',
                    input_file_id='file_xyz',
                    object='batch',
                    status='cancelling',
                )

        @dataclass
        class _FilesNamespace:
            parent: TestBatchAPIErrorHandling.MockOpenAIWithAPIError

            async def create(self, **kwargs: Any) -> MagicMock:
                if self.parent.error_on == 'files.create':
                    self.parent._raise_error()
                mock_file = MagicMock()  # pragma: no cover
                mock_file.id = 'file_uploaded_123'  # pragma: no cover
                return mock_file  # pragma: no cover

            async def content(self, file_id: str) -> MagicMock:
                if self.parent.error_on == 'files.content':
                    self.parent._raise_error()
                mock_response = MagicMock()  # pragma: no cover
                mock_response.text = ''  # pragma: no cover
                return mock_response  # pragma: no cover

    async def test_batch_create_api_status_error(self, allow_model_requests: None):
        """Test batch_create raises ModelHTTPError on APIStatusError."""
        mock_client = self.MockOpenAIWithAPIError(error_type='status', error_on='files.create')
        model = mock_client.create_model()

        messages_1: list[ModelMessage] = [ModelRequest.user_text_prompt('Hello')]
        messages_2: list[ModelMessage] = [ModelRequest.user_text_prompt('World')]
        requests = [
            ('req-1', messages_1, ModelRequestParameters()),
            ('req-2', messages_2, ModelRequestParameters()),
        ]

        with pytest.raises(ModelHTTPError) as exc_info:
            await model.batch_create(requests)

        assert exc_info.value.status_code == 429

    async def test_batch_create_api_connection_error(self, allow_model_requests: None):
        """Test batch_create raises ModelAPIError on APIConnectionError."""
        mock_client = self.MockOpenAIWithAPIError(error_type='connection', error_on='files.create')
        model = mock_client.create_model()

        messages_1: list[ModelMessage] = [ModelRequest.user_text_prompt('Hello')]
        messages_2: list[ModelMessage] = [ModelRequest.user_text_prompt('World')]
        requests = [
            ('req-1', messages_1, ModelRequestParameters()),
            ('req-2', messages_2, ModelRequestParameters()),
        ]

        with pytest.raises(ModelAPIError) as exc_info:
            await model.batch_create(requests)

        assert 'Connection failed' in str(exc_info.value)

    async def test_batch_status_api_status_error(self, allow_model_requests: None):
        """Test batch_status raises ModelHTTPError on APIStatusError."""
        mock_client = self.MockOpenAIWithAPIError(error_type='status', error_on='batches.retrieve')
        model = mock_client.create_model()

        batch = OpenAIBatch(
            id='batch_123',
            status=BatchStatus.IN_PROGRESS,
            created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
            input_file_id='file_input',
        )

        with pytest.raises(ModelHTTPError) as exc_info:
            await model.batch_status(batch)

        assert exc_info.value.status_code == 429

    async def test_batch_status_api_connection_error(self, allow_model_requests: None):
        """Test batch_status raises ModelAPIError on APIConnectionError."""
        mock_client = self.MockOpenAIWithAPIError(error_type='connection', error_on='batches.retrieve')
        model = mock_client.create_model()

        batch = OpenAIBatch(
            id='batch_123',
            status=BatchStatus.IN_PROGRESS,
            created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
            input_file_id='file_input',
        )

        with pytest.raises(ModelAPIError) as exc_info:
            await model.batch_status(batch)

        assert 'Connection failed' in str(exc_info.value)

    async def test_batch_cancel_api_status_error(self, allow_model_requests: None):
        """Test batch_cancel raises ModelHTTPError on APIStatusError."""
        mock_client = self.MockOpenAIWithAPIError(error_type='status', error_on='batches.cancel')
        model = mock_client.create_model()

        batch = OpenAIBatch(
            id='batch_123',
            status=BatchStatus.IN_PROGRESS,
            created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
            input_file_id='file_input',
        )

        with pytest.raises(ModelHTTPError) as exc_info:
            await model.batch_cancel(batch)

        assert exc_info.value.status_code == 429

    async def test_batch_cancel_api_connection_error(self, allow_model_requests: None):
        """Test batch_cancel raises ModelAPIError on APIConnectionError."""
        mock_client = self.MockOpenAIWithAPIError(error_type='connection', error_on='batches.cancel')
        model = mock_client.create_model()

        batch = OpenAIBatch(
            id='batch_123',
            status=BatchStatus.IN_PROGRESS,
            created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
            input_file_id='file_input',
        )

        with pytest.raises(ModelAPIError) as exc_info:
            await model.batch_cancel(batch)

        assert 'Connection failed' in str(exc_info.value)

    async def test_batch_results_api_status_error(self, allow_model_requests: None):
        """Test batch_results raises ModelHTTPError on APIStatusError when fetching file."""
        mock_client = self.MockOpenAIWithAPIError(error_type='status', error_on='files.content')
        model = mock_client.create_model()

        batch = OpenAIBatch(
            id='batch_123',
            status=BatchStatus.COMPLETED,
            created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
            input_file_id='file_input',
            output_file_id='file_output',
        )

        with pytest.raises(ModelHTTPError) as exc_info:
            await model.batch_results(batch)

        assert exc_info.value.status_code == 429

    async def test_batch_results_api_connection_error(self, allow_model_requests: None):
        """Test batch_results raises ModelAPIError on APIConnectionError when fetching file."""
        mock_client = self.MockOpenAIWithAPIError(error_type='connection', error_on='files.content')
        model = mock_client.create_model()

        batch = OpenAIBatch(
            id='batch_123',
            status=BatchStatus.COMPLETED,
            created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
            input_file_id='file_input',
            output_file_id='file_output',
        )

        with pytest.raises(ModelAPIError) as exc_info:
            await model.batch_results(batch)

        assert 'Connection failed' in str(exc_info.value)


class TestBatchJSONParseErrorHandling:
    """Tests for JSON parse error handling in batch results."""

    async def test_batch_results_malformed_json(self, allow_model_requests: None):
        """Test batch_results handles malformed JSON lines gracefully."""
        # Mix of valid and invalid JSON
        valid_line = json.dumps(
            {
                'id': 'resp_1',
                'custom_id': 'req-1',
                'response': {
                    'status_code': 200,
                    'body': {
                        'id': 'chatcmpl-1',
                        'object': 'chat.completion',
                        'created': 1704067200,
                        'model': 'gpt-5-mini',
                        'choices': [
                            {
                                'index': 0,
                                'message': {'role': 'assistant', 'content': 'Success!'},
                                'finish_reason': 'stop',
                            }
                        ],
                    },
                },
            }
        )
        invalid_line = 'this is not valid json {'
        another_valid = json.dumps(
            {
                'id': 'resp_2',
                'custom_id': 'req-2',
                'response': {
                    'status_code': 200,
                    'body': {
                        'id': 'chatcmpl-2',
                        'object': 'chat.completion',
                        'created': 1704067200,
                        'model': 'gpt-5-mini',
                        'choices': [
                            {
                                'index': 0,
                                'message': {'role': 'assistant', 'content': 'Also success!'},
                                'finish_reason': 'stop',
                            }
                        ],
                    },
                },
            }
        )

        output_content = '\n'.join([valid_line, invalid_line, another_valid])

        mock_client = MockOpenAIBatch()
        mock_client.file_content_responses['file_output'] = output_content

        model = mock_client.create_model()

        batch = OpenAIBatch(
            id='batch_123',
            status=BatchStatus.COMPLETED,
            created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
            input_file_id='file_input',
            output_file_id='file_output',
        )

        results = await model.batch_results(batch)

        # Should have 3 results: 2 successful + 1 JSON parse error
        assert len(results) == 3

        # First result should be successful
        assert results[0].custom_id == 'req-1'
        assert results[0].is_successful is True

        # Second result should be the JSON parse error
        assert results[1].custom_id == 'parse_error_1'
        assert results[1].is_successful is False
        assert results[1].error is not None
        assert results[1].error.code == 'json_parse_error'
        assert 'Failed to parse result line' in results[1].error.message

        # Third result should also be successful
        assert results[2].custom_id == 'req-2'
        assert results[2].is_successful is True


# --- High-Level model_request_batch Tests ---


class TestModelRequestBatch:
    """Tests for the high-level model_request_batch function."""

    async def test_model_request_batch_success(self, allow_model_requests: None):
        """Test model_request_batch completes successfully."""
        from pydantic_ai.direct import model_request_batch

        # Set up mock that returns completed status immediately
        mock_client = MockOpenAIBatch()

        # First call creates the batch
        mock_client.batches_list = [
            OpenAIBatchResponse(
                id='batch_123',
                completion_window='24h',
                created_at=1704067200,
                endpoint='/v1/chat/completions',
                input_file_id='file_input',
                object='batch',
                status='completed',
                output_file_id='file_output',
                request_counts=BatchRequestCounts(completed=1, failed=0, total=1),
            )
        ]

        # Set up results
        output_line = {
            'id': 'resp_123',
            'custom_id': 'req-1',
            'response': {
                'status_code': 200,
                'body': {
                    'id': 'chatcmpl-123',
                    'object': 'chat.completion',
                    'created': 1704067200,
                    'model': 'gpt-5-mini',
                    'choices': [
                        {
                            'index': 0,
                            'message': {'role': 'assistant', 'content': 'Test response'},
                            'finish_reason': 'stop',
                        }
                    ],
                },
            },
        }
        mock_client.file_content_responses['file_output'] = json.dumps(output_line)

        model = mock_client.create_model()

        messages_1: list[ModelMessage] = [ModelRequest.user_text_prompt('Hello')]
        messages_2: list[ModelMessage] = [ModelRequest.user_text_prompt('World')]
        requests: list[tuple[str, list[ModelMessage]]] = [
            ('req-1', messages_1),
            ('req-2', messages_2),
        ]

        results = await model_request_batch(model, requests, poll_interval=0.01)

        assert len(results) >= 1  # At least one result
        assert results[0].custom_id == 'req-1'
        assert results[0].is_successful is True

    async def test_model_request_batch_timeout(self, allow_model_requests: None):
        """Test model_request_batch raises TimeoutError on timeout."""
        import asyncio

        from pydantic_ai.direct import model_request_batch

        # Set up mock that never completes
        mock_client = MockOpenAIBatch()
        mock_client.batches_list = [
            OpenAIBatchResponse(
                id='batch_123',
                completion_window='24h',
                created_at=1704067200,
                endpoint='/v1/chat/completions',
                input_file_id='file_input',
                object='batch',
                status='in_progress',  # Never completes
            )
        ]

        model = mock_client.create_model()

        messages_1: list[ModelMessage] = [ModelRequest.user_text_prompt('Hello')]
        messages_2: list[ModelMessage] = [ModelRequest.user_text_prompt('World')]
        requests: list[tuple[str, list[ModelMessage]]] = [
            ('req-1', messages_1),
            ('req-2', messages_2),
        ]

        with pytest.raises(asyncio.TimeoutError, match='timed out'):
            await model_request_batch(model, requests, poll_interval=0.01, timeout=0.05)

    async def test_model_request_batch_cancellation(self, allow_model_requests: None):
        """Test model_request_batch handles cancellation."""
        import asyncio

        from pydantic_ai.direct import model_request_batch

        # Set up mock that never completes
        mock_client = MockOpenAIBatch()
        mock_client.batches_list = [
            OpenAIBatchResponse(
                id='batch_123',
                completion_window='24h',
                created_at=1704067200,
                endpoint='/v1/chat/completions',
                input_file_id='file_input',
                object='batch',
                status='in_progress',  # Never completes
            )
        ]

        model = mock_client.create_model()

        messages_1: list[ModelMessage] = [ModelRequest.user_text_prompt('Hello')]
        messages_2: list[ModelMessage] = [ModelRequest.user_text_prompt('World')]
        requests: list[tuple[str, list[ModelMessage]]] = [
            ('req-1', messages_1),
            ('req-2', messages_2),
        ]

        # Create task and cancel it
        task = asyncio.create_task(model_request_batch(model, requests, poll_interval=0.1))
        await asyncio.sleep(0.05)  # Let it start
        task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await task

    def test_model_request_batch_sync(self, allow_model_requests: None):
        """Test the synchronous model_request_batch_sync wrapper."""
        from pydantic_ai.direct import model_request_batch_sync

        # Set up mock that returns completed status immediately
        mock_client = MockOpenAIBatch()
        mock_client.batches_list = [
            OpenAIBatchResponse(
                id='batch_sync',
                completion_window='24h',
                created_at=1704067200,
                endpoint='/v1/chat/completions',
                input_file_id='file_input',
                object='batch',
                status='completed',
                output_file_id='file_output',
                request_counts=BatchRequestCounts(completed=2, failed=0, total=2),
            )
        ]

        # Set up results for both requests (OpenAI batch requires at least 2 requests)
        output_lines = '\n'.join(
            [
                json.dumps(
                    {
                        'id': 'resp_sync_1',
                        'custom_id': 'req-sync-1',
                        'response': {
                            'status_code': 200,
                            'body': {
                                'id': 'chatcmpl-sync-1',
                                'object': 'chat.completion',
                                'created': 1704067200,
                                'model': 'gpt-5-mini',
                                'choices': [
                                    {
                                        'index': 0,
                                        'message': {'role': 'assistant', 'content': 'Sync response 1'},
                                        'finish_reason': 'stop',
                                    }
                                ],
                            },
                        },
                    }
                ),
                json.dumps(
                    {
                        'id': 'resp_sync_2',
                        'custom_id': 'req-sync-2',
                        'response': {
                            'status_code': 200,
                            'body': {
                                'id': 'chatcmpl-sync-2',
                                'object': 'chat.completion',
                                'created': 1704067200,
                                'model': 'gpt-5-mini',
                                'choices': [
                                    {
                                        'index': 0,
                                        'message': {'role': 'assistant', 'content': 'Sync response 2'},
                                        'finish_reason': 'stop',
                                    }
                                ],
                            },
                        },
                    }
                ),
            ]
        )
        mock_client.file_content_responses['file_output'] = output_lines

        model = mock_client.create_model()

        messages_1: list[ModelMessage] = [ModelRequest.user_text_prompt('Hello')]
        messages_2: list[ModelMessage] = [ModelRequest.user_text_prompt('World')]
        requests: list[tuple[str, list[ModelMessage]]] = [
            ('req-sync-1', messages_1),
            ('req-sync-2', messages_2),
        ]

        results = model_request_batch_sync(model, requests, poll_interval=0.01)

        assert len(results) == 2
        result_map = {r.custom_id: r for r in results}
        assert result_map['req-sync-1'].is_successful is True
        assert result_map['req-sync-2'].is_successful is True

    async def test_model_request_batch_with_per_request_params(self, allow_model_requests: None):
        """Test model_request_batch with per-request ModelRequestParameters (3-tuple format)."""
        from pydantic_ai.direct import model_request_batch

        # Set up mock
        mock_client = MockOpenAIBatch()
        mock_client.batches_list = [
            OpenAIBatchResponse(
                id='batch_params',
                completion_window='24h',
                created_at=1704067200,
                endpoint='/v1/chat/completions',
                input_file_id='file_input',
                object='batch',
                status='completed',
                output_file_id='file_output',
                request_counts=BatchRequestCounts(completed=2, failed=0, total=2),
            )
        ]

        # Set up results for both requests
        output_lines = '\n'.join(
            [
                json.dumps(
                    {
                        'id': 'resp_1',
                        'custom_id': 'req-1',
                        'response': {
                            'status_code': 200,
                            'body': {
                                'id': 'chatcmpl-1',
                                'object': 'chat.completion',
                                'created': 1704067200,
                                'model': 'gpt-5-mini',
                                'choices': [
                                    {
                                        'index': 0,
                                        'message': {'role': 'assistant', 'content': 'Response 1'},
                                        'finish_reason': 'stop',
                                    }
                                ],
                            },
                        },
                    }
                ),
                json.dumps(
                    {
                        'id': 'resp_2',
                        'custom_id': 'req-2',
                        'response': {
                            'status_code': 200,
                            'body': {
                                'id': 'chatcmpl-2',
                                'object': 'chat.completion',
                                'created': 1704067200,
                                'model': 'gpt-5-mini',
                                'choices': [
                                    {
                                        'index': 0,
                                        'message': {'role': 'assistant', 'content': 'Response 2'},
                                        'finish_reason': 'stop',
                                    }
                                ],
                            },
                        },
                    }
                ),
            ]
        )
        mock_client.file_content_responses['file_output'] = output_lines

        model = mock_client.create_model()

        # Create different parameters for each request
        params_1 = ModelRequestParameters(allow_text_output=True)
        params_2 = ModelRequestParameters(allow_text_output=False)

        messages_1: list[ModelMessage] = [ModelRequest.user_text_prompt('Hello')]
        messages_2: list[ModelMessage] = [ModelRequest.user_text_prompt('World')]

        # Use 3-tuple format with per-request parameters
        requests: list[tuple[str, list[ModelMessage], ModelRequestParameters]] = [
            ('req-1', messages_1, params_1),
            ('req-2', messages_2, params_2),
        ]

        results = await model_request_batch(model, requests, poll_interval=0.01)

        assert len(results) == 2
        assert results[0].custom_id == 'req-1'
        assert results[1].custom_id == 'req-2'


class TestInstrumentedModelBatch:
    """Tests for batch operations through instrumented/wrapper models."""

    async def test_instrumented_model_forwards_batch_create(self, allow_model_requests: None):
        """Test that InstrumentedModel properly forwards batch_create to wrapped model."""
        from pydantic_ai.models.instrumented import InstrumentedModel

        # Set up mock
        mock_client = MockOpenAIBatch()
        mock_client.batches_list = [
            OpenAIBatchResponse(
                id='batch_instrumented',
                completion_window='24h',
                created_at=1704067200,
                endpoint='/v1/chat/completions',
                input_file_id='file_input',
                object='batch',
                status='completed',
                output_file_id='file_output',
                request_counts=BatchRequestCounts(completed=2, failed=0, total=2),
            )
        ]

        base_model = mock_client.create_model()
        instrumented_model = InstrumentedModel(base_model)

        messages: list[ModelMessage] = [ModelRequest.user_text_prompt('Hello')]
        requests: list[tuple[str, list[ModelMessage], ModelRequestParameters]] = [
            ('req-1', messages, ModelRequestParameters()),
            ('req-2', messages, ModelRequestParameters()),
        ]

        batch = await instrumented_model.batch_create(requests)

        assert batch.id == 'batch_instrumented'
        assert batch.status == BatchStatus.COMPLETED
        assert len(mock_client.batch_create_calls) == 1

    async def test_instrumented_model_forwards_batch_status(self, allow_model_requests: None):
        """Test that InstrumentedModel properly forwards batch_status to wrapped model."""
        from pydantic_ai.models.instrumented import InstrumentedModel

        # Set up mock
        mock_client = MockOpenAIBatch()
        mock_client.batches_list = [
            OpenAIBatchResponse(
                id='batch_status_test',
                completion_window='24h',
                created_at=1704067200,
                endpoint='/v1/chat/completions',
                input_file_id='file_input',
                object='batch',
                status='in_progress',
                request_counts=BatchRequestCounts(completed=1, failed=0, total=2),
            )
        ]

        base_model = mock_client.create_model()
        instrumented_model = InstrumentedModel(base_model)

        batch = OpenAIBatch(
            id='batch_status_test',
            status=BatchStatus.IN_PROGRESS,
            created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
        )

        updated = await instrumented_model.batch_status(batch)

        assert updated.id == 'batch_status_test'
        assert updated.status == BatchStatus.IN_PROGRESS

    async def test_instrumented_model_forwards_batch_results(self, allow_model_requests: None):
        """Test that InstrumentedModel properly forwards batch_results to wrapped model."""
        from pydantic_ai.models.instrumented import InstrumentedModel

        # Set up mock with output file
        mock_client = MockOpenAIBatch()
        output_lines = json.dumps(
            {
                'id': 'resp_1',
                'custom_id': 'req-1',
                'response': {
                    'status_code': 200,
                    'body': {
                        'id': 'chatcmpl-1',
                        'object': 'chat.completion',
                        'created': 1704067200,
                        'model': 'gpt-5-mini',
                        'choices': [
                            {
                                'index': 0,
                                'message': {'role': 'assistant', 'content': 'Hello!'},
                                'finish_reason': 'stop',
                            }
                        ],
                    },
                },
            }
        )
        mock_client.file_content_responses['file_output'] = output_lines

        base_model = mock_client.create_model()
        instrumented_model = InstrumentedModel(base_model)

        batch = OpenAIBatch(
            id='batch_results_test',
            status=BatchStatus.COMPLETED,
            created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
            output_file_id='file_output',
        )

        results = await instrumented_model.batch_results(batch)

        assert len(results) == 1
        assert results[0].custom_id == 'req-1'
        assert results[0].is_successful is True

    async def test_instrumented_model_forwards_batch_cancel(self, allow_model_requests: None):
        """Test that InstrumentedModel properly forwards batch_cancel to wrapped model."""
        from pydantic_ai.models.instrumented import InstrumentedModel

        mock_client = MockOpenAIBatch()
        base_model = mock_client.create_model()
        instrumented_model = InstrumentedModel(base_model)

        batch = OpenAIBatch(
            id='batch_cancel_test',
            status=BatchStatus.IN_PROGRESS,
            created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
        )

        updated = await instrumented_model.batch_cancel(batch)

        assert updated.id == 'batch_cancel_test'
        assert updated.status == BatchStatus.CANCELLING


class TestBatchResultsValidationError:
    """Tests for ValidationError handling in batch_results."""

    async def test_validation_error_converted_to_batch_error(self, allow_model_requests: None):
        """Test that ValidationError from model_validate is converted to BatchError."""
        # Set up mock with invalid response body that will fail validation
        mock_client = MockOpenAIBatch()
        output_lines = json.dumps(
            {
                'id': 'resp_1',
                'custom_id': 'req-invalid',
                'response': {
                    'status_code': 200,
                    'body': {
                        # Missing required fields like 'id', 'object', 'created', 'model'
                        'choices': [],
                    },
                },
            }
        )
        mock_client.file_content_responses['file_output'] = output_lines

        model = mock_client.create_model()

        batch = OpenAIBatch(
            id='batch_validation_test',
            status=BatchStatus.COMPLETED,
            created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
            output_file_id='file_output',
        )

        results = await model.batch_results(batch)

        # Should not raise, but return a BatchError
        assert len(results) == 1
        assert results[0].custom_id == 'req-invalid'
        assert results[0].is_successful is False
        assert results[0].error is not None
        assert results[0].error.code == 'validation_error'
        assert 'Failed to validate response' in results[0].error.message

    async def test_mixed_valid_and_invalid_responses(self, allow_model_requests: None):
        """Test that valid and invalid responses are handled correctly together."""
        mock_client = MockOpenAIBatch()
        output_lines = '\n'.join(
            [
                # Valid response
                json.dumps(
                    {
                        'id': 'resp_1',
                        'custom_id': 'req-valid',
                        'response': {
                            'status_code': 200,
                            'body': {
                                'id': 'chatcmpl-1',
                                'object': 'chat.completion',
                                'created': 1704067200,
                                'model': 'gpt-5-mini',
                                'choices': [
                                    {
                                        'index': 0,
                                        'message': {'role': 'assistant', 'content': 'Valid response'},
                                        'finish_reason': 'stop',
                                    }
                                ],
                            },
                        },
                    }
                ),
                # Invalid response (missing required fields)
                json.dumps(
                    {
                        'id': 'resp_2',
                        'custom_id': 'req-invalid',
                        'response': {
                            'status_code': 200,
                            'body': {
                                # Invalid - missing required fields
                                'not_a_valid_field': 'bad data',
                            },
                        },
                    }
                ),
            ]
        )
        mock_client.file_content_responses['file_output'] = output_lines

        model = mock_client.create_model()

        batch = OpenAIBatch(
            id='batch_mixed_test',
            status=BatchStatus.COMPLETED,
            created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
            output_file_id='file_output',
        )

        results = await model.batch_results(batch)

        # Should get 2 results
        assert len(results) == 2

        # First result should be valid
        result_map = {r.custom_id: r for r in results}
        assert result_map['req-valid'].is_successful is True
        assert result_map['req-valid'].response is not None

        # Second result should have validation error
        assert result_map['req-invalid'].is_successful is False
        assert result_map['req-invalid'].error is not None
        assert result_map['req-invalid'].error.code == 'validation_error'


class TestModelRequestBatchPolling:
    """Tests for model_request_batch polling loop coverage."""

    async def test_model_request_batch_without_timeout_polls_until_complete(self, allow_model_requests: None):
        """Test model_request_batch polls correctly when no timeout is set.

        This exercises the branch in direct.py where timeout is None,
        ensuring the polling loop runs without checking timeout condition.
        """
        from pydantic_ai.direct import model_request_batch

        # Track status check calls
        status_check_count = 0

        # Create a custom mock that transitions from in_progress to completed
        class TransitioningMock(MockOpenAIBatch):
            @property
            def batches(self) -> Any:
                return self._TransitioningBatchesNamespace(self)

            @dataclass
            class _TransitioningBatchesNamespace:
                parent: TransitioningMock

                async def create(self, **kwargs: Any) -> OpenAIBatchResponse:
                    self.parent.batch_create_calls.append(kwargs)
                    return OpenAIBatchResponse(
                        id='batch_polling',
                        completion_window='24h',
                        created_at=1704067200,
                        endpoint='/v1/chat/completions',
                        input_file_id=kwargs.get('input_file_id', 'file_xyz'),
                        object='batch',
                        status='in_progress',  # Start as in_progress
                    )

                async def retrieve(self, batch_id: str) -> OpenAIBatchResponse:
                    nonlocal status_check_count
                    status_check_count += 1
                    # Transition to completed on second call
                    if status_check_count >= 2:
                        return OpenAIBatchResponse(
                            id=batch_id,
                            completion_window='24h',
                            created_at=1704067200,
                            endpoint='/v1/chat/completions',
                            input_file_id='file_xyz',
                            object='batch',
                            status='completed',
                            output_file_id='file_output',
                            request_counts=BatchRequestCounts(completed=2, failed=0, total=2),
                        )
                    return OpenAIBatchResponse(
                        id=batch_id,
                        completion_window='24h',
                        created_at=1704067200,
                        endpoint='/v1/chat/completions',
                        input_file_id='file_xyz',
                        object='batch',
                        status='in_progress',
                    )

        mock_client = TransitioningMock()

        # Set up results for when completed
        output_lines = [
            {
                'id': 'resp_1',
                'custom_id': 'req-poll-1',
                'response': {
                    'status_code': 200,
                    'body': {
                        'id': 'chatcmpl-poll-1',
                        'object': 'chat.completion',
                        'created': 1704067200,
                        'model': 'gpt-5-mini',
                        'choices': [
                            {
                                'index': 0,
                                'message': {'role': 'assistant', 'content': 'Polled response 1'},
                                'finish_reason': 'stop',
                            }
                        ],
                    },
                },
            },
            {
                'id': 'resp_2',
                'custom_id': 'req-poll-2',
                'response': {
                    'status_code': 200,
                    'body': {
                        'id': 'chatcmpl-poll-2',
                        'object': 'chat.completion',
                        'created': 1704067200,
                        'model': 'gpt-5-mini',
                        'choices': [
                            {
                                'index': 0,
                                'message': {'role': 'assistant', 'content': 'Polled response 2'},
                                'finish_reason': 'stop',
                            }
                        ],
                    },
                },
            },
        ]
        mock_client.file_content_responses['file_output'] = '\n'.join(json.dumps(line) for line in output_lines)

        model = mock_client.create_model()

        messages: list[ModelMessage] = [ModelRequest.user_text_prompt('Test polling')]
        requests: list[tuple[str, list[ModelMessage]]] = [
            ('req-poll-1', messages),
            ('req-poll-2', messages),
        ]

        # Call without timeout - should poll until complete
        results = await model_request_batch(model, requests, poll_interval=0.01)

        # Verify polling happened
        assert status_check_count >= 2, 'Should have checked status at least twice'

        # Verify results
        assert len(results) == 2
        result_map = {r.custom_id: r for r in results}
        assert result_map['req-poll-1'].is_successful is True
        assert result_map['req-poll-2'].is_successful is True
