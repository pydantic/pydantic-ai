"""Tests for OpenAI batch functionality based on real examples."""

from __future__ import annotations as _annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel

from pydantic_ai import RunContext
from pydantic_ai.batches.openai import (
    BatchJob,
    BatchResult,
    OpenAIBatchModel,
    create_chat_request,
)
from pydantic_ai.tools import Tool


class WeatherResult(BaseModel):
    location: str
    temperature: float
    condition: str
    humidity: int


def get_weather(ctx: RunContext[None], location: str, units: str = 'celsius') -> str:
    """Get current weather information for a location."""
    return f'Weather in {location}: 22°{units[0].upper()}, sunny, 60% humidity'


def calculate(ctx: RunContext[None], expression: str) -> str:
    """Perform mathematical calculations."""
    try:
        result = eval(expression)  # Don't use eval in production!
        return f'Result: {result}'
    except Exception as e:
        return f'Error: {str(e)}'


class TestCreateChatRequest:
    """Test create_chat_request function."""

    def test_basic_chat_request(self):
        """Test creating basic chat request like in batch_crt.py"""
        request = create_chat_request(
            custom_id='math-question', prompt='What is 2+2?', model='gpt-4o-mini', max_tokens=50
        )

        assert request.custom_id == 'math-question'
        assert request.body['model'] == 'gpt-4o-mini'
        assert request.body['max_tokens'] == 50
        assert request.body['messages'] == [{'role': 'user', 'content': 'What is 2+2?'}]

    def test_creative_request_with_temperature(self):
        """Test creative request with temperature like in batch_crt.py"""
        request = create_chat_request(
            custom_id='creative-writing',
            prompt='Write a short poem about coding',
            model='gpt-4o-mini',
            max_tokens=100,
            temperature=0.8,
        )

        assert request.custom_id == 'creative-writing'
        assert request.body['temperature'] == 0.8
        assert request.body['max_tokens'] == 100

    def test_request_with_tools(self):
        """Test request with tools like in batch_tool.py"""
        weather_tool = Tool(get_weather)
        calc_tool = Tool(calculate)
        tools = [weather_tool.tool_def, calc_tool.tool_def]

        request = create_chat_request(
            custom_id='weather-tokyo',
            prompt="What's the weather like in Tokyo?",
            model='gpt-4o-mini',
            tools=tools,
            max_tokens=150,
        )

        assert request.custom_id == 'weather-tokyo'
        assert 'tools' in request.body
        assert len(request.body['tools']) == 2
        assert request.body['tools'][0]['function']['name'] == 'get_weather'

    def test_structured_output_native(self):
        """Test structured output with native mode like in structured_output.py"""
        request = create_chat_request(
            custom_id='structured-paris',
            prompt='Get weather information for Paris',
            model='gpt-4o-mini',
            output_type=WeatherResult,
            output_mode='native',
            max_tokens=200,
        )

        assert request.custom_id == 'structured-paris'
        assert 'response_format' in request.body
        assert request.body['response_format']['type'] == 'json_schema'

    def test_structured_output_tool_mode(self):
        """Test structured output with tool mode like in structured_output.py"""
        request = create_chat_request(
            custom_id='tool-mode-1',
            prompt='Analyze the weather in Berlin',
            model='gpt-4o-mini',
            output_type=WeatherResult,
            output_mode='tool',
            max_tokens=200,
        )

        assert request.custom_id == 'tool-mode-1'
        assert 'tools' in request.body
        assert 'tool_choice' in request.body


@pytest.fixture
def mock_openai_batch_model(monkeypatch: pytest.MonkeyPatch):
    """Create a mocked OpenAI batch model."""
    monkeypatch.setattr('pydantic_ai.models.ALLOW_MODEL_REQUESTS', True)

    with patch('pydantic_ai.providers.openai.AsyncOpenAI') as mock_openai:
        mock_client = MagicMock()
        mock_openai.return_value = mock_client

        model = OpenAIBatchModel('openai:gpt-4o-mini')
        return model, mock_client


class TestBasicBatchWorkflow:
    """Test basic batch workflow like in batch_crt.py examples."""

    async def test_create_basic_batch_job(self, mock_openai_batch_model: Any) -> None:
        """Test creating basic batch job with metadata."""
        batch_model, mock_client = mock_openai_batch_model

        # Mock batch creation
        mock_batch = MagicMock()
        mock_batch.id = 'batch_test_123'
        mock_client.batches.create = AsyncMock(return_value=mock_batch)
        mock_client.files.create = AsyncMock(return_value=MagicMock(id='file_123'))

        # Create requests like in batch_crt.py
        requests = [
            create_chat_request(custom_id='math-question', prompt='What is 2+2?', model='gpt-4o-mini', max_tokens=50),
            create_chat_request(
                custom_id='creative-writing',
                prompt='Write a short poem about coding',
                model='gpt-4o-mini',
                max_tokens=100,
                temperature=0.8,
            ),
            create_chat_request(
                custom_id='explanation',
                prompt='Explain quantum computing in one sentence',
                model='gpt-4o-mini',
                max_tokens=75,
            ),
        ]

        # Submit batch job with metadata
        batch_id = await batch_model.batch_create_job(
            requests=requests, metadata={'project': 'my-batch-job', 'version': '1.0'}
        )

        assert batch_id == 'batch_test_123'
        mock_client.files.create.assert_called_once()
        mock_client.batches.create.assert_called_once()

        # Check metadata was passed
        call_args = mock_client.batches.create.call_args
        assert call_args.kwargs['metadata'] == {'project': 'my-batch-job', 'version': '1.0'}

    async def test_batch_status_completed(self, mock_openai_batch_model: Any) -> None:
        """Test getting batch status when completed like in check_status.py output."""
        batch_model, mock_client = mock_openai_batch_model

        # Mock completed batch status
        mock_batch = MagicMock()
        mock_batch.id = 'batch_6887c4d206c88190b4e06c4f3867e8ca'
        mock_batch.object = 'batch'
        mock_batch.endpoint = '/v1/chat/completions'
        mock_batch.errors = None
        mock_batch.input_file_id = 'file-Jfyv6WaoGG1i8ukdEL7cJz'
        mock_batch.completion_window = '24h'
        mock_batch.status = 'completed'
        mock_batch.output_file_id = 'file-AiJT2XheqDiVYc5a1FA8Y8'
        mock_batch.error_file_id = None
        mock_batch.created_at = 1753728210
        mock_batch.in_progress_at = 1753728210
        mock_batch.expires_at = 1753814610
        mock_batch.finalizing_at = 1753728292
        mock_batch.completed_at = 1753728293
        mock_batch.failed_at = None
        mock_batch.expired_at = None
        mock_batch.cancelling_at = None
        mock_batch.cancelled_at = None
        mock_batch.metadata = {}

        # Mock request_counts
        mock_request_counts = MagicMock()
        mock_request_counts.model_dump.return_value = {'completed': 3, 'failed': 0, 'total': 3}
        mock_batch.request_counts = mock_request_counts

        mock_client.batches.retrieve = AsyncMock(return_value=mock_batch)

        # Get status
        job_info = await batch_model.batch_get_status('batch_6887c4d206c88190b4e06c4f3867e8ca')

        assert isinstance(job_info, BatchJob)
        assert job_info.id == 'batch_6887c4d206c88190b4e06c4f3867e8ca'
        assert job_info.status == 'completed'
        assert job_info.completion_window == '24h'
        assert job_info.request_counts == {'completed': 3, 'failed': 0, 'total': 3}
        assert job_info.created_at == 1753728210

    async def test_retrieve_completed_results(self, mock_openai_batch_model: Any) -> None:
        """Test retrieving completed batch results like in get_res.py output."""
        batch_model, mock_client = mock_openai_batch_model

        # Mock batch status (completed)
        mock_batch = MagicMock()
        mock_batch.status = 'completed'
        mock_batch.output_file_id = 'file-AiJT2XheqDiVYc5a1FA8Y8'
        mock_client.batches.retrieve = AsyncMock(return_value=mock_batch)

        # Mock file content with real response format from get_res.py
        mock_file_content = b"""{"id": "batch_req_1", "custom_id": "request-1", "response": {"body": {"choices": [{"message": {"content": "2 + 2 = 4"}}]}}, "error": null}
{"id": "batch_req_2", "custom_id": "request-2", "response": {"body": {"choices": [{"message": {"content": "In lines of code, a world takes flight,\\nFrom dawn's first thought to late at night."}}]}}, "error": null}
{"id": "batch_req_3", "custom_id": "request-3", "response": {"body": {"choices": [{"message": {"content": "Quantum computing is a type of computing that uses quantum bits, or qubits, leveraging principles like superposition and entanglement to perform complex calculations exponentially faster than classical computers for certain problems."}}]}}, "error": null}"""

        mock_file_response = MagicMock()
        mock_file_response.read.return_value = mock_file_content
        mock_client.files.content = AsyncMock(return_value=mock_file_response)

        # Retrieve results
        results = await batch_model.batch_retrieve_job('batch_test_123')

        assert len(results) == 3
        assert all(isinstance(r, BatchResult) for r in results)

        # Check first result (math question)
        assert results[0].custom_id == 'request-1'
        assert results[0].output == '2 + 2 = 4'
        assert results[0].error is None

        # Check second result (creative writing)
        assert results[1].custom_id == 'request-2'
        assert 'In lines of code, a world takes flight' in results[1].output
        assert results[1].error is None

        # Check third result (explanation)
        assert results[2].custom_id == 'request-3'
        assert 'Quantum computing' in results[2].output
        assert results[2].error is None


class TestBatchWithTools:
    """Test batch functionality with tools like in batch_tool.py and get_tool.py."""

    async def test_batch_with_tool_calls(self, mock_openai_batch_model: Any) -> None:
        """Test batch job with tool calls like in get_tool.py output."""
        batch_model, mock_client = mock_openai_batch_model

        # Mock batch creation
        mock_batch = MagicMock()
        mock_batch.id = 'batch_tools_123'
        mock_client.batches.create = AsyncMock(return_value=mock_batch)
        mock_client.files.create = AsyncMock(return_value=MagicMock(id='file_123'))

        # Mock completed batch status
        mock_batch_status = MagicMock()
        mock_batch_status.status = 'completed'
        mock_batch_status.output_file_id = 'file_456'
        mock_client.batches.retrieve = AsyncMock(return_value=mock_batch_status)

        # Mock file content with tool calls like in get_tool.py output
        mock_file_content = b"""{"id": "batch_req_1", "custom_id": "weather-tokyo", "response": {"body": {"choices": [{"message": {"tool_calls": [{"function": {"name": "get_weather", "arguments": "{\\"location\\": \\"Tokyo\\"}"}}]}}]}}, "error": null}
{"id": "batch_req_2", "custom_id": "calculation", "response": {"body": {"choices": [{"message": {"tool_calls": [{"function": {"name": "calculate", "arguments": "{\\"expression\\": \\"15 * 23 + 7\\"}"}}]}}]}}, "error": null}
{"id": "batch_req_3", "custom_id": "weather-london", "response": {"body": {"choices": [{"message": {"tool_calls": [{"function": {"name": "get_weather", "arguments": "{\\"location\\": \\"London\\"}"}}]}}]}}, "error": null}"""

        mock_file_response = MagicMock()
        mock_file_response.read.return_value = mock_file_content
        mock_client.files.content = AsyncMock(return_value=mock_file_response)

        # Create tools
        weather_tool = Tool(get_weather)
        calc_tool = Tool(calculate)
        tools = [weather_tool.tool_def, calc_tool.tool_def]

        # Create requests like in batch_tool.py
        requests = [
            create_chat_request(
                custom_id='weather-tokyo',
                prompt="What's the weather like in Tokyo?",
                model='gpt-4o-mini',
                tools=tools,
                max_tokens=150,
            ),
            create_chat_request(
                custom_id='calculation',
                prompt='Calculate 15 * 23 + 7',
                model='gpt-4o-mini',
                tools=tools,
                max_tokens=100,
            ),
            create_chat_request(
                custom_id='weather-london',
                prompt="What's the weather in London?",
                model='gpt-4o-mini',
                tools=tools,
                max_tokens=150,
            ),
        ]

        # Submit batch
        batch_id = await batch_model.batch_create_job(requests)
        assert batch_id == 'batch_tools_123'

        # Retrieve results
        results = await batch_model.batch_retrieve_job(batch_id)
        assert len(results) == 3

        # Check tool calls like in get_tool.py output
        tokyo_result = next(r for r in results if r.custom_id == 'weather-tokyo')
        assert len(tokyo_result.tool_calls) == 1
        assert tokyo_result.tool_calls[0]['function']['name'] == 'get_weather'

        calc_result = next(r for r in results if r.custom_id == 'calculation')
        assert len(calc_result.tool_calls) == 1
        assert calc_result.tool_calls[0]['function']['name'] == 'calculate'

        london_result = next(r for r in results if r.custom_id == 'weather-london')
        assert len(london_result.tool_calls) == 1
        assert london_result.tool_calls[0]['function']['name'] == 'get_weather'


class TestStructuredOutput:
    """Test structured output like in structured_output.py."""

    async def test_batch_with_structured_output(self, mock_openai_batch_model: Any) -> None:
        """Test batch with structured output modes."""
        batch_model, mock_client = mock_openai_batch_model

        # Mock batch creation
        mock_batch = MagicMock()
        mock_batch.id = 'batch_struct_123'
        mock_client.batches.create = AsyncMock(return_value=mock_batch)
        mock_client.files.create = AsyncMock(return_value=MagicMock(id='file_123'))

        # Create requests like in structured_output.py
        requests = [
            create_chat_request(
                custom_id='structured-paris',
                prompt='Get weather information for Paris and format it properly',
                model='gpt-4o-mini',
                output_type=WeatherResult,
                output_mode='native',
                max_tokens=200,
            ),
            create_chat_request(
                custom_id='tool-mode-1',
                prompt='Analyze the weather in Berlin',
                model='gpt-4o-mini',
                output_type=WeatherResult,
                output_mode='tool',
                max_tokens=200,
            ),
        ]

        # Submit batch
        batch_id = await batch_model.batch_create_job(requests)
        assert batch_id == 'batch_struct_123'

        # Verify requests were structured correctly
        create_call = mock_client.batches.create.call_args
        assert create_call is not None


class TestErrorHandling:
    """Test error handling scenarios."""

    async def test_batch_create_job_api_error(self, mock_openai_batch_model: Any) -> None:
        """Test error handling when batch_create_job API call fails."""
        from pydantic_ai import ModelHTTPError
        from pydantic_ai.models.openai import APIStatusError

        batch_model, mock_client = mock_openai_batch_model

        # Mock API error during file creation
        mock_client.files.create = AsyncMock(
            side_effect=APIStatusError(
                message='File upload failed',
                response=MagicMock(status_code=400),
                body={'error': {'message': 'File upload failed'}},
            )
        )

        requests = [create_chat_request(custom_id='test-req', prompt='Hello', model='gpt-4o-mini', max_tokens=50)]

        with pytest.raises(ModelHTTPError) as exc_info:
            await batch_model.batch_create_job(requests)

        assert exc_info.value.status_code == 400

    async def test_batch_create_job_batch_api_error(self, mock_openai_batch_model: Any) -> None:
        """Test error handling when batch creation API call fails."""
        from pydantic_ai import ModelHTTPError
        from pydantic_ai.models.openai import APIStatusError

        batch_model, mock_client = mock_openai_batch_model

        # Mock successful file creation but failed batch creation
        mock_client.files.create = AsyncMock(return_value=MagicMock(id='file_123'))
        mock_client.batches.create = AsyncMock(
            side_effect=APIStatusError(
                message='Batch creation failed',
                response=MagicMock(status_code=429),
                body={'error': {'message': 'Rate limit exceeded'}},
            )
        )

        requests = [create_chat_request(custom_id='test-req', prompt='Hello', model='gpt-4o-mini', max_tokens=50)]

        with pytest.raises(ModelHTTPError) as exc_info:
            await batch_model.batch_create_job(requests)

        assert exc_info.value.status_code == 429

    async def test_batch_get_status_api_error(self, mock_openai_batch_model: Any) -> None:
        """Test error handling when batch_get_status API call fails."""
        from pydantic_ai import ModelHTTPError
        from pydantic_ai.models.openai import APIStatusError

        batch_model, mock_client = mock_openai_batch_model

        mock_client.batches.retrieve = AsyncMock(
            side_effect=APIStatusError(
                message='Batch not found',
                response=MagicMock(status_code=404),
                body={'error': {'message': 'Batch not found'}},
            )
        )

        with pytest.raises(ModelHTTPError) as exc_info:
            await batch_model.batch_get_status('nonexistent_batch')

        assert exc_info.value.status_code == 404

    async def test_batch_retrieve_job_not_completed(self, mock_openai_batch_model: Any) -> None:
        """Test error when trying to retrieve results from non-completed batch."""
        batch_model, mock_client = mock_openai_batch_model

        # Mock batch status as in_progress
        mock_batch = MagicMock()
        mock_batch.status = 'in_progress'
        mock_batch.output_file_id = None
        mock_client.batches.retrieve = AsyncMock(return_value=mock_batch)

        with pytest.raises(ValueError, match=r'Batch .* is not completed. Status: in_progress'):
            await batch_model.batch_retrieve_job('batch_123')

    async def test_batch_retrieve_job_no_output_file(self, mock_openai_batch_model: Any) -> None:
        """Test error when completed batch has no output file."""
        batch_model, mock_client = mock_openai_batch_model

        # Mock batch status as completed but no output file
        mock_batch = MagicMock()
        mock_batch.status = 'completed'
        mock_batch.output_file_id = None
        mock_client.batches.retrieve = AsyncMock(return_value=mock_batch)

        with pytest.raises(ValueError, match=r'Batch .* has no output file'):
            await batch_model.batch_retrieve_job('batch_123')

    async def test_batch_retrieve_job_file_api_error(self, mock_openai_batch_model: Any) -> None:
        """Test error handling when file content retrieval fails."""
        from pydantic_ai import ModelHTTPError
        from pydantic_ai.models.openai import APIStatusError

        batch_model, mock_client = mock_openai_batch_model

        # Mock successful batch status check
        mock_batch = MagicMock()
        mock_batch.status = 'completed'
        mock_batch.output_file_id = 'file_123'
        mock_client.batches.retrieve = AsyncMock(return_value=mock_batch)

        # Mock file content retrieval error
        mock_client.files.content = AsyncMock(
            side_effect=APIStatusError(
                message='File not found',
                response=MagicMock(status_code=404),
                body={'error': {'message': 'File not found'}},
            )
        )

        with pytest.raises(ModelHTTPError) as exc_info:
            await batch_model.batch_retrieve_job('batch_123')

        assert exc_info.value.status_code == 404


class TestBatchCancelAndList:
    """Test batch cancel and list functionality."""

    async def test_batch_cancel_job(self, mock_openai_batch_model: Any) -> None:
        """Test cancelling a batch job."""
        batch_model, mock_client = mock_openai_batch_model

        # Mock cancelled batch
        mock_batch = MagicMock()
        mock_batch.id = 'batch_cancel_123'
        mock_batch.object = 'batch'
        mock_batch.endpoint = '/v1/chat/completions'
        mock_batch.errors = None
        mock_batch.input_file_id = 'file-123'
        mock_batch.completion_window = '24h'
        mock_batch.status = 'cancelled'
        mock_batch.output_file_id = None
        mock_batch.error_file_id = None
        mock_batch.created_at = 1753728210
        mock_batch.in_progress_at = None
        mock_batch.expires_at = 1753814610
        mock_batch.finalizing_at = None
        mock_batch.completed_at = None
        mock_batch.failed_at = None
        mock_batch.expired_at = None
        mock_batch.cancelling_at = 1753728250
        mock_batch.cancelled_at = 1753728260
        mock_batch.request_counts = None
        mock_batch.metadata = {}

        mock_client.batches.cancel = AsyncMock(return_value=mock_batch)

        result = await batch_model.batch_cancel_job('batch_cancel_123')

        assert isinstance(result, BatchJob)
        assert result.id == 'batch_cancel_123'
        assert result.status == 'cancelled'
        assert result.cancelling_at == 1753728250
        assert result.cancelled_at == 1753728260
        mock_client.batches.cancel.assert_called_once_with('batch_cancel_123')

    async def test_batch_cancel_job_api_error(self, mock_openai_batch_model: Any) -> None:
        """Test error handling when batch cancellation fails."""
        from pydantic_ai import ModelHTTPError
        from pydantic_ai.models.openai import APIStatusError

        batch_model, mock_client = mock_openai_batch_model

        mock_client.batches.cancel = AsyncMock(
            side_effect=APIStatusError(
                message='Cannot cancel completed batch',
                response=MagicMock(status_code=400),
                body={'error': {'message': 'Cannot cancel completed batch'}},
            )
        )

        with pytest.raises(ModelHTTPError) as exc_info:
            await batch_model.batch_cancel_job('batch_123')

        assert exc_info.value.status_code == 400

    async def test_batch_list_jobs(self, mock_openai_batch_model: Any) -> None:
        """Test listing batch jobs."""
        batch_model, mock_client = mock_openai_batch_model

        # Mock batch list response
        mock_batch1 = MagicMock()
        mock_batch1.id = 'batch_1'
        mock_batch1.object = 'batch'
        mock_batch1.endpoint = '/v1/chat/completions'
        mock_batch1.errors = None
        mock_batch1.input_file_id = 'file-1'
        mock_batch1.completion_window = '24h'
        mock_batch1.status = 'completed'
        mock_batch1.output_file_id = 'file-out-1'
        mock_batch1.error_file_id = None
        mock_batch1.created_at = 1753728210
        mock_batch1.in_progress_at = 1753728210
        mock_batch1.expires_at = 1753814610
        mock_batch1.finalizing_at = 1753728292
        mock_batch1.completed_at = 1753728293
        mock_batch1.failed_at = None
        mock_batch1.expired_at = None
        mock_batch1.cancelling_at = None
        mock_batch1.cancelled_at = None
        mock_batch1.request_counts = None
        mock_batch1.metadata = {}

        mock_batch2 = MagicMock()
        mock_batch2.id = 'batch_2'
        mock_batch2.object = 'batch'
        mock_batch2.endpoint = '/v1/chat/completions'
        mock_batch2.errors = None
        mock_batch2.input_file_id = 'file-2'
        mock_batch2.completion_window = '24h'
        mock_batch2.status = 'in_progress'
        mock_batch2.output_file_id = None
        mock_batch2.error_file_id = None
        mock_batch2.created_at = 1753728300
        mock_batch2.in_progress_at = 1753728300
        mock_batch2.expires_at = 1753814700
        mock_batch2.finalizing_at = None
        mock_batch2.completed_at = None
        mock_batch2.failed_at = None
        mock_batch2.expired_at = None
        mock_batch2.cancelling_at = None
        mock_batch2.cancelled_at = None
        mock_batch2.request_counts = None
        mock_batch2.metadata = {}

        mock_list_response = MagicMock()
        mock_list_response.data = [mock_batch1, mock_batch2]
        mock_client.batches.list = AsyncMock(return_value=mock_list_response)

        results = await batch_model.batch_list_jobs(limit=10)

        assert len(results) == 2
        assert all(isinstance(job, BatchJob) for job in results)
        assert results[0].id == 'batch_1'
        assert results[0].status == 'completed'
        assert results[1].id == 'batch_2'
        assert results[1].status == 'in_progress'
        mock_client.batches.list.assert_called_once_with(limit=10)

    async def test_batch_list_jobs_api_error(self, mock_openai_batch_model: Any) -> None:
        """Test error handling when batch list fails."""
        from pydantic_ai import ModelHTTPError
        from pydantic_ai.models.openai import APIStatusError

        batch_model, mock_client = mock_openai_batch_model

        mock_client.batches.list = AsyncMock(
            side_effect=APIStatusError(
                message='Unauthorized',
                response=MagicMock(status_code=403),
                body={'error': {'message': 'Unauthorized'}},
            )
        )

        with pytest.raises(ModelHTTPError) as exc_info:
            await batch_model.batch_list_jobs()

        assert exc_info.value.status_code == 403


class TestBatchResultMethods:
    """Test BatchResult methods."""

    def test_batch_result_get_tool_call_arguments_success(self):
        """Test successful tool call argument parsing."""
        result = BatchResult(
            id='test_id',
            custom_id='test_custom',
            response={
                'body': {
                    'choices': [
                        {
                            'message': {
                                'tool_calls': [
                                    {
                                        'function': {
                                            'name': 'test_function',
                                            'arguments': '{"location": "Tokyo", "units": "celsius"}',
                                        }
                                    }
                                ]
                            }
                        }
                    ]
                }
            },
            error=None,
        )

        args = result.get_tool_call_arguments()
        assert args == {'location': 'Tokyo', 'units': 'celsius'}

    def test_batch_result_get_tool_call_arguments_specific_index(self):
        """Test tool call argument parsing with specific index."""
        result = BatchResult(
            id='test_id',
            custom_id='test_custom',
            response={
                'body': {
                    'choices': [
                        {
                            'message': {
                                'tool_calls': [
                                    {
                                        'function': {
                                            'name': 'first_function',
                                            'arguments': '{"arg1": "value1"}',
                                        }
                                    },
                                    {
                                        'function': {
                                            'name': 'second_function',
                                            'arguments': '{"arg2": "value2"}',
                                        }
                                    },
                                ]
                            }
                        }
                    ]
                }
            },
            error=None,
        )

        # Test first tool call (index 0)
        args0 = result.get_tool_call_arguments(0)
        assert args0 == {'arg1': 'value1'}

        # Test second tool call (index 1)
        args1 = result.get_tool_call_arguments(1)
        assert args1 == {'arg2': 'value2'}

    def test_batch_result_get_tool_call_arguments_no_response(self):
        """Test tool call argument parsing when no response."""
        result = BatchResult(id='test_id', custom_id='test_custom', response=None, error=None)

        args = result.get_tool_call_arguments()
        assert args is None

    def test_batch_result_get_tool_call_arguments_no_tool_calls(self):
        """Test tool call argument parsing when no tool calls."""
        result = BatchResult(
            id='test_id',
            custom_id='test_custom',
            response={'body': {'choices': [{'message': {'content': 'Hello'}}]}},
            error=None,
        )

        args = result.get_tool_call_arguments()
        assert args is None

    def test_batch_result_get_tool_call_arguments_index_out_of_range(self):
        """Test tool call argument parsing with out of range index."""
        result = BatchResult(
            id='test_id',
            custom_id='test_custom',
            response={
                'body': {
                    'choices': [
                        {'message': {'tool_calls': [{'function': {'name': 'test', 'arguments': '{"arg": "value"}'}}]}}
                    ]
                }
            },
            error=None,
        )

        args = result.get_tool_call_arguments(5)  # Index 5 doesn't exist
        assert args is None

    def test_batch_result_get_tool_call_arguments_invalid_json(self):
        """Test tool call argument parsing with invalid JSON."""
        result = BatchResult(
            id='test_id',
            custom_id='test_custom',
            response={
                'body': {
                    'choices': [
                        {
                            'message': {
                                'tool_calls': [
                                    {
                                        'function': {
                                            'name': 'test_function',
                                            'arguments': 'invalid json {',
                                        }
                                    }
                                ]
                            }
                        }
                    ]
                }
            },
            error=None,
        )

        args = result.get_tool_call_arguments()
        assert args is None

    def test_batch_result_get_tool_call_arguments_malformed_response(self):
        """Test tool call argument parsing with malformed response structure."""
        result = BatchResult(
            id='test_id',
            custom_id='test_custom',
            response={'body': {'choices': [{'message': {}}]}},  # Missing tool_calls
            error=None,
        )

        args = result.get_tool_call_arguments()
        assert args is None


class TestMessageBuilding:
    """Test message building functionality through create_chat_request."""

    def test_create_chat_request_with_string_prompt(self):
        """Test create_chat_request with simple string prompt."""
        request = create_chat_request(custom_id='string-test', prompt='Hello world', model='gpt-4o-mini', max_tokens=50)

        assert request.custom_id == 'string-test'
        assert request.body['messages'] == [{'role': 'user', 'content': 'Hello world'}]

    def test_create_chat_request_with_system_prompt(self):
        """Test create_chat_request with system prompt."""
        request = create_chat_request(
            custom_id='system-test',
            prompt='Hello world',
            model='gpt-4o-mini',
            system_prompt='You are a helpful assistant',
            max_tokens=50,
        )

        expected_messages = [
            {'role': 'system', 'content': 'You are a helpful assistant'},
            {'role': 'user', 'content': 'Hello world'},
        ]
        assert request.body['messages'] == expected_messages

    def test_create_chat_request_with_user_prompt_part(self):
        """Test create_chat_request with UserPromptPart."""
        from pydantic_ai.messages import UserPromptPart

        user_part = UserPromptPart(content='Hello from UserPromptPart')
        request = create_chat_request(custom_id='user-part-test', prompt=user_part, model='gpt-4o-mini', max_tokens=50)

        assert request.custom_id == 'user-part-test'
        # The function should handle UserPromptPart conversion
        assert len(request.body['messages']) == 1
        assert request.body['messages'][0]['role'] == 'user'

    def test_create_chat_request_with_user_prompt_part_list(self):
        """Test create_chat_request with list of UserPromptPart."""
        from pydantic_ai.messages import UserPromptPart

        user_parts = [
            UserPromptPart(content='First part'),
            UserPromptPart(content='Second part'),
            UserPromptPart(content='Third part'),
        ]
        request = create_chat_request(
            custom_id='user-parts-list-test', prompt=user_parts, model='gpt-4o-mini', max_tokens=50
        )

        assert request.custom_id == 'user-parts-list-test'
        # The function should handle list of UserPromptParts
        assert len(request.body['messages']) == 1
        assert request.body['messages'][0]['role'] == 'user'

    def test_create_chat_request_with_complex_user_content(self):
        """Test create_chat_request with UserPromptPart containing complex content."""
        from pydantic_ai.messages import BinaryContent, UserPromptPart

        # Create a UserPromptPart with mixed content
        binary_content = BinaryContent(data=b'fake image data', media_type='image/jpeg')
        user_part = UserPromptPart(content=['Text content', binary_content])

        request = create_chat_request(
            custom_id='complex-content-test', prompt=user_part, model='gpt-4o-mini', max_tokens=50
        )

        assert request.custom_id == 'complex-content-test'
        # The function should handle complex UserPromptPart conversion
        assert len(request.body['messages']) == 1
        assert request.body['messages'][0]['role'] == 'user'


class TestPromptedOutputMode:
    """Test prompted output mode functionality."""

    def test_structured_output_prompted_mode(self):
        """Test structured output with prompted mode."""
        request = create_chat_request(
            custom_id='prompted-test',
            prompt='Get weather for Berlin',
            model='gpt-4o-mini',
            output_type=WeatherResult,
            output_mode='prompted',
            max_tokens=200,
        )

        assert request.custom_id == 'prompted-test'
        assert 'response_format' not in request.body  # No response_format for prompted mode
        assert 'tools' not in request.body  # No tools for prompted mode

        # Check that system prompt was enhanced with schema instructions
        messages = request.body['messages']
        system_message = next((msg for msg in messages if msg['role'] == 'system'), None)
        assert system_message is not None
        assert 'JSON' in system_message['content']
        assert 'schema' in system_message['content']

    def test_structured_output_prompted_mode_with_existing_system_prompt(self):
        """Test prompted mode with existing system prompt."""
        request = create_chat_request(
            custom_id='prompted-with-system',
            prompt='Get weather for Berlin',
            model='gpt-4o-mini',
            system_prompt='You are a weather expert.',
            output_type=WeatherResult,
            output_mode='prompted',
            max_tokens=200,
        )

        messages = request.body['messages']
        system_message = next((msg for msg in messages if msg['role'] == 'system'), None)
        assert system_message is not None

        # Should contain both original system prompt and schema instructions
        content = system_message['content']
        assert 'You are a weather expert.' in content
        assert 'JSON' in content
        assert 'schema' in content


class TestModelInitialization:
    """Test OpenAIBatchModel initialization."""

    def test_openai_batch_model_with_non_openai_model(self, monkeypatch: pytest.MonkeyPatch):
        """Test error when initializing with non-OpenAI model."""
        monkeypatch.setattr('pydantic_ai.models.ALLOW_MODEL_REQUESTS', True)

        # Mock infer_model in wrapper module to return a non-OpenAI model
        from pydantic_ai.models.anthropic import AnthropicModel

        mock_anthropic_model = MagicMock(spec=AnthropicModel)
        monkeypatch.setattr('pydantic_ai.models.wrapper.infer_model', MagicMock(return_value=mock_anthropic_model))

        # This should raise ValueError since it's not an OpenAI model
        with pytest.raises(ValueError, match=r'OpenAIBatchModel requires an OpenAI model'):
            from typing import cast

            from pydantic_ai.models import KnownModelName

            OpenAIBatchModel(cast(KnownModelName, 'any-model'))

    def test_openai_batch_model_with_openai_model_string(self, monkeypatch: pytest.MonkeyPatch):
        """Test successful initialization with OpenAI model string."""
        monkeypatch.setattr('pydantic_ai.models.ALLOW_MODEL_REQUESTS', True)

        with patch('pydantic_ai.providers.openai.AsyncOpenAI') as mock_openai:
            mock_client = MagicMock()
            mock_openai.return_value = mock_client

            batch_model = OpenAIBatchModel('openai:gpt-4o-mini')
            # The model_name property returns just the model part, not the provider prefix
            assert batch_model.model_name == 'gpt-4o-mini'

    def test_openai_batch_model_client_property(self, monkeypatch: pytest.MonkeyPatch):
        """Test that client property returns AsyncOpenAI client."""
        monkeypatch.setattr('pydantic_ai.models.ALLOW_MODEL_REQUESTS', True)

        with patch('pydantic_ai.providers.openai.AsyncOpenAI') as mock_openai:
            mock_client = MagicMock()
            mock_client.batches = MagicMock()
            mock_client.files = MagicMock()
            mock_openai.return_value = mock_client

            batch_model = OpenAIBatchModel('openai:gpt-4o-mini')
            client = batch_model.client
            # The client should be an AsyncOpenAI instance
            assert hasattr(client, 'batches')
            assert hasattr(client, 'files')


class TestAdditionalErrorScenarios:
    """Test additional error scenarios and edge cases."""

    def test_batch_result_output_property_edge_cases(self):
        """Test BatchResult.output property with various edge cases."""
        # Test with missing choices
        result_no_choices = BatchResult(
            id='test_id',
            custom_id='test_custom',
            response={'body': {}},
            error=None,
        )
        assert result_no_choices.output is None

        # Test with empty choices
        result_empty_choices = BatchResult(
            id='test_id',
            custom_id='test_custom',
            response={'body': {'choices': []}},
            error=None,
        )
        assert result_empty_choices.output is None

        # Test with missing message
        result_no_message = BatchResult(
            id='test_id',
            custom_id='test_custom',
            response={'body': {'choices': [{}]}},
            error=None,
        )
        assert result_no_message.output is None

    def test_batch_result_tool_calls_property_edge_cases(self):
        """Test BatchResult.tool_calls property with various edge cases."""
        # Test with missing choices
        result_no_choices = BatchResult(
            id='test_id',
            custom_id='test_custom',
            response={'body': {}},
            error=None,
        )
        assert result_no_choices.tool_calls == []

        # Test with empty choices
        result_empty_choices = BatchResult(
            id='test_id',
            custom_id='test_custom',
            response={'body': {'choices': []}},
            error=None,
        )
        assert result_empty_choices.tool_calls == []

        # Test with missing message
        result_no_message = BatchResult(
            id='test_id',
            custom_id='test_custom',
            response={'body': {'choices': [{}]}},
            error=None,
        )
        assert result_no_message.tool_calls == []

        # Test with message but no tool_calls
        result_no_tool_calls = BatchResult(
            id='test_id',
            custom_id='test_custom',
            response={'body': {'choices': [{'message': {'content': 'Hello'}}]}},
            error=None,
        )
        assert result_no_tool_calls.tool_calls == []

    def test_create_chat_request_default_output_mode(self):
        """Test that default output mode is 'tool' when output_type is provided."""
        request = create_chat_request(
            custom_id='default-mode-test',
            prompt='Get weather data',
            model='gpt-4o-mini',
            output_type=WeatherResult,  # No output_mode specified
        )

        # Should default to 'tool' mode
        assert 'tools' in request.body
        assert 'tool_choice' in request.body

    def test_native_output_with_description_and_strict(self):
        """Test native output mode with description and strict=True."""
        from unittest.mock import MagicMock

        from pydantic import BaseModel

        from pydantic_ai._output import StructuredTextOutputSchema

        # Mock the StructuredTextOutputSchema to have description and strict=True
        mock_object_def = MagicMock()
        mock_object_def.name = 'TestResult'
        mock_object_def.description = 'A test result with description'
        mock_object_def.strict = True
        mock_object_def.json_schema = {'type': 'object', 'properties': {'value': {'type': 'string'}}}

        with patch('pydantic_ai._output.OutputSchema.build') as mock_build:
            mock_schema = MagicMock(spec=StructuredTextOutputSchema)
            mock_schema.mode = 'native'
            mock_schema.object_def = mock_object_def
            mock_build.return_value = mock_schema

            request = create_chat_request(
                custom_id='strict-test',
                prompt='Get strict result',
                model='gpt-4o-mini',
                output_type=BaseModel,  # Using BaseModel directly
                output_mode='native',
            )

            # Check that response_format is set with description and strict
            assert 'response_format' in request.body
            response_format = request.body['response_format']
            assert response_format['type'] == 'json_schema'
            json_schema = response_format['json_schema']

            # This should trigger the description and strict lines (75, 77)
            assert json_schema['name'] == 'TestResult'
            assert json_schema['description'] == 'A test result with description'
            assert json_schema['strict'] is True
            assert 'schema' in json_schema

    def test_tool_output_with_existing_tools(self):
        """Test tool output mode when body already has tools."""
        from pydantic_ai.tools import ToolDefinition

        # Create a mock tool definition
        existing_tool = ToolDefinition(
            name='existing_tool',
            description='An existing tool',
            parameters_json_schema={'type': 'object', 'properties': {}},
        )

        request = create_chat_request(
            custom_id='tools-test',
            prompt='Use tools',
            model='gpt-4o-mini',
            tools=[existing_tool],  # Add existing tools first
            output_type=WeatherResult,
            output_mode='tool',
        )

        # Should have both existing tools and output tools
        assert 'tools' in request.body
        tools = request.body['tools']
        assert len(tools) >= 2  # At least existing + output tool

    def test_batch_result_output_none_response(self):
        """Test BatchResult.output when response is None."""
        result = BatchResult(
            id='test-id',
            custom_id='test-custom',
            response=None,  # This should trigger line 275
            error=None,
        )

        # Should return None when response is None
        assert result.output is None


class TestHelperFunctions:
    """Test helper functions to achieve 100% coverage."""

    def test_get_weather_function(self) -> None:
        """Test get_weather function directly."""
        from unittest.mock import Mock

        ctx = Mock()
        result = get_weather(ctx, 'Paris')
        assert result == 'Weather in Paris: 22°C, sunny, 60% humidity'

        result_fahrenheit = get_weather(ctx, 'New York', 'fahrenheit')
        assert result_fahrenheit == 'Weather in New York: 22°F, sunny, 60% humidity'

    def test_calculate_function(self) -> None:
        """Test calculate function directly."""
        from unittest.mock import Mock

        ctx = Mock()

        # Test successful calculation
        result = calculate(ctx, '2 + 3')
        assert result == 'Result: 5'

        # Test error case
        result_error = calculate(ctx, 'invalid_expression')
        assert result_error.startswith('Error:')


class TestBranchCoverage:
    """Test specific branches to achieve 100% coverage."""

    def test_file_content_with_empty_lines(self, mock_openai_batch_model: Any):
        """Test batch_retrieve_job with empty lines in file content (line 486->485)."""
        batch_model, mock_client = mock_openai_batch_model

        # Mock batch status (completed)
        mock_batch = MagicMock()
        mock_batch.status = 'completed'
        mock_batch.output_file_id = 'file_456'
        mock_client.batches.retrieve = AsyncMock(return_value=mock_batch)

        # Mock file content with empty lines and whitespace
        mock_file_content = b"""{"id": "batch_req_1", "custom_id": "test-1", "response": {"body": {"choices": [{"message": {"content": "Hello"}}]}}, "error": null}

{"id": "batch_req_2", "custom_id": "test-2", "response": {"body": {"choices": [{"message": {"content": "World"}}]}}, "error": null}

"""  # Empty lines and whitespace

        mock_file_response = MagicMock()
        mock_file_response.read.return_value = mock_file_content
        mock_client.files.content = AsyncMock(return_value=mock_file_response)

        async def run_test():
            results = await batch_model.batch_retrieve_job('batch_test_123')
            # Should only have 2 results, empty lines should be skipped
            assert len(results) == 2
            assert results[0].custom_id == 'test-1'
            assert results[1].custom_id == 'test-2'

        import asyncio

        asyncio.run(run_test())

    def test_create_request_with_non_structured_output_type(self):
        """Test native output with non-StructuredTextOutputSchema."""
        from pydantic import BaseModel

        class SimpleModel(BaseModel):
            value: str

        # Mock the OutputSchema.build to return a schema that's not StructuredTextOutputSchema
        with patch('pydantic_ai._output.OutputSchema.build') as mock_build:
            mock_schema = MagicMock()
            mock_schema.mode = 'native'
            # This will make isinstance(output_schema, StructuredTextOutputSchema) return False
            mock_schema.__class__.__name__ = 'NotStructuredTextOutputSchema'
            mock_build.return_value = mock_schema

            request = create_chat_request(
                custom_id='non-structured-test',
                prompt='Test request',
                model='gpt-4o-mini',
                output_type=SimpleModel,
                output_mode='native',
            )

            # Should not have response_format since it's not StructuredTextOutputSchema
            assert 'response_format' not in request.body

    def test_create_request_with_tool_mode_multiple_output_tools(self):
        """Test tool mode with multiple output tools (no tool_choice)."""
        from pydantic import BaseModel

        class MultiToolModel(BaseModel):
            result1: str
            result2: str

        # Mock the OutputSchema.build to return a schema with multiple tools
        with patch('pydantic_ai._output.OutputSchema.build') as mock_build:
            mock_schema = MagicMock()
            mock_schema.mode = 'tool'

            # Mock toolset with multiple tools
            mock_tool_def1 = MagicMock()
            mock_tool_def1.name = 'tool1'
            mock_tool_def1.description = 'First tool'
            mock_tool_def1.parameters_json_schema = {'type': 'object'}

            mock_tool_def2 = MagicMock()
            mock_tool_def2.name = 'tool2'
            mock_tool_def2.description = 'Second tool'
            mock_tool_def2.parameters_json_schema = {'type': 'object'}

            mock_toolset = MagicMock()
            mock_toolset._tool_defs = [mock_tool_def1, mock_tool_def2]
            mock_schema.toolset = mock_toolset
            mock_build.return_value = mock_schema

            request = create_chat_request(
                custom_id='multi-tool-test',
                prompt='Test request',
                model='gpt-4o-mini',
                output_type=MultiToolModel,
                output_mode='tool',
            )

            # Should have tools but no tool_choice since there are multiple tools
            assert 'tools' in request.body
            assert len(request.body['tools']) == 2
            assert 'tool_choice' not in request.body

    def test_create_request_with_tool_mode_no_toolset(self):
        """Test tool mode with no toolset."""
        from pydantic import BaseModel

        class NoToolModel(BaseModel):
            result: str

        # Mock the OutputSchema.build to return a schema with no toolset
        with patch('pydantic_ai._output.OutputSchema.build') as mock_build:
            mock_schema = MagicMock()
            mock_schema.mode = 'tool'
            mock_schema.toolset = None  # No toolset
            mock_build.return_value = mock_schema

            request = create_chat_request(
                custom_id='no-toolset-test',
                prompt='Test request',
                model='gpt-4o-mini',
                output_type=NoToolModel,
                output_mode='tool',
            )

            # Should not have tools since there's no toolset
            assert 'tools' not in request.body

    def test_create_request_with_prompted_mode_non_prompted_schema(self):
        """Test prompted mode with non-PromptedOutputSchema."""
        from pydantic import BaseModel

        class NonPromptedModel(BaseModel):
            result: str

        # Mock the OutputSchema.build to return a schema that's not PromptedOutputSchema
        with patch('pydantic_ai._output.OutputSchema.build') as mock_build:
            mock_schema = MagicMock()
            mock_schema.mode = 'prompted'
            # This will make isinstance(output_schema, PromptedOutputSchema) return False
            mock_schema.__class__.__name__ = 'NotPromptedOutputSchema'
            mock_build.return_value = mock_schema

            request = create_chat_request(
                custom_id='non-prompted-test',
                prompt='Test request',
                model='gpt-4o-mini',
                output_type=NonPromptedModel,
                output_mode='prompted',
            )

            # Messages should remain unchanged since it's not PromptedOutputSchema
            assert request.body['messages'] == [{'role': 'user', 'content': 'Test request'}]

    def test_create_request_with_unknown_output_mode(self):
        """Test create_chat_request with an output mode that's not handled."""
        from pydantic import BaseModel

        class TestModel(BaseModel):
            result: str

        # Mock the OutputSchema.build to return a schema with an unknown mode
        with patch('pydantic_ai._output.OutputSchema.build') as mock_build:
            mock_schema = MagicMock()
            mock_schema.mode = 'unknown_mode'  # This will not match any of native/tool/prompted
            mock_build.return_value = mock_schema

            request = create_chat_request(
                custom_id='unknown-mode-test',
                prompt='Test request',
                model='gpt-4o-mini',
                output_type=TestModel,
                output_mode='unknown_mode',  # type: ignore[arg-type]
            )

            # Should still return a valid request, just without output processing
            assert request.custom_id == 'unknown-mode-test'
            # Should not have special formatting since mode is unknown
            assert 'response_format' not in request.body
            assert 'tool_choice' not in request.body
