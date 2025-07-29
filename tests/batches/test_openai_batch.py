"""Tests for OpenAI batch functionality based on real examples."""

from __future__ import annotations as _annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel

from pydantic_ai import RunContext

try:
    from pydantic_ai.batches.openai import (
        BatchJob,
        BatchResult,
        OpenAIBatchModel,
        create_chat_request,
    )
except ImportError:
    pytest.skip('openai is not installed', allow_module_level=True)

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
