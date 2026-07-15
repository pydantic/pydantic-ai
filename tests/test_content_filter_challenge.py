import pytest
from pydantic_ai import Agent, ModelMessage, ModelResponse, TextPart, ThinkingPart, ToolCallPart, RequestUsage
from pydantic_ai.exceptions import ContentFilterError
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models import Model, ModelRequestParameters, StreamedResponse
from pydantic_ai.messages import ModelResponseStreamEvent
from pydantic_ai._run_context import RunContext
from pydantic_ai.settings import ModelSettings
from typing import AsyncIterator, AsyncGenerator
from datetime import datetime
from contextlib import asynccontextmanager

# Tests for agent.run()

async def test_run_content_filter_empty():
    async def filtered_response(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(
            parts=[],
            model_name='test-model',
            finish_reason='content_filter',
            provider_details={'finish_reason': 'content_filter'},
        )

    model = FunctionModel(function=filtered_response, model_name='test-model')
    agent = Agent(model)

    with pytest.raises(ContentFilterError, match="Content filter triggered. Finish reason: 'content_filter'"):
        await agent.run('Trigger filter')

async def test_run_content_filter_partial_text():
    async def filtered_response(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(
            parts=[TextPart('Partially generated content...')],
            model_name='test-model',
            finish_reason='content_filter',
            provider_details={'finish_reason': 'content_filter'},
        )

    model = FunctionModel(function=filtered_response, model_name='test-model')
    agent = Agent(model)

    with pytest.raises(ContentFilterError, match="Content filter triggered. Finish reason: 'content_filter'"):
        await agent.run('Trigger filter')

async def test_run_content_filter_thinking():
    async def filtered_response(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(
            parts=[ThinkingPart('Thinking about this...', signature='thinking_sig')],
            model_name='test-model',
            finish_reason='content_filter',
            provider_details={'finish_reason': 'content_filter'},
        )

    model = FunctionModel(function=filtered_response, model_name='test-model')
    agent = Agent(model)

    with pytest.raises(ContentFilterError, match="Content filter triggered. Finish reason: 'content_filter'"):
        await agent.run('Trigger filter')

async def test_run_content_filter_tool_call():
    async def filtered_response(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(
            parts=[ToolCallPart('some_tool', {'arg': 1}, 'call_id')],
            model_name='test-model',
            finish_reason='content_filter',
            provider_details={'finish_reason': 'content_filter'},
        )

    model = FunctionModel(function=filtered_response, model_name='test-model')
    agent = Agent(model)

    with pytest.raises(ContentFilterError, match="Content filter triggered. Finish reason: 'content_filter'"):
        await agent.run('Trigger filter')


# Tests for agent.run_stream()

class ChallengeStreamedResponse(StreamedResponse):
    def __init__(self, model_request_parameters, parts, finish_reason=None):
        super().__init__(model_request_parameters=model_request_parameters)
        self.parts = parts
        self.finish_reason = finish_reason
        if finish_reason:
            self.provider_details = {'finish_reason': finish_reason}

    async def _get_event_iterator(self) -> AsyncIterator[ModelResponseStreamEvent]:
        self._usage = RequestUsage()
        for idx, part in enumerate(self.parts):
            yield self._parts_manager.handle_part(
                vendor_part_id=idx,
                part=part,
            )

    @property
    def model_name(self) -> str:
        return 'challenge-stream-model'

    @property
    def provider_name(self) -> str:
        return 'test'

    @property
    def provider_url(self) -> str:
        return 'https://test.example.com'

    @property
    def timestamp(self) -> datetime:
        return datetime(2026, 1, 1)

class ChallengeStreamModel(Model):
    def __init__(self, parts, finish_reason=None):
        super().__init__()
        self.parts = parts
        self.finish_reason = finish_reason

    @property
    def system(self) -> str:
        return 'test'

    @property
    def model_name(self) -> str:
        return 'challenge-stream-model'

    @property
    def base_url(self) -> str:
        return 'https://test.example.com'

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        return ModelResponse(parts=self.parts, finish_reason=self.finish_reason)

    @asynccontextmanager
    async def request_stream(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
        run_context: RunContext | None = None,
    ) -> AsyncGenerator[StreamedResponse, None]:
        yield ChallengeStreamedResponse(
            model_request_parameters=model_request_parameters,
            parts=self.parts,
            finish_reason=self.finish_reason,
        )

async def test_stream_content_filter_empty():
    model = ChallengeStreamModel(parts=[], finish_reason='content_filter')
    agent = Agent(model)

    with pytest.raises(ContentFilterError, match="Content filter triggered. Finish reason: 'content_filter'"):
        async with agent.run_stream('Trigger filter') as stream:
            await stream.get_output()

async def test_stream_content_filter_partial_text():
    model = ChallengeStreamModel(parts=[TextPart('Partially generated content...')], finish_reason='content_filter')
    agent = Agent(model)

    with pytest.raises(ContentFilterError, match="Content filter triggered. Finish reason: 'content_filter'"):
        async with agent.run_stream('Trigger filter') as stream:
            await stream.get_output()

async def test_stream_content_filter_thinking():
    model = ChallengeStreamModel(parts=[ThinkingPart('Thinking about this...', signature='thinking_sig')], finish_reason='content_filter')
    agent = Agent(model)

    with pytest.raises(ContentFilterError, match="Content filter triggered. Finish reason: 'content_filter'"):
        async with agent.run_stream('Trigger filter') as stream:
            await stream.get_output()
