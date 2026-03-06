from __future__ import annotations as _annotations

import pytest
from pydantic import BaseModel

from pydantic_ai import Agent
from pydantic_ai.output import NativeOutput, PromptedOutput

from ..conftest import try_import
from .mock_openai import MockOpenAI, completion_message, get_mock_chat_completion_kwargs

with try_import() as imports_successful:
    from openai.types.chat.chat_completion_message import ChatCompletionMessage

    from pydantic_ai.models.ollama import OllamaModel
    from pydantic_ai.providers.ollama import OllamaProvider


pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='openai not installed'),
]


def test_ollama_native_output_sends_response_format(allow_model_requests: None) -> None:
    class CityLocation(BaseModel):
        city: str
        country: str

    mock_client = MockOpenAI.create_mock(
        completion_message(ChatCompletionMessage(content='{"city":"Paris","country":"France"}', role='assistant'))
    )

    model = OllamaModel('qwen3', provider=OllamaProvider(openai_client=mock_client))
    agent = Agent(model, output_type=NativeOutput(CityLocation))

    agent.run_sync('What is the capital of France?')

    kwargs = get_mock_chat_completion_kwargs(mock_client)[-1]
    response_format = kwargs['response_format']

    assert response_format['type'] == 'json_schema'
    json_schema = response_format['json_schema']
    assert 'schema' in json_schema
    assert 'strict' not in json_schema

    schema = json_schema['schema']
    assert schema['type'] == 'object'
    assert 'city' in schema['properties'] and 'country' in schema['properties']


def test_ollama_json_object_output_sends_response_format(allow_model_requests: None) -> None:
    class CityLocation(BaseModel):
        city: str
        country: str

    mock_client = MockOpenAI.create_mock(
        completion_message(ChatCompletionMessage(content='{"city":"Paris","country":"France"}', role='assistant'))
    )

    model = OllamaModel('qwen3', provider=OllamaProvider(openai_client=mock_client))
    agent = Agent(model, output_type=PromptedOutput(CityLocation))

    agent.run_sync('What is the capital of France?')

    kwargs = get_mock_chat_completion_kwargs(mock_client)[-1]
    response_format = kwargs['response_format']

    assert response_format == {'type': 'json_object'}
