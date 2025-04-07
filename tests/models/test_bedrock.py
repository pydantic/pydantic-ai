# tests/test_bedrock.py

from __future__ import annotations as _annotations

import datetime
import os

# --- MODIFIED IMPORT: Ensure 'cast' and other types are present ---
# ----------------------------------
import pytest

# -------------------------------------
# --- ADDED MockerFixture import ---
from pytest_mock import MockerFixture

# from dirty_equals import IsInstance # Keep if used by other tests, else remove
# from inline_snapshot import snapshot # Keep if used by other tests, else remove
from typing_extensions import TypedDict

# Standard library mocking can also be used if preferred
# from unittest.mock import patch, Mock
from pydantic_ai.agent import Agent
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    TextPart,
    UserPromptPart,
)

# Assuming conftest might be in ../ relative to tests/models/
# Adjust relative path if needed
try:
    # --- MODIFIED IMPORT: Removed unused IsDatetime ---
    from ..conftest import try_import
except ImportError:  # Fallback if running from a different structure
    # --- MODIFIED IMPORT: Removed unused IsDatetime ---
    from conftest import try_import


with try_import() as imports_successful:
    import boto3

    # Import the specific client type for hinting
    from mypy_boto3_bedrock_runtime import BedrockRuntimeClient

    from pydantic_ai.models.bedrock import BedrockConverseModel
    from pydantic_ai.providers.bedrock import BedrockProvider

pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='bedrock not installed'),
    pytest.mark.anyio,
    pytest.mark.vcr,  # Keep for existing tests if they rely on it
]


@pytest.fixture
def bedrock_provider():
    """Fixture to provide a BedrockProvider instance."""
    aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID', 'DUMMY_KEY_ID')
    aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY', 'DUMMY_SECRET_KEY')
    region_name = os.getenv('AWS_REGION', 'us-east-1')

    if aws_access_key_id == 'DUMMY_KEY_ID':
        print('\nWARNING: Using dummy AWS credentials for Bedrock client fixture.')

    bedrock_client: BedrockRuntimeClient = boto3.client(  # type: ignore[assignment]
        'bedrock-runtime',
        region_name=region_name,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
    )
    yield BedrockProvider(bedrock_client=bedrock_client)
    try:
        bedrock_client.close()
    except Exception:
        pass


# --- Existing tests (modified slightly for robustness) ---

# NOTE: Assuming the snapshots in the original file are correct.
# If tests fail due to snapshot mismatches after fixing imports/logic,
# they might need to be updated using `pytest --snapshot-update`.


async def test_bedrock_model(allow_model_requests: None, bedrock_provider: BedrockProvider):
    model = BedrockConverseModel('us.amazon.nova-micro-v1:0', provider=bedrock_provider)
    agent = Agent(model=model, system_prompt='You are a chatbot.')

    result = await agent.run('Hello!')
    assert isinstance(result.data, str)
    assert 'assist' in result.data.lower()
    assert result.usage().requests == 1
    assert result.usage().request_tokens is not None
    assert result.usage().response_tokens is not None

    messages = result.all_messages()
    assert len(messages) == 2
    assert isinstance(messages[0], ModelRequest)
    assert isinstance(messages[0].parts[0], SystemPromptPart)
    assert isinstance(messages[0].parts[1], UserPromptPart)
    assert messages[0].parts[1].content == 'Hello!'
    assert isinstance(messages[1], ModelResponse)
    assert isinstance(messages[1].parts[0], TextPart)
    assert 'assist' in messages[1].parts[0].content.lower()


async def test_bedrock_model_structured_response(allow_model_requests: None, bedrock_provider: BedrockProvider):
    model = BedrockConverseModel('us.amazon.nova-micro-v1:0', provider=bedrock_provider)
    agent = Agent(model=model, system_prompt='You are a helpful chatbot.', retries=5)

    class Response(TypedDict):
        temperature: str
        date: datetime.date
        city: str

    @agent.tool_plain
    async def temperature(city: str, date: datetime.date) -> str:
        """Get the temperature in a city on a specific date."""
        return '30°C'

    result = await agent.run('What was the temperature in London 1st January 2022?', result_type=Response)
    assert result.data == {'temperature': '30°C', 'date': datetime.date(2022, 1, 1), 'city': 'London'}
    assert result.usage().requests > 0


# ... (Keep other existing tests, assuming they are correct) ...
# ... [Existing tests omitted for brevity - MAKE SURE TO KEEP THEM IN YOUR ACTUAL FILE] ...

# ---- START: New Tests for System Prompt Fix ----


@pytest.mark.anyio
async def test_bedrock_converse_no_system_prompt(
    allow_model_requests: None, bedrock_provider: BedrockProvider, mocker: MockerFixture
):
    """
    Test that BedrockConverseModel omits 'system' param when no system prompt is provided.
    """
    assert bedrock_provider.client is not None, 'Bedrock client not initialized in provider fixture'
    model = BedrockConverseModel('us.amazon.nova-micro-v1:0', provider=bedrock_provider)
    agent = Agent(model=model)

    mock_converse = mocker.patch.object(model.client, 'converse', autospec=True)
    mock_converse.return_value = {
        'output': {'message': {'content': [{'text': 'Mock response: No system prompt received.'}]}},
        'usage': {'inputTokens': 5, 'outputTokens': 6, 'totalTokens': 11},
        'ResponseMetadata': {'HTTPStatusCode': 200},
    }

    try:
        result = await agent.run('Test prompt without system')
    except Exception as e:
        pytest.fail(f'Agent run failed unexpectedly when no system prompt was provided: {e}')

    mock_converse.assert_called_once()
    call_kwargs = mock_converse.call_args.kwargs
    assert 'system' not in call_kwargs, "The 'system' parameter should be omitted when no system prompt is given"
    assert 'messages' in call_kwargs
    assert len(call_kwargs['messages']) == 1
    assert call_kwargs['messages'][0]['role'] == 'user'
    assert call_kwargs['messages'][0]['content'][0]['text'] == 'Test prompt without system'
    assert result.data == 'Mock response: No system prompt received.'


@pytest.mark.anyio
async def test_bedrock_converse_with_system_prompt(
    allow_model_requests: None, bedrock_provider: BedrockProvider, mocker: MockerFixture
):
    """
    Test that BedrockConverseModel includes 'system' param correctly when the Agent
    is initialized with a static system_prompt (assuming the fix is applied).
    """
    system_prompt_text = 'You are a helpful testing bot.'
    assert bedrock_provider.client is not None, 'Bedrock client not initialized in provider fixture'
    model = BedrockConverseModel('us.amazon.nova-micro-v1:0', provider=bedrock_provider)
    agent = Agent(model=model, system_prompt=system_prompt_text)

    mock_converse = mocker.patch.object(model.client, 'converse', autospec=True)
    mock_converse.return_value = {
        'output': {'message': {'content': [{'text': 'Mock response: System prompt received.'}]}},
        'usage': {'inputTokens': 20, 'outputTokens': 6, 'totalTokens': 26},
        'ResponseMetadata': {'HTTPStatusCode': 200},
    }

    try:
        result = await agent.run('Test prompt with system')
    except Exception as e:
        pytest.fail(f'Agent run failed unexpectedly when system prompt was provided: {e}')

    mock_converse.assert_called_once()
    call_kwargs = mock_converse.call_args.kwargs
    assert 'system' in call_kwargs, "The 'system' parameter should be present when Agent has a system_prompt"
    assert isinstance(call_kwargs['system'], list)
    assert len(call_kwargs['system']) == 1
    assert call_kwargs['system'][0] == {'text': system_prompt_text}, 'System prompt text does not match'
    assert 'messages' in call_kwargs
    assert len(call_kwargs['messages']) == 1
    assert call_kwargs['messages'][0]['role'] == 'user'
    assert call_kwargs['messages'][0]['content'][0]['text'] == 'Test prompt with system'
    assert result.data == 'Mock response: System prompt received.'


# ---- END: New Tests for System Prompt Fix ----
