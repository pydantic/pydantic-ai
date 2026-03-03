"""Tests for ContextWindowExceeded exception detection across providers."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import httpx
import pytest

from pydantic_ai import Agent
from pydantic_ai.exceptions import ContextWindowExceeded
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.providers.gateway import gateway_provider

from .conftest import try_import

pytestmark = [pytest.mark.anyio, pytest.mark.vcr]

with try_import() as openai_imports_successful:
    from openai import APIStatusError as OpenAIAPIStatusError

    from pydantic_ai.models.openai import (
        OpenAIChatModel,
        OpenAIResponsesModel,
        _check_context_window_exceeded as openai_check,  # pyright: ignore[reportPrivateUsage]
    )
    from pydantic_ai.providers.openai import OpenAIProvider

with try_import() as anthropic_imports_successful:
    from anthropic import APIStatusError as AnthropicAPIStatusError

    from pydantic_ai.models.anthropic import (
        AnthropicModel,
        _check_context_window_exceeded as anthropic_check,  # pyright: ignore[reportPrivateUsage]
    )

with try_import() as groq_imports_successful:
    from groq import APIStatusError as GroqAPIStatusError

    from pydantic_ai.models.groq import (
        GroqModel,
        _check_context_window_exceeded as groq_check,  # pyright: ignore[reportPrivateUsage]
    )
    from pydantic_ai.providers.groq import GroqProvider

with try_import() as google_imports_successful:
    from pydantic_ai.models.google import GoogleModel
    from pydantic_ai.providers.google import GoogleProvider

with try_import() as bedrock_imports_successful:
    from botocore.exceptions import ClientError as BotoClientError

    from pydantic_ai.models.bedrock import (
        BedrockConverseModel,
        _check_context_window_exceeded as bedrock_check,  # pyright: ignore[reportPrivateUsage]
    )

with try_import() as mistral_imports_successful:
    from mistralai.models import SDKError as MistralSDKError

    from pydantic_ai.models.mistral import (
        MistralModel,
        _check_context_window_exceeded as mistral_check,  # pyright: ignore[reportPrivateUsage]
    )
    from pydantic_ai.providers.mistral import MistralProvider

with try_import() as cohere_imports_successful:
    from cohere.core.api_error import ApiError as CohereApiError

    from pydantic_ai.models.cohere import (
        CohereModel,
        _check_context_window_exceeded as cohere_check,  # pyright: ignore[reportPrivateUsage]
    )
    from pydantic_ai.providers.cohere import CohereProvider

HUGE_PROMPT = 'word ' * 150_000
ANTHROPIC_HUGE_PROMPT = 'word ' * 250_000
GOOGLE_HUGE_PROMPT = 'word ' * 1_100_000

_MOCK_REQUEST = httpx.Request('POST', 'https://example.com')


def _mock_response(status_code: int) -> httpx.Response:
    return httpx.Response(status_code, request=_MOCK_REQUEST)


# ==================== VCR integration tests ====================


@pytest.mark.skipif(not openai_imports_successful(), reason='openai not installed')
async def test_openai_context_window_exceeded(allow_model_requests: None, openai_api_key: str):
    """Test that OpenAI Chat context length exceeded errors raise ContextWindowExceeded."""
    model = OpenAIChatModel('gpt-4o-mini', provider=OpenAIProvider(api_key=openai_api_key))
    agent = Agent(model)

    with pytest.raises(ContextWindowExceeded) as exc_info:
        await agent.run(HUGE_PROMPT)

    assert exc_info.value.status_code == 400
    assert exc_info.value.model_name == 'gpt-4o-mini'


@pytest.mark.skipif(not openai_imports_successful(), reason='openai not installed')
async def test_openai_responses_context_window_exceeded(allow_model_requests: None, openai_api_key: str):
    """Test that OpenAI Responses API context length exceeded errors raise ContextWindowExceeded."""
    model = OpenAIResponsesModel('gpt-4o-mini', provider=OpenAIProvider(api_key=openai_api_key))
    agent = Agent(model)

    with pytest.raises(ContextWindowExceeded) as exc_info:
        await agent.run(HUGE_PROMPT)

    assert exc_info.value.status_code == 400
    assert exc_info.value.model_name == 'gpt-4o-mini'


@pytest.mark.skipif(not anthropic_imports_successful(), reason='anthropic not installed')
async def test_anthropic_context_window_exceeded(allow_model_requests: None, gateway_api_key: str):
    """Test that Anthropic context length exceeded errors raise ContextWindowExceeded."""
    model = AnthropicModel('claude-haiku-4-5', provider=gateway_provider('anthropic', api_key=gateway_api_key))
    agent = Agent(model)

    with pytest.raises(ContextWindowExceeded) as exc_info:
        await agent.run(ANTHROPIC_HUGE_PROMPT)

    assert exc_info.value.status_code == 400
    assert exc_info.value.model_name == 'claude-haiku-4-5'


@pytest.mark.skipif(not groq_imports_successful(), reason='groq not installed')
async def test_groq_context_window_exceeded(allow_model_requests: None, groq_api_key: str):
    """Test that Groq context length exceeded errors raise ContextWindowExceeded."""
    model = GroqModel('llama-3.1-8b-instant', provider=GroqProvider(api_key=groq_api_key))
    agent = Agent(model)

    with pytest.raises(ContextWindowExceeded) as exc_info:
        await agent.run(HUGE_PROMPT)

    assert exc_info.value.status_code == 400
    assert exc_info.value.model_name == 'llama-3.1-8b-instant'


@pytest.mark.skipif(not google_imports_successful(), reason='google-genai not installed')
async def test_google_context_window_exceeded(allow_model_requests: None, gemini_api_key: str):
    """Test that Google context length exceeded errors raise ContextWindowExceeded."""
    model = GoogleModel('gemini-2.0-flash', provider=GoogleProvider(api_key=gemini_api_key))
    agent = Agent(model)

    with pytest.raises(ContextWindowExceeded) as exc_info:
        await agent.run(GOOGLE_HUGE_PROMPT)

    assert exc_info.value.status_code == 400
    assert exc_info.value.model_name == 'gemini-2.0-flash'


@pytest.mark.skipif(not bedrock_imports_successful(), reason='boto3 not installed')
async def test_bedrock_context_window_exceeded(allow_model_requests: None, gateway_api_key: str):
    """Test that Bedrock context length exceeded errors raise ContextWindowExceeded."""
    model = BedrockConverseModel(
        'us.amazon.nova-micro-v1:0', provider=gateway_provider('bedrock', api_key=gateway_api_key)
    )
    agent = Agent(model)

    with pytest.raises(ContextWindowExceeded) as exc_info:
        await agent.run(HUGE_PROMPT)

    assert exc_info.value.status_code == 400
    assert exc_info.value.model_name == 'us.amazon.nova-micro-v1:0'


@pytest.mark.skipif(not mistral_imports_successful(), reason='mistral not installed')
async def test_mistral_context_window_exceeded(allow_model_requests: None, mistral_api_key: str):
    """Test that Mistral context length exceeded errors raise ContextWindowExceeded."""
    model = MistralModel('mistral-small-latest', provider=MistralProvider(api_key=mistral_api_key))
    agent = Agent(model)

    with pytest.raises(ContextWindowExceeded) as exc_info:
        await agent.run(HUGE_PROMPT)

    assert exc_info.value.status_code == 400
    assert exc_info.value.model_name == 'mistral-small-latest'


@pytest.mark.skipif(not cohere_imports_successful(), reason='cohere not installed')
async def test_cohere_context_window_exceeded(allow_model_requests: None, co_api_key: str):
    """Test that Cohere context length exceeded errors raise ContextWindowExceeded."""
    model = CohereModel('command-r7b-12-2024', provider=CohereProvider(api_key=co_api_key))
    agent = Agent(model)

    with pytest.raises(ContextWindowExceeded) as exc_info:
        await agent.run(HUGE_PROMPT)

    assert exc_info.value.status_code == 400
    assert exc_info.value.model_name == 'command-r7b-12-2024'


# ==================== Unit tests for _check_context_window_exceeded branches ====================


def _openai_api_error(status_code: int, body: object) -> OpenAIAPIStatusError:
    return OpenAIAPIStatusError(message='error', response=_mock_response(status_code), body=body)


@pytest.mark.skipif(not openai_imports_successful(), reason='openai not installed')
class TestOpenAICheckContextWindow:
    def test_nested_error_code(self):
        exc = _openai_api_error(400, {'error': {'code': 'context_length_exceeded'}})
        result = openai_check(exc, 'gpt-4o')
        assert isinstance(result, ContextWindowExceeded)

    def test_non_400_returns_none(self):
        exc = _openai_api_error(500, {'code': 'context_length_exceeded'})
        assert openai_check(exc, 'gpt-4o') is None

    def test_no_match_returns_none(self):
        exc = _openai_api_error(400, {'error': {'code': 'other_error'}})
        assert openai_check(exc, 'gpt-4o') is None

    def test_non_dict_body_returns_none(self):
        exc = _openai_api_error(400, 'not a dict')
        assert openai_check(exc, 'gpt-4o') is None


def _anthropic_api_error(status_code: int, body: object) -> AnthropicAPIStatusError:
    return AnthropicAPIStatusError(message='error', response=_mock_response(status_code), body=body)


@pytest.mark.skipif(not anthropic_imports_successful(), reason='anthropic not installed')
class TestAnthropicCheckContextWindow:
    def test_no_match_returns_none(self):
        exc = _anthropic_api_error(400, {'error': {'type': 'invalid_request_error', 'message': 'some other error'}})
        assert anthropic_check(exc, 'claude-haiku-4-5') is None

    def test_wrong_type_returns_none(self):
        exc = _anthropic_api_error(400, {'error': {'type': 'authentication_error', 'message': 'prompt is too long'}})
        assert anthropic_check(exc, 'claude-haiku-4-5') is None

    def test_non_dict_body_returns_none(self):
        exc = _anthropic_api_error(400, 'not a dict')
        assert anthropic_check(exc, 'claude-haiku-4-5') is None


def _groq_api_error(status_code: int, body: object) -> GroqAPIStatusError:
    return GroqAPIStatusError(message='error', response=_mock_response(status_code), body=body)


@pytest.mark.skipif(not groq_imports_successful(), reason='groq not installed')
class TestGroqCheckContextWindow:
    def test_code_only(self):
        exc = _groq_api_error(400, {'error': {'code': 'context_length_exceeded'}})
        result = groq_check(exc, 'llama-3.1-8b-instant')
        assert isinstance(result, ContextWindowExceeded)

    def test_no_match_returns_none(self):
        exc = _groq_api_error(400, {'error': {'type': 'other', 'code': 'other'}})
        assert groq_check(exc, 'llama-3.1-8b-instant') is None

    def test_non_400_returns_none(self):
        exc = _groq_api_error(429, {'error': {'code': 'context_length_exceeded'}})
        assert groq_check(exc, 'llama-3.1-8b-instant') is None

    def test_non_dict_body_returns_none(self):
        exc = _groq_api_error(400, 'not a dict')
        assert groq_check(exc, 'llama-3.1-8b-instant') is None


@pytest.mark.skipif(not mistral_imports_successful(), reason='mistral not installed')
class TestMistralCheckContextWindow:
    @staticmethod
    def _sdk_error(status_code: int, body: str | None = None) -> MistralSDKError:
        return MistralSDKError(
            message='error',
            raw_response=_mock_response(status_code),
            body=body,
        )

    def test_json_string_body_code(self):
        exc = self._sdk_error(400, '{"code": 3051, "message": "too large"}')
        result = mistral_check(exc, 'mistral-small-latest')
        assert isinstance(result, ContextWindowExceeded)

    def test_json_string_body_message_pattern(self):
        exc = self._sdk_error(400, '{"message": "maximum context length exceeded"}')
        result = mistral_check(exc, 'mistral-small-latest')
        assert isinstance(result, ContextWindowExceeded)

    def test_json_string_body_no_match(self):
        exc = self._sdk_error(400, '{"message": "some other error"}')
        assert mistral_check(exc, 'mistral-small-latest') is None

    def test_non_json_string_body(self):
        exc = self._sdk_error(400, 'not json at all')
        assert mistral_check(exc, 'mistral-small-latest') is None

    def test_json_string_non_dict(self):
        exc = self._sdk_error(400, '"just a string"')
        assert mistral_check(exc, 'mistral-small-latest') is None

    def test_non_400_returns_none(self):
        exc = self._sdk_error(500, '{"code": 3051}')
        assert mistral_check(exc, 'mistral-small-latest') is None

    def test_dict_body(self):
        """When SDKError.body is already a dict (not a JSON string)."""
        exc = MistralSDKError(
            message='error',
            raw_response=_mock_response(400),
            body={'code': '3051', 'message': 'error'},  # pyright: ignore[reportArgumentType]
        )
        result = mistral_check(exc, 'mistral-small-latest')
        assert isinstance(result, ContextWindowExceeded)


@pytest.mark.skipif(not cohere_imports_successful(), reason='cohere not installed')
class TestCohereCheckContextWindow:
    @staticmethod
    def _api_error(status_code: int | None, body: object) -> CohereApiError:
        return CohereApiError(status_code=status_code, body=body)

    def test_match(self):
        exc = self._api_error(400, {'message': 'too many tokens in the input'})
        result = cohere_check(exc, 'command-r')
        assert isinstance(result, ContextWindowExceeded)

    def test_no_match_returns_none(self):
        exc = self._api_error(400, {'message': 'some other error'})
        assert cohere_check(exc, 'command-r') is None

    def test_no_status_code_returns_none(self):
        exc = self._api_error(None, {'message': 'too many tokens'})
        assert cohere_check(exc, 'command-r') is None

    def test_non_dict_body_returns_none(self):
        exc = self._api_error(400, 'not a dict')
        assert cohere_check(exc, 'command-r') is None


@pytest.mark.skipif(not bedrock_imports_successful(), reason='boto3 not installed')
class TestBedrockCheckContextWindow:
    @staticmethod
    def _client_error(status_code: int, message: str) -> BotoClientError:
        return BotoClientError(
            {
                'Error': {'Code': 'ValidationException', 'Message': message},
                'ResponseMetadata': {
                    'RequestId': '',
                    'HostId': '',
                    'HTTPStatusCode': status_code,
                    'HTTPHeaders': {},
                    'RetryAttempts': 0,
                },
            },
            'Converse',
        )

    def test_no_match_returns_none(self):
        exc = self._client_error(400, 'some other validation error')
        assert bedrock_check(exc, 'nova-micro') is None

    def test_non_400_returns_none(self):
        exc = self._client_error(500, 'input is too long')
        assert bedrock_check(exc, 'nova-micro') is None


# ==================== Unit tests for count_tokens context window detection ====================


@pytest.mark.skipif(not anthropic_imports_successful(), reason='anthropic not installed')
async def test_anthropic_count_tokens_context_window(allow_model_requests: None):
    """Test that Anthropic count_tokens raises ContextWindowExceeded on context window errors."""
    model = AnthropicModel('claude-haiku-4-5', provider=gateway_provider('anthropic', api_key='mock-key'))
    error = AnthropicAPIStatusError(
        message='error',
        response=_mock_response(400),
        body={'error': {'type': 'invalid_request_error', 'message': 'prompt is too long for this model'}},
    )
    params = ModelRequestParameters()
    with patch.object(model.client.beta.messages, 'count_tokens', new_callable=AsyncMock, side_effect=error):
        with pytest.raises(ContextWindowExceeded) as exc_info:
            await model.count_tokens([], None, params)

    assert exc_info.value.model_name == 'claude-haiku-4-5'


@pytest.mark.skipif(not bedrock_imports_successful(), reason='boto3 not installed')
async def test_bedrock_count_tokens_context_window(allow_model_requests: None):
    """Test that Bedrock count_tokens raises ContextWindowExceeded on context window errors."""
    model = BedrockConverseModel('us.amazon.nova-micro-v1:0', provider=gateway_provider('bedrock', api_key='mock-key'))
    error = BotoClientError(
        {
            'Error': {'Code': 'ValidationException', 'Message': 'input is too long for the model'},
            'ResponseMetadata': {
                'RequestId': '',
                'HostId': '',
                'HTTPStatusCode': 400,
                'HTTPHeaders': {},
                'RetryAttempts': 0,
            },
        },
        'CountTokens',
    )
    with patch.object(model, 'client') as mock_client:
        mock_client.count_tokens.side_effect = error
        with pytest.raises(ContextWindowExceeded) as exc_info:
            params = ModelRequestParameters()
            await model.count_tokens([], None, params)

    assert exc_info.value.model_name == 'us.amazon.nova-micro-v1:0'
