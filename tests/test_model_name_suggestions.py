from __future__ import annotations

import json
from contextlib import AsyncExitStack
from dataclasses import dataclass
from typing import Literal

import httpx
import httpx2
import pytest
from inline_snapshot import snapshot
from typing_extensions import assert_never

from pydantic_ai import Agent, ModelHTTPError

from .conftest import IsStr, TestEnv, try_import

with try_import() as openai_imports:
    from pydantic_ai.models.openai import OpenAIChatModel, OpenAIResponsesModel
    from pydantic_ai.providers.openai import OpenAIProvider

with try_import() as anthropic_imports:
    from pydantic_ai.models.anthropic import AnthropicModel
    from pydantic_ai.providers.anthropic import AnthropicProvider

with try_import() as google_imports:
    from pydantic_ai.models.google import GoogleModel
    from pydantic_ai.providers.google import GoogleProvider

with try_import() as bedrock_imports:
    from botocore.awsrequest import AWSPreparedRequest

    from pydantic_ai.models.bedrock import BedrockConverseModel
    from pydantic_ai.providers.bedrock import BedrockProvider

with try_import() as groq_imports:
    from pydantic_ai.models.groq import GroqModel
    from pydantic_ai.providers.groq import GroqProvider

pytestmark = [pytest.mark.anyio, pytest.mark.vcr]


ProviderCase = Literal['openai-responses', 'openai-chat', 'anthropic', 'google', 'bedrock', 'groq']


@dataclass(frozen=True)
class Case:
    provider: ProviderCase
    model_name: str
    expected_request: tuple[str, ...]
    expected_body: object
    expected_status_code: int
    expected_suggestion: str
    marks: tuple[pytest.MarkDecorator, ...]


CASES = [
    Case(
        provider='openai-responses',
        model_name='gpt-5.2-proo',
        expected_request=('gpt-5.2-proo',),
        expected_body=snapshot(
            {
                'message': "The requested model 'gpt-5.2-proo' does not exist.",
                'type': 'invalid_request_error',
                'param': 'model',
                'code': 'model_not_found',
            }
        ),
        expected_status_code=400,
        expected_suggestion='openai:gpt-5.2-pro',
        marks=(pytest.mark.skipif(not openai_imports(), reason='openai not installed'),),
    ),
    Case(
        provider='openai-chat',
        model_name='gpt-5.2-proo',
        expected_request=('gpt-5.2-proo',),
        expected_body=snapshot(
            {
                'message': 'The model `gpt-5.2-proo` does not exist or you do not have access to it.',
                'type': 'invalid_request_error',
                'param': None,
                'code': 'model_not_found',
            }
        ),
        expected_status_code=404,
        expected_suggestion='openai:gpt-5.2-pro',
        marks=(pytest.mark.skipif(not openai_imports(), reason='openai not installed'),),
    ),
    Case(
        provider='anthropic',
        model_name='claude-sonet-4-5',
        expected_request=('claude-sonet-4-5',),
        expected_body=snapshot(
            {
                'type': 'error',
                'error': {'type': 'not_found_error', 'message': 'model: claude-sonet-4-5'},
                'request_id': IsStr(),
            }
        ),
        expected_status_code=404,
        expected_suggestion='anthropic:claude-sonnet-4-5',
        marks=(pytest.mark.skipif(not anthropic_imports(), reason='anthropic not installed'),),
    ),
    Case(
        provider='google',
        model_name='gemini-3.6-flahs',
        expected_request=('/v1beta/models/gemini-3.6-flahs:generateContent',),
        expected_body=snapshot(
            {
                'error': {
                    'code': 404,
                    'message': 'models/gemini-3.6-flahs is not found for API version v1beta, or is not supported for generateContent. Call ModelService.ListModels to see the list of available models and their supported methods.',
                    'status': 'NOT_FOUND',
                }
            }
        ),
        expected_status_code=404,
        expected_suggestion='google:gemini-3.6-flash',
        marks=(pytest.mark.skipif(not google_imports(), reason='google not installed'),),
    ),
    Case(
        provider='bedrock',
        model_name='us.amazon.nova-micro-v1:O',
        expected_request=(
            'https://bedrock-runtime.us-east-1.amazonaws.com/model/us.amazon.nova-micro-v1%3AO/converse',
        ),
        expected_body=snapshot(
            {
                'Error': {'Message': 'The provided model identifier is invalid.', 'Code': '400'},
                'ResponseMetadata': {
                    'HTTPStatusCode': 400,
                    'HTTPHeaders': {
                        'connection': 'keep-alive',
                        'content-length': '56',
                        'content-type': 'application/json',
                    },
                    'RetryAttempts': 0,
                },
            }
        ),
        expected_status_code=400,
        expected_suggestion='bedrock:us.amazon.nova-micro-v1:0',
        marks=(pytest.mark.skipif(not bedrock_imports(), reason='bedrock not installed'),),
    ),
    Case(
        provider='groq',
        model_name='llama-3.3-70b-versatlie',
        expected_request=('llama-3.3-70b-versatlie',),
        expected_body=snapshot(
            {
                'error': {
                    'message': 'The model `llama-3.3-70b-versatlie` does not exist or you do not have access to it.',
                    'type': 'invalid_request_error',
                    'code': 'model_not_found',
                }
            }
        ),
        expected_status_code=404,
        expected_suggestion='groq:llama-3.3-70b-versatile',
        marks=(pytest.mark.skipif(not groq_imports(), reason='groq not installed'),),
    ),
]


@pytest.mark.parametrize('case', [pytest.param(case, id=case.provider, marks=case.marks) for case in CASES])
async def test_model_name_suggestion(case: Case, request: pytest.FixtureRequest, allow_model_requests: None):
    requested_identifiers: list[str] = []

    async def capture_model_request(request: httpx.Request | httpx2.Request) -> None:
        requested_identifiers.append(json.loads(request.content)['model'])

    async def capture_google_request(request: httpx.Request | httpx2.Request) -> None:
        requested_identifiers.append(request.url.path)

    def capture_bedrock_request(request: AWSPreparedRequest, **_: object) -> None:
        requested_identifiers.append(request.url)

    async with AsyncExitStack() as stack:
        if case.provider in ('openai-responses', 'openai-chat'):
            api_key: str = request.getfixturevalue('openai_api_key')
            http_client = await stack.enter_async_context(
                httpx2.AsyncClient(event_hooks={'request': [capture_model_request]})
            )
            openai_provider = OpenAIProvider(api_key=api_key, http_client=http_client)
            if case.provider == 'openai-responses':
                model = OpenAIResponsesModel(case.model_name, provider=openai_provider)
            else:
                model = OpenAIChatModel(case.model_name, provider=openai_provider)
        elif case.provider == 'anthropic':
            api_key: str = request.getfixturevalue('anthropic_api_key')
            http_client = await stack.enter_async_context(
                httpx2.AsyncClient(event_hooks={'request': [capture_model_request]})
            )
            model = AnthropicModel(
                case.model_name,
                provider=AnthropicProvider(api_key=api_key, http_client=http_client),
            )
        elif case.provider == 'google':
            api_key: str = request.getfixturevalue('gemini_api_key')
            http_client = await stack.enter_async_context(
                httpx2.AsyncClient(event_hooks={'request': [capture_google_request]})
            )
            model = GoogleModel(
                case.model_name,
                provider=GoogleProvider(api_key=api_key, http_client=http_client),
            )
        elif case.provider == 'bedrock':
            bedrock_provider: BedrockProvider = request.getfixturevalue('bedrock_provider')
            model = BedrockConverseModel(case.model_name, provider=bedrock_provider)
            event = 'before-send.bedrock-runtime.Converse'
            model.client.meta.events.register_last(event, capture_bedrock_request)
            stack.callback(model.client.meta.events.unregister, event, capture_bedrock_request)
        elif case.provider == 'groq':
            api_key: str = request.getfixturevalue('groq_api_key')
            http_client = await stack.enter_async_context(
                httpx.AsyncClient(event_hooks={'request': [capture_model_request]})
            )
            model = GroqModel(
                case.model_name,
                provider=GroqProvider(api_key=api_key, http_client=http_client),
            )
        else:
            assert_never(case.provider)

        with pytest.raises(ModelHTTPError) as exc_info:
            await Agent(model).run('hello')

    assert tuple(requested_identifiers) == case.expected_request
    assert exc_info.value.status_code == case.expected_status_code
    assert exc_info.value.body == case.expected_body
    assert exc_info.value.suggested_model_id == case.expected_suggestion
    assert exc_info.value.__cause__ is not None
    assert str(exc_info.value).endswith(f'. Did you mean {case.expected_suggestion!r}?')


@pytest.mark.skipif(not openai_imports(), reason='openai not installed')
async def test_inferred_model_name_suggestion(allow_model_requests: None, openai_api_key: str, env: TestEnv):
    env.set('OPENAI_API_KEY', openai_api_key)

    with pytest.raises(ModelHTTPError) as exc_info:
        await Agent('openai:gpt-5.2-proo').run('hello')

    assert str(exc_info.value).endswith(". Did you mean 'openai:gpt-5.2-pro'?")
    assert exc_info.value.__cause__ is not None
