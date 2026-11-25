"""Capture the request bodies OpenAI, Anthropic and Gemini receive, to check strict mode on the wire."""

import json
from typing import Any

import httpx2
import pytest

from pydantic_ai.models import Model
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.anthropic import AnthropicProvider
from pydantic_ai.providers.openai import OpenAIProvider
from tests.conftest import try_import

with try_import() as google_installed:
    from pydantic_ai.models.google import GoogleModel
    from pydantic_ai.providers.google import GoogleProvider

PROVIDERS = (
    'openai',
    'anthropic',
    pytest.param('google', marks=pytest.mark.skipif(not google_installed(), reason='google-genai not installed')),
)


def capturing_model(provider: str, bodies: list[dict[str, Any]]) -> Model:
    """A real `provider` model whose requests are appended to `bodies` and then rejected with a 400."""

    def respond(request: httpx2.Request) -> httpx2.Response:
        bodies.append(json.loads(request.content))
        return httpx2.Response(400, json={'error': {'message': 'captured', 'type': 'invalid_request_error'}})

    client = httpx2.AsyncClient(transport=httpx2.MockTransport(respond), timeout=600)
    if provider == 'openai':
        return OpenAIChatModel('gpt-5', provider=OpenAIProvider(api_key='test', http_client=client))
    if provider == 'anthropic':
        return AnthropicModel('claude-sonnet-4-6', provider=AnthropicProvider(api_key='test', http_client=client))
    return GoogleModel('gemini-2.5-pro', provider=GoogleProvider(api_key='test', http_client=client))


def strict_tools(provider: str, body: dict[str, Any]) -> dict[str, bool]:
    """Each tool `body` declares, and whether strict mode is on for it."""
    if provider == 'openai':
        return {tool['function']['name']: tool['function'].get('strict', False) for tool in body['tools']}
    if provider == 'anthropic':
        return {tool['name']: tool.get('strict', False) for tool in body['tools']}
    validated = body.get('toolConfig', {}).get('functionCallingConfig', {}).get('mode') == 'VALIDATED'
    return {declaration['name']: validated for tool in body['tools'] for declaration in tool['functionDeclarations']}
