from __future__ import annotations as _annotations

import inspect
import subprocess
import sys
from importlib.metadata import requires

import httpx2
import pytest
from packaging.requirements import Requirement

from pydantic_ai._http import create_async_httpx2_client

from .conftest import try_import

with try_import() as anthropic_imports_successful:
    from anthropic import AsyncAnthropic

with try_import() as groq_imports_successful:
    from groq import AsyncGroq

with try_import() as cohere_imports_successful:
    from cohere import AsyncClientV2, TextAssistantMessageResponseContentItem, UserChatMessageV2

with try_import() as huggingface_imports_successful:
    from huggingface_hub import AsyncInferenceClient

with try_import() as google_imports_successful:
    from google.genai import Client as GoogleClient
    from google.genai.types import HttpOptions

with try_import() as mistral_imports_successful:
    from mistralai.client import Mistral

# Prepended to every subprocess script below: legacy HTTPX is installed in the test environment, so the
# scripts have to make it unimportable to stand in for an install that never had it.
_BLOCK_HTTPX = """
import sys


class BlockHttpx:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'httpx' or fullname.startswith('httpx.'):
            raise ImportError('httpx is not installed')


sys.meta_path.insert(0, BlockHttpx())
"""

_HTTPX_FREE_CORE = (
    _BLOCK_HTTPX
    + """
import asyncio
import typing

from pydantic_ai import Agent
from pydantic_ai.agent.spec import AgentSpec
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.providers import Provider

assert asyncio.run(Agent('test').run('hello')).output
assert AgentSpec.model_json_schema_with_capabilities()
assert typing.get_type_hints(Provider)


async def stream_function(messages, info):
    for word in ['hello ', 'world']:
        yield word


async def break_out_of_stream():
    async with Agent(FunctionModel(stream_function=stream_function)).run_stream('hello') as result:
        async for _ in result.stream_output():
            break


asyncio.run(break_out_of_stream())
assert not any(name == 'httpx' or name.startswith('httpx.') for name in sys.modules), 'the SDK-less core imported httpx'
"""
)

_HTTPX_FREE_OPENAI = (
    _BLOCK_HTTPX
    + """
import asyncio
import warnings

import httpx2

from pydantic_ai.providers.gateway import gateway_provider
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.github import GitHubProvider
from pydantic_ai.providers.openai import OpenAIProvider


async def construct_providers():
    async with httpx2.AsyncClient() as client:
        provider = OpenAIProvider(api_key='test', http_client=client)
        assert provider.client._client is client
        OpenAIChatModel('gpt-4o', provider=provider)

        gateway = gateway_provider(
            'openai', api_key='test', base_url='https://gateway.example.com', http_client=client
        )
        assert gateway.client._client is client


asyncio.run(construct_providers())

# The deprecated GitHub Models provider is not migrated to HTTPX2, so it stays importable but says
# what to install instead of dying on `openai`'s dropped legacy HTTPX dependency.
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    try:
        GitHubProvider(api_key='test')
    except ImportError as exc:
        assert 'Please install `httpx` to use the GitHub Models provider' in str(exc), exc
    else:
        raise AssertionError('`GitHubProvider` built a client without legacy httpx')

assert not any(name == 'httpx' or name.startswith('httpx.') for name in sys.modules), 'OpenAI providers imported httpx'
"""
)


_HTTPX_FREE_ANTHROPIC = (
    _BLOCK_HTTPX
    + """
import asyncio

import httpx2

from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.providers.anthropic import AnthropicProvider
from pydantic_ai.providers.gateway import gateway_provider


async def construct_providers():
    async with httpx2.AsyncClient() as client:
        provider = AnthropicProvider(api_key='test', http_client=client)
        assert provider.client._client is client
        AnthropicModel('claude-sonnet-4-6', provider=provider)

        gateway = gateway_provider(
            'anthropic', api_key='test', base_url='https://gateway.example.com', http_client=client
        )
        assert gateway.client._client is client


asyncio.run(construct_providers())

assert not any(name == 'httpx' or name.startswith('httpx.') for name in sys.modules), 'Anthropic providers imported httpx'
"""
)


def test_core_runs_without_httpx() -> None:
    result = subprocess.run(
        [sys.executable, '-W', 'error', '-c', _HTTPX_FREE_CORE],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert result.stderr == ''


def test_openai_providers_run_without_httpx() -> None:
    result = subprocess.run(
        [sys.executable, '-W', 'error', '-c', _HTTPX_FREE_OPENAI],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert result.stderr == ''


@pytest.mark.skipif(not anthropic_imports_successful(), reason='anthropic not installed')
def test_anthropic_providers_run_without_httpx() -> None:
    result = subprocess.run(
        [sys.executable, '-W', 'error', '-c', _HTTPX_FREE_ANTHROPIC],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert result.stderr == ''


async def test_httpx2_client_constructs_without_blocking() -> None:
    """`_http.py` preloads `httpcore2` at import so client construction can't block the event loop.

    The assertion that nothing blocks is the autouse `blockbuster` fixture, which raises `BlockingError`
    on a blocking call inside a scanned library frame — so this test is the regression guard for that
    preload only on the CI lanes that enable BlockBuster. Elsewhere it asserts construction alone.
    """
    async with create_async_httpx2_client() as client:
        assert isinstance(client, httpx2.AsyncClient)


@pytest.mark.skipif(not anthropic_imports_successful(), reason='anthropic not installed')
async def test_anthropic_accepts_httpx2_client() -> None:
    async with httpx2.AsyncClient() as client:
        anthropic_client = AsyncAnthropic(api_key='test', http_client=client)

        assert anthropic_client._client is client  # pyright: ignore[reportPrivateUsage]


@pytest.mark.skipif(not groq_imports_successful(), reason='groq not installed')
async def test_groq_still_rejects_httpx2_client() -> None:
    async with httpx2.AsyncClient() as client:
        with pytest.raises(TypeError, match=r'Expected an instance of `httpx\.AsyncClient`'):
            AsyncGroq(api_key='test', http_client=client)  # pyright: ignore[reportArgumentType]


@pytest.mark.skipif(not google_imports_successful(), reason='google-genai not installed')
async def test_google_accepts_httpx2_client() -> None:
    async with httpx2.AsyncClient() as client:
        google_client = GoogleClient(api_key='test', http_options=HttpOptions(httpx_async_client=client))

        assert google_client._api_client._async_httpx_client is client  # pyright: ignore[reportPrivateUsage]


@pytest.mark.skipif(not mistral_imports_successful(), reason='mistral not installed')
async def test_mistral_accepts_httpx2_client() -> None:
    async with httpx2.AsyncClient() as client:
        mistral_client = Mistral(api_key='test', async_client=client)  # pyright: ignore[reportArgumentType]

        assert mistral_client.sdk_configuration.async_client is client


def _requirement_names(distribution: str) -> set[str]:
    return {Requirement(value).name for value in requires(distribution) or []}


@pytest.mark.skipif(not cohere_imports_successful(), reason='cohere not installed')
async def test_cohere_drives_an_httpx2_client_but_declares_legacy_httpx() -> None:
    """Cohere holds `CohereProvider` back for typing and packaging reasons, not runtime ones.

    Unlike Groq, Cohere never checks the client's type: `httpx_client` is duck-typed all the way down
    to `AsyncClient.request`, so an HTTPX2 client both constructs and completes a request. What blocks
    the migration is the `pyright: ignore` below — `httpx_client` is still annotated
    `httpx.AsyncClient` — plus Cohere pulling legacy HTTPX into the install unconditionally.

    Trips when Cohere widens that annotation, because `reportUnnecessaryTypeIgnoreComment` then fails
    the unused ignore, or when it drops the legacy HTTPX requirement: migrate `CohereProvider` then.
    Trips the other way if the request stops working, which means the duck-typing broke and the
    migration is further off than this file records.
    """

    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(
            200,
            json={
                'id': 'test-id',
                'finish_reason': 'COMPLETE',
                'message': {'role': 'assistant', 'content': [{'type': 'text', 'text': 'hello'}]},
            },
        )

    async with httpx2.AsyncClient(transport=httpx2.MockTransport(handler)) as client:
        cohere_client = AsyncClientV2(api_key='test', httpx_client=client)  # pyright: ignore[reportArgumentType]

        response = await cohere_client.chat(model='command-r', messages=[UserChatMessageV2(content='hello')])

    assert response.message.content == [TextAssistantMessageResponseContentItem(text='hello')]
    assert 'httpx' in _requirement_names('cohere')


@pytest.mark.skipif(not huggingface_imports_successful(), reason='huggingface-hub not installed')
def test_huggingface_offers_no_client_injection_and_keeps_legacy_httpx() -> None:
    """HuggingFace has no per-client injection seam to migrate, so it gets no client-acceptance canary.

    `AsyncInferenceClient` takes no client at all: it lazily builds a legacy `httpx.AsyncClient` from
    the process-global `huggingface_hub.set_async_client_factory` hook. That is why
    `HuggingFaceProvider` rejects `http_client` and accepts a whole `hf_client` instead, and why the
    only thing left to pin is the constructor's lack of a seam. Its module-level `from httpx import
    AsyncClient` resolves only because `huggingface-hub` requires legacy HTTPX unconditionally — the
    `huggingface` extra never declares HTTPX itself.

    Trips when `AsyncInferenceClient` grows a client or transport parameter, which is a real injection
    seam to migrate, or when `huggingface-hub` drops legacy HTTPX, which breaks the provider's import
    and means the extra needs its own HTTPX dependency.
    """
    parameters = inspect.signature(AsyncInferenceClient).parameters

    assert [name for name in parameters if 'client' in name or 'transport' in name] == []
    assert 'httpx' in _requirement_names('huggingface-hub')


# MCP gets no canary of its own. `MCPToolset(http_client=...)` never hands the client to an SDK
# constructor — it is wrapped in an `httpx_client_factory` that FastMCP only calls at connect time —
# so there is nothing to assert in-process, and the readiness signal is packaging-level anyway:
# `pydantic_ai.mcp`'s module-level `import httpx` resolves only because `fastmcp-slim[client]` pulls
# legacy HTTPX in transitively. `.github/scripts/check_http_dependencies.py` already asserts exactly
# that, and fails the moment FastMCP drops legacy HTTPX; migrate `mcp.py` then.
