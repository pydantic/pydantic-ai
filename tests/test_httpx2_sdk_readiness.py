from __future__ import annotations as _annotations

import subprocess
import sys
import textwrap

import httpx2
import pytest

from .conftest import try_import

with try_import() as anthropic_imports_successful:
    from anthropic import AsyncAnthropic

with try_import() as groq_imports_successful:
    from groq import AsyncGroq

with try_import() as google_imports_successful:
    from google.genai import Client as GoogleClient
    from google.genai.types import HttpOptions

with try_import() as mistral_imports_successful:
    from mistralai.client import Mistral

_HTTPX_FREE_CORE = """
import sys


class BlockHttpx:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'httpx' or fullname.startswith('httpx.'):
            raise ImportError('httpx is not installed')


sys.meta_path.insert(0, BlockHttpx())

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

_HTTPX_FREE_OPENAI = """
import sys


class BlockHttpx:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'httpx' or fullname.startswith('httpx.'):
            raise ImportError('httpx is not installed')


sys.meta_path.insert(0, BlockHttpx())

import asyncio

import httpx2

from pydantic_ai.providers.gateway import gateway_provider
from pydantic_ai.models.openai import OpenAIChatModel
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
assert not any(name == 'httpx' or name.startswith('httpx.') for name in sys.modules), 'OpenAI providers imported httpx'
"""


def test_core_runs_without_httpx() -> None:
    result = subprocess.run(
        [sys.executable, '-W', 'error', '-c', textwrap.dedent(_HTTPX_FREE_CORE)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert result.stderr == ''


def test_openai_providers_run_without_httpx() -> None:
    result = subprocess.run(
        [sys.executable, '-W', 'error', '-c', textwrap.dedent(_HTTPX_FREE_OPENAI)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert result.stderr == ''


@pytest.mark.skipif(not anthropic_imports_successful(), reason='anthropic not installed')
async def test_anthropic_still_rejects_httpx2_client() -> None:
    async with httpx2.AsyncClient() as client:
        with pytest.raises(TypeError, match=r'Expected an instance of `httpx\.AsyncClient`'):
            AsyncAnthropic(api_key='test', http_client=client)  # pyright: ignore[reportArgumentType]


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
