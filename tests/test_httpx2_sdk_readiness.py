from __future__ import annotations as _annotations

import subprocess
import sys
import textwrap

import httpx2
import pytest
from anthropic import AsyncAnthropic
from google.genai import Client as GoogleClient
from google.genai.types import HttpOptions
from groq import AsyncGroq
from pydantic import ValidationError

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


def test_core_runs_without_httpx() -> None:
    result = subprocess.run(
        [sys.executable, '-W', 'error', '-c', textwrap.dedent(_HTTPX_FREE_CORE)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert result.stderr == ''


async def test_anthropic_still_rejects_httpx2_client() -> None:
    async with httpx2.AsyncClient() as client:
        with pytest.raises(TypeError, match=r'Expected an instance of `httpx\.AsyncClient`'):
            AsyncAnthropic(api_key='test', http_client=client)  # pyright: ignore[reportArgumentType]


async def test_groq_still_rejects_httpx2_client() -> None:
    async with httpx2.AsyncClient() as client:
        with pytest.raises(TypeError, match=r'Expected an instance of `httpx\.AsyncClient`'):
            AsyncGroq(api_key='test', http_client=client)  # pyright: ignore[reportArgumentType]


async def test_google_still_rejects_httpx2_client() -> None:
    async with httpx2.AsyncClient() as client:
        with pytest.raises(ValidationError, match='httpx_async_client'):
            GoogleClient(
                api_key='test',
                http_options=HttpOptions(
                    httpx_async_client=client,  # pyright: ignore[reportArgumentType]
                ),
            )
