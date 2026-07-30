"""Guards on where `httpx` is still allowed to appear.

Pydantic AI uses `httpx2` for the HTTP it owns end to end. `httpx` survives only at the provider
SDK boundary, because every SDK below runs `isinstance(http_client, httpx.AsyncClient)` and raises
`TypeError: Invalid 'http_client' argument` otherwise. Two things follow, one test each:

- `httpx` is not a dependency of `pydantic-ai-slim`; it arrives transitively, from whichever
  provider SDK the user installs. The SDK-less core must therefore import and run without it.
- The SDK rejection still happens. That test **fails as soon as an SDK ships httpx2 support** —
  CI resolves the locked versions, so the signal arrives on the monthly Dependabot bump that
  brings the new SDK in. A failure there is good news: drop that provider from the list and move
  its `http_client` parameter to `httpx2`.
"""

from __future__ import annotations as _annotations

import subprocess
import sys
import textwrap
from collections.abc import Callable
from typing import Any

import httpx2
import pytest

from .conftest import try_import

with try_import() as openai_imports_successful:
    from pydantic_ai.providers.openai import OpenAIProvider

with try_import() as anthropic_imports_successful:
    from pydantic_ai.providers.anthropic import AnthropicProvider

with try_import() as groq_imports_successful:
    from pydantic_ai.providers.groq import GroqProvider


def _openai_provider(http_client: httpx2.AsyncClient) -> Any:
    return OpenAIProvider(api_key='api-key', http_client=http_client)  # pyright: ignore[reportArgumentType]


def _anthropic_provider(http_client: httpx2.AsyncClient) -> Any:
    return AnthropicProvider(api_key='api-key', http_client=http_client)  # pyright: ignore[reportArgumentType]


def _groq_provider(http_client: httpx2.AsyncClient) -> Any:
    return GroqProvider(api_key='api-key', http_client=http_client)  # pyright: ignore[reportArgumentType]


@pytest.mark.parametrize(
    'build_provider',
    [
        pytest.param(
            _openai_provider,
            id='openai',
            marks=pytest.mark.skipif(not openai_imports_successful(), reason='need to install openai'),
        ),
        pytest.param(
            _anthropic_provider,
            id='anthropic',
            marks=pytest.mark.skipif(not anthropic_imports_successful(), reason='need to install anthropic'),
        ),
        pytest.param(
            _groq_provider,
            id='groq',
            marks=pytest.mark.skipif(not groq_imports_successful(), reason='need to install groq'),
        ),
    ],
)
def test_sdk_still_rejects_httpx2_client(build_provider: Callable[[httpx2.AsyncClient], Any]) -> None:
    """When this fails, the SDK accepts `httpx2` — see the module docstring."""
    with pytest.raises(TypeError, match='http_client'):
        build_provider(httpx2.AsyncClient())


# Runs in a subprocess because `httpx` is already imported in this session (the provider SDKs pull
# it in), and blocking a module that is in `sys.modules` proves nothing.
_HTTPX_FREE_CORE = """
import sys


class BlockHttpx:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'httpx' or fullname.startswith('httpx.'):
            raise ImportError('httpx is not installed')


sys.meta_path.insert(0, BlockHttpx())

import asyncio

from pydantic_ai import Agent
from pydantic_ai.agent.spec import AgentSpec
from pydantic_ai.models.function import FunctionModel

assert Agent('test').run_sync('hello').output

# `ModelSettings` is a field type on several pydantic models, so building one resolves this
# module's annotations — including `timeout`'s `httpx.Timeout` arm.
assert AgentSpec.model_json_schema_with_capabilities()

# `Provider` is public and subclassable, so its `httpx`-typed attributes have to stay resolvable.
import typing

from pydantic_ai.providers import Provider

assert typing.get_type_hints(Provider)


async def stream_function(messages, info):
    for word in ['hello ', 'world']:
        yield word


async def break_out_of_stream():
    async with Agent(FunctionModel(stream_function=stream_function)).run_stream('hello') as result:
        async for _ in result.stream_output():
            break


asyncio.run(break_out_of_stream())

assert 'httpx' not in sys.modules, 'the SDK-less core imported httpx'
"""


def test_core_runs_without_httpx() -> None:
    """`pydantic-ai-slim` with no provider SDK installed must import and run without `httpx`.

    `httpx` is not in its dependencies — it comes transitively from whichever SDK the user
    installs — so anything in the core that reaches for it at runtime is a packaging bug that
    only surfaces for the slimmest installs.

    The early `break` covers the subtler half. `StreamedResponse`'s cancel guard names its
    suppressed errors as `except self.get_stream_cancel_errors():`, and Python evaluates that
    expression for *every* exception passing through — including the `GeneratorExit` an ordinary
    early `break` raises. `FunctionModel` doesn't override the method, so the `break` lands on the
    base default and on the continuation composite that folds it in.

    `stderr` is asserted because that failure is silent in the exit code: the errors surface while
    `asyncio.run` finalizes async generators, which logs them and still exits 0.
    """
    result = subprocess.run(
        [sys.executable, '-c', textwrap.dedent(_HTTPX_FREE_CORE)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert 'httpx' not in result.stderr, result.stderr
