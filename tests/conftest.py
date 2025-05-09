from __future__ import annotations as _annotations

import asyncio
import importlib.util
import os
import re
import secrets
import sys
from collections.abc import AsyncIterator, Iterator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Any, Callable

import httpx
import pytest
from _pytest.assertion.rewrite import AssertionRewritingHook
from pytest_mock import MockerFixture
from typing_extensions import TypeAlias
from vcr import VCR

import pydantic_ai.models
from pydantic_ai.messages import BinaryContent
from pydantic_ai.models import Model, cached_async_http_client

__all__ = 'IsDatetime', 'IsFloat', 'IsNow', 'IsStr', 'IsInt', 'TestEnv', 'ClientWithHandler', 'try_import'


pydantic_ai.models.ALLOW_MODEL_REQUESTS = False

if TYPE_CHECKING:
    from pydantic_ai.providers.bedrock import BedrockProvider

    def IsDatetime(*args: Any, **kwargs: Any) -> datetime: ...
    def IsFloat(*args: Any, **kwargs: Any) -> float: ...
    def IsInt(*args: Any, **kwargs: Any) -> int: ...
    def IsNow(*args: Any, **kwargs: Any) -> datetime: ...
    def IsStr(*args: Any, **kwargs: Any) -> str: ...
else:
    from dirty_equals import IsDatetime, IsFloat, IsInt, IsNow as _IsNow, IsStr

    def IsNow(*args: Any, **kwargs: Any):
        # Increase the default value of `delta` to 10 to reduce test flakiness on overburdened machines
        if 'delta' not in kwargs:  # pragma: no branch
            kwargs['delta'] = 10
        return _IsNow(*args, **kwargs)


class TestEnv:
    __test__ = False

    def __init__(self):
        self.envars: dict[str, str | None] = {}

    def set(self, name: str, value: str) -> None:
        self.envars[name] = os.getenv(name)
        os.environ[name] = value

    def remove(self, name: str) -> None:
        self.envars[name] = os.environ.pop(name, None)

    def reset(self) -> None:
        for name, value in self.envars.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


@pytest.fixture
def env() -> Iterator[TestEnv]:
    test_env = TestEnv()

    yield test_env

    test_env.reset()


@pytest.fixture
def anyio_backend():
    return 'asyncio'


@pytest.fixture
def allow_model_requests():
    with pydantic_ai.models.override_allow_model_requests(True):
        yield


@pytest.fixture
async def client_with_handler() -> AsyncIterator[ClientWithHandler]:
    client: httpx.AsyncClient | None = None

    def create_client(handler: Callable[[httpx.Request], httpx.Response]) -> httpx.AsyncClient:
        nonlocal client
        assert client is None, 'client_with_handler can only be called once'
        client = httpx.AsyncClient(mounts={'all://': httpx.MockTransport(handler)})
        return client

    try:
        yield create_client
    finally:
        if client:
            await client.aclose()


ClientWithHandler: TypeAlias = Callable[[Callable[[httpx.Request], httpx.Response]], httpx.AsyncClient]


# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false
@pytest.fixture
def create_module(tmp_path: Path, request: pytest.FixtureRequest) -> Callable[[str], Any]:
    """Taken from `pydantic/tests/conftest.py`, create module object, execute and return it."""

    def run(
        source_code: str,
        rewrite_assertions: bool = True,
        module_name_prefix: str | None = None,
    ) -> ModuleType:
        """Create module object, execute and return it.

        Can be used as a decorator of the function from the source code of which the module will be constructed.

        Args:
            source_code: Python source code of the module
            rewrite_assertions: whether to rewrite assertions in module or not
            module_name_prefix: string prefix to use in the name of the module, does not affect the name of the file.

        """

        # Max path length in Windows is 260. Leaving some buffer here
        max_name_len = 240 - len(str(tmp_path))
        # Windows does not allow these characters in paths. Linux bans slashes only.
        sanitized_name = re.sub('[' + re.escape('<>:"/\\|?*') + ']', '-', request.node.name)[:max_name_len]
        module_name = f'{sanitized_name}_{secrets.token_hex(5)}'
        path = tmp_path / f'{module_name}.py'
        path.write_text(source_code)
        filename = str(path)

        if module_name_prefix:  # pragma: no cover
            module_name = module_name_prefix + module_name

        if rewrite_assertions:
            loader = AssertionRewritingHook(config=request.config)
            loader.mark_rewrite(module_name)
        else:  # pragma: no cover
            loader = None

        spec = importlib.util.spec_from_file_location(module_name, filename, loader=loader)
        sys.modules[module_name] = module = importlib.util.module_from_spec(spec)  # pyright: ignore[reportArgumentType]
        spec.loader.exec_module(module)  # pyright: ignore[reportOptionalMemberAccess]
        return module

    return run


@contextmanager
def try_import() -> Iterator[Callable[[], bool]]:
    import_success = False

    def check_import() -> bool:
        return import_success

    try:
        yield check_import
    except ImportError:
        pass
    else:
        import_success = True


@pytest.fixture(autouse=True)
def set_event_loop() -> Iterator[None]:
    new_loop = asyncio.new_event_loop()
    asyncio.set_event_loop(new_loop)
    yield
    new_loop.close()


def raise_if_exception(e: Any) -> None:
    if isinstance(e, Exception):
        raise e


def pytest_recording_configure(config: Any, vcr: VCR):
    from . import json_body_serializer

    vcr.register_serializer('yaml', json_body_serializer)


@pytest.fixture(scope='module')
def vcr_config():
    return {
        'ignore_localhost': True,
        # Note: additional header filtering is done inside the serializer
        'filter_headers': ['authorization', 'x-api-key'],
        'decode_compressed_response': True,
    }


@pytest.fixture(autouse=True)
async def close_cached_httpx_client() -> AsyncIterator[None]:
    yield
    for provider in [
        'openai',
        'anthropic',
        'azure',
        'google-gla',
        'google-vertex',
        'groq',
        'mistral',
        'cohere',
        'deepseek',
        None,
    ]:
        await cached_async_http_client(provider=provider).aclose()


@pytest.fixture(scope='session')
def assets_path() -> Path:
    return Path(__file__).parent / 'assets'


@pytest.fixture(scope='session')
def audio_content(assets_path: Path) -> BinaryContent:
    audio_bytes = assets_path.joinpath('marcelo.mp3').read_bytes()
    return BinaryContent(data=audio_bytes, media_type='audio/mpeg')


@pytest.fixture(scope='session')
def image_content(assets_path: Path) -> BinaryContent:
    image_bytes = assets_path.joinpath('kiwi.png').read_bytes()
    return BinaryContent(data=image_bytes, media_type='image/png')


@pytest.fixture(scope='session')
def video_content(assets_path: Path) -> BinaryContent:
    video_bytes = assets_path.joinpath('small_video.mp4').read_bytes()
    return BinaryContent(data=video_bytes, media_type='video/mp4')


@pytest.fixture(scope='session')
def document_content(assets_path: Path) -> BinaryContent:
    pdf_bytes = assets_path.joinpath('dummy.pdf').read_bytes()
    return BinaryContent(data=pdf_bytes, media_type='application/pdf')


@pytest.fixture(scope='session')
def openai_api_key() -> str:
    return os.getenv('OPENAI_API_KEY', 'mock-api-key')


@pytest.fixture(scope='session')
def gemini_api_key() -> str:
    return os.getenv('GEMINI_API_KEY', 'mock-api-key')


@pytest.fixture(scope='session')
def groq_api_key() -> str:
    return os.getenv('GROQ_API_KEY', 'mock-api-key')


@pytest.fixture(scope='session')
def anthropic_api_key() -> str:
    return os.getenv('ANTHROPIC_API_KEY', 'mock-api-key')


@pytest.fixture(scope='session')
def co_api_key() -> str:
    return os.getenv('CO_API_KEY', 'mock-api-key')


@pytest.fixture(scope='session')
def mistral_api_key() -> str:
    return os.getenv('MISTRAL_API_KEY', 'mock-api-key')


@pytest.fixture(scope='session')
def bedrock_provider():
    try:
        import boto3

        from pydantic_ai.providers.bedrock import BedrockProvider

        bedrock_client = boto3.client(
            'bedrock-runtime',
            region_name=os.getenv('AWS_REGION', 'us-east-1'),
            aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID', 'AKIA6666666666666666'),
            aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY', '6666666666666666666666666666666666666666'),
        )
        yield BedrockProvider(bedrock_client=bedrock_client)
        bedrock_client.close()
    except ImportError:
        pytest.skip('boto3 is not installed')


@pytest.fixture()
def model(
    request: pytest.FixtureRequest,
    openai_api_key: str,
    anthropic_api_key: str,
    mistral_api_key: str,
    groq_api_key: str,
    co_api_key: str,
    gemini_api_key: str,
    bedrock_provider: BedrockProvider,
) -> Model:  # pragma: lax no cover
    try:
        if request.param == 'openai':
            from pydantic_ai.models.openai import OpenAIModel
            from pydantic_ai.providers.openai import OpenAIProvider

            return OpenAIModel('o3-mini', provider=OpenAIProvider(api_key=openai_api_key))
        elif request.param == 'anthropic':
            from pydantic_ai.models.anthropic import AnthropicModel
            from pydantic_ai.providers.anthropic import AnthropicProvider

            return AnthropicModel('claude-3-5-sonnet-latest', provider=AnthropicProvider(api_key=anthropic_api_key))
        elif request.param == 'mistral':
            from pydantic_ai.models.mistral import MistralModel
            from pydantic_ai.providers.mistral import MistralProvider

            return MistralModel('ministral-8b-latest', provider=MistralProvider(api_key=mistral_api_key))
        elif request.param == 'groq':
            from pydantic_ai.models.groq import GroqModel
            from pydantic_ai.providers.groq import GroqProvider

            return GroqModel('llama3-8b-8192', provider=GroqProvider(api_key=groq_api_key))
        elif request.param == 'cohere':
            from pydantic_ai.models.cohere import CohereModel
            from pydantic_ai.providers.cohere import CohereProvider

            return CohereModel('command-r-plus', provider=CohereProvider(api_key=co_api_key))
        elif request.param == 'gemini':
            from pydantic_ai.models.gemini import GeminiModel
            from pydantic_ai.providers.google_gla import GoogleGLAProvider

            return GeminiModel('gemini-1.5-flash', provider=GoogleGLAProvider(api_key=gemini_api_key))
        elif request.param == 'bedrock':
            from pydantic_ai.models.bedrock import BedrockConverseModel

            return BedrockConverseModel('us.amazon.nova-micro-v1:0', provider=bedrock_provider)
        else:
            raise ValueError(f'Unknown model: {request.param}')
    except ImportError:
        pytest.skip(f'{request.param} is not installed')


@pytest.fixture
def mock_snapshot_id(mocker: MockerFixture):
    i = 0

    def generate_snapshot_id(node_id: str) -> str:
        nonlocal i
        i += 1
        return f'{node_id}:{i}'

    return mocker.patch('pydantic_graph.nodes.generate_snapshot_id', side_effect=generate_snapshot_id)
