from __future__ import annotations as _annotations

import asyncio
import importlib.util
import logging
import os
import re
import secrets
import sys
from collections.abc import AsyncIterator, Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from functools import cached_property
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Any, TypeAlias, cast

import httpx
import pytest
from _pytest.assertion.rewrite import AssertionRewritingHook
from pytest_mock import MockerFixture
from vcr import VCR, request as vcr_request

import pydantic_ai.models
from pydantic_ai import Agent, BinaryContent
from pydantic_ai.models import Model, cached_async_http_client

__all__ = (
    'IsDatetime',
    'IsFloat',
    'IsNow',
    'IsStr',
    'IsBytes',
    'IsInt',
    'IsInstance',
    'TestEnv',
    'ClientWithHandler',
    'try_import',
)

# Configure VCR logger to WARNING as it is too verbose by default
# specifically, it logs every request and response including binary
# content in Cassette.append, which is causing log downloads from
# GitHub action to fail.
logging.getLogger('vcr.cassette').setLevel(logging.WARNING)

pydantic_ai.models.ALLOW_MODEL_REQUESTS = False

if TYPE_CHECKING:
    from typing import TypeVar

    from pydantic_ai.providers.bedrock import BedrockProvider

    T = TypeVar('T')

    def IsInstance(arg: type[T]) -> T: ...
    def IsDatetime(*args: Any, **kwargs: Any) -> datetime: ...
    def IsFloat(*args: Any, **kwargs: Any) -> float: ...
    def IsInt(*args: Any, **kwargs: Any) -> int: ...
    def IsNow(*args: Any, **kwargs: Any) -> datetime: ...
    def IsStr(*args: Any, **kwargs: Any) -> str: ...
    def IsSameStr(*args: Any, **kwargs: Any) -> str: ...
    def IsBytes(*args: Any, **kwargs: Any) -> bytes: ...
else:
    from dirty_equals import IsBytes, IsDatetime, IsFloat, IsInstance, IsInt, IsNow as _IsNow, IsStr

    def IsNow(*args: Any, **kwargs: Any):
        # Increase the default value of `delta` to 10 to reduce test flakiness on overburdened machines
        if 'delta' not in kwargs:  # pragma: no branch
            kwargs['delta'] = 10
        return _IsNow(*args, **kwargs)

    class IsSameStr(IsStr):
        """
        Checks if the value is a string, and that subsequent uses have the same value as the first one.

        Example:
        ```python {test="skip"}
        assert events == [
            {
                'type': 'RUN_STARTED',
                'threadId': (thread_id := IsSameStr()),
                'runId': (run_id := IsSameStr()),
            },
            {'type': 'TEXT_MESSAGE_START', 'messageId': (message_id := IsSameStr()), 'role': 'assistant'},
            {'type': 'TEXT_MESSAGE_CONTENT', 'messageId': message_id, 'delta': 'success '},
            {
                'type': 'TEXT_MESSAGE_CONTENT',
                'messageId': message_id,
                'delta': '(no tool calls)',
            },
            {'type': 'TEXT_MESSAGE_END', 'messageId': message_id},
            {
                'type': 'RUN_FINISHED',
                'threadId': thread_id,
                'runId': run_id,
            },
        ]
        ```
        """

        _first_other: str | None = None

        def equals(self, other: Any) -> bool:
            if self._first_other is None:
                self._first_other = other
                return super().equals(other)
            else:
                return other == self._first_other


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
                os.environ[name] = value  # pragma: lax no cover


@pytest.fixture
def env() -> Iterator[TestEnv]:
    test_env = TestEnv()

    yield test_env

    test_env.reset()


@pytest.fixture(scope='session')
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
        if client:  # pragma: no branch
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


@pytest.fixture(scope='session', autouse=True)
def event_loop() -> Iterator[None]:
    new_loop = asyncio.new_event_loop()
    asyncio.set_event_loop(new_loop)
    yield
    new_loop.close()


@pytest.fixture(autouse=True)
def no_instrumentation_by_default():
    Agent.instrument_all(False)


try:
    import logfire

    logfire.DEFAULT_LOGFIRE_INSTANCE.config.ignore_no_config = True

    @pytest.fixture(autouse=True)
    def fresh_logfire():
        logfire.shutdown(flush=False)

except ImportError:
    pass


def raise_if_exception(e: Any) -> None:
    if isinstance(e, Exception):
        raise e


def pytest_recording_configure(config: Any, vcr: VCR):
    from . import json_body_serializer

    vcr.register_serializer('yaml', json_body_serializer)

    def method_matcher(r1: vcr_request.Request, r2: vcr_request.Request) -> None:
        if r1.method.upper() != r2.method.upper():
            raise AssertionError(f'{r1.method} != {r2.method}')

    vcr.register_matcher('method', method_matcher)


@pytest.fixture(autouse=True)
def mock_vcr_aiohttp_content(mocker: MockerFixture):
    try:
        from vcr.stubs import aiohttp_stubs
    except ImportError:  # pragma: lax no cover
        return

    # google-genai calls `self.response_stream.content.readline()` where `self.response_stream` is a `MockClientResponse`,
    # which creates a new `MockStream` each time instead of returning the same one, resulting in the readline cursor not being respected.
    # So we turn `content` into a cached property to return the same one each time.
    # VCR issue: https://github.com/kevin1024/vcrpy/issues/927. Once that's is resolved, we can remove this patch.
    cached_content = cached_property(aiohttp_stubs.MockClientResponse.content.fget)  # type: ignore
    cached_content.__set_name__(aiohttp_stubs.MockClientResponse, 'content')
    mocker.patch('vcr.stubs.aiohttp_stubs.MockClientResponse.content', new=cached_content)
    mocker.patch('vcr.stubs.aiohttp_stubs.MockStream.set_exception', return_value=None)


@pytest.fixture(scope='module')
def vcr_config():
    return {
        'ignore_localhost': True,
        # Note: additional header filtering is done inside the serializer
        'filter_headers': ['authorization', 'x-api-key'],
        'decode_compressed_response': True,
    }


@pytest.fixture(autouse=True)
async def close_cached_httpx_client(anyio_backend: str) -> AsyncIterator[None]:
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
def text_document_content(assets_path: Path) -> BinaryContent:
    content = assets_path.joinpath('dummy.txt').read_text()
    bin_content = BinaryContent(data=content.encode(), media_type='text/plain')
    return bin_content


@pytest.fixture(scope='session')
def deepseek_api_key() -> str:
    return os.getenv('DEEPSEEK_API_KEY', 'mock-api-key')


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
def openrouter_api_key() -> str:
    return os.getenv('OPENROUTER_API_KEY', 'mock-api-key')


@pytest.fixture(scope='session')
def huggingface_api_key() -> str:
    return os.getenv('HF_TOKEN', 'hf_token')


@pytest.fixture(scope='session')
def heroku_inference_key() -> str:
    return os.getenv('HEROKU_INFERENCE_KEY', 'mock-api-key')


@pytest.fixture(scope='session')
def cerebras_api_key() -> str:
    return os.getenv('CEREBRAS_API_KEY', 'mock-api-key')


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
            aws_session_token=os.getenv('AWS_SESSION_TOKEN', None),
        )
        yield BedrockProvider(bedrock_client=bedrock_client)
        bedrock_client.close()
    except ImportError:  # pragma: lax no cover
        pytest.skip('boto3 is not installed')


@pytest.fixture()
def vertex_provider_auth(mocker: MockerFixture) -> None:  # pragma: lax no cover
    # Locally, we authenticate via `gcloud` CLI, so we don't need to patch anything.
    if not os.getenv('CI', False):
        return  # pragma: lax no cover

    try:
        from google.genai import _api_client
    except ImportError:
        return  # do nothing if this isn't installed

    @dataclass
    class NoOpCredentials:
        token = 'my-token'
        quota_project_id = 'pydantic-ai'

        def refresh(self, request: httpx.Request): ...

        def expired(self) -> bool:
            return False

    return_value = (NoOpCredentials(), 'pydantic-ai')
    mocker.patch.object(_api_client, 'load_auth', return_value=return_value)


@pytest.fixture()
async def vertex_provider(vertex_provider_auth: None):  # pragma: lax no cover
    # NOTE: You need to comment out this line to rewrite the cassettes locally.
    if not os.getenv('CI', False):
        pytest.skip('Requires properly configured local google vertex config to pass')

    try:
        from pydantic_ai.providers.google import GoogleProvider, VertexAILocation
    except ImportError:  # pragma: lax no cover
        pytest.skip('google is not installed')

    project = os.getenv('GOOGLE_PROJECT', 'pydantic-ai')
    location = os.getenv('GOOGLE_LOCATION', 'global')
    yield GoogleProvider(project=project, location=cast(VertexAILocation, location))


@pytest.fixture()
def model(
    request: pytest.FixtureRequest,
    openai_api_key: str,
    anthropic_api_key: str,
    mistral_api_key: str,
    groq_api_key: str,
    co_api_key: str,
    gemini_api_key: str,
    huggingface_api_key: str,
    bedrock_provider: BedrockProvider,
) -> Model:  # pragma: lax no cover
    try:
        if request.param == 'test':
            from pydantic_ai.models.test import TestModel

            return TestModel()
        elif request.param == 'openai':
            from pydantic_ai.models.openai import OpenAIChatModel
            from pydantic_ai.providers.openai import OpenAIProvider

            return OpenAIChatModel('o3-mini', provider=OpenAIProvider(api_key=openai_api_key))
        elif request.param == 'anthropic':
            from pydantic_ai.models.anthropic import AnthropicModel
            from pydantic_ai.providers.anthropic import AnthropicProvider

            return AnthropicModel('claude-sonnet-4-5', provider=AnthropicProvider(api_key=anthropic_api_key))
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
            from pydantic_ai.models.gemini import GeminiModel  # type: ignore[reportDeprecated]
            from pydantic_ai.providers.google_gla import GoogleGLAProvider  # type: ignore[reportDeprecated]

            return GeminiModel('gemini-1.5-flash', provider=GoogleGLAProvider(api_key=gemini_api_key))  # type: ignore[reportDeprecated]
        elif request.param == 'google':
            from pydantic_ai.models.google import GoogleModel
            from pydantic_ai.providers.google import GoogleProvider

            return GoogleModel('gemini-1.5-flash', provider=GoogleProvider(api_key=gemini_api_key))
        elif request.param == 'bedrock':
            from pydantic_ai.models.bedrock import BedrockConverseModel

            return BedrockConverseModel('us.amazon.nova-micro-v1:0', provider=bedrock_provider)
        elif request.param == 'huggingface':
            from pydantic_ai.models.huggingface import HuggingFaceModel
            from pydantic_ai.providers.huggingface import HuggingFaceProvider

            return HuggingFaceModel(
                'Qwen/Qwen2.5-72B-Instruct',
                provider=HuggingFaceProvider(provider_name='nebius', api_key=huggingface_api_key),
            )
        elif request.param == 'outlines':
            from outlines.models.transformers import from_transformers
            from transformers import AutoModelForCausalLM, AutoTokenizer

            from pydantic_ai.models.outlines import OutlinesModel

            return OutlinesModel(
                from_transformers(
                    AutoModelForCausalLM.from_pretrained('erwanf/gpt2-mini'),
                    AutoTokenizer.from_pretrained('erwanf/gpt2-mini'),
                )
            )
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
