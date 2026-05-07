from __future__ import annotations as _annotations

import asyncio
import importlib.util
import logging
import os
import re
import secrets
import sys
from collections.abc import AsyncIterator, Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from functools import cached_property
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Any, TypeAlias, TypeVar, cast, overload

import httpx
import pytest
from _pytest.assertion.rewrite import AssertionRewritingHook
from pytest_mock import MockerFixture
from vcr import VCR, request as vcr_request

import pydantic_ai.models
from pydantic_ai import Agent, BinaryContent, BinaryImage, Embedder
from pydantic_ai.messages import (
    BuiltinToolCallPart,
    BuiltinToolReturnPart,
    DocumentUrl,
    FilePart,
    ImageUrl,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
    SystemPromptPart,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
    VideoUrl,
)
from pydantic_ai.models import DEFAULT_HTTP_TIMEOUT, Model

from ._inline_snapshot import Builder, Custom, customize

__all__ = (
    'IsDatetime',
    'IsFloat',
    'IsNow',
    'IsStr',
    'IsBytes',
    'IsInt',
    'IsInstance',
    'IsList',
    'TestEnv',
    'ClientWithHandler',
    'try_import',
    'SNAPSHOT_BYTES_COLLAPSE_THRESHOLD',
)

# Configure VCR logger to WARNING as it is too verbose by default
# specifically, it logs every request and response including binary
# content in Cassette.append, which is causing log downloads from
# GitHub action to fail.
logging.getLogger('vcr.cassette').setLevel(logging.WARNING)

pydantic_ai.models.ALLOW_MODEL_REQUESTS = False

os.environ.setdefault('HF_HUB_DISABLE_PROGRESS_BARS', '1')

if TYPE_CHECKING:
    from typing import TypeVar

    from pydantic_ai.providers.bedrock import BedrockProvider
    from pydantic_ai.providers.xai import XaiProvider

    T = TypeVar('T')

    def IsInstance(arg: type[T]) -> T: ...
    def IsDatetime(*args: Any, **kwargs: Any) -> datetime: ...
    def IsFloat(*args: Any, **kwargs: Any) -> float: ...
    def IsInt(*args: Any, **kwargs: Any) -> int: ...
    def IsNow(*args: Any, **kwargs: Any) -> datetime: ...
    def IsStr(*args: Any, **kwargs: Any) -> str: ...
    def IsSameStr(*args: Any, **kwargs: Any) -> str: ...
    def IsBytes(*args: Any, **kwargs: Any) -> bytes: ...
    def IsList(*args: T, **kwargs: Any) -> list[T]: ...
else:
    from dirty_equals import IsBytes, IsDatetime, IsFloat, IsInstance, IsInt, IsList, IsNow as _IsNow, IsStr

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


SNAPSHOT_BYTES_COLLAPSE_THRESHOLD = 50


def sanitize_filename(name: str, max_len: int) -> str:
    """Sanitize a string for safe use as a filename across platforms."""
    # Windows does not allow these characters in paths. Linux bans slashes only.
    return re.sub('[' + re.escape('<>:"/\\|?*') + ']', '-', name)[:max_len]


@customize
def binary_handler(value: Any) -> Any | None:  # pragma: no cover
    # Use IsBytes() for large byte sequences in snapshots.
    if isinstance(value, bytes) and len(value) > SNAPSHOT_BYTES_COLLAPSE_THRESHOLD:
        return IsBytes()


@customize
def isdatetime_handler(value: Any, builder: Builder) -> Any | None:  # pragma: no cover
    # Use IsDatetime() for datetime values in snapshots.
    if isinstance(value, datetime):
        return IsDatetime()


@customize
def content_handler(value: Any, builder: Builder) -> Custom | None:  # pragma: no cover
    # special handler for types which need an identifier argument for __init__ but declare an _identifier in the class
    if isinstance(value, BinaryImage):
        return builder.create_call(
            BinaryImage,
            [],
            {
                # prevent generation of IsBytes() because it does not work together with Pydantic models
                'data': builder.create_code(f'{value.data!r}'),
                'media_type': value.media_type,
                'identifier': builder.with_default(value.identifier, None),
                'vendor_metadata': builder.with_default(value.vendor_metadata, None),
                # kind is always "binary"
                # 'kind': builder.with_default(value.kind, 'binary'),
            },
        )

    if isinstance(value, BinaryContent):
        return builder.create_call(
            BinaryContent,
            [],
            {
                # prevent generation of IsBytes() because it does not work together with Pydantic models
                'data': builder.create_code(f'{value.data!r}'),
                'media_type': value.media_type,
                'identifier': builder.with_default(value.identifier, None),
                'vendor_metadata': builder.with_default(value.vendor_metadata, None),
                'kind': builder.with_default(value.kind, 'binary'),
            },
        )

    for cls, kind in [(VideoUrl, 'video-url'), (DocumentUrl, 'document-url'), (ImageUrl, 'image-url')]:
        if type(value) is cls:
            return builder.create_call(
                cls,
                [],
                {
                    'url': value.url,
                    'media_type': builder.with_default(value.media_type, None),
                    # TODO: identifier is not used for == comparison should we ignore it?
                    'identifier': builder.with_default(value.identifier, None),
                    'force_download': builder.with_default(value.force_download, False),
                    'vendor_metadata': builder.with_default(value.vendor_metadata, None),
                    'kind': builder.with_default(value.kind, kind),
                },
            )


@customize
def variable_handler(value: Any, builder: Builder, local_vars: dict[str, Any]) -> Custom | None:  # pragma: no cover
    for name, local_variable in local_vars.items():
        # use local_function.__qualname__ when there exist a local_function with the wanted name
        if hasattr(local_variable, '__qualname__') and value == local_variable.__qualname__:
            return builder.create_code(f'{name}.__qualname__')

        # use `part.tool_call_id` when there is a local variable part with the wanted id
        if name == 'part' and hasattr(local_variable, 'tool_call_id') and local_variable.tool_call_id == value:
            return builder.create_code(f'{name}.tool_call_id')

        # skip IsSameStr variables that haven't been compared yet (no value captured)
        if hasattr(local_variable, '_first_other') and local_variable._first_other is None:
            continue

        # uses local variables like part* *_content or thread_id when their value is equal to the wanted value in the snapshot
        if (
            (name.startswith('part') or name.endswith('_content') or name in ('thread_id',))
            and name != 'parts'
            and local_variable == value
        ):
            return builder.create_code(name)

        # match the local var like `var := IsSameStr()` a second time
        if type(local_variable) is IsSameStr and local_variable == value:
            return builder.create_code(name)


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
        sanitized_name = sanitize_filename(request.node.name, max_name_len)
        module_name = f'{sanitized_name}_{secrets.token_hex(5)}'
        path = tmp_path / f'{module_name}.py'
        path.write_text(source_code, encoding='utf-8')
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
    Embedder.instrument_all(False)


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


_AWS_ACCOUNT_ID_IN_ARN = re.compile(r'(arn(?:%3A|:)aws(?:%3A|:)bedrock(?:%3A|:)[^:%]*(?:%3A|:))\d{12}((?:%3A|:))')
_SCRUBBED_AWS_ACCOUNT_ID = r'\g<1>123456789012\2'


def pytest_recording_configure(config: Any, vcr: VCR):
    from . import json_body_serializer

    vcr.register_serializer('yaml', json_body_serializer)

    def method_matcher(r1: vcr_request.Request, r2: vcr_request.Request) -> None:
        if r1.method.upper() != r2.method.upper():
            raise AssertionError(f'{r1.method} != {r2.method}')

    def path_matcher(r1: vcr_request.Request, r2: vcr_request.Request) -> None:
        """Match URL paths after scrubbing AWS account IDs from ARNs."""
        path1 = _AWS_ACCOUNT_ID_IN_ARN.sub(_SCRUBBED_AWS_ACCOUNT_ID, r1.path)
        path2 = _AWS_ACCOUNT_ID_IN_ARN.sub(_SCRUBBED_AWS_ACCOUNT_ID, r2.path)
        if path1 != path2:
            raise AssertionError(f'{path1} != {path2}')

    vcr.register_matcher('method', method_matcher)
    vcr.register_matcher('path', path_matcher)

    def scrub_aws_account_id(request: vcr_request.Request) -> vcr_request.Request:
        request.uri = _AWS_ACCOUNT_ID_IN_ARN.sub(_SCRUBBED_AWS_ACCOUNT_ID, request.uri)
        return request

    vcr.before_record_request = scrub_aws_account_id


def pytest_addoption(parser: Any) -> None:
    parser.addoption(
        '--xai-proto-include-json',
        action='store_true',
        default=True,
        dest='xai_proto_include_json',
        help='Include JSON representations in xAI proto cassette YAML files.',
    )
    parser.addoption(
        '--run-gateway-live',
        action='store_true',
        default=False,
        help='Run live gateway smoke tests that make real paid model requests.',
    )


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
        'filter_headers': ['authorization', 'x-api-key', 'cookie'],
        'decode_compressed_response': True,
    }


_HttpClientCache: TypeAlias = 'dict[tuple[int, int], httpx.AsyncClient]'


@pytest.fixture(autouse=True)
def track_httpx_clients(monkeypatch: pytest.MonkeyPatch) -> Iterator[_HttpClientCache]:
    """Monkeypatch `create_async_http_client` in all loaded modules and track created clients.

    Within a single test, calls with the same (timeout, connect) args reuse the same
    httpx.AsyncClient. On teardown, all clients are closed — no process-global state leaks.

    This is a sync fixture so it applies to both sync and async tests. For async tests, the
    companion ``close_httpx_clients`` fixture handles async cleanup first.
    """
    cache: _HttpClientCache = {}
    original = pydantic_ai.models.create_async_http_client

    def cached_per_test(**kwargs: Any) -> httpx.AsyncClient:
        key = (kwargs.get('timeout', DEFAULT_HTTP_TIMEOUT), kwargs.get('connect', 5))
        if key not in cache or cache[key].is_closed:
            cache[key] = original(**kwargs)
        return cache[key]

    for mod in list(sys.modules.values()):
        if getattr(mod, 'create_async_http_client', None) is original:
            monkeypatch.setattr(mod, 'create_async_http_client', cached_per_test)

    yield cache

    unclosed = [c for c in cache.values() if not c.is_closed]
    if unclosed:  # pragma: no cover

        async def _close_all() -> None:
            for client in unclosed:
                await client.aclose()

        asyncio.run(_close_all())


@pytest.fixture(autouse=True)
async def close_httpx_clients(anyio_backend: str, track_httpx_clients: _HttpClientCache) -> AsyncIterator[None]:
    """Close tracked HTTP clients after async tests."""
    yield
    for client in track_httpx_clients.values():
        if not client.is_closed:
            await client.aclose()


@pytest.fixture(autouse=True, scope='session')
def patch_google_genai_gc_crash():
    """Work around google-genai BaseApiClient GC crash.

    BaseApiClient.__del__ schedules aclose() during GC, which crashes when the
    object was only partially initialized (_async_httpx_client never set).
    Remove when https://github.com/googleapis/python-genai/issues/2023 closes.
    """
    try:
        from google.genai._api_client import BaseApiClient
    except ImportError:
        yield
        return

    original_aclose = BaseApiClient.aclose

    async def safe_aclose(self: BaseApiClient) -> None:
        if hasattr(self, '_async_httpx_client'):
            await original_aclose(self)
        else:  # pragma: lax no cover
            # In some test runs, the `if` above will always run, so we get an `if -> exit` branch coverage miss.
            # This is a workaround to specify that the `else` branch may not be hit (as we don't have `lax no branch`)
            pass

    BaseApiClient.aclose = safe_aclose
    yield
    BaseApiClient.aclose = original_aclose


@pytest.fixture(scope='session')
def assets_path() -> Path:
    return Path(__file__).parent / 'assets'


@pytest.fixture(scope='session')
def audio_content(assets_path: Path) -> BinaryContent:
    audio_bytes = assets_path.joinpath('marcelo.mp3').read_bytes()
    return BinaryContent(data=audio_bytes, media_type='audio/mpeg')


@pytest.fixture(scope='session')
def image_content(assets_path: Path) -> BinaryImage:
    image_bytes = assets_path.joinpath('kiwi.jpg').read_bytes()
    return BinaryImage(data=image_bytes, media_type='image/jpeg')


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
    content = assets_path.joinpath('dummy.txt').read_text(encoding='utf-8')
    bin_content = BinaryContent(data=content.encode(), media_type='text/plain')
    return bin_content


os.environ.pop('OPENAI_BASE_URL', None)
os.environ.pop('ANTHROPIC_BASE_URL', None)


@pytest.fixture(scope='session')
def deepseek_api_key() -> str:
    return os.getenv('DEEPSEEK_API_KEY', 'mock-api-key')


@pytest.fixture(scope='session')
def openai_api_key() -> str:
    return os.getenv('OPENAI_API_KEY', 'mock-api-key')


@pytest.fixture(scope='session')
def gemini_api_key() -> str:
    return os.getenv('GEMINI_API_KEY', os.getenv('GOOGLE_API_KEY', 'mock-api-key'))


@pytest.fixture(scope='session')
def groq_api_key() -> str:
    return os.getenv('GROQ_API_KEY', 'mock-api-key')


@pytest.fixture(scope='session')
def anthropic_api_key() -> str:
    return os.getenv('ANTHROPIC_API_KEY', 'mock-api-key')


@pytest.fixture(scope='session')
def gateway_api_key() -> str | None:
    return os.getenv('PYDANTIC_AI_GATEWAY_API_KEY', os.getenv('PAIG_API_KEY'))


@pytest.fixture(scope='session')
def co_api_key() -> str:
    return os.getenv('CO_API_KEY', 'mock-api-key')


@pytest.fixture(scope='session')
def voyage_api_key() -> str:
    return os.getenv('VOYAGE_API_KEY', 'mock-api-key')


@pytest.fixture(scope='session')
def mistral_api_key() -> str:
    return os.getenv('MISTRAL_API_KEY', 'mock-api-key')


@pytest.fixture(scope='session')
def openrouter_api_key() -> str:
    return os.getenv('OPENROUTER_API_KEY', 'mock-api-key')


@pytest.fixture(scope='session')
def ollama_api_key() -> str:
    return os.getenv('OLLAMA_API_KEY', 'mock-api-key')


@pytest.fixture(scope='session')
def huggingface_api_key() -> str:
    return os.getenv('HF_TOKEN', 'hf_token')


@pytest.fixture(autouse=True, scope='session')
def _patch_hf_provider_mappings():
    """Populate the SDK's hardcoded model mappings to avoid sync HTTP calls during VCR tests.

    The HuggingFace SDK makes a synchronous HTTP call to resolve provider mappings at request time,
    which is incompatible with VCR's async test infrastructure.
    """
    try:
        from huggingface_hub.hf_api import InferenceProviderMapping
        from huggingface_hub.inference._providers._common import HARDCODED_MODEL_INFERENCE_MAPPING
    except ImportError:
        return

    models: list[tuple[str, str, str]] = [
        ('together', 'deepseek-ai/DeepSeek-R1', 'conversational'),
        ('nebius', 'Qwen/Qwen2.5-VL-72B-Instruct', 'conversational'),
        ('nebius', 'Qwen/Qwen2.5-72B-Instruct', 'conversational'),
    ]

    for provider, model_id, task in models:
        HARDCODED_MODEL_INFERENCE_MAPPING[provider][model_id] = InferenceProviderMapping(
            provider=provider,
            hf_model_id=model_id,
            providerId=model_id,
            status='live',
            task=task,
        )


@pytest.fixture(scope='session')
def heroku_inference_key() -> str:
    return os.getenv('HEROKU_INFERENCE_KEY', 'mock-api-key')


@pytest.fixture(scope='session')
def cerebras_api_key() -> str:
    return os.getenv('CEREBRAS_API_KEY', 'mock-api-key')


@pytest.fixture(scope='session')
def xai_api_key() -> str:
    return os.getenv('XAI_API_KEY', 'mock-api-key')


@pytest.fixture(scope='session')
def tavily_api_key() -> str:
    return os.getenv('TAVILY_API_KEY', 'mock-api-key')


@pytest.fixture(scope='function')  # Needs to be function scoped to get the request node name
def xai_provider(request: pytest.FixtureRequest) -> Iterator[XaiProvider | None]:
    """xAI provider fixture backed by protobuf cassettes.

    Mirrors the `bedrock_provider` pattern: yields a provider, and callers can use `provider.client`.
    Returns None for non-xAI tests to avoid loading cassettes unnecessarily.
    """
    if 'xai' not in request.node.name:
        yield None
        return

    try:
        from pydantic_ai.providers.xai import XaiProvider
        from tests.models.xai_proto_cassettes import xai_proto_cassette_session
    except ImportError:  # pragma: no cover
        pytest.skip('xai_sdk not installed')

    cassette_name = sanitize_filename(request.node.name, 240)
    test_module = cast(str, request.node.fspath.basename.replace('.py', ''))
    cassette_path = Path(__file__).parent / 'models' / 'cassettes' / test_module / f'{cassette_name}.xai.yaml'
    record_mode: str | None
    try:
        # Provided by `pytest-recording` as `--record-mode=...` (dest is typically `record_mode`).
        record_mode = cast(Any, request.config).getoption('record_mode')
    except Exception:  # pragma: no cover
        record_mode = None
    include_debug_json = bool(cast(Any, request.config).getoption('xai_proto_include_json'))
    session = xai_proto_cassette_session(
        cassette_path,
        record_mode=record_mode,
        include_debug_json=include_debug_json,
    )
    provider = XaiProvider(xai_client=cast(Any, session.client))
    try:
        yield provider
    finally:
        session.dump_if_recording()


@pytest.fixture(scope='session')
def bedrock_provider():
    try:
        import boto3

        from pydantic_ai.providers.bedrock import BedrockProvider

        bearer_token = os.getenv('AWS_BEARER_TOKEN_BEDROCK')
        if bearer_token:  # pragma: no cover
            provider = BedrockProvider(
                api_key=bearer_token,
                region_name=os.getenv('AWS_REGION', 'us-east-1'),
            )
            yield provider
            provider.client.close()
        else:  # pragma: lax no cover
            if os.getenv('AWS_PROFILE'):
                bedrock_client = boto3.client(
                    'bedrock-runtime',
                    region_name=os.getenv('AWS_REGION', 'us-east-1'),
                )
            else:
                bedrock_client = boto3.client(
                    'bedrock-runtime',
                    region_name=os.getenv('AWS_REGION', 'us-east-1'),
                    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID', 'AKIA6666666666666666'),
                    aws_secret_access_key=os.getenv(
                        'AWS_SECRET_ACCESS_KEY', '6666666666666666666666666666666666666666'
                    ),
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
    mocker.patch('pydantic_ai.providers.google_vertex.google.auth.default', return_value=return_value)


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
                    AutoModelForCausalLM.from_pretrained('hf-internal-testing/tiny-random-gpt2'),
                    AutoTokenizer.from_pretrained('hf-internal-testing/tiny-random-gpt2'),
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


@pytest.fixture
def disable_ssrf_protection_for_vcr():
    """Disable SSRF protection for VCR compatibility.

    VCR cassettes record requests with the original hostname. Since SSRF protection
    resolves hostnames to IPs before making requests, we need to disable the validation
    for VCR tests to match the pre-recorded cassettes.

    This fixture patches validate_and_resolve_url to return the hostname in place
    of the resolved IP, allowing the request URL to use the original hostname.
    """
    from unittest.mock import patch

    from pydantic_ai._ssrf import ResolvedUrl, extract_host_and_port

    async def mock_validate_and_resolve(url: str, allow_local: bool) -> ResolvedUrl:
        hostname, path, port, is_https = extract_host_and_port(url)
        # Return hostname in place of resolved IP - this allows VCR matching
        return ResolvedUrl(resolved_ip=hostname, hostname=hostname, port=port, is_https=is_https, path=path)

    with patch('pydantic_ai._ssrf.validate_and_resolve_url', mock_validate_and_resolve):
        yield


_RequestPartT = TypeVar('_RequestPartT', bound=SystemPromptPart | UserPromptPart | ToolReturnPart | RetryPromptPart)
_ResponsePartT = TypeVar(
    '_ResponsePartT',
    bound=TextPart | ToolCallPart | BuiltinToolCallPart | BuiltinToolReturnPart | ThinkingPart | FilePart,
)


@overload
def iter_message_parts(
    messages: Sequence[ModelMessage],
    message_type: type[ModelRequest],
    part_type: type[_RequestPartT],
) -> Iterator[_RequestPartT]: ...


@overload
def iter_message_parts(
    messages: Sequence[ModelMessage],
    message_type: type[ModelResponse],
    part_type: type[_ResponsePartT],
) -> Iterator[_ResponsePartT]: ...


def iter_message_parts(
    messages: Sequence[ModelMessage],
    message_type: type[ModelRequest] | type[ModelResponse],
    part_type: type[_RequestPartT] | type[_ResponsePartT],
) -> Iterator[_RequestPartT | _ResponsePartT]:
    """Iterate over all parts of a given type in messages of a given type."""
    for msg in messages:  # pragma: no branch
        if isinstance(msg, message_type):
            for part in msg.parts:
                if isinstance(part, part_type):
                    yield part


# endregion
