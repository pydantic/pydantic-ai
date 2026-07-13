"""Tests for the web chat UI module."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any, Literal
from unittest.mock import AsyncMock

import pytest

from pydantic_ai import Agent, ModelSettings
from pydantic_ai.messages import ModelMessage
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, DeltaToolCalls, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.native_tools import SUPPORTED_NATIVE_TOOLS, AbstractNativeTool, MCPServerTool
from pydantic_ai.profiles import ModelProfile
from pydantic_ai.profiles.google import GoogleModelProfile
from pydantic_ai.profiles.groq import GroqModelProfile
from pydantic_ai.profiles.openai import OpenAIModelProfile
from pydantic_ai.tools import DeferredToolRequests

from ._inline_snapshot import snapshot
from .conftest import try_import

with try_import() as starlette_import_successful:
    import httpx
    from starlette.applications import Starlette
    from starlette.responses import Response
    from starlette.testclient import TestClient

    import pydantic_ai.ui._web.app as app_module
    from pydantic_ai.native_tools import WebSearchTool
    from pydantic_ai.ui._web import create_web_app
    from pydantic_ai.ui._web.app import _get_ui_html  # pyright: ignore[reportPrivateUsage]
    from pydantic_ai.ui.vercel_ai import VercelAIAdapter

with try_import() as openai_import_successful:
    import openai  # noqa: F401 # pyright: ignore[reportUnusedImport]

pytestmark = [
    pytest.mark.skipif(not starlette_import_successful(), reason='starlette not installed'),
]


def test_agent_to_web():
    """Test the Agent.to_web() method."""
    agent = Agent('test')
    app = agent.to_web()

    assert isinstance(app, Starlette)


def test_agent_to_web_with_model_instances():
    """Test to_web() accepts model instances, not just strings."""
    agent = Agent(TestModel())
    model_instance = TestModel()

    # List with instances
    app = agent.to_web(models=[model_instance, 'test'])
    assert isinstance(app, Starlette)

    # Dict with instances
    app = agent.to_web(models={'Custom': model_instance, 'Test': 'test'})
    assert isinstance(app, Starlette)


@pytest.mark.anyio
async def test_model_instance_preserved_in_dispatch(monkeypatch: pytest.MonkeyPatch):
    """Test that model instances are preserved and used in dispatch, not reconstructed from string."""
    model_instance = TestModel(custom_output_text='Custom output')
    agent = Agent()
    app = create_web_app(agent, models=[model_instance])

    # Mock dispatch_request to capture the model parameter
    mock_dispatch = AsyncMock(return_value=Response(content=b'', status_code=200))
    monkeypatch.setattr(VercelAIAdapter, 'dispatch_request', mock_dispatch)

    with TestClient(app) as client:
        client.post(
            '/api/chat',
            json={
                'trigger': 'submit-message',
                'id': 'test-id',
                'messages': [
                    {
                        'id': 'msg-1',
                        'role': 'user',
                        'parts': [{'type': 'text', 'text': 'Hello'}],
                    }
                ],
                'model': 'test:test',
                'builtinTools': [],
            },
        )

    # Verify dispatch_request was called with the original model instance
    mock_dispatch.assert_called_once()
    call_kwargs = mock_dispatch.call_args.kwargs
    assert call_kwargs['model'] is model_instance, 'Model instance should be preserved, not reconstructed from string'


def test_agent_to_web_with_deps():
    """Test to_web() accepts deps parameter."""

    @dataclass
    class MyDeps:
        api_key: str

    agent: Agent[MyDeps, str] = Agent(TestModel(), deps_type=MyDeps)
    deps = MyDeps(api_key='test-key')

    app = agent.to_web(deps=deps)
    assert isinstance(app, Starlette)


def test_agent_to_web_with_model_settings():
    """Test to_web() accepts model_settings parameter."""
    agent = Agent(TestModel())
    settings = ModelSettings(temperature=0.5, max_tokens=100)

    app = agent.to_web(model_settings=settings)
    assert isinstance(app, Starlette)


def test_chat_app_health_endpoint():
    """Test the /api/health endpoint."""
    agent = Agent('test')
    app = create_web_app(agent)

    with TestClient(app) as client:
        response = client.get('/api/health')
        assert response.status_code == 200
        assert response.json() == {'ok': True}


def test_chat_app_configure_endpoint():
    """Test the /api/configure endpoint with explicit models and tools."""

    agent = Agent('test')
    app = create_web_app(
        agent,
        models=['test'],
        native_tools=[WebSearchTool()],
    )

    with TestClient(app) as client:
        response = client.get('/api/configure')
        assert response.status_code == 200
        assert response.json() == snapshot(
            {
                'models': [
                    {'id': 'test:test', 'name': 'Test', 'builtinTools': ['web_search']},
                    {'id': 'test', 'name': 'Test', 'builtinTools': ['web_search']},
                ],
                'builtinTools': [{'id': 'web_search', 'name': 'Web Search'}],
            }
        )


def test_chat_app_configure_endpoint_empty():
    """Test the /api/configure endpoint with no models or tools."""
    agent = Agent('test')
    app = create_web_app(agent)

    with TestClient(app) as client:
        response = client.get('/api/configure')
        assert response.status_code == 200
        assert response.json() == snapshot(
            {'models': [{'id': 'test:test', 'name': 'Test', 'builtinTools': []}], 'builtinTools': []}
        )


@pytest.mark.skipif(not openai_import_successful(), reason='openai not installed')
def test_chat_app_configure_preserves_chat_vs_responses(monkeypatch: pytest.MonkeyPatch):
    """Test that openai-chat: and openai-responses: models are kept as separate entries."""
    monkeypatch.setenv('OPENAI_API_KEY', 'test-key')

    agent = Agent('test')
    app = create_web_app(
        agent,
        models=['openai-chat:gpt-4o', 'openai-responses:gpt-4o'],
    )

    with TestClient(app) as client:
        response = client.get('/api/configure')
        assert response.status_code == 200
        data = response.json()
        model_ids = [m['id'] for m in data['models']]
        assert 'openai-chat:gpt-4o' in model_ids
        assert 'openai-responses:gpt-4o' in model_ids
        assert len([m for m in model_ids if 'gpt-4o' in m]) == 2


def _stub_cdn_fetch(monkeypatch: pytest.MonkeyPatch, content: bytes) -> list[int]:
    """Stub `httpx.AsyncClient` to return `content` on every fetch, returning a `[count]` list.

    Used by the cache tests to drive `_get_ui_html`'s CDN/url branch without a real network
    request. The returned single-element list is incremented on each fetch so a test can assert
    how many times the cache missed.
    """
    fetch_count = [0]

    class MockResponse:
        def __init__(self) -> None:
            self.content = content

        def raise_for_status(self) -> None:
            pass

    class MockAsyncClient:
        async def __aenter__(self) -> MockAsyncClient:
            return self

        async def __aexit__(self, *args: Any) -> None:
            pass

        async def get(self, url: str) -> MockResponse:
            fetch_count[0] += 1
            return MockResponse()

    monkeypatch.setattr(app_module.httpx, 'AsyncClient', MockAsyncClient)
    return fetch_count


@pytest.fixture
def isolated_ui_cache(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Isolate the index route's HTML cache to a temp dir and stub the CDN fetch.

    The index route caches the default UI HTML under the shared user cache dir; without
    per-test isolation, tests that serve `/` race on the same file across xdist workers
    (a non-atomic write being read mid-write), and miss the cache into a real CDN request.
    """
    monkeypatch.setattr(app_module, '_get_cache_dir', lambda: tmp_path)
    _stub_cdn_fetch(monkeypatch, b'<html>Test UI</html>')


def test_chat_app_index_endpoint(isolated_ui_cache: None):
    """Test that the index endpoint serves HTML with proper caching headers."""
    agent = Agent('test')
    app = create_web_app(agent)

    with TestClient(app) as client:
        response = client.get('/')
        assert response.status_code == 200
        assert response.headers['content-type'] == 'text/html; charset=utf-8'
        assert 'cache-control' in response.headers
        assert response.headers['cache-control'] == 'public, max-age=3600'
        assert len(response.content) > 0


@pytest.mark.anyio
async def test_get_ui_html_cdn_fetch(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Test that _get_ui_html fetches from CDN when filesystem cache misses."""
    monkeypatch.setattr(app_module, '_get_cache_dir', lambda: tmp_path)

    test_content = b'<html>Test UI</html>'
    _stub_cdn_fetch(monkeypatch, test_content)

    result = await _get_ui_html()

    assert result == test_content
    cache_file: Path = tmp_path / f'{app_module.CHAT_UI_VERSION}.html'
    assert cache_file.exists()
    assert cache_file.read_bytes() == test_content


@pytest.mark.anyio
async def test_get_ui_html_filesystem_cache_hit(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Test that _get_ui_html returns cached content from filesystem."""
    monkeypatch.setattr(app_module, '_get_cache_dir', lambda: tmp_path)

    test_content = b'<html>Cached UI</html>'
    cache_file = tmp_path / f'{app_module.CHAT_UI_VERSION}.html'
    cache_file.write_bytes(test_content)

    result = await _get_ui_html()

    assert result == test_content


def test_get_cache_dir_uses_xdg_cache_home(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """`_get_cache_dir` derives its path from `XDG_CACHE_HOME` and creates the directory.

    The index-route tests monkeypatch `_get_cache_dir` for isolation, so this is the only
    test that exercises its real body.
    """
    monkeypatch.setenv('XDG_CACHE_HOME', str(tmp_path))

    cache_dir = app_module._get_cache_dir()  # pyright: ignore[reportPrivateUsage]

    assert cache_dir == tmp_path / 'pydantic-ai' / 'web-ui'
    assert cache_dir.is_dir()


@pytest.mark.anyio
async def test_get_ui_html_refetches_empty_cache_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setattr(app_module, '_get_cache_dir', lambda: tmp_path)

    cache_file = tmp_path / f'{app_module.CHAT_UI_VERSION}.html'
    cache_file.write_bytes(b'')
    test_content = b'<html>Recovered UI</html>'
    fetch_count = _stub_cdn_fetch(monkeypatch, test_content)

    result = await _get_ui_html()

    assert result == test_content
    assert cache_file.read_bytes() == test_content
    assert fetch_count[0] == 1


def test_write_cached_file_removes_temp_file_on_replace_error(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """A failed `os.replace` unlinks the temp file and leaves the destination intact.

    The cleanup path only fires when the rename fails, which `_get_ui_html` can't trigger on demand,
    so the private helper is driven directly with a forced `os.replace` failure.
    """
    cache_file = tmp_path / f'{app_module.CHAT_UI_VERSION}.html'
    cache_file.write_bytes(b'old content')
    temp_paths: list[Path] = []

    def fail_replace(src: str | os.PathLike[str], dst: str | os.PathLike[str]) -> None:
        temp_paths.append(Path(src))
        assert Path(dst) == cache_file
        assert Path(src).exists()
        raise OSError('replace failed')

    monkeypatch.setattr(app_module.os, 'replace', fail_replace)

    with pytest.raises(OSError, match='replace failed'):
        app_module._write_cached_file(cache_file, b'new content')  # pyright: ignore[reportPrivateUsage]

    assert cache_file.read_bytes() == b'old content'
    assert temp_paths
    assert not temp_paths[0].exists()


def test_write_cached_file_closes_temp_handle_before_replace(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """The temp file handle is closed before `os.replace`, so the rename can't fail on Windows.

    Windows refuses to replace a file that still has an open handle, so an `os.replace` fired while
    the `NamedTemporaryFile` handle is open would break the atomic write there. Capturing the temp
    file object and asserting it is already closed when `os.replace` runs pins the close-before-replace
    ordering on every platform (POSIX allows renaming an open file, so it would otherwise hide the bug).
    """
    cache_file = tmp_path / f'{app_module.CHAT_UI_VERSION}.html'
    temp_files: list[IO[bytes]] = []
    real_named_temporary_file = app_module.tempfile.NamedTemporaryFile

    def capturing_named_temporary_file(*, dir: Path, prefix: str, delete: bool) -> IO[bytes]:
        tmp_file = real_named_temporary_file(dir=dir, prefix=prefix, delete=delete)
        temp_files.append(tmp_file)
        return tmp_file

    monkeypatch.setattr(app_module.tempfile, 'NamedTemporaryFile', capturing_named_temporary_file)

    closed_at_replace: list[bool] = []
    real_replace = app_module.os.replace

    def instrumented_replace(src: str | os.PathLike[str], dst: str | os.PathLike[str]) -> None:
        closed_at_replace.append(temp_files[0].closed)
        real_replace(src, dst)

    monkeypatch.setattr(app_module.os, 'replace', instrumented_replace)

    content = b'<html>UI</html>'
    app_module._write_cached_file(cache_file, content)  # pyright: ignore[reportPrivateUsage]

    assert closed_at_replace == [True]
    assert cache_file.read_bytes() == content


@pytest.mark.anyio
async def test_get_ui_html_cache_write_is_atomic(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """The destination cache file only ever materializes complete, via an atomic `os.replace`.

    A direct `write_bytes` to the destination truncates it before writing the content, so a
    concurrent reader can catch it existing-but-empty. Interposing on `os.replace` lets us assert
    deterministically (no timing/threads) that the destination materializes only through the atomic
    rename, and that the rename source already holds the complete content.
    """
    monkeypatch.setattr(app_module, '_get_cache_dir', lambda: tmp_path)

    full_content = b'<html>complete UI document</html>'
    _stub_cdn_fetch(monkeypatch, full_content)

    cache_file = tmp_path / f'{app_module.CHAT_UI_VERSION}.html'
    real_replace = app_module.os.replace
    replaced_targets: list[Path] = []

    def instrumented_replace(src: str | os.PathLike[str], dst: str | os.PathLike[str]) -> None:
        assert Path(src).read_bytes() == full_content
        assert not cache_file.exists()
        replaced_targets.append(Path(dst))
        real_replace(src, dst)

    monkeypatch.setattr(app_module.os, 'replace', instrumented_replace)

    result = await _get_ui_html()

    assert result == full_content
    assert cache_file.read_bytes() == full_content
    assert replaced_targets == [cache_file]


def test_chat_app_index_caching(isolated_ui_cache: None):
    """Test that the UI HTML is cached after first fetch."""
    agent = Agent('test')
    app = create_web_app(agent)

    with TestClient(app) as client:
        response1 = client.get('/')
        response2 = client.get('/')

        assert response1.content == response2.content
        assert response1.status_code == 200
        assert response2.status_code == 200


@pytest.mark.anyio
async def test_post_chat_endpoint():
    """Test the POST /api/chat endpoint."""
    agent = Agent(TestModel(custom_output_text='Hello from test!'))
    app = create_web_app(agent)

    with TestClient(app) as client:
        response = client.post(
            '/api/chat',
            json={
                'trigger': 'submit-message',
                'id': 'test-message-id',
                'messages': [
                    {
                        'id': 'msg-1',
                        'role': 'user',
                        'parts': [{'type': 'text', 'text': 'Hello'}],
                    }
                ],
                'model': 'test:test',
                'builtinTools': [],
            },
        )

        assert response.status_code == 200


def _parse_sse_chunk_types(body: str) -> list[str]:
    """Extract the ordered `type` of each `data:` chunk from a Vercel AI SSE stream body."""
    types: list[str] = []
    for line in body.splitlines():
        if line.startswith('data: ') and (payload := line.removeprefix('data: ')) != '[DONE]':
            types.append(json.loads(payload)['type'])
    return types


@pytest.mark.anyio
@pytest.mark.parametrize('sdk_version', [None, 5, 6, 7])
async def test_post_chat_streams_tool_approval(allow_model_requests: None, sdk_version: Literal[5, 6, 7] | None):
    """The bundled web path targets Vercel AI SDK v7, so a tool call that requires approval streams a
    `tool-approval-request` chunk the v7 UI renders as approve/reject buttons.

    `sdk_version=None` exercises the default (`create_web_app` bundles the v7 UI, so it targets 7);
    explicit `6`/`7` match (7 emits the same wire as 6), while `5` falls back to `tool-input-available`
    with no approval chunk.

    Not a VCR test: this asserts the server→client SSE stream shape, which has no provider API to
    record. `FunctionModel` deterministically drives the deferred tool call the wire format hinges on.
    """

    async def stream_function(
        _messages: list[ModelMessage], agent_info: AgentInfo
    ) -> AsyncIterator[DeltaToolCalls | str]:
        yield {0: DeltaToolCall(name='delete_file', json_args='{"path": "test.txt"}', tool_call_id='delete_1')}

    agent = Agent(model=FunctionModel(stream_function=stream_function), output_type=[str, DeferredToolRequests])

    @agent.tool_plain(requires_approval=True)
    def delete_file(path: str) -> str:
        return f'Deleted {path}'  # pragma: no cover

    app = create_web_app(agent) if sdk_version is None else create_web_app(agent, sdk_version=sdk_version)

    with TestClient(app) as client:
        response = client.post(
            '/api/chat',
            json={
                'trigger': 'submit-message',
                'id': 'test-id',
                'messages': [{'id': 'msg-1', 'role': 'user', 'parts': [{'type': 'text', 'text': 'Delete test.txt'}]}],
                'builtinTools': [],
            },
        )
        assert response.status_code == 200
        chunk_types = _parse_sse_chunk_types(response.text)

    if sdk_version == 5:
        assert 'tool-approval-request' not in chunk_types
        assert 'tool-input-available' in chunk_types
    else:
        # v6/v7 emit `tool-input-available` (carrying the tool args the UI renders in the prompt)
        # before `tool-approval-request` (which only carries approval_id + tool_call_id), so the
        # v7 UI can show the pending call's input alongside the approve/reject buttons.
        assert 'tool-input-available' in chunk_types
        assert 'tool-approval-request' in chunk_types
        assert chunk_types.index('tool-input-available') < chunk_types.index('tool-approval-request')


def test_chat_app_options_endpoint():
    """Test the OPTIONS /api/chat endpoint (CORS preflight)."""
    agent = Agent('test')
    app = create_web_app(agent)

    with TestClient(app) as client:
        response = client.options('/api/chat')
        assert response.status_code == 200


def test_mcp_server_tool_label():
    """Test MCPServerTool.label property."""
    tool = MCPServerTool(id='test-server', url='https://example.com')
    assert tool.label == 'MCP: test-server'


def test_model_profile():
    """Test Model.profile cached property."""
    model = TestModel()
    assert model.profile is not None


@pytest.mark.parametrize('profile_name', ['base', 'openai', 'google', 'groq'])
def test_supported_native_tools(profile_name: str):
    """Test `profile.get('supported_native_tools', SUPPORTED_NATIVE_TOOLS)` returns proper tool types."""
    if profile_name == 'base':
        profile: ModelProfile = ModelProfile()
    elif profile_name == 'openai':
        profile = OpenAIModelProfile()
    elif profile_name == 'google':
        profile = GoogleModelProfile()
    else:
        profile = GroqModelProfile()

    result = profile.get('supported_native_tools', SUPPORTED_NATIVE_TOOLS)
    assert isinstance(result, frozenset)
    assert all(issubclass(t, AbstractNativeTool) for t in result)


def test_post_chat_invalid_model():
    """Test POST /api/chat returns 400 when model is not in allowed list."""
    agent = Agent(TestModel(custom_output_text='Hello'))
    # Use 'test' as the allowed model, then send a different model in the request
    app = create_web_app(agent, models=['test'])

    with TestClient(app) as client:
        response = client.post(
            '/api/chat',
            json={
                'trigger': 'submit-message',
                'id': 'test-id',
                'messages': [
                    {
                        'id': 'msg-1',
                        'role': 'user',
                        'parts': [{'type': 'text', 'text': 'Hello'}],
                    }
                ],
                'model': 'test:different_model',
                'builtinTools': [],
            },
        )

        assert response.status_code == 400
        assert response.json() == snapshot({'error': 'Model "test:different_model" is not in the allowed models list'})


def test_post_chat_invalid_builtin_tool():
    """Test POST /api/chat returns 400 when builtin tool is not in allowed list."""
    agent = Agent(TestModel(custom_output_text='Hello'))
    app = create_web_app(agent, native_tools=[WebSearchTool()])

    with TestClient(app) as client:
        response = client.post(
            '/api/chat',
            json={
                'trigger': 'submit-message',
                'id': 'test-id',
                'messages': [
                    {
                        'id': 'msg-1',
                        'role': 'user',
                        'parts': [{'type': 'text', 'text': 'Hello'}],
                    }
                ],
                'model': 'test:test',
                'builtinTools': ['code_execution'],  # Not in allowed list
            },
        )

        assert response.status_code == 400
        assert response.json() == snapshot(
            {'error': "Builtin tool(s) ['code_execution'] not in the allowed tools list"}
        )


def test_model_label_openrouter():
    """Test Model.label handles OpenRouter-style names with /."""
    model = TestModel(model_name='meta-llama/llama-3-70b')
    assert model.label == snapshot('Llama 3 70b')


def test_agent_to_web_with_instructions():
    """Test to_web() accepts instructions parameter."""
    agent = Agent(TestModel())
    app = agent.to_web(instructions='Always respond in Spanish')
    assert isinstance(app, Starlette)


@pytest.mark.anyio
async def test_instructions_passed_to_dispatch(monkeypatch: pytest.MonkeyPatch):
    """Test that instructions from create_web_app are passed to dispatch_request."""
    agent = Agent(TestModel(custom_output_text='Hello'))
    app = create_web_app(agent, instructions='Always respond in Spanish')

    # Mock dispatch_request to capture the instructions parameter
    mock_dispatch = AsyncMock(return_value=Response(content=b'', status_code=200))
    monkeypatch.setattr(VercelAIAdapter, 'dispatch_request', mock_dispatch)

    with TestClient(app) as client:
        client.post(
            '/api/chat',
            json={
                'trigger': 'submit-message',
                'id': 'test-id',
                'messages': [
                    {
                        'id': 'msg-1',
                        'role': 'user',
                        'parts': [{'type': 'text', 'text': 'Hello'}],
                    }
                ],
                'model': 'test:test',
                'builtinTools': [],
            },
        )

    # Verify dispatch_request was called with instructions
    mock_dispatch.assert_called_once()
    call_kwargs = mock_dispatch.call_args.kwargs
    assert call_kwargs['instructions'] == 'Always respond in Spanish'


@pytest.mark.anyio
async def test_get_ui_html_custom_url(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Test that _get_ui_html fetches from custom URL when provided."""
    monkeypatch.setattr(app_module, '_get_cache_dir', lambda: tmp_path)

    test_content = b'<html>Custom CDN UI</html>'
    captured_url: list[str] = []

    class MockResponse:
        content = test_content

        def raise_for_status(self) -> None:
            pass

    class MockAsyncClient:
        async def __aenter__(self) -> MockAsyncClient:
            return self

        async def __aexit__(self, *args: Any) -> None:
            pass

        async def get(self, url: str) -> MockResponse:
            captured_url.append(url)
            return MockResponse()

    monkeypatch.setattr(app_module.httpx, 'AsyncClient', MockAsyncClient)

    # URL is used directly, no version templating
    custom_url = 'https://my-internal-cdn.example.com/ui/index.html'
    result = await _get_ui_html(html_source=custom_url)

    assert result == test_content
    assert len(captured_url) == 1
    assert captured_url[0] == custom_url


@pytest.mark.anyio
async def test_get_ui_html_custom_url_caching(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Test that custom URLs are cached to filesystem and not re-fetched."""
    monkeypatch.setattr(app_module, '_get_cache_dir', lambda: tmp_path)

    test_content = b'<html>Cached Custom UI</html>'
    fetch_count = _stub_cdn_fetch(monkeypatch, test_content)

    custom_url = 'https://my-internal-cdn.example.com/ui/cached.html'

    # First call should fetch from URL
    result1 = await _get_ui_html(html_source=custom_url)
    assert result1 == test_content
    assert fetch_count[0] == 1

    # Verify cache file was created
    url_hash = hashlib.sha256(custom_url.encode()).hexdigest()[:16]
    cache_file = tmp_path / f'url_{url_hash}.html'
    assert cache_file.exists()
    assert cache_file.read_bytes() == test_content

    # Second call should use cache, not fetch again
    result2 = await _get_ui_html(html_source=custom_url)
    assert result2 == test_content
    assert fetch_count[0] == 1  # Still 1, not 2


def test_agent_to_web_with_html_source():
    """Test that Agent.to_web() accepts html_source parameter."""
    agent = Agent('test')
    app = agent.to_web(html_source='https://custom-cdn.example.com/ui/index.html')

    assert isinstance(app, Starlette)


@pytest.mark.anyio
async def test_get_ui_html_local_file_path_string(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Test that _get_ui_html supports local file paths as strings."""
    # Create a test HTML file
    test_html = b'<html><body>Local UI Content</body></html>'
    local_file = tmp_path / 'custom-ui.html'
    local_file.write_bytes(test_html)

    result = await app_module._get_ui_html(html_source=str(local_file))  # pyright: ignore[reportPrivateUsage]

    assert result == test_html


@pytest.mark.anyio
async def test_get_ui_html_local_file_path_instance(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Test that _get_ui_html supports Path instances."""
    # Create a test HTML file
    test_html = b'<html><body>Path Instance UI</body></html>'
    local_file = tmp_path / 'path-ui.html'
    local_file.write_bytes(test_html)

    result = await app_module._get_ui_html(html_source=local_file)  # pyright: ignore[reportPrivateUsage]

    assert result == test_html


@pytest.mark.anyio
async def test_get_ui_html_local_file_not_found(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Test that _get_ui_html raises FileNotFoundError for missing local file paths."""
    # Try to use a non-existent local file path
    nonexistent_path = str(tmp_path / 'nonexistent-ui.html')

    with pytest.raises(FileNotFoundError, match='Local UI file not found'):
        await app_module._get_ui_html(html_source=nonexistent_path)  # pyright: ignore[reportPrivateUsage]


@pytest.mark.anyio
async def test_get_ui_html_source_instance_not_found(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Test that _get_ui_html raises FileNotFoundError for missing Path instances."""
    # Try to use a non-existent Path instance
    nonexistent_path = tmp_path / 'nonexistent-ui.html'

    with pytest.raises(FileNotFoundError, match='Local UI file not found'):
        await app_module._get_ui_html(html_source=nonexistent_path)  # pyright: ignore[reportPrivateUsage]


def test_chat_app_index_file_not_found(tmp_path: Path):
    """Test that index endpoint raises FileNotFoundError for non-existent html_source file."""
    agent = Agent('test')
    nonexistent_file = tmp_path / 'nonexistent-ui.html'
    app = create_web_app(agent, html_source=str(nonexistent_file))

    with TestClient(app, raise_server_exceptions=True) as client:
        with pytest.raises(FileNotFoundError, match='Local UI file not found'):
            client.get('/')


def test_chat_app_index_http_error(monkeypatch: pytest.MonkeyPatch):
    """Test that index endpoint raises HTTPStatusError when CDN fetch fails."""

    class MockResponse:
        status_code = 500

    class MockAsyncClient:
        async def __aenter__(self) -> MockAsyncClient:
            return self

        async def __aexit__(self, *args: Any) -> None:
            pass

        async def get(self, url: str) -> None:
            response = MockResponse()
            raise httpx.HTTPStatusError('Server error', request=None, response=response)  # type: ignore

    monkeypatch.setattr(app_module.httpx, 'AsyncClient', MockAsyncClient)
    # Use a fresh temp dir so there's no cached file
    monkeypatch.setattr(app_module, '_get_cache_dir', lambda: Path('/tmp/nonexistent-cache-dir-for-test'))

    agent = Agent('test')
    app = create_web_app(agent)

    with TestClient(app, raise_server_exceptions=True) as client:
        with pytest.raises(httpx.HTTPStatusError):
            client.get('/')
