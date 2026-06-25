"""Tests for the PyaiDocs capability."""

from __future__ import annotations

import importlib
from pathlib import Path

import httpx
import pytest
from pydantic import TypeAdapter
from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from pydantic_ai_harness.experimental import HarnessExperimentalWarning
from pydantic_ai_harness.experimental.docs import PyaiDocs, PyaiDocsToolset, PyaiDocsTopic

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    """Run async tests on the asyncio backend (matching upstream pydantic-ai)."""
    return 'asyncio'


class _FakeClient:
    """Stand-in for `httpx.AsyncClient` that returns a canned response or raises."""

    def __init__(self, *, text: str = '', status: int = 200, error: httpx.HTTPError | None = None) -> None:
        self._text = text
        self._status = status
        self._error = error

    async def __aenter__(self) -> _FakeClient:
        return self

    async def __aexit__(self, *exc: object) -> bool:
        return False

    async def get(self, url: str) -> httpx.Response:
        if self._error is not None:
            raise self._error
        return httpx.Response(self._status, text=self._text, request=httpx.Request('GET', url))


def _install_fake_httpx(
    monkeypatch: pytest.MonkeyPatch,
    *,
    text: str = '',
    status: int = 200,
    error: httpx.HTTPError | None = None,
) -> None:
    """Replace `httpx.AsyncClient` with a factory yielding a `_FakeClient`."""

    def factory(*args: object, **kwargs: object) -> _FakeClient:
        return _FakeClient(text=text, status=status, error=error)

    monkeypatch.setattr(httpx, 'AsyncClient', factory)


class TestPyaiDocsToolset:
    async def test_local_hit_is_cached(self, tmp_path: Path) -> None:
        (tmp_path / 'hooks.md').write_text('# Hooks local', encoding='utf-8')
        cache: dict[PyaiDocsTopic, str] = {}
        toolset = PyaiDocsToolset[None](local_docs_path=tmp_path, cache=cache)

        assert await toolset.read_pyai_docs(PyaiDocsTopic.hooks) == '# Hooks local'
        assert cache[PyaiDocsTopic.hooks] == '# Hooks local'

        # Second call serves from cache: removing the file does not change the result.
        (tmp_path / 'hooks.md').unlink()
        assert await toolset.read_pyai_docs(PyaiDocsTopic.hooks) == '# Hooks local'

    async def test_remote_fallback_without_local_and_caching_disabled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_fake_httpx(monkeypatch, text='# Capabilities remote')
        toolset = PyaiDocsToolset[None](local_docs_path=None, cache=None)

        assert await toolset.read_pyai_docs(PyaiDocsTopic.capabilities) == '# Capabilities remote'

    async def test_remote_fallback_when_local_file_missing(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _install_fake_httpx(monkeypatch, text='# Agent remote')
        cache: dict[PyaiDocsTopic, str] = {}
        toolset = PyaiDocsToolset[None](local_docs_path=tmp_path, cache=cache)

        assert await toolset.read_pyai_docs(PyaiDocsTopic.agent) == '# Agent remote'
        assert cache[PyaiDocsTopic.agent] == '# Agent remote'

    async def test_remote_error_without_local_checkout(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_fake_httpx(monkeypatch, error=httpx.ConnectError('boom'))
        toolset = PyaiDocsToolset[None](local_docs_path=None, cache=None)

        with pytest.raises(RuntimeError, match='no local checkout configured'):
            await toolset.read_pyai_docs(PyaiDocsTopic.tools)

    async def test_remote_error_reports_local_path(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        _install_fake_httpx(monkeypatch, status=404)
        toolset = PyaiDocsToolset[None](local_docs_path=tmp_path, cache=None)

        with pytest.raises(RuntimeError, match=str(tmp_path)):
            await toolset.read_pyai_docs(PyaiDocsTopic.toolsets)

    def test_tools_advanced_value_coerces_to_member(self) -> None:
        # The LLM passes the enum VALUE; pydantic coerces it back to the member.
        # `tools_advanced` is the one topic whose name != value, so lock the mapping.
        assert PyaiDocsTopic.tools_advanced.value == 'tools-advanced'
        assert TypeAdapter(PyaiDocsTopic).validate_python('tools-advanced') is PyaiDocsTopic.tools_advanced

    async def test_tools_advanced_reads_hyphenated_file(self, tmp_path: Path) -> None:
        (tmp_path / 'tools-advanced.md').write_text('# Tools advanced local', encoding='utf-8')
        toolset = PyaiDocsToolset[None](local_docs_path=tmp_path, cache=None)

        assert await toolset.read_pyai_docs(PyaiDocsTopic.tools_advanced) == '# Tools advanced local'

    async def test_local_path_expands_user(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.setenv('HOME', str(tmp_path))
        (tmp_path / 'hooks.md').write_text('# Hooks home', encoding='utf-8')
        toolset = PyaiDocsToolset[None](local_docs_path=Path('~'), cache=None)

        assert await toolset.read_pyai_docs(PyaiDocsTopic.hooks) == '# Hooks home'


class TestPyaiDocsCapability:
    def test_resolved_path_prefers_constructor_arg(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.setenv('PYDANTIC_AI_HARNESS_DOCS_PATH', '/env/ignored')
        assert PyaiDocs[None](local_docs_path=tmp_path)._resolved_local_path() == tmp_path

    def test_resolved_path_falls_back_to_env(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.setenv('PYDANTIC_AI_HARNESS_DOCS_PATH', str(tmp_path))
        assert PyaiDocs[None]()._resolved_local_path() == tmp_path

    def test_resolved_path_none_when_unset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv('PYDANTIC_AI_HARNESS_DOCS_PATH', raising=False)
        assert PyaiDocs[None]()._resolved_local_path() is None

    def test_get_toolset_shares_cache_when_enabled(self) -> None:
        capability = PyaiDocs[None]()
        toolset = capability.get_toolset()
        assert isinstance(toolset, PyaiDocsToolset)
        assert toolset._cache is capability._cache

    def test_get_toolset_disables_cache(self) -> None:
        toolset = PyaiDocs[None](cache=False).get_toolset()
        assert isinstance(toolset, PyaiDocsToolset)
        assert toolset._cache is None

    def test_instructions_mention_the_tool(self) -> None:
        instructions = PyaiDocs[None]().get_instructions()
        assert isinstance(instructions, str)
        assert 'read_pyai_docs' in instructions

    def test_serialization_name(self) -> None:
        assert PyaiDocs.get_serialization_name() == 'PyaiDocs'


class TestThroughAgent:
    async def test_tool_returns_local_doc(self, tmp_path: Path) -> None:
        (tmp_path / 'capabilities.md').write_text('# Capabilities doc', encoding='utf-8')

        def call_then_finish(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            if len(messages) == 1:
                return ModelResponse(parts=[ToolCallPart('read_pyai_docs', {'topic': 'capabilities'})])
            return ModelResponse(parts=[TextPart('done')])

        agent = Agent(FunctionModel(call_then_finish), capabilities=[PyaiDocs(local_docs_path=tmp_path)])
        result = await agent.run('go')

        assert result.output == 'done'
        returns = [
            part.content
            for message in result.all_messages()
            for part in message.parts
            if isinstance(part, ToolReturnPart) and part.tool_name == 'read_pyai_docs'
        ]
        assert returns == ['# Capabilities doc']


def test_import_emits_experimental_warning() -> None:
    module = importlib.import_module('pydantic_ai_harness.experimental.docs')
    with pytest.warns(HarnessExperimentalWarning, match='docs'):
        importlib.reload(module)
