"""Tests for the PydanticAIDocs capability."""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest
from pydantic import TypeAdapter
from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from pydantic_ai_harness.pydantic_ai_docs import PydanticAIDocs, PydanticAIDocsToolset, PydanticAIDocsTopic

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


class TestPydanticAIDocsToolset:
    async def test_local_hit_is_cached(self, tmp_path: Path) -> None:
        (tmp_path / 'hooks.md').write_text('# Hooks local', encoding='utf-8')
        cache: dict[PydanticAIDocsTopic, str] = {}
        toolset = PydanticAIDocsToolset[object](local_docs_path=tmp_path, cache=cache)

        assert await toolset.read_pyai_docs(PydanticAIDocsTopic.hooks) == '# Hooks local'
        assert cache[PydanticAIDocsTopic.hooks] == '# Hooks local'

        # Second call serves from cache: removing the file does not change the result.
        (tmp_path / 'hooks.md').unlink()
        assert await toolset.read_pyai_docs(PydanticAIDocsTopic.hooks) == '# Hooks local'

    async def test_remote_fallback_without_local_and_caching_disabled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_fake_httpx(monkeypatch, text='# Capabilities remote')
        toolset = PydanticAIDocsToolset[object](local_docs_path=None, cache=None)

        assert await toolset.read_pyai_docs(PydanticAIDocsTopic.capabilities) == '# Capabilities remote'

    async def test_remote_fallback_when_local_file_missing(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _install_fake_httpx(monkeypatch, text='# Agent remote')
        cache: dict[PydanticAIDocsTopic, str] = {}
        toolset = PydanticAIDocsToolset[object](local_docs_path=tmp_path, cache=cache)

        assert await toolset.read_pyai_docs(PydanticAIDocsTopic.agent) == '# Agent remote'
        assert cache[PydanticAIDocsTopic.agent] == '# Agent remote'

    async def test_remote_error_without_local_checkout(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_fake_httpx(monkeypatch, error=httpx.ConnectError('boom'))
        toolset = PydanticAIDocsToolset[object](local_docs_path=None, cache=None)

        with pytest.raises(RuntimeError, match='no local checkout configured'):
            await toolset.read_pyai_docs(PydanticAIDocsTopic.tools)

    async def test_remote_error_reports_local_path(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        _install_fake_httpx(monkeypatch, status=404)
        toolset = PydanticAIDocsToolset[object](local_docs_path=tmp_path, cache=None)

        with pytest.raises(RuntimeError, match=str(tmp_path)):
            await toolset.read_pyai_docs(PydanticAIDocsTopic.toolsets)

    def test_tools_advanced_value_coerces_to_member(self) -> None:
        # The LLM passes the enum VALUE; pydantic coerces it back to the member.
        # `tools_advanced` is the one topic whose name != value, so lock the mapping.
        assert PydanticAIDocsTopic.tools_advanced.value == 'tools-advanced'
        assert TypeAdapter(PydanticAIDocsTopic).validate_python('tools-advanced') is PydanticAIDocsTopic.tools_advanced

    async def test_tools_advanced_reads_hyphenated_file(self, tmp_path: Path) -> None:
        (tmp_path / 'tools-advanced.md').write_text('# Tools advanced local', encoding='utf-8')
        toolset = PydanticAIDocsToolset[object](local_docs_path=tmp_path, cache=None)

        assert await toolset.read_pyai_docs(PydanticAIDocsTopic.tools_advanced) == '# Tools advanced local'

    async def test_local_path_expands_user(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.setenv('HOME', str(tmp_path))
        (tmp_path / 'hooks.md').write_text('# Hooks home', encoding='utf-8')
        toolset = PydanticAIDocsToolset[object](local_docs_path=Path('~'), cache=None)

        assert await toolset.read_pyai_docs(PydanticAIDocsTopic.hooks) == '# Hooks home'


class TestPydanticAIDocsCapability:
    def test_resolved_path_prefers_constructor_arg(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.setenv('PYDANTIC_AI_HARNESS_DOCS_PATH', '/env/ignored')
        assert PydanticAIDocs[object](local_docs_path=tmp_path)._resolved_local_path() == tmp_path

    def test_resolved_path_falls_back_to_env(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.setenv('PYDANTIC_AI_HARNESS_DOCS_PATH', str(tmp_path))
        assert PydanticAIDocs[object]()._resolved_local_path() == tmp_path

    def test_resolved_path_none_when_unset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv('PYDANTIC_AI_HARNESS_DOCS_PATH', raising=False)
        assert PydanticAIDocs[object]()._resolved_local_path() is None

    def test_get_toolset_shares_cache_when_enabled(self) -> None:
        capability = PydanticAIDocs[object]()
        toolset = capability.get_toolset()
        assert isinstance(toolset, PydanticAIDocsToolset)
        assert toolset._cache is capability._cache

    def test_get_toolset_disables_cache(self) -> None:
        toolset = PydanticAIDocs[object](cache=False).get_toolset()
        assert isinstance(toolset, PydanticAIDocsToolset)
        assert toolset._cache is None

    def test_instructions_mention_the_tool(self) -> None:
        instructions = PydanticAIDocs[object]().get_instructions()
        assert isinstance(instructions, str)
        assert 'read_pyai_docs' in instructions

    def test_serialization_name(self) -> None:
        assert PydanticAIDocs.get_serialization_name() == 'PydanticAIDocs'


class TestThroughAgent:
    async def test_tool_returns_local_doc(self, tmp_path: Path) -> None:
        (tmp_path / 'capabilities.md').write_text('# Capabilities doc', encoding='utf-8')

        def call_then_finish(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            if len(messages) == 1:
                return ModelResponse(parts=[ToolCallPart('read_pyai_docs', {'topic': 'capabilities'})])
            return ModelResponse(parts=[TextPart('done')])

        agent = Agent(FunctionModel(call_then_finish), capabilities=[PydanticAIDocs(local_docs_path=tmp_path)])
        result = await agent.run('go')

        assert result.output == 'done'
        returns = [
            part.content
            for message in result.all_messages()
            for part in message.parts
            if isinstance(part, ToolReturnPart) and part.tool_name == 'read_pyai_docs'
        ]
        assert returns == ['# Capabilities doc']
