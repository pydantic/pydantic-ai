"""Tests for the PydanticAIDocs capability."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx
import pytest
from pydantic import TypeAdapter

from pydantic_ai import Agent, RunContext
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.usage import RunUsage
from pydantic_ai.workspaces import LocalWorkspaceBackend, Workspace
from pydantic_ai_harness.pydantic_ai_docs import PydanticAIDocs, PydanticAIDocsToolset, PydanticAIDocsTopic


def _run_context(workspace: Workspace | None = None) -> RunContext[None]:
    """Minimal `RunContext` for direct toolset invocations."""
    if workspace is None:
        return RunContext[None](deps=None, model=TestModel(), usage=RunUsage(), prompt=None, messages=[], run_step=0)
    return RunContext[None](
        deps=None, model=TestModel(), usage=RunUsage(), prompt=None, messages=[], run_step=0, workspace=workspace
    )


async def _call_docs_tool(
    toolset: PydanticAIDocsToolset[object], workspace: Workspace | None, name: str, **tool_args: Any
) -> str:
    ctx = _run_context(workspace=workspace)
    tools = await toolset.get_tools(ctx)
    result = await toolset.call_tool(name, tool_args, ctx, tools[name])
    assert isinstance(result, str)
    return result


@pytest.fixture
def workspace(tmp_path: Path) -> Workspace:
    return Workspace(LocalWorkspaceBackend(working_dir=tmp_path))


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
    async def test_local_hit_is_cached(self, tmp_path: Path, workspace: Workspace) -> None:
        (tmp_path / 'hooks.md').write_text('# Hooks local', encoding='utf-8')
        cache: dict[PydanticAIDocsTopic, str] = {}
        toolset = PydanticAIDocsToolset[object](local_docs_path=tmp_path, cache=cache)

        assert (
            await _call_docs_tool(toolset, workspace, 'read_pyai_docs', topic=PydanticAIDocsTopic.hooks)
            == '# Hooks local'
        )
        assert cache[PydanticAIDocsTopic.hooks] == '# Hooks local'

        # Second call serves from cache: removing the file does not change the result.
        (tmp_path / 'hooks.md').unlink()
        assert (
            await _call_docs_tool(toolset, workspace, 'read_pyai_docs', topic=PydanticAIDocsTopic.hooks)
            == '# Hooks local'
        )

    async def test_remote_fallback_without_local_and_caching_disabled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_fake_httpx(monkeypatch, text='# Capabilities remote')
        toolset = PydanticAIDocsToolset[object](local_docs_path=None, cache=None)

        assert (
            await _call_docs_tool(toolset, None, 'read_pyai_docs', topic=PydanticAIDocsTopic.capabilities)
            == '# Capabilities remote'
        )

    async def test_remote_fallback_when_local_file_missing(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, workspace: Workspace
    ) -> None:
        _install_fake_httpx(monkeypatch, text='# Agent remote')
        cache: dict[PydanticAIDocsTopic, str] = {}
        toolset = PydanticAIDocsToolset[object](local_docs_path=tmp_path, cache=cache)

        assert (
            await _call_docs_tool(toolset, workspace, 'read_pyai_docs', topic=PydanticAIDocsTopic.agent)
            == '# Agent remote'
        )
        assert cache[PydanticAIDocsTopic.agent] == '# Agent remote'

    async def test_remote_fallback_when_local_parent_is_not_a_directory(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, workspace: Workspace
    ) -> None:
        _install_fake_httpx(monkeypatch, text='# Agent remote')
        local_parent = tmp_path / 'not-a-directory'
        local_parent.write_text('file', encoding='utf-8')
        toolset = PydanticAIDocsToolset[object](local_docs_path=local_parent, cache=None)

        result = await _call_docs_tool(toolset, workspace, 'read_pyai_docs', topic=PydanticAIDocsTopic.agent)

        assert result == '# Agent remote'

    async def test_remote_error_without_local_checkout(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_fake_httpx(monkeypatch, error=httpx.ConnectError('boom'))
        toolset = PydanticAIDocsToolset[object](local_docs_path=None, cache=None)

        with pytest.raises(RuntimeError, match='no local checkout configured'):
            await _call_docs_tool(toolset, None, 'read_pyai_docs', topic=PydanticAIDocsTopic.tools)

    async def test_remote_error_reports_local_path(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, workspace: Workspace
    ) -> None:
        _install_fake_httpx(monkeypatch, status=404)
        toolset = PydanticAIDocsToolset[object](local_docs_path=tmp_path, cache=None)

        with pytest.raises(RuntimeError, match=str(tmp_path)):
            await _call_docs_tool(toolset, workspace, 'read_pyai_docs', topic=PydanticAIDocsTopic.toolsets)

    def test_tools_advanced_value_coerces_to_member(self) -> None:
        # The LLM passes the enum VALUE; pydantic coerces it back to the member.
        # `tools_advanced` is the one topic whose name != value, so lock the mapping.
        assert PydanticAIDocsTopic.tools_advanced.value == 'tools-advanced'
        assert TypeAdapter(PydanticAIDocsTopic).validate_python('tools-advanced') is PydanticAIDocsTopic.tools_advanced

    async def test_tools_advanced_reads_hyphenated_file(self, tmp_path: Path, workspace: Workspace) -> None:
        (tmp_path / 'tools-advanced.md').write_text('# Tools advanced local', encoding='utf-8')
        toolset = PydanticAIDocsToolset[object](local_docs_path=tmp_path, cache=None)

        assert (
            await _call_docs_tool(toolset, workspace, 'read_pyai_docs', topic=PydanticAIDocsTopic.tools_advanced)
            == '# Tools advanced local'
        )

    async def test_relative_local_path_uses_workspace_working_dir(self, tmp_path: Path, workspace: Workspace) -> None:
        (tmp_path / 'hooks.md').write_text('# Hooks home', encoding='utf-8')
        toolset = PydanticAIDocsToolset[object](local_docs_path=Path('.'), cache=None)

        assert (
            await _call_docs_tool(toolset, workspace, 'read_pyai_docs', topic=PydanticAIDocsTopic.hooks)
            == '# Hooks home'
        )

    async def test_tilde_is_rejected(self, workspace: Workspace) -> None:
        toolset = PydanticAIDocsToolset[object](local_docs_path=Path('~/docs'), cache=None)

        with pytest.raises(UserError, match='do not expand'):
            await _call_docs_tool(toolset, workspace, 'read_pyai_docs', topic=PydanticAIDocsTopic.hooks)


class TestPydanticAIDocsCapability:
    def test_resolved_path_prefers_constructor_arg(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.setenv('PYDANTIC_AI_HARNESS_DOCS_PATH', '/env/ignored')
        assert PydanticAIDocs[object](local_docs_path=tmp_path)._resolved_local_path() == tmp_path  # pyright: ignore[reportPrivateUsage]

    def test_resolved_path_falls_back_to_env(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.setenv('PYDANTIC_AI_HARNESS_DOCS_PATH', str(tmp_path))
        assert PydanticAIDocs[object]()._resolved_local_path() == tmp_path  # pyright: ignore[reportPrivateUsage]

    def test_resolved_path_none_when_unset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv('PYDANTIC_AI_HARNESS_DOCS_PATH', raising=False)
        assert PydanticAIDocs[object]()._resolved_local_path() is None  # pyright: ignore[reportPrivateUsage]

    def test_get_toolset_shares_cache_when_enabled(self) -> None:
        capability = PydanticAIDocs[object]()
        toolset = capability.get_toolset()
        assert isinstance(toolset, PydanticAIDocsToolset)
        assert toolset._cache is capability._cache  # pyright: ignore[reportPrivateUsage]

    def test_get_toolset_disables_cache(self) -> None:
        toolset = PydanticAIDocs[object](cache=False).get_toolset()
        assert isinstance(toolset, PydanticAIDocsToolset)
        assert toolset._cache is None  # pyright: ignore[reportPrivateUsage]

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
        backend = LocalWorkspaceBackend(working_dir=tmp_path)
        result = await agent.run('go', workspace=backend)

        assert result.output == 'done'
        returns = [
            part.content
            for message in result.all_messages()
            for part in message.parts
            if isinstance(part, ToolReturnPart) and part.tool_name == 'read_pyai_docs'
        ]
        assert returns == ['# Capabilities doc']

    async def test_local_path_without_workspace_fails_the_run(self, tmp_path: Path) -> None:
        agent = Agent(TestModel(call_tools=[]), capabilities=[PydanticAIDocs(local_docs_path=tmp_path)])

        with pytest.raises(UserError, match='`PydanticAIDocs` needs a workspace'):
            await agent.run('go')

    async def test_no_local_path_needs_no_workspace(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv('PYDANTIC_AI_HARNESS_DOCS_PATH', raising=False)
        agent = Agent(TestModel(call_tools=[]), capabilities=[PydanticAIDocs()])

        assert (await agent.run('go')).output

    async def test_cache_is_isolated_between_run_workspaces(self, tmp_path: Path) -> None:
        first_root = tmp_path / 'first'
        second_root = tmp_path / 'second'
        for root, content in ((first_root, '# First workspace'), (second_root, '# Second workspace')):
            docs = root / 'docs'
            docs.mkdir(parents=True)
            (docs / 'capabilities.md').write_text(content, encoding='utf-8')

        def call_then_finish(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            if len(messages) == 1:
                return ModelResponse(parts=[ToolCallPart('read_pyai_docs', {'topic': 'capabilities'})])
            return ModelResponse(parts=[TextPart('done')])

        agent = Agent(FunctionModel(call_then_finish), capabilities=[PydanticAIDocs(local_docs_path=Path('docs'))])

        returned: list[str] = []
        for root in (first_root, second_root):
            backend = LocalWorkspaceBackend(working_dir=root)
            result = await agent.run('go', workspace=backend)
            returned.extend(
                part.content
                for message in result.all_messages()
                for part in message.parts
                if isinstance(part, ToolReturnPart)
                and part.tool_name == 'read_pyai_docs'
                and isinstance(part.content, str)
            )

        assert returned == ['# First workspace', '# Second workspace']
