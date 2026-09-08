"""Tests for the RepoContext capability."""

from __future__ import annotations

import json
import sys
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any, Literal

import pytest
from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ToolReturnPart, UserPromptPart
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, DeltaToolCalls, FunctionModel
from pydantic_ai.models.test import TestModel

from pydantic_ai_harness import HarnessDeprecationWarning
from pydantic_ai_harness.filesystem import FileSystem
from pydantic_ai_harness.repo_context import (
    AgentContextInventory,
    ContextFile,
    RepoContext,
    RepoContextToolset,
)
from pydantic_ai_harness.repo_context._inventory import scan_assets
from pydantic_ai_harness.repo_context._loader import (
    discover_instruction_files,
    find_dir_context_file,
    render_context_files,
)

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return 'asyncio'


def _write(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding='utf-8')
    return path


class TestDiscoverInstructionFiles:
    def test_walk_up_ancestor_first(self, tmp_path: Path) -> None:
        _write(tmp_path / 'CLAUDE.md', 'root')
        workspace = tmp_path / 'a' / 'b'
        _write(workspace / 'CLAUDE.md', 'leaf')
        files = discover_instruction_files(workspace, tmp_path, ('CLAUDE.md',))
        assert [f.content for f in files] == ['root', 'leaf']

    def test_home_none_only_workspace(self, tmp_path: Path) -> None:
        _write(tmp_path / 'CLAUDE.md', 'root')
        workspace = tmp_path / 'a'
        _write(workspace / 'CLAUDE.md', 'leaf')
        files = discover_instruction_files(workspace, None, ('CLAUDE.md',))
        assert [f.content for f in files] == ['leaf']

    def test_home_equals_workspace(self, tmp_path: Path) -> None:
        _write(tmp_path / 'CLAUDE.md', 'only')
        files = discover_instruction_files(tmp_path, tmp_path, ('CLAUDE.md',))
        assert [f.content for f in files] == ['only']

    def test_home_not_ancestor_falls_back_to_workspace(self, tmp_path: Path) -> None:
        workspace = tmp_path / 'a'
        _write(workspace / 'CLAUDE.md', 'leaf')
        unrelated = tmp_path / 'other'
        unrelated.mkdir()
        files = discover_instruction_files(workspace, unrelated, ('CLAUDE.md',))
        assert [f.content for f in files] == ['leaf']

    def test_both_filenames_within_dir_order(self, tmp_path: Path) -> None:
        _write(tmp_path / 'CLAUDE.md', 'claude')
        _write(tmp_path / 'AGENTS.md', 'agents')
        files = discover_instruction_files(tmp_path, None, ('CLAUDE.md', 'AGENTS.md'))
        assert [f.content for f in files] == ['claude', 'agents']

    @pytest.mark.skipif(sys.platform == 'win32', reason='symlinks need privileges on Windows')
    def test_symlink_deduped_by_realpath(self, tmp_path: Path) -> None:
        _write(tmp_path / 'CLAUDE.md', 'shared')
        (tmp_path / 'AGENTS.md').symlink_to(tmp_path / 'CLAUDE.md')
        files = discover_instruction_files(tmp_path, None, ('CLAUDE.md', 'AGENTS.md'))
        assert len(files) == 1

    def test_identical_content_deduped_by_hash(self, tmp_path: Path) -> None:
        _write(tmp_path / 'CLAUDE.md', 'same')
        workspace = tmp_path / 'a'
        _write(workspace / 'CLAUDE.md', 'same')
        files = discover_instruction_files(workspace, tmp_path, ('CLAUDE.md',))
        assert [f.content for f in files] == ['same']

    def test_missing_files_skipped(self, tmp_path: Path) -> None:
        files = discover_instruction_files(tmp_path, None, ('CLAUDE.md',))
        assert files == []

    def test_non_utf8_file_does_not_crash(self, tmp_path: Path) -> None:
        (tmp_path / 'CLAUDE.md').write_bytes(b'caf\xe9 instructions')
        files = discover_instruction_files(tmp_path, None, ('CLAUDE.md',))
        assert len(files) == 1
        assert 'instructions' in files[0].content


class TestFindDirContextFile:
    def test_first_existing_wins(self, tmp_path: Path) -> None:
        _write(tmp_path / 'AGENTS.md', 'agents')
        found = find_dir_context_file(tmp_path, ('CLAUDE.md', 'AGENTS.md'))
        assert found is not None
        assert found.content == 'agents'

    def test_none_when_absent(self, tmp_path: Path) -> None:
        assert find_dir_context_file(tmp_path, ('CLAUDE.md',)) is None

    def test_non_utf8_file_does_not_crash(self, tmp_path: Path) -> None:
        (tmp_path / 'CLAUDE.md').write_bytes(b'caf\xe9 instructions')
        found = find_dir_context_file(tmp_path, ('CLAUDE.md',))
        assert found is not None
        assert 'instructions' in found.content


class TestRender:
    def test_label_outside_workspace_falls_back_to_posix(self, tmp_path: Path) -> None:
        outside = _write(tmp_path / 'outer' / 'CLAUDE.md', 'x')
        cf = ContextFile(directory=outside.parent, path=outside, content='x')
        rendered = render_context_files([cf], relative_to=tmp_path / 'inner')
        assert outside.as_posix() in rendered

    def test_trailing_newline_does_not_open_a_gap_before_the_closing_tag(self, tmp_path: Path) -> None:
        # The terminator goes; the two spaces before it stay, being a hard line break in Markdown.
        path = _write(tmp_path / 'CLAUDE.md', 'be nice  \n')
        cf = ContextFile(directory=path.parent, path=path, content=path.read_text(encoding='utf-8'))
        assert (
            render_context_files([cf], relative_to=tmp_path)
            == '<context-file path="CLAUDE.md">\nbe nice  \n</context-file>'
        )


class TestInstructions:
    def test_includes_files_and_inventory_hint(self, tmp_path: Path) -> None:
        _write(tmp_path / 'CLAUDE.md', 'be nice')
        cap = RepoContext[object](workspace_dir=tmp_path)
        instructions = cap.get_instructions()
        assert isinstance(instructions, str)
        assert 'be nice' in instructions
        assert 'inventory_agent_context' in instructions

    def test_none_when_all_disabled(self, tmp_path: Path) -> None:
        cap = RepoContext[object](workspace_dir=tmp_path, autoload_instructions=False, expose_inventory_tool=False)
        assert cap.get_instructions() is None

    def test_autoload_off_keeps_inventory_hint(self, tmp_path: Path) -> None:
        _write(tmp_path / 'CLAUDE.md', 'ignored')
        cap = RepoContext[object](workspace_dir=tmp_path, autoload_instructions=False)
        instructions = cap.get_instructions()
        assert isinstance(instructions, str)
        assert 'ignored' not in instructions
        assert 'inventory_agent_context' in instructions

    def test_no_files_no_inventory_is_none(self, tmp_path: Path) -> None:
        cap = RepoContext[object](workspace_dir=tmp_path, expose_inventory_tool=False)
        assert cap.get_instructions() is None

    def test_files_cached_across_calls(self, tmp_path: Path) -> None:
        _write(tmp_path / 'CLAUDE.md', 'first')
        cap = RepoContext[object](workspace_dir=tmp_path)
        assert cap.get_instructions() is not None and 'first' in cap.get_instructions()  # type: ignore[operator]
        _write(tmp_path / 'CLAUDE.md', 'second')
        # Read-once: the cached result is reused, so the edit is not picked up.
        assert 'second' not in cap.get_instructions()  # type: ignore[operator]


class TestToolset:
    def test_get_toolset_none_when_disabled(self, tmp_path: Path) -> None:
        assert RepoContext[object](workspace_dir=tmp_path, expose_inventory_tool=False).get_toolset() is None

    def test_get_toolset_present(self, tmp_path: Path) -> None:
        assert isinstance(RepoContext[object](workspace_dir=tmp_path).get_toolset(), RepoContextToolset)

    async def test_inventory_tool_runs_through_agent(self, tmp_path: Path) -> None:
        _write(tmp_path / '.claude' / 'skills' / 'foo' / 'SKILL.md', 'skill')
        agent = Agent(
            TestModel(call_tools=['inventory_agent_context']),
            capabilities=[RepoContext[object](workspace_dir=tmp_path)],
        )
        result = await agent.run('go')
        assert 'inventory_agent_context' in result.output


class TestScanAssets:
    def test_full_shape(self, tmp_path: Path) -> None:
        _write(tmp_path / '.claude' / 'skills' / 'foo' / 'SKILL.md', 's')
        _write(tmp_path / '.claude' / 'agents' / 'bar.md', 'a')
        _write(tmp_path / '.claude' / 'settings.json', '{}')
        inv = scan_assets(tmp_path, ('.claude', '.agents', '.codex', '.grok'))
        by_root = {r.root: r for r in inv.roots}
        claude = by_root['.claude']
        assert claude.exists
        assert claude.skills == ['.claude/skills/foo/SKILL.md']
        assert claude.agents == ['.claude/agents/bar.md']
        assert claude.settings == '.claude/settings.json'
        assert by_root['.agents'].exists is False
        assert by_root['.codex'].notes is not None
        assert by_root['.grok'].notes is not None

    def test_existing_root_without_settings(self, tmp_path: Path) -> None:
        _write(tmp_path / '.claude' / 'skills' / 'foo' / 'SKILL.md', 's')
        inv = scan_assets(tmp_path, ('.claude',))
        assert inv.roots[0].settings is None
        assert inv.roots[0].notes is None

    def test_returns_model(self, tmp_path: Path) -> None:
        assert isinstance(scan_assets(tmp_path, ()), AgentContextInventory)

    @pytest.mark.skipif(sys.platform == 'win32', reason='symlinks need privileges on Windows')
    def test_symlinked_asset_escaping_workspace(self, tmp_path: Path) -> None:
        workspace = tmp_path / 'ws'
        outside = _write(tmp_path / 'outside' / 'foo' / 'SKILL.md', 's')
        link = workspace / '.claude' / 'skills' / 'foo' / 'SKILL.md'
        link.parent.mkdir(parents=True)
        link.symlink_to(outside)
        inv = scan_assets(workspace, ('.claude',))
        claude = inv.roots[0]
        assert claude.exists
        assert len(claude.skills) == 1
        assert claude.skills[0].endswith('.claude/skills/foo/SKILL.md')


def _tool_returns(messages: list[ModelMessage]) -> int:
    return sum(isinstance(part, ToolReturnPart) for message in messages for part in message.parts)


def _repo_notes(messages: list[ModelMessage]) -> list[str]:
    notes: list[str] = []
    for message in messages:
        for part in message.parts:
            if not isinstance(part, UserPromptPart) or not isinstance(part.content, str):
                continue
            if '<repo-context>' in part.content or '<context-file ' in part.content:
                notes.append(part.content)
    return notes


class TestNestedTraversal:
    @pytest.mark.parametrize(('nested_inject', 'includes_body'), [('pointer', False), ('contents', True)])
    async def test_filesystem_event_enqueues_once_before_next_request(
        self, tmp_path: Path, nested_inject: Literal['pointer', 'contents'], includes_body: bool
    ) -> None:
        _write(tmp_path / 'sub' / 'AGENTS.md', 'NESTED BODY')
        _write(tmp_path / 'sub' / 'one.py', 'one')
        _write(tmp_path / 'sub' / 'two.py', 'two')

        async def stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[DeltaToolCalls | str]:
            returns = _tool_returns(messages)
            if returns == 0:
                yield {0: DeltaToolCall(name='read_file', json_args='{"path":"sub/one.py"}', tool_call_id='one')}
            elif returns == 1:
                notes = _repo_notes(messages)
                assert len(notes) == 1
                assert 'sub/AGENTS.md' in notes[0]
                assert ('NESTED BODY' in notes[0]) is includes_body
                yield {0: DeltaToolCall(name='read_file', json_args='{"path":"sub/two.py"}', tool_call_id='two')}
            else:
                assert len(_repo_notes(messages)) == 1
                yield 'done'

        await Agent(
            FunctionModel(stream_function=stream),
            capabilities=[
                FileSystem(root_dir=tmp_path),
                RepoContext(
                    workspace_dir=tmp_path,
                    autoload_instructions=False,
                    expose_inventory_tool=False,
                    nested_traversal=True,
                    nested_inject=nested_inject,
                ),
            ],
        ).run('go')

    async def test_filesystem_rooted_below_the_workspace_resolves_against_its_own_root(self, tmp_path: Path) -> None:
        project = tmp_path / 'project'
        _write(project / 'sub' / 'AGENTS.md', 'NESTED BODY')
        _write(project / 'sub' / 'one.py', 'one')
        # A decoy at the path the event would name if it were wrongly rebased onto the workspace.
        _write(tmp_path / 'sub' / 'AGENTS.md', 'DECOY BODY')

        async def stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[DeltaToolCalls | str]:
            if _tool_returns(messages) == 0:
                yield {0: DeltaToolCall(name='read_file', json_args='{"path":"sub/one.py"}', tool_call_id='one')}
            else:
                notes = _repo_notes(messages)
                assert len(notes) == 1
                assert 'project/sub/AGENTS.md' in notes[0]
                yield 'done'

        await Agent(
            FunctionModel(stream_function=stream),
            capabilities=[
                FileSystem(root_dir=project),
                RepoContext(
                    workspace_dir=tmp_path,
                    autoload_instructions=False,
                    expose_inventory_tool=False,
                    nested_traversal=True,
                ),
            ],
        ).run('go')

    async def test_filesystem_rooted_outside_the_workspace_enqueues_nothing(self, tmp_path: Path) -> None:
        workspace = tmp_path / 'workspace'
        workspace.mkdir()
        elsewhere = tmp_path / 'elsewhere'
        _write(elsewhere / 'AGENTS.md', 'OUTSIDE BODY')
        _write(elsewhere / 'one.py', 'one')

        async def stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[DeltaToolCalls | str]:
            if _tool_returns(messages) == 0:
                yield {0: DeltaToolCall(name='list_directory', json_args='{"path":"."}', tool_call_id='list')}
            else:
                assert _repo_notes(messages) == []
                yield 'done'

        await Agent(
            FunctionModel(stream_function=stream),
            capabilities=[
                FileSystem(root_dir=elsewhere),
                RepoContext(
                    workspace_dir=workspace,
                    autoload_instructions=False,
                    expose_inventory_tool=False,
                    nested_traversal=True,
                ),
            ],
        ).run('go')

    async def test_customized_sniff_fallback_warns_and_supports_non_event_tool(self, tmp_path: Path) -> None:
        _write(tmp_path / 'sub' / 'AGENTS.md', 'NESTED BODY')

        async def stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[DeltaToolCalls | str]:
            if _tool_returns(messages) == 0:
                yield {0: DeltaToolCall(name='list_dir', json_args='{"target":"sub"}', tool_call_id='list')}
            else:
                assert len(_repo_notes(messages)) == 1
                yield 'done'

        def list_dir(target: str) -> list[dict[str, str]]:
            return [{'name': 'AGENTS.md'}, {'name': 'one.py'}]

        with pytest.warns(
            HarnessDeprecationWarning,
            match='Traversal detection now reacts to `FileReadEvent` and `DirectoryListedEvent`',
        ):
            capability = RepoContext(
                workspace_dir=tmp_path,
                autoload_instructions=False,
                expose_inventory_tool=False,
                nested_traversal=True,
                traversal_tool_names=frozenset({'list_dir'}),
                traversal_path_arg='target',
            )

        await Agent(FunctionModel(stream_function=stream), capabilities=[capability], tools=[list_dir]).run('go')

    async def test_traversal_into_a_directory_without_a_context_file_enqueues_nothing(self, tmp_path: Path) -> None:
        _write(tmp_path / 'sub' / 'one.py', 'one')

        async def stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[DeltaToolCalls | str]:
            if _tool_returns(messages) == 0:
                yield {0: DeltaToolCall(name='read_file', json_args='{"path":"sub/one.py"}', tool_call_id='one')}
            else:
                assert _repo_notes(messages) == []
                yield 'done'

        await Agent(
            FunctionModel(stream_function=stream),
            capabilities=[
                FileSystem(root_dir=tmp_path),
                RepoContext(
                    workspace_dir=tmp_path,
                    autoload_instructions=False,
                    expose_inventory_tool=False,
                    nested_traversal=True,
                ),
            ],
        ).run('go')

    async def test_customized_sniff_ignores_a_non_string_path_and_accepts_an_absolute_one(self, tmp_path: Path) -> None:
        _write(tmp_path / 'sub' / 'AGENTS.md', 'NESTED BODY')

        async def stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[DeltaToolCalls | str]:
            returns = _tool_returns(messages)
            if returns == 0:
                yield {0: DeltaToolCall(name='list_dir', json_args='{"target":7}', tool_call_id='bad')}
            elif returns == 1:
                assert _repo_notes(messages) == []
                args = json.dumps({'target': str(tmp_path / 'sub')})
                yield {0: DeltaToolCall(name='list_dir', json_args=args, tool_call_id='abs')}
            else:
                assert len(_repo_notes(messages)) == 1
                yield 'done'

        def list_dir(target: Any) -> str:
            return 'listed'

        with pytest.warns(HarnessDeprecationWarning, match='Traversal detection now reacts'):
            capability = RepoContext(
                workspace_dir=tmp_path,
                autoload_instructions=False,
                expose_inventory_tool=False,
                nested_traversal=True,
                traversal_tool_names=frozenset({'list_dir'}),
                traversal_path_arg='target',
            )

        await Agent(FunctionModel(stream_function=stream), capabilities=[capability], tools=[list_dir]).run('go')

    async def test_customized_sniff_labels_a_context_file_outside_the_workspace(self, tmp_path: Path) -> None:
        workspace = tmp_path / 'workspace'
        workspace.mkdir()
        outside = _write(tmp_path / 'outside' / 'AGENTS.md', 'OUTSIDE BODY')

        async def stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[DeltaToolCalls | str]:
            if _tool_returns(messages) == 0:
                args = json.dumps({'target': str(outside.parent)})
                yield {0: DeltaToolCall(name='list_dir', json_args=args, tool_call_id='out')}
            else:
                notes = _repo_notes(messages)
                assert len(notes) == 1
                assert outside.as_posix() in notes[0]
                yield 'done'

        def list_dir(target: Any) -> str:
            return 'listed'

        with pytest.warns(HarnessDeprecationWarning, match='Traversal detection now reacts'):
            capability = RepoContext(
                workspace_dir=workspace,
                autoload_instructions=False,
                expose_inventory_tool=False,
                nested_traversal=True,
                traversal_tool_names=frozenset({'list_dir'}),
                traversal_path_arg='target',
            )

        await Agent(FunctionModel(stream_function=stream), capabilities=[capability], tools=[list_dir]).run('go')


class TestForRunAndMisc:
    def test_serialization_name(self) -> None:
        assert RepoContext.get_serialization_name() == 'RepoContext'
