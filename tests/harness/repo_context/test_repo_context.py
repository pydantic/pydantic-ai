"""Tests for the RepoContext capability."""

from __future__ import annotations

import json
import sys
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any, Literal
from unittest.mock import AsyncMock, MagicMock

import pytest

from pydantic_ai import Agent, RunContext
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import ModelMessage, ModelRequest, ToolReturnPart, UserPromptPart
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, DeltaToolCalls, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.usage import RunUsage
from pydantic_ai.workspaces import LocalWorkspaceBackend, Workspace
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


@pytest.fixture
def workspace(tmp_path: Path) -> Workspace:
    return Workspace(LocalWorkspaceBackend(working_dir=tmp_path))


def _run_context(workspace: Workspace) -> RunContext[object]:
    return RunContext[object](
        deps=None, model=TestModel(), usage=RunUsage(), prompt=None, messages=[], run_step=0, workspace=workspace
    )


def _write(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding='utf-8')
    return path


def _render_capability_instructions(capability: RepoContext[object], ctx: RunContext[object]) -> str | None:
    instructions = capability.get_instructions()
    assert callable(instructions)
    rendered = instructions(ctx)
    assert isinstance(rendered, str) or rendered is None
    return rendered


def _tool_returns(messages: list[ModelMessage]) -> int:
    return sum(isinstance(part, ToolReturnPart) for message in messages for part in message.parts)


def _repo_notes(messages: list[ModelMessage]) -> list[str]:
    notes: list[str] = []
    for message in messages:
        for part in message.parts:
            if (
                isinstance(part, UserPromptPart)
                and isinstance(part.content, str)
                and ('<repo-context>' in part.content or '<context-file ' in part.content)
            ):
                notes.append(part.content)
    return notes


class TestDiscoverInstructionFiles:
    async def test_does_not_load_instruction_link_outside_scanned_directory(
        self, tmp_path: Path, workspace: Workspace
    ) -> None:
        directory = tmp_path / 'project'
        directory.mkdir()
        outside = _write(tmp_path / 'AGENTS.md', 'untrusted')
        (directory / 'AGENTS.md').symlink_to(outside)
        assert await discover_instruction_files(workspace, directory, None, ('AGENTS.md',)) == []
        assert await find_dir_context_file(workspace, directory, ('AGENTS.md',)) is None

    async def test_walk_up_ancestor_first(self, tmp_path: Path, workspace: Workspace) -> None:
        _write(tmp_path / 'CLAUDE.md', 'root')
        workspace_dir = tmp_path / 'a' / 'b'
        _write(workspace_dir / 'CLAUDE.md', 'leaf')
        files = await discover_instruction_files(workspace, workspace_dir, tmp_path, ('CLAUDE.md',))
        assert [f.content for f in files] == ['root', 'leaf']

    async def test_home_none_only_workspace(self, tmp_path: Path, workspace: Workspace) -> None:
        _write(tmp_path / 'CLAUDE.md', 'root')
        workspace_dir = tmp_path / 'a'
        _write(workspace_dir / 'CLAUDE.md', 'leaf')
        files = await discover_instruction_files(workspace, workspace_dir, None, ('CLAUDE.md',))
        assert [f.content for f in files] == ['leaf']

    async def test_home_equals_workspace(self, tmp_path: Path, workspace: Workspace) -> None:
        _write(tmp_path / 'CLAUDE.md', 'only')
        files = await discover_instruction_files(workspace, tmp_path, tmp_path, ('CLAUDE.md',))
        assert [f.content for f in files] == ['only']

    async def test_home_not_ancestor_falls_back_to_workspace(self, tmp_path: Path, workspace: Workspace) -> None:
        workspace_dir = tmp_path / 'a'
        _write(workspace_dir / 'CLAUDE.md', 'leaf')
        unrelated = tmp_path / 'other'
        unrelated.mkdir()
        files = await discover_instruction_files(workspace, workspace_dir, unrelated, ('CLAUDE.md',))
        assert [f.content for f in files] == ['leaf']

    async def test_both_filenames_within_dir_order(self, tmp_path: Path, workspace: Workspace) -> None:
        _write(tmp_path / 'CLAUDE.md', 'claude')
        _write(tmp_path / 'AGENTS.md', 'agents')
        files = await discover_instruction_files(workspace, tmp_path, None, ('CLAUDE.md', 'AGENTS.md'))
        assert [f.content for f in files] == ['claude', 'agents']

    @pytest.mark.skipif(sys.platform == 'win32', reason='symlinks need privileges on Windows')
    async def test_symlink_deduped_by_content_hash(self, tmp_path: Path, workspace: Workspace) -> None:
        # Symlink-realpath dedup is dropped with the workspace migration (isolation is
        # the workspace's job); the content-hash dedup still catches shared bytes.
        _write(tmp_path / 'CLAUDE.md', 'shared')
        (tmp_path / 'AGENTS.md').symlink_to(tmp_path / 'CLAUDE.md')
        files = await discover_instruction_files(workspace, tmp_path, None, ('CLAUDE.md', 'AGENTS.md'))
        assert len(files) == 1

    async def test_identical_content_deduped_by_hash(self, tmp_path: Path, workspace: Workspace) -> None:
        _write(tmp_path / 'CLAUDE.md', 'same')
        workspace_dir = tmp_path / 'a'
        _write(workspace_dir / 'CLAUDE.md', 'same')
        files = await discover_instruction_files(workspace, workspace_dir, tmp_path, ('CLAUDE.md',))
        assert [f.content for f in files] == ['same']

    async def test_missing_files_skipped(self, tmp_path: Path, workspace: Workspace) -> None:
        files = await discover_instruction_files(workspace, tmp_path, None, ('CLAUDE.md',))
        assert files == []

    @pytest.mark.parametrize('error', [FileNotFoundError(), NotADirectoryError()])
    async def test_unreadable_model_path_does_not_block_capability_setup(self, tmp_path: Path, error: OSError) -> None:
        workspace = MagicMock(spec=Workspace)
        workspace.working_dir = AsyncMock(return_value=tmp_path.as_posix())
        workspace.stat = AsyncMock(side_effect=error)
        cap = RepoContext[object](expose_inventory_tool=False)
        ctx = _run_context(workspace=workspace)

        await cap.before_run(ctx)

        assert cap.get_instructions() is not None

    async def test_duplicate_filename_is_read_once(self, tmp_path: Path, workspace: Workspace) -> None:
        _write(tmp_path / 'CLAUDE.md', 'once')

        files = await discover_instruction_files(workspace, tmp_path, None, ('CLAUDE.md', 'CLAUDE.md'))

        assert [file.content for file in files] == ['once']

    async def test_non_utf8_file_does_not_crash(self, tmp_path: Path, workspace: Workspace) -> None:
        (tmp_path / 'CLAUDE.md').write_bytes(b'caf\xe9 instructions')
        files = await discover_instruction_files(workspace, tmp_path, None, ('CLAUDE.md',))
        assert len(files) == 1
        assert 'instructions' in files[0].content


class TestFindDirContextFile:
    async def test_first_existing_wins(self, tmp_path: Path, workspace: Workspace) -> None:
        _write(tmp_path / 'AGENTS.md', 'agents')
        found = await find_dir_context_file(workspace, tmp_path, ('CLAUDE.md', 'AGENTS.md'))
        assert found is not None
        assert found.content == 'agents'

    async def test_none_when_absent(self, tmp_path: Path, workspace: Workspace) -> None:
        assert await find_dir_context_file(workspace, tmp_path, ('CLAUDE.md',)) is None

    async def test_non_utf8_file_does_not_crash(self, tmp_path: Path, workspace: Workspace) -> None:
        (tmp_path / 'CLAUDE.md').write_bytes(b'caf\xe9 instructions')
        found = await find_dir_context_file(workspace, tmp_path, ('CLAUDE.md',))
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
    async def test_includes_files_and_inventory_hint(self, tmp_path: Path, workspace: Workspace) -> None:
        _write(tmp_path / 'CLAUDE.md', 'be nice')
        cap = RepoContext[object]()
        ctx = _run_context(workspace=workspace)
        await cap.before_run(ctx)
        instructions = _render_capability_instructions(cap, ctx)
        assert isinstance(instructions, str)
        assert 'be nice' in instructions
        assert 'inventory_agent_context' in instructions

    def test_none_when_all_disabled(self, tmp_path: Path) -> None:
        cap = RepoContext[object](autoload_instructions=False, expose_inventory_tool=False)
        assert cap.get_instructions() is None

    async def test_autoload_off_keeps_inventory_hint(self, tmp_path: Path, workspace: Workspace) -> None:
        _write(tmp_path / 'CLAUDE.md', 'ignored')
        cap = RepoContext[object](autoload_instructions=False)
        await cap.before_run(_run_context(workspace=workspace))
        instructions = cap.get_instructions()
        assert isinstance(instructions, str)
        assert 'ignored' not in instructions
        assert 'inventory_agent_context' in instructions

    async def test_no_files_no_inventory_is_none(self, tmp_path: Path, workspace: Workspace) -> None:
        cap = RepoContext[object](expose_inventory_tool=False)
        ctx = _run_context(workspace=workspace)
        await cap.before_run(ctx)
        assert _render_capability_instructions(cap, ctx) is None

    async def test_files_cached_across_calls(self, tmp_path: Path, workspace: Workspace) -> None:
        _write(tmp_path / 'CLAUDE.md', 'first')
        cap = RepoContext[object]()
        ctx = _run_context(workspace=workspace)
        await cap.before_run(ctx)
        first = _render_capability_instructions(cap, ctx)
        assert first is not None and 'first' in first
        _write(tmp_path / 'CLAUDE.md', 'second')
        second = _render_capability_instructions(cap, ctx)
        # Read-once: `before_run` loaded the file, so subsequent edits are not picked up.
        assert second is not None and 'second' not in second

    @pytest.mark.parametrize('autoload_instructions', [True, False])
    async def test_no_workspace_fails_the_run(self, autoload_instructions: bool) -> None:
        agent = Agent(TestModel(), capabilities=[RepoContext[object](autoload_instructions=autoload_instructions)])

        with pytest.raises(UserError, match='`RepoContext` needs a workspace'):
            await agent.run('go')

    async def test_walk_up_stops_at_a_home_dir_given_as_text(self, tmp_path: Path) -> None:
        _write(tmp_path / 'CLAUDE.md', 'home instructions')
        (tmp_path / 'repo').mkdir()
        cap = RepoContext[object](home_dir=str(tmp_path), expose_inventory_tool=False)
        ctx = _run_context(workspace=Workspace(LocalWorkspaceBackend(tmp_path / 'repo')))
        await cap.before_run(ctx)
        assert 'home instructions' in (_render_capability_instructions(cap, ctx) or '')


class TestToolset:
    def test_get_toolset_none_when_disabled(self, tmp_path: Path) -> None:
        assert RepoContext[object](expose_inventory_tool=False).get_toolset() is None

    def test_get_toolset_present(self, tmp_path: Path) -> None:
        assert isinstance(RepoContext[object]().get_toolset(), RepoContextToolset)

    async def test_inventory_tool_runs_through_agent(self, tmp_path: Path) -> None:
        _write(tmp_path / '.claude' / 'skills' / 'foo' / 'SKILL.md', 'skill')
        agent = Agent(
            TestModel(call_tools=['inventory_agent_context']),
            capabilities=[RepoContext[object]()],
        )
        backend = LocalWorkspaceBackend(working_dir=tmp_path)
        result = await agent.run('go', workspace=backend)
        assert 'inventory_agent_context' in result.output


class TestScanAssets:
    async def test_full_shape(self, tmp_path: Path, workspace: Workspace) -> None:
        _write(tmp_path / '.claude' / 'skills' / 'foo' / 'SKILL.md', 's')
        _write(tmp_path / '.claude' / 'agents' / 'bar.md', 'a')
        _write(tmp_path / '.claude' / 'settings.json', '{}')
        inv = await scan_assets(workspace, tmp_path, ('.claude', '.agents', '.codex', '.grok'))
        by_root = {r.root: r for r in inv.roots}
        claude = by_root['.claude']
        assert claude.exists
        assert claude.skills == ['.claude/skills/foo/SKILL.md']
        assert claude.agents == ['.claude/agents/bar.md']
        assert claude.settings == '.claude/settings.json'
        assert by_root['.agents'].exists is False
        assert by_root['.codex'].notes is not None
        assert by_root['.grok'].notes is not None

    async def test_existing_root_without_settings(self, tmp_path: Path, workspace: Workspace) -> None:
        _write(tmp_path / '.claude' / 'skills' / 'foo' / 'SKILL.md', 's')
        inv = await scan_assets(workspace, tmp_path, ('.claude',))
        assert inv.roots[0].settings is None
        assert inv.roots[0].notes is None

    async def test_root_without_skills_directory(self, tmp_path: Path, workspace: Workspace) -> None:
        _write(tmp_path / '.claude' / 'agents' / 'helper.md', 'agent')
        inv = await scan_assets(workspace, tmp_path, ('.claude',))
        assert inv.roots[0].skills == []
        assert inv.roots[0].agents == ['.claude/agents/helper.md']

    async def test_skill_walk_has_a_depth_bound(self, tmp_path: Path, workspace: Workspace) -> None:
        near = tmp_path / '.claude' / 'skills'
        for index in range(8):
            near /= f'level-{index}'
        _write(near / 'SKILL.md', 'near')
        _write(near / 'level-8' / 'SKILL.md', 'deep')

        inventory = await scan_assets(workspace, tmp_path, ('.claude',))

        assert inventory.roots[0].skills == [f'.claude/skills/{"/".join(f"level-{i}" for i in range(8))}/SKILL.md']

    async def test_file_at_asset_root_is_not_a_directory(self, tmp_path: Path, workspace: Workspace) -> None:
        _write(tmp_path / '.claude', 'not a directory')

        inv = await scan_assets(workspace, tmp_path, ('.claude',))

        assert inv.roots[0].exists is False

    async def test_unrecognized_inventory_entries_are_ignored(self, tmp_path: Path, workspace: Workspace) -> None:
        _write(tmp_path / '.claude' / 'README.md', 'not an agent')
        _write(tmp_path / '.claude' / 'skills' / 'README.md', 'not a skill')
        _write(tmp_path / '.claude' / 'settings.json', '{}')

        inv = await scan_assets(workspace, tmp_path, ('.claude',))

        assert inv.roots[0].settings == '.claude/settings.json'
        assert inv.roots[0].skills == []

    async def test_returns_model(self, tmp_path: Path, workspace: Workspace) -> None:
        assert isinstance(await scan_assets(workspace, tmp_path, ()), AgentContextInventory)

    @pytest.mark.skipif(sys.platform == 'win32', reason='symlinks need privileges on Windows')
    async def test_symlinked_asset_uses_confined_display_path(self, tmp_path: Path, workspace: Workspace) -> None:
        workspace_dir = tmp_path / 'ws'
        outside = _write(tmp_path / 'outside' / 'foo' / 'SKILL.md', 's')
        link = workspace_dir / '.claude' / 'skills' / 'foo' / 'SKILL.md'
        link.parent.mkdir(parents=True)
        link.symlink_to(outside)
        inv = await scan_assets(workspace, workspace_dir, ('.claude',))
        claude = inv.roots[0]
        assert claude.exists
        assert len(claude.skills) == 1
        assert claude.skills == ['.claude/skills/foo/SKILL.md']

    @pytest.mark.skipif(sys.platform == 'win32', reason='symlinks need privileges on Windows')
    async def test_skill_walk_lists_symlink_cycles_once(self, tmp_path: Path, workspace: Workspace) -> None:
        skills = tmp_path / '.claude' / 'skills'
        _write(skills / 'a' / 'SKILL.md', 's')
        # Three links to `.` would take thousands of listings (and report duplicates) without deduplication.
        for index in range(3):
            (skills / f'loop{index}').symlink_to('.')

        inventory = await scan_assets(workspace, tmp_path, ('.claude',))

        assert inventory.roots[0].skills == ['.claude/skills/a/SKILL.md']


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
                    autoload_instructions=False,
                    expose_inventory_tool=False,
                    nested_traversal=True,
                    nested_inject=nested_inject,
                ),
            ],
        ).run('go', workspace=LocalWorkspaceBackend(working_dir=tmp_path))

    async def test_filesystem_rooted_above_the_workspace_resolves_against_its_own_root(self, tmp_path: Path) -> None:
        project = tmp_path / 'project'
        _write(project / 'sub' / 'AGENTS.md', 'NESTED BODY')
        _write(project / 'sub' / 'one.py', 'one')
        # A decoy at the path the event would name if it were wrongly rebased onto the working directory.
        _write(project / 'project' / 'sub' / 'AGENTS.md', 'DECOY BODY')

        async def stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[DeltaToolCalls | str]:
            if _tool_returns(messages) == 0:
                yield {0: DeltaToolCall(name='read_file', json_args='{"path":"sub/one.py"}', tool_call_id='one')}
            else:
                notes = _repo_notes(messages)
                assert len(notes) == 1
                assert '(sub/AGENTS.md)' in notes[0]
                yield 'done'

        await Agent(
            FunctionModel(stream_function=stream),
            capabilities=[
                FileSystem(root_dir=tmp_path),
                RepoContext(autoload_instructions=False, expose_inventory_tool=False, nested_traversal=True),
            ],
        ).run('go', workspace=LocalWorkspaceBackend(working_dir=project))

    async def test_traversal_outside_the_working_directory_enqueues_nothing(self, tmp_path: Path) -> None:
        workspace = tmp_path / 'workspace'
        workspace.mkdir()
        elsewhere = tmp_path / 'elsewhere'
        _write(elsewhere / 'AGENTS.md', 'OUTSIDE BODY')
        _write(elsewhere / 'one.py', 'one')

        async def stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[DeltaToolCalls | str]:
            if _tool_returns(messages) == 0:
                yield {
                    0: DeltaToolCall(name='list_directory', json_args='{"path":"../elsewhere"}', tool_call_id='list')
                }
            else:
                assert _repo_notes(messages) == []
                yield 'done'

        await Agent(
            FunctionModel(stream_function=stream),
            capabilities=[
                FileSystem(root_dir=tmp_path),
                RepoContext(autoload_instructions=False, expose_inventory_tool=False, nested_traversal=True),
            ],
        ).run('go', workspace=LocalWorkspaceBackend(working_dir=workspace))

    @pytest.mark.parametrize('remove_before_return', [False, True])
    async def test_customized_sniff_fallback_warns_and_supports_non_event_tool(
        self, tmp_path: Path, remove_before_return: bool
    ) -> None:
        _write(tmp_path / 'sub' / 'AGENTS.md', 'NESTED BODY')

        async def stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[DeltaToolCalls | str]:
            if _tool_returns(messages) == 0:
                yield {0: DeltaToolCall(name='list_dir', json_args='{"target":"sub"}', tool_call_id='list')}
            else:
                assert len(_repo_notes(messages)) == (0 if remove_before_return else 1)
                yield 'done'

        def list_dir(target: str) -> list[dict[str, str]]:
            if remove_before_return:
                (tmp_path / target / 'AGENTS.md').unlink()
                (tmp_path / target).rmdir()
            return [{'name': 'AGENTS.md'}, {'name': 'one.py'}]

        with pytest.warns(
            HarnessDeprecationWarning,
            match='Traversal detection now reacts to `FileReadEvent` and `DirectoryListedEvent`',
        ):
            capability = RepoContext(
                autoload_instructions=False,
                expose_inventory_tool=False,
                nested_traversal=True,
                traversal_tool_names=frozenset({'list_dir'}),
                traversal_path_arg='target',
            )

        await Agent(FunctionModel(stream_function=stream), capabilities=[capability], tools=[list_dir]).run(
            'go', workspace=LocalWorkspaceBackend(working_dir=tmp_path)
        )

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
                    autoload_instructions=False,
                    expose_inventory_tool=False,
                    nested_traversal=True,
                ),
            ],
        ).run('go', workspace=LocalWorkspaceBackend(working_dir=tmp_path))

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
                autoload_instructions=False,
                expose_inventory_tool=False,
                nested_traversal=True,
                traversal_tool_names=frozenset({'list_dir'}),
                traversal_path_arg='target',
            )

        await Agent(FunctionModel(stream_function=stream), capabilities=[capability], tools=[list_dir]).run(
            'go', workspace=LocalWorkspaceBackend(working_dir=tmp_path)
        )

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
                autoload_instructions=False,
                expose_inventory_tool=False,
                nested_traversal=True,
                traversal_tool_names=frozenset({'list_dir'}),
                traversal_path_arg='target',
            )

        await Agent(FunctionModel(stream_function=stream), capabilities=[capability], tools=[list_dir]).run(
            'go', workspace=LocalWorkspaceBackend(working_dir=workspace)
        )


class TestForRunAndMisc:
    async def test_agent_reuse_isolates_instruction_state_between_workspaces(self, tmp_path: Path) -> None:
        first_root = tmp_path / 'first'
        second_root = tmp_path / 'second'
        _write(first_root / 'CLAUDE.md', 'first workspace instructions')
        _write(second_root / 'CLAUDE.md', 'second workspace instructions')
        captured: list[list[ModelMessage]] = []

        async def model(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
            del info
            captured.append(messages)
            yield 'done'

        agent = Agent(
            FunctionModel(stream_function=model),
            capabilities=[RepoContext[object](expose_inventory_tool=False)],
        )

        instructions: list[str] = []
        for root in (first_root, second_root):
            backend = LocalWorkspaceBackend(working_dir=root)
            await agent.run('go', workspace=backend)
            first_request = captured[-1][0]
            assert isinstance(first_request, ModelRequest)
            instructions.append(first_request.instructions or '')

        assert 'first workspace instructions' in instructions[0]
        assert 'second workspace instructions' not in instructions[0]
        assert 'second workspace instructions' in instructions[1]
        assert 'first workspace instructions' not in instructions[1]

    def test_serialization_name(self) -> None:
        assert RepoContext.get_serialization_name() == 'RepoContext'


async def test_disabled_nested_traversal_ignores_filesystem_event(tmp_path: Path) -> None:
    _write(tmp_path / 'sub' / 'AGENTS.md', 'NESTED BODY')
    _write(tmp_path / 'sub' / 'one.py', 'one')

    async def stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[DeltaToolCalls | str]:
        if _tool_returns(messages) == 0:
            yield {0: DeltaToolCall(name='read_file', json_args='{"path":"sub/one.py"}')}
        else:
            assert not _repo_notes(messages)
            yield 'done'

    await Agent(
        FunctionModel(stream_function=stream),
        capabilities=[
            FileSystem(root_dir=tmp_path),
            RepoContext(nested_traversal=False, expose_inventory_tool=False),
        ],
    ).run('go', workspace=LocalWorkspaceBackend(working_dir=tmp_path))
