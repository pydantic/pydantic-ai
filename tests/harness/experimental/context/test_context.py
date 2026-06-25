"""Tests for the RepoContext capability."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from pydantic_ai import Agent, RunContext
from pydantic_ai.messages import ToolCallPart
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.usage import RunUsage

from pydantic_ai_harness.experimental.context import (
    AgentContextInventory,
    ContextFile,
    RepoContext,
    RepoContextToolset,
)
from pydantic_ai_harness.experimental.context._inventory import scan_assets
from pydantic_ai_harness.experimental.context._loader import (
    discover_instruction_files,
    find_dir_context_file,
    render_context_files,
)

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return 'asyncio'


def _run_context() -> RunContext[None]:
    return RunContext[None](
        deps=None,
        model=TestModel(),
        usage=RunUsage(),
        prompt=None,
        messages=[],
        run_step=0,
    )


def _call(tool_name: str, **args: str) -> tuple[ToolCallPart, ToolDefinition, dict[str, str]]:
    return ToolCallPart(tool_name=tool_name, args=args), ToolDefinition(name=tool_name), args


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


class TestInstructions:
    def test_includes_files_and_inventory_hint(self, tmp_path: Path) -> None:
        _write(tmp_path / 'CLAUDE.md', 'be nice')
        cap = RepoContext[None](workspace_dir=tmp_path)
        instructions = cap.get_instructions()
        assert isinstance(instructions, str)
        assert 'be nice' in instructions
        assert 'inventory_agent_context' in instructions

    def test_none_when_all_disabled(self, tmp_path: Path) -> None:
        cap = RepoContext[None](workspace_dir=tmp_path, autoload_instructions=False, expose_inventory_tool=False)
        assert cap.get_instructions() is None

    def test_autoload_off_keeps_inventory_hint(self, tmp_path: Path) -> None:
        _write(tmp_path / 'CLAUDE.md', 'ignored')
        cap = RepoContext[None](workspace_dir=tmp_path, autoload_instructions=False)
        instructions = cap.get_instructions()
        assert isinstance(instructions, str)
        assert 'ignored' not in instructions
        assert 'inventory_agent_context' in instructions

    def test_no_files_no_inventory_is_none(self, tmp_path: Path) -> None:
        cap = RepoContext[None](workspace_dir=tmp_path, expose_inventory_tool=False)
        assert cap.get_instructions() is None

    def test_files_cached_across_calls(self, tmp_path: Path) -> None:
        _write(tmp_path / 'CLAUDE.md', 'first')
        cap = RepoContext[None](workspace_dir=tmp_path)
        assert cap.get_instructions() is not None and 'first' in cap.get_instructions()  # type: ignore[operator]
        _write(tmp_path / 'CLAUDE.md', 'second')
        # Read-once: the cached result is reused, so the edit is not picked up.
        assert 'second' not in cap.get_instructions()  # type: ignore[operator]


class TestToolset:
    def test_get_toolset_none_when_disabled(self, tmp_path: Path) -> None:
        assert RepoContext[None](workspace_dir=tmp_path, expose_inventory_tool=False).get_toolset() is None

    def test_get_toolset_present(self, tmp_path: Path) -> None:
        assert isinstance(RepoContext[None](workspace_dir=tmp_path).get_toolset(), RepoContextToolset)

    async def test_inventory_tool_runs_through_agent(self, tmp_path: Path) -> None:
        _write(tmp_path / '.claude' / 'skills' / 'foo' / 'SKILL.md', 'skill')
        agent = Agent(
            TestModel(call_tools=['inventory_agent_context']), capabilities=[RepoContext[None](workspace_dir=tmp_path)]
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


class TestNestedTraversal:
    async def test_off_by_default_returns_result(self, tmp_path: Path) -> None:
        _write(tmp_path / 'sub' / 'CLAUDE.md', 'nested')
        cap = RepoContext[None](workspace_dir=tmp_path)
        call, tool_def, args = _call('list_directory', path='sub')
        out = await cap.after_tool_execute(_run_context(), call=call, tool_def=tool_def, args=args, result='listing')
        assert out == 'listing'

    async def test_pointer_appended_on_first_traversal(self, tmp_path: Path) -> None:
        _write(tmp_path / 'sub' / 'CLAUDE.md', 'nested')
        cap = RepoContext[None](workspace_dir=tmp_path, nested_traversal=True)
        call, tool_def, args = _call('list_directory', path='sub')
        out = await cap.after_tool_execute(_run_context(), call=call, tool_def=tool_def, args=args, result='listing')
        assert out.startswith('listing')
        assert 'sub/CLAUDE.md' in out
        assert 'nested' not in out

    async def test_second_traversal_no_reappend(self, tmp_path: Path) -> None:
        _write(tmp_path / 'sub' / 'CLAUDE.md', 'nested')
        cap = RepoContext[None](workspace_dir=tmp_path, nested_traversal=True)
        call, tool_def, args = _call('list_directory', path='sub')
        ctx = _run_context()
        first = await cap.after_tool_execute(ctx, call=call, tool_def=tool_def, args=args, result='one')
        second = await cap.after_tool_execute(ctx, call=call, tool_def=tool_def, args=args, result='two')
        assert 'CLAUDE.md' in first
        assert second == 'two'

    async def test_tool_name_not_matched(self, tmp_path: Path) -> None:
        _write(tmp_path / 'sub' / 'CLAUDE.md', 'nested')
        cap = RepoContext[None](workspace_dir=tmp_path, nested_traversal=True)
        call, tool_def, args = _call('write_file', path='sub')
        out = await cap.after_tool_execute(_run_context(), call=call, tool_def=tool_def, args=args, result='r')
        assert out == 'r'

    async def test_non_str_path_arg_ignored(self, tmp_path: Path) -> None:
        cap = RepoContext[None](workspace_dir=tmp_path, nested_traversal=True)
        call = ToolCallPart(tool_name='list_directory', args={'path': 123})
        out = await cap.after_tool_execute(
            _run_context(), call=call, tool_def=ToolDefinition(name='list_directory'), args={'path': 123}, result='r'
        )
        assert out == 'r'

    async def test_dir_without_context_file_untouched(self, tmp_path: Path) -> None:
        (tmp_path / 'sub').mkdir()
        cap = RepoContext[None](workspace_dir=tmp_path, nested_traversal=True)
        call, tool_def, args = _call('list_directory', path='sub')
        out = await cap.after_tool_execute(_run_context(), call=call, tool_def=tool_def, args=args, result='r')
        assert out == 'r'

    async def test_read_file_uses_parent_dir(self, tmp_path: Path) -> None:
        _write(tmp_path / 'sub' / 'CLAUDE.md', 'nested')
        target = _write(tmp_path / 'sub' / 'code.py', 'x = 1')
        cap = RepoContext[None](workspace_dir=tmp_path, nested_traversal=True)
        call, tool_def, args = _call('read_file', path=str(target))
        out = await cap.after_tool_execute(_run_context(), call=call, tool_def=tool_def, args=args, result='file body')
        assert 'CLAUDE.md' in out

    async def test_contents_mode_inlines_body(self, tmp_path: Path) -> None:
        _write(tmp_path / 'sub' / 'CLAUDE.md', 'NESTED BODY')
        cap = RepoContext[None](workspace_dir=tmp_path, nested_traversal=True, nested_inject='contents')
        call, tool_def, args = _call('list_directory', path='sub')
        out = await cap.after_tool_execute(_run_context(), call=call, tool_def=tool_def, args=args, result='r')
        assert 'NESTED BODY' in out

    async def test_label_falls_back_when_dir_outside_workspace(self, tmp_path: Path) -> None:
        workspace = tmp_path / 'ws'
        workspace.mkdir()
        outside = _write(tmp_path / 'outside' / 'CLAUDE.md', 'nested').parent
        cap = RepoContext[None](workspace_dir=workspace, nested_traversal=True)
        call, tool_def, args = _call('list_directory', path=str(outside))
        out = await cap.after_tool_execute(_run_context(), call=call, tool_def=tool_def, args=args, result='r')
        assert outside.resolve().as_posix() in out

    async def test_non_str_result_returned_unchanged(self, tmp_path: Path) -> None:
        _write(tmp_path / 'sub' / 'CLAUDE.md', 'nested')
        cap = RepoContext[None](
            workspace_dir=tmp_path, nested_traversal=True, traversal_tool_names=frozenset({'list_dir', 'read_file'})
        )
        call, tool_def, args = _call('list_dir', path='sub')
        listing = [{'name': 'CLAUDE.md'}, {'name': 'code.py'}]
        out = await cap.after_tool_execute(_run_context(), call=call, tool_def=tool_def, args=args, result=listing)
        assert out is listing

    async def test_string_result_still_gets_note(self, tmp_path: Path) -> None:
        _write(tmp_path / 'sub' / 'CLAUDE.md', 'nested')
        cap = RepoContext[None](workspace_dir=tmp_path, nested_traversal=True)
        call, tool_def, args = _call('list_directory', path='sub')
        out = await cap.after_tool_execute(_run_context(), call=call, tool_def=tool_def, args=args, result='listing')
        assert out.startswith('listing')
        assert 'CLAUDE.md' in out


class TestForRunAndMisc:
    async def test_for_run_isolates_state(self, tmp_path: Path) -> None:
        _write(tmp_path / 'sub' / 'CLAUDE.md', 'nested')
        base = RepoContext[None](workspace_dir=tmp_path, nested_traversal=True)
        run_cap = await base.for_run(_run_context())
        call, tool_def, args = _call('list_directory', path='sub')
        await run_cap.after_tool_execute(_run_context(), call=call, tool_def=tool_def, args=args, result='r')
        fresh = await base.for_run(_run_context())
        out = await fresh.after_tool_execute(_run_context(), call=call, tool_def=tool_def, args=args, result='r2')
        assert 'CLAUDE.md' in out

    def test_serialization_name(self) -> None:
        assert RepoContext.get_serialization_name() == 'RepoContext'
