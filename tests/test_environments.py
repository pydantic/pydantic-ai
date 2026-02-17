"""Tests for pydantic_ai.environments — ExecutionEnvironment, ExecutionToolset, LocalEnvironment, and MemoryEnvironment."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest
from inline_snapshot import snapshot

from pydantic_ai import ToolCallPart
from pydantic_ai._run_context import RunContext
from pydantic_ai._tool_manager import ToolManager
from pydantic_ai.environments import ExecuteResult, ExecutionToolset, FileInfo
from pydantic_ai.environments.local import LocalEnvironment
from pydantic_ai.environments.memory import MemoryEnvironment
from pydantic_ai.exceptions import UnexpectedModelBehavior
from pydantic_ai.models.test import TestModel
from pydantic_ai.usage import RunUsage

pytestmark = pytest.mark.anyio


def build_run_context(deps: Any = None, run_step: int = 0) -> RunContext[Any]:
    return RunContext(
        deps=deps,
        model=TestModel(),
        usage=RunUsage(),
        prompt=None,
        messages=[],
        run_step=run_step,
    )


# --- Data types ---


def test_execute_result():
    result = ExecuteResult(output='hello\n', exit_code=0)
    assert result.output == 'hello\n'
    assert result.exit_code == 0
    assert result.truncated is False


def test_execute_result_truncated():
    result = ExecuteResult(output='data', exit_code=1, truncated=True)
    assert result.truncated is True


def test_file_info():
    info = FileInfo(name='test.py', path='src/test.py', is_dir=False, size=42)
    assert info.name == 'test.py'
    assert info.is_dir is False
    assert info.size == 42


def test_file_info_directory():
    info = FileInfo(name='src', path='src', is_dir=True)
    assert info.is_dir is True
    assert info.size is None


# --- LocalEnvironment: execute ---


async def test_local_execute_basic(tmp_path: Path):
    async with LocalEnvironment(tmp_path) as env:
        result = await env.execute('echo hello')
        assert result.exit_code == 0
        assert 'hello' in result.output


async def test_local_execute_exit_code(tmp_path: Path):
    async with LocalEnvironment(tmp_path) as env:
        result = await env.execute('exit 42')
        assert result.exit_code == 42


async def test_local_execute_timeout(tmp_path: Path):
    async with LocalEnvironment(tmp_path) as env:
        result = await env.execute('sleep 10', timeout=0.5)
        assert result.exit_code == -1
        assert 'timed out' in result.output.lower()


async def test_local_execute_stderr(tmp_path: Path):
    async with LocalEnvironment(tmp_path) as env:
        result = await env.execute('echo error >&2')
        assert 'error' in result.output


# --- LocalEnvironment: environment variables ---


async def test_local_env_vars_baseline(tmp_path: Path):
    async with LocalEnvironment(tmp_path, env_vars={'MY_VAR': 'baseline'}) as env:
        result = await env.execute('echo $MY_VAR')
        assert 'baseline' in result.output


async def test_local_env_vars_per_call(tmp_path: Path):
    async with LocalEnvironment(tmp_path) as env:
        result = await env.execute('echo $CALL_VAR', env={'CALL_VAR': 'per_call'})
        assert 'per_call' in result.output


async def test_local_env_vars_merged(tmp_path: Path):
    async with LocalEnvironment(tmp_path, env_vars={'BASE': 'one'}) as env:
        result = await env.execute('echo $BASE $EXTRA', env={'EXTRA': 'two'})
        assert 'one' in result.output
        assert 'two' in result.output


async def test_local_env_vars_per_call_overrides_baseline(tmp_path: Path):
    async with LocalEnvironment(tmp_path, env_vars={'VAR': 'old'}) as env:
        result = await env.execute('echo $VAR', env={'VAR': 'new'})
        assert 'new' in result.output
        assert 'old' not in result.output


async def test_local_inherit_env_true(tmp_path: Path):
    os.environ['_TEST_INHERIT_CHECK'] = 'inherited'
    try:
        async with LocalEnvironment(tmp_path, inherit_env=True) as env:
            result = await env.execute('echo $_TEST_INHERIT_CHECK')
            assert 'inherited' in result.output
    finally:
        del os.environ['_TEST_INHERIT_CHECK']


async def test_local_inherit_env_false(tmp_path: Path):
    os.environ['_TEST_INHERIT_CHECK'] = 'should_not_see'
    try:
        async with LocalEnvironment(tmp_path, inherit_env=False) as env:
            result = await env.execute('echo x${_TEST_INHERIT_CHECK}x')
            assert result.output.strip() == 'xx'
    finally:
        del os.environ['_TEST_INHERIT_CHECK']


async def test_local_inherit_env_false_with_explicit_vars(tmp_path: Path):
    async with LocalEnvironment(tmp_path, env_vars={'ONLY_THIS': 'yes'}, inherit_env=False) as env:
        result = await env.execute('/bin/echo $ONLY_THIS')
        assert 'yes' in result.output


# --- LocalEnvironment: file operations ---


async def test_local_write_and_read(tmp_path: Path):
    async with LocalEnvironment(tmp_path) as env:
        await env.write_file('test.txt', 'line one\nline two\n')
        content = await env.read_file('test.txt')
        assert isinstance(content, str)
        assert 'line one' in content
        assert 'line two' in content


async def test_local_read_line_numbers(tmp_path: Path):
    async with LocalEnvironment(tmp_path) as env:
        await env.write_file('numbered.txt', 'alpha\nbeta\ngamma\n')
        content = await env.read_file('numbered.txt')
        assert content == snapshot("""\
     1\talpha
     2\tbeta
     3\tgamma
""")


async def test_local_read_with_offset_limit(tmp_path: Path):
    async with LocalEnvironment(tmp_path) as env:
        lines = '\n'.join(f'line {i}' for i in range(20))
        await env.write_file('long.txt', lines)

        content = await env.read_file('long.txt', offset=5, limit=3)
        assert content == snapshot("""\
     6\tline 5
     7\tline 6
     8\tline 7
... (12 more lines. Use offset=8 to continue reading.)
""")


async def test_local_read_continuation_hint(tmp_path: Path):
    async with LocalEnvironment(tmp_path) as env:
        lines = '\n'.join(f'line {i}' for i in range(20))
        await env.write_file('long.txt', lines)

        content = await env.read_file('long.txt', offset=0, limit=5)
        assert content == snapshot("""\
     1\tline 0
     2\tline 1
     3\tline 2
     4\tline 3
     5\tline 4
... (15 more lines. Use offset=5 to continue reading.)
""")


async def test_local_read_offset_out_of_bounds(tmp_path: Path):
    async with LocalEnvironment(tmp_path) as env:
        await env.write_file('short.txt', 'one\ntwo\n')
        with pytest.raises(ValueError, match='Offset 100 exceeds file length'):
            await env.read_file('short.txt', offset=100)


async def test_local_read_directory_error(tmp_path: Path):
    async with LocalEnvironment(tmp_path) as env:
        (tmp_path / 'subdir').mkdir()
        with pytest.raises(FileNotFoundError, match='is a directory'):
            await env.read_file('subdir')


async def test_local_read_nonexistent(tmp_path: Path):
    async with LocalEnvironment(tmp_path) as env:
        with pytest.raises(FileNotFoundError):
            await env.read_file('nonexistent.txt')


async def test_local_write_creates_parent_dirs(tmp_path: Path):
    async with LocalEnvironment(tmp_path) as env:
        await env.write_file('deep/nested/dir/file.txt', 'content')
        content = await env.read_file('deep/nested/dir/file.txt')
        assert isinstance(content, str)
        assert 'content' in content


async def test_local_write_binary(tmp_path: Path):
    async with LocalEnvironment(tmp_path) as env:
        await env.write_file('binary.bin', b'\x00\x01\x02\x03')
        assert (tmp_path / 'binary.bin').read_bytes() == b'\x00\x01\x02\x03'


async def test_local_read_file_bytes(tmp_path: Path):
    # Create a minimal PNG (1x1 transparent pixel)
    png_data = (
        b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01'
        b'\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89'
        b'\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01'
        b'\r\n\xb4\x00\x00\x00\x00IEND\xaeB`\x82'
    )
    async with LocalEnvironment(tmp_path) as env:
        await env.write_file('image.png', png_data)
        result = await env.read_file_bytes('image.png')
        assert isinstance(result, bytes)
        assert result == png_data


# --- LocalEnvironment: edit_file ---


async def test_local_edit_single_replacement(tmp_path: Path):
    async with LocalEnvironment(tmp_path) as env:
        await env.write_file('edit.txt', 'foo bar baz')
        count = await env.edit_file('edit.txt', 'bar', 'BAR')
        assert count == 1
        content = (tmp_path / 'edit.txt').read_text()
        assert content == 'foo BAR baz'


async def test_local_edit_replace_all(tmp_path: Path):
    async with LocalEnvironment(tmp_path) as env:
        await env.write_file('edit.txt', 'aaa bbb aaa')
        count = await env.edit_file('edit.txt', 'aaa', 'xxx', replace_all=True)
        assert count == 2
        content = (tmp_path / 'edit.txt').read_text()
        assert content == 'xxx bbb xxx'


async def test_local_edit_not_found(tmp_path: Path):
    async with LocalEnvironment(tmp_path) as env:
        await env.write_file('edit.txt', 'hello world')
        with pytest.raises(ValueError, match='not found'):
            await env.edit_file('edit.txt', 'missing', 'replacement')


async def test_local_edit_ambiguous_without_replace_all(tmp_path: Path):
    async with LocalEnvironment(tmp_path) as env:
        await env.write_file('edit.txt', 'dup dup dup')
        with pytest.raises(ValueError, match='3 times'):
            await env.edit_file('edit.txt', 'dup', 'unique')


async def test_local_edit_nonexistent_file(tmp_path: Path):
    async with LocalEnvironment(tmp_path) as env:
        with pytest.raises(FileNotFoundError):
            await env.edit_file('missing.txt', 'old', 'new')


async def test_local_edit_multiline(tmp_path: Path):
    async with LocalEnvironment(tmp_path) as env:
        await env.write_file('code.py', 'def foo():\n    return "old"\n\nprint("test")\n')
        count = await env.edit_file('code.py', 'def foo():\n    return "old"', 'def foo():\n    return "new"')
        assert count == 1
        content = (tmp_path / 'code.py').read_text()
        assert 'return "new"' in content
        assert 'return "old"' not in content
        assert 'print("test")' in content


# --- LocalEnvironment: ls ---


async def test_local_ls(tmp_path: Path):
    async with LocalEnvironment(tmp_path) as env:
        await env.write_file('a.txt', 'a')
        await env.write_file('b.txt', 'b')
        (tmp_path / 'subdir').mkdir()

        entries = await env.ls('.')
        names = {e.name for e in entries}
        assert 'a.txt' in names
        assert 'b.txt' in names
        assert 'subdir' in names

        dirs = [e for e in entries if e.is_dir]
        files = [e for e in entries if not e.is_dir]
        assert any(d.name == 'subdir' for d in dirs)
        assert all(f.size is not None and f.size > 0 for f in files)


async def test_local_ls_not_a_directory(tmp_path: Path):
    async with LocalEnvironment(tmp_path) as env:
        await env.write_file('file.txt', 'content')
        with pytest.raises(NotADirectoryError):
            await env.ls('file.txt')


# --- LocalEnvironment: glob ---


async def test_local_glob(tmp_path: Path):
    async with LocalEnvironment(tmp_path) as env:
        await env.write_file('src/main.py', '# main')
        await env.write_file('src/utils.py', '# utils')
        await env.write_file('src/data.json', '{}')

        matches = await env.glob('**/*.py')
        assert len(matches) == 2
        assert any('main.py' in m for m in matches)
        assert any('utils.py' in m for m in matches)
        assert not any('data.json' in m for m in matches)


async def test_local_glob_no_matches(tmp_path: Path):
    async with LocalEnvironment(tmp_path) as env:
        matches = await env.glob('**/*.rs')
        assert matches == []


# --- LocalEnvironment: grep ---


async def test_local_grep(tmp_path: Path):
    async with LocalEnvironment(tmp_path) as env:
        await env.write_file('a.py', 'def hello():\n    pass\n')
        await env.write_file('b.py', 'x = 1\n')

        result = await env.grep('hello')
        assert 'a.py' in result
        assert 'hello' in result
        assert 'b.py' not in result


async def test_local_grep_with_glob_pattern(tmp_path: Path):
    async with LocalEnvironment(tmp_path) as env:
        await env.write_file('code.py', 'target = 1\n')
        await env.write_file('code.js', 'target = 2\n')

        result = await env.grep('target', glob_pattern='*.py')
        assert 'code.py' in result
        assert 'code.js' not in result


async def test_local_grep_line_numbers(tmp_path: Path):
    async with LocalEnvironment(tmp_path) as env:
        await env.write_file('test.txt', 'alpha\nbeta\ngamma\nbeta\n')

        result = await env.grep('beta')
        assert result == snapshot('test.txt:2:beta\ntest.txt:4:beta')


async def test_local_grep_no_matches(tmp_path: Path):
    async with LocalEnvironment(tmp_path) as env:
        await env.write_file('test.txt', 'nothing interesting')
        result = await env.grep('nonexistent_pattern')
        assert result == ''


async def test_local_grep_skips_hidden_files(tmp_path: Path):
    async with LocalEnvironment(tmp_path) as env:
        await env.write_file('visible.py', 'target_string\n')
        (tmp_path / '.hidden').mkdir()
        (tmp_path / '.hidden' / 'secret.py').write_text('target_string\n')
        (tmp_path / '.dotfile').write_text('target_string\n')

        result = await env.grep('target_string')
        assert 'visible.py' in result
        assert '.hidden' not in result
        assert '.dotfile' not in result


# --- LocalEnvironment: create_process ---


async def test_local_create_process(tmp_path: Path):
    async with LocalEnvironment(tmp_path) as env:
        proc = await env.create_process('echo interactive')
        async with proc:
            data = await proc.recv(timeout=5)
            assert b'interactive' in data


async def test_local_create_process_env(tmp_path: Path):
    async with LocalEnvironment(tmp_path) as env:
        proc = await env.create_process('echo $PROC_VAR', env={'PROC_VAR': 'from_process'})
        async with proc:
            data = await proc.recv(timeout=5)
            assert b'from_process' in data


async def test_local_create_process_stdin(tmp_path: Path):
    async with LocalEnvironment(tmp_path) as env:
        # Use head -1 so the process exits after reading one line
        proc = await env.create_process('head -1')
        async with proc:
            await proc.send(b'hello from stdin\n')
            data = await proc.recv(timeout=5)
            assert b'hello from stdin' in data


async def test_local_process_wait(tmp_path: Path):
    async with LocalEnvironment(tmp_path) as env:
        proc = await env.create_process('exit 7')
        async with proc:
            rc = await proc.wait(timeout=5)
            assert rc == 7


async def test_local_process_kill(tmp_path: Path):
    async with LocalEnvironment(tmp_path) as env:
        proc = await env.create_process('sleep 60')
        # Don't use async with — we want to test manual kill
        await proc.kill()
        assert proc.returncode is not None


# --- LocalEnvironment: path traversal ---


async def test_local_path_traversal_blocked(tmp_path: Path):
    async with LocalEnvironment(tmp_path) as env:
        with pytest.raises(PermissionError, match='outside the environment root'):
            await env.read_file('../../../etc/passwd')


async def test_local_path_traversal_write_blocked(tmp_path: Path):
    async with LocalEnvironment(tmp_path) as env:
        with pytest.raises(PermissionError, match='outside the environment root'):
            await env.write_file('../escape.txt', 'malicious')


# --- LocalEnvironment: creates root dir ---


async def test_local_creates_root_dir(tmp_path: Path):
    root = tmp_path / 'new_root'
    assert not root.exists()
    async with LocalEnvironment(root) as env:
        assert root.exists()
        result = await env.execute('echo works')
        assert 'works' in result.output


# --- ExecutionToolset ---


async def test_toolset_tool_names():
    toolset = ExecutionToolset(LocalEnvironment('.'))
    tool_names = sorted(toolset.tools.keys())
    assert tool_names == snapshot(['bash', 'edit_file', 'glob', 'grep', 'read_file', 'write_file'])


async def test_toolset_include_flags():
    toolset = ExecutionToolset(
        LocalEnvironment('.'),
        include_bash=False,
        include_file_tools=False,
        include_search_tools=False,
    )
    assert toolset.tools == {}


async def test_toolset_include_bash_only():
    toolset = ExecutionToolset(
        LocalEnvironment('.'),
        include_bash=True,
        include_file_tools=False,
        include_search_tools=False,
    )
    assert sorted(toolset.tools.keys()) == ['bash']


async def test_toolset_bash_tool(tmp_path: Path):
    env = LocalEnvironment(tmp_path)
    toolset = ExecutionToolset(env)
    ctx = build_run_context(None)
    manager = await ToolManager[None](toolset).for_run_step(ctx)

    async with env:
        result = await manager.handle_call(ToolCallPart(tool_name='bash', args={'command': 'echo hello'}))
        assert result == snapshot("""\
hello

Exit code: 0\
""")


async def test_toolset_read_write_tools(tmp_path: Path):
    env = LocalEnvironment(tmp_path)
    toolset = ExecutionToolset(env)
    ctx = build_run_context(None)
    manager = await ToolManager[None](toolset).for_run_step(ctx)

    async with env:
        # Write
        write_result = await manager.handle_call(
            ToolCallPart(tool_name='write_file', args={'path': 'test.txt', 'content': 'hello world'})
        )
        assert write_result == snapshot('File written: test.txt')

        # Read
        read_result = await manager.handle_call(ToolCallPart(tool_name='read_file', args={'path': 'test.txt'}))
        assert read_result == snapshot('     1\thello world\n')


async def test_toolset_edit_retry_on_error(tmp_path: Path):
    env = LocalEnvironment(tmp_path)
    toolset = ExecutionToolset(env, max_retries=0)
    ctx = build_run_context(None)
    manager = await ToolManager[None](toolset).for_run_step(ctx)

    async with env:
        await env.write_file('test.txt', 'content')

        # Edit with non-matching string: ModelRetry is raised by tool, but with max_retries=0
        # the ToolManager wraps it into UnexpectedModelBehavior
        with pytest.raises(UnexpectedModelBehavior, match='exceeded max retries count of 0'):
            await manager.handle_call(
                ToolCallPart(
                    tool_name='edit_file',
                    args={'path': 'test.txt', 'old_string': 'nonexistent', 'new_string': 'replacement'},
                )
            )


async def test_toolset_glob_tool(tmp_path: Path):
    env = LocalEnvironment(tmp_path)
    toolset = ExecutionToolset(env)
    ctx = build_run_context(None)
    manager = await ToolManager[None](toolset).for_run_step(ctx)

    async with env:
        await env.write_file('a.py', '# a')
        await env.write_file('b.py', '# b')

        result = await manager.handle_call(ToolCallPart(tool_name='glob', args={'pattern': '*.py'}))
        assert result == snapshot("""\
a.py
b.py\
""")


async def test_toolset_grep_tool(tmp_path: Path):
    env = LocalEnvironment(tmp_path)
    toolset = ExecutionToolset(env)
    ctx = build_run_context(None)
    manager = await ToolManager[None](toolset).for_run_step(ctx)

    async with env:
        await env.write_file('search.py', 'def find_me():\n    pass\n')

        result = await manager.handle_call(ToolCallPart(tool_name='grep', args={'pattern': 'find_me'}))
        assert result == snapshot('search.py:1:def find_me():')


# --- ExecutionToolset: error handling ---


async def test_toolset_read_nonexistent_returns_error(tmp_path: Path):
    """read_file on a nonexistent file returns an error string instead of crashing."""
    env = LocalEnvironment(tmp_path)
    toolset = ExecutionToolset(env)
    ctx = build_run_context(None)
    manager = await ToolManager[None](toolset).for_run_step(ctx)

    async with env:
        result = await manager.handle_call(ToolCallPart(tool_name='read_file', args={'path': 'nope.txt'}))
        assert 'Error:' in str(result)


async def test_toolset_read_path_traversal_returns_error(tmp_path: Path):
    """read_file with path traversal returns an error string."""
    env = LocalEnvironment(tmp_path)
    toolset = ExecutionToolset(env)
    ctx = build_run_context(None)
    manager = await ToolManager[None](toolset).for_run_step(ctx)

    async with env:
        result = await manager.handle_call(ToolCallPart(tool_name='read_file', args={'path': '../../etc/passwd'}))
        assert 'Error:' in str(result)


async def test_toolset_write_path_traversal_returns_error(tmp_path: Path):
    """write_file with path traversal returns an error string."""
    env = LocalEnvironment(tmp_path)
    toolset = ExecutionToolset(env)
    ctx = build_run_context(None)
    manager = await ToolManager[None](toolset).for_run_step(ctx)

    async with env:
        result = await manager.handle_call(
            ToolCallPart(tool_name='write_file', args={'path': '../../tmp/evil.txt', 'content': 'bad'})
        )
        assert 'Error:' in str(result)


async def test_toolset_glob_path_traversal_returns_error(tmp_path: Path):
    """glob with path traversal returns an error string."""
    env = LocalEnvironment(tmp_path)
    toolset = ExecutionToolset(env)
    ctx = build_run_context(None)
    manager = await ToolManager[None](toolset).for_run_step(ctx)

    async with env:
        result = await manager.handle_call(
            ToolCallPart(tool_name='glob', args={'pattern': '*.py', 'path': '../../etc'})
        )
        assert 'Error:' in str(result)


async def test_toolset_grep_invalid_regex_returns_error(tmp_path: Path):
    """grep with invalid regex returns an error string."""
    env = LocalEnvironment(tmp_path)
    toolset = ExecutionToolset(env)
    ctx = build_run_context(None)
    manager = await ToolManager[None](toolset).for_run_step(ctx)

    async with env:
        await env.write_file('test.txt', 'content')

        result = await manager.handle_call(ToolCallPart(tool_name='grep', args={'pattern': '[invalid'}))
        assert 'Error:' in str(result)


async def test_toolset_read_offset_out_of_bounds_returns_error(tmp_path: Path):
    """read_file with offset past EOF returns an error string."""
    env = LocalEnvironment(tmp_path)
    toolset = ExecutionToolset(env)
    ctx = build_run_context(None)
    manager = await ToolManager[None](toolset).for_run_step(ctx)

    async with env:
        await env.write_file('short.txt', 'one\ntwo\n')

        result = await manager.handle_call(
            ToolCallPart(tool_name='read_file', args={'path': 'short.txt', 'offset': 100})
        )
        assert 'Error:' in str(result)
        assert 'Offset 100 exceeds' in str(result)


async def test_toolset_read_continuation_hint(tmp_path: Path):
    """read_file includes continuation hint when there are more lines."""
    env = LocalEnvironment(tmp_path)
    toolset = ExecutionToolset(env)
    ctx = build_run_context(None)
    manager = await ToolManager[None](toolset).for_run_step(ctx)

    async with env:
        lines = '\n'.join(f'line {i}' for i in range(20))
        await env.write_file('long.txt', lines)

        result = await manager.handle_call(
            ToolCallPart(tool_name='read_file', args={'path': 'long.txt', 'offset': 0, 'limit': 5})
        )
        assert result == snapshot("""\
     1	line 0
     2	line 1
     3	line 2
     4	line 3
     5	line 4
... (15 more lines. Use offset=5 to continue reading.)
""")


# --- ExecutionToolset: approval flags ---


async def test_toolset_require_bash_approval():
    """require_bash_approval sets requires_approval on the bash tool."""
    toolset = ExecutionToolset(require_bash_approval=True)
    ctx = build_run_context(None)
    tools = await toolset.get_tools(ctx)
    assert tools['bash'].tool_def.kind == 'unapproved'
    # Other tools should be normal
    assert tools['read_file'].tool_def.kind == 'function'


async def test_toolset_require_write_approval():
    """require_write_approval sets requires_approval on write_file and edit_file."""
    toolset = ExecutionToolset(require_write_approval=True)
    ctx = build_run_context(None)
    tools = await toolset.get_tools(ctx)
    assert tools['write_file'].tool_def.kind == 'unapproved'
    assert tools['edit_file'].tool_def.kind == 'unapproved'
    # read_file and search tools should NOT require approval
    assert tools['read_file'].tool_def.kind == 'function'
    assert tools['glob'].tool_def.kind == 'function'
    assert tools['grep'].tool_def.kind == 'function'


async def test_toolset_default_no_approval():
    """By default, no tools require approval."""
    toolset = ExecutionToolset()
    ctx = build_run_context(None)
    tools = await toolset.get_tools(ctx)
    for tool in tools.values():
        assert tool.tool_def.kind == 'function'


# --- ExecutionToolset: environment management ---


async def test_toolset_environment_property():
    env = LocalEnvironment('.')
    toolset = ExecutionToolset(env)
    assert toolset.environment is env
    assert toolset.required_environment is env


async def test_toolset_no_environment_returns_none():
    toolset = ExecutionToolset()
    assert toolset.environment is None


async def test_toolset_no_environment_required_raises():
    toolset = ExecutionToolset()
    with pytest.raises(RuntimeError, match='No execution environment configured'):
        _ = toolset.required_environment


async def test_toolset_use_environment():
    env1 = LocalEnvironment('/tmp/env1')
    env2 = LocalEnvironment('/tmp/env2')
    toolset = ExecutionToolset(env1)

    assert toolset.environment is env1
    with toolset.use_environment(env2):
        assert toolset.environment is env2
    assert toolset.environment is env1


async def test_toolset_use_environment_no_default():
    env = LocalEnvironment('.')
    toolset = ExecutionToolset()

    assert toolset.environment is None

    with toolset.use_environment(env):
        assert toolset.environment is env

    assert toolset.environment is None


async def test_toolset_system_prompt():
    toolset = ExecutionToolset(LocalEnvironment('.'))
    prompt = toolset.system_prompt
    assert 'bash' in prompt
    assert 'read_file' in prompt
    assert 'write_file' in prompt


async def test_toolset_tool_name_conflict_hint():
    toolset = ExecutionToolset(LocalEnvironment('.'))
    assert 'PrefixedToolset' in toolset.tool_name_conflict_hint


# --- ExecutionToolset: lifecycle ---


async def test_toolset_lifecycle(tmp_path: Path):
    env = LocalEnvironment(tmp_path)
    toolset = ExecutionToolset(env)

    async with toolset:
        result = await env.execute('echo lifecycle')
        assert 'lifecycle' in result.output


# --- ExecutionToolset: image support ---


async def test_toolset_image_support_disabled(tmp_path: Path):
    env = LocalEnvironment(tmp_path)
    toolset = ExecutionToolset(env, image_support=False)
    ctx = build_run_context(None)
    manager = await ToolManager[None](toolset).for_run_step(ctx)

    async with env:
        await env.write_file('photo.png', b'\x89PNG\r\n\x1a\n')
        result = await manager.handle_call(ToolCallPart(tool_name='read_file', args={'path': 'photo.png'}))
        assert result == snapshot('[Image file: photo.png — image_support is disabled on this toolset]')


# --- LocalEnvironment: grep output modes ---


async def test_local_grep_files_with_matches(tmp_path: Path):
    async with LocalEnvironment(tmp_path) as env:
        await env.write_file('a.py', 'target = 1\nother = 2\n')
        await env.write_file('b.py', 'target = 3\ntarget = 4\n')
        await env.write_file('c.py', 'nothing here\n')

        result = await env.grep('target', output_mode='files_with_matches')
        lines = result.strip().splitlines()
        assert sorted(lines) == ['a.py', 'b.py']


async def test_local_grep_count(tmp_path: Path):
    async with LocalEnvironment(tmp_path) as env:
        await env.write_file('a.py', 'target = 1\nother = 2\n')
        await env.write_file('b.py', 'target = 3\ntarget = 4\n')
        await env.write_file('c.py', 'nothing here\n')

        result = await env.grep('target', output_mode='count')
        lines = sorted(result.strip().splitlines())
        assert lines == ['a.py:1', 'b.py:2']


async def test_local_grep_content_default(tmp_path: Path):
    """Default output_mode is 'content' with file:line:text format."""
    async with LocalEnvironment(tmp_path) as env:
        await env.write_file('test.py', 'hello\nworld\n')

        result = await env.grep('hello')
        assert result == snapshot('test.py:1:hello')


# --- LocalEnvironment: binary file detection ---


async def test_local_grep_skips_binary_files(tmp_path: Path):
    async with LocalEnvironment(tmp_path) as env:
        await env.write_file('text.py', 'findme = True\n')
        await env.write_file('binary.pyc', b'\x00\x01\x02findme\x03\x04')

        result = await env.grep('findme')
        assert 'text.py' in result
        assert 'binary.pyc' not in result


async def test_local_grep_binary_detection_first_8kb(tmp_path: Path):
    """Binary detection checks only the first 8KB."""
    async with LocalEnvironment(tmp_path) as env:
        # File with null byte after 8KB — should be treated as text
        content = 'findme\n' + ('x' * 8200) + '\x00'
        await env.write_file('mostly_text.txt', content)

        result = await env.grep('findme')
        assert 'mostly_text.txt' in result


# --- Toolset: grep output_mode ---


async def test_toolset_grep_files_with_matches(tmp_path: Path):
    env = LocalEnvironment(tmp_path)
    toolset = ExecutionToolset(env)
    ctx = build_run_context(None)
    manager = await ToolManager[None](toolset).for_run_step(ctx)

    async with env:
        await env.write_file('a.py', 'target = 1\n')
        await env.write_file('b.py', 'other = 2\n')

        result = await manager.handle_call(
            ToolCallPart(tool_name='grep', args={'pattern': 'target', 'output_mode': 'files_with_matches'})
        )
        assert result == snapshot('a.py')


async def test_toolset_grep_count(tmp_path: Path):
    env = LocalEnvironment(tmp_path)
    toolset = ExecutionToolset(env)
    ctx = build_run_context(None)
    manager = await ToolManager[None](toolset).for_run_step(ctx)

    async with env:
        await env.write_file('a.py', 'x = 1\nx = 2\nx = 3\n')

        result = await manager.handle_call(
            ToolCallPart(tool_name='grep', args={'pattern': 'x', 'output_mode': 'count'})
        )
        assert result == snapshot('a.py:3')


# --- MemoryEnvironment ---


async def test_memory_read_write():
    async with MemoryEnvironment() as env:
        await env.write_file('test.txt', 'hello world\n')
        content = await env.read_file('test.txt')
        assert content == snapshot("""\
     1\thello world
""")


async def test_memory_initial_files():
    env = MemoryEnvironment(files={'a.txt': 'alpha', 'b.txt': 'beta'})
    async with env:
        a = await env.read_file('a.txt')
        assert isinstance(a, str)
        assert 'alpha' in a
        b = await env.read_file('b.txt')
        assert isinstance(b, str)
        assert 'beta' in b


async def test_memory_read_nonexistent():
    async with MemoryEnvironment() as env:
        with pytest.raises(FileNotFoundError):
            await env.read_file('nope.txt')


async def test_memory_read_directory_error():
    env = MemoryEnvironment(files={'dir/file.txt': 'content'})
    async with env:
        with pytest.raises(FileNotFoundError, match='is a directory'):
            await env.read_file('dir')


async def test_memory_read_offset_limit():
    lines = '\n'.join(f'line {i}' for i in range(20))
    env = MemoryEnvironment(files={'long.txt': lines})
    async with env:
        content = await env.read_file('long.txt', offset=5, limit=3)
        assert isinstance(content, str)
        assert 'line 5' in content
        assert 'line 7' in content
        assert 'line 4' not in content
        assert 'line 8' not in content


async def test_memory_read_continuation_hint():
    lines = '\n'.join(f'line {i}' for i in range(20))
    env = MemoryEnvironment(files={'long.txt': lines})
    async with env:
        content = await env.read_file('long.txt', offset=0, limit=5)
        assert isinstance(content, str)
        assert '15 more lines' in content
        assert 'offset=5' in content


async def test_memory_read_offset_out_of_bounds():
    env = MemoryEnvironment(files={'short.txt': 'one\ntwo\n'})
    async with env:
        with pytest.raises(ValueError, match='Offset 100 exceeds'):
            await env.read_file('short.txt', offset=100)


async def test_memory_edit_file():
    env = MemoryEnvironment(files={'code.py': 'old_value = 1'})
    async with env:
        count = await env.edit_file('code.py', 'old_value', 'new_value')
        assert count == 1
        content = await env.read_file('code.py')
        assert isinstance(content, str)
        assert 'new_value' in content
        assert 'old_value' not in content


async def test_memory_edit_file_not_found():
    async with MemoryEnvironment() as env:
        with pytest.raises(FileNotFoundError):
            await env.edit_file('nope.txt', 'a', 'b')


async def test_memory_edit_string_not_found():
    env = MemoryEnvironment(files={'f.txt': 'hello'})
    async with env:
        with pytest.raises(ValueError, match='not found'):
            await env.edit_file('f.txt', 'missing', 'replacement')


async def test_memory_edit_ambiguous():
    env = MemoryEnvironment(files={'f.txt': 'dup dup dup'})
    async with env:
        with pytest.raises(ValueError, match='3 times'):
            await env.edit_file('f.txt', 'dup', 'x')


async def test_memory_edit_replace_all():
    env = MemoryEnvironment(files={'f.txt': 'aaa bbb aaa'})
    async with env:
        count = await env.edit_file('f.txt', 'aaa', 'xxx', replace_all=True)
        assert count == 2
        content = await env.read_file('f.txt')
        assert isinstance(content, str)
        assert 'xxx bbb xxx' in content


async def test_memory_ls():
    env = MemoryEnvironment(
        files={
            'a.txt': 'a',
            'b.txt': 'bb',
            'sub/c.txt': 'ccc',
        }
    )
    async with env:
        entries = await env.ls('.')
        names = {e.name for e in entries}
        assert names == {'a.txt', 'b.txt', 'sub'}

        dirs = [e for e in entries if e.is_dir]
        files = [e for e in entries if not e.is_dir]
        assert len(dirs) == 1
        assert dirs[0].name == 'sub'
        assert all(f.size is not None for f in files)


async def test_memory_ls_subdirectory():
    env = MemoryEnvironment(files={'sub/a.txt': 'a', 'sub/b.txt': 'b'})
    async with env:
        entries = await env.ls('sub')
        names = {e.name for e in entries}
        assert names == {'a.txt', 'b.txt'}


async def test_memory_ls_not_a_directory():
    async with MemoryEnvironment() as env:
        with pytest.raises(NotADirectoryError):
            await env.ls('nonexistent')


async def test_memory_glob():
    env = MemoryEnvironment(
        files={
            'src/main.py': '# main',
            'src/utils.py': '# utils',
            'src/data.json': '{}',
        }
    )
    async with env:
        matches = await env.glob('*.py', path='src')
        assert sorted(matches) == ['src/main.py', 'src/utils.py']


async def test_memory_glob_no_matches():
    env = MemoryEnvironment(files={'a.py': ''})
    async with env:
        matches = await env.glob('*.rs')
        assert matches == []


async def test_memory_grep_content():
    env = MemoryEnvironment(
        files={
            'a.py': 'def hello():\n    pass\n',
            'b.py': 'x = 1\n',
        }
    )
    async with env:
        result = await env.grep('hello')
        assert result == snapshot('a.py:1:def hello():')


async def test_memory_grep_files_with_matches():
    env = MemoryEnvironment(
        files={
            'a.py': 'target = 1\n',
            'b.py': 'target = 2\ntarget = 3\n',
            'c.py': 'nothing\n',
        }
    )
    async with env:
        result = await env.grep('target', output_mode='files_with_matches')
        lines = sorted(result.strip().splitlines())
        assert lines == ['a.py', 'b.py']


async def test_memory_grep_count():
    env = MemoryEnvironment(
        files={
            'a.py': 'x = 1\n',
            'b.py': 'x = 2\nx = 3\n',
        }
    )
    async with env:
        result = await env.grep('x', output_mode='count')
        lines = sorted(result.strip().splitlines())
        assert lines == ['a.py:1', 'b.py:2']


async def test_memory_grep_skips_binary():
    env = MemoryEnvironment(
        files={
            'text.py': 'findme = True\n',
            'binary.dat': b'\x00\x01findme\x02',
        }
    )
    async with env:
        result = await env.grep('findme')
        assert 'text.py' in result
        assert 'binary.dat' not in result


async def test_memory_grep_skips_hidden():
    env = MemoryEnvironment(
        files={
            'visible.py': 'target\n',
            '.hidden/secret.py': 'target\n',
        }
    )
    async with env:
        result = await env.grep('target')
        assert 'visible.py' in result
        assert '.hidden' not in result


async def test_memory_grep_with_glob_pattern():
    env = MemoryEnvironment(
        files={
            'code.py': 'target\n',
            'code.js': 'target\n',
        }
    )
    async with env:
        result = await env.grep('target', glob_pattern='*.py')
        assert 'code.py' in result
        assert 'code.js' not in result


async def test_memory_execute_with_handler():
    def handler(cmd: str) -> ExecuteResult:
        return ExecuteResult(output=f'ran: {cmd}\n', exit_code=0)

    async with MemoryEnvironment(command_handler=handler) as env:
        result = await env.execute('echo hello')
        assert result.output == 'ran: echo hello\n'
        assert result.exit_code == 0


async def test_memory_execute_no_handler():
    async with MemoryEnvironment() as env:
        with pytest.raises(RuntimeError, match='no command_handler'):
            await env.execute('echo hello')


async def test_memory_create_process_not_supported():
    async with MemoryEnvironment() as env:
        with pytest.raises(NotImplementedError):
            await env.create_process('echo hello')


async def test_memory_write_binary():
    async with MemoryEnvironment() as env:
        await env.write_file('data.bin', b'\x00\x01\x02')
        # read_file returns text; read_file_bytes returns raw bytes
        content = await env.read_file('data.bin')
        assert isinstance(content, str)
        raw = await env.read_file_bytes('data.bin')
        assert raw == b'\x00\x01\x02'


async def test_memory_read_file_bytes():
    png_data = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR'
    env = MemoryEnvironment(files={'img.png': png_data})
    async with env:
        result = await env.read_file_bytes('img.png')
        assert isinstance(result, bytes)
        assert result == png_data


# --- MemoryEnvironment with ExecutionToolset ---


async def test_memory_toolset_integration():
    """MemoryEnvironment works with ExecutionToolset for full agent testing."""
    env = MemoryEnvironment(files={'main.py': 'print("hello")\n'})
    toolset = ExecutionToolset(env, include_bash=False)
    ctx = build_run_context(None)
    manager = await ToolManager[None](toolset).for_run_step(ctx)

    async with env:
        # read_file
        result = await manager.handle_call(ToolCallPart(tool_name='read_file', args={'path': 'main.py'}))
        assert result == snapshot('     1\tprint("hello")\n')

        # write_file
        result = await manager.handle_call(
            ToolCallPart(tool_name='write_file', args={'path': 'new.py', 'content': 'x = 1'})
        )
        assert result == snapshot('File written: new.py')

        # glob
        result = await manager.handle_call(ToolCallPart(tool_name='glob', args={'pattern': '*.py'}))
        assert result == snapshot("""\
main.py
new.py\
""")

        # grep
        result = await manager.handle_call(ToolCallPart(tool_name='grep', args={'pattern': 'hello'}))
        assert result == snapshot('main.py:1:print("hello")')


# --- Docker/E2B instantiation tests ---


def test_docker_sandbox_instantiation():
    """DockerSandbox can be constructed without starting Docker."""
    try:
        from pydantic_ai.environments.docker import DockerSandbox
    except ImportError:
        pytest.skip('docker package not installed')

    # Verify construction succeeds with default and custom settings
    sandbox = DockerSandbox(image='python:3.12-slim')
    assert isinstance(sandbox, DockerSandbox)

    sandbox_with_opts = DockerSandbox(
        image='node:20-slim',
        packages=['express'],
        package_manager='npm',
        memory_limit='512m',
        cpu_limit=1.0,
        network_disabled=True,
    )
    assert isinstance(sandbox_with_opts, DockerSandbox)


def test_e2b_sandbox_instantiation():
    """E2BSandbox can be constructed without connecting to E2B."""
    try:
        from pydantic_ai.environments.e2b import E2BSandbox
    except ImportError:
        pytest.skip('e2b-code-interpreter package not installed')

    sandbox = E2BSandbox(template='base', api_key='test-key', timeout=60)
    assert isinstance(sandbox, E2BSandbox)


# --- Agent-level integration test ---


async def test_agent_with_execution_toolset():
    """Agent with ExecutionToolset runs end-to-end using TestModel and MemoryEnvironment."""
    from pydantic_ai import Agent

    env = MemoryEnvironment(
        files={'data.txt': 'hello world\n'},
        command_handler=lambda cmd: ExecuteResult(output=f'executed: {cmd}\n', exit_code=0),
    )
    toolset = ExecutionToolset(env)

    agent = Agent('test', toolsets=[toolset])

    async with env:
        result = await agent.run('Read the file data.txt')
        # The TestModel will call tools and we verify it completes without error
        assert result.output is not None
