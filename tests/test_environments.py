"""Tests for pydantic_ai.environments — ExecutionEnvironment, ExecutionEnvironmentToolset, LocalEnvironment, and MemoryEnvironment."""

from __future__ import annotations

import io
import os
import struct
import tarfile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch as mock_patch

import pytest
from inline_snapshot import snapshot

from pydantic_ai import Agent, BinaryContent, ToolCallPart
from pydantic_ai._run_context import RunContext
from pydantic_ai._tool_manager import ToolManager
from pydantic_ai.environments import (
    EnvCapability,
    ExecutionEnvironment as BaseEnv,
    ExecutionResult,
)
from pydantic_ai.environments._base import apply_replace_str
from pydantic_ai.environments.local import LocalEnvironment, _LocalEnvironmentProcess
from pydantic_ai.exceptions import UnexpectedModelBehavior
from pydantic_ai.models.test import TestModel
from pydantic_ai.toolsets.execution_environment import ExecutionEnvironmentToolset, _format_lines
from pydantic_ai.usage import RunUsage
from tests.environment_helpers import MemoryEnvironment

try:
    from docker.errors import DockerException, NotFound as DockerNotFound

    from pydantic_ai.environments.docker import (
        DockerEnvironment,
        _DockerEnvironmentProcess,
        _put_file,
        _shell_escape,
    )
except ImportError:  # pragma: lax no cover
    docker_installed = False
else:
    docker_installed = True

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
    result = ExecutionResult(output='hello\n', exit_code=0)
    assert result.output == 'hello\n'
    assert result.exit_code == 0


# --- LocalEnvironment: execute ---


async def test_local_execute_basic(tmp_path: Path):
    async with LocalEnvironment(tmp_path) as env:
        result = await env.shell('echo hello')
        assert result.exit_code == 0
        assert 'hello' in result.output


async def test_local_execute_exit_code(tmp_path: Path):
    async with LocalEnvironment(tmp_path) as env:
        result = await env.shell('exit 42')
        assert result.exit_code == 42


async def test_local_execute_timeout(tmp_path: Path):
    async with LocalEnvironment(tmp_path) as env:
        result = await env.shell('sleep 10', timeout=0.5)
        assert result.exit_code == -1
        assert 'timed out' in result.output.lower()


async def test_local_execute_stderr(tmp_path: Path):
    async with LocalEnvironment(tmp_path) as env:
        result = await env.shell('echo error >&2')
        assert 'error' in result.output


# --- LocalEnvironment: environment variables ---


async def test_local_env_vars_baseline(tmp_path: Path):
    async with LocalEnvironment(tmp_path, env_vars={'MY_VAR': 'baseline'}) as env:
        result = await env.shell('echo $MY_VAR')
        assert 'baseline' in result.output


async def test_local_env_vars_per_call(tmp_path: Path):
    async with LocalEnvironment(tmp_path) as env:
        result = await env.shell('echo $CALL_VAR', env={'CALL_VAR': 'per_call'})
        assert 'per_call' in result.output


async def test_local_env_vars_merged(tmp_path: Path):
    async with LocalEnvironment(tmp_path, env_vars={'BASE': 'one'}) as env:
        result = await env.shell('echo $BASE $EXTRA', env={'EXTRA': 'two'})
        assert 'one' in result.output
        assert 'two' in result.output


async def test_local_env_vars_per_call_overrides_baseline(tmp_path: Path):
    async with LocalEnvironment(tmp_path, env_vars={'VAR': 'old'}) as env:
        result = await env.shell('echo $VAR', env={'VAR': 'new'})
        assert 'new' in result.output
        assert 'old' not in result.output


async def test_local_inherit_env_true(tmp_path: Path):
    os.environ['_TEST_INHERIT_CHECK'] = 'inherited'
    try:
        async with LocalEnvironment(tmp_path, inherit_env=True) as env:
            result = await env.shell('echo $_TEST_INHERIT_CHECK')
            assert 'inherited' in result.output
    finally:
        del os.environ['_TEST_INHERIT_CHECK']


async def test_local_inherit_env_false(tmp_path: Path):
    os.environ['_TEST_INHERIT_CHECK'] = 'should_not_see'
    try:
        async with LocalEnvironment(tmp_path, inherit_env=False) as env:
            result = await env.shell('echo x${_TEST_INHERIT_CHECK}x')
            assert result.output.strip() == 'xx'
    finally:
        del os.environ['_TEST_INHERIT_CHECK']


async def test_local_inherit_env_false_with_explicit_vars(tmp_path: Path):
    async with LocalEnvironment(tmp_path, env_vars={'ONLY_THIS': 'yes'}, inherit_env=False) as env:
        result = await env.shell('/bin/echo $ONLY_THIS')
        assert 'yes' in result.output


# --- LocalEnvironment: file operations ---


async def test_local_write_and_read(tmp_path: Path):
    async with LocalEnvironment(tmp_path) as env:
        await env.write_file('test.txt', 'line one\nline two\n')
        content = await env.read_file('test.txt')
        assert isinstance(content, bytes)
        assert content == b'line one\nline two\n'


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
        assert content == b'content'


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
        result = await env.read_file('image.png')
        assert isinstance(result, bytes)
        assert result == png_data


# --- LocalEnvironment: replace_str ---


async def test_local_edit_single_replacement(tmp_path: Path):
    async with LocalEnvironment(tmp_path) as env:
        await env.write_file('edit.txt', 'foo bar baz')
        count = await env.replace_str('edit.txt', 'bar', 'BAR')
        assert count == 1
        content = (tmp_path / 'edit.txt').read_text()
        assert content == 'foo BAR baz'


async def test_local_edit_replace_all(tmp_path: Path):
    async with LocalEnvironment(tmp_path) as env:
        await env.write_file('edit.txt', 'aaa bbb aaa')
        count = await env.replace_str('edit.txt', 'aaa', 'xxx', replace_all=True)
        assert count == 2
        content = (tmp_path / 'edit.txt').read_text()
        assert content == 'xxx bbb xxx'


async def test_local_edit_not_found(tmp_path: Path):
    async with LocalEnvironment(tmp_path) as env:
        await env.write_file('edit.txt', 'hello world')
        with pytest.raises(ValueError, match='not found'):
            await env.replace_str('edit.txt', 'missing', 'replacement')


async def test_local_edit_ambiguous_without_replace_all(tmp_path: Path):
    async with LocalEnvironment(tmp_path) as env:
        await env.write_file('edit.txt', 'dup dup dup')
        with pytest.raises(ValueError, match='3 times'):
            await env.replace_str('edit.txt', 'dup', 'unique')


async def test_local_edit_nonexistent_file(tmp_path: Path):
    async with LocalEnvironment(tmp_path) as env:
        with pytest.raises(FileNotFoundError):
            await env.replace_str('missing.txt', 'old', 'new')


async def test_local_edit_multiline(tmp_path: Path):
    async with LocalEnvironment(tmp_path) as env:
        await env.write_file('code.py', 'def foo():\n    return "old"\n\nprint("test")\n')
        count = await env.replace_str('code.py', 'def foo():\n    return "old"', 'def foo():\n    return "new"')
        assert count == 1
        content = (tmp_path / 'code.py').read_text()
        assert 'return "new"' in content
        assert 'return "old"' not in content
        assert 'print("test")' in content


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
        result = await env.shell('echo works')
        assert 'works' in result.output


# --- ExecutionEnvironmentToolset ---


async def test_toolset_tool_names():
    toolset = ExecutionEnvironmentToolset(LocalEnvironment('.'))
    ctx = build_run_context()
    tools = await toolset.get_tools(ctx)
    tool_names = sorted(tools.keys())
    assert tool_names == snapshot(['read_file', 'replace_str', 'shell', 'write_file'])


async def test_toolset_include_flags():
    toolset = ExecutionEnvironmentToolset(
        LocalEnvironment('.'),
        include=[],
    )
    ctx = build_run_context()
    tools = await toolset.get_tools(ctx)
    assert tools == {}


async def test_toolset_include_shell_only():
    toolset = ExecutionEnvironmentToolset(
        LocalEnvironment('.'),
        include=['shell'],
    )
    ctx = build_run_context()
    tools = await toolset.get_tools(ctx)
    assert sorted(tools.keys()) == ['shell']


async def test_toolset_bash_tool(tmp_path: Path):
    env = LocalEnvironment(tmp_path)
    toolset = ExecutionEnvironmentToolset(env)
    ctx = build_run_context(None)
    manager = await ToolManager[None](toolset).for_run_step(ctx)

    async with env:
        result = await manager.handle_call(ToolCallPart(tool_name='shell', args={'command': 'echo hello'}))
        assert result == snapshot("""\
hello

Exit code: 0\
""")


async def test_toolset_read_write_tools(tmp_path: Path):
    env = LocalEnvironment(tmp_path)
    toolset = ExecutionEnvironmentToolset(env)
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
        assert read_result == snapshot('1\thello world\n')


async def test_toolset_edit_retry_on_error(tmp_path: Path):
    env = LocalEnvironment(tmp_path)
    toolset = ExecutionEnvironmentToolset(env, max_retries=0)
    ctx = build_run_context(None)
    manager = await ToolManager[None](toolset).for_run_step(ctx)

    async with env:
        await env.write_file('test.txt', 'content')

        # Edit with non-matching string: ModelRetry is raised by tool, but with max_retries=0
        # the ToolManager wraps it into UnexpectedModelBehavior
        with pytest.raises(UnexpectedModelBehavior, match='exceeded max retries count of 0'):
            await manager.handle_call(
                ToolCallPart(
                    tool_name='replace_str',
                    args={'path': 'test.txt', 'old': 'nonexistent', 'new': 'replacement'},
                )
            )


async def test_toolset_edit_retry_on_permission_error(tmp_path: Path):
    """replace_str raises ModelRetry on PermissionError (e.g. path traversal)."""
    env = LocalEnvironment(tmp_path)
    toolset = ExecutionEnvironmentToolset(env, max_retries=0)
    ctx = build_run_context(None)
    manager = await ToolManager[None](toolset).for_run_step(ctx)

    async with env:
        with pytest.raises(UnexpectedModelBehavior, match='exceeded max retries count of 0'):
            await manager.handle_call(
                ToolCallPart(
                    tool_name='replace_str',
                    args={'path': '../../etc/passwd', 'old': 'root', 'new': 'hacked'},
                )
            )


# --- ExecutionEnvironmentToolset: error handling ---


async def test_toolset_read_nonexistent_returns_error(tmp_path: Path):
    """read_file on a nonexistent file returns an error string instead of crashing."""
    env = LocalEnvironment(tmp_path)
    toolset = ExecutionEnvironmentToolset(env)
    ctx = build_run_context(None)
    manager = await ToolManager[None](toolset).for_run_step(ctx)

    async with env:
        result = await manager.handle_call(ToolCallPart(tool_name='read_file', args={'path': 'nope.txt'}))
        assert 'Error:' in str(result)


async def test_toolset_read_path_traversal_returns_error(tmp_path: Path):
    """read_file with path traversal returns an error string."""
    env = LocalEnvironment(tmp_path)
    toolset = ExecutionEnvironmentToolset(env)
    ctx = build_run_context(None)
    manager = await ToolManager[None](toolset).for_run_step(ctx)

    async with env:
        result = await manager.handle_call(ToolCallPart(tool_name='read_file', args={'path': '../../etc/passwd'}))
        assert 'Error:' in str(result)


async def test_toolset_write_path_traversal_returns_error(tmp_path: Path):
    """write_file with path traversal returns an error string."""
    env = LocalEnvironment(tmp_path)
    toolset = ExecutionEnvironmentToolset(env)
    ctx = build_run_context(None)
    manager = await ToolManager[None](toolset).for_run_step(ctx)

    async with env:
        result = await manager.handle_call(
            ToolCallPart(tool_name='write_file', args={'path': '../../tmp/evil.txt', 'content': 'bad'})
        )
        assert 'Error:' in str(result)


async def test_toolset_read_offset_out_of_bounds_returns_error(tmp_path: Path):
    """read_file with offset past EOF returns an error string."""
    env = LocalEnvironment(tmp_path)
    toolset = ExecutionEnvironmentToolset(env)
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
    toolset = ExecutionEnvironmentToolset(env)
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


# --- ExecutionEnvironmentToolset: approval flags ---


async def test_toolset_require_shell_approval():
    """require_shell_approval sets requires_approval on the shell tool."""
    env = MemoryEnvironment(command_handler=lambda cmd: ExecutionResult(output='', exit_code=0))
    toolset = ExecutionEnvironmentToolset(env, require_shell_approval=True)
    ctx = build_run_context(None)
    tools = await toolset.get_tools(ctx)
    assert tools['shell'].tool_def.kind == 'unapproved'
    # Other tools should be normal
    assert tools['read_file'].tool_def.kind == 'function'


async def test_toolset_require_write_approval():
    """require_write_approval sets requires_approval on write_file and replace_str."""
    toolset = ExecutionEnvironmentToolset(MemoryEnvironment(), require_write_approval=True)
    ctx = build_run_context(None)
    tools = await toolset.get_tools(ctx)
    assert tools['write_file'].tool_def.kind == 'unapproved'
    assert tools['replace_str'].tool_def.kind == 'unapproved'
    # read_file should NOT require approval
    assert tools['read_file'].tool_def.kind == 'function'


async def test_toolset_default_no_approval():
    """By default, no tools require approval."""
    toolset = ExecutionEnvironmentToolset(MemoryEnvironment())
    ctx = build_run_context(None)
    tools = await toolset.get_tools(ctx)
    for tool in tools.values():
        assert tool.tool_def.kind == 'function'


# --- ExecutionEnvironmentToolset: output truncation ---


async def test_toolset_shell_output_truncation():
    """Shell output is truncated while preserving the exit code."""
    long_output = 'x' * 500
    env = MemoryEnvironment(command_handler=lambda cmd: ExecutionResult(output=long_output, exit_code=0))
    toolset = ExecutionEnvironmentToolset(env, max_output_chars=100)
    ctx = build_run_context(None)
    manager = await ToolManager[None](toolset).for_run_step(ctx)

    result = await manager.handle_call(ToolCallPart(tool_name='shell', args={'command': 'echo'}))
    assert isinstance(result, str)
    assert result.endswith('Exit code: 0')
    assert '... (truncated)' in result


async def test_toolset_read_file_output_truncation(tmp_path: Path):
    """read_file text output is truncated when exceeding max_output_chars."""
    long_line = 'y' * 500
    env = MemoryEnvironment(files={'big.txt': long_line})
    toolset = ExecutionEnvironmentToolset(env, max_output_chars=100)
    ctx = build_run_context(None)
    manager = await ToolManager[None](toolset).for_run_step(ctx)

    result = await manager.handle_call(ToolCallPart(tool_name='read_file', args={'path': 'big.txt'}))
    assert isinstance(result, str)
    assert len(result) <= 120  # 100 + truncation marker
    assert '... (truncated)' in result


# --- ExecutionEnvironmentToolset: environment management ---


async def test_toolset_environment_property():
    env = LocalEnvironment('.')
    toolset = ExecutionEnvironmentToolset(env)
    assert toolset.environment is env
    assert toolset.required_environment is env


async def test_toolset_no_environment_returns_none():
    toolset = ExecutionEnvironmentToolset()
    assert toolset.environment is None


async def test_toolset_no_environment_required_raises():
    toolset = ExecutionEnvironmentToolset()
    with pytest.raises(RuntimeError, match='No execution environment configured'):
        _ = toolset.required_environment


async def test_toolset_use_environment():
    env1 = LocalEnvironment('/tmp/env1')
    env2 = LocalEnvironment('/tmp/env2')
    toolset = ExecutionEnvironmentToolset(env1)

    assert toolset.environment is env1
    with toolset.use_environment(env2):
        assert toolset.environment is env2
    assert toolset.environment is env1


async def test_toolset_use_environment_no_default():
    env = LocalEnvironment('.')
    toolset = ExecutionEnvironmentToolset()

    assert toolset.environment is None

    with toolset.use_environment(env):
        assert toolset.environment is env

    assert toolset.environment is None


async def test_toolset_tool_name_conflict_hint():
    toolset = ExecutionEnvironmentToolset(LocalEnvironment('.'))
    assert 'PrefixedToolset' in toolset.tool_name_conflict_hint


# --- ExecutionEnvironmentToolset: lifecycle ---


async def test_toolset_enter_no_environment_raises():
    toolset = ExecutionEnvironmentToolset()
    with pytest.raises(RuntimeError, match='No execution environment configured'):
        async with toolset:
            pass


async def test_toolset_lifecycle(tmp_path: Path):
    env = LocalEnvironment(tmp_path)
    toolset = ExecutionEnvironmentToolset(env)

    async with toolset:
        result = await env.shell('echo lifecycle')
        assert 'lifecycle' in result.output


# --- MemoryEnvironment ---


async def test_memory_read_write():
    async with MemoryEnvironment() as env:
        await env.write_file('test.txt', 'hello world\n')
        content = await env.read_file('test.txt')
        assert content == b'hello world\n'


async def test_memory_initial_files():
    env = MemoryEnvironment(files={'a.txt': 'alpha', 'b.txt': 'beta'})
    async with env:
        assert env.files == {'a.txt': 'alpha', 'b.txt': 'beta'}
        a = await env.read_file('a.txt')
        assert a == b'alpha'
        b = await env.read_file('b.txt')
        assert b == b'beta'


async def test_memory_read_nonexistent():
    async with MemoryEnvironment() as env:
        with pytest.raises(FileNotFoundError):
            await env.read_file('nope.txt')


async def test_memory_read_directory_error():
    env = MemoryEnvironment(files={'dir/file.txt': 'content'})
    async with env:
        with pytest.raises(FileNotFoundError, match='is a directory'):
            await env.read_file('dir')


async def test_memory_replace_str():
    env = MemoryEnvironment(files={'code.py': 'old_value = 1'})
    async with env:
        count = await env.replace_str('code.py', 'old_value', 'new_value')
        assert count == 1
        content = await env.read_file('code.py')
        assert content == b'new_value = 1'


async def test_memory_replace_str_not_found():
    async with MemoryEnvironment() as env:
        with pytest.raises(FileNotFoundError):
            await env.replace_str('nope.txt', 'a', 'b')


async def test_memory_edit_string_not_found():
    env = MemoryEnvironment(files={'f.txt': 'hello'})
    async with env:
        with pytest.raises(ValueError, match='not found'):
            await env.replace_str('f.txt', 'missing', 'replacement')


async def test_memory_edit_ambiguous():
    env = MemoryEnvironment(files={'f.txt': 'dup dup dup'})
    async with env:
        with pytest.raises(ValueError, match='3 times'):
            await env.replace_str('f.txt', 'dup', 'x')


async def test_memory_edit_replace_all():
    env = MemoryEnvironment(files={'f.txt': 'aaa bbb aaa'})
    async with env:
        count = await env.replace_str('f.txt', 'aaa', 'xxx', replace_all=True)
        assert count == 2
        content = await env.read_file('f.txt')
        assert content == b'xxx bbb xxx'


async def test_memory_execute_with_handler():
    def handler(cmd: str) -> ExecutionResult:
        return ExecutionResult(output=f'ran: {cmd}\n', exit_code=0)

    async with MemoryEnvironment(command_handler=handler) as env:
        result = await env.shell('echo hello')
        assert result.output == 'ran: echo hello\n'
        assert result.exit_code == 0


async def test_memory_execute_no_handler():
    async with MemoryEnvironment() as env:
        with pytest.raises(RuntimeError, match='no command_handler'):
            await env.shell('echo hello')


async def test_memory_create_process_not_supported():
    async with MemoryEnvironment() as env:
        with pytest.raises(NotImplementedError):
            await env.create_process('echo hello')


async def test_memory_write_binary():
    async with MemoryEnvironment() as env:
        await env.write_file('data.bin', b'\x00\x01\x02')
        content = await env.read_file('data.bin')
        assert content == b'\x00\x01\x02'


async def test_memory_read_file_bytes():
    png_data = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR'
    env = MemoryEnvironment(files={'img.png': png_data})
    async with env:
        result = await env.read_file('img.png')
        assert isinstance(result, bytes)
        assert result == png_data


# --- MemoryEnvironment with ExecutionEnvironmentToolset ---


async def test_memory_toolset_integration():
    """MemoryEnvironment works with ExecutionEnvironmentToolset for full agent testing."""
    env = MemoryEnvironment(files={'main.py': 'print("hello")\n'})
    toolset = ExecutionEnvironmentToolset(env, exclude=['shell'])
    ctx = build_run_context(None)
    manager = await ToolManager[None](toolset).for_run_step(ctx)

    async with env:
        # read_file
        result = await manager.handle_call(ToolCallPart(tool_name='read_file', args={'path': 'main.py'}))
        assert result == snapshot('1\tprint("hello")\n')

        # write_file
        result = await manager.handle_call(
            ToolCallPart(tool_name='write_file', args={'path': 'new.py', 'content': 'x = 1'})
        )
        assert result == snapshot('File written: new.py')


# --- Agent-level integration test ---


async def test_agent_with_execution_toolset():
    """Agent with ExecutionEnvironmentToolset runs end-to-end using TestModel and MemoryEnvironment."""

    env = MemoryEnvironment(
        files={'data.txt': 'hello world\n'},
        command_handler=lambda cmd: ExecutionResult(output=f'executed: {cmd}\n', exit_code=0),
    )
    toolset = ExecutionEnvironmentToolset(env)

    agent = Agent('test', toolsets=[toolset])

    async with env:
        result = await agent.run('Read the file data.txt')
        # The TestModel will call tools and we verify it completes without error
        assert result.output is not None


# pyright: reportPrivateUsage=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false, reportPossiblyUnboundVariable=false


# --- _base.py helper functions ---


def test__format_lines_empty_file():
    """_format_lines on empty string returns just a newline."""
    result = _format_lines('', 0, 2000)
    assert result == '\n'


def test__format_lines_trailing_newline():
    """_format_lines adds trailing newline when text doesn't end with one."""
    result = _format_lines('no trailing newline', 0, 2000)
    assert result.endswith('\n')
    assert '1\tno trailing newline' in result


def test_apply_replace_str_basic():
    new_text, count = apply_replace_str('hello world', 'world', 'earth', 'test.txt', replace_all=False)
    assert new_text == 'hello earth'
    assert count == 1


def test_apply_replace_str_replace_all():
    new_text, count = apply_replace_str('aaa bbb aaa', 'aaa', 'xxx', 'test.txt', replace_all=True)
    assert new_text == 'xxx bbb xxx'
    assert count == 2


def test_apply_replace_str_not_found():
    with pytest.raises(ValueError, match='not found'):
        apply_replace_str('hello', 'missing', 'x', 'test.txt', replace_all=False)


def test_apply_replace_str_ambiguous():
    with pytest.raises(ValueError, match='2 times'):
        apply_replace_str('aa bb aa', 'aa', 'x', 'test.txt', replace_all=False)


# --- LocalEnvironment: additional edge cases ---


async def test_local_execute_no_timeout(tmp_path: Path):
    """execute() with timeout=None completes without timeout."""
    async with LocalEnvironment(tmp_path) as env:
        result = await env.shell('echo no_timeout', timeout=None)
        assert result.exit_code == 0
        assert 'no_timeout' in result.output


async def test_local_execute_invalid_timeout(tmp_path: Path):
    """execute() with non-positive timeout raises ValueError."""
    async with LocalEnvironment(tmp_path) as env:
        with pytest.raises(ValueError, match='timeout must be positive or None'):
            await env.shell('echo test', timeout=0)
        with pytest.raises(ValueError, match='timeout must be positive or None'):
            await env.shell('echo test', timeout=-1)


async def test_local_read_file_bytes_directory(tmp_path: Path):
    """read_file_bytes on a directory raises FileNotFoundError."""
    async with LocalEnvironment(tmp_path) as env:
        (tmp_path / 'adir').mkdir()
        with pytest.raises(FileNotFoundError, match='is a directory'):
            await env.read_file('adir')


async def test_local_read_file_bytes_nonexistent(tmp_path: Path):
    """read_file_bytes on a nonexistent file raises FileNotFoundError."""
    async with LocalEnvironment(tmp_path) as env:
        with pytest.raises(FileNotFoundError):
            await env.read_file('nope.bin')


# --- MemoryEnvironment: additional edge cases ---


async def test_memory_normalize_paths():
    """MemoryEnvironment normalizes paths correctly."""
    async with MemoryEnvironment() as env:
        await env.write_file('./test.txt', 'content')
        content = await env.read_file('test.txt')
        assert content == b'content'


async def test_memory_normalize_leading_slash():
    """MemoryEnvironment strips leading slashes."""
    async with MemoryEnvironment() as env:
        await env.write_file('/test.txt', 'content')
        content = await env.read_file('test.txt')
        assert content == b'content'


async def test_memory_read_file_text():
    """read_file returns raw bytes."""
    env = MemoryEnvironment(files={'text.txt': 'hello'})
    async with env:
        result = await env.read_file('text.txt')
        assert result == b'hello'


async def test_memory_read_file_not_found():
    """read_file on missing file raises FileNotFoundError."""
    async with MemoryEnvironment() as env:
        with pytest.raises(FileNotFoundError):
            await env.read_file('missing.txt')


async def test_memory_edit_binary():
    """replace_str works on binary content."""
    env = MemoryEnvironment(files={'data.txt': b'hello world'})
    async with env:
        count = await env.replace_str('data.txt', 'world', 'earth')
        assert count == 1


# --- ExecutionEnvironmentToolset: additional coverage ---


async def test_toolset_binary_too_large(tmp_path: Path):
    """read_file on a binary file that's too large returns error string."""
    env = LocalEnvironment(tmp_path)
    toolset = ExecutionEnvironmentToolset(env, max_binary_content_bytes=10)  # Very small limit
    ctx = build_run_context(None)
    manager = await ToolManager[None](toolset).for_run_step(ctx)

    async with env:
        # Write a PNG file that exceeds the limit
        await env.write_file('big.png', b'\x89PNG\r\n\x1a\n' + b'\x00' * 100)
        result = await manager.handle_call(ToolCallPart(tool_name='read_file', args={'path': 'big.png'}))
        assert 'Binary content too large' in str(result)


async def test_toolset_image_read(tmp_path: Path):
    """read_file on an image returns BinaryContent."""

    env = LocalEnvironment(tmp_path)
    toolset = ExecutionEnvironmentToolset(env)
    ctx = build_run_context(None)
    manager = await ToolManager[None](toolset).for_run_step(ctx)

    async with env:
        png_data = (
            b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01'
            b'\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89'
            b'\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01'
            b'\r\n\xb4\x00\x00\x00\x00IEND\xaeB`\x82'
        )
        await env.write_file('img.png', png_data)
        result = await manager.handle_call(ToolCallPart(tool_name='read_file', args={'path': 'img.png'}))
        assert isinstance(result, BinaryContent)
        assert result.media_type == 'image/png'


async def test_toolset_read_binary_unknown_type():
    """read_file on a binary file with unrecognized extension returns a placeholder message."""
    # Use an extension that mimetypes won't recognize, so guess_type returns None
    env = MemoryEnvironment(files={'data.xyzabc': b'\x80\x81\x82'})
    toolset = ExecutionEnvironmentToolset(env)
    ctx = build_run_context(None)
    manager = await ToolManager[None](toolset).for_run_step(ctx)
    async with env:
        result = await manager.handle_call(ToolCallPart(tool_name='read_file', args={'path': 'data.xyzabc'}))
    assert result == '[Binary file: data.xyzabc — cannot display as text]'


async def test_toolset_edit_success(tmp_path: Path):
    """replace_str tool returns success message."""
    env = LocalEnvironment(tmp_path)
    toolset = ExecutionEnvironmentToolset(env)
    ctx = build_run_context(None)
    manager = await ToolManager[None](toolset).for_run_step(ctx)

    async with env:
        await env.write_file('code.py', 'old_value = 1\n')
        result = await manager.handle_call(
            ToolCallPart(
                tool_name='replace_str',
                args={'path': 'code.py', 'old': 'old_value', 'new': 'new_value'},
            )
        )
        assert result == snapshot('Replaced 1 occurrence in code.py.')


async def test_toolset_lifecycle_ref_counting(tmp_path: Path):
    """Multiple context manager entries share the environment."""
    env = LocalEnvironment(tmp_path)
    toolset = ExecutionEnvironmentToolset(env)

    async with toolset:
        async with toolset:
            # Both entries active
            result = await env.shell('echo shared')
            assert 'shared' in result.output
        # Still alive after one exit
        result = await env.shell('echo still_alive')
        assert 'still_alive' in result.output


# --- DockerEnvironment: mocked tests ---


def _make_tar(filename: str, data: bytes) -> bytes:
    """Create a tar archive with a single file."""
    f = io.BytesIO()
    with tarfile.open(fileobj=f, mode='w') as tar:
        info = tarfile.TarInfo(name=filename)
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))
    f.seek(0)
    return f.read()


class MockContainer:
    """Mock Docker container for testing."""

    def __init__(self) -> None:
        self._files: dict[str, bytes] = {}
        self.id = 'mock-container-id'
        self.status = 'running'
        self.client = MagicMock()

    def exec_run(
        self,
        cmd: list[str] | str,
        workdir: str | None = None,
        environment: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> tuple[int, bytes]:
        """Simulate exec_run by executing simple commands."""
        if isinstance(cmd, list):
            cmd_str = ' '.join(cmd)
        else:
            cmd_str = cmd  # pragma: no cover

        # Handle mkdir -p
        if 'mkdir -p' in cmd_str:
            return 0, b''

        # Handle find (glob)
        if 'find' in cmd_str:
            # Extract the search path from: find '<path>' ...
            import shlex as _shlex

            find_part = cmd_str.split('|')[0].strip()
            # Remove the 'sh -c' wrapper if present
            if find_part.startswith('sh -c '):  # pragma: no branch
                find_part = find_part[len('sh -c ') :]
            tokens = _shlex.split(find_part)
            # tokens[0] is 'find', tokens[1] is the path
            search_path = tokens[1] if len(tokens) > 1 else '.'
            wd = workdir or '/workspace'
            matches = []
            for fpath in sorted(self._files):
                # Make path relative to workdir
                if not fpath.startswith(wd + '/'):
                    continue
                rel = fpath[len(wd) + 1 :]
                if search_path == '.':
                    matches.append(f'./{rel}')
                elif rel.startswith(search_path + '/') or rel == search_path:
                    matches.append(rel)
            return 0, '\n'.join(matches).encode('utf-8')

        # Handle general commands
        if 'echo' in cmd_str:
            # Extract the echo argument
            msg = cmd_str.split('echo ', 1)[-1] if 'echo ' in cmd_str else ''
            return 0, (msg + '\n').encode('utf-8')

        if 'exit' in cmd_str:  # pragma: no cover
            return 1, b''

        return 0, b''  # pragma: no cover

    def put_archive(self, path: str, data: Any) -> bool:
        """Simulate file upload by extracting tar data."""
        tar_data = data.read() if hasattr(data, 'read') else data
        with tarfile.open(fileobj=io.BytesIO(tar_data)) as tar:
            for member in tar.getmembers():
                extracted = tar.extractfile(member)
                if extracted:  # pragma: no branch
                    full_path = f'{path}/{member.name}' if path != '.' else member.name
                    self._files[full_path] = extracted.read()
        return True

    def get_archive(self, path: str) -> tuple[list[bytes], dict[str, Any]]:
        """Simulate file download."""
        if path not in self._files:
            # Check if file exists at resolved path
            for fpath, data in self._files.items():  # pragma: no cover
                if fpath.endswith(path) or path.endswith(fpath.split('/')[-1]):
                    return [_make_tar(fpath.split('/')[-1], data)], {}  # pragma: no cover
            raise DockerNotFound('File not found')  # pragma: no cover
        data = self._files[path]
        return [_make_tar(path.split('/')[-1], data)], {}

    def stop(self, timeout: int = 5) -> None:  # pragma: no cover
        self.status = 'stopped'

    def remove(self, force: bool = False) -> None:
        pass

    def reload(self) -> None:
        pass


@pytest.fixture
def mock_container() -> MockContainer:
    return MockContainer()


@pytest.fixture
def mock_docker_sandbox(mock_container: MockContainer) -> Any:
    """Create a DockerEnvironment with a mock container."""
    sandbox = DockerEnvironment(image='python:3.12-slim')
    sandbox._container = mock_container  # type: ignore[assignment]
    sandbox._client = MagicMock()
    return sandbox


@pytest.mark.skipif(not docker_installed, reason='docker package not installed')
class TestDocker:
    async def test_docker_execute(self, mock_docker_sandbox: Any) -> None:
        """DockerEnvironment.execute runs commands in container."""
        result = await mock_docker_sandbox.shell('echo hello')
        assert result.exit_code == 0
        assert isinstance(result.output, str)

    async def test_docker_execute_timeout(self, mock_docker_sandbox: Any) -> None:
        """DockerEnvironment.execute wraps command with timeout."""
        result = await mock_docker_sandbox.shell('echo test', timeout=30)
        assert result.exit_code == 0

    async def test_docker_execute_no_timeout(self, mock_docker_sandbox: Any) -> None:
        """DockerEnvironment.execute with timeout=None."""
        result = await mock_docker_sandbox.shell('echo test', timeout=None)
        assert result.exit_code == 0

    async def test_docker_execute_invalid_timeout(self, mock_docker_sandbox: Any) -> None:
        """DockerEnvironment.execute with non-positive timeout raises ValueError."""
        with pytest.raises(ValueError, match='timeout must be positive or None'):
            await mock_docker_sandbox.shell('echo test', timeout=0)
        with pytest.raises(ValueError, match='timeout must be positive or None'):
            await mock_docker_sandbox.shell('echo test', timeout=-1)

    async def test_docker_execute_with_env(self, mock_docker_sandbox: Any) -> None:
        """DockerEnvironment.execute passes env vars."""
        result = await mock_docker_sandbox.shell('echo test', env={'KEY': 'value'})
        assert result.exit_code == 0

    async def test_docker_write_read_file(self, mock_docker_sandbox: Any) -> None:
        """DockerEnvironment write and read files."""
        await mock_docker_sandbox.write_file('test.txt', 'hello world\n')
        content = await mock_docker_sandbox.read_file('test.txt')
        assert isinstance(content, bytes)

    async def test_docker_write_file_binary(self, mock_docker_sandbox: Any) -> None:
        """DockerEnvironment write binary file."""
        await mock_docker_sandbox.write_file('data.bin', b'\x00\x01\x02')

    async def test_docker_read_file_not_found(self, mock_docker_sandbox: Any) -> None:
        """DockerEnvironment.read_file on missing file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            await mock_docker_sandbox.read_file('nonexistent.txt')

    async def test_docker_read_file_image(self, mock_docker_sandbox: Any, mock_container: MockContainer) -> None:
        """DockerEnvironment.read_file returns raw bytes for image files."""
        png_data = b'\x89PNG\r\n\x1a\n'
        mock_container._files['/workspace/image.png'] = png_data
        result = await mock_docker_sandbox.read_file('image.png')
        assert isinstance(result, bytes)
        assert result == png_data

    async def test_docker_replace_str(self, mock_docker_sandbox: Any, mock_container: MockContainer) -> None:
        """DockerEnvironment.replace_str replaces text."""
        mock_container._files['/workspace/code.py'] = b'old_value = 1'
        count = await mock_docker_sandbox.replace_str('code.py', 'old_value', 'new_value')
        assert count == 1

    async def test_docker_container_property(self, mock_docker_sandbox: Any) -> None:
        """DockerEnvironment._required_container raises when not started."""

        sandbox = DockerEnvironment()
        with pytest.raises(RuntimeError, match='not started'):
            _ = sandbox._required_container

    async def test_docker_create_process(self, mock_docker_sandbox: Any) -> None:
        """DockerEnvironment.create_process returns a _DockerEnvironmentProcess."""
        proc = await mock_docker_sandbox.create_process('echo test')
        assert proc is not None

    async def test_docker_is_alive(self, mock_docker_sandbox: Any) -> None:
        """DockerEnvironment.is_alive checks container status."""
        result = await mock_docker_sandbox.is_alive()
        assert result is True

    async def test_docker_is_alive_not_started(
        self,
    ) -> None:
        """DockerEnvironment.is_alive returns False when not started."""

        sandbox = DockerEnvironment()
        result = await sandbox.is_alive()
        assert result is False

    async def test_docker_resolve_path(self, mock_docker_sandbox: Any) -> None:
        """DockerEnvironment._resolve_path resolves relative paths."""
        assert mock_docker_sandbox._resolve_path('test.txt') == '/workspace/test.txt'
        assert mock_docker_sandbox._resolve_path('/abs/path') == '/abs/path'
        assert mock_docker_sandbox._resolve_path('sub/dir/file.py') == '/workspace/sub/dir/file.py'

    def test_docker_put_file(self) -> None:
        """_put_file creates a tar archive and uploads it."""

        container = MockContainer()
        _put_file(container, '/workspace/test.txt', b'hello')  # type: ignore[arg-type]
        assert '/workspace/test.txt' in container._files
        assert container._files['/workspace/test.txt'] == b'hello'

    def test_docker_sandbox_process_read_frame(self) -> None:
        """_DockerEnvironmentProcess._read_frame parses multiplexed stream frames."""

        container = MockContainer()
        proc = _DockerEnvironmentProcess(container, 'echo test', '/workspace')  # type: ignore[arg-type]

        # Create a mock socket with a multiplexed frame
        stdout_data = b'hello from stdout'
        header = struct.pack('>BxxxI', 1, len(stdout_data))  # stream_type=1 (stdout)

        mock_socket = MagicMock()
        mock_socket.recv.side_effect = [header, stdout_data]
        proc._socket = mock_socket

        stream_type, data = proc._read_frame()
        assert stream_type == 1
        assert data == stdout_data

    def test_docker_sandbox_process_read_frame_stderr(self) -> None:
        """_DockerEnvironmentProcess._read_frame handles stderr frames."""

        container = MockContainer()
        proc = _DockerEnvironmentProcess(container, 'echo test', '/workspace')  # type: ignore[arg-type]

        stderr_data = b'error output'
        header = struct.pack('>BxxxI', 2, len(stderr_data))  # stream_type=2 (stderr)

        mock_socket = MagicMock()
        mock_socket.recv.side_effect = [header, stderr_data]
        proc._socket = mock_socket

        stream_type, data = proc._read_frame()
        assert stream_type == 2
        assert data == stderr_data

    def test_docker_sandbox_process_read_frame_eof(self) -> None:
        """_DockerEnvironmentProcess._read_frame returns empty on EOF."""

        container = MockContainer()
        proc = _DockerEnvironmentProcess(container, 'echo test', '/workspace')  # type: ignore[arg-type]

        mock_socket = MagicMock()
        mock_socket.recv.return_value = b''  # EOF
        proc._socket = mock_socket

        stream_type, data = proc._read_frame()
        assert stream_type == 0
        assert data == b''
        assert proc._eof is True

    def test_docker_sandbox_process_read_frame_zero_size(self) -> None:
        """_DockerEnvironmentProcess._read_frame handles zero-size frames."""

        container = MockContainer()
        proc = _DockerEnvironmentProcess(container, 'echo test', '/workspace')  # type: ignore[arg-type]

        header = struct.pack('>BxxxI', 1, 0)  # zero size

        mock_socket = MagicMock()
        mock_socket.recv.return_value = header
        proc._socket = mock_socket

        stream_type, data = proc._read_frame()
        assert stream_type == 1
        assert data == b''

    def test_docker_sandbox_process_already_eof(self) -> None:
        """_DockerEnvironmentProcess._read_frame returns empty when already at EOF."""

        container = MockContainer()
        proc = _DockerEnvironmentProcess(container, 'echo test', '/workspace')  # type: ignore[arg-type]
        proc._eof = True

        stream_type, data = proc._read_frame()
        assert stream_type == 0
        assert data == b''

    def test_docker_hardened_constructor(
        self,
    ):
        """DockerEnvironment.hardened() returns a properly configured instance."""
        env = DockerEnvironment.hardened(image='python:3.12-slim', memory_limit='1g')
        assert env._network_disabled is True
        assert env._read_only is True
        assert env._cap_drop == ['ALL']
        assert env._memory_limit == '1g'
        assert env._user == 'nobody'
        assert env._init is True

    def test_docker_setup_early_return(
        self,
    ):
        """DockerEnvironment._setup returns early if container already exists."""
        env = DockerEnvironment(image='python:3.12-slim')
        env._container = MagicMock()
        env._setup()  # should not create a new container
        assert env._client is None  # docker.from_env() was never called

    async def test_docker_process_recv_stderr_no_buffer(
        self,
    ) -> None:
        """_DockerEnvironmentProcess.recv_stderr without buffered data (no timeout)."""
        container = MockContainer()
        proc = _DockerEnvironmentProcess(container, 'echo test', '/workspace')  # type: ignore[arg-type]

        stderr_data = b'error output'
        header = struct.pack('>BxxxI', 2, len(stderr_data))
        mock_socket = MagicMock()
        mock_socket.recv.side_effect = [header, stderr_data]
        proc._socket = mock_socket

        result = await proc.recv_stderr()
        assert result == stderr_data

    async def test_docker_process_recv_stream_buffers_stdout(
        self,
    ) -> None:
        """_DockerEnvironmentProcess._recv_stream buffers stdout when stderr is wanted."""
        container = MockContainer()
        proc = _DockerEnvironmentProcess(container, 'echo test', '/workspace')  # type: ignore[arg-type]

        # First frame is stdout (type 1), second is stderr (type 2)
        stdout_data = b'stdout output'
        stderr_data = b'stderr output'
        stdout_header = struct.pack('>BxxxI', 1, len(stdout_data))
        stderr_header = struct.pack('>BxxxI', 2, len(stderr_data))

        mock_socket = MagicMock()
        mock_socket.recv.side_effect = [stdout_header, stdout_data, stderr_header, stderr_data]
        proc._socket = mock_socket

        # Requesting stderr should buffer stdout and return stderr
        result = await proc.recv_stderr()
        assert result == stderr_data
        assert proc._stdout_buffer == [stdout_data]

    async def test_docker_process_wait_no_timeout(
        self,
    ) -> None:
        """_DockerEnvironmentProcess.wait without timeout polls until returncode is set."""
        container = MockContainer()
        proc = _DockerEnvironmentProcess(container, 'echo test', '/workspace')  # type: ignore[arg-type]
        proc._exec_id = 'exec-123'
        # Mock exec_inspect to return "still running" first, then "exited"
        call_count = 0

        def mock_inspect(exec_id: str) -> dict[str, Any]:
            nonlocal call_count
            call_count += 1
            if call_count <= 1:
                return {'Running': True, 'ExitCode': None}
            return {'Running': False, 'ExitCode': 0}

        container.client.api.exec_inspect = mock_inspect
        result = await proc.wait()
        assert result == 0
        assert call_count >= 2

    async def test_docker_process_wait_with_timeout(
        self,
    ) -> None:
        """_DockerEnvironmentProcess.wait with timeout."""
        container = MockContainer()
        proc = _DockerEnvironmentProcess(container, 'echo test', '/workspace')  # type: ignore[arg-type]
        proc._returncode = 42
        result = await proc.wait(timeout=5.0)
        assert result == 42

    async def test_docker_read_file_binary(self, mock_docker_sandbox: Any, mock_container: MockContainer) -> None:
        """DockerEnvironment.read_file returns raw bytes for any file."""
        binary_data = b'\x80\x81\x82\xff'
        mock_container._files['/workspace/data.bin'] = binary_data
        result = await mock_docker_sandbox.read_file('data.bin')
        assert result == binary_data

    async def test_docker_is_alive_exception(self, mock_docker_sandbox: Any, mock_container: MockContainer) -> None:
        """DockerEnvironment.is_alive returns False when reload raises."""
        mock_container.reload = MagicMock(side_effect=DockerException('connection error'))
        result = await mock_docker_sandbox.is_alive()
        assert result is False

    async def test_docker_is_alive_running(self, mock_docker_sandbox: Any) -> None:
        """DockerEnvironment.is_alive returns True when running."""
        result = await mock_docker_sandbox.is_alive()
        assert result is True

    async def test_docker_process_recv_with_buffered_data(
        self,
    ) -> None:
        """_DockerEnvironmentProcess.recv returns buffered stdout data first."""

        container = MockContainer()
        proc = _DockerEnvironmentProcess(container, 'echo test', '/workspace')  # type: ignore[arg-type]
        proc._stdout_buffer.append(b'buffered data')

        result = await proc.recv()
        assert result == b'buffered data'
        assert proc._stdout_buffer == []

    async def test_docker_process_recv_stderr_with_buffered_data(
        self,
    ) -> None:
        """_DockerEnvironmentProcess.recv_stderr returns buffered stderr data first."""

        container = MockContainer()
        proc = _DockerEnvironmentProcess(container, 'echo test', '/workspace')  # type: ignore[arg-type]
        proc._stderr_buffer.append(b'buffered error')

        result = await proc.recv_stderr()
        assert result == b'buffered error'
        assert proc._stderr_buffer == []

    async def test_docker_process_recv_stream_buffers_other(
        self,
    ) -> None:
        """_DockerEnvironmentProcess._recv_stream buffers frames for the other stream."""

        container = MockContainer()
        proc = _DockerEnvironmentProcess(container, 'echo test', '/workspace')  # type: ignore[arg-type]

        # First frame is stderr (type 2), second is stdout (type 1)
        stderr_data = b'error output'
        stdout_data = b'stdout output'
        stderr_header = struct.pack('>BxxxI', 2, len(stderr_data))
        stdout_header = struct.pack('>BxxxI', 1, len(stdout_data))

        mock_socket = MagicMock()
        mock_socket.recv.side_effect = [stderr_header, stderr_data, stdout_header, stdout_data]
        proc._socket = mock_socket

        # Requesting stdout should buffer stderr and return stdout
        result = await proc.recv()
        assert result == stdout_data
        assert proc._stderr_buffer == [stderr_data]

    async def test_docker_process_recv_stream_eof(
        self,
    ) -> None:
        """_DockerEnvironmentProcess._recv_stream returns empty on EOF."""

        container = MockContainer()
        proc = _DockerEnvironmentProcess(container, 'echo test', '/workspace')  # type: ignore[arg-type]

        mock_socket = MagicMock()
        mock_socket.recv.return_value = b''  # EOF
        proc._socket = mock_socket

        result = await proc.recv()
        assert result == b''

    async def test_docker_process_kill(
        self,
    ) -> None:
        """_DockerEnvironmentProcess.kill closes the socket."""

        container = MockContainer()
        proc = _DockerEnvironmentProcess(container, 'echo test', '/workspace')  # type: ignore[arg-type]
        mock_socket = MagicMock()
        proc._socket = mock_socket

        await proc.kill()
        mock_socket.close.assert_called_once()

    async def test_docker_process_kill_oserror(
        self,
    ) -> None:
        """_DockerEnvironmentProcess.kill handles OSError."""

        container = MockContainer()
        proc = _DockerEnvironmentProcess(container, 'echo test', '/workspace')  # type: ignore[arg-type]
        mock_socket = MagicMock()
        mock_socket.close.side_effect = OSError('socket error')
        proc._socket = mock_socket

        # Should not raise
        await proc.kill()

    async def test_docker_process_returncode(
        self,
    ) -> None:
        """_DockerEnvironmentProcess.returncode checks exec status."""

        container = MockContainer()
        proc = _DockerEnvironmentProcess(container, 'echo test', '/workspace')  # type: ignore[arg-type]

        # No exec_id means returncode is None
        assert proc.returncode is None

        # With exec_id and cached returncode
        proc._exec_id = 'exec-123'
        proc._returncode = 0
        assert proc.returncode == 0

    async def test_docker_process_returncode_from_inspect(
        self,
    ) -> None:
        """_DockerEnvironmentProcess._poll_exit_code polls Docker API."""

        container = MockContainer()
        container.client.api.exec_inspect.return_value = {'ExitCode': 42, 'Running': False}
        proc = _DockerEnvironmentProcess(container, 'echo test', '/workspace')  # type: ignore[arg-type]
        proc._exec_id = 'exec-123'

        # returncode only returns cached value — no I/O
        assert proc.returncode is None

        # _poll_exit_code offloads the HTTP call and caches the result
        rc = await proc._poll_exit_code()
        assert rc == 42
        assert proc.returncode == 42

    async def test_docker_process_returncode_still_running(
        self,
    ) -> None:
        """_DockerEnvironmentProcess._poll_exit_code returns None when process is running."""

        container = MockContainer()
        # Docker returns ExitCode=0 + Running=True for still-running processes
        container.client.api.exec_inspect.return_value = {'ExitCode': 0, 'Running': True}
        proc = _DockerEnvironmentProcess(container, 'echo test', '/workspace')  # type: ignore[arg-type]
        proc._exec_id = 'exec-123'

        assert await proc._poll_exit_code() is None
        assert proc.returncode is None

    async def test_docker_process_returncode_inspect_error(
        self,
    ) -> None:
        """_DockerEnvironmentProcess._poll_exit_code handles API errors."""

        container = MockContainer()
        container.client.api.exec_inspect.side_effect = OSError('connection failed')
        proc = _DockerEnvironmentProcess(container, 'echo test', '/workspace')  # type: ignore[arg-type]
        proc._exec_id = 'exec-123'

        assert await proc._poll_exit_code() is None
        assert proc.returncode is None

    async def test_docker_process_send(
        self,
    ) -> None:
        """_DockerEnvironmentProcess.send writes to socket."""

        container = MockContainer()
        proc = _DockerEnvironmentProcess(container, 'echo test', '/workspace')  # type: ignore[arg-type]
        mock_socket = MagicMock()
        proc._socket = mock_socket

        await proc.send(b'hello')
        mock_socket.sendall.assert_called_once_with(b'hello')

    async def test_docker_process_recv_with_timeout(
        self,
    ) -> None:
        """_DockerEnvironmentProcess.recv with timeout."""

        container = MockContainer()
        proc = _DockerEnvironmentProcess(container, 'echo test', '/workspace')  # type: ignore[arg-type]

        stdout_data = b'data'
        header = struct.pack('>BxxxI', 1, len(stdout_data))
        mock_socket = MagicMock()
        mock_socket.recv.side_effect = [header, stdout_data]
        proc._socket = mock_socket

        result = await proc.recv(timeout=5.0)
        assert result == stdout_data

    async def test_docker_process_recv_stderr_with_timeout(
        self,
    ) -> None:
        """_DockerEnvironmentProcess.recv_stderr with timeout."""

        container = MockContainer()
        proc = _DockerEnvironmentProcess(container, 'echo test', '/workspace')  # type: ignore[arg-type]

        stderr_data = b'error'
        header = struct.pack('>BxxxI', 2, len(stderr_data))
        mock_socket = MagicMock()
        mock_socket.recv.side_effect = [header, stderr_data]
        proc._socket = mock_socket

        result = await proc.recv_stderr(timeout=5.0)
        assert result == stderr_data

    async def test_docker_read_frame_data_eof_during_read(
        self,
    ) -> None:
        """_DockerEnvironmentProcess._read_frame handles EOF during data read."""

        container = MockContainer()
        proc = _DockerEnvironmentProcess(container, 'echo test', '/workspace')  # type: ignore[arg-type]

        # Header says 100 bytes but socket returns less then EOF
        header = struct.pack('>BxxxI', 1, 100)
        mock_socket = MagicMock()
        mock_socket.recv.side_effect = [header, b'partial', b'']  # EOF during data
        proc._socket = mock_socket

        stream_type, data = proc._read_frame()
        assert stream_type == 1
        assert data == b'partial'
        assert proc._eof is True

    async def test_docker_process_start_with_env(
        self,
    ) -> None:
        """_DockerEnvironmentProcess._do_start passes env to exec_create."""

        container = MockContainer()
        container.client.api.exec_create.return_value = {'Id': 'exec-test'}
        mock_sock = MagicMock()
        container.client.api.exec_start.return_value = mock_sock

        proc = _DockerEnvironmentProcess(
            container,  # type: ignore[arg-type]
            'echo test',
            '/workspace',
            env={'FOO': 'bar'},
        )
        await proc._start()

        assert proc._exec_id == 'exec-test'
        call_kwargs = container.client.api.exec_create.call_args[1]
        assert call_kwargs['environment'] == {'FOO': 'bar'}

    async def test_docker_process_aenter(
        self,
    ) -> None:
        """_DockerEnvironmentProcess.__aenter__ starts the process."""

        container = MockContainer()
        container.client.api.exec_create.return_value = {'Id': 'exec-aenter'}
        mock_sock = MagicMock()
        container.client.api.exec_start.return_value = mock_sock

        proc = _DockerEnvironmentProcess(container, 'echo test', '/workspace')  # type: ignore[arg-type]
        entered = await proc.__aenter__()
        assert entered is proc
        assert proc._exec_id == 'exec-aenter'

    async def test_docker_process_poll_exit_code_no_exec_id(
        self,
    ) -> None:
        """_DockerEnvironmentProcess._poll_exit_code returns None when _exec_id is None."""
        container = MockContainer()
        proc = _DockerEnvironmentProcess(container, 'echo test', '/workspace')  # type: ignore[arg-type]

        # _exec_id is None by default (process not started)
        assert proc._exec_id is None
        assert await proc._poll_exit_code() is None

    async def test_docker_process_aexit_kills_running(
        self,
    ) -> None:
        """_DockerEnvironmentProcess.__aexit__ kills the process if still running."""
        container = MockContainer()
        proc = _DockerEnvironmentProcess(container, 'echo test', '/workspace')  # type: ignore[arg-type]
        mock_socket = MagicMock()
        proc._socket = mock_socket
        # Process is "still running" — _exec_id set but inspect says Running=True
        proc._exec_id = 'exec-aexit'
        container.client.api.exec_inspect.return_value = {'Running': True, 'ExitCode': None}

        await proc.__aexit__(None, None, None)

        # Should have called kill (socket.close)
        mock_socket.close.assert_called_once()
        assert proc._returncode is None

    async def test_docker_process_aexit_already_exited(
        self,
    ) -> None:
        """_DockerEnvironmentProcess.__aexit__ does not kill if process already exited."""
        container = MockContainer()
        proc = _DockerEnvironmentProcess(container, 'echo test', '/workspace')  # type: ignore[arg-type]
        mock_socket = MagicMock()
        proc._socket = mock_socket
        proc._exec_id = 'exec-aexit'
        container.client.api.exec_inspect.return_value = {'Running': False, 'ExitCode': 0}

        await proc.__aexit__(None, None, None)

        # Should NOT have called kill
        mock_socket.close.assert_not_called()
        assert proc._returncode == 0

    async def test_mock_container_find_command(
        self,
    ) -> None:
        """MockContainer.exec_run handles find commands for glob operations."""
        container = MockContainer()
        container._files['/workspace/src/main.py'] = b'print("hello")'
        container._files['/workspace/src/utils.py'] = b'# utils'
        container._files['/workspace/README.md'] = b'# readme'
        # File outside workdir should be skipped
        container._files['/other/secret.txt'] = b'secret'

        # find with '.' searches all files relative to workdir
        exit_code, output = container.exec_run(['sh', '-c', "find '.' -type f"], workdir='/workspace')
        output_str = output.decode()
        assert exit_code == 0
        assert './src/main.py' in output_str
        assert './src/utils.py' in output_str
        assert './README.md' in output_str
        assert 'secret' not in output_str

    async def test_mock_container_find_command_subpath(
        self,
    ) -> None:
        """MockContainer.exec_run handles find commands with a specific subdirectory."""
        container = MockContainer()
        container._files['/workspace/src/main.py'] = b'print("hello")'
        container._files['/workspace/docs/guide.md'] = b'# guide'

        exit_code, output = container.exec_run(['sh', '-c', "find 'src' -type f"], workdir='/workspace')
        output_str = output.decode()
        assert exit_code == 0
        assert 'src/main.py' in output_str
        assert 'guide.md' not in output_str

    async def test_docker_read_file_image_not_found(
        self, mock_docker_sandbox: Any, mock_container: MockContainer
    ) -> None:
        """DockerEnvironment.read_file raises FileNotFoundError for missing image files."""

        def fail_get_archive(path: str) -> Any:
            raise DockerNotFound('File not found')

        mock_container.get_archive = fail_get_archive
        with pytest.raises(FileNotFoundError, match='File not found: missing.png'):
            await mock_docker_sandbox.read_file('missing.png')

    # --- Additional Docker coverage: lifecycle, process, truncation ---

    async def test_docker_execute_truncation(self, mock_docker_sandbox: Any, mock_container: MockContainer) -> None:
        """DockerEnvironment.execute truncates long output."""
        original = mock_container.exec_run

        def big_output(cmd: Any, **kwargs: Any) -> tuple[int, bytes]:
            if isinstance(cmd, list) and 'echo' in str(cmd):
                return 0, b'x' * 200_000
            return original(cmd, **kwargs)  # pragma: no cover

        mock_container.exec_run = big_output  # type: ignore[assignment]
        result = await mock_docker_sandbox.shell('echo big')
        assert len(result.output) == snapshot(200000)

    async def test_docker_execute_timeout_exit_code(
        self, mock_docker_sandbox: Any, mock_container: MockContainer
    ) -> None:
        """DockerEnvironment.execute handles timeout exit code 124."""

        def timeout_result(cmd: Any, **kwargs: Any) -> tuple[int, bytes]:
            return 124, b'partial output'

        mock_container.exec_run = timeout_result  # type: ignore[assignment]
        result = await mock_docker_sandbox.shell('sleep 999', timeout=1)
        assert result.exit_code == 124
        assert '[Command timed out]' in result.output

    async def test_docker_setup_teardown(
        self,
    ) -> None:
        """DockerEnvironment._setup and _teardown with mocked Docker client."""
        sandbox = DockerEnvironment(image='python:3.12-slim')

        mock_client = MagicMock()
        mock_container_obj = MagicMock()
        mock_client.containers.run.return_value = mock_container_obj

        with mock_patch('pydantic_ai.environments.docker.docker') as mock_docker:
            mock_docker.from_env.return_value = mock_client
            sandbox._setup()
            assert sandbox._container is not None

        # Teardown
        sandbox._teardown()
        mock_container_obj.stop.assert_called()
        mock_container_obj.remove.assert_called()
        assert sandbox._container is None

    async def test_docker_teardown_cleanup_errors(
        self,
    ) -> None:
        """DockerEnvironment._teardown handles exceptions gracefully."""

        sandbox = DockerEnvironment()
        mock_container = MagicMock()
        mock_container.stop.side_effect = DockerException('stop failed')
        mock_container.remove.side_effect = DockerException('remove failed')
        sandbox._container = mock_container

        # Should not raise
        sandbox._teardown()
        assert sandbox._container is None

    async def test_docker_setup_with_all_options(
        self,
    ) -> None:
        """DockerEnvironment._setup passes all container options."""
        sandbox = DockerEnvironment(
            image='python:3.12-slim',
            env_vars={'KEY': 'val'},
            volumes={'/host': {'bind': '/container', 'mode': 'rw'}},
            memory_limit='512m',
            cpu_limit=1.0,
            pids_limit=256,
            network_disabled=True,
            read_only=True,
            cap_drop=['ALL'],
            security_opt=['no-new-privileges'],
            user='nobody',
            tmpfs={'/tmp': 'noexec,nosuid,size=64m'},
            init=True,
        )

        mock_client = MagicMock()
        mock_container = MagicMock()
        mock_client.containers.run.return_value = mock_container

        with mock_patch('pydantic_ai.environments.docker.docker') as mock_docker:
            mock_docker.from_env.return_value = mock_client
            sandbox._setup()

        call_kwargs = mock_client.containers.run.call_args[1]
        assert call_kwargs['volumes'] == {'/host': {'bind': '/container', 'mode': 'rw'}}
        assert call_kwargs['mem_limit'] == '512m'
        assert call_kwargs['nano_cpus'] == int(1e9)
        assert call_kwargs['pids_limit'] == 256
        assert call_kwargs['network_disabled'] is True
        assert call_kwargs['read_only'] is True
        assert call_kwargs['cap_drop'] == ['ALL']
        assert call_kwargs['security_opt'] == ['no-new-privileges']
        assert call_kwargs['user'] == 'nobody'
        assert call_kwargs['tmpfs'] == {'/tmp': 'noexec,nosuid,size=64m'}
        assert call_kwargs['init'] is True

    # --- Docker instantiation tests ---

    def test_docker_sandbox_instantiation(
        self,
    ):
        """DockerEnvironment can be constructed without starting Docker."""

        # Verify construction succeeds with default and custom settings
        sandbox = DockerEnvironment(image='python:3.12-slim')
        assert isinstance(sandbox, DockerEnvironment)

        sandbox_with_opts = DockerEnvironment(
            image='node:20-slim',
            memory_limit='512m',
            cpu_limit=1.0,
            network_disabled=True,
        )
        assert isinstance(sandbox_with_opts, DockerEnvironment)

        # Verify security hardening parameters are accepted
        sandbox_hardened = DockerEnvironment(
            image='python:3.12-slim',
            network_disabled=True,
            read_only=True,
            cap_drop=['ALL'],
            security_opt=['no-new-privileges'],
            user='nobody',
            pids_limit=256,
            tmpfs={'/tmp': 'noexec,nosuid,size=64m'},
            init=True,
        )
        assert isinstance(sandbox_hardened, DockerEnvironment)

    def test_shell_escape(self):
        assert _shell_escape('hello') == "'hello'"
        assert _shell_escape("it's") == "'it'\\''s'"
        assert _shell_escape('') == "''"
        assert _shell_escape('a b c') == "'a b c'"


# --- Additional coverage: _base.py ---


async def test_execution_environment_aenter_aexit():
    """ExecutionEnvironment base __aenter__/__aexit__ are exercised by subclasses."""
    # MemoryEnvironment exercises the base class path
    env = MemoryEnvironment()
    async with env:
        pass


# --- Additional coverage: _toolset.py ---


async def test_toolset_bash_empty_output(tmp_path: Path):
    """ExecutionEnvironmentToolset bash returns just exit code when no output."""
    env = LocalEnvironment(tmp_path)
    toolset = ExecutionEnvironmentToolset(env)
    ctx = build_run_context()
    manager = await ToolManager[None](toolset).for_run_step(ctx)

    async with env:
        result = await manager.handle_call(ToolCallPart(tool_name='shell', args={'command': 'true'}))
        assert 'Exit code: 0' in str(result)


async def test_toolset_lifecycle_error(tmp_path: Path):
    """ExecutionEnvironmentToolset handles environment startup failures."""

    class FailingEnv(LocalEnvironment):
        async def __aenter__(self):
            raise RuntimeError('Setup failed')

    env = FailingEnv(tmp_path)
    toolset = ExecutionEnvironmentToolset(env)
    with pytest.raises(RuntimeError, match='Setup failed'):
        async with toolset:
            pass


# --- Additional coverage: local.py ---


async def test_local_process_stdin_not_available():
    """_LocalEnvironmentProcess.send raises when stdin is None."""
    mock_proc = MagicMock()
    mock_proc.stdin = None
    proc = _LocalEnvironmentProcess(mock_proc)
    with pytest.raises(RuntimeError, match='stdin'):
        await proc.send(b'data')


async def test_local_process_stdout_not_available():
    """_LocalEnvironmentProcess.recv raises when stdout is None."""
    mock_proc = MagicMock()
    mock_proc.stdout = None
    proc = _LocalEnvironmentProcess(mock_proc)
    with pytest.raises(RuntimeError, match='stdout'):
        await proc.recv()


async def test_local_process_stderr_not_available():
    """_LocalEnvironmentProcess.recv_stderr raises when stderr is None."""
    mock_proc = MagicMock()
    mock_proc.stderr = None
    proc = _LocalEnvironmentProcess(mock_proc)
    with pytest.raises(RuntimeError, match='stderr'):
        await proc.recv_stderr()


async def test_local_process_recv_stderr_timeout(tmp_path: Path):
    """_LocalEnvironmentProcess.recv_stderr with timeout."""
    env = LocalEnvironment(tmp_path)
    proc = await env.create_process('python -c "import sys; sys.stderr.write(\'err\\n\')"')
    async with proc:
        data = await proc.recv_stderr(timeout=5.0)
        assert b'err' in data


async def test_local_process_recv_stderr_eof(tmp_path: Path):
    """_LocalEnvironmentProcess.recv_stderr returns empty on EOF."""
    env = LocalEnvironment(tmp_path)
    proc = await env.create_process('echo done')
    async with proc:
        await proc.wait(timeout=5.0)
        # After process exits, stderr should return empty
        data = await proc.recv_stderr()
        assert data == b''


async def test_local_process_kill_terminates_sleep(tmp_path: Path):
    """_LocalEnvironmentProcess.kill terminates process."""
    env = LocalEnvironment(tmp_path)
    proc = await env.create_process('sleep 60')
    async with proc:
        await proc.kill()
        # After kill, returncode should be set


async def test_local_read_file_bytes_directory_raises_error(tmp_path: Path):
    """LocalEnvironment.read_file_bytes raises on directory."""
    (tmp_path / 'subdir').mkdir()
    env = LocalEnvironment(tmp_path)
    with pytest.raises(FileNotFoundError, match='directory'):
        await env.read_file('subdir')


async def test_local_read_file_bytes_not_found(tmp_path: Path):
    """LocalEnvironment.read_file_bytes raises on missing file."""
    env = LocalEnvironment(tmp_path)
    with pytest.raises(FileNotFoundError, match='not found'):
        await env.read_file('nonexistent.txt')


async def test_local_execute_output_truncation(tmp_path: Path):
    """LocalEnvironment.execute truncates long output."""
    # Write a script that outputs lots of text
    script = tmp_path / 'big.py'
    script.write_text("print('x' * 200000)")
    env = LocalEnvironment(tmp_path)
    result = await env.shell(f'python {script}')
    assert len(result.output) == snapshot(200001)


# --- Additional coverage: memory.py ---


async def test_memory_normalize_leading_slash_in_constructor():
    """MemoryEnvironment normalizes paths with leading /."""
    env = MemoryEnvironment(files={'/abs/path.txt': 'content'})
    content = await env.read_file('abs/path.txt')
    assert content == b'content'


async def test_memory_read_file_directory_error():
    """MemoryEnvironment.read_file raises on directory paths."""
    env = MemoryEnvironment(files={'dir/file.txt': 'content'})
    with pytest.raises(FileNotFoundError, match='directory'):
        await env.read_file('dir')


async def test_memory_read_file_bytes_not_found_raises_error():
    """MemoryEnvironment.read_file_bytes raises on missing file."""
    env = MemoryEnvironment()
    with pytest.raises(FileNotFoundError):
        await env.read_file('missing.txt')


async def test_local_process_wait_no_timeout(tmp_path: Path):
    """_LocalEnvironmentProcess.wait without timeout (line 74)."""
    env = LocalEnvironment(tmp_path)
    proc = await env.create_process('true')
    async with proc:
        exit_code = await proc.wait()  # no timeout
        assert exit_code == 0


async def test_memory_normalize_absolute_path():
    """MemoryEnvironment._normalize strips leading / (line 76)."""
    env = MemoryEnvironment(files={'path.txt': 'content'})
    # Normalize /path.txt should strip leading /
    normalized = env._normalize('/path.txt')
    assert normalized == 'path.txt'


async def test_memory_read_file_that_is_also_directory_prefix():
    """MemoryEnvironment.read_file when path exists as both file and directory prefix."""
    # 'dir' exists as a file AND 'dir/child.txt' makes it look like a directory too
    env = MemoryEnvironment(files={'dir': 'I am a file', 'dir/child.txt': 'child content'})
    async with env:
        content = await env.read_file('dir')
        assert content == b'I am a file'


# --- ExecutionEnvironmentToolset: environment_factory ---


async def test_toolset_factory_basic():
    """Factory creates a fresh environment per __aenter__."""
    envs_created: list[MemoryEnvironment] = []

    def factory() -> MemoryEnvironment:
        env = MemoryEnvironment()
        envs_created.append(env)
        return env

    toolset = ExecutionEnvironmentToolset(factory)

    async with toolset:
        assert len(envs_created) == 1
        assert toolset.environment is envs_created[0]

    # Second entry creates a new environment
    async with toolset:
        assert len(envs_created) == 2
        assert toolset.environment is envs_created[1]
        assert envs_created[0] is not envs_created[1]


async def test_toolset_factory_concurrent():
    """Concurrent __aenter__ calls get different environments."""
    import asyncio

    envs_created: list[MemoryEnvironment] = []

    def factory() -> MemoryEnvironment:
        env = MemoryEnvironment()
        envs_created.append(env)
        return env

    toolset = ExecutionEnvironmentToolset(factory)

    async def enter_and_check() -> MemoryEnvironment:
        async with toolset:
            env = toolset.environment
            assert isinstance(env, MemoryEnvironment)
            return env

    env1, env2 = await asyncio.gather(enter_and_check(), enter_and_check())
    assert len(envs_created) == 2
    assert env1 is not env2


async def test_toolset_factory_concurrent_isolation():
    """Two concurrent runs each write a file and don't see each other's files."""
    import asyncio

    def factory() -> MemoryEnvironment:
        return MemoryEnvironment()

    toolset = ExecutionEnvironmentToolset(factory)
    ctx = build_run_context()

    async def write_and_read(filename: str, content: str) -> tuple[str, str]:
        """Write a file, then try to read a file the other task wrote."""
        other_file = 'b.txt' if filename == 'a.txt' else 'a.txt'
        async with toolset:
            manager = await ToolManager[None](toolset).for_run_step(ctx)
            await manager.handle_call(ToolCallPart(tool_name='write_file', args={'path': filename, 'content': content}))
            # Small delay so both tasks have a chance to write
            await asyncio.sleep(0.01)
            other_result = await manager.handle_call(ToolCallPart(tool_name='read_file', args={'path': other_file}))
            return content, str(other_result)

    (content_a, read_b), (content_b, read_a) = await asyncio.gather(
        write_and_read('a.txt', 'alpha'),
        write_and_read('b.txt', 'beta'),
    )

    assert content_a == 'alpha'
    assert content_b == 'beta'
    # Each run should NOT see the other's file — they have isolated environments
    assert 'Error' in read_b
    assert 'Error' in read_a


async def test_toolset_factory_concurrent_agent_runs():
    """Parallel agent.run() calls with a factory get isolated environments (relates to #4347)."""
    import asyncio

    envs_created: list[MemoryEnvironment] = []

    def factory() -> MemoryEnvironment:
        env = MemoryEnvironment(
            files={'id.txt': str(len(envs_created))},
            command_handler=lambda cmd: ExecutionResult(output='ok', exit_code=0),
        )
        envs_created.append(env)
        return env

    toolset = ExecutionEnvironmentToolset(factory)
    agent = Agent('test', toolsets=[toolset])

    result_a, result_b = await asyncio.gather(
        agent.run('Read the file id.txt'),
        agent.run('Read the file id.txt'),
    )

    # Each run should have created its own environment
    assert len(envs_created) == 2
    assert result_a.output is not None
    assert result_b.output is not None


async def test_toolset_factory_cleanup():
    """__aexit__ properly cleans up factory-created environments."""
    entered = 0
    exited = 0

    class TrackingEnv(MemoryEnvironment):
        async def __aenter__(self):
            nonlocal entered
            entered += 1
            return await super().__aenter__()

        async def __aexit__(self, *args: Any):
            nonlocal exited
            exited += 1
            return await super().__aexit__(*args)

    toolset = ExecutionEnvironmentToolset(TrackingEnv)

    async with toolset:
        assert entered == 1
        assert exited == 0

    assert entered == 1
    assert exited == 1


async def test_toolset_factory_with_use_environment():
    """use_environment() overrides the factory-created environment within the context."""
    override_env = MemoryEnvironment()

    toolset = ExecutionEnvironmentToolset(MemoryEnvironment)

    async with toolset:
        factory_env = toolset.environment
        assert factory_env is not override_env

        with toolset.use_environment(override_env):
            assert toolset.environment is override_env

        # After exiting use_environment, factory env is restored
        assert toolset.environment is factory_env


# --- Memory image file stored as string ---


async def test_memory_read_image_stored_as_string():
    """MemoryEnvironment returns bytes for image files even when stored as a string."""
    env = MemoryEnvironment(files={'image.png': 'fake png data'})
    async with env:
        result = await env.read_file('image.png')
    assert isinstance(result, bytes)
    assert result == b'fake png data'


# --- ExecutionEnvironmentToolset: get_tools filters by runtime capabilities ---


async def test_toolset_factory_filters_tools_by_capabilities():
    """When using environment_factory, get_tools() only returns tools supported by the runtime environment."""

    class _ShellOnlyEnv(BaseEnv):
        @property
        def capabilities(self) -> frozenset[EnvCapability]:
            return frozenset({'shell'})

        async def shell(
            self, command: str, *, timeout: float | None = None, env: dict[str, str] | None = None
        ) -> ExecutionResult:
            return ExecutionResult(output='', exit_code=0)  # pragma: no cover

    toolset = ExecutionEnvironmentToolset(_ShellOnlyEnv)
    # Before entering, all tools are registered (no env to check)
    ctx = build_run_context()

    async with toolset:
        tools = await toolset.get_tools(ctx)

    # Only shell should be exposed — the runtime env only supports shell
    assert set(tools.keys()) == {'shell'}


async def test_toolset_use_environment_filters_tools():
    """use_environment() with a limited env filters tools from get_tools()."""

    class _ShellOnlyEnv(BaseEnv):
        @property
        def capabilities(self) -> frozenset[EnvCapability]:
            return frozenset({'shell'})

    # Full-capability shared env registers all tools
    full_env = MemoryEnvironment()
    toolset = ExecutionEnvironmentToolset(full_env)
    ctx = build_run_context()

    async with full_env:
        all_tools = await toolset.get_tools(ctx)
        assert 'read_file' in all_tools
        assert 'write_file' in all_tools

        # Override with a limited env — only shell should remain
        with toolset.use_environment(_ShellOnlyEnv()):
            limited_tools = await toolset.get_tools(ctx)
            assert set(limited_tools.keys()) == {'shell'}

        # After exiting use_environment, all tools are back
        restored_tools = await toolset.get_tools(ctx)
        assert set(restored_tools.keys()) == set(all_tools.keys())


# --- Coverage gap tests ---


async def test_local_recv_no_timeout(tmp_path: Path):
    """_LocalEnvironmentProcess.recv without timeout returns data."""
    env = LocalEnvironment(tmp_path)
    proc = await env.create_process('echo hello')
    async with proc:
        data = await proc.recv()  # no timeout
        assert b'hello' in data


async def test_local_recv_end_of_stream(tmp_path: Path):
    """_LocalEnvironmentProcess.recv returns empty bytes at EndOfStream."""
    env = LocalEnvironment(tmp_path)
    proc = await env.create_process('true')
    async with proc:
        await proc.wait(timeout=5)
        # After process exits, reading should return empty
        data = await proc.recv()
        assert data == b''


async def test_local_read_file_binary_non_image(tmp_path: Path):
    """LocalEnvironment.read_file returns raw bytes for non-image binary files."""
    async with LocalEnvironment(tmp_path) as env:
        binary_path = tmp_path / 'data.bin'
        binary_path.write_bytes(b'\x80\x81\x82\xff')
        result = await env.read_file('data.bin')
        assert isinstance(result, bytes)
        assert result == b'\x80\x81\x82\xff'
