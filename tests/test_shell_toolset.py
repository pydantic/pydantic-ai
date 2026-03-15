"""Tests for ShellToolset, TextEditorToolset, ApplyPatchToolset, and NativeToolDefinition types."""

from __future__ import annotations

import json
import warnings
from typing import Any
from unittest.mock import AsyncMock, Mock

import anyio
import pytest

from pydantic_ai.exceptions import ModelRetry
from pydantic_ai.models import warn_native_tool_fallback
from pydantic_ai.tools import (
    ApplyPatchNativeDefinition,
    NativeToolDefinition,
    ShellNativeDefinition,
    TextEditorNativeDefinition,
    ToolDefinition,
)
from pydantic_ai.toolsets.apply_patch import ApplyPatchOperation, ApplyPatchOutput, ApplyPatchToolset
from pydantic_ai.toolsets.shell import (
    ShellExecutor,
    ShellOutput,
    ShellToolset,
)
from pydantic_ai.toolsets.text_editor import TextEditorOutput, TextEditorToolset

from .conftest import try_import

with try_import() as anthropic_installed:
    pass

with try_import() as openai_installed:
    pass

# === NativeToolDefinition types ===


def test_shell_native_definition():
    defn = ShellNativeDefinition()
    assert defn.kind == 'shell'


def test_text_editor_native_definition():
    defn = TextEditorNativeDefinition()
    assert defn.kind == 'text_editor'
    assert defn.max_characters is None

    defn_with_max = TextEditorNativeDefinition(max_characters=10000)
    assert defn_with_max.max_characters == 10000


def test_apply_patch_native_definition():
    defn = ApplyPatchNativeDefinition()
    assert defn.kind == 'apply_patch'


def test_tool_definition_native_field():
    """ToolDefinition has native_definition field, defaulting to None."""
    td = ToolDefinition(name='test')
    assert td.native_definition is None

    td_with_native = ToolDefinition(
        name='test_shell',
        native_definition=ShellNativeDefinition(),
    )
    assert td_with_native.native_definition is not None
    assert isinstance(td_with_native.native_definition, ShellNativeDefinition)


def test_native_tool_definition_union():
    """NativeToolDefinition is a union of all native definitions."""
    # Just verify the types are assignable
    defns: list[NativeToolDefinition] = [
        ShellNativeDefinition(),
        TextEditorNativeDefinition(),
        ApplyPatchNativeDefinition(),
    ]
    assert len(defns) == 3


# === ShellToolset ===


@pytest.fixture
def mock_executor() -> ShellExecutor:
    executor = AsyncMock(spec=ShellExecutor)
    executor.execute = AsyncMock(return_value=ShellOutput(output='hello world', exit_code=0))
    executor.restart = AsyncMock(return_value=ShellOutput(output='Shell session restarted.', exit_code=0))
    executor.close = AsyncMock()
    return executor


class TestShellToolsetGetTools:
    async def test_get_tools_returns_correct_definition(self, mock_executor: ShellExecutor):
        toolset: ShellToolset[None] = ShellToolset(executor=mock_executor)
        from pydantic_ai import RunContext

        ctx = RunContext[None](
            deps=None,
            model=None,  # type: ignore
            usage=None,  # type: ignore
            prompt='test',
            retries={},
        )
        tools = await toolset.get_tools(ctx)

        assert 'shell' in tools
        tool_def = tools['shell'].tool_def
        assert tool_def.name == 'shell'
        assert tool_def.sequential is True
        assert isinstance(tool_def.native_definition, ShellNativeDefinition)
        assert tool_def.parameters_json_schema['type'] == 'object'
        assert 'command' in tool_def.parameters_json_schema['properties']
        assert 'restart' in tool_def.parameters_json_schema['properties']

    async def test_get_tools_custom_name(self, mock_executor: ShellExecutor):
        toolset: ShellToolset[None] = ShellToolset(executor=mock_executor, tool_name='my_bash')
        from pydantic_ai import RunContext

        ctx = RunContext[None](
            deps=None,
            model=None,  # type: ignore
            usage=None,  # type: ignore
            prompt='test',
            retries={},
        )
        tools = await toolset.get_tools(ctx)
        assert 'my_bash' in tools
        assert tools['my_bash'].tool_def.name == 'my_bash'


class TestShellToolsetCallTool:
    async def test_call_tool_execute(self, mock_executor: ShellExecutor):
        toolset: ShellToolset[None] = ShellToolset(executor=mock_executor)
        from pydantic_ai import RunContext

        ctx = RunContext[None](
            deps=None,
            model=None,  # type: ignore
            usage=None,  # type: ignore
            prompt='test',
            retries={},
        )
        tools = await toolset.get_tools(ctx)
        tool = tools['shell']

        result = await toolset.call_tool('shell', {'command': 'echo hello'}, ctx, tool)
        parsed = json.loads(result)
        assert parsed['output'] == 'hello world'
        assert parsed['exit_code'] == 0
        mock_executor.execute.assert_called_once_with('echo hello')  # type: ignore

    async def test_call_tool_restart(self, mock_executor: ShellExecutor):
        toolset: ShellToolset[None] = ShellToolset(executor=mock_executor)
        from pydantic_ai import RunContext

        ctx = RunContext[None](
            deps=None,
            model=None,  # type: ignore
            usage=None,  # type: ignore
            prompt='test',
            retries={},
        )
        tools = await toolset.get_tools(ctx)
        tool = tools['shell']

        result = await toolset.call_tool('shell', {'command': '', 'restart': True}, ctx, tool)
        parsed = json.loads(result)
        assert parsed['output'] == 'Shell session restarted.'
        mock_executor.restart.assert_called_once()  # type: ignore

    async def test_call_tool_truncation(self, mock_executor: ShellExecutor):
        long_output = 'x' * 300_000
        mock_executor.execute = AsyncMock(return_value=ShellOutput(output=long_output, exit_code=0))

        toolset: ShellToolset[None] = ShellToolset(executor=mock_executor, max_output_chars=100)
        from pydantic_ai import RunContext

        ctx = RunContext[None](
            deps=None,
            model=None,  # type: ignore
            usage=None,  # type: ignore
            prompt='test',
            retries={},
        )
        tools = await toolset.get_tools(ctx)
        tool = tools['shell']

        result = await toolset.call_tool('shell', {'command': 'big'}, ctx, tool)
        parsed = json.loads(result)
        assert '[output truncated' in parsed['output']
        assert '100 chars of 300000 total' in parsed['output']

    async def test_call_tool_timeout(self):
        import asyncio

        class SlowExecutor:
            async def execute(self, command: str) -> ShellOutput:
                await asyncio.sleep(10)
                return ShellOutput(output='', exit_code=0)  # pragma: no cover

            async def restart(self) -> ShellOutput:
                return ShellOutput(output='Shell session restarted.', exit_code=0)

            async def close(self) -> None:
                pass

        toolset: ShellToolset[None] = ShellToolset(executor=SlowExecutor(), timeout=0.01)
        from pydantic_ai import RunContext

        ctx = RunContext[None](
            deps=None,
            model=None,  # type: ignore
            usage=None,  # type: ignore
            prompt='test',
            retries={},
        )
        tools = await toolset.get_tools(ctx)
        tool = tools['shell']

        with pytest.raises(ModelRetry, match='timed out'):
            await toolset.call_tool('shell', {'command': 'slow'}, ctx, tool)


class TestShellToolsetLocal:
    def test_local_factory(self):
        toolset: ShellToolset[None] = ShellToolset.local(cwd='/tmp')
        # Verify the executor satisfies the ShellExecutor protocol
        assert callable(toolset.executor.execute)
        assert callable(toolset.executor.restart)
        assert callable(toolset.executor.close)
        assert toolset.id is None

    async def test_local_executor_basic(self):
        executor = ShellToolset.local().executor
        try:
            with anyio.fail_after(10):
                result = await executor.execute('echo hello')
            assert 'hello' in result.output
            assert result.exit_code == 0
        finally:
            await executor.close()

    async def test_local_executor_restart(self):
        executor = ShellToolset.local().executor
        try:
            with anyio.fail_after(10):
                result = await executor.restart()
            assert 'restarted' in result.output.lower()
        finally:
            await executor.close()

    async def test_local_executor_nonzero_exit(self):
        executor = ShellToolset.local().executor
        try:
            with anyio.fail_after(10):
                result = await executor.execute('exit 42')
            assert result.exit_code == 42
        finally:
            await executor.close()

    async def test_local_executor_no_trailing_newline(self):
        executor = ShellToolset.local().executor
        try:
            with anyio.fail_after(10):
                result = await executor.execute('printf "no newline"')
            assert 'no newline' in result.output
        finally:
            await executor.close()

    async def test_local_executor_unexpected_eof_returns_process_exit_code(self, monkeypatch: pytest.MonkeyPatch):
        class FakeStdin:
            def __init__(self) -> None:
                self.written = b''

            def write(self, data: bytes) -> None:
                self.written += data

            async def drain(self) -> None:
                return None

        class FakeStdout:
            async def readline(self) -> bytes:
                return b''

        class FakeProcess:
            def __init__(self) -> None:
                self.stdin = FakeStdin()
                self.stdout = FakeStdout()
                self.returncode: int | None = None
                self.wait_called = False

            async def wait(self) -> int:
                self.wait_called = True
                self.returncode = 17
                return 17

        executor = ShellToolset.local().executor
        process = FakeProcess()

        async def ensure_process() -> FakeProcess:
            return process

        monkeypatch.setattr(executor, '_ensure_process', ensure_process)

        result = await executor.execute('exit 17')
        assert result == ShellOutput(output='', exit_code=17)
        assert process.stdin.written.startswith(b'exit 17\n')
        assert process.wait_called is True

    async def test_local_executor_missing_exit_code_uses_default(self, monkeypatch: pytest.MonkeyPatch):
        class FakeStdin:
            def write(self, data: bytes) -> None:
                self.written = data

            async def drain(self) -> None:
                return None

        class FakeStdout:
            def __init__(self, sentinel: str) -> None:
                self._lines = [f'{sentinel}\n'.encode()]

            async def readline(self) -> bytes:
                return self._lines.pop(0) if self._lines else b''

        class FakeProcess:
            def __init__(self, sentinel: str) -> None:
                self.stdin = FakeStdin()
                self.stdout = FakeStdout(sentinel)
                self.returncode = None

        executor = ShellToolset.local().executor
        sentinel = f'__PYDANTIC_AI_EXIT_{id(executor)}__'
        process = FakeProcess(sentinel)

        async def ensure_process() -> FakeProcess:
            return process

        monkeypatch.setattr(executor, '_ensure_process', ensure_process)

        result = await executor.execute('echo hello')
        assert result == ShellOutput(output='', exit_code=0)

    async def test_local_executor_state_persists(self):
        """cd and export persist across execute calls."""
        executor = ShellToolset.local().executor
        try:
            with anyio.fail_after(10):
                await executor.execute('export MY_TEST_VAR=hello123')
                result = await executor.execute('echo $MY_TEST_VAR')
            assert 'hello123' in result.output
        finally:
            await executor.close()

    async def test_aexit_closes_executor(self, mock_executor: ShellExecutor):
        toolset: ShellToolset[None] = ShellToolset(executor=mock_executor)
        await toolset.__aexit__()
        mock_executor.close.assert_called_once()  # type: ignore

    async def test_local_executor_close_kills_process_after_timeout(self, monkeypatch: pytest.MonkeyPatch):
        import asyncio

        loop = asyncio.get_running_loop()
        process = Mock()
        process.returncode = None
        # First wait() call hangs (used by wait_for timeout); subsequent calls resolve immediately
        hang_future = loop.create_future()
        call_count = 0

        def mock_wait() -> Any:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return hang_future
            # After kill(), wait() should resolve immediately
            done_future = loop.create_future()
            done_future.set_result(None)
            return done_future

        process.wait = mock_wait
        process.terminate = Mock()
        process.kill = Mock()

        executor = ShellToolset.local().executor
        executor._process = process  # pyright: ignore[reportAttributeAccessIssue]

        original_wait_for = asyncio.wait_for

        async def timeout_wait_for(awaitable: Any, *args: Any, **kwargs: Any) -> Any:
            return await original_wait_for(awaitable, timeout=0)

        monkeypatch.setattr('pydantic_ai.toolsets.shell.asyncio.wait_for', timeout_wait_for)

        await executor.close()

        process.terminate.assert_called_once()
        process.kill.assert_called_once()
        assert executor._process is None  # pyright: ignore[reportAttributeAccessIssue]


@pytest.mark.skipif(not anthropic_installed(), reason='anthropic not installed')
class TestFallbackWarning:
    def test_native_tool_fallback_warns(self):
        """Native tool on unsupported provider emits a fallback warning."""
        # Use unique provider name to avoid parallel test interference
        provider = 'test_fallback_shell'

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            warn_native_tool_fallback('shell', provider)
        assert len(w) == 1
        assert 'falling back to function tool format' in str(w[0].message)
        assert provider in str(w[0].message)

        # Second call should not warn (once per kind+provider)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            warn_native_tool_fallback('shell', provider)
        assert len(w) == 0


# === TextEditorToolset ===


def _make_run_context():
    from pydantic_ai import RunContext

    return RunContext[None](
        deps=None,
        model=None,  # type: ignore
        usage=None,  # type: ignore
        prompt='test',
        retries={},
    )


class TestTextEditorToolsetGetTools:
    async def test_get_tools_returns_correct_definition(self):
        mock_execute = AsyncMock(return_value=TextEditorOutput(output='OK'))
        toolset: TextEditorToolset[None] = TextEditorToolset(execute=mock_execute)
        assert toolset.id is None
        ctx = _make_run_context()
        tools = await toolset.get_tools(ctx)

        assert 'str_replace_based_edit_tool' in tools
        tool_def = tools['str_replace_based_edit_tool'].tool_def
        assert tool_def.name == 'str_replace_based_edit_tool'
        assert isinstance(tool_def.native_definition, TextEditorNativeDefinition)
        assert tool_def.native_definition.max_characters is None

    async def test_get_tools_with_max_characters(self):
        mock_execute = AsyncMock(return_value=TextEditorOutput(output='OK'))
        toolset: TextEditorToolset[None] = TextEditorToolset(execute=mock_execute, max_characters=50000)
        ctx = _make_run_context()
        tools = await toolset.get_tools(ctx)

        tool_def = tools['str_replace_based_edit_tool'].tool_def
        assert isinstance(tool_def.native_definition, TextEditorNativeDefinition)
        assert tool_def.native_definition.max_characters == 50000

    async def test_get_tools_custom_name(self):
        mock_execute = AsyncMock(return_value=TextEditorOutput(output='OK'))
        toolset: TextEditorToolset[None] = TextEditorToolset(execute=mock_execute, tool_name='my_editor')
        ctx = _make_run_context()
        tools = await toolset.get_tools(ctx)
        assert 'my_editor' in tools
        assert tools['my_editor'].tool_def.name == 'my_editor'


class TestTextEditorToolsetCallTool:
    async def test_call_tool_dispatches_to_execute(self):
        received_cmd: Any = None

        async def mock_execute(cmd: Any) -> TextEditorOutput:
            nonlocal received_cmd
            received_cmd = cmd
            return TextEditorOutput(output='file contents here')

        toolset: TextEditorToolset[None] = TextEditorToolset(execute=mock_execute)
        ctx = _make_run_context()
        tools = await toolset.get_tools(ctx)
        tool = tools['str_replace_based_edit_tool']

        result = await toolset.call_tool(
            'str_replace_based_edit_tool',
            {'command': 'view', 'path': '/tmp/test.py'},
            ctx,
            tool,
        )
        parsed = json.loads(result)
        assert parsed['output'] == 'file contents here'
        assert parsed['success'] is True
        assert received_cmd is not None
        assert received_cmd['command'] == 'view'
        assert received_cmd['path'] == '/tmp/test.py'

    async def test_call_tool_failure(self):
        async def mock_execute(cmd: Any) -> TextEditorOutput:
            return TextEditorOutput(output='File not found', success=False)

        toolset: TextEditorToolset[None] = TextEditorToolset(execute=mock_execute)
        ctx = _make_run_context()
        tools = await toolset.get_tools(ctx)
        tool = tools['str_replace_based_edit_tool']

        result = await toolset.call_tool(
            'str_replace_based_edit_tool',
            {'command': 'view', 'path': '/nonexistent'},
            ctx,
            tool,
        )
        parsed = json.loads(result)
        assert parsed['output'] == 'File not found'
        assert parsed['success'] is False


# === ApplyPatchToolset ===


class TestApplyPatchToolsetGetTools:
    async def test_get_tools_returns_correct_definition(self):
        mock_execute = AsyncMock(return_value=ApplyPatchOutput(status='completed'))
        toolset: ApplyPatchToolset[None] = ApplyPatchToolset(execute=mock_execute)
        assert toolset.id is None
        ctx = _make_run_context()
        tools = await toolset.get_tools(ctx)

        assert 'apply_patch' in tools
        tool_def = tools['apply_patch'].tool_def
        assert tool_def.name == 'apply_patch'
        assert isinstance(tool_def.native_definition, ApplyPatchNativeDefinition)

    async def test_get_tools_custom_name(self):
        mock_execute = AsyncMock(return_value=ApplyPatchOutput(status='completed'))
        toolset: ApplyPatchToolset[None] = ApplyPatchToolset(execute=mock_execute, tool_name='my_patcher')
        ctx = _make_run_context()
        tools = await toolset.get_tools(ctx)
        assert 'my_patcher' in tools
        assert tools['my_patcher'].tool_def.name == 'my_patcher'


class TestApplyPatchToolsetCallTool:
    async def test_call_tool_dispatches_to_execute(self):
        received_op: Any = None

        async def mock_execute(op: Any) -> ApplyPatchOutput:
            nonlocal received_op
            received_op = op
            return ApplyPatchOutput(status='completed', output='Patch applied')

        toolset: ApplyPatchToolset[None] = ApplyPatchToolset(execute=mock_execute)
        ctx = _make_run_context()
        tools = await toolset.get_tools(ctx)
        tool = tools['apply_patch']

        result = await toolset.call_tool(
            'apply_patch',
            {'operation_type': 'update_file', 'path': '/tmp/test.py', 'diff': '--- a\n+++ b\n@@ ...\n-old\n+new'},
            ctx,
            tool,
        )
        parsed = json.loads(result)
        assert parsed['status'] == 'completed'
        assert parsed['output'] == 'Patch applied'
        assert isinstance(received_op, ApplyPatchOperation)
        assert received_op.operation_type == 'update_file'
        assert received_op.path == '/tmp/test.py'
        assert received_op.diff is not None

    async def test_call_tool_create_file(self):
        async def mock_execute(op: Any) -> ApplyPatchOutput:
            return ApplyPatchOutput(status='completed')

        toolset: ApplyPatchToolset[None] = ApplyPatchToolset(execute=mock_execute)
        ctx = _make_run_context()
        tools = await toolset.get_tools(ctx)
        tool = tools['apply_patch']

        result = await toolset.call_tool(
            'apply_patch',
            {'operation_type': 'create_file', 'path': '/tmp/new.py', 'content': 'print("hello")'},
            ctx,
            tool,
        )
        parsed = json.loads(result)
        assert parsed['status'] == 'completed'

    async def test_call_tool_failure(self):
        async def mock_execute(op: Any) -> ApplyPatchOutput:
            return ApplyPatchOutput(status='failed', output='Patch conflict')

        toolset: ApplyPatchToolset[None] = ApplyPatchToolset(execute=mock_execute)
        ctx = _make_run_context()
        tools = await toolset.get_tools(ctx)
        tool = tools['apply_patch']

        result = await toolset.call_tool(
            'apply_patch',
            {'operation_type': 'delete_file', 'path': '/tmp/gone.py'},
            ctx,
            tool,
        )
        parsed = json.loads(result)
        assert parsed['status'] == 'failed'
        assert parsed['output'] == 'Patch conflict'


# === Fallback warnings for new native tool types ===


class TestFallbackWarningNewTypes:
    @pytest.mark.skipif(not anthropic_installed(), reason='anthropic not installed')
    def test_text_editor_fallback_warns(self):
        """Text editor native tool on unsupported provider emits a fallback warning."""
        provider = 'test_fallback_text_editor'

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            warn_native_tool_fallback('text_editor', provider)
        assert len(w) == 1
        assert 'text_editor' in str(w[0].message)
        assert provider in str(w[0].message)

    @pytest.mark.skipif(not openai_installed(), reason='openai not installed')
    def test_apply_patch_fallback_warns(self):
        """Apply patch native tool on unsupported provider emits a fallback warning."""
        provider = 'test_fallback_apply_patch'

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            warn_native_tool_fallback('apply_patch', provider)
        assert len(w) == 1
        assert 'apply_patch' in str(w[0].message)
        assert provider in str(w[0].message)


# === id property coverage ===


def test_shell_toolset_id():
    toolset: ShellToolset[None] = ShellToolset.local()
    assert toolset.id is None


async def test_text_editor_toolset_id():
    toolset: TextEditorToolset[None] = TextEditorToolset(execute=AsyncMock(return_value=TextEditorOutput(output='')))
    assert toolset.id is None


async def test_apply_patch_toolset_id():
    toolset: ApplyPatchToolset[None] = ApplyPatchToolset(
        execute=AsyncMock(return_value=ApplyPatchOutput(status='completed'))
    )
    assert toolset.id is None


# === Shell executor edge cases ===


async def test_local_executor_corrupt_sentinel():
    """Cover ValueError branch when exit code can't be parsed."""
    executor = ShellToolset.local().executor
    try:
        with anyio.fail_after(10):
            # echo a line that looks like a sentinel but has a non-integer exit code
            sentinel = f'__PYDANTIC_AI_EXIT_{id(executor)}__'
            result = await executor.execute(f'echo "{sentinel} notanumber"')
            # The sentinel will be detected but parseInt will fail -> exit_code=1
            # However, the REAL sentinel from our wrapper will come after
            # Just verify execution completes without hanging
            assert isinstance(result.exit_code, int)
    finally:
        await executor.close()
