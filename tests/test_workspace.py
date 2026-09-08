"""Tests for the workspace interface and its lazy backend contract."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import anyio
import pytest
from pydantic import TypeAdapter

from pydantic_ai import Agent, RunContext, UserError
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.durable_exec._workspace import guard_workflow_workspace
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.run import AgentRunResult
from pydantic_ai.usage import RunUsage
from pydantic_ai.workspaces import (
    FileWindow,
    LocalWorkspace,
    ReadOnlyWorkspace,
    UnavailableWorkspace,
    Workspace,
    WorkspaceBackend,
    WorkspaceError,
    WorkspaceRef,
    WorkspaceTimeoutError,
    WrapperWorkspace,
)

from .workspace_fakes import (
    ConnectOnlyWorkspaceCapability,
    DecliningWorkspaceCapability,
    FakeEntry,
    FakeWorkspace,
    FakeWorkspaceResult,
    RunOnlyWorkspaceBackend,
    WorkspaceCapability,
)

pytestmark = pytest.mark.anyio


async def test_wrapper_overrides_apply_to_text_and_window_reads():
    backend = FakeWorkspace('wrapper', {'/workspace/file.txt': b'inner'})

    class ReadingWrapper(WrapperWorkspace):
        async def read_bytes(self, path: str) -> bytes:
            return b'outer\nvalue\n'

    workspace = ReadingWrapper(Workspace(backend))
    assert await workspace.read_text('file.txt') == 'outer\nvalue\n'
    assert (await workspace.read_file('file.txt', limit=1)).lines == ('outer',)


async def test_wrapper_overrides_apply_to_text_writes():
    backend = FakeWorkspace('wrapper')
    writes: list[tuple[str, bytes]] = []

    class WritingWrapper(WrapperWorkspace):
        async def write_bytes(self, path: str, data: bytes) -> None:
            writes.append((path, data))

    workspace = WritingWrapper(Workspace(backend))
    await workspace.write_text('file.txt', 'outer')
    assert writes == [('file.txt', b'outer')]
    assert backend.files == {}


async def test_stacked_wrappers_preserve_delegation_identity_and_refs():
    ref = WorkspaceRef(provider='fake', id='stacked')
    backend = FakeWorkspace('wrapper', {'/workspace/file.txt': b'inner'}, ref=ref)
    inner = WrapperWorkspace(Workspace(backend))
    outer = WrapperWorkspace(inner)

    assert outer.wrapped is inner
    assert inner.wrapped.backend is backend
    assert outer.backend is inner
    assert outer.ref == ref

    events: list[str] = []

    class LoggedWorkspace(WrapperWorkspace):
        def __init__(self, wrapped: Workspace, name: str):
            super().__init__(wrapped)
            self.name = name

        async def read_bytes(self, path: str) -> bytes:
            events.append(f'{self.name} before')
            data = await self.wrapped.read_bytes(path)
            events.append(f'{self.name} after')
            return data

    inner_logged = LoggedWorkspace(Workspace(backend), 'inner')
    outer_logged = LoggedWorkspace(inner_logged, 'outer')
    assert events == []
    assert await outer_logged.read_text('file.txt') == 'inner'
    assert events == ['outer before', 'inner before', 'inner after', 'outer after']


def _tool_call_model(tool_name: str = 'probe') -> FunctionModel:
    def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(parts=[ToolCallPart(tool_name, {})])
        return ModelResponse(parts=[TextPart('done')])

    return FunctionModel(model)


async def test_resolve_normalizes_paths_against_the_working_directory() -> None:
    workspace = Workspace(FakeWorkspace('resolve'))

    assert await workspace.resolve('sub/../notes.txt') == '/workspace/notes.txt'
    assert await workspace.resolve('/abs/./x') == '/abs/x'
    assert await workspace.resolve('x', base='/elsewhere') == '/elsewhere/x'


async def test_resolve_rejects_a_relative_base() -> None:
    with pytest.raises(ValueError, match="base must be an absolute path, got 'relative'"):
        await Workspace(FakeWorkspace('resolve')).resolve('file.txt', base='relative')


async def test_flat_file_operations_use_the_backend_filesystem() -> None:
    backend = FakeWorkspace('files', {'/workspace/data.txt': b'hello'})
    workspace = Workspace(backend)

    assert await workspace.read_bytes('data.txt') == b'hello'
    assert (await workspace.stat('data.txt')).path == '/workspace/data.txt'
    assert await workspace.exists('data.txt')
    assert (await workspace.list_dir('.'))[0].path == '/workspace/data.txt'
    await workspace.make_dir('new-dir')
    await workspace.write_bytes('new.txt', b'new')
    await workspace.write_text('data.txt', 'updated')
    await workspace.remove('new.txt')

    assert backend.files['/workspace/data.txt'] == b'updated'
    assert not await workspace.exists('new.txt')


async def test_text_helpers_resolve_relative_paths() -> None:
    backend = FakeWorkspace('text', {'/workspace/data.txt': b'old'})
    workspace = Workspace(backend)

    await workspace.write_text('data.txt', 'updated')

    assert await workspace.read_text('data.txt') == 'updated'
    assert backend.files['/workspace/data.txt'] == b'updated'


async def test_run_only_backend_supports_bounded_reads_through_shell() -> None:
    inner = FakeWorkspace('run-only', {'/workspace/data.txt': b'one\ntwo\nthree\n'})
    commands: list[str | Sequence[str]] = []

    class RunOnlyBackend:
        @property
        def ref(self) -> WorkspaceRef | None:
            return inner.ref

        async def run(
            self,
            command: str | Sequence[str],
            *,
            shell: bool = False,
            cwd: str | None = None,
            env: Mapping[str, str] | None = None,
            timeout: float | None = None,
        ) -> FakeWorkspaceResult:
            commands.append(command)
            return await inner.run(command, shell=shell, cwd=cwd, env=env, timeout=timeout)

        async def working_dir(self) -> str:
            return await inner.working_dir()

    workspace = Workspace(RunOnlyBackend())
    assert workspace.ref is None
    window = await workspace.read_file('data.txt', limit=2)

    assert window.lines == ('one', 'two')
    assert commands == [['sed', '-n', '1,3p;3q', '/workspace/data.txt']]
    assert inner.reads == []
    with pytest.raises(WorkspaceError, match='invalid base64'):
        await workspace.read_file('data.txt')


@pytest.mark.parametrize(
    'result',
    [
        pytest.param(FakeWorkspaceResult(exit_code=127, stderr='sed: not found'), id='no-sed'),
        pytest.param(FakeWorkspaceResult(exit_code=2), id='nonzero'),
        pytest.param(FakeWorkspaceResult(stderr='warning'), id='stderr'),
    ],
)
async def test_bounded_read_shell_failures_fall_back_to_filesystem(result: FakeWorkspaceResult) -> None:
    class FailedSed(FakeWorkspace):
        async def run(
            self,
            command: str | Sequence[str],
            *,
            shell: bool = False,
            cwd: str | None = None,
            env: Mapping[str, str] | None = None,
            timeout: float | None = None,
        ) -> FakeWorkspaceResult:
            if not isinstance(command, str) and list(command[:2]) == ['sed', '-n']:
                return result
            return await super().run(command, shell=shell, cwd=cwd, env=env, timeout=timeout)

    backend = FailedSed('failed-sed', {'/workspace/data.txt': b'one\ntwo\nthree\n'})

    workspace = Workspace(backend)
    assert workspace.ref == backend.ref
    # Only the `sed` slice is broken here; ordinary commands still run.
    assert (await workspace.run(['true'])).stdout == 'connected'
    window = await workspace.read_file('data.txt', offset=2, limit=1)

    assert window.lines == ('two',)
    assert window.has_more is True
    assert window.total_lines == 3
    assert backend.reads == ['/workspace/data.txt']


async def test_bounded_read_falls_back_to_the_shell_filesystem_when_sed_is_missing(tmp_path: Path) -> None:
    path = tmp_path / 'data.txt'
    path.write_text('one\ntwo\nthree\n')

    class NoSedBackend(RunOnlyWorkspaceBackend):
        async def run(
            self,
            command: str | Sequence[str],
            *,
            shell: bool = False,
            cwd: str | None = None,
            env: Mapping[str, str] | None = None,
            timeout: float | None = None,
        ) -> FakeWorkspaceResult:
            if not isinstance(command, str) and list(command[:2]) == ['sed', '-n']:
                return FakeWorkspaceResult(exit_code=127, stderr='sed: not found')
            result = await super().run(command, shell=shell, cwd=cwd, env=env, timeout=timeout)
            return FakeWorkspaceResult(exit_code=result.exit_code, stdout=result.stdout, stderr=result.stderr)

    window = await Workspace(NoSedBackend(LocalWorkspace(tmp_path))).read_file('data.txt', offset=2, limit=1)

    assert (window.lines, window.has_more, window.total_lines) == (('two',), True, 3)


async def test_native_filesystem_fallback_is_used_when_fake_sed_is_unavailable() -> None:
    backend = FakeWorkspace('no-sed', {'/workspace/data.txt': b'one\ntwo\n'}, sed=False)

    window = await Workspace(backend).read_file('data.txt', limit=1)

    assert window.lines == ('one',)
    assert backend.reads == ['/workspace/data.txt']


async def test_run_only_backend_has_a_complete_binary_safe_shell_filesystem(tmp_path: Path) -> None:
    backend = RunOnlyWorkspaceBackend(LocalWorkspace(tmp_path))
    workspace = Workspace(backend)
    payload = bytes(range(256)) * 800
    filename = "nested/weird '\n blob.bin"

    assert backend.ref is None
    await workspace.write_bytes(filename, payload)

    assert await workspace.read_bytes(filename) == payload
    assert (await workspace.stat(filename)).size == len(payload)
    assert await workspace.exists(filename)
    entries = await workspace.list_dir('nested')
    assert [(entry.name, entry.is_dir) for entry in entries] == [("weird '\n blob.bin", False)]
    # The encoded write is chunked below Linux's independent per-argument limit.
    assert max(len(command.encode()) for command in backend.commands if isinstance(command, str)) < 128 * 1024

    await workspace.make_dir('nested/directory')
    assert (await workspace.stat('nested/directory')).is_dir
    assert await workspace.exists('nested/directory')
    await workspace.remove('nested')
    assert not await workspace.exists(filename)
    with pytest.raises(FileNotFoundError):
        await workspace.read_bytes(filename)


@pytest.mark.parametrize('cleanup_fails', [False, True])
async def test_shell_write_preserves_the_original_error_when_cleanup_fails(tmp_path: Path, cleanup_fails: bool) -> None:
    cleanup_attempted = False

    class FailedWriteBackend(RunOnlyWorkspaceBackend):
        async def run(
            self,
            command: str | Sequence[str],
            *,
            shell: bool = False,
            cwd: str | None = None,
            env: Mapping[str, str] | None = None,
            timeout: float | None = None,
        ) -> FakeWorkspaceResult:
            nonlocal cleanup_attempted
            if isinstance(command, str) and command.startswith('rm -f '):
                cleanup_attempted = True
                if cleanup_fails:
                    raise RuntimeError('cleanup failed')
            if isinstance(command, str) and 'base64 -d' in command:
                raise RuntimeError('write failed')
            result = await super().run(command, shell=shell, cwd=cwd, env=env, timeout=timeout)
            return FakeWorkspaceResult(exit_code=result.exit_code, stdout=result.stdout, stderr=result.stderr)

    with pytest.raises(RuntimeError, match='write failed'):
        await Workspace(FailedWriteBackend(LocalWorkspace(tmp_path))).write_bytes('data.bin', b'data')

    assert cleanup_attempted


async def test_shell_stat_rejects_an_invalid_size(tmp_path: Path) -> None:
    class InvalidStatBackend(RunOnlyWorkspaceBackend):
        async def run(
            self,
            command: str | Sequence[str],
            *,
            shell: bool = False,
            cwd: str | None = None,
            env: Mapping[str, str] | None = None,
            timeout: float | None = None,
        ) -> FakeWorkspaceResult:
            if isinstance(command, str) and 'wc -c' in command:
                return FakeWorkspaceResult(stdout='not-a-size')
            result = await super().run(command, shell=shell, cwd=cwd, env=env, timeout=timeout)
            return FakeWorkspaceResult(exit_code=result.exit_code, stdout=result.stdout, stderr=result.stderr)

    workspace = Workspace(InvalidStatBackend(LocalWorkspace(tmp_path)))
    await workspace.write_bytes('data.bin', b'data')
    with pytest.raises(WorkspaceError, match='invalid size'):
        await workspace.stat('data.bin')


async def test_shell_list_dir_rejects_invalid_encoded_output(tmp_path: Path) -> None:
    class InvalidListingBackend(RunOnlyWorkspaceBackend):
        async def run(
            self,
            command: str | Sequence[str],
            *,
            shell: bool = False,
            cwd: str | None = None,
            env: Mapping[str, str] | None = None,
            timeout: float | None = None,
        ) -> FakeWorkspaceResult:
            if isinstance(command, str) and 'find ' in command:
                return FakeWorkspaceResult(stdout='/w==')
            result = await super().run(command, shell=shell, cwd=cwd, env=env, timeout=timeout)
            return FakeWorkspaceResult(exit_code=result.exit_code, stdout=result.stdout, stderr=result.stderr)

    workspace = Workspace(InvalidListingBackend(LocalWorkspace(tmp_path)))
    await workspace.make_dir('directory')
    with pytest.raises(WorkspaceError, match='invalid directory listing'):
        await workspace.list_dir('.')


async def test_empty_shell_window_tolerates_a_backend_without_stat() -> None:
    class NoStatBackend(FakeWorkspace):
        async def stat(self, path: str) -> FakeEntry:
            raise NotImplementedError

        async def run(
            self,
            command: str | Sequence[str],
            *,
            shell: bool = False,
            cwd: str | None = None,
            env: Mapping[str, str] | None = None,
            timeout: float | None = None,
        ) -> FakeWorkspaceResult:
            return FakeWorkspaceResult(stdout='')

    window = await Workspace(NoStatBackend('no-stat')).read_file('empty.txt', limit=1)

    assert window == FileWindow(lines=(), start_line=1, has_more=False, total_lines=None)


async def test_shell_list_dir_does_not_hide_find_failure(tmp_path: Path) -> None:
    class FailedFindBackend(RunOnlyWorkspaceBackend):
        async def run(
            self,
            command: str | Sequence[str],
            *,
            shell: bool = False,
            cwd: str | None = None,
            env: Mapping[str, str] | None = None,
            timeout: float | None = None,
        ) -> FakeWorkspaceResult:
            if isinstance(command, str) and 'find ' in command:
                command = command.replace('find ', 'false ', 1)
            result = await super().run(command, shell=shell, cwd=cwd, env=env, timeout=timeout)
            return FakeWorkspaceResult(exit_code=result.exit_code, stdout=result.stdout, stderr=result.stderr)

    with pytest.raises(WorkspaceError):
        await Workspace(FailedFindBackend(LocalWorkspace(tmp_path))).list_dir('.')


async def test_slice_timeout_falls_back_to_filesystem() -> None:
    class TimedOutSed(FakeWorkspace):
        async def run(
            self,
            command: str | Sequence[str],
            *,
            shell: bool = False,
            cwd: str | None = None,
            env: Mapping[str, str] | None = None,
            timeout: float | None = None,
        ) -> FakeWorkspaceResult:
            raise WorkspaceTimeoutError('sed timed out', timeout=timeout)

    backend = TimedOutSed('timed-out-sed', {'/workspace/data.txt': b'one\ntwo\nthree\n'})

    window = await Workspace(backend).read_file('data.txt', offset=2, limit=1)

    assert window.lines == ('two',)
    assert backend.reads == ['/workspace/data.txt']


async def test_a_file_without_a_trailing_newline_reads_to_its_last_line() -> None:
    """The last line still counts, and the window still knows it reached the end."""
    backend = FakeWorkspace('no-trailing-newline', {'/workspace/data.txt': b'one\ntwo'})

    window = await Workspace(backend).read_file('data.txt', limit=5)

    assert window.lines == ('one', 'two')
    assert window.has_more is False
    assert window.total_lines == 2


@pytest.mark.parametrize('kwargs', [{'offset': 0}, {'limit': 0}])
async def test_read_file_rejects_invalid_window_values(kwargs: dict[str, int]) -> None:
    with pytest.raises(ValueError):
        await Workspace(FakeWorkspace('invalid-window')).read_file('data.txt', **kwargs)


@pytest.mark.parametrize('offset', [1, 4])
async def test_bounded_read_returns_empty_window_at_or_past_empty_file(offset: int) -> None:
    backend = FakeWorkspace('empty-file', {'/workspace/data.txt': b''})

    window = await Workspace(backend).read_file('data.txt', offset=offset, limit=2)

    assert (window.lines, window.start_line, window.has_more, window.total_lines) == ((), offset, False, None)


async def test_bounded_read_reports_more_lines_only_when_the_window_is_short() -> None:
    backend = FakeWorkspace('window', {'/workspace/data.txt': b'one\ntwo\nthree\n'})
    workspace = Workspace(backend)

    partial = await workspace.read_file('data.txt', offset=1, limit=2)
    ending = await workspace.read_file('data.txt', offset=2, limit=2)

    assert (partial.lines, partial.has_more, partial.total_lines) == (('one', 'two'), True, None)
    assert (ending.lines, ending.has_more, ending.total_lines) == (('two', 'three'), False, 3)


async def test_full_read_uses_filesystem_and_preserves_decoding_contracts() -> None:
    backend = FakeWorkspace('full-read', {'/workspace/data.txt': b'one\ntwo\nthree'})
    workspace = Workspace(backend)

    window = await workspace.read_file('data.txt', offset=2)

    assert (window.lines, window.has_more, window.total_lines) == (('two', 'three'), False, 3)
    assert window.text == 'two\nthree'
    assert backend.reads == ['/workspace/data.txt']

    backend.files['/workspace/bad.txt'] = b'one\xfftwo\n'
    assert (await workspace.read_file('bad.txt')).lines == ('one�two',)
    with pytest.raises(UnicodeDecodeError):
        await workspace.read_text('bad.txt')


async def test_bounded_read_through_read_only_workspace_uses_filesystem() -> None:
    backend = FakeWorkspace('read-only', {'/workspace/data.txt': b'one\ntwo\nthree\n'})
    workspace = Workspace(ReadOnlyWorkspace(Workspace(backend)))

    window = await workspace.read_file('data.txt', offset=2, limit=1)

    assert window.lines == ('two',)
    assert backend.reads == ['/workspace/data.txt']


async def test_unavailable_workspace_uses_the_configured_reason_for_every_operation() -> None:
    reason = 'workspace disabled by policy'
    backend = UnavailableWorkspace(reason)
    # No environment exists, so there is no identity a later run could reconnect to.
    assert backend.ref is None
    operations = [
        backend.run(['true']),
        backend.working_dir(),
        backend.read_bytes('/file'),
        backend.write_bytes('/file', b'data'),
        backend.stat('/file'),
        backend.list_dir('/'),
        backend.make_dir('/dir'),
        backend.remove('/file'),
        backend.exists('/file'),
    ]

    for operation in operations:
        with pytest.raises(UserError, match='workspace disabled by policy'):
            await operation


async def test_bare_run_context_workspace_explains_how_to_attach_one() -> None:
    ctx = RunContext[None](deps=None, model=TestModel(), usage=RunUsage())

    with pytest.raises(UserError, match=r'pass `workspace=`.*capability'):
        await ctx.workspace.run(['true'])


async def test_explicit_backend_wins_over_a_capability_backend() -> None:
    capability = WorkspaceCapability()
    explicit = FakeWorkspace('explicit')
    observed: list[Workspace] = []
    agent = Agent(_tool_call_model(), capabilities=[capability])

    @agent.tool
    async def probe(ctx: RunContext[Any]) -> str:
        observed.append(ctx.workspace)
        return 'ok'

    await agent.run('go', workspace=explicit)

    assert observed[0].backend is explicit
    assert capability.refs == []


async def test_missing_paths_raise_the_builtin_error_through_the_workspace() -> None:
    """The protocol promises `FileNotFoundError` for every operation that needs an existing path."""
    workspace = Workspace(FakeWorkspace('missing-paths'))

    for operation in (
        workspace.read_bytes('gone.txt'),
        workspace.stat('gone.txt'),
        workspace.remove('gone.txt'),
    ):
        with pytest.raises(FileNotFoundError):
            await operation

    # The `sed` fast path falls through to the filesystem so a missing file is not an empty window.
    with pytest.raises(FileNotFoundError):
        await workspace.read_file('gone.txt', limit=1)


async def test_workspace_is_selected_from_the_per_run_capability() -> None:
    bootstrap_backend = FakeWorkspace('bootstrap')
    run_backend = FakeWorkspace('per-run')

    class PerRunWorkspace(AbstractCapability[Any]):
        id = 'per_run_workspace'

        def __init__(self, backend: FakeWorkspace, replacement: PerRunWorkspace | None = None) -> None:
            self.backend = backend
            self.replacement = replacement

        async def for_run(self, ctx: RunContext[Any]) -> AbstractCapability[Any]:
            return self.replacement or self

        def get_workspace(self, ctx: RunContext[Any], *, ref: WorkspaceRef | None) -> WorkspaceBackend:
            return self.backend

    run_capability = PerRunWorkspace(run_backend)
    bootstrap_capability = PerRunWorkspace(bootstrap_backend, run_capability)
    result = await Agent(TestModel(), capabilities=[bootstrap_capability]).run('go')

    assert result.workspace.backend is run_backend
    assert bootstrap_backend.ref is None


async def test_the_result_carries_the_workspace_the_run_used() -> None:
    """`result.workspace` is the same object tools saw, so a caller can keep working in it."""
    capability = WorkspaceCapability()
    observed: list[Workspace] = []
    agent = Agent(_tool_call_model(), capabilities=[capability])

    @agent.tool
    async def probe(ctx: RunContext[Any]) -> str:
        observed.append(ctx.workspace)
        return (await ctx.workspace.run(['true'])).stdout

    result = await agent.run('go')

    assert result.workspace is observed[0]
    assert result.workspace.ref == WorkspaceRef(provider='fake', id='fake-capability')

    # Handing it to a second run continues in the same environment rather than making a new one.
    second = await agent.run('again', workspace=result.workspace)
    assert second.workspace is result.workspace
    assert capability.refs == [None]
    assert capability.backend.create_calls == 1


async def test_a_result_still_round_trips_through_json_when_a_workspace_was_used() -> None:
    """The workspace is a live handle, so it is left out of the serialized result rather than breaking it."""
    agent = Agent(_tool_call_model(), capabilities=[WorkspaceCapability()])

    @agent.tool
    async def probe(ctx: RunContext[Any]) -> str:
        return (await ctx.workspace.run(['true'])).stdout

    result = await agent.run('go')
    adapter = TypeAdapter(AgentRunResult[str])
    restored = adapter.validate_json(adapter.dump_json(result))

    assert restored == result
    with pytest.raises(UserError, match='created outside an agent run'):
        await restored.workspace.run(['true'])


async def test_a_result_built_outside_a_run_explains_that_no_workspace_is_attached() -> None:
    result = AgentRunResult[str]('output')

    with pytest.raises(UserError, match='created outside an agent run'):
        await result.workspace.run(['true'])


async def test_two_capabilities_supplying_a_workspace_name_both() -> None:
    """One run, one workspace: a second supplier is a configuration mistake, not a silent winner."""

    class SecondWorkspaceCapability(WorkspaceCapability):
        id = 'second-workspace'

    agent = Agent(TestModel(), capabilities=[WorkspaceCapability(), SecondWorkspaceCapability()])

    with pytest.raises(UserError, match='WorkspaceCapability and SecondWorkspaceCapability both did'):
        await agent.run('go')


async def test_deferred_capability_never_contributes_a_backend() -> None:
    capability = WorkspaceCapability()
    capability.defer_loading = True
    observed: list[Workspace] = []
    agent = Agent(_tool_call_model(), capabilities=[capability])

    @agent.tool
    async def probe(ctx: RunContext[Any]) -> str:
        observed.append(ctx.workspace)
        return 'ok'

    await agent.run('go')

    assert isinstance(observed[0].backend, UnavailableWorkspace)
    assert capability.refs == []


def test_backend_construction_does_no_io() -> None:
    backend = FakeWorkspace('lazy')

    assert backend.ref is None
    assert backend.create_calls == 0
    assert backend.attach_calls == 0


async def test_first_operation_creates_the_environment_once() -> None:
    backend = FakeWorkspace('fresh')

    await backend.run(['true'])
    await backend.working_dir()

    assert backend.create_calls == 1
    assert backend.attach_calls == 0
    assert backend.ref is not None


async def test_concurrent_first_operations_create_one_environment() -> None:
    backend = FakeWorkspace('concurrent')

    async with anyio.create_task_group() as tg:
        tg.start_soon(backend.run, ['true'])
        tg.start_soon(backend.working_dir)

    assert backend.create_calls == 1
    assert backend.ref is not None


async def test_create_backend_ref_is_set_after_its_first_operation() -> None:
    backend = FakeWorkspace('identity')
    assert backend.ref is None

    await backend.run(['true'])

    assert backend.ref == WorkspaceRef(provider='fake', id='fake-identity')


async def test_workspace_ref_forwards_backend_identity() -> None:
    backend = FakeWorkspace('ref')
    workspace = Workspace(backend)

    assert workspace.ref is None
    await workspace.run(['true'])

    assert workspace.ref == WorkspaceRef(provider='fake', id='fake-ref')


def test_workspace_wrap_is_idempotent() -> None:
    backend = FakeWorkspace('wrapped')
    workspace = Workspace.wrap(backend)

    assert isinstance(workspace, Workspace)
    assert workspace.backend is backend
    assert Workspace.wrap(workspace) is workspace


async def test_run_rejects_relative_cwd() -> None:
    with pytest.raises(ValueError, match='absolute'):
        await Workspace(FakeWorkspace('cwd')).run(['true'], cwd='relative')


async def test_two_capabilities_cannot_supply_the_workspace() -> None:
    class FirstWorkspaceCapability(AbstractCapability[Any]):
        def get_workspace(self, ctx: RunContext[Any], *, ref: WorkspaceRef | None) -> WorkspaceBackend:
            return FakeWorkspace('first')

    class SecondWorkspaceCapability(AbstractCapability[Any]):
        def get_workspace(self, ctx: RunContext[Any], *, ref: WorkspaceRef | None) -> WorkspaceBackend:
            return FakeWorkspace('second')

    agent = Agent(_tool_call_model(), capabilities=[FirstWorkspaceCapability(), SecondWorkspaceCapability()])

    with pytest.raises(UserError, match=r'FirstWorkspaceCapability.*SecondWorkspaceCapability'):
        await agent.run('go')


async def test_declining_capability_leaves_the_run_workspace_unavailable() -> None:
    capability = DecliningWorkspaceCapability()
    agent = Agent(_tool_call_model(), capabilities=[capability])

    @agent.tool
    async def probe(ctx: RunContext[Any]) -> str:
        assert isinstance(ctx.workspace.backend, UnavailableWorkspace)
        await ctx.workspace.run(['true'])
        return 'unreachable'  # pragma: no cover

    with pytest.raises(UserError, match='No workspace is attached'):
        await agent.run('go')
    assert capability.calls == 1


async def test_unrecognized_workspace_ref_is_rejected() -> None:
    agent = Agent(_tool_call_model(), capabilities=[DecliningWorkspaceCapability()])

    with pytest.raises(UserError, match="No capability can supply workspace 'missing'"):
        await agent.run('go', workspace=WorkspaceRef(provider='fake', id='missing'))


async def test_capability_backend_is_available_without_connecting_during_run_setup() -> None:
    capability = WorkspaceCapability()
    seen: list[Workspace] = []
    agent = Agent(_tool_call_model(), capabilities=[capability])

    @agent.tool
    async def probe(ctx: RunContext[Any]) -> str:
        seen.append(ctx.workspace)
        return 'ok'

    await agent.run('go')

    assert seen[0].backend is capability.backend
    assert capability.backend.create_calls == 0


async def test_run_never_cleans_up_the_workspace() -> None:
    backend = FakeWorkspace('persistent')
    agent = Agent(_tool_call_model(), capabilities=[])

    @agent.tool
    async def probe(ctx: RunContext[Any]) -> str:
        await ctx.workspace.run(['true'])
        return 'ok'

    await agent.run('go', workspace=backend)

    assert backend.cleanup_calls == []


async def test_failed_run_never_cleans_up_the_workspace() -> None:
    backend = FakeWorkspace('failed')
    agent = Agent(_tool_call_model('explode'))

    @agent.tool
    async def explode(ctx: RunContext[Any]) -> str:
        await ctx.workspace.run(['true'])
        raise RuntimeError('boom')

    with pytest.raises(RuntimeError, match='boom'):
        await agent.run('go', workspace=backend)

    assert backend.cleanup_calls == []


async def test_cancelled_run_never_cleans_up_the_workspace() -> None:
    backend = FakeWorkspace('cancelled')
    agent = Agent(_tool_call_model())
    entered = anyio.Event()

    @agent.tool
    async def probe(ctx: RunContext[Any]) -> str:
        await ctx.workspace.run(['true'])
        entered.set()
        await anyio.sleep(60)
        return 'unreachable'  # pragma: no cover

    async with anyio.create_task_group() as tg:

        async def run_agent() -> None:
            await agent.run('go', workspace=backend)

        tg.start_soon(run_agent)
        await entered.wait()
        tg.cancel_scope.cancel()

    assert backend.cleanup_calls == []


async def test_guard_workflow_workspace_only_rejects_a_live_handle() -> None:
    ref = WorkspaceRef(provider='fake', id='existing')

    assert guard_workflow_workspace(ref, live_error='live workspace') is ref
    assert guard_workflow_workspace(None, live_error='live workspace') is None
    with pytest.raises(UserError, match='live workspace'):
        guard_workflow_workspace(FakeWorkspace('live'), live_error='live workspace')
    with pytest.raises(UserError, match='deprecated wrapper'):
        guard_workflow_workspace(ref, live_error='live workspace', ref_error='deprecated wrapper')


async def test_capability_can_supply_a_backend_for_an_explicit_ref() -> None:
    capability = ConnectOnlyWorkspaceCapability()
    agent = Agent(_tool_call_model(), capabilities=[capability])

    @agent.tool
    async def probe(ctx: RunContext[Any]) -> str:
        return (await ctx.workspace.run(['true'])).stdout

    result: AgentRunResult[Any] = await agent.run('go', workspace=WorkspaceRef(provider='fake', id='existing'))

    assert result.output == 'done'
    assert capability.ids == ['existing']
    assert result.workspace.ref == WorkspaceRef(provider='fake', id='existing')
    assert await result.workspace.working_dir() == '/workspace'

    # This capability only attaches: with no ref it declines and the run gets the unavailable default.
    without_ref: AgentRunResult[Any] = await Agent(TestModel(), capabilities=[capability]).run('go')

    assert isinstance(without_ref.workspace.backend, UnavailableWorkspace)
    assert capability.ids == ['existing']
