"""Focused public tests for the Modal workspace capability."""

from __future__ import annotations

import sys
import warnings
from pathlib import Path
from typing import Any

import anyio
import pytest
from pydantic_ai import Agent
from pydantic_ai.capabilities import PrefixTools
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import RunContext
from pydantic_ai.usage import RunUsage
from pydantic_ai.workspaces import (
    Workspace,
    WorkspaceError,
    WorkspaceRef,
    WorkspaceTimeoutError,
    WorkspaceUnavailableError,
)

import pydantic_ai_harness.modal_sandbox as modal_sandbox_package
from pydantic_ai_harness._warn import HarnessDeprecationWarning
from pydantic_ai_harness.coder import Coder
from pydantic_ai_harness.filesystem import FileSystem
from pydantic_ai_harness.modal_sandbox import ModalSandbox, ModalSandboxBackend

from .._tool_calls import call_tools
from .conftest import skip_or_fail_live_tier
from .fake_modal import FakeModal

pytestmark = pytest.mark.anyio(backends=['asyncio'])


async def test_backend_acquires_fresh_workspace_and_records_ref(fake_modal: FakeModal) -> None:
    backend = ModalSandboxBackend()
    assert not fake_modal.sandboxes
    native = await backend.get_sandbox()
    assert native is fake_modal.sandboxes[0]
    assert backend.ref == WorkspaceRef(provider='modal', id=native.object_id)


async def test_missing_modal_extra_has_install_hint(fake_modal: FakeModal, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, 'modal', None)
    with pytest.raises(UserError, match=r'pydantic-ai-harness\[modal\]'):
        await ModalSandboxBackend().get_sandbox()


async def test_backend_attaches_explicit_ref_without_create(fake_modal: FakeModal) -> None:
    backend = ModalSandboxBackend(ref=WorkspaceRef(provider='modal', id='existing'))
    await backend.get_sandbox()
    assert fake_modal.attach_ids == ['existing']
    assert not fake_modal.create_kwargs


@pytest.mark.parametrize(
    ('name', 'expected'),
    [('AuthError', WorkspaceUnavailableError), ('InvalidError', WorkspaceError), ('ConnectionError', None)],
)
async def test_attach_failures_keep_reference_and_do_not_create(
    fake_modal: FakeModal, name: str, expected: type[Exception] | None
) -> None:
    fake_modal.attach_error = fake_modal.exception(name)('failed')
    backend = ModalSandboxBackend(ref=WorkspaceRef(provider='modal', id='existing'))
    with pytest.raises(expected or fake_modal.exception(name)) as exc_info:
        await backend.get_sandbox()
    assert backend.ref == WorkspaceRef(provider='modal', id='existing')
    assert not fake_modal.create_kwargs
    assert fake_modal.attach_error in (exc_info.value, exc_info.value.__cause__)


async def test_native_workspace_identity_is_immediate(fake_modal: FakeModal) -> None:
    native = await ModalSandboxBackend().get_sandbox()
    backend = ModalSandboxBackend(sandbox=native)
    assert backend.ref == WorkspaceRef(provider='modal', id=native.object_id)
    assert await backend.get_sandbox() is native


async def test_native_workspace_and_ref_conflict(fake_modal: FakeModal) -> None:
    native = await ModalSandboxBackend().get_sandbox()
    with pytest.raises(ValueError, match='either `sandbox` or `ref`'):
        ModalSandboxBackend(sandbox=native, ref=WorkspaceRef(provider='modal', id='other'))


async def test_filesystem_not_directory_error_uses_builtin_exception(fake_modal: FakeModal) -> None:
    backend = ModalSandboxBackend()
    await backend.get_sandbox()
    fake_modal.sandboxes[0].fs_error = fake_modal.exception('SandboxFilesystemNotADirectoryError')('file')
    with pytest.raises(NotADirectoryError, match='Not a directory'):
        await backend.read_bytes('/file/child')


async def test_command_start_timeout_is_bounded(fake_modal: FakeModal) -> None:
    fake_modal.exec_hangs = True
    backend = ModalSandboxBackend()
    with pytest.raises(WorkspaceTimeoutError, match='Command timed out after 0.01s'):
        with anyio.fail_after(0.2):
            await backend.run(['echo', 'hello'], timeout=0.01)


async def test_create_timeout_is_a_retryable_timeout(fake_modal: FakeModal, monkeypatch: pytest.MonkeyPatch) -> None:
    # An image build or pull may still be running: report a transient timeout.
    fake_modal.create_gate = anyio.Event()
    fake_modal.create_before_gate = True
    monkeypatch.setattr('pydantic_ai_harness.modal_sandbox._backend._CREATE_TIMEOUT', 0.01)
    backend = ModalSandboxBackend()
    with pytest.raises(TimeoutError, match='image build or pull') as exc_info:
        await backend.get_sandbox()
    assert not isinstance(exc_info.value, WorkspaceError)


async def test_an_sdk_timeout_during_create_propagates_unchanged(fake_modal: FakeModal) -> None:
    error = TimeoutError('grpc deadline')
    fake_modal.create_error = error
    with pytest.raises(TimeoutError) as exc_info:
        await ModalSandboxBackend().get_sandbox()
    assert exc_info.value is error


async def test_command_timeout_starts_once_the_sandbox_is_acquired(fake_modal: FakeModal) -> None:
    fake_modal.create_gate = anyio.Event()
    backend = ModalSandboxBackend()
    async with anyio.create_task_group() as tg:

        async def release() -> None:
            # Creating the sandbox outlasts the timeout; the command itself fits in it comfortably.
            await anyio.sleep(1.1)
            assert fake_modal.create_gate is not None
            fake_modal.create_gate.set()

        tg.start_soon(release)
        result = await backend.run(['echo', 'hello'], timeout=1)
        assert result.stdout == 'echo hello\n'
    assert fake_modal.sandboxes[0].exec_calls[0].timeout == 1


async def test_command_timeout_keeps_captured_output(fake_modal: FakeModal) -> None:
    fake_modal.responder = lambda argv, timeout: ('partial stdout', 'partial stderr', 0)
    fake_modal.wait_hangs = True
    backend = ModalSandboxBackend()
    with pytest.raises(WorkspaceTimeoutError) as exc_info:
        await backend.run(['echo', 'hello'], timeout=0.01)
    assert exc_info.value.stdout == 'partial stdout'
    assert exc_info.value.stderr == 'partial stderr'


async def test_command_timeout_keeps_completed_stderr_when_stdout_reader_hangs(fake_modal: FakeModal) -> None:
    fake_modal.responder = lambda argv, timeout: ('partial stdout', 'partial stderr', 0)
    fake_modal.stdout_hangs = True
    backend = ModalSandboxBackend()
    with pytest.raises(WorkspaceTimeoutError) as exc_info:
        await backend.run(['echo', 'hello'], timeout=0.01)
    assert exc_info.value.stdout == ''
    assert exc_info.value.stderr == 'partial stderr'


def test_capability_takes_the_base_class_options_and_the_creation_settings() -> None:
    capability = ModalSandbox(id='modal', description='a sandbox', image='python:3.13-slim', env={'A': '1'})
    assert (capability.id, capability.description, capability.defer_loading) == ('modal', 'a sandbox', False)
    assert capability == ModalSandbox(id='modal', description='a sandbox', image='python:3.13-slim', env={'A': '1'})
    assert capability.get_toolset() is None
    assert capability.get_instructions() is None


def test_modal_docs_name_default_user_and_directory() -> None:
    for path in (Path('docs/modal-sandbox.md'), Path('pydantic_ai_harness/modal_sandbox/README.md')):
        text = path.read_text()
        assert 'root user' in text
        assert 'relative paths' in text
        assert "working_dir='/workspace'" in text


def test_modal_docs_distinguish_command_timeout_from_sandbox_lifetime() -> None:
    for path in (Path('docs/modal-sandbox.md'), Path('pydantic_ai_harness/modal_sandbox/README.md')):
        text = path.read_text()
        assert 'defer_loading=True' in text
        assert 'Removed. Use `sandbox_timeout`, which bounds every command.' not in text
        assert 'does not apply to attached sandboxes' in text


def test_modal_coder_examples_explain_eager_creation_and_set_working_dir() -> None:
    for path in (Path('docs/modal-sandbox.md'), Path('pydantic_ai_harness/modal_sandbox/README.md')):
        text = path.read_text()
        assert 'Coder(repo_context=False)' in text
        assert 'when the run starts' in text
        assert "ModalSandbox(working_dir='/workspace')" in text


def test_modal_capability_repr_does_not_expose_env_secrets() -> None:
    assert 'private-token' not in repr(ModalSandbox(env={'TOKEN': 'private-token'}))


def test_defer_loading_is_refused() -> None:
    with pytest.raises(UserError, match="skipped when the run's workspace is chosen"):
        ModalSandbox(id='modal', defer_loading=True)


@pytest.mark.parametrize(
    ('legacy', 'guidance'),
    [
        ({'sandbox_id': 'sb-1'}, "workspace=WorkspaceRef(provider='modal', id=sandbox_id)"),
        ({'session': object()}, 'ModalSandboxBackend(sandbox=<modal.Sandbox>)'),
        ({'default_command_timeout': 5.0}, 'Shell(default_timeout=...)'),
        ({'max_command_timeout': 60}, 'sandbox lifetime (`sandbox_timeout`) bounds every command'),
        ({'max_output_bytes': 1}, 'Shell(max_output_chars=...)'),
        ({'max_output_lines': 1}, 'ToolOutputLimits'),
        ({'max_read_bytes': 1}, 'FileSystem(max_read_lines=..., max_read_chars=...)'),
        ({'instructions': ''}, "belongs in the agent's `instructions`"),
    ],
)
def test_previous_constructor_arguments_are_refused_with_guidance(legacy: dict[str, Any], guidance: str) -> None:
    # The previous `ModalSandbox` bundled its own tools; each of its arguments now points at
    # where that setting lives, rather than failing as an unknown keyword.
    with pytest.raises(UserError) as exc_info:
        ModalSandbox(**legacy)  # pyright: ignore[reportArgumentType]
    message = str(exc_info.value)
    (name,) = legacy
    assert message.startswith(f'`ModalSandbox` no longer accepts `{name}`.')
    assert 'add `Shell()` and/or `FileSystem()`' in message
    assert f'- `{name}`: ' in message
    assert guidance in message
    assert message.endswith('#upgrading-from-the-previous-modalsandbox')


def test_several_previous_arguments_are_reported_together() -> None:
    with pytest.raises(UserError, match=r'no longer accepts `sandbox_id`, `instructions`') as exc_info:
        ModalSandbox(sandbox_id='sb-1', instructions='')  # pyright: ignore[reportArgumentType]
    assert str(exc_info.value).count('\n- `') == 2


def test_an_unknown_argument_is_still_a_type_error() -> None:
    with pytest.raises(TypeError, match="unexpected keyword argument 'imag'"):
        ModalSandbox(imag='python:3.13-slim')  # pyright: ignore[reportArgumentType]


async def test_capability_declines_foreign_ref(fake_modal: FakeModal) -> None:
    capability = ModalSandbox()
    ctx = RunContext(deps=None, model=TestModel(), usage=RunUsage())
    assert capability.get_workspace(ctx, ref=WorkspaceRef(provider='other', id='foreign')) is None
    assert not fake_modal.sandboxes


async def test_backend_rejects_foreign_ref(fake_modal: FakeModal) -> None:
    with pytest.raises(ValueError, match="expected 'modal'"):
        ModalSandboxBackend(ref=WorkspaceRef(provider='other', id='foreign'))
    assert not fake_modal.sandboxes


async def test_agent_without_workspace_tool_does_not_create(fake_modal: FakeModal) -> None:
    agent = Agent(TestModel(), capabilities=[ModalSandbox()])
    with pytest.warns(UserWarning, match='registers no tools'):
        result = await agent.run('go')
    assert result.output
    assert not fake_modal.sandboxes


async def test_agent_runs_without_history_create_fresh_workspaces(fake_modal: FakeModal) -> None:
    agent = Agent(TestModel(call_tools=['run_command']), capabilities=[ModalSandbox()])

    @agent.tool
    async def run_command(ctx: RunContext[object]) -> str:
        return (await ctx.workspace.run(['printf', 'ok'])).stdout

    await agent.run('go')
    await agent.run('go')
    assert len(fake_modal.sandboxes) == 2


async def test_agent_history_attaches_same_workspace(fake_modal: FakeModal) -> None:
    def model(messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        request = messages[-1]
        if isinstance(request, ModelRequest) and any(isinstance(part, ToolReturnPart) for part in request.parts):
            return ModelResponse(parts=[TextPart(content='done')])
        return ModelResponse(parts=[ToolCallPart(tool_name='run_command', args={}, tool_call_id='call')])

    agent = Agent(FunctionModel(model), capabilities=[ModalSandbox()])
    tool_results: list[str] = []

    @agent.tool
    async def run_command(ctx: RunContext[object]) -> str:
        if await ctx.workspace.exists('/marker.txt'):
            result = await ctx.workspace.read_text('/marker.txt')
        else:
            await ctx.workspace.write_text('/marker.txt', 'persisted')
            result = 'created'
        tool_results.append(result)
        return result

    first = await agent.run('go')
    second = await agent.run('go', message_history=first.all_messages())
    assert first.output == 'done'
    assert second.output == 'done'
    assert tool_results == ['created', 'persisted']
    assert len(fake_modal.sandboxes) == 1
    assert first.workspace.ref is not None
    assert fake_modal.attach_ids == [first.workspace.ref.id]


async def test_concurrent_operations_share_one_acquisition(fake_modal: FakeModal) -> None:
    backend = ModalSandboxBackend()
    fake_modal.create_gate = anyio.Event()

    async with anyio.create_task_group() as tg:
        tg.start_soon(backend.run, ['printf', 'one'])
        tg.start_soon(backend.run, ['printf', 'two'])
        while not fake_modal.create_started:
            await anyio.sleep(0)
        fake_modal.create_gate.set()
    assert fake_modal.owned_creates == 1
    assert len(fake_modal.sandboxes[0].exec_calls) == 2


async def test_filesystem_operation_is_not_blocked_by_command_wait(
    fake_modal: FakeModal,
) -> None:
    fake_modal.wait_hangs = True
    backend = ModalSandboxBackend()
    async with anyio.create_task_group() as tg:
        tg.start_soon(backend.run, ['sleep', '1'])
        while not fake_modal.sandboxes:
            await anyio.sleep(0)
        fake_modal.sandboxes[0].files['/marker.txt'] = b'marker'
        with anyio.fail_after(0.2):
            assert await Workspace(backend).read_text('/marker.txt') == 'marker'
        tg.cancel_scope.cancel()


async def test_failed_acquisition_can_retry(fake_modal: FakeModal) -> None:
    backend = ModalSandboxBackend()
    fake_modal.create_error = fake_modal.exception('ConnectionError')('temporary')
    with pytest.raises(fake_modal.exception('ConnectionError')):
        await backend.get_sandbox()
    fake_modal.create_error = None
    await backend.get_sandbox()
    assert fake_modal.owned_creates == 1


async def test_cancellation_after_modal_created_the_sandbox_keeps_its_ref(fake_modal: FakeModal) -> None:
    # Cancelled while Modal's create response is in flight, the backend still records the
    # sandbox, so `ref` names it and the next operation reuses it instead of creating another.
    backend = ModalSandboxBackend()
    fake_modal.create_gate = anyio.Event()
    with anyio.CancelScope() as scope:
        async with anyio.create_task_group() as tg:
            tg.start_soon(backend.get_sandbox)
            while not fake_modal.create_started:
                await anyio.sleep(0)
            scope.cancel()
            fake_modal.create_gate.set()
    assert scope.cancelled_caught
    assert backend.ref == WorkspaceRef(provider='modal', id='sb-owned')
    await backend.run(['true'])
    assert fake_modal.owned_creates == 1


async def test_agent_uses_modal_sandbox(fake_modal: FakeModal) -> None:
    agent = Agent(TestModel(call_tools=['run_command']), capabilities=[ModalSandbox()])

    @agent.tool
    async def run_command(ctx: RunContext[object]) -> str:
        result = await ctx.workspace.run(['printf', 'hello'])
        return result.stdout

    result = await agent.run('go')
    assert 'run_command' in result.output
    assert fake_modal.sandboxes[0].exec_calls[0].argv[-2:] == ['printf', 'hello']


@pytest.mark.parametrize(
    ('name', 'replacement'),
    [
        ('ModalSandboxSession', 'ModalSandboxBackend(sandbox=<modal.Sandbox>)'),
        ('ModalSandboxExecResult', 'pydantic_ai.workspaces.CommandResult'),
        ('ModalSandboxError', 'pydantic_ai.workspaces.WorkspaceError'),
        ('ModalSandboxTerminalError', 'pydantic_ai.workspaces.WorkspaceUnavailableError'),
        ('ModalSandboxUnavailableError', 'pydantic_ai.workspaces.WorkspaceUnavailableError'),
        ('ModalSandboxAuthError', 'pydantic_ai.workspaces.WorkspaceUnavailableError'),
    ],
)
def test_removed_names_raise_import_error_naming_the_replacement(name: str, replacement: str) -> None:
    with pytest.raises(ImportError) as exc_info:
        getattr(modal_sandbox_package, name)
    message = str(exc_info.value)
    assert message.startswith(f'`{name}` was removed from `pydantic_ai_harness.modal_sandbox`.')
    assert replacement in message
    assert message.endswith('#upgrading-from-the-previous-modalsandbox')
    assert exc_info.value.name == name


def test_removed_name_fails_a_from_import_with_the_guidance() -> None:
    with pytest.raises(ImportError, match='ModalSandboxBackend'):
        from pydantic_ai_harness.modal_sandbox import ModalSandboxSession  # noqa: F401, I001, PLC0415  # pyright: ignore[reportUnusedImport]


def test_other_missing_names_are_attribute_errors() -> None:
    with pytest.raises(AttributeError, match="has no attribute 'ModalSandboxTypo'"):
        getattr(modal_sandbox_package, 'ModalSandboxTypo')


@pytest.mark.parametrize('value', [0, -5, 1.5, True, None])
def test_sandbox_timeout_must_be_a_positive_integer(value: Any) -> None:
    with pytest.raises(UserError, match=rf'sandbox_timeout must be an integer of at least 1, got {value!r}\.'):
        ModalSandbox(sandbox_timeout=value)


@pytest.mark.parametrize('value', [0, -5, 1.5, True])
def test_idle_timeout_must_be_a_positive_integer_or_none(value: Any) -> None:
    with pytest.raises(UserError, match=rf'idle_timeout must be an integer of at least 1 or None, got {value!r}\.'):
        ModalSandbox(idle_timeout=value)


@pytest.mark.parametrize('working_dir', ['relative/dir', '', 'C:\\work'])
def test_working_dir_must_be_an_absolute_posix_path(working_dir: str) -> None:
    with pytest.raises(UserError, match='working_dir must be an absolute POSIX path or None'):
        ModalSandbox(working_dir=working_dir)


async def test_creation_settings_reach_the_sandbox(fake_modal: FakeModal) -> None:
    capability = ModalSandbox(sandbox_timeout=10, idle_timeout=2, working_dir='/work', env={'A': '1'})
    backend = capability.get_workspace(RunContext(deps=None, model=TestModel(), usage=RunUsage()), ref=None)
    assert isinstance(backend, ModalSandboxBackend)
    await backend.get_sandbox()
    created = fake_modal.create_kwargs[-1]
    assert (created['timeout'], created['idle_timeout'], created['workdir'], created['env']) == (
        10,
        2,
        '/work',
        {'A': '1'},
    )


def test_workdir_is_a_deprecated_alias_of_working_dir() -> None:
    with pytest.warns(
        HarnessDeprecationWarning,
        match=r'`ModalSandbox\(workdir=...\)` has been renamed to `ModalSandbox\(working_dir=...\)`',
    ):
        capability = ModalSandbox(workdir='/work')
    assert capability.working_dir == '/work'


def test_workdir_and_working_dir_together_are_refused() -> None:
    with pytest.raises(UserError, match='Pass `working_dir` only'):
        ModalSandbox(workdir='/a', working_dir='/b')


async def test_agent_without_workspace_tools_warns_once_per_process(fake_modal: FakeModal) -> None:
    # `ModalSandbox(image=...)` alone was the documented usage when the capability had its own
    # tools; it still builds, so the warning is what tells the user the model lost the sandbox.
    agent = Agent(TestModel(), capabilities=[ModalSandbox(image='python:3.13-slim')])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        await agent.run('go')
        await agent.run('go again')
    messages = [str(warning.message) for warning in caught if issubclass(warning.category, UserWarning)]
    assert len(messages) == 1
    assert messages[0].startswith("`ModalSandbox` supplies the Modal sandbox as the run's `ctx.workspace`")
    assert 'ModalSandbox(warn_if_no_tools=False)' in messages[0]
    assert messages[0].endswith('#upgrading-from-the-previous-modalsandbox')
    assert not fake_modal.sandboxes


async def test_warn_if_no_tools_false_silences_the_warning(fake_modal: FakeModal) -> None:
    agent = Agent(TestModel(), capabilities=[ModalSandbox(warn_if_no_tools=False)])
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        await agent.run('go')


async def test_agent_with_filesystem_does_not_warn(fake_modal: FakeModal) -> None:
    fake_modal.responder = lambda argv, timeout: ('/root\n', '', 0)
    agent = Agent(TestModel(call_tools=[]), capabilities=[ModalSandbox(), FileSystem()])
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        await agent.run('go')


async def test_prefixed_filesystem_does_not_warn(fake_modal: FakeModal) -> None:
    agent = Agent(TestModel(call_tools=[]), capabilities=[ModalSandbox(), PrefixTools(FileSystem(), prefix='repo')])
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        await agent.run('go')


async def test_agent_with_custom_workspace_tool_does_not_warn(fake_modal: FakeModal) -> None:
    agent = Agent(TestModel(call_tools=[]), capabilities=[ModalSandbox()])

    @agent.tool
    async def grep(ctx: RunContext[object], pattern: str) -> str:
        return (await ctx.workspace.run(['grep', '-r', pattern, '.'])).stdout  # pragma: no cover - not called

    with warnings.catch_warnings():
        warnings.simplefilter('error')
        await agent.run('go')


async def test_coder_tools_run_in_the_sandbox(fake_modal: FakeModal, tmp_path: Path) -> None:
    # The fake's host mode runs the sandbox's commands and file operations under `tmp_path`.
    fake_modal.host_root = tmp_path.resolve()
    results = await call_tools(
        [ModalSandbox(), Coder()],
        [
            ('write_file', {'path': 'hello.py', 'content': "print('hello from the sandbox')\n"}),
            ('shell', {'command': 'python3 hello.py'}),
        ],
    )
    assert (tmp_path / 'hello.py').read_text() == "print('hello from the sandbox')\n"
    assert 'hello from the sandbox' in results[1]
    assert len(fake_modal.sandboxes) == 1


@pytest.mark.parametrize(('require', 'outcome'), [('', pytest.skip.Exception), ('1', pytest.fail.Exception)])
def test_live_tier_with_empty_credentials_skips_unless_required(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, require: str, outcome: type[BaseException]
) -> None:
    # An empty secret is how CI spells a missing one; `MODAL_REQUIRE_LIVE` makes that fail.
    monkeypatch.setenv('PYDANTIC_AI_HARNESS_MODAL_LIVE', '1')
    monkeypatch.setenv('MODAL_TOKEN_ID', '')
    monkeypatch.setenv('MODAL_TOKEN_SECRET', '')
    monkeypatch.setenv('HOME', str(tmp_path))
    monkeypatch.setenv('MODAL_REQUIRE_LIVE', require)
    with pytest.raises(outcome):
        skip_or_fail_live_tier()
