"""Workspaces under `TemporalDurability`, against a real sandboxed worker.

Not VCR tests: the behavior under test is where each workspace call runs (an activity, or directly
inside one), which needs the Temporal server's history rather than a provider recording. A fake
remote provider stands in for a real one: its environments live in the worker process, so the
sandboxed workflow can construct a backend for one but only an activity can touch it.
"""

from __future__ import annotations

import errno
import os
import shutil
import sys
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import timedelta
from importlib.machinery import ModuleSpec
from pathlib import Path
from typing import Any, Literal

import pytest
from pydantic import TypeAdapter

from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import AbstractCapability, Capability, LocalWorkspace
from pydantic_ai.exceptions import UserError
from pydantic_ai.models.test import TestModel
from pydantic_ai.usage import RunUsage
from pydantic_ai.workspaces import (
    CommandResult,
    FileEntry,
    ReadOnlyWorkspace,
    SupportsCommands,
    SupportsFilesystem,
    Workspace,
    WorkspaceBackend,
    WorkspaceCommand,
    WorkspaceReadOnlyError,
    WorkspaceRef,
    WorkspaceTimeoutError,
    WorkspaceUnavailableError,
)
from pydantic_ai.workspaces.unavailable import UnavailableWorkspace

try:
    from temporalio import activity, workflow
    from temporalio.activity import _Definition as ActivityDefinition  # pyright: ignore[reportPrivateUsage]
    from temporalio.client import Client, WorkflowFailureError
    from temporalio.common import RetryPolicy
    from temporalio.testing import ActivityEnvironment
    from temporalio.worker import Replayer, Worker
    from temporalio.worker.workflow_sandbox import SandboxedWorkflowRunner
    from temporalio.workflow import ActivityConfig

    from pydantic_ai.durable_exec._workspace import DurableWorkspace, WorkspaceCall, execute_call, raise_error
    from pydantic_ai.durable_exec.prefect import PrefectDurability
    from pydantic_ai.durable_exec.temporal import (
        AgentPlugin,
        PydanticAIPlugin,
        TemporalDurability,
        _workflow_runner,  # pyright: ignore[reportPrivateUsage]
    )
    from pydantic_ai.durable_exec.temporal._operation_backend import workspace_run_activity_config
    from pydantic_ai.durable_exec.temporal._run_context import TemporalRunContext, deserialize_run_context
    from pydantic_ai.durable_exec.temporal._toolset import with_non_retryable_errors
    from pydantic_ai.durable_exec.temporal._transports import _WorkspaceCallWire

except ImportError:  # pragma: lax no cover
    pytest.skip('temporal not installed', allow_module_level=True)


# The 3.14 durable-exec CI leg takes this skip; every other leg falls through.
if sys.version_info >= (3, 14):  # pragma: lax no cover
    pytest.skip(
        'temporalio sandbox is incompatible with Python 3.14: '
        'sandbox module state accumulates across validation cycles causing import failures after ~22 workflows '
        '(remove when https://github.com/temporalio/sdk-python/issues/1326 closes)',
        allow_module_level=True,
    )

try:
    import logfire  # pyright: ignore[reportUnusedImport]  # noqa: F401
except ImportError:  # pragma: lax no cover
    pytest.skip('logfire not installed', allow_module_level=True)

try:
    import mcp  # pyright: ignore[reportUnusedImport]  # noqa: F401
except ImportError:  # pragma: lax no cover
    pytest.skip('mcp not installed', allow_module_level=True)

try:
    import openai  # pyright: ignore[reportUnusedImport]  # noqa: F401
except ImportError:  # pragma: lax no cover
    pytest.skip('openai not installed', allow_module_level=True)


with workflow.unsafe.imports_passed_through():
    from ..._inline_snapshot import snapshot

    # Loads `vcr`, which Temporal doesn't like without passing through the import
    from ...conftest import IsStr
    from ._shared import (
        BASE_ACTIVITY_CONFIG,
        TASK_QUEUE,
        _workflow_failure_cause,  # pyright: ignore[reportPrivateUsage]
    )

_REF_ADAPTER: TypeAdapter[WorkspaceRef | None] = TypeAdapter(WorkspaceRef | None)

pytestmark = [pytest.mark.anyio, pytest.mark.filterwarnings('ignore::pydantic.PydanticDeprecatedSince20')]


@pytest.mark.parametrize('durability', [TemporalDurability, PrefectDurability])
def test_instructions_only_capability_needs_no_toolset_id(
    durability: type[TemporalDurability] | type[PrefectDurability],
) -> None:
    agent = Agent(TestModel(), name='instructions_only', capabilities=[Capability(instructions='x'), durability()])
    assert agent is not None


def test_temporal_runner_passes_installed_harness_through(monkeypatch: pytest.MonkeyPatch) -> None:
    from pydantic_ai.durable_exec import temporal

    runner = SandboxedWorkflowRunner()

    def installed(module: str) -> ModuleSpec | None:
        return ModuleSpec(module, loader=None) if module == 'pydantic_ai_harness' else None

    monkeypatch.setattr(temporal, 'find_spec', installed)
    configured = _workflow_runner(runner)
    assert isinstance(configured, SandboxedWorkflowRunner)
    assert 'pydantic_ai_harness' in configured.restrictions.passthrough_modules

    def absent(module: str) -> ModuleSpec | None:
        return None

    monkeypatch.setattr(temporal, 'find_spec', absent)
    configured = _workflow_runner(runner)
    assert isinstance(configured, SandboxedWorkflowRunner)
    assert 'pydantic_ai_harness' not in configured.restrictions.passthrough_modules


def test_workspace_failures_do_not_retry_temporal_activities() -> None:
    policy = with_non_retryable_errors(RetryPolicy())
    assert {WorkspaceTimeoutError.__name__, WorkspaceReadOnlyError.__name__, WorkspaceUnavailableError.__name__} <= set(
        policy.non_retryable_error_types or []
    )


def test_workspace_run_activity_has_time_for_command_and_cleanup() -> None:
    config = ActivityConfig(start_to_close_timeout=timedelta(seconds=60))
    assert workspace_run_activity_config(config, 120).get('start_to_close_timeout') == timedelta(seconds=150)
    assert workspace_run_activity_config(config, None).get('start_to_close_timeout') == timedelta(hours=1)
    assert config.get('start_to_close_timeout') == timedelta(seconds=60)


async def test_unavailable_workspace_reason_survives_activity_context() -> None:
    agent = Agent(TestModel(), name='unavailable')
    ctx = RunContext(
        deps=None, model=TestModel(), usage=RunUsage(), workspace=Workspace(UnavailableWorkspace('disabled by policy'))
    )
    restored = deserialize_run_context(
        TemporalRunContext, TemporalRunContext.serialize_run_context(ctx), deps=None, agent=agent
    )
    with pytest.raises(WorkspaceUnavailableError, match='disabled by policy'):
        await restored.workspace.working_dir()


# --- A fake remote provider ---------------------------------------------------------------------
#
# Environments are held in this module, in the worker process. The workflow sandbox re-imports the
# module and so gets its own empty copy, which is the point: constructing a backend there is pure,
# and only an activity (which runs in the worker process proper) reaches the environments.

_ENVIRONMENTS: dict[str, dict[str, bytes]] = {}
_PROVIDER_LOG: list[str] = []


def _reset_provider() -> None:
    _ENVIRONMENTS.clear()
    _PROVIDER_LOG.clear()


class RemoteBackend(WorkspaceBackend, SupportsCommands, SupportsFilesystem):
    """A backend that creates or attaches on first use, like a real remote provider's."""

    def __init__(self, ref: WorkspaceRef | None) -> None:
        self._ref = ref

    @property
    def ref(self) -> WorkspaceRef | None:
        return self._ref

    def _files(self) -> dict[str, bytes]:
        assert activity.in_activity(), 'the fake provider must only be reached from an activity'
        if self._ref is None:
            env_id = f'env-{len(_ENVIRONMENTS) + 1}'
            _ENVIRONMENTS[env_id] = {}
            _PROVIDER_LOG.append(f'create:{env_id}')
            self._ref = WorkspaceRef(provider='remote', id=env_id)
        elif self._ref.id not in _ENVIRONMENTS:
            raise WorkspaceUnavailableError(f'environment {self._ref.id!r} does not exist')
        else:
            _PROVIDER_LOG.append(f'attach:{self._ref.id}')
        return _ENVIRONMENTS[self._ref.id]

    async def run(
        self,
        command: WorkspaceCommand,
        *,
        shell: bool = False,
        cwd: str | None = None,
        env: Mapping[str, str] | None = None,
        timeout: float | None = None,
    ) -> CommandResult:
        self._files()
        if isinstance(command, str) != shell:
            raise TypeError('a shell string needs `shell=True`, an argv sequence needs `shell=False`')
        return CommandResult(exit_code=0, stdout=f'ran:{" ".join(command)}', stderr='')

    async def working_dir(self) -> str:
        self._files()
        return '/remote'

    async def read_bytes(self, path: str) -> bytes:
        files = self._files()
        if path not in files:
            raise FileNotFoundError(path)
        return files[path]

    async def write_bytes(self, path: str, data: bytes) -> None:
        self._files()[path] = data

    async def stat(self, path: str) -> FileEntry:
        files = self._files()
        if path not in files:
            raise FileNotFoundError(path)
        return FileEntry(name=path.rsplit('/', 1)[-1], path=path, is_dir=False, size=len(files[path]))

    async def list_dir(self, path: str) -> Sequence[FileEntry]:
        return [
            FileEntry(name=file.rsplit('/', 1)[-1], path=file, is_dir=False, size=len(data))
            for file, data in sorted(self._files().items())
        ]

    async def make_dir(self, path: str) -> None:
        self._files()

    async def remove(self, path: str) -> None:
        files = self._files()
        if path not in files:
            raise FileNotFoundError(path)
        del files[path]

    async def exists(self, path: str) -> bool:
        return path in self._files()


@dataclass
class RemoteWorkspaces(AbstractCapability[Any]):
    read_only: bool = False

    def get_workspace(self, ctx: RunContext[Any], *, ref: WorkspaceRef | None) -> WorkspaceBackend | None:
        if ref is not None and ref.provider != 'remote':
            return None
        backend = RemoteBackend(ref)
        return ReadOnlyWorkspace(Workspace(backend)) if self.read_only else backend


class WriteInHook(AbstractCapability[Any]):
    """Touches the workspace from workflow-side hooks, where every call has to be an activity."""

    async def before_run(self, ctx: RunContext[Any]) -> None:
        assert workflow.in_workflow()
        assert isinstance(ctx.workspace, DurableWorkspace)
        await ctx.workspace.write_text('hook.txt', f'before_run in {await ctx.workspace.working_dir()}')


def _activity_names(history: Any) -> list[str]:
    return [
        event.activity_task_scheduled_event_attributes.activity_type.name
        for event in history.events
        if event.HasField('activity_task_scheduled_event_attributes')
    ]


# --- A fresh workspace: one `ensure`, parallel tools, hooks and the result ----------------------

fresh_agent = Agent(
    TestModel(call_tools=['write_left', 'write_right']),
    name='fresh',
    capabilities=[WriteInHook(), RemoteWorkspaces(), TemporalDurability(activity_config=BASE_ACTIVITY_CONFIG)],
)


@fresh_agent.tool
async def write_left(ctx: RunContext[Any]) -> str:
    assert activity.in_activity()
    assert not isinstance(ctx.workspace, DurableWorkspace)
    await ctx.workspace.write_text('left.txt', 'L')
    return await ctx.workspace.read_text('hook.txt')


@fresh_agent.tool
async def write_right(ctx: RunContext[Any]) -> str:
    assert not isinstance(ctx.workspace, DurableWorkspace)
    await ctx.workspace.write_text('right.txt', 'R')
    return (await ctx.workspace.run(['pwd'])).stdout


@workflow.defn
class FreshWorkspaceWorkflow:
    @workflow.run
    async def run(self) -> dict[str, Any]:
        result = await fresh_agent.run('Use both tools.')
        assert isinstance(result.workspace, DurableWorkspace)
        entries = await result.workspace.list_dir('.')
        return {
            'output': result.output,
            'ref': _REF_ADAPTER.dump_python(result.workspace.ref, mode='json'),
            'response_refs': [
                _REF_ADAPTER.dump_python(message.workspace_ref, mode='json')
                for message in result.all_messages()
                if message.kind == 'response'
            ],
            'working_dir': await result.workspace.working_dir(),
            'resolved': await result.workspace.resolve('nested/file.txt'),
            'files': [entry.path for entry in entries],
            'hook': await result.workspace.read_text('hook.txt'),
        }


async def test_fresh_workspace_is_provisioned_once_and_shared_by_every_side(client: Client) -> None:
    _reset_provider()

    async with Worker(
        client, task_queue=TASK_QUEUE, workflows=[FreshWorkspaceWorkflow], plugins=[AgentPlugin(fresh_agent)]
    ):
        handle = await client.start_workflow(
            FreshWorkspaceWorkflow.run, id=f'{FreshWorkspaceWorkflow.__name__}-{uuid.uuid4()}', task_queue=TASK_QUEUE
        )
        output = await handle.result()
        history = await handle.fetch_history()

    assert output == snapshot(
        {
            'output': '{"write_left":"before_run in /remote","write_right":"ran:pwd"}',
            'ref': {'provider': 'remote', 'id': 'env-1'},
            'response_refs': [{'provider': 'remote', 'id': 'env-1'}, {'provider': 'remote', 'id': 'env-1'}],
            'working_dir': '/remote',
            'resolved': '/remote/nested/file.txt',
            'files': ['/remote/hook.txt', '/remote/left.txt', '/remote/right.txt'],
            'hook': 'before_run in /remote',
        }
    )
    # Exactly one environment for the whole run, created by the one `ensure` activity; the hook's
    # write and the workflow-side reads are activities, the tools' calls are not.
    assert list(_ENVIRONMENTS) == ['env-1']
    assert _PROVIDER_LOG[0] == 'create:env-1'
    assert 'create:' not in ' '.join(_PROVIDER_LOG[1:])
    assert _activity_names(history) == snapshot(
        [
            'agent__fresh__capability__workspace__call',
            'agent__fresh__capability__workspace__call',
            'agent__fresh__model_request',
            'agent__fresh__toolset__<agent>__call_tool',
            'agent__fresh__toolset__<agent>__call_tool',
            'agent__fresh__model_request',
            'agent__fresh__capability__workspace__call',
            'agent__fresh__capability__workspace__call',
        ]
    )

    replay = await Replayer(workflows=[FreshWorkspaceWorkflow], plugins=[PydanticAIPlugin()]).replay_workflow(history)
    assert replay.replay_failure is None
    # Replay dispatched nothing to the provider.
    assert list(_ENVIRONMENTS) == ['env-1']


# --- Policy wrappers come back inside the activity ----------------------------------------------

read_only_agent = Agent(
    TestModel(call_tools=['try_write']),
    name='read_only',
    capabilities=[RemoteWorkspaces(read_only=True), TemporalDurability(activity_config=BASE_ACTIVITY_CONFIG)],
)


@read_only_agent.tool
async def try_write(ctx: RunContext[Any]) -> str:
    assert isinstance(ctx.workspace, ReadOnlyWorkspace)
    try:
        await ctx.workspace.write_text('nope.txt', 'x')
    except WorkspaceReadOnlyError as error:
        return f'blocked: {error}'
    return 'wrote'  # pragma: no cover


@workflow.defn
class ReadOnlyWorkflow:
    @workflow.run
    async def run(self, write_after: bool) -> str:
        result = await read_only_agent.run('Try to write.')
        assert isinstance(result.workspace, DurableWorkspace)
        assert isinstance(result.workspace.wrapped, ReadOnlyWorkspace)
        if write_after:
            await result.workspace.make_dir('sub')
        return result.output


async def test_read_only_policy_is_enforced_inside_the_activity_and_from_the_workflow(client: Client) -> None:
    _reset_provider()

    async with Worker(
        client, task_queue=TASK_QUEUE, workflows=[ReadOnlyWorkflow], plugins=[AgentPlugin(read_only_agent)]
    ):
        output = await client.execute_workflow(
            ReadOnlyWorkflow.run, False, id=f'{ReadOnlyWorkflow.__name__}-{uuid.uuid4()}', task_queue=TASK_QUEUE
        )
        assert output == snapshot(
            '{"try_write":"blocked: This workspace is read-only: running commands and modifying files are disabled. Reading files, listing directories, and checking that paths exist are allowed."}'
        )

        # From workflow code the refusal crosses the activity boundary as data and is re-raised as
        # the same `WorkspaceReadOnlyError`, which fails the workflow instead of the workflow task.
        with pytest.raises(WorkflowFailureError) as exc_info:
            await client.execute_workflow(
                ReadOnlyWorkflow.run, True, id=f'{ReadOnlyWorkflow.__name__}-{uuid.uuid4()}', task_queue=TASK_QUEUE
            )
    cause = _workflow_failure_cause(exc_info.value)
    assert cause.type == 'WorkspaceReadOnlyError'
    assert cause.message.startswith('This workspace is read-only')


# --- Binary content and expected errors cross the activity boundary ----------------------------

binary_agent = Agent(
    TestModel(),
    name='binary',
    capabilities=[RemoteWorkspaces(), TemporalDurability(activity_config=BASE_ACTIVITY_CONFIG)],
)

_BINARY = b'\xff\xfe\x00\x01binary\x80'


@workflow.defn
class BinaryWorkflow:
    @workflow.run
    async def run(self) -> dict[str, Any]:
        result = await binary_agent.run('Nothing to do.')
        workspace = result.workspace
        await workspace.write_bytes('blob.bin', _BINARY)
        await workspace.make_dir('sub')
        await workspace.write_text('sub/gone.txt', 'x')
        await workspace.remove('sub/gone.txt')
        entry = await workspace.stat('blob.bin')
        try:
            await workspace.stat('sub/gone.txt')
        except FileNotFoundError as error:
            stat_missing = f'{type(error).__name__}:{error}'
        else:  # pragma: no cover
            stat_missing = 'none'
        try:
            await workspace.read_text('blob.bin')
        except UnicodeDecodeError as error:
            decode_error = f'{type(error).__name__}:{error.reason}:{error.object == _BINARY}'
        else:  # pragma: no cover
            decode_error = 'none'
        try:
            await workspace.remove('missing.txt')
        except FileNotFoundError as error:
            missing = f'{type(error).__name__}:{error}'
        else:  # pragma: no cover
            missing = 'none'
        try:
            await workspace.run('echo hi')
        except TypeError as error:
            argument_error = f'{type(error).__name__}:{error}'
        else:  # pragma: no cover
            argument_error = 'none'
        return {
            'round_trip': (await workspace.read_bytes('blob.bin')) == _BINARY,
            'size': entry.size,
            'exists': [await workspace.exists('blob.bin'), await workspace.exists('nope')],
            'decode_error': decode_error,
            'missing': missing,
            'stat_missing': stat_missing,
            'argument_error': argument_error,
        }


async def test_binary_content_and_expected_errors_cross_the_activity_boundary(client: Client) -> None:
    _reset_provider()

    async with Worker(client, task_queue=TASK_QUEUE, workflows=[BinaryWorkflow], plugins=[AgentPlugin(binary_agent)]):
        output = await client.execute_workflow(
            BinaryWorkflow.run, id=f'{BinaryWorkflow.__name__}-{uuid.uuid4()}', task_queue=TASK_QUEUE
        )

    assert output == snapshot(
        {
            'round_trip': True,
            'size': 11,
            'exists': [True, False],
            'decode_error': 'UnicodeDecodeError:invalid start byte:True',
            'missing': 'FileNotFoundError:/remote/missing.txt',
            'stat_missing': 'FileNotFoundError:/remote/sub/gone.txt',
            'argument_error': 'TypeError:a shell string needs `shell=True`, an argv sequence needs `shell=False`',
        }
    )
    assert _ENVIRONMENTS['env-1']['/remote/blob.bin'] == _BINARY


@workflow.defn
class LargeWriteWorkflow:
    @workflow.run
    async def run(self) -> str:
        result = await binary_agent.run('Nothing to do.')
        try:
            await result.workspace.write_bytes('big.bin', b'x' * 2_000_000)
        except UserError as error:
            return str(error)
        return 'unexpected success'


async def test_large_workflow_write_fails_before_scheduling_activity(client: Client) -> None:
    _reset_provider()
    async with Worker(
        client, task_queue=TASK_QUEUE, workflows=[LargeWriteWorkflow], plugins=[AgentPlugin(binary_agent)]
    ):
        message = await client.execute_workflow(
            LargeWriteWorkflow.run,
            id=f'{LargeWriteWorkflow.__name__}-{uuid.uuid4()}',
            task_queue=TASK_QUEUE,
            execution_timeout=timedelta(seconds=10),
        )
    assert 'too large for Temporal' in message
    assert 'big.bin' not in _ENVIRONMENTS['env-1']


async def test_unicode_encode_error_crosses_workspace_boundary() -> None:
    class EncodingBackend(RemoteBackend):
        async def working_dir(self) -> str:
            return '/remote'

        async def write_bytes(self, path: str, data: bytes) -> None:
            raise UnicodeEncodeError('ascii', 'café', 3, 4, 'ordinal not in range')

    result = await execute_call(Workspace(EncodingBackend(None)), WorkspaceCall(method='write_bytes', path='x'))
    assert result.error is not None
    with pytest.raises(UnicodeEncodeError) as exc_info:
        raise_error(result.error)
    assert (exc_info.value.encoding, exc_info.value.object, exc_info.value.start, exc_info.value.end) == (
        'ascii',
        'café',
        3,
        4,
    )


async def test_workspace_os_error_preserves_type_and_both_filenames() -> None:
    class RenameBackend(RemoteBackend):
        async def working_dir(self) -> str:
            return '/remote'

        async def remove(self, path: str) -> None:
            raise FileNotFoundError(errno.ENOENT, 'No such file', path, None, '/remote/new')

    workspace = Workspace(RenameBackend(None))
    with pytest.raises(FileNotFoundError) as plain:
        await workspace.remove('old')
    result = await execute_call(workspace, WorkspaceCall(method='remove', path='old'))
    assert result.error is not None
    with pytest.raises(FileNotFoundError) as durable:
        raise_error(result.error)
    assert type(durable.value) is type(plain.value)
    assert durable.value.errno == plain.value.errno
    assert durable.value.filename2 == plain.value.filename2
    assert str(durable.value) == str(plain.value)


async def test_deterministic_os_error_crosses_workspace_boundary_without_retry() -> None:
    class LongNameBackend(RemoteBackend):
        async def working_dir(self) -> str:
            return '/remote'

        async def read_bytes(self, path: str) -> bytes:
            raise OSError(errno.ENAMETOOLONG, 'File name too long', path)

    result = await execute_call(Workspace(LongNameBackend(None)), WorkspaceCall(method='read_bytes', path='long-name'))
    assert result.error is not None
    with pytest.raises(OSError) as exc_info:
        raise_error(result.error)
    assert exc_info.value.errno == errno.ENAMETOOLONG
    assert exc_info.value.filename == '/remote/long-name'


# --- An uncaught workspace error fails the workflow instead of hanging it ----------------------


class ReadMissingInHook(AbstractCapability[Any]):
    async def before_run(self, ctx: RunContext[Any]) -> None:
        await ctx.workspace.read_text('missing.txt')


uncaught_agent = Agent(
    TestModel(),
    name='uncaught',
    capabilities=[ReadMissingInHook(), RemoteWorkspaces(), TemporalDurability(activity_config=BASE_ACTIVITY_CONFIG)],
)


@workflow.defn
class UncaughtErrorWorkflow:
    @workflow.run
    async def run(self) -> str:
        return (await uncaught_agent.run('Nothing to do.')).output


async def test_uncaught_workspace_error_in_workflow_code_fails_the_workflow(client: Client) -> None:
    _reset_provider()

    async with Worker(
        client, task_queue=TASK_QUEUE, workflows=[UncaughtErrorWorkflow], plugins=[AgentPlugin(uncaught_agent)]
    ):
        with pytest.raises(WorkflowFailureError) as exc_info:
            await client.execute_workflow(
                UncaughtErrorWorkflow.run,
                id=f'{UncaughtErrorWorkflow.__name__}-{uuid.uuid4()}',
                task_queue=TASK_QUEUE,
                execution_timeout=timedelta(seconds=30),
            )

    cause = _workflow_failure_cause(exc_info.value)
    assert (cause.type, cause.message) == ('FileNotFoundError', '/remote/missing.txt')


# --- `LocalWorkspace` end to end ---------------------------------------------------------------

# Resolved from the environment rather than `tempfile`, which the workflow sandbox re-imports and
# which probes the filesystem the first time it is asked for a directory.
_LOCAL_DIR = os.path.join(os.environ.get('TMPDIR', '/tmp'), 'pydantic_ai_temporal_local_workspace')

local_agent = Agent(
    TestModel(call_tools=['write_note']),
    name='local',
    capabilities=[LocalWorkspace(_LOCAL_DIR), TemporalDurability(activity_config=BASE_ACTIVITY_CONFIG)],
)


@local_agent.tool
async def write_note(ctx: RunContext[Any]) -> str:
    await ctx.workspace.write_text('note.txt', 'on disk')
    return (await ctx.workspace.run(['cat', 'note.txt'])).stdout


@workflow.defn
class LocalWorkspaceWorkflow:
    @workflow.run
    async def run(self) -> dict[str, Any]:
        result = await local_agent.run('Write the note.')
        return {
            'output': result.output,
            'ref': _REF_ADAPTER.dump_python(result.workspace.ref, mode='json'),
            'working_dir': await result.workspace.working_dir(),
            'note': await result.workspace.read_text('note.txt'),
        }


async def test_local_workspace_end_to_end(client: Client) -> None:
    shutil.rmtree(_LOCAL_DIR, ignore_errors=True)
    Path(_LOCAL_DIR).mkdir(parents=True)
    try:
        async with Worker(
            client, task_queue=TASK_QUEUE, workflows=[LocalWorkspaceWorkflow], plugins=[AgentPlugin(local_agent)]
        ):
            handle = await client.start_workflow(
                LocalWorkspaceWorkflow.run,
                id=f'{LocalWorkspaceWorkflow.__name__}-{uuid.uuid4()}',
                task_queue=TASK_QUEUE,
            )
            output = await handle.result()
            history = await handle.fetch_history()

        canonical = os.path.realpath(_LOCAL_DIR)
        assert output == snapshot(
            {
                'output': '{"write_note":"on disk"}',
                'ref': {'provider': 'local', 'id': IsStr()},
                'working_dir': IsStr(),
                'note': 'on disk',
            }
        )
        assert output['ref']['id'] == _LOCAL_DIR
        assert output['working_dir'] == canonical
        assert (Path(_LOCAL_DIR) / 'note.txt').read_text() == 'on disk'
        assert _activity_names(history) == snapshot(
            [
                'agent__local__capability__workspace__call',
                'agent__local__model_request',
                'agent__local__toolset__<agent>__call_tool',
                'agent__local__model_request',
                'agent__local__capability__workspace__call',
            ]
        )
    finally:
        shutil.rmtree(_LOCAL_DIR, ignore_errors=True)


# --- `workspace=` inside the workflow ----------------------------------------------------------

explicit_agent = Agent(
    TestModel(call_tools=['read_seed']),
    name='explicit',
    capabilities=[RemoteWorkspaces(), TemporalDurability(activity_config=BASE_ACTIVITY_CONFIG)],
)


@explicit_agent.tool
async def read_seed(ctx: RunContext[Any]) -> str:
    return await ctx.workspace.read_text('seed.txt')


ExplicitKind = Literal[
    'ref', 'live_with_ref', 'live_read_only', 'live_fresh', 'foreign_ref', 'previous_result', 'dead_ref'
]


@workflow.defn
class ExplicitWorkspaceWorkflow:
    @workflow.run
    async def run(self, kind: ExplicitKind) -> dict[str, Any]:
        seeded = WorkspaceRef(provider='remote', id='seeded')
        if kind == 'ref':
            workspace: Any = seeded
        elif kind == 'live_with_ref':
            workspace = RemoteBackend(seeded)
        elif kind == 'live_read_only':
            workspace = ReadOnlyWorkspace(Workspace(RemoteBackend(seeded)))
        elif kind == 'live_fresh':
            workspace = RemoteBackend(None)
        elif kind == 'foreign_ref':
            workspace = WorkspaceRef(provider='other', id='x')
        elif kind == 'dead_ref':
            workspace = WorkspaceRef(provider='remote', id='expired')
        else:
            workspace = (await explicit_agent.run('Read the seed.', workspace=seeded)).workspace
        result = await explicit_agent.run('Read the seed.', workspace=workspace)
        return {
            'output': result.output,
            'ref': _REF_ADAPTER.dump_python(result.workspace.ref, mode='json'),
        }


@pytest.mark.parametrize('kind', ['ref', 'live_with_ref', 'previous_result'])
async def test_explicit_workspace_that_a_capability_recognizes_attaches(client: Client, kind: ExplicitKind) -> None:
    _reset_provider()
    _ENVIRONMENTS['seeded'] = {'/remote/seed.txt': b'seed'}

    async with Worker(
        client, task_queue=TASK_QUEUE, workflows=[ExplicitWorkspaceWorkflow], plugins=[AgentPlugin(explicit_agent)]
    ):
        output = await client.execute_workflow(
            ExplicitWorkspaceWorkflow.run,
            kind,
            id=f'{ExplicitWorkspaceWorkflow.__name__}-{kind}-{uuid.uuid4()}',
            task_queue=TASK_QUEUE,
        )

    assert output == snapshot({'output': '{"read_seed":"seed"}', 'ref': {'provider': 'remote', 'id': 'seeded'}})
    assert list(_ENVIRONMENTS) == ['seeded']
    assert 'create:' not in ' '.join(_PROVIDER_LOG)


@pytest.mark.parametrize(
    ('kind', 'message'),
    [
        (
            'live_fresh',
            'A live workspace cannot be passed to `workspace=` inside a Temporal workflow: a backend or wrapper '
            'cannot cross the durable boundary, and it has no `WorkspaceRef` yet, so no capability could reattach '
            'to its environment. Pass a `WorkspaceRef` (or `result.workspace` from a run on this agent) and attach '
            'a capability whose `get_workspace` supplies it; a policy wrapper such as `ReadOnlyWorkspace` belongs '
            'on that capability (for example `LocalWorkspace(..., read_only=True)`), not around the argument.',
        ),
        (
            'foreign_ref',
            "Workspace `other:x` was passed to the run, but none of the agent's workspace capabilities recognized it.",
        ),
    ],
)
async def test_explicit_workspace_nobody_can_rebuild_is_rejected(
    client: Client, kind: ExplicitKind, message: str
) -> None:
    _reset_provider()

    async with Worker(
        client, task_queue=TASK_QUEUE, workflows=[ExplicitWorkspaceWorkflow], plugins=[AgentPlugin(explicit_agent)]
    ):
        with pytest.raises(WorkflowFailureError) as exc_info:
            await client.execute_workflow(
                ExplicitWorkspaceWorkflow.run,
                kind,
                id=f'{ExplicitWorkspaceWorkflow.__name__}-{kind}-{uuid.uuid4()}',
                task_queue=TASK_QUEUE,
            )

    cause = _workflow_failure_cause(exc_info.value)
    assert (cause.type, cause.message) == ('UserError', message)
    assert _ENVIRONMENTS == {}


async def test_explicit_read_only_workspace_cannot_lose_its_policy(client: Client) -> None:
    _reset_provider()
    _ENVIRONMENTS['seeded'] = {'/remote/seed.txt': b'seed'}
    async with Worker(
        client, task_queue=TASK_QUEUE, workflows=[ExplicitWorkspaceWorkflow], plugins=[AgentPlugin(explicit_agent)]
    ):
        with pytest.raises(WorkflowFailureError) as exc_info:
            await client.execute_workflow(
                ExplicitWorkspaceWorkflow.run,
                'live_read_only',
                id=f'{ExplicitWorkspaceWorkflow.__name__}-read-only-{uuid.uuid4()}',
                task_queue=TASK_QUEUE,
            )
    cause = _workflow_failure_cause(exc_info.value)
    assert cause.type == 'UserError'
    assert 'read_only on the capability' in cause.message


async def test_a_dead_environment_fails_the_workflow_with_the_workspace_error(client: Client) -> None:
    """`ensure` cannot attach to an environment that no longer exists; the error crosses and fails the workflow."""
    _reset_provider()

    async with Worker(
        client, task_queue=TASK_QUEUE, workflows=[ExplicitWorkspaceWorkflow], plugins=[AgentPlugin(explicit_agent)]
    ):
        with pytest.raises(WorkflowFailureError) as exc_info:
            await client.execute_workflow(
                ExplicitWorkspaceWorkflow.run,
                'dead_ref',
                id=f'{ExplicitWorkspaceWorkflow.__name__}-dead-{uuid.uuid4()}',
                task_queue=TASK_QUEUE,
                execution_timeout=timedelta(seconds=30),
            )

    cause = _workflow_failure_cause(exc_info.value)
    assert (cause.type, cause.message) == ('WorkspaceUnavailableError', "environment 'expired' does not exist")


# --- A capability that creates environments must recognize their refs ---------------------------


class AmnesiacWorkspaces(AbstractCapability[Any]):
    def get_workspace(self, ctx: RunContext[Any], *, ref: WorkspaceRef | None) -> WorkspaceBackend | None:
        return RemoteBackend(None) if ref is None else None


amnesiac_agent = Agent(
    TestModel(),
    name='amnesiac',
    capabilities=[AmnesiacWorkspaces(), TemporalDurability(activity_config=BASE_ACTIVITY_CONFIG)],
)


@workflow.defn
class AmnesiacWorkflow:
    @workflow.run
    async def run(self) -> str:
        return (await amnesiac_agent.run('Nothing to do.')).output


async def test_a_creating_capability_must_recognize_the_ref_it_created(client: Client) -> None:
    """The `ensure` activity created an environment the workflow cannot rebuild a workspace for."""
    _reset_provider()

    async with Worker(
        client, task_queue=TASK_QUEUE, workflows=[AmnesiacWorkflow], plugins=[AgentPlugin(amnesiac_agent)]
    ):
        with pytest.raises(WorkflowFailureError) as exc_info:
            await client.execute_workflow(
                AmnesiacWorkflow.run, id=f'{AmnesiacWorkflow.__name__}-{uuid.uuid4()}', task_queue=TASK_QUEUE
            )

    cause = _workflow_failure_cause(exc_info.value)
    assert cause.type == 'UserError'
    assert cause.message == snapshot(
        "No capability can supply workspace 'env-1' from provider 'remote', which the run just created. A "
        '`get_workspace` hook that creates an environment must also recognize its ref.'
    )


async def test_activity_refuses_a_ref_no_worker_capability_recognizes() -> None:
    """An `ensure` activity for a ref the worker's capabilities cannot rebuild fails with an explanation."""
    agent = Agent(TestModel(), name='ctx', capabilities=[RemoteWorkspaces(), TemporalDurability()])
    durability = TemporalDurability.from_agent(agent)
    assert durability is not None
    ensure = next(
        item
        for item in durability.temporal_activities
        if ActivityDefinition.must_from_callable(item).name == 'agent__ctx__capability__workspace__call'  # pyright: ignore[reportUnknownMemberType]
    )
    wire = _WorkspaceCallWire(
        call=WorkspaceCall(method='ensure'),
        ref=WorkspaceRef(provider='other', id='x'),
        serialized_run_context={'run_id': 'r', 'workspace_ref': {'provider': 'other', 'id': 'x'}},
    )

    with pytest.raises(UserError, match="No capability can supply the workspace 'x' from provider 'other'"):
        await ActivityEnvironment().run(ensure, wire, None)


def test_activity_run_context_rebuilds_the_workspace_from_the_serialized_ref() -> None:
    """The restore path on its own: a custom context that sets `workspace` wins, a missing capability explains."""
    agent = Agent(TestModel(), name='ctx', capabilities=[RemoteWorkspaces(read_only=True), TemporalDurability()])
    durability = TemporalDurability.from_agent(agent)
    assert durability is not None
    serialized = {'run_id': 'r', 'workspace_ref': {'provider': 'remote', 'id': 'seeded'}}

    restored = durability.deserialize_operation_run_context(serialized, None)
    assert isinstance(restored.workspace, ReadOnlyWorkspace)
    assert restored.workspace.ref == WorkspaceRef(provider='remote', id='seeded')

    class OwnWorkspace(TemporalRunContext[Any]):
        @classmethod
        def deserialize_run_context(cls, ctx: dict[str, Any], deps: Any) -> OwnWorkspace:
            return cls(**{**ctx, 'workspace': Workspace(RemoteBackend(None))}, deps=deps)

    own = Agent(
        TestModel(), name='ctx', capabilities=[RemoteWorkspaces(), TemporalDurability(run_context_type=OwnWorkspace)]
    )
    own_durability = TemporalDurability.from_agent(own)
    assert own_durability is not None
    own_ctx = own_durability.deserialize_operation_run_context(serialized, None)
    assert type(own_ctx.workspace) is Workspace and own_ctx.workspace.ref is None

    no_ref = durability.deserialize_operation_run_context({'run_id': 'r'}, None)
    assert no_ref.workspace.ref is None

    foreign = durability.deserialize_operation_run_context(
        {'run_id': 'r', 'workspace_ref': {'provider': 'other', 'id': 'x'}}, None
    )
    assert foreign.workspace.ref is None
