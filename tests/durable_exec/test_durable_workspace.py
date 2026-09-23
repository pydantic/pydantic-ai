"""The durable workspace layer, exercised through a fake in-process engine.

Not VCR tests: nothing here talks to a model provider, and the behavior under test is where each
workspace call executes (in a durable unit, or directly inside one), which a recording could not
observe. Each engine's own suite covers the same scenarios against its real runtime; this file pins
the engine-agnostic contract the base owns: one `ensure` unit per run, unit-per-call dispatch from
workflow code, direct calls inside units, errors crossing as data, and what `workspace=` accepts
inside a container.
"""

from __future__ import annotations

import pickle
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

import anyio
import pytest
from inline_snapshot import snapshot

from pydantic_ai import Agent, RunContext, UserError
from pydantic_ai.capabilities import AbstractCapability, Capability, LocalWorkspace, WrapperCapability
from pydantic_ai.capabilities.abstract import CapabilityOrdering, WrapRunHandler
from pydantic_ai.durable_exec import (
    JSON_CODEC,
    BaseDurabilityCapability,
    CallableOperationBackend,
    DurabilityEngineSpec,
    DurableOperationId,
    JournalOperationNamer,
    OperationConfigRole,
    RoleBasedOperationConfig,
    WorkspaceOperationId,
)
from pydantic_ai.durable_exec._workspace import (
    DurableWorkspace,
    ReadBytesArguments,
    RunArguments,
    UnicodeDecodeDetails,
    WorkspaceOperationError,
    WorkspaceOperationResult,
    WriteBytesArguments,
    raise_operation_error,
    workspace_operation_error,
)
from pydantic_ai.models.test import TestModel
from pydantic_ai.run import AgentRunResult
from pydantic_ai.usage import RunUsage
from pydantic_ai.workspaces import (
    FileEntry,
    ReadOnlyWorkspace,
    Workspace,
    WorkspaceBackend,
    WorkspaceError,
    WorkspaceReadOnlyError,
    WorkspaceRef,
    WorkspaceTimeoutError,
    WorkspaceUnavailableError,
)

from ..workspace_fakes import FakeWorkspace, InMemoryProvider, WorkspaceCapability

pytestmark = pytest.mark.anyio


class _Backend(CallableOperationBackend[dict[str, Any]]):
    def __init__(self, durability: FakeDurability) -> None:
        super().__init__(
            namer=JournalOperationNamer(durability.name),
            config=RoleBasedOperationConfig(model={}, event={}, capability={}, tool={}),
        )
        self._durability = durability

    async def execute(
        self,
        *,
        operation_id: DurableOperationId,
        name: str,
        body: Callable[[], Awaitable[object]],
        cache_key: tuple[object, ...],
        config: dict[str, Any],
    ) -> object:
        self._durability.units.append(name)
        return await body()


class FakeDurability(BaseDurabilityCapability[Any]):
    """An in-process engine that journals nothing but records which units ran."""

    engine_spec = DurabilityEngineSpec(
        engine_name='Fake',
        durable_unit_noun='unit',
        durable_container_noun='journal',
        codec=JSON_CODEC,
    )
    in_container = True

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.units: list[str] = []

    @property
    def in_durable_context(self) -> bool:
        return self.in_container

    def get_durable_operation_backend(self) -> _Backend:
        return _Backend(self)


class TransparentDurability(FakeDurability):
    in_container = False


class RebuildingDurability(FakeDurability):
    """An engine whose units rebuild the workspace elsewhere, so per-run policy must match."""

    engine_spec = DurabilityEngineSpec(
        engine_name='Fake',
        durable_unit_noun='unit',
        durable_container_noun='journal',
        codec=JSON_CODEC,
        workspace_rebuilt_in_unit=True,
    )


class FreshWorkspaces(AbstractCapability[Any]):
    """A factory-style supplier: a new backend per call, attaching when a ref is given."""

    def __init__(self) -> None:
        self.backends: list[FakeWorkspace] = []

    def get_workspace(self, ctx: RunContext[Any], *, ref: WorkspaceRef | None) -> WorkspaceBackend | None:
        if ref is not None and ref.provider != 'fake':
            return None
        backend = FakeWorkspace(ref.id if ref is not None else 'fresh', ref=ref)
        self.backends.append(backend)
        return backend


def _run_context() -> RunContext[None]:
    return RunContext(deps=None, model=TestModel(), usage=RunUsage())


def _workspace_units(durability: FakeDurability) -> list[str]:
    # `for_agent` binds a shallow copy, so the user's instance shares the recorded list but not the name.
    return [name.split('__workspace__')[1] for name in durability.units if '__workspace__' in name]


async def test_ensure_runs_once_before_hooks_and_every_side_shares_one_environment() -> None:
    """`ensure` runs before `before_run`, a hook's call is a unit, a tool's call is direct, and the result's is a unit."""
    supplier = FreshWorkspaces()
    durability = FakeDurability()
    seen: list[str] = []

    class Hook(AbstractCapability[Any]):
        async def before_run(self, ctx: RunContext[Any]) -> None:
            ref = ctx.workspace.ref
            assert ref is not None
            seen.append(f'before_run:{ref.id}')
            await ctx.workspace.write_text('note.txt', 'from the hook')

    agent = Agent(TestModel(call_tools=['probe']), name='ws', capabilities=[Hook(), supplier, durability])

    @agent.tool
    async def probe(ctx: RunContext[Any]) -> str:
        assert isinstance(ctx.workspace, DurableWorkspace)
        return await ctx.workspace.read_text('note.txt')

    result = await agent.run('go')

    assert result.output == snapshot('{"probe":"from the hook"}')
    assert seen == ['before_run:fake-fresh']
    assert result.workspace.ref == WorkspaceRef(provider='fake', id='fake-fresh')
    # The result's workspace still dispatches units after the run, inside the container.
    assert await result.workspace.exists('note.txt') is True
    assert _workspace_units(durability) == snapshot(['ensure', 'write_text', 'exists'])
    # `ensure` created one environment on the live backend an in-process unit shares with the
    # container, so nothing was rebuilt or reattached.
    assert [(backend.name, backend.create_calls, backend.attach_calls) for backend in supplier.backends] == snapshot(
        [('fresh', 1, 0)]
    )


async def test_working_dir_and_resolve_answer_locally_after_ensure() -> None:
    durability = FakeDurability()
    agent = Agent(TestModel(), name='ws', capabilities=[WorkspaceCapability(), durability])

    result = await agent.run('go')

    assert await result.workspace.working_dir() == '/workspace'
    assert await result.workspace.resolve('a/b') == '/workspace/a/b'
    assert await result.workspace.resolve('/abs/../x') == '/x'
    assert _workspace_units(durability) == ['ensure']


async def test_parallel_first_uses_share_one_ensure() -> None:
    """Two hooks touching the workspace at once, before the companion's eager `ensure`, take the lazy path once."""
    supplier = FreshWorkspaces()
    durability = FakeDurability()

    class EarlyWrapRun(AbstractCapability[Any]):
        # Listed before the durability's companion, so this wraps outside it and its
        # pre-handler code runs before the eager `ensure`.
        def get_ordering(self) -> CapabilityOrdering:
            return CapabilityOrdering(position='outermost')

        async def wrap_run(self, ctx: RunContext[Any], *, handler: WrapRunHandler) -> AgentRunResult[Any]:
            assert ctx.workspace.ref is None
            async with anyio.create_task_group() as tg:
                tg.start_soon(ctx.workspace.write_text, 'a.txt', 'a')
                tg.start_soon(ctx.workspace.write_text, 'b.txt', 'b')
                tg.start_soon(ctx.workspace.working_dir)
            return await handler()

    agent = Agent(TestModel(), name='ws', capabilities=[EarlyWrapRun(), supplier, durability])
    result = await agent.run('go')

    assert result.workspace.ref == WorkspaceRef(provider='fake', id='fake-fresh')
    assert _workspace_units(durability) == snapshot(['ensure', 'write_text', 'write_text'])
    assert sum(backend.create_calls for backend in supplier.backends) == 1


async def test_policy_wrapper_is_enforced_inside_the_unit(tmp_path: Path) -> None:
    durability = FakeDurability()
    agent = Agent(TestModel(), name='ws', capabilities=[LocalWorkspace(tmp_path, read_only=True), durability])

    result = await agent.run('go')

    assert isinstance(result.workspace, DurableWorkspace)
    assert isinstance(result.workspace.wrapped, ReadOnlyWorkspace)
    with pytest.raises(UserError, match='read-only'):
        await result.workspace.write_text('x.txt', 'x')
    assert _workspace_units(durability) == ['ensure', 'write_text']


async def test_expected_errors_cross_as_data_and_re_raise(tmp_path: Path) -> None:
    durability = FakeDurability()
    agent = Agent(TestModel(), name='ws', capabilities=[LocalWorkspace(tmp_path), durability])
    result = await agent.run('go')
    (tmp_path / 'latin.txt').write_bytes(b'caf\xe9')

    with pytest.raises(FileNotFoundError):
        await result.workspace.read_bytes('missing.txt')
    with pytest.raises(UnicodeDecodeError) as error:
        await result.workspace.read_text('latin.txt')
    assert error.value.object == b'caf\xe9'
    with pytest.raises(TypeError):
        await result.workspace.run('echo hi')
    with pytest.raises(ValueError, match='offset'):
        await result.workspace.read_file('latin.txt', offset=0)
    # Every failure was a completed unit, not a failed one.
    assert _workspace_units(durability) == ['ensure', 'read_bytes', 'read_text', 'run', 'read_file']


async def test_backend_is_not_reachable_from_workflow_code() -> None:
    supplier = WorkspaceCapability()
    agent = Agent(TestModel(), name='ws', capabilities=[supplier, FakeDurability()])
    result = await agent.run('go')

    with pytest.raises(UserError, match=r'`workspace\.backend` is not available in durable workflow code'):
        result.workspace.backend
    # Nested wrappers are still reachable for callers that only want to inspect the chain.
    assert isinstance(result.workspace, DurableWorkspace)
    assert result.workspace.wrapped.backend is supplier.backend


async def test_ensure_rejects_a_backend_without_a_ref() -> None:
    class RefLess(FakeWorkspace):
        @property
        def ref(self) -> None:
            return None

    agent = Agent(TestModel(), name='ws', capabilities=[WorkspaceCapability(RefLess('refless')), FakeDurability()])

    with pytest.raises(UserError, match='must report a `ref` once any operation has completed'):
        await agent.run('go')


async def test_an_unexpected_backend_error_fails_the_unit() -> None:
    class Flaky(FakeWorkspace):
        async def read_bytes(self, path: str) -> bytes:
            raise ConnectionError('provider hiccup')

    agent = Agent(TestModel(), name='ws', capabilities=[WorkspaceCapability(Flaky('flaky')), FakeDurability()])
    result = await agent.run('go')

    with pytest.raises(ConnectionError, match='provider hiccup'):
        await result.workspace.read_text('x.txt')


async def test_ensure_failure_surfaces_as_the_workspace_error() -> None:
    class Dead(FakeWorkspace):
        async def working_dir(self) -> str:
            raise WorkspaceUnavailableError('environment expired')

    agent = Agent(TestModel(), name='ws', capabilities=[WorkspaceCapability(Dead('dead')), FakeDurability()])

    with pytest.raises(WorkspaceUnavailableError, match='environment expired'):
        await agent.run('go')


async def test_durable_agent_outside_the_container_keeps_the_selected_workspace() -> None:
    supplier = WorkspaceCapability()
    observed: list[Workspace] = []
    agent = Agent(TestModel(call_tools=['probe']), name='ws', capabilities=[supplier, TransparentDurability()])

    @agent.tool
    async def probe(ctx: RunContext[Any]) -> str:
        observed.append(ctx.workspace)
        return 'ok'

    result = await agent.run('go')

    assert type(result.workspace) is Workspace
    assert result.workspace is observed[0]
    assert result.workspace.backend is supplier.backend
    durability = TransparentDurability.from_agent(agent)
    assert durability is not None
    assert [name for name in durability.units if '__workspace__' in name] == []


async def test_result_workspace_calls_directly_once_the_container_has_ended() -> None:
    """Every method of a `DurableWorkspace` reaches the wrapped workspace once no container is active."""
    supplier = WorkspaceCapability(FakeWorkspace('direct', files={'/workspace/seed.txt': b'seed\n'}))
    durability = FakeDurability()
    agent = Agent(TestModel(), name='ws', capabilities=[supplier, durability])
    result = await agent.run('go')
    workspace = result.workspace
    assert isinstance(workspace, DurableWorkspace)

    bound = FakeDurability.from_agent(agent)
    assert bound is not None
    bound.in_container = False
    assert workspace.backend is workspace.wrapped
    await workspace.write_text('after.txt', 'done')
    await workspace.write_bytes('after.bin', b'\x00')
    await workspace.make_dir('sub')
    assert await workspace.read_text('after.txt') == 'done'
    assert await workspace.read_bytes('after.bin') == b'\x00'
    assert (await workspace.read_file('seed.txt')).lines == ('seed',)
    assert (await workspace.stat('after.txt')).size == 4
    assert {entry.name for entry in await workspace.list_dir('.')} >= {'after.txt', 'after.bin', 'seed.txt'}
    assert (await workspace.run(['true'])).stdout == 'connected'
    assert await workspace.working_dir() == '/workspace'
    await workspace.remove('after.bin')
    assert await workspace.exists('after.bin') is False
    assert _workspace_units(durability) == ['ensure']

    # A wrapper that never ran `ensure` asks the wrapped workspace for its working directory too.
    unensured = DurableWorkspace(
        Workspace(FakeWorkspace('never')), durability=TransparentDurability(), ctx=_run_context()
    )
    assert await unensured.working_dir() == '/workspace'


async def test_no_units_are_bound_without_a_construction_time_supplier() -> None:
    durability = FakeDurability()
    Agent(TestModel(), name='ws', capabilities=[durability])
    bound = FakeDurability.from_agent(Agent(TestModel(), name='ws', capabilities=[durability]))
    assert bound is not None
    assert bound._bound_workspace_operations == {}  # pyright: ignore[reportPrivateUsage]

    # Attaching one per run inside the container is then refused rather than run non-durably.
    supplier = WorkspaceCapability()
    with pytest.raises(UserError, match='no capability supplied workspaces when the agent was constructed'):
        await Agent(TestModel(), name='ws', capabilities=[durability]).run('go', capabilities=[supplier])


async def test_a_wrapper_capability_supplying_workspaces_binds_the_units() -> None:
    class SuppliesThroughWrapper(WrapperCapability[Any]):
        def get_workspace(self, ctx: RunContext[Any], *, ref: WorkspaceRef | None) -> WorkspaceBackend | None:
            return FakeWorkspace('wrapped')

    durability = FakeDurability()
    agent = Agent(TestModel(), name='ws', capabilities=[SuppliesThroughWrapper(Capability(id='inner')), durability])

    result = await agent.run('go')

    assert result.workspace.ref == WorkspaceRef(provider='fake', id='fake-wrapped')
    assert _workspace_units(durability) == ['ensure']


async def test_explicit_workspace_argument_inside_the_container() -> None:
    """A live instance is only accepted for its identity, and only when a capability claims it."""
    supplier = FreshWorkspaces()
    durability = FakeDurability()
    agent = Agent(TestModel(), name='ws', capabilities=[supplier, durability])
    first = await agent.run('go')

    # A previous result's `DurableWorkspace` unwraps to its ref; the run attaches through the capability.
    second = await agent.run('again', workspace=first.workspace)
    assert second.workspace is not first.workspace
    assert second.workspace.ref == first.workspace.ref
    assert isinstance(second.workspace, DurableWorkspace)
    assert second.workspace.wrapped.backend is supplier.backends[-1]

    # A live instance whose ref a capability recognizes is replaced by the capability-built one.
    third = await agent.run('once more', workspace=FakeWorkspace('x', ref=WorkspaceRef(provider='fake', id='x')))
    assert third.workspace.ref == WorkspaceRef(provider='fake', id='x')
    assert supplier.backends[-1].attach_calls == 1

    with pytest.raises(UserError, match='it has no `WorkspaceRef` yet'):
        await agent.run('nope', workspace=FakeWorkspace('unborn'))
    with pytest.raises(UserError, match="no capability on this agent recognizes workspace 'y' from provider 'other'"):
        await agent.run('nope', workspace=FakeWorkspace('y', ref=WorkspaceRef(provider='other', id='y')))
    with pytest.raises(UserError, match='belongs on that capability'):
        await agent.run('nope', workspace=ReadOnlyWorkspace(Workspace(FakeWorkspace('policy'))))


async def test_forwarded_durable_workspace_without_a_ref_asks_for_a_fresh_environment() -> None:
    supplier = FreshWorkspaces()
    durability = FakeDurability()
    agent = Agent(TestModel(), name='ws', capabilities=[supplier, durability])
    unensured = DurableWorkspace(Workspace(FakeWorkspace('parent')), durability=durability, ctx=_run_context())

    result = await agent.run('go', workspace=unensured)

    assert result.workspace.ref == WorkspaceRef(provider='fake', id='fake-fresh')

    declining = Agent(TestModel(), name='ws', capabilities=[LocalWorkspace('/tmp'), durability])
    other = DurableWorkspace(Workspace(FakeWorkspace('parent')), durability=durability, ctx=_run_context())
    # `LocalWorkspace` supplies without a ref, so this claims a fresh local workspace instead.
    assert (await declining.run('go', workspace=other)).workspace.ref == WorkspaceRef(provider='local', id='/tmp')


async def test_per_run_policy_must_match_the_construction_tree_when_units_rebuild(tmp_path: Path) -> None:
    class ReadOnlyPerRun(LocalWorkspace[Any]):
        async def for_run(self, ctx: RunContext[Any]) -> AbstractCapability[Any]:
            return LocalWorkspace(self.working_dir, read_only=True)

    agent = Agent(TestModel(), name='ws', capabilities=[ReadOnlyPerRun(tmp_path), RebuildingDurability()])
    with pytest.raises(UserError, match=r'\(ReadOnlyWorkspace, Workspace\) differs from .* \(Workspace\)'):
        await agent.run('go')

    # The same replacement is fine for an engine whose units use the run's live workspace.
    result = await Agent(TestModel(), name='ws', capabilities=[ReadOnlyPerRun(tmp_path), FakeDurability()]).run('go')
    with pytest.raises(UserError, match='read-only'):
        await result.workspace.make_dir('sub')


async def test_sub_agent_run_from_a_unit_uses_the_forwarded_workspace_directly() -> None:
    """Inside a unit the container check may still hold; the unit marker keeps the sub-run direct."""
    supplier = WorkspaceCapability()
    durability = FakeDurability()
    child = Agent(TestModel(call_tools=['child_probe']), name='child', capabilities=[supplier, durability])
    parent = Agent(TestModel(call_tools=['delegate']), name='parent', capabilities=[supplier, durability])
    child_workspaces: list[Workspace] = []

    @child.tool
    async def child_probe(ctx: RunContext[Any]) -> str:
        child_workspaces.append(ctx.workspace)
        return await ctx.workspace.read_text('shared.txt')

    @parent.tool
    async def delegate(ctx: RunContext[Any]) -> str:
        await ctx.workspace.write_text('shared.txt', 'from parent')
        return (await child.run('go', workspace=ctx.workspace)).output

    result = await parent.run('go')

    assert result.output == snapshot('{"delegate":"{\\"child_probe\\":\\"from parent\\"}"}')
    # The child saw the parent's live durable workspace, and its calls went direct: one `ensure`
    # for the parent run, and no unit for anything the tools did.
    assert type(child_workspaces[0]) is DurableWorkspace
    assert [name for name in durability.units if '__workspace__' in name] == ['parent__workspace__ensure']


async def test_every_method_runs_as_a_unit_against_a_provider_environment() -> None:
    provider = InMemoryProvider()
    durability = FakeDurability()
    agent = Agent(TestModel(), name='ws', capabilities=[provider.capability(), durability])
    result = await agent.run('go')
    workspace = result.workspace

    await workspace.make_dir('sub')
    await workspace.write_text('sub/a.txt', 'alpha')
    await workspace.write_bytes('b.bin', b'\x00\x01')
    assert (await workspace.run(['ls'])).stdout == 'ran:ls'
    assert (await workspace.stat('sub/a.txt')).size == 5
    assert [entry.name for entry in await workspace.list_dir('.')] == ['b.bin', 'a.txt']
    assert (await workspace.read_file('sub/a.txt')).lines == ('alpha',)
    assert await workspace.exists('b.bin') is True
    await workspace.remove('b.bin')
    assert await workspace.exists('b.bin') is False
    with pytest.raises(FileNotFoundError):
        await workspace.remove('b.bin')
    with pytest.raises(FileNotFoundError):
        await workspace.stat('b.bin')
    with pytest.raises(FileNotFoundError):
        await workspace.read_bytes('b.bin')
    with pytest.raises(TypeError, match='shell'):
        await workspace.run('ls', shell=False)
    assert _workspace_units(durability) == snapshot(
        [
            'ensure',
            'make_dir',
            'write_text',
            'write_bytes',
            'run',
            'stat',
            'list_dir',
            'read_file',
            'exists',
            'remove',
            'exists',
            'remove',
            'stat',
            'read_bytes',
            'run',
        ]
    )
    assert provider.log == ['create:env-1']

    # A ref from another provider is declined, and a ref to a vanished environment cannot attach.
    with pytest.raises(UserError, match="No capability can supply workspace 'x'"):
        await agent.run('go', workspace=WorkspaceRef(provider='other', id='x'))
    with pytest.raises(WorkspaceUnavailableError, match="environment 'expired' does not exist"):
        await agent.run('go', workspace=WorkspaceRef(provider='fake', id='expired'))


def test_workspace_role_falls_back_to_the_capability_config() -> None:
    config = RoleBasedOperationConfig(model={'m': 1}, event={'e': 1}, capability={'c': 1}, tool={'t': 1})
    assert config.base('workspace', operation_id=WorkspaceOperationId('run')) == {'c': 1}
    explicit = RoleBasedOperationConfig(
        model={'m': 1}, event={'e': 1}, capability={'c': 1}, tool={'t': 1}, workspace={'w': 1}
    )
    assert explicit.base('workspace', operation_id=WorkspaceOperationId('run')) == {'w': 1}


def test_non_utf8_bytes_round_trip_through_json_and_pickle() -> None:
    raw = b'\xff\xfe\x00binary'
    params = WriteBytesArguments(path='/a', data=raw)
    dumped = JSON_CODEC.dump(WriteBytesArguments, params)
    # Pydantic's base64 mode uses the URL-safe alphabet.
    assert dumped == {'path': '/a', 'data': '__4AYmluYXJ5'}
    assert JSON_CODEC.load(WriteBytesArguments, dumped) == params
    assert pickle.loads(pickle.dumps(params)) == params

    result = WorkspaceOperationResult[bytes](value=raw)
    assert (
        JSON_CODEC.load(WorkspaceOperationResult[bytes], JSON_CODEC.dump(WorkspaceOperationResult[bytes], result))
        == result
    )
    entries = WorkspaceOperationResult[list[FileEntry]](value=[FileEntry(name='a', path='/a', is_dir=False, size=1)])
    assert (
        JSON_CODEC.load(
            WorkspaceOperationResult[list[FileEntry]],
            JSON_CODEC.dump(WorkspaceOperationResult[list[FileEntry]], entries),
        )
        == entries
    )
    assert JSON_CODEC.load(
        RunArguments, JSON_CODEC.dump(RunArguments, RunArguments(command=['ls', '-l']))
    ) == RunArguments(command=['ls', '-l'])
    assert JSON_CODEC.load(ReadBytesArguments, {'path': '/x'}) == ReadBytesArguments(path='/x')


@pytest.mark.parametrize(
    'error',
    [
        WorkspaceTimeoutError('slow', stdout='partial', stderr='err', timeout=1.5),
        WorkspaceUnavailableError('gone'),
        WorkspaceReadOnlyError('read-only'),
        WorkspaceError('broken'),
        FileNotFoundError('/missing'),
        NotADirectoryError('/file'),
        IsADirectoryError('/dir'),
        PermissionError('/root'),
        FileExistsError('/there'),
        UnicodeDecodeError('utf-8', b'\xff', 0, 1, 'invalid start byte'),
        NotImplementedError('no stat'),
        UserError('policy'),
        TypeError('shell'),
        ValueError('offset'),
    ],
)
def test_error_table_round_trips_every_kind(error: Exception) -> None:
    data = workspace_operation_error(error)
    assert data is not None
    restored = JSON_CODEC.load(WorkspaceOperationError, JSON_CODEC.dump(WorkspaceOperationError, data))
    with pytest.raises(type(error)) as raised:
        raise_operation_error(restored)
    assert str(raised.value) == str(error)
    assert type(raised.value) is type(error)
    if isinstance(error, WorkspaceTimeoutError):
        assert isinstance(raised.value, WorkspaceTimeoutError)
        assert (raised.value.stdout, raised.value.stderr, raised.value.timeout) == ('partial', 'err', 1.5)
    if isinstance(error, UnicodeDecodeError):
        assert isinstance(raised.value, UnicodeDecodeError)
        assert raised.value.object == b'\xff'


def test_unexpected_errors_fail_the_unit() -> None:
    assert workspace_operation_error(ConnectionError('flaky')) is None
    assert workspace_operation_error(OSError('other')) is None
    assert (
        WorkspaceOperationError(
            kind='unicode_decode',
            message='m',
            decode=UnicodeDecodeDetails(encoding='utf-8', object=b'', start=0, end=0, reason='r'),
        ).decode
        is not None
    )


async def test_run_never_ensures_without_an_attached_workspace() -> None:
    durability = FakeDurability()
    agent = Agent(TestModel(), name='ws', capabilities=[FreshWorkspaces(), durability])

    result = await agent.run('go', workspace=WorkspaceRef(provider='fake', id='known'))
    assert result.workspace.ref == WorkspaceRef(provider='fake', id='known')
    # A known ref still gets one `ensure`, which journals the working directory and attaches once.
    assert _workspace_units(durability) == ['ensure']


def test_journal_names_with_a_workspace_supplier() -> None:
    durability = FakeDurability()
    bound = FakeDurability.from_agent(
        Agent(TestModel(), name='compat', capabilities=[WorkspaceCapability(), durability])
    )
    assert bound is not None
    namer = JournalOperationNamer('compat')
    names = {
        namer.operation_name(WorkspaceOperationId(method))
        for method in bound._bound_workspace_operations  # pyright: ignore[reportPrivateUsage]
    }
    assert names == snapshot(
        {
            'compat__workspace__ensure',
            'compat__workspace__run',
            'compat__workspace__read_bytes',
            'compat__workspace__write_bytes',
            'compat__workspace__stat',
            'compat__workspace__list_dir',
            'compat__workspace__make_dir',
            'compat__workspace__remove',
            'compat__workspace__exists',
            'compat__workspace__read_text',
            'compat__workspace__write_text',
            'compat__workspace__read_file',
        }
    )


def test_operation_config_role_includes_workspace() -> None:
    role: OperationConfigRole = 'workspace'
    assert role == 'workspace'
