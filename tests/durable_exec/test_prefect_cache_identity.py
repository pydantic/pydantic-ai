"""Prefect task cache keys: which task runs replay which recorded result.

`PrefectDurability` runs each model request, tool call and durable operation as a Prefect task, and a flow retry
replays a task whose cache key it has recorded. The keys come in two kinds:

- Run-scoped: workspace calls, custom capability operations, a run's start time and its default run ID carry their
  run's identity and an ordinal counted per run (prefect.md "Durable Capability Operations", "Workspaces" and "Run
  and Conversation IDs"). The key of a run's k-th call is therefore the same however other runs' calls interleave
  with it, which a flow retry can change, and no two calls in a flow share a key.
- Run-agnostic: model requests, tool calls and tool listings leave out per-run identifiers, so identical inputs
  reuse results across runs in one flow (prefect.md "Caching and Idempotency").

Either way, a key hashes values: equal values give equal keys, whether or not their objects are shared.

Not VCR tests: no model request goes out, and no Prefect server runs. `task` and `FlowRunContext.get` are replaced by
a stand-in flow attempt, which keeps the counters a flow attempt keeps and computes each task run's key with
`PrefectAgentInputs`. That is the part of the default cache policy that varies within a flow: the task's source and
the flow run's ID, which it adds, are the same for every task run here. The projections and counters that build a
key are internal, and no public API exposes a key without a Prefect server, so the property imports them.
"""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Callable, Generator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Literal, get_args
from unittest.mock import MagicMock

import pytest
from hypothesis import HealthCheck, given, settings, strategies as st
from pydantic import BaseModel

from pydantic_ai import Agent, BinaryContent, RunContext
from pydantic_ai.durable_exec._base import _ModelRequestCacheIdentity  # pyright: ignore[reportPrivateUsage]
from pydantic_ai.durable_exec._capability_operation import CapabilityCacheIdentity, CapabilityOperationParams
from pydantic_ai.durable_exec._operation import CapabilityOperationId, ModelRequestParams
from pydantic_ai.durable_exec._workspace import (
    WORKSPACE_OPERATION_ID,
    WorkspaceCall,
    WorkspaceCallCacheIdentity,
    WorkspaceCallParams,
)
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, TextPart, ToolCallPart, UserPromptPart
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.toolsets import DynamicToolset, FunctionToolset
from pydantic_ai.usage import RunUsage

from ..conftest import try_import

with try_import() as imports_successful:
    from prefect.context import FlowRunContext

    from pydantic_ai.durable_exec.prefect import (
        PrefectDurability,
        TaskConfig,
        _durability as prefect_durability,  # pyright: ignore[reportPrivateUsage]
        _operation_backend as prefect_operation_backend,
    )
    from pydantic_ai.durable_exec.prefect._cache_policies import PrefectAgentInputs

pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='prefect not installed'),
    # As in `tests/test_history_properties.py`: Hypothesis notes this deprecation when another thread touches
    # `random` during a draw, which no strategy here does.
    pytest.mark.filterwarnings(
        'ignore:Do not use the `random` module inside strategies:hypothesis.errors.HypothesisDeprecationWarning'
    ),
]

PROPERTY_SETTINGS = settings(
    max_examples=50,
    derandomize=True,
    database=None,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
"""A small, derandomized batch, so a run is reproducible and fits the per-test budget, as in
`tests/test_history_properties.py`."""

AGENT_NAME = 'cache_identity'


# --- A stand-in flow attempt -------------------------------------------------------------------------------------


@dataclass
class _FlowAttempt:
    """What `PrefectDurability` reads from a flow run: the counters Prefect rebuilds on each attempt.

    It stands in for `prefect.task` too, recording the cache key of each task run, by task name, and running it.
    """

    task_run_dynamic_keys: dict[str, object] = field(default_factory=dict[str, object])
    flow_run: str = 'flow-run'
    keys: list[tuple[str, str]] = field(default_factory=list[tuple[str, str]])

    def task(self, fn: Callable[..., Awaitable[object]]) -> _TaskRun:
        return _TaskRun(self, fn)


@dataclass
class _TaskRun:
    attempt: _FlowAttempt
    fn: Callable[..., Awaitable[object]]

    def with_options(self, *, name: str, **options: object) -> Callable[..., Awaitable[object]]:
        async def run(*args: object) -> object:
            inputs = dict(inspect.signature(self.fn).bind(*args).arguments)
            key = PrefectAgentInputs().compute_key(MagicMock(), inputs, {})
            assert key is not None
            self.attempt.keys.append((name, key))
            return await self.fn(*args)

        return run


@contextmanager
def _flow_attempt() -> Generator[_FlowAttempt]:
    attempt = _FlowAttempt()
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(FlowRunContext, 'get', lambda: attempt)
        monkeypatch.setattr(prefect_operation_backend, 'task', attempt.task)
        monkeypatch.setattr(prefect_durability, 'task', attempt.task)
        yield attempt


def _run_context(run_id: str, *, deps: object = None, metadata: dict[str, object] | None = None) -> RunContext[object]:
    return RunContext[object](deps=deps, model=TestModel(), usage=RunUsage(), run_id=run_id, metadata=metadata)


async def _done() -> object:
    return None


# --- Run-scoped keys: a run's k-th call keeps its key however runs interleave ----------------------------------------

RunScopedCall = Callable[[PrefectDurability[object], str, str], Awaitable[object]]
"""One call a run makes in flow code: `(durability, run_id, payload)`."""


async def _workspace_call(durability: PrefectDurability[object], run_id: str, payload: str) -> object:
    params = WorkspaceCallParams(
        run_context=_run_context(run_id), ref=None, call=WorkspaceCall(method='read_bytes', path=payload)
    )
    backend = durability.get_durable_operation_backend()
    assert isinstance(backend, prefect_operation_backend.PrefectOperationBackend)
    return await backend.execute(
        operation_id=WORKSPACE_OPERATION_ID,
        name=f'{AGENT_NAME}__workspace',
        body=_done,
        cache_key=WorkspaceCallCacheIdentity().project(params),
        config=TaskConfig(),
    )


async def _capability_operation(durability: PrefectDurability[object], run_id: str, payload: str) -> object:
    params = CapabilityOperationParams(run_context=_run_context(run_id), arguments={'note': payload})
    backend = durability.get_durable_operation_backend()
    assert isinstance(backend, prefect_operation_backend.PrefectOperationBackend)
    return await backend.execute(
        operation_id=CapabilityOperationId('notes', operation='save'),
        name=f'{AGENT_NAME}__notes__save',
        body=_done,
        cache_key=CapabilityCacheIdentity().project(params),
        config=TaskConfig(),
    )


async def _run_start_time(durability: PrefectDurability[object], run_id: str, payload: str) -> object:
    return await durability.get_run_start_time(run_id)


async def _default_run_id(durability: PrefectDurability[object], run_id: str, payload: str) -> object:
    """A run with no explicit ID: `run_id` stands for the run's prompt, and `payload` for its dependencies."""
    return await durability._get_default_run_id_for_inputs(  # pyright: ignore[reportPrivateUsage]
        lambda: {
            'prompt': run_id,
            'messages': [],
            'message_history_ids': [],
            'conversation_id': None,
            'deps': payload,
            'metadata': None,
        }
    )


RUN_SCOPED_CALLS: dict[str, RunScopedCall] = {
    'workspace_call': _workspace_call,
    'capability_operation': _capability_operation,
    'run_start_time': _run_start_time,
    'default_run_id': _default_run_id,
}

Runs = dict[str, list[str]]
"""Each run's ID, with the payloads of the calls it makes, in order."""


@st.composite
def _interleaved_runs(draw: st.DrawFn) -> tuple[Runs, list[str]]:
    """Runs with distinct IDs, and an order of all their calls in one flow attempt: the run of each call in turn.

    IDs draw from a small alphabet with the separator, so a key that joined a run ID to another field without its
    length could collide. Payloads repeat, so a run makes identical calls.
    """
    run_ids = draw(st.lists(st.text(alphabet='a:1', min_size=1, max_size=3), min_size=2, max_size=3, unique=True))
    runs = {run_id: draw(st.lists(st.sampled_from(['x', 'y']), min_size=1, max_size=3)) for run_id in run_ids}
    order = draw(st.permutations([run_id for run_id, payloads in runs.items() for _ in payloads]))
    return runs, order


async def _keys_by_call(call: RunScopedCall, runs: Runs, order: Sequence[str]) -> dict[tuple[str, int], str]:
    """The cache key of each run's k-th call, made in `order` in a fresh flow attempt."""
    durability = PrefectDurability[object](name=AGENT_NAME)
    keys: dict[tuple[str, int], str] = {}
    made = dict.fromkeys(runs, 0)
    with _flow_attempt() as attempt:
        for run_id in order:
            k = made[run_id]
            made[run_id] += 1
            await call(durability, run_id, runs[run_id][k])
            [(_, keys[run_id, k])] = attempt.keys[len(keys) :]
    return keys


@pytest.mark.parametrize('projection', RUN_SCOPED_CALLS)
@PROPERTY_SETTINGS
@given(interleaved=_interleaved_runs())
def test_run_scoped_key_ignores_other_runs(projection: str, interleaved: tuple[Runs, list[str]]) -> None:
    """A run's k-th call gets the key it gets when the runs go one after another, and no two calls share a key."""
    runs, order = interleaved
    one_after_another = [run_id for run_id, payloads in runs.items() for _ in payloads]
    call = RUN_SCOPED_CALLS[projection]
    with asyncio.Runner() as runner:
        expected = runner.run(_keys_by_call(call, runs, one_after_another))
        keys = runner.run(_keys_by_call(call, runs, order))
    assert keys == expected
    assert len(set(keys.values())) == len(keys)


# --- Run-agnostic keys: runs that differ only in their identity share results ---------------------------------------

_RUN_IDENTITIES = st.lists(
    st.tuples(st.text(alphabet='a:1', min_size=1, max_size=3), st.text(alphabet='b:2', min_size=1, max_size=3)),
    min_size=2,
    max_size=2,
    unique=True,
)
"""Two runs' `(run_id, conversation_id)`."""


@PROPERTY_SETTINGS
@given(prompt=st.text(max_size=8), value=st.text(max_size=8), identities=_RUN_IDENTITIES)
def test_model_and_tool_keys_leave_out_the_run(prompt: str, value: str, identities: list[tuple[str, str]]) -> None:
    """Two runs of one flow that differ only in their IDs and timestamps give their model requests, tool listing and
    tool call the same keys (prefect.md "Caching and Idempotency"), so the second replays the first's results.

    The model gives its tool call a fixed ID, as a provider's ID would be on replay of the model request's result.
    """

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(parts=[ToolCallPart('echo', {'value': value}, tool_call_id='call-1')])
        return ModelResponse(parts=[TextPart('done')])

    def echo(value: str) -> str:
        return value

    def tools(ctx: RunContext[None]) -> FunctionToolset[None]:
        return FunctionToolset([echo], id='echo')

    agent = Agent(
        FunctionModel(respond),
        name=AGENT_NAME,
        deps_type=type(None),
        toolsets=[DynamicToolset(tools, id='echo_tools')],
        capabilities=[PrefectDurability()],
    )

    async def run_each() -> list[list[tuple[str, str]]]:
        with _flow_attempt() as attempt:
            keys: list[list[tuple[str, str]]] = []
            for run_id, conversation_id in identities:
                before = len(attempt.keys)
                await agent.run(prompt, run_id=run_id, conversation_id=conversation_id)
                keys.append(attempt.keys[before:])
            return keys

    with asyncio.Runner() as runner:
        first, second = runner.run(run_each())
    assert {name.split(':')[0] for name, _ in first} == {'Discover Tools', 'Model Request', 'Call Tool'}
    assert second == first


# --- Shared and copied values give the same key ---------------------------------------------------------------------

Holder = Literal[
    'message',
    'deps_dict',
    'deps_list',
    'deps_dataclass',
    'deps_model',
    'metadata_dict',
    'metadata_model',
]
"""Where a value appears twice in a model request's inputs: in the prompt, or in the run's dependencies or metadata,
in a container of each kind they hold."""

HOLDERS: tuple[Holder, ...] = get_args(Holder)

_MODEL_HOLDERS: frozenset[Holder] = frozenset({'deps_model', 'metadata_model'})

Damage = Literal['binary_in_hashable_model']
"""Non-UTF-8 bytes in a Pydantic model Prefect can hash, which it hashes with `cloudpickle`."""


@dataclass
class _Pair:
    first: object
    second: object


class _PairModel(BaseModel):
    first: str | bytes
    second: str | bytes


@dataclass(frozen=True)
class SharedValue:
    holder: Holder
    value: str | bytes


@st.composite
def shared_values(draw: st.DrawFn, *, without: frozenset[Damage] = frozenset()) -> SharedValue:
    holder = draw(st.sampled_from(HOLDERS))
    binary = st.binary(max_size=3)
    if holder in _MODEL_HOLDERS and 'binary_in_hashable_model' in without:
        binary = st.text(max_size=3).map(str.encode)
    return SharedValue(holder, draw(st.one_of(st.text(max_size=3), binary)))


def _copy(value: str | bytes) -> str | bytes:
    """An equal value in an object of its own."""
    return bytes(bytearray(value)) if isinstance(value, bytes) else ''.join(list(value))


def _model_request_key(shared: SharedValue, *, copied: bool) -> str:
    """The key of a model request whose inputs hold `shared.value` twice: as one object, or as an object and a copy."""
    first, second = shared.value, _copy(shared.value) if copied else shared.value
    prompt: list[str | BinaryContent] = ['Describe these.']
    deps: object = None
    metadata: dict[str, object] | None = None
    match shared.holder:
        case 'message':
            prompt += [
                item if isinstance(item, str) else BinaryContent(item, media_type='application/octet-stream')
                for item in (first, second)
            ]
        case 'deps_dict':
            deps = {'first': first, 'second': second}
        case 'deps_list':
            deps = [first, second]
        case 'deps_dataclass':
            deps = _Pair(first, second)
        case 'deps_model':
            deps = _PairModel(first=first, second=second)
        case 'metadata_dict':
            metadata = {'first': first, 'second': second}
        case 'metadata_model':
            metadata = {'pair': _PairModel(first=first, second=second)}
    params = ModelRequestParams(
        messages=[ModelRequest(parts=[UserPromptPart(prompt)])],
        model_settings=None,
        model_request_parameters=ModelRequestParameters(),
        run_context=_run_context('run', deps=deps, metadata=metadata),
    )
    inputs = {'operation_name': 'Model Request', 'logical_inputs': _ModelRequestCacheIdentity().project(params)}
    key = PrefectAgentInputs().compute_key(MagicMock(), inputs, {})
    assert key is not None
    return key


@dataclass(frozen=True)
class KnownFinding:
    id: str
    damage: Damage
    """What draws the finding, left out of `test_equal_values_give_equal_keys`."""
    tracked_by: str
    reproducer: SharedValue


KNOWN_FINDINGS = [
    KnownFinding(
        'prefect-cache-key-sharing',
        'binary_in_hashable_model',
        'https://github.com/pydantic/pydantic-ai/issues/ISSUE-PENDING-prefect-cache-key-sharing',
        SharedValue('deps_model', b'\xff\xfe'),
    ),
]


@PROPERTY_SETTINGS
@given(shared=shared_values(without=frozenset(finding.damage for finding in KNOWN_FINDINGS)))
def test_equal_values_give_equal_keys(shared: SharedValue) -> None:
    """A value held twice as one object, and as an object and its copy, give a model request the same key, so a flow
    retry that rebuilds its inputs from replayed results, with different sharing, replays the request (#12188)."""
    assert _model_request_key(shared, copied=True) == _model_request_key(shared, copied=False)


@pytest.mark.parametrize(
    'finding',
    [
        pytest.param(
            finding,
            id=finding.id,
            marks=pytest.mark.xfail(
                strict=True,
                raises=AssertionError,
                reason=f'{finding.id} ({finding.tracked_by}): a Pydantic model Prefect can hash keeps the '
                '`cloudpickle` fallback, whose digest depends on object sharing',
            ),
        )
        for finding in KNOWN_FINDINGS
    ],
)
def test_known_finding_is_still_present(finding: KnownFinding) -> None:
    assert _model_request_key(finding.reproducer, copied=True) == _model_request_key(finding.reproducer, copied=False)
