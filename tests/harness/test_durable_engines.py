"""Every capability case under every durable engine, in-process.

Three checks run over the buildable cases of `tests/capability_cases.py` and `tests/harness/capability_cases.py`:

- A run lists the toolsets the agent was built with. A durable engine registers each toolset when it binds and
  dispatches a run's tool calls to the registered one by `id`, so a toolset a capability builds again for each run
  is refused inside a workflow, or listed in one place and called in another. This check needs no engine, and
  leaves out the cases that refuse every engine.
- Every engine in `ENGINES` builds the agent, unless `REFUSALS` says the case refuses that engine by design. Which
  engines a refusal covers follows from each engine's `engine_spec`.
- A case's `calls` run against the run context each engine rebuilds inside its durable unit: through
  `TemporalDurability`'s own serialization and a `dataclasses.replace` copy of what it rebuilds, end to end on the
  Azure Durable Task in-memory worker, and end to end in Render child tasks. A tool that reads a field the unit's
  context doesn't carry, or emits an event the unit can't deliver, fails its cell. A tool a capability runs in the
  workflow for that engine is called as the engine calls it.

A cell is a case and an engine, or `NO_ENGINE` for the first check. `DURABLE` lists each cell that fails on main,
with a reason linking the issue that tracks it. A listed cell that passes fails too, so fixing a capability fails
the suite until its entry is removed.
"""

from __future__ import annotations

import importlib.util
import inspect
import re
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable, Iterable, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import anyio
import pytest
from _pytest.mark import ParameterSet

from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.durable_exec import JSON_CODEC, BaseDurabilityCapability, ToolsetKind
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models import ModelRequestContext
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.toolsets import AbstractToolset, WrapperToolset
from pydantic_ai.toolsets.abstract import ToolsetTool

from .. import capability_cases as core
from ..append_only import ToolCall, build_capabilities, discover_capabilities, skip_unless_installed
from ..capability_cases import (
    _ISSUES,  # pyright: ignore[reportPrivateUsage]
    _NO_DURABLE_TOOLSET_ID,  # pyright: ignore[reportPrivateUsage]
    _TOOLSET_WITHOUT_ID,  # pyright: ignore[reportPrivateUsage]
    _USERS_TOOLSET,  # pyright: ignore[reportPrivateUsage]
    _WRAPPED_TOOLSET,  # pyright: ignore[reportPrivateUsage]
    CORE_CAPABILITY_PACKAGES,
    CapabilityCase,
    Unbuildable,
    shipped_capability_types,
)
from ..manifests import Exempt
from .capability_cases import CASES as HARNESS_CASES, HARNESS_CAPABILITY_PACKAGES


@dataclass(frozen=True)
class Engine:
    """A durable engine, built as a worker builds it, and the module its optional dependency group provides."""

    requires: str
    build: Callable[[], BaseDurabilityCapability[None]]


# Each builder imports its engine's SDK, so collecting this module imports none of them.
def _temporal() -> BaseDurabilityCapability[None]:
    from pydantic_ai.durable_exec.temporal import TemporalDurability

    return TemporalDurability[None]()


def _dbos() -> BaseDurabilityCapability[None]:
    from pydantic_ai.durable_exec.dbos import DBOSDurability

    return DBOSDurability[None]()


def _prefect() -> BaseDurabilityCapability[None]:
    from pydantic_ai.durable_exec.prefect import PrefectDurability

    return PrefectDurability[None]()


def _absurd() -> BaseDurabilityCapability[None]:
    from pydantic_ai_harness.absurd import AbsurdDurability

    return AbsurdDurability[None]()


def _aws_lambda() -> BaseDurabilityCapability[None]:
    from pydantic_ai_harness.aws_lambda import AWSLambdaDurability

    return AWSLambdaDurability[None]()


def _azure_durable_task() -> BaseDurabilityCapability[None]:
    from pydantic_ai_harness.azure_durable_task import AzureDurableTaskDurability

    return AzureDurableTaskDurability[None]()


def _pgtask() -> BaseDurabilityCapability[None]:
    from pydantic_ai_harness.pgtask import PGTaskDurability

    return PGTaskDurability[None]()


def _render() -> BaseDurabilityCapability[None]:
    from render import Workflows

    from pydantic_ai_harness.render import RenderWorkflows

    # Binding registers the agent's operation tasks on the app by name, so each agent gets an app of its own.
    return RenderWorkflows[None](Workflows())


ENGINES: dict[str, Engine] = {
    'TemporalDurability': Engine('temporalio', _temporal),
    'DBOSDurability': Engine('dbos', _dbos),
    'PrefectDurability': Engine('prefect', _prefect),
    'AbsurdDurability': Engine('absurd_sdk', _absurd),
    'AWSLambdaDurability': Engine('aws_durable_execution_sdk_python', _aws_lambda),
    'AzureDurableTaskDurability': Engine('durabletask', _azure_durable_task),
    'PGTaskDurability': Engine('pgtask', _pgtask),
    'RenderWorkflows': Engine('render', _render),
}
"""Every durable engine, keyed by its durability capability's class name."""

NO_ENGINE = 'no engine'
"""The engine of a cell for the check that a run lists the toolsets the agent was built with, which needs none."""

CASES: dict[str, CapabilityCase] = {**core.CASES, **HARNESS_CASES}
PUBLIC = discover_capabilities((*CORE_CAPABILITY_PACKAGES, *HARNESS_CAPABILITY_PACKAGES))
"""The public capabilities, which the suites build."""
BUILT = sorted(
    name
    for name, case in CASES.items()
    if name in PUBLIC and not isinstance(case.build, Unbuildable) and name not in ENGINES
)
"""The cases built here: every buildable public capability that isn't itself an engine."""


@dataclass(frozen=True)
class Refusal:
    """A case that an engine refuses to build an agent with, by design."""

    reason: str
    error: str
    """A pattern the engine's `UserError` matches."""
    kind: ToolsetKind | None = None
    """The kind of the toolset that has no `id`, or `None` when the case refuses every engine for another reason.

    An engine that doesn't wrap function or MCP toolsets builds the agent with such a toolset; every engine refuses
    a dynamic one."""


_LEAF_WITHOUT_ID = 'need to have a unique `id`'
_OPERATIONS_WITHOUT_ID = Refusal(
    "it contributes durable operations, whose persisted names need the capability's `id`, and its case sets none",
    'contributes durable operations and needs an explicit `id`',
)

REFUSALS: dict[str, Refusal] = {
    'Capability': Refusal(_TOOLSET_WITHOUT_ID, _LEAF_WITHOUT_ID, 'function'),
    'PydanticAIDocs': Refusal(_TOOLSET_WITHOUT_ID, _LEAF_WITHOUT_ID, 'function'),
    'BrowserUse': Refusal(_TOOLSET_WITHOUT_ID, _LEAF_WITHOUT_ID, 'function'),
    'Toolset': Refusal(_USERS_TOOLSET, _LEAF_WITHOUT_ID, 'function'),
    'MCP': Refusal(_USERS_TOOLSET, _LEAF_WITHOUT_ID, 'mcp'),
    'PrefixTools': Refusal(_WRAPPED_TOOLSET, _LEAF_WITHOUT_ID, 'function'),
    'DynamicCapability': Refusal(
        'a capability function contributes a dynamic toolset, which has an id only when the `DynamicCapability` '
        'wrapping it is given one',
        _LEAF_WITHOUT_ID,
        'dynamic',
    ),
    'ApprovalReviewer': _OPERATIONS_WITHOUT_ID,
    'InputGuardrail': _OPERATIONS_WITHOUT_ID,
    'OutputGuardrail': _OPERATIONS_WITHOUT_ID,
    'PromptInjectionDefender': _OPERATIONS_WITHOUT_ID,
    'ToolCallJudge': _OPERATIONS_WITHOUT_ID,
    'ToolGuardrail': _OPERATIONS_WITHOUT_ID,
    'Peers': Refusal(_NO_DURABLE_TOOLSET_ID, 'Peers does not support durable execution'),
    'PlaywrightBrowser': Refusal(
        'it refuses durable agents: a live browser session cannot be replayed',
        'PlaywrightBrowser does not support durable execution',
    ),
}
"""The cases an engine refuses to build an agent with, by design, keyed by class name."""

REGISTERED = [name for name in BUILT if (refusal := REFUSALS.get(name)) is None or refusal.kind is not None]
"""The cases some engine builds an agent with, whose toolsets an engine registers: every case but those that refuse
every engine."""

_RUNTIME_BUILT_TOOLSET = (
    'it builds its toolset again for each run under the same `id`, which a durable engine refuses inside a '
    f'workflow: {_ISSUES}ISSUE-PENDING-runtime-built-toolsets'
)
_AZURE_TRACER = (
    "its tools open spans on `ctx.tracer`, which an Azure Durable Task activity's run context does not carry: "
    f'{_ISSUES}ISSUE-PENDING-azure-tracer'
)
_EMITS_IN_UNIT = (
    "its tools emit events from inside the durable unit, whose run context can't reach the run's event stream: "
    f'{_ISSUES}7971'
)

DURABLE: dict[tuple[str, str], Exempt] = {
    ('CompactOnRequest', NO_ENGINE): Exempt(_RUNTIME_BUILT_TOOLSET),
    ('Researcher', NO_ENGINE): Exempt(f'its `web_search` and `web_fetch` toolsets: {_RUNTIME_BUILT_TOOLSET}'),
    ('Memory', 'AzureDurableTaskDurability'): Exempt(_AZURE_TRACER),
    ('SmartSearch', 'AzureDurableTaskDurability'): Exempt(_AZURE_TRACER),
    ('Planning', 'AzureDurableTaskDurability'): Exempt(_EMITS_IN_UNIT),
    ('Planning', 'TemporalDurability'): Exempt(_EMITS_IN_UNIT),
}
"""Each cell that fails on main, keyed by case and engine name."""


def _assert_holds(cell: tuple[str, str], breaks: Sequence[str]) -> None:
    """`breaks` says how `cell` fails: it must say nothing, unless `DURABLE` lists the cell."""
    if exempt := DURABLE.get(cell):
        assert breaks, f'{cell} now passes: remove its `DURABLE` entry ({exempt.reason})'
    else:
        assert not breaks, '; '.join(breaks)


async def _build(name: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> list[AbstractCapability[None]]:
    """Everything the case of `name` builds, after skipping it unless its dependency is installed."""
    case = CASES[name]
    skip_unless_installed(name, case.requires)
    for key, value in case.env.items():
        monkeypatch.setenv(key, value)
    # Built off the event loop, as an app builds its agent: building can import a client library.
    return await anyio.to_thread.run_sync(build_capabilities, name, PUBLIC[name].load, case.fixture, tmp_path)


def _agent_name() -> str:
    """A name no other agent in the session has: DBOS registers each agent's steps by name, process-wide."""
    return f'cell_{uuid.uuid4().hex}'


def _answering_model() -> FunctionModel:
    async def stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
        yield 'done'

    return FunctionModel(lambda messages, info: ModelResponse(parts=[TextPart('done')]), stream_function=stream)


def _calling_model(made: tuple[ToolCall, ...]) -> FunctionModel:
    """A model that makes the calls in order, one per response, then answers; call `i` gets the ID `call_{i}`."""

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        step = sum(isinstance(message, ModelResponse) for message in messages)
        if step == len(made):
            return ModelResponse(parts=[TextPart('done')])
        name, args = made[step]
        return ModelResponse(parts=[ToolCallPart(name, args, tool_call_id=f'call_{step}')])

    return FunctionModel(respond)


def _leaves(toolsets: Iterable[AbstractToolset[None]]) -> list[AbstractToolset[None]]:
    leaves: list[AbstractToolset[None]] = []
    for toolset in toolsets:
        toolset.apply(leaves.append)
    return leaves


@dataclass
class _ListedToolsets(AbstractCapability[None]):
    """Records the toolset each request lists its tools from."""

    toolsets: list[AbstractToolset[None]] = field(default_factory=list[AbstractToolset[None]])

    async def before_model_request(
        self, ctx: RunContext[None], request_context: ModelRequestContext
    ) -> ModelRequestContext:
        assert ctx.tool_manager is not None
        self.toolsets.append(ctx.tool_manager.toolset)
        return request_context


def test_every_engine_has_an_entry() -> None:
    found, skipped = shipped_capability_types(('pydantic_ai.durable_exec', *HARNESS_CAPABILITY_PACKAGES))
    engines = {
        name for name, cls in found.items() if issubclass(cls, BaseDurabilityCapability) and not inspect.isabstract(cls)
    }
    assert sorted(engines - ENGINES.keys()) == [], 'add an `ENGINES` entry for each new durable engine'
    if not skipped:
        assert sorted(ENGINES.keys() - engines) == [], '`ENGINES` names an engine that no longer exists'


def test_every_table_entry_names_a_cell() -> None:
    """An entry for a case or engine that no longer exists would hide the next one added under its name."""
    cells = {(name, NO_ENGINE) for name in REGISTERED} | {(name, engine) for name in BUILT for engine in ENGINES}
    assert sorted(DURABLE.keys() - cells) == []
    assert sorted(REFUSALS.keys() - set(BUILT)) == []
    assert [cell for cell, exempt in DURABLE.items() if not exempt.reason.strip()] == []


@pytest.mark.parametrize('name', REGISTERED)
async def test_a_run_lists_the_toolsets_the_agent_was_built_with(
    name: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    listed = _ListedToolsets()
    agent = Agent(
        _answering_model(), deps_type=type(None), capabilities=[*await _build(name, tmp_path, monkeypatch), listed]
    )
    await agent.run('go')
    assert listed.toolsets, 'the run made no request'
    built = {id(leaf) for leaf in _leaves(agent.toolsets)}
    breaks = [
        f'the run lists {leaf.label}, which the agent was not built with'
        for leaf in _leaves(listed.toolsets[:1])
        # An id in angle brackets names a role the framework fills per run, such as `<output>`.
        if leaf.id is not None and not leaf.id.startswith('<') and id(leaf) not in built
    ]
    _assert_holds((name, NO_ENGINE), breaks)


@dataclass
class _InActivity(WrapperToolset[None]):
    """Calls each tool with the run context its Temporal activity rebuilds from the serialized one."""

    copy: bool = False
    """Call it with a `dataclasses.replace` copy of that context instead, as code in the activity can make."""

    # `Any`, as in the `WrapperToolset.call_tool` this overrides.
    async def call_tool(
        self, name: str, tool_args: dict[str, Any], ctx: RunContext[None], tool: ToolsetTool[None]
    ) -> Any:
        from pydantic_ai.durable_exec.temporal import TemporalDurability

        key = TemporalDurability.engine_spec.tool_config_key
        config = (tool.tool_def.metadata or {}).get(key) if key is not None else None
        if config is False or (config is None and tool.tool_def.durable is False):
            # The engine runs this tool in the workflow, on the run's own context.
            return await super().call_tool(name, tool_args, ctx, tool)
        assert ctx.agent is not None
        durability = TemporalDurability[None].from_agent(ctx.agent)
        assert durability is not None
        serialized = durability.run_context_type.serialize_run_context(ctx)
        wire = JSON_CODEC.load(dict[str, object], JSON_CODEC.dump(dict[str, object], serialized))
        restored = durability.deserialize_operation_run_context(wire, ctx.deps)
        return await super().call_tool(name, tool_args, replace(restored) if self.copy else restored, tool)


@dataclass
class _RunToolsInActivities(AbstractCapability[None]):
    """Wraps the run's toolset in `_InActivity`."""

    copy: bool

    def get_wrapper_toolset(self, toolset: AbstractToolset[None]) -> AbstractToolset[None]:
        return _InActivity(toolset, copy=self.copy)


async def _build_calls(name: str, tmp_path: Path) -> tuple[list[AbstractCapability[None]], tuple[ToolCall, ...]]:
    """A fresh build of what the case's calls run against, in a directory of its own, as tools keep state there."""
    calls = CASES[name].calls
    assert calls is not None
    directory = tmp_path / uuid.uuid4().hex
    directory.mkdir(parents=True)
    built = await anyio.to_thread.run_sync(build_capabilities, name, PUBLIC[name].load, calls.build, directory)
    return built, calls.made


async def _temporal_breaks(name: str, tmp_path: Path) -> list[str]:
    from pydantic_ai.durable_exec.temporal import TemporalDurability

    breaks: list[str] = []
    for copy in (False, True):
        capabilities, made = await _build_calls(name, tmp_path)
        agent = Agent(
            _calling_model(made),
            name=_agent_name(),
            deps_type=type(None),
            capabilities=[*capabilities, _RunToolsInActivities(copy=copy), TemporalDurability[None]()],
        )
        try:
            await agent.run('go')
        except UserError as exc:
            breaks.append(f'{"a copy of " if copy else ""}the rebuilt run context: {exc}')
    return breaks


async def _azure_breaks(name: str, tmp_path: Path) -> list[str]:
    from durabletask.client import OrchestrationFailedError

    from pydantic_ai_harness.azure_durable_task import AzureDurableTaskDurability

    from .azure_durable_task.worker import InMemoryWorker, registered_agent, run_agent

    capabilities, made = await _build_calls(name, tmp_path)
    agent = Agent(
        _calling_model(made),
        name=_agent_name(),
        deps_type=type(None),
        capabilities=[*capabilities, AzureDurableTaskDurability[None]()],
    )
    worker = InMemoryWorker()

    def run() -> None:
        with registered_agent(worker, agent):
            run_agent(worker, agent, 'go', deps=None)

    try:
        # The orchestration runs on a loop of its own, which refuses to start under a running one.
        await anyio.to_thread.run_sync(run)
    except OrchestrationFailedError as exc:
        return [str(exc)]
    return []


async def _render_breaks(name: str, tmp_path: Path) -> list[str]:
    from render import Workflows

    from pydantic_ai_harness.render import RenderWorkflows

    from .render.conftest import RecordingTaskContext, run_agent_in_task

    capabilities, made = await _build_calls(name, tmp_path)
    runtime = RenderWorkflows[None](Workflows())
    agent = Agent(_calling_model(made), name=_agent_name(), deps_type=type(None), capabilities=[*capabilities, runtime])
    try:
        await run_agent_in_task(agent, runtime, RecordingTaskContext())
    except UserError as exc:
        return [str(exc)]
    return []


_RUNNERS: dict[str, Callable[[str, Path], Awaitable[list[str]]]] = {
    'TemporalDurability': _temporal_breaks,
    'AzureDurableTaskDurability': _azure_breaks,
    'RenderWorkflows': _render_breaks,
}
"""How a case's calls run on each engine whose durable units rebuild the run context from a serialized one.

The other engines run each unit in the run's own process, on the run's own context."""

_CELLS: list[ParameterSet] = [
    pytest.param(name, engine, marks=pytest.mark.moves_cache_prefix(reason=reason))
    if engine in _RUNNERS and (calls := CASES[name].calls) is not None and (reason := calls.moves_cache_prefix)
    else pytest.param(name, engine)
    for name in BUILT
    for engine in sorted(ENGINES)
]
"""Every case under every engine; a cell whose calls rewrite history is marked `moves_cache_prefix`."""


@pytest.mark.parametrize(('name', 'engine'), _CELLS)
async def test_the_engine_runs_the_case(
    name: str, engine: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    if importlib.util.find_spec(ENGINES[engine].requires) is None:  # pragma: lax no cover
        pytest.skip(f'{engine} needs `{ENGINES[engine].requires}`, from an optional group that is not installed')
    capabilities = await _build(name, tmp_path, monkeypatch)
    durability = ENGINES[engine].build()
    refusal = REFUSALS.get(name)
    # An id-less dynamic toolset is refused before the engine looks at the kinds of toolset it wraps.
    if refusal is not None and refusal.kind in ('function', 'mcp'):
        refusal = refusal if refusal.kind in durability.engine_spec.wrapped_toolset_kinds else None
    breaks: list[str] = []
    try:
        Agent(_answering_model(), name=_agent_name(), deps_type=type(None), capabilities=[*capabilities, durability])
    except UserError as exc:
        if refusal is None or not re.search(refusal.error, str(exc)):
            breaks.append(f'building the agent raised: {exc}')
    else:
        if refusal is not None:
            breaks.append(f'the agent was built, but `REFUSALS` says it is refused: {refusal.reason}')
    if refusal is None and CASES[name].calls is not None and (runner := _RUNNERS.get(engine)) is not None:
        breaks += await runner(name, tmp_path / 'calls')
    _assert_holds((name, engine), breaks)
