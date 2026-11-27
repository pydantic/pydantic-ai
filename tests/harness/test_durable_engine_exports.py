"""`pydantic_ai_harness.temporal`, `.dbos` and `.prefect` re-export the core durable engines.

Until Pydantic AI v3 the code lives in `pydantic_ai.durable_exec.<engine>`, so the harness modules must hand out
the very same objects, minus the deprecated names, and recorded workflow, activity, step and task names must not
depend on which path an agent was built from. Each engine skips when its SDK isn't installed; all three run in the
all-extras cells.
"""

from __future__ import annotations

import importlib
from types import ModuleType

import pytest

import pydantic_ai_harness
from pydantic_ai import Agent
from pydantic_ai.durable_exec import (
    DurableOperationId,
    EventStreamHandlerId,
    ModelRequestId,
    ToolsetCallToolId,
    ToolsetGetInstructionsId,
    ToolsetGetToolsId,
    ToolsetValidateToolArgumentsId,
)
from pydantic_ai.models.test import TestModel
from pydantic_ai.toolsets import FunctionToolset

_SDKS = {'temporal': 'temporalio', 'dbos': 'dbos', 'prefect': 'prefect'}
_ENGINES = list(_SDKS)

# The `@deprecated` names in each core `__all__`. They keep their core import path until v3 and get no new one.
_DEPRECATED = {
    'temporal': {'TemporalAgent'},
    'dbos': {'DBOSAgent'},
    'prefect': {'PrefectAgent', 'PrefectFunctionToolset', 'PrefectMCPToolset'},
}
_ROOT_EXPORTS = {'temporal': 'TemporalDurability', 'dbos': 'DBOSDurability', 'prefect': 'PrefectDurability'}

_OPERATION_IDS: list[DurableOperationId] = [
    ModelRequestId(None, streaming=False, model_name='test'),
    ModelRequestId('other', streaming=True, model_name='test'),
    EventStreamHandlerId(),
    ToolsetGetToolsId('mcp', toolset_id='mcp'),
    ToolsetGetInstructionsId('mcp'),
    ToolsetCallToolId('mcp', toolset_id='mcp'),
    ToolsetValidateToolArgumentsId('function', toolset_id='functions'),
    ToolsetCallToolId('function', toolset_id='functions'),
]


def _core_and_harness(engine: str) -> tuple[ModuleType, ModuleType]:
    pytest.importorskip(_SDKS[engine])
    return (
        importlib.import_module(f'pydantic_ai.durable_exec.{engine}'),
        importlib.import_module(f'pydantic_ai_harness.{engine}'),
    )


@pytest.mark.parametrize('engine', _ENGINES)
def test_every_harness_name_is_the_core_object(engine: str) -> None:
    core, harness = _core_and_harness(engine)
    assert harness.__all__
    for name in harness.__all__:
        assert getattr(harness, name) is getattr(core, name), name


@pytest.mark.parametrize('engine', _ENGINES)
def test_harness_reexports_every_core_name_except_the_deprecated_ones(engine: str) -> None:
    core, harness = _core_and_harness(engine)
    deprecated = {name for name in core.__all__ if getattr(getattr(core, name), '__deprecated__', None)}
    assert deprecated == _DEPRECATED[engine]
    assert set(harness.__all__) == set(core.__all__) - deprecated
    for name in deprecated:
        assert not hasattr(harness, name), name


@pytest.mark.parametrize('engine', _ENGINES)
def test_root_export_is_the_core_capability(engine: str) -> None:
    core, harness = _core_and_harness(engine)
    name = _ROOT_EXPORTS[engine]
    assert name in pydantic_ai_harness.__all__
    assert getattr(pydantic_ai_harness, name) is getattr(core, name) is getattr(harness, name)


def test_temporal_activity_names_do_not_depend_on_the_import_path() -> None:
    pytest.importorskip('temporalio')
    from temporalio.activity import _Definition as ActivityDefinition  # pyright: ignore[reportPrivateUsage]

    from pydantic_ai.durable_exec.temporal import TemporalDurability as CoreTemporalDurability
    from pydantic_ai_harness.temporal import TemporalDurability

    names: list[set[str | None]] = []
    for durability_class in (TemporalDurability, CoreTemporalDurability):
        agent = Agent(
            TestModel(), name='exports', toolsets=[FunctionToolset(id='functions')], capabilities=[durability_class()]
        )
        durability = durability_class.from_agent(agent)
        assert durability is not None
        names.append(
            {
                ActivityDefinition.must_from_callable(activity).name  # pyright: ignore[reportUnknownMemberType]
                for activity in durability.temporal_activities
            }
        )
    harness_names, core_names = names
    assert {'agent__exports__model_request', 'agent__exports__toolset__functions__call_tool'} <= harness_names
    assert harness_names == core_names


def test_dbos_step_names_do_not_depend_on_the_import_path() -> None:
    pytest.importorskip('dbos')
    from pydantic_ai.durable_exec.dbos import DBOSOperationNamer as CoreDBOSOperationNamer
    from pydantic_ai_harness.dbos import DBOSOperationNamer

    harness_names = [DBOSOperationNamer('exports').operation_name(operation_id) for operation_id in _OPERATION_IDS]
    assert 'exports__model.request' in harness_names
    assert harness_names == [
        CoreDBOSOperationNamer('exports').operation_name(operation_id) for operation_id in _OPERATION_IDS
    ]


def test_prefect_task_names_do_not_depend_on_the_import_path() -> None:
    pytest.importorskip('prefect')
    from pydantic_ai.durable_exec.prefect import PrefectOperationNamer as CorePrefectOperationNamer
    from pydantic_ai_harness.prefect import PrefectOperationNamer

    harness_names = [
        PrefectOperationNamer().invocation_name(operation_id, label='get_weather') for operation_id in _OPERATION_IDS
    ]
    assert harness_names[0].operation_name == 'Model Request: test'
    assert harness_names == [
        CorePrefectOperationNamer().invocation_name(operation_id, label='get_weather')
        for operation_id in _OPERATION_IDS
    ]
