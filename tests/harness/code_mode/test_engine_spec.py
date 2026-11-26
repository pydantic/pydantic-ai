"""How `CodeMode` runs Monty in workflow code, read from the durable engine's `DurabilityEngineSpec`.

`thread_wakeable_loop=False` sends every Monty call through a blocking portal, and
`replays_orchestrator_code=True` turns off the elapsed-time limit. A fake engine flips each flag, and
each built-in engine's values are pinned, so every engine keeps the path it ran on before the flags.
"""

from __future__ import annotations

import importlib
import threading
from dataclasses import replace
from datetime import datetime
from typing import Any

import pytest
from pydantic_monty import OsFunction

from pydantic_ai import Agent
from pydantic_ai.durable_exec import BaseDurabilityCapability, DurabilityEngineSpec
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai_harness.code_mode import CodeMode, CodeModeOS, CodeModeResourceLimits

from .._recording_durability import RecordingDurability


class _UnwakeableLoopDurability(RecordingDurability):
    """An engine whose workflow code runs on a loop that can't be woken from another thread."""

    engine_spec = replace(RecordingDurability.engine_spec, thread_wakeable_loop=False)


class _ReplayingDurability(RecordingDurability):
    """An engine that replays its workflow code against the recorded history."""

    engine_spec = replace(RecordingDurability.engine_spec, replays_orchestrator_code=True)


def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


async def _run_code(
    code: str,
    durability: RecordingDurability,
    *,
    os_access: CodeModeOS | None = None,
    resource_limits: CodeModeResourceLimits | None = None,
) -> ToolReturnPart:
    """Run `code` once through `run_code` in `durability`'s workflow code, and return what the model got back."""

    def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(parts=[ToolCallPart('run_code', {'code': code})])
        return ModelResponse(parts=[TextPart('done')])

    agent = Agent(
        FunctionModel(model),
        name='engine_spec',
        deps_type=type(None),
        capabilities=[CodeMode[None](os_access=os_access, resource_limits=resource_limits), durability],
    )
    agent.tool_plain(add)
    result = await agent.run('go')
    request = result.all_messages()[2]
    assert isinstance(request, ModelRequest)
    [part] = request.parts
    assert isinstance(part, ToolReturnPart)
    return part


class TestCodeModeEngineSpec:
    @pytest.mark.parametrize(
        ('durability_type', 'through_portal'),
        [
            pytest.param(RecordingDurability, False, id='wakeable'),
            pytest.param(_UnwakeableLoopDurability, True, id='unwakeable'),
        ],
    )
    async def test_an_unwakeable_loop_calls_monty_through_the_portal(
        self, durability_type: type[RecordingDurability], through_portal: bool
    ) -> None:
        """Through the portal, a plain `os_access` handler runs on the run's own thread instead of Monty's."""
        threads: list[threading.Thread] = []

        def handler(*, name: OsFunction, args: tuple[Any, ...], kwargs: dict[str, Any], **_: Any) -> Any:
            threads.append(threading.current_thread())
            return datetime(2026, 9, 26)

        part = await _run_code('import datetime\ndatetime.datetime.now().year', durability_type(), os_access=handler)

        assert part.outcome != 'retried'
        assert len(threads) == 1
        assert (threads[0] is threading.current_thread()) is through_portal

    @pytest.mark.parametrize(
        ('durability_type', 'time_limited'),
        [
            pytest.param(RecordingDurability, True, id='not-replayed'),
            pytest.param(_ReplayingDurability, False, id='replayed'),
        ],
    )
    async def test_replayed_workflow_code_has_no_elapsed_time_limit(
        self, durability_type: type[RecordingDurability], time_limited: bool
    ) -> None:
        """A limit that fires on one execution but not its replay would make them diverge."""
        busy = 'total = 0\nfor item in range(100_000):\n    total += item\ntotal'

        part = await _run_code(busy, durability_type(), resource_limits={'max_duration_seconds': 1e-6})

        assert (part.outcome == 'retried') is time_limited


@pytest.mark.parametrize(
    ('module', 'name', 'replays_orchestrator_code', 'thread_wakeable_loop'),
    [
        pytest.param('pydantic_ai.durable_exec.temporal', 'TemporalDurability', True, False, id='temporal'),
        pytest.param(
            'pydantic_ai_harness.azure_durable_task', 'AzureDurableTaskDurability', True, False, id='azure-durable-task'
        ),
        pytest.param('pydantic_ai.durable_exec.dbos', 'DBOSDurability', False, True, id='dbos'),
        pytest.param('pydantic_ai.durable_exec.prefect', 'PrefectDurability', False, True, id='prefect'),
        pytest.param('pydantic_ai_harness.aws_lambda', 'AWSLambdaDurability', False, True, id='aws-lambda'),
        pytest.param('pydantic_ai_harness.absurd', 'AbsurdDurability', False, True, id='absurd'),
        pytest.param('pydantic_ai_harness.pgtask', 'PGTaskDurability', False, True, id='pgtask'),
        pytest.param('pydantic_ai_harness.render', 'RenderWorkflows', False, True, id='render'),
    ],
)
def test_built_in_engine_spec_flags(
    module: str, name: str, replays_orchestrator_code: bool, thread_wakeable_loop: bool
) -> None:
    """Each built-in engine declares the flags that keep `CodeMode` and `DynamicWorkflow` on their earlier path."""
    try:
        durability: type[BaseDurabilityCapability[Any]] = getattr(importlib.import_module(module), name)
    except ImportError:  # pragma: lax no cover - the engine's optional extra isn't installed
        pytest.skip(f'{module} needs its optional extra')
    spec: DurabilityEngineSpec = durability.engine_spec
    assert (spec.replays_orchestrator_code, spec.thread_wakeable_loop) == (
        replays_orchestrator_code,
        thread_wakeable_loop,
    )
