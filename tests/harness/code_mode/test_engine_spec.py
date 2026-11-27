"""How `CodeMode` runs Monty in workflow code, read from the durable engine's `DurabilityEngineSpec`.

`thread_wakeable_loop=False` sends every Monty call through a blocking portal, and
`replays_orchestrator_code=True` turns off the elapsed-time limit. A fake engine flips each flag, and
each built-in engine's values are pinned, so every engine keeps the path it ran on before the flags.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from datetime import datetime
from typing import Any

import pytest
from pydantic_monty import OsFunction

from pydantic_ai import Agent
from pydantic_ai.durable_exec import DurabilityEngineSpec
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai_harness.code_mode import CodeMode, CodeModeOS, CodeModeResourceLimits
from tests.conftest import try_import

from .._recording_durability import RecordingDurability, ReplayingDurability, UnwakeableLoopDurability

with try_import() as temporal_available:
    from pydantic_ai.durable_exec.temporal import TemporalDurability
with try_import() as azure_available:
    from pydantic_ai_harness.azure_durable_task import AzureDurableTaskDurability
with try_import() as dbos_available:
    from pydantic_ai.durable_exec.dbos import DBOSDurability
with try_import() as prefect_available:
    from pydantic_ai.durable_exec.prefect import PrefectDurability
with try_import() as aws_lambda_available:
    from pydantic_ai_harness.aws_lambda import AWSLambdaDurability
with try_import() as absurd_available:
    from pydantic_ai_harness.absurd import AbsurdDurability
with try_import() as pgtask_available:
    from pydantic_ai_harness.pgtask import PGTaskDurability
with try_import() as render_available:
    from pydantic_ai_harness.render import RenderWorkflows

_BUSY = 'total = 0\nfor item in range(100_000):\n    total += item\ntotal'
"""A snippet that runs far longer than a 1 µs elapsed-time limit."""


def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


async def _run_code(
    code: str,
    durability: RecordingDurability | None,
    *,
    os_access: CodeModeOS | None = None,
    resource_limits: CodeModeResourceLimits | None = None,
) -> ToolReturnPart:
    """Run `code` once through `run_code`, in `durability`'s workflow code if given, and return what the model got back."""

    def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(parts=[ToolCallPart('run_code', {'code': code})])
        return ModelResponse(parts=[TextPart('done')])

    code_mode = CodeMode[None](os_access=os_access, resource_limits=resource_limits)
    agent = Agent(
        FunctionModel(model),
        name='engine_spec',
        deps_type=type(None),
        capabilities=[code_mode] if durability is None else [code_mode, durability],
    )
    agent.tool_plain(add)
    result = await agent.run('go')
    request = result.all_messages()[2]
    assert isinstance(request, ModelRequest)
    [part] = request.parts
    assert isinstance(part, ToolReturnPart)
    return part


async def _handler_threads(durability: RecordingDurability | None) -> list[threading.Thread]:
    """The threads a plain `os_access` handler ran on: the run's own one through the portal, Monty's otherwise."""
    threads: list[threading.Thread] = []

    def handler(*, name: OsFunction, args: tuple[Any, ...], kwargs: dict[str, Any], **_: Any) -> Any:
        threads.append(threading.current_thread())
        return datetime(2026, 9, 26)

    part = await _run_code('import datetime\ndatetime.datetime.now().year', durability, os_access=handler)
    assert part.outcome != 'retried'
    return threads


class TestCodeModeEngineSpec:
    @pytest.mark.parametrize(
        ('durability_type', 'through_portal'),
        [
            pytest.param(RecordingDurability, False, id='wakeable'),
            pytest.param(UnwakeableLoopDurability, True, id='unwakeable'),
        ],
    )
    async def test_an_unwakeable_loop_calls_monty_through_the_portal(
        self, durability_type: type[RecordingDurability], through_portal: bool
    ) -> None:
        threads = await _handler_threads(durability_type())

        assert len(threads) == 1
        assert (threads[0] is threading.current_thread()) is through_portal

    @pytest.mark.parametrize(
        ('durability_type', 'time_limited'),
        [
            pytest.param(RecordingDurability, True, id='not-replayed'),
            pytest.param(ReplayingDurability, False, id='replayed'),
        ],
    )
    async def test_replayed_workflow_code_has_no_elapsed_time_limit(
        self, durability_type: type[RecordingDurability], time_limited: bool
    ) -> None:
        """A limit that fires on one execution but not its replay would make them diverge."""
        part = await _run_code(_BUSY, durability_type(), resource_limits={'max_duration_seconds': 1e-6})

        assert (part.outcome == 'retried') is time_limited

    @pytest.mark.skipif(not azure_available(), reason='durabletask not installed')
    async def test_an_agent_without_a_capability_gets_the_running_engines_spec(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An agent with no durability capability, like a sub-agent, still runs in the orchestration's code."""

        def in_durable_task_orchestration() -> bool:
            return True

        monkeypatch.setattr(
            'pydantic_ai_harness._monty_exec.in_durable_task_orchestration', in_durable_task_orchestration
        )

        assert await _handler_threads(None) == [threading.current_thread()]
        part = await _run_code(_BUSY, None, resource_limits={'max_duration_seconds': 1e-6})
        assert part.outcome != 'retried'


@pytest.mark.parametrize(
    ('engine_spec', 'replays_orchestrator_code', 'thread_wakeable_loop'),
    [
        pytest.param(
            lambda: TemporalDurability.engine_spec,
            True,
            False,
            id='temporal',
            marks=pytest.mark.skipif(not temporal_available(), reason='temporalio not installed'),
        ),
        pytest.param(
            lambda: AzureDurableTaskDurability.engine_spec,
            True,
            False,
            id='azure-durable-task',
            marks=pytest.mark.skipif(not azure_available(), reason='durabletask not installed'),
        ),
        pytest.param(
            lambda: DBOSDurability.engine_spec,
            False,
            True,
            id='dbos',
            marks=pytest.mark.skipif(not dbos_available(), reason='dbos not installed'),
        ),
        pytest.param(
            lambda: PrefectDurability.engine_spec,
            False,
            True,
            id='prefect',
            marks=pytest.mark.skipif(not prefect_available(), reason='prefect not installed'),
        ),
        pytest.param(
            lambda: AWSLambdaDurability.engine_spec,
            False,
            True,
            id='aws-lambda',
            marks=pytest.mark.skipif(not aws_lambda_available(), reason='aws-durable-execution-sdk not installed'),
        ),
        pytest.param(
            lambda: AbsurdDurability.engine_spec,
            False,
            True,
            id='absurd',
            marks=pytest.mark.skipif(not absurd_available(), reason='absurd not installed'),
        ),
        pytest.param(
            lambda: PGTaskDurability.engine_spec,
            False,
            True,
            id='pgtask',
            marks=pytest.mark.skipif(not pgtask_available(), reason='pgtask not installed'),
        ),
        pytest.param(
            lambda: RenderWorkflows.engine_spec,
            False,
            True,
            id='render',
            marks=pytest.mark.skipif(not render_available(), reason='render not installed'),
        ),
    ],
)
def test_built_in_engine_spec_flags(
    engine_spec: Callable[[], DurabilityEngineSpec], replays_orchestrator_code: bool, thread_wakeable_loop: bool
) -> None:
    """Each built-in engine declares the flags that keep `CodeMode` and `DynamicWorkflow` on their earlier path."""
    spec = engine_spec()
    assert (spec.replays_orchestrator_code, spec.thread_wakeable_loop) == (
        replays_orchestrator_code,
        thread_wakeable_loop,
    )
