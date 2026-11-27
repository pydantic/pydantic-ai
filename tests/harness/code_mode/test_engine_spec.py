"""How `CodeMode` runs Monty in workflow code, read from the durable engine's `DurabilityEngineSpec`.

`thread_wakeable_loop=False` sends every Monty call through a blocking portal, and
`replays_orchestrator_code=True` turns off the elapsed-time limit. A fake engine flips each flag; each
built-in engine's values are pinned in that engine's own test module.
"""

from __future__ import annotations

import threading

import pytest

from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai_harness.code_mode import CodeMode, CodeModeResourceLimits

from .._recording_durability import (
    RecordingDurability,
    ReplayingDurability,
    UnboundedReplayingDurability,
    UnwakeableLoopDurability,
)
from .stored_sessions import SERVER, stored_session_pool

_BUSY = 'total = 0\nfor item in range(100_000):\n    total += item\ntotal'
"""A snippet that runs far longer than a 1 µs elapsed-time limit."""


def _portal_threads() -> list[str]:
    """Names of the live `anyio` blocking-portal threads.

    `anyio` names them `asyncio-portal-<id>`, or `Thread-<n> (run_blocking_portal)` in older versions.
    """
    return [thread.name for thread in threading.enumerate() if 'portal' in thread.name]


async def _run_code(
    code: str,
    durability: RecordingDurability,
    *,
    resource_limits: CodeModeResourceLimits | None = None,
    portals: list[list[str]] | None = None,
) -> ToolReturnPart:
    """Run `code` once through `run_code` in `durability`'s workflow code, and return what the model got back.

    The snippet can call `probe()`, which appends the portal threads open while the snippet waits on it to `portals`.
    """

    def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(parts=[ToolCallPart('run_code', {'code': code})])
        return ModelResponse(parts=[TextPart('done')])

    code_mode = CodeMode[None](resource_limits=resource_limits)
    agent = Agent(
        FunctionModel(model),
        name='engine_spec',
        deps_type=type(None),
        capabilities=[code_mode, durability],
    )

    @agent.tool_plain
    def probe() -> str:
        """Report the portal threads open while the snippet waits on this call."""
        assert portals is not None, 'only a snippet run with `portals` calls `probe()`'
        portals.append(_portal_threads())
        return 'probed'

    result = await agent.run('go')
    request = result.all_messages()[2]
    assert isinstance(request, ModelRequest)
    [part] = request.parts
    assert isinstance(part, ToolReturnPart)
    return part


async def _through_portal(durability: RecordingDurability) -> bool:
    """Whether the snippet's Monty calls went through a portal, which is open while a nested tool runs."""
    portals: list[list[str]] = []
    part = await _run_code('await probe()', durability, portals=portals)
    assert part.outcome != 'retried'
    assert len(portals) == 1
    return bool(portals[0])


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
        assert await _through_portal(durability_type()) is through_portal

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

    @pytest.mark.parametrize('passed_for_run', [pytest.param(False, id='on-agent'), pytest.param(True, id='for-run')])
    async def test_an_engine_passed_for_the_run_runs_turns_as_activities(self, passed_for_run: bool) -> None:
        """A durability capability passed with `run(capabilities=...)` decides how snippets run, as one on the agent does.

        In an engine that replays code it doesn't stop when it blocks, each turn must run as a durable
        operation, where `max_duration_seconds` applies, instead of in workflow code with no limit.
        """
        durability = UnboundedReplayingDurability()

        def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            if len(messages) == 1:
                return ModelResponse(parts=[ToolCallPart('run_code', {'code': 'await probe()'})])
            return ModelResponse(parts=[TextPart('done')])

        code_mode = CodeMode[None](monty_sandbox=stored_session_pool)
        agent = Agent(
            FunctionModel(model),
            name='engine_spec',
            deps_type=type(None),
            capabilities=[code_mode] if passed_for_run else [code_mode, durability],
        )

        @agent.tool_plain
        def probe() -> str:
            """Answer the snippet."""
            return 'probed'

        SERVER.reset()
        try:
            result = await agent.run('go', capabilities=[durability] if passed_for_run else None)
        finally:
            SERVER.reset()

        request = result.all_messages()[2]
        assert isinstance(request, ModelRequest)
        [part] = request.parts
        assert isinstance(part, ToolReturnPart)
        assert part.outcome != 'retried'
        # Each turn ran as a durable operation, and no Monty call ran through a portal in workflow code.
        assert any(name.endswith('monty_turn') for name, _ in durability.calls)
        assert _portal_threads() == []
