"""PROTOTYPE: characterize callback-driven re-entrancy across extension surfaces.

The question is whether process isolation plus strict xfails can make deadlocks and
state-mutation failures bounded, visible, and cheap to add without inventing a
surface-specific test harness.
"""

from __future__ import annotations

import multiprocessing
from collections.abc import Callable
from multiprocessing.synchronize import Event as ProcessEvent

import anyio
import pytest
from typing_extensions import Self

from pydantic_ai import Agent, FunctionToolset, RunContext
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart
from pydantic_ai.models.concurrency import ConcurrencyLimitedModel
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import ToolDefinition

_STARTUP_TIMEOUT = 10
_COMPLETION_TIMEOUT = 0.5


def _run_and_signal(operation: Callable[[], None], started: ProcessEvent) -> None:
    started.set()
    operation()


def _assert_completes_in_subprocess(operation: Callable[[], None]) -> None:
    if 'forkserver' in multiprocessing.get_all_start_methods():
        context = multiprocessing.get_context('forkserver')
    else:
        context = multiprocessing.get_context('spawn')
    started = context.Event()
    process = context.Process(target=_run_and_signal, args=(operation, started))
    try:
        process.start()
        if not started.wait(_STARTUP_TIMEOUT):
            if process.is_alive():
                process.kill()
            process.join()
            pytest.fail(f'{operation.__name__} did not start within {_STARTUP_TIMEOUT} seconds')

        process.join(_COMPLETION_TIMEOUT)

        if process.is_alive():
            process.kill()
            process.join()
            pytest.fail(f'{operation.__name__} did not complete within {_COMPLETION_TIMEOUT} seconds')

        assert process.exitcode == 0
    finally:
        process.close()


def _complete_normally() -> None:
    pass


def test_completion_contract_accepts_a_successful_operation() -> None:
    _assert_completes_in_subprocess(_complete_normally)


class _ReenteringToolset(FunctionToolset[None]):
    def __init__(self) -> None:
        super().__init__()
        self.agent: Agent[None, str] | None = None

    async def __aenter__(self) -> Self:
        assert self.agent is not None
        await self.agent.__aenter__()
        return self


async def _reenter_agent_lifecycle() -> None:
    toolset = _ReenteringToolset()
    agent = Agent(TestModel(), toolsets=[toolset], deps_type=type(None))
    toolset.agent = agent

    async with agent:
        pass


def _run_agent_lifecycle_reentry() -> None:
    anyio.run(_reenter_agent_lifecycle)


@pytest.mark.xfail(strict=True, reason='PROTOTYPE: toolset callback re-entry currently deadlocks')
def test_agent_lifecycle_callback_reentry_completes() -> None:
    _assert_completes_in_subprocess(_run_agent_lifecycle_reentry)


async def _reenter_limited_model() -> None:
    agent: Agent[None, str] | None = None
    reentered = False

    async def model_function(_messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        nonlocal reentered
        if not reentered:
            reentered = True
            assert agent is not None
            await agent.run('inner')
        return ModelResponse(parts=[TextPart(content='done')])

    model = ConcurrencyLimitedModel(FunctionModel(model_function), limiter=1)
    agent = Agent(model)
    await agent.run('outer')


def _run_limited_model_reentry() -> None:
    anyio.run(_reenter_limited_model)


@pytest.mark.xfail(strict=True, reason='PROTOTYPE: recursive limited-model request currently self-waits')
def test_limited_model_callback_reentry_completes() -> None:
    _assert_completes_in_subprocess(_run_limited_model_reentry)


def _first_tool() -> str:
    return 'first'


def _second_tool() -> str:
    return 'second'


async def _mutate_toolset_during_prepare() -> None:
    toolset = FunctionToolset[None]()

    async def prepare(_ctx: RunContext[None], tool_def: ToolDefinition) -> ToolDefinition:
        toolset.add_function(_second_tool)
        return tool_def

    toolset.add_function(_first_tool, prepare=prepare)
    await Agent(TestModel(), toolsets=[toolset], deps_type=type(None)).run('prepare tools')


def _run_toolset_prepare_mutation() -> None:
    anyio.run(_mutate_toolset_during_prepare)


@pytest.mark.xfail(strict=True, reason='PROTOTYPE: prepare callback mutation currently invalidates iteration')
def test_tool_prepare_registration_is_deferred_to_the_next_snapshot() -> None:
    _assert_completes_in_subprocess(_run_toolset_prepare_mutation)
