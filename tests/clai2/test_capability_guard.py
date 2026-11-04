"""Fail-soft run setup: a plugin capability that rejects its configuration costs itself, not the turn."""

from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path

import pytest

from pydantic_ai import Agent, AgentRunResult, RunContext
from pydantic_ai.capabilities import AbstractCapability, Hooks
from pydantic_ai.capabilities.abstract import WrapRunHandler
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import ModelMessage
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_clai2 import Session
from pydantic_clai2.runtime.capability_guard import CapabilitySetupError, PluginGuard, raised_here, setup_errors


@dataclass
class RefusesForRun(AbstractCapability[None]):
    async def for_run(self, ctx: RunContext[None]) -> AbstractCapability[None]:
        raise UserError('bad for_run setting')


@dataclass
class RefusesWrapRun(AbstractCapability[None]):
    async def wrap_run(self, ctx: RunContext[None], *, handler: WrapRunHandler) -> AgentRunResult[object]:
        raise UserError('bad wrap_run setting')


@dataclass
class FailsAfterStart(AbstractCapability[None]):
    async def wrap_run(self, ctx: RunContext[None], *, handler: WrapRunHandler) -> AgentRunResult[object]:
        await handler()
        raise UserError('raised once the run was under way')


def session(agent: Agent[None, str], tmp_path: Path, *plugins: AbstractCapability[None]) -> Session[None, str]:
    return Session(agent, deps=None, plugins=plugins, workspace=tmp_path)


@pytest.mark.parametrize('capability', [RefusesForRun(), RefusesWrapRun()])
async def test_setup_error_fails_the_turn_closed_and_is_reported(
    tmp_path: Path, capability: AbstractCapability[None]
) -> None:
    """Nothing is retried, so no other capability is set up twice; the caller leaves it out of later turns."""
    reported: list[CapabilitySetupError] = []
    guard = PluginGuard[None](capability, plugin='broken')
    conversation = session(Agent(TestModel(custom_output_text='done')), tmp_path, guard)
    conversation.on_setup_error = reported.append
    with pytest.raises(CapabilitySetupError, match="Plugin 'broken': UserError: bad "):
        await conversation.prompt('hello')
    assert [(error.plugin, error.capability) for error in reported] == [('broken', capability)]
    conversation.on_setup_error = None
    with pytest.raises(CapabilitySetupError):
        await conversation.prompt('again')
    conversation.plugins = ()
    assert (await conversation.prompt('without it')).output == 'done'


async def test_unguarded_or_foreign_setup_errors_still_fail(tmp_path: Path) -> None:
    """A guard the session did not add, here one bound to the agent, is not the session's to drop."""
    reported: list[CapabilitySetupError] = []
    guard = PluginGuard[None](RefusesWrapRun(), plugin='bound')
    bound = session(Agent(TestModel(), deps_type=type(None), capabilities=[guard]), tmp_path)
    bound.on_setup_error = reported.append
    with pytest.raises(CapabilitySetupError, match="Plugin 'bound'"):
        await bound.prompt('hello')
    unguarded = session(Agent(TestModel()), tmp_path, RefusesWrapRun())
    unguarded.on_setup_error = reported.append
    with pytest.raises(UserError, match='bad wrap_run setting'):
        await unguarded.prompt('hello')
    assert reported == []
    error = CapabilitySetupError(plugin='x', capability=object(), error=UserError('x'))
    assert not raised_here([RefusesWrapRun()], error)


async def test_simultaneous_setup_errors_report_every_failing_capability(tmp_path: Path) -> None:
    """Core sets capabilities up concurrently, so two refusing at once arrive as one exception group."""
    reported: list[CapabilitySetupError] = []
    conversation = session(
        Agent(TestModel(custom_output_text='done')),
        tmp_path,
        PluginGuard[None](RefusesForRun(), plugin='one'),
        PluginGuard[None](RefusesForRun(), plugin='two'),
    )
    conversation.on_setup_error = reported.append
    with pytest.raises(ExceptionGroup):
        await conversation.prompt('hello')
    assert sorted(error.plugin for error in reported) == ['one', 'two']


async def test_simultaneous_foreign_setup_errors_still_fail(tmp_path: Path) -> None:
    guards = [PluginGuard[None](RefusesForRun(), plugin=name) for name in ('one', 'two')]
    bound = Agent(TestModel(), deps_type=type(None), capabilities=guards)
    with pytest.raises(ExceptionGroup):
        await session(bound, tmp_path).prompt('hello')


def test_setup_errors_only_when_every_failure_is_one() -> None:
    error = CapabilitySetupError(plugin='x', capability=object(), error=UserError('x'))
    assert setup_errors(error) == [error]
    assert setup_errors(ValueError('x')) is None
    assert setup_errors(ExceptionGroup('x', [error, ExceptionGroup('y', [error])])) == [error, error]
    assert setup_errors(ExceptionGroup('x', [error, ValueError('x')])) is None


async def test_errors_after_setup_propagate(tmp_path: Path) -> None:
    reported: list[CapabilitySetupError] = []

    async def broken_model(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
        raise UserError('model misconfigured')
        yield ''  # pragma: no cover -- makes this an async generator.

    model_run = session(
        Agent(FunctionModel(stream_function=broken_model)), tmp_path, PluginGuard[None](Hooks(), plugin='p')
    )
    model_run.on_setup_error = reported.append
    with pytest.raises(UserError, match='model misconfigured'):
        await model_run.prompt('hello')

    agent = Agent(TestModel(call_tools=['explode']))

    @agent.tool_plain
    def explode() -> str:
        raise UserError('tool misconfigured')

    tool_run = session(agent, tmp_path)
    tool_run.plugins = (PluginGuard[None](FailsAfterStart(), plugin='late'),)
    tool_run.on_setup_error = reported.append
    with pytest.raises(UserError, match='tool misconfigured'):
        await tool_run.prompt('hello')
    tool_run.agent = Agent(TestModel())
    with pytest.raises(UserError, match='raised once the run was under way'):
        await tool_run.prompt('hello')
    assert reported == []
