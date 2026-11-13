"""Lifecycle of the toolsets a `DynamicToolset` factory returns, through agent runs.

These tests drive runs with `TestModel` and `FunctionModel` rather than cassettes: the toolset lifecycle runs
before and around each model request, and they assert the tool definitions the model receives and the order of
the hooks, which no recorded provider response can change.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Self

import pytest

from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import AbstractCapability, AgentCapability, PrefixTools, Toolset
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.toolsets import AbstractToolset, CombinedToolset, DynamicToolset, FunctionToolset

from .function_model_prefix import TOOLS_CHANGE_MID_RUN


def greet() -> str:
    return 'hello'


def greet_toolset(ctx: RunContext[None]) -> AbstractToolset[None]:
    return FunctionToolset([greet])


@pytest.mark.parametrize('dynamic', [False, True])
@pytest.mark.parametrize(
    ('make_capability', 'tool_name'),
    [
        pytest.param(lambda: PrefixTools(Toolset(greet_toolset), prefix='ns'), 'ns_greet', id='prefix-tools'),
        pytest.param(lambda: Toolset(DynamicToolset(greet_toolset)), 'greet', id='dynamic-toolset'),
    ],
)
async def test_dynamic_capability_initializes_nested_toolset_factory(
    make_capability: Callable[[], AbstractCapability[None]], tool_name: str, dynamic: bool
):
    """A capability holding a nested dynamic toolset exposes the same tools when resolved dynamically."""
    capability = make_capability()
    model = TestModel(custom_output_text='done')
    capabilities: list[AgentCapability[None]] = [lambda ctx: capability] if dynamic else [capability]
    agent = Agent(model, deps_type=type(None), capabilities=capabilities)

    result = await agent.run('hello')

    assert result.output == 'done'
    assert model.last_model_request_parameters is not None
    assert [tool.name for tool in model.last_model_request_parameters.function_tools] == [tool_name]


class LifecycleToolset(FunctionToolset[None]):
    def __init__(self, events: list[str], *, prepared: bool = False):
        super().__init__([greet])
        self.events = events
        self.is_prepared = prepared
        self.entered = False

    async def for_run(self, ctx: RunContext[None]) -> AbstractToolset[None]:
        self.events.append('for_run')
        return LifecycleToolset(self.events, prepared=True)

    async def __aenter__(self) -> Self:
        assert self.is_prepared, 'The factory result must be prepared before entry'
        assert not self.entered
        self.entered = True
        self.events.append('enter')
        return self

    async def for_run_step(self, ctx: RunContext[None]) -> AbstractToolset[None]:
        assert self.entered
        self.events.append(f'step:{ctx.run_step}')
        return self

    async def __aexit__(self, *args: Any) -> None:
        assert self.entered
        self.entered = False
        self.events.append('exit')


@pytest.mark.parametrize('per_run_step', [False, True])
async def test_dynamic_toolset_prepares_stable_factory_result_once(per_run_step: bool):
    """A fresh prepared copy stays entered when the factory returns its original again."""
    events: list[str] = []
    original = LifecycleToolset(events)
    factory_calls = 0

    def factory(ctx: RunContext[None]) -> AbstractToolset[None]:
        nonlocal factory_calls
        factory_calls += 1
        return original

    agent = Agent(TestModel(), deps_type=type(None), toolsets=[DynamicToolset(factory, per_run_step=per_run_step)])
    for _ in range(2):
        await agent.run('hello')

    assert events == ['for_run', 'enter', 'step:1', 'step:2', 'exit'] * 2
    assert factory_calls == (4 if per_run_step else 2)
    assert not original.is_prepared
    assert not original.entered


@pytest.mark.moves_cache_prefix(reason=TOOLS_CHANGE_MID_RUN)
async def test_dynamic_toolset_prepares_replacements_and_removes_tools():
    """Changing a factory result exits its prepared copy before entering the next one."""
    events: list[str] = []
    originals = iter([LifecycleToolset(events), LifecycleToolset(events), None])
    offered_tools: list[list[str]] = []

    def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        names = [tool.name for tool in info.function_tools]
        offered_tools.append(names)
        if names:
            return ModelResponse(parts=[ToolCallPart('greet', {})])
        return ModelResponse(parts=[TextPart('done')])

    agent = Agent(FunctionModel(model), deps_type=type(None), toolsets=[lambda ctx: next(originals)])
    assert (await agent.run('hello')).output == 'done'

    assert offered_tools == [['greet'], ['greet'], []]
    assert events == ['for_run', 'enter', 'step:1', 'exit', 'for_run', 'enter', 'step:2', 'exit']


async def test_dynamic_toolset_preparation_failure_exits_only_active_copy():
    """A replacement that fails preparation must never be entered or exited."""
    events: list[str] = []

    class FailingPreparationToolset(LifecycleToolset):
        async def for_run(self, ctx: RunContext[None]) -> AbstractToolset[None]:
            events.append('prepare:failed')
            raise RuntimeError('preparation failed')

    original = LifecycleToolset(events)
    replacement = FailingPreparationToolset(events)
    originals = iter([original, replacement])
    agent = Agent(TestModel(), deps_type=type(None), toolsets=[lambda ctx: next(originals)])

    with pytest.raises(RuntimeError, match='preparation failed'):
        await agent.run('hello')

    assert events == ['for_run', 'enter', 'step:1', 'exit', 'prepare:failed']
    assert not original.entered
    assert not replacement.entered


@pytest.mark.parametrize(
    'replace_factory_result',
    [False, pytest.param(True, marks=pytest.mark.moves_cache_prefix(reason=TOOLS_CHANGE_MID_RUN))],
)
async def test_dynamic_toolset_exits_entered_combined_after_step_replacement(replace_factory_result: bool):
    """Per-step operational replacements must not discard the entered container's cleanup."""
    events: list[str] = []

    class ReplacingToolset(FunctionToolset[None]):
        async def for_run_step(self, ctx: RunContext[None]) -> AbstractToolset[None]:
            return FunctionToolset([greet])

    class ResourceToolset(FunctionToolset[None]):
        async def __aenter__(self) -> Self:
            events.append('enter')
            return self

        async def __aexit__(self, *args: Any) -> None:
            events.append('exit')

    combined = CombinedToolset([ReplacingToolset(), ResourceToolset()])
    replacement = FunctionToolset[None]()
    offered_tools: list[list[str]] = []

    def factory(ctx: RunContext[None]) -> AbstractToolset[None]:
        if replace_factory_result and ctx.run_step > 1:
            # The resource from the first factory result must already be closed
            # when the model receives the new result's tool list.
            return replacement
        return combined

    def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        offered_tools.append([tool.name for tool in info.function_tools])
        if len(offered_tools) == 1:
            return ModelResponse(parts=[ToolCallPart('greet', {})])
        if replace_factory_result:
            assert events == ['enter', 'exit']
        return ModelResponse(parts=[TextPart('done')])

    agent = Agent(FunctionModel(model), deps_type=type(None), toolsets=[factory])
    assert (await agent.run('hello')).output == 'done'
    assert offered_tools == [['greet'], [] if replace_factory_result else ['greet']]
    assert events == ['enter', 'exit']
