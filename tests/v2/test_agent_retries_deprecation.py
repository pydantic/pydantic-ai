"""1.x deprecation warnings for split agent retry fields.

`retries` remains the canonical retry configuration. The 1.x compatibility shims
`tool_retries`, `output_retries`, and run/override `output_retries` warn before
the split fields are removed in v2.
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import BaseModel

from pydantic_ai import Agent, AgentSpec
from pydantic_ai._warnings import PydanticAIDeprecationWarning
from pydantic_ai.agent import WrapperAgent
from pydantic_ai.exceptions import UnexpectedModelBehavior
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.output import ToolOutput


class Foo(BaseModel):
    a: int
    b: str


def _invalid_output_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    assert info.output_tools is not None
    return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, '{"bad_field": 1}')])


@pytest.mark.parametrize(
    'kwargs, expected_warning',
    [
        ({'tool_retries': 5}, r'`Agent\(tool_retries=\.\.\.\)` is deprecated'),
        ({'output_retries': 5}, r'`Agent\(output_retries=\.\.\.\)` is deprecated'),
    ],
    ids=['tool_retries', 'output_retries'],
)
def test_agent_init_split_retry_kwargs_warn(kwargs: dict[str, Any], expected_warning: str):
    with pytest.warns(PydanticAIDeprecationWarning, match=expected_warning):
        Agent('test', **kwargs)


def test_agent_spec_tool_retries_warns_and_sets_tool_budget():
    with pytest.warns(PydanticAIDeprecationWarning, match=r'`AgentSpec\.tool_retries` is deprecated'):
        agent = Agent.from_spec({'model': 'test', 'tool_retries': 5})

    assert agent._max_tool_retries == 5  # pyright: ignore[reportPrivateUsage]
    assert agent._max_output_retries == 1  # pyright: ignore[reportPrivateUsage]


def test_agent_spec_output_retries_warns_and_sets_output_budget():
    with pytest.warns(PydanticAIDeprecationWarning, match=r'`AgentSpec\.output_retries` is deprecated'):
        agent = Agent.from_spec({'model': 'test', 'retries': 3, 'output_retries': 10})

    assert agent._max_tool_retries == 3  # pyright: ignore[reportPrivateUsage]
    assert agent._max_output_retries == 10  # pyright: ignore[reportPrivateUsage]


def test_agent_spec_split_retry_fields_warn_on_model_construction():
    with pytest.warns(PydanticAIDeprecationWarning, match=r'`AgentSpec\.tool_retries` is deprecated'):
        tool_spec = AgentSpec(model='test', tool_retries=5)
    assert tool_spec.tool_retries == 5

    with pytest.warns(PydanticAIDeprecationWarning, match=r'`AgentSpec\.output_retries` is deprecated'):
        output_spec = AgentSpec(model='test', output_retries=6)
    assert output_spec.output_retries == 6


def test_from_spec_preserves_zero_split_retry_budgets_during_deprecation():
    output_call_count = 0

    def output_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal output_call_count
        output_call_count += 1
        return _invalid_output_model(messages, info)

    with pytest.warns(PydanticAIDeprecationWarning, match=r'`AgentSpec\.output_retries` is deprecated'):
        output_agent = Agent.from_spec(
            {'model': 'test', 'output_retries': 0},
            model=FunctionModel(output_model),
            output_type=ToolOutput(Foo),
        )

    with pytest.raises(UnexpectedModelBehavior, match=r'Exceeded maximum output retries \(0\)'):
        output_agent.run_sync('Hello')

    assert output_call_count == 1

    tool_call_count = 0

    def tool_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal tool_call_count
        tool_call_count += 1
        if tool_call_count == 1:
            return ModelResponse(parts=[ToolCallPart('plain_tool', '{"bad_field": 1}')])
        if tool_call_count == 2:
            return ModelResponse(parts=[ToolCallPart('plain_tool', '{"x": 1}')])
        return ModelResponse(parts=[TextPart('done')])

    with pytest.warns(PydanticAIDeprecationWarning, match=r'`AgentSpec\.tool_retries` is deprecated'):
        tool_agent = Agent.from_spec(
            {'model': 'test', 'tool_retries': 0},
            model=FunctionModel(tool_model),
        )

    @tool_agent.tool_plain
    def plain_tool(x: int) -> str:
        return str(x)

    with pytest.raises(UnexpectedModelBehavior, match=r"Tool 'plain_tool' exceeded max retries count of 0"):
        tool_agent.run_sync('Hello')

    result = tool_agent.run_sync('Hello again')
    assert result.output == 'done'
    assert tool_call_count == 3


@pytest.mark.parametrize(
    'run_kwargs, expected_budget',
    [
        ({'output_retries': 0}, 0),
        ({'retries': {'output': 0}, 'output_retries': 5}, 0),
        ({'output_retries': None}, 3),
    ],
    ids=['output_retries', 'retries-wins-over-output_retries', 'output_retries-none-keeps-default'],
)
def test_run_output_retries_kwarg_warns(run_kwargs: dict[str, Any], expected_budget: int):
    call_count = 0

    def return_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal call_count
        call_count += 1
        return _invalid_output_model(messages, info)

    agent = Agent(FunctionModel(return_model), output_type=ToolOutput(Foo), retries={'output': 3})

    with pytest.warns(PydanticAIDeprecationWarning, match=r'agent\.run_sync\(output_retries=\.\.\.\)` is deprecated'):
        with pytest.raises(UnexpectedModelBehavior, match=rf'Exceeded maximum output retries \({expected_budget}\)'):
            agent.run_sync('Hello', **run_kwargs)

    assert call_count == expected_budget + 1


def test_override_output_retries_kwarg_warns():
    call_count = 0

    def return_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal call_count
        call_count += 1
        return _invalid_output_model(messages, info)

    agent = Agent(FunctionModel(return_model), output_type=ToolOutput(Foo), retries={'output': 3})

    with pytest.warns(PydanticAIDeprecationWarning, match=r'agent\.override\(output_retries=\.\.\.\)` is deprecated'):
        with agent.override(output_retries=0):
            with pytest.raises(UnexpectedModelBehavior, match=r'Exceeded maximum output retries \(0\)'):
                agent.run_sync('Hello')

    assert call_count == 1


def test_wrapper_override_forwards_deprecated_output_retries_kwarg():
    call_count = 0

    def return_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal call_count
        call_count += 1
        return _invalid_output_model(messages, info)

    agent = Agent(FunctionModel(return_model), output_type=ToolOutput(Foo), retries={'output': 3})
    wrapped = WrapperAgent(agent)

    with pytest.warns(PydanticAIDeprecationWarning, match=r'agent\.override\(output_retries=\.\.\.\)` is deprecated'):
        with wrapped.override(output_retries=5, retries=0):
            with pytest.raises(UnexpectedModelBehavior, match=r'Exceeded maximum output retries \(0\)'):
                wrapped.run_sync('Hello')

    assert call_count == 1


def test_agent_init_split_retry_none_warns():
    with pytest.warns(PydanticAIDeprecationWarning, match=r'`Agent\(tool_retries=\.\.\.\)` is deprecated'):
        Agent(TestModel(), tool_retries=None)  # pyright: ignore[reportCallIssue]
