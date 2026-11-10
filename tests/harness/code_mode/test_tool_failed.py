from __future__ import annotations

from pydantic_ai import Agent, ModelRetry, ToolFailed
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai_harness import CodeMode


def test_nested_tool_failed_returns_failed_result_without_retry() -> None:
    model_calls = 0

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal model_calls
        model_calls += 1
        if model_calls == 1:
            return ModelResponse(parts=[ToolCallPart(tool_name='run_code', args={'code': 'await fail()'})])
        return ModelResponse(parts=[TextPart(content='done')])

    agent = Agent(FunctionModel(respond), capabilities=[CodeMode()], retries=1)

    @agent.tool_plain
    def fail() -> None:
        raise ToolFailed('nested failure')

    result = agent.run_sync('run the code')
    run_code_return = next(
        part
        for message in result.all_messages()
        for part in message.parts
        if isinstance(part, ToolReturnPart) and part.tool_name == 'run_code'
    )
    assert model_calls == 2
    assert run_code_return.outcome == 'failed'
    assert run_code_return.content == 'nested failure'


def test_nested_model_retry_still_retries() -> None:
    model_calls = 0

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal model_calls
        model_calls += 1
        if model_calls == 1:
            return ModelResponse(parts=[ToolCallPart(tool_name='run_code', args={'code': 'await retry()'})])
        return ModelResponse(parts=[TextPart(content='done')])

    agent = Agent(FunctionModel(respond), capabilities=[CodeMode()], retries=1)

    @agent.tool_plain
    def retry() -> None:
        raise ModelRetry('retry nested tool')

    result = agent.run_sync('run the code')
    run_code_return = next(
        part
        for message in result.all_messages()
        for part in message.parts
        if isinstance(part, ToolReturnPart) and part.tool_name == 'run_code'
    )
    assert model_calls == 2
    assert run_code_return.outcome == 'retried'
    assert 'retry nested tool' in str(run_code_return.content)


def test_repeated_nested_tool_failed_does_not_exhaust_run_code_retries() -> None:
    failure_count = 4
    model_calls = 0

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal model_calls
        model_calls += 1
        if model_calls <= failure_count:
            return ModelResponse(parts=[ToolCallPart(tool_name='run_code', args={'code': 'await fail()'})])
        return ModelResponse(parts=[TextPart(content='done')])

    agent = Agent(FunctionModel(respond), capabilities=[CodeMode()], retries=1)

    @agent.tool_plain
    def fail() -> None:
        raise ToolFailed('repeated nested failure')

    result = agent.run_sync('run the code')
    run_code_returns = [
        part
        for message in result.all_messages()
        for part in message.parts
        if isinstance(part, ToolReturnPart) and part.tool_name == 'run_code'
    ]
    assert model_calls == failure_count + 1
    assert len(run_code_returns) == failure_count
    assert all(part.outcome == 'failed' for part in run_code_returns)


def test_native_tool_failed_remains_failed_without_retry() -> None:
    model_calls = 0

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal model_calls
        model_calls += 1
        if model_calls == 1:
            return ModelResponse(parts=[ToolCallPart(tool_name='fail', args={})])
        return ModelResponse(parts=[TextPart(content='done')])

    agent = Agent(FunctionModel(respond), retries=1)

    @agent.tool_plain
    def fail() -> None:
        raise ToolFailed('native failure')

    result = agent.run_sync('call the tool')
    tool_return = next(
        part
        for message in result.all_messages()
        for part in message.parts
        if isinstance(part, ToolReturnPart) and part.tool_name == 'fail'
    )
    assert model_calls == 2
    assert tool_return.outcome == 'failed'
    assert tool_return.content == 'native failure'
