from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor

import pytest

from pydantic_ai import Agent, UserError, _utils
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel


def return_inner_result(_: list[ModelMessage], __: AgentInfo) -> ModelResponse:
    return ModelResponse(parts=[TextPart('inner result')])


def make_tool_delegate(inner_agent: Agent[None, str], *, stream: bool = False) -> Agent[None, str]:
    def call_delegate(_: list[ModelMessage], __: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[ToolCallPart('delegate', '{"prompt": "hello"}')])

    agent = Agent(FunctionModel(call_delegate))

    @agent.tool_plain
    def delegate(prompt: str) -> str:
        if stream:
            with inner_agent.run_stream_sync(prompt) as result:
                return result.get_output()
        return inner_agent.run_sync(prompt).output

    return agent


def make_output_delegate(inner_agent: Agent[None, str]) -> Agent[None, str]:
    def call_delegate(_: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        assert info.output_tools is not None
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, '{"prompt": "hello"}')])

    def delegate(prompt: str) -> str:
        return inner_agent.run_sync(prompt).output

    return Agent(FunctionModel(call_delegate), output_type=delegate)


NESTED_RUN_SYNC_ERROR = (
    r'`Agent\.run_sync\(\)` cannot be called from a synchronous callback running inside an agent run\. '
    r'Make the callback `async def` and use `await agent\.run\(\.\.\.\)` instead\.'
)


@pytest.mark.parametrize('make_agent', [make_tool_delegate, make_output_delegate])
async def test_run_sync_from_sync_callback_is_rejected(
    make_agent: Callable[[Agent[None, str]], Agent[None, str]],
) -> None:
    inner_agent = Agent(FunctionModel(return_inner_result))
    outer_agent = make_agent(inner_agent)

    with pytest.raises(UserError, match=NESTED_RUN_SYNC_ERROR):
        await outer_agent.run('delegate')

    assert (await inner_agent.run('after callback')).output == 'inner result'


def test_run_sync_from_sync_callback_is_rejected_with_sync_outer_run() -> None:
    inner_agent = Agent(FunctionModel(return_inner_result))
    outer_agent = make_output_delegate(inner_agent)

    with pytest.raises(UserError, match=NESTED_RUN_SYNC_ERROR):
        outer_agent.run_sync('delegate')


def test_run_sync_from_sync_callback_is_rejected_with_bounded_executor() -> None:
    inner_agent = Agent(FunctionModel(return_inner_result))
    outer_agent = make_tool_delegate(inner_agent)

    with ThreadPoolExecutor(max_workers=1) as executor:
        with Agent.using_thread_executor(executor):
            with pytest.raises(UserError, match=NESTED_RUN_SYNC_ERROR):
                outer_agent.run_sync('delegate')

        assert executor.submit(inner_agent.run_sync, 'after callback').result().output == 'inner result'


async def test_run_stream_sync_from_sync_callback_is_rejected() -> None:
    inner_agent = Agent(FunctionModel(return_inner_result))
    outer_agent = make_tool_delegate(inner_agent, stream=True)

    with pytest.raises(
        UserError,
        match=(
            r'`Agent\.run_stream_sync\(\)` cannot be called from a synchronous callback running inside an agent run\. '
            r'Make the callback `async def` and use `async with agent\.run_stream\(\.\.\.\)` instead\.'
        ),
    ):
        await outer_agent.run('delegate')


async def test_run_sync_from_inline_sync_callback_is_rejected() -> None:
    inner_agent = Agent(FunctionModel(return_inner_result))
    outer_agent = make_tool_delegate(inner_agent)

    with _utils.disable_threads():
        with pytest.raises(UserError, match=NESTED_RUN_SYNC_ERROR):
            await outer_agent.run('delegate')

    assert (await inner_agent.run('after callback')).output == 'inner result'


async def test_async_callback_can_delegate_to_agent() -> None:
    call_count = 0

    def call_delegate(_: list[ModelMessage], __: AgentInfo) -> ModelResponse:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return ModelResponse(parts=[ToolCallPart('delegate', '{"prompt": "hello"}')])
        return ModelResponse(parts=[TextPart('outer result')])

    inner_agent = Agent(FunctionModel(return_inner_result))
    outer_agent = Agent(FunctionModel(call_delegate))

    @outer_agent.tool_plain
    async def delegate(prompt: str) -> str:
        return (await inner_agent.run(prompt)).output

    result = await outer_agent.run('delegate')

    assert result.output == 'outer result'
