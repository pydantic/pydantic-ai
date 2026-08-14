from __future__ import annotations

import asyncio
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any, cast

import pytest

from pydantic_ai import Agent, RunContext, UserError, _utils
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel


def return_inner_result(_: list[ModelMessage], __: AgentInfo) -> ModelResponse:
    return ModelResponse(parts=[TextPart('inner result')])


def call_delegate(_: list[ModelMessage], __: AgentInfo) -> ModelResponse:
    return ModelResponse(parts=[ToolCallPart('delegate', '{"prompt": "hello"}')])


def make_tool_delegate(inner_agent: Agent[None, str]) -> Agent[None, str]:
    agent = Agent(FunctionModel(call_delegate))

    @agent.tool_plain
    def delegate(prompt: str) -> str:
        return inner_agent.run_sync(prompt).output

    return agent


def make_output_delegate(inner_agent: Agent[None, str]) -> Agent[None, str]:
    def call_delegate(_: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        assert info.output_tools is not None
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, '{"prompt": "hello"}')])

    def delegate(prompt: str) -> str:
        return inner_agent.run_sync(prompt).output

    return Agent(FunctionModel(call_delegate), output_type=delegate)


def make_awaitable_tool_delegate(inner_agent: Agent[None, str]) -> Agent[None, str]:
    async def run_delegate(prompt: str) -> str:
        return inner_agent.run_sync(prompt).output

    agent = Agent(FunctionModel(call_delegate))

    @agent.tool_plain
    def delegate(prompt: str) -> Any:
        return run_delegate(prompt)

    return agent


NESTED_RUN_SYNC_ERROR = (
    r'`Agent\.run_sync\(\)` cannot be called from inside another agent run\. '
    r'Make the calling function `async def` and use `await agent\.run\(\.\.\.\)` instead\.'
)


@pytest.mark.parametrize('make_agent', [make_tool_delegate, make_output_delegate, make_awaitable_tool_delegate])
async def test_run_sync_from_sync_callback_is_rejected(
    make_agent: Callable[[Agent[None, str]], Agent[None, str]],
) -> None:
    inner_agent = Agent(FunctionModel(return_inner_result))
    outer_agent = make_agent(inner_agent)

    with pytest.raises(UserError, match=NESTED_RUN_SYNC_ERROR):
        await outer_agent.run('delegate')


async def test_run_sync_from_metadata_callback_is_rejected() -> None:
    inner_agent = Agent(FunctionModel(return_inner_result))

    def metadata(_: RunContext) -> dict[str, Any]:
        return cast(dict[str, Any], inner_agent.run_sync('delegate'))

    outer_agent = Agent(FunctionModel(return_inner_result), metadata=metadata)

    with pytest.raises(UserError, match=NESTED_RUN_SYNC_ERROR):
        await outer_agent.run('outer')


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


async def test_run_stream_sync_from_sync_callback_is_rejected() -> None:
    inner_agent = Agent(FunctionModel(return_inner_result))
    outer_agent = Agent(FunctionModel(call_delegate))

    @outer_agent.tool_plain
    def delegate(prompt: str) -> str:
        return inner_agent.run_stream_sync(prompt).get_output()

    with pytest.raises(
        UserError,
        match=(
            r'`Agent\.run_stream_sync\(\)` cannot be called from inside another agent run\. '
            r'Make the calling function `async def` and use `async with agent\.run_stream\(\.\.\.\)` instead\.'
        ),
    ):
        await outer_agent.run('delegate')


async def test_run_sync_from_inline_sync_callback_is_rejected() -> None:
    inner_agent = Agent(FunctionModel(return_inner_result))
    outer_agent = make_tool_delegate(inner_agent)

    with _utils.disable_threads():
        with pytest.raises(UserError, match=NESTED_RUN_SYNC_ERROR):
            await outer_agent.run('delegate')


async def test_run_sync_succeeds_in_child_task_after_parent_run() -> None:
    parent_finished = asyncio.Event()
    inner_agent = Agent(FunctionModel(return_inner_result))
    outer_agent = Agent(FunctionModel(return_inner_result))

    async def run_after_parent() -> str:
        await parent_finished.wait()
        return (await asyncio.to_thread(inner_agent.run_sync, 'after parent')).output

    async with outer_agent.iter('parent') as run:
        child_task = asyncio.create_task(run_after_parent())
        async for _ in run:
            pass

    parent_finished.set()
    assert await child_task == 'inner result'


async def test_run_sync_in_child_task_is_rejected_while_parent_run_is_active() -> None:
    inner_finished = asyncio.Event()
    delegate_agent = Agent(FunctionModel(return_inner_result))
    inner_agent = Agent(FunctionModel(return_inner_result))
    outer_agent = Agent(FunctionModel(return_inner_result))

    async def run_after_inner() -> None:
        await inner_finished.wait()
        await asyncio.to_thread(delegate_agent.run_sync, 'while parent is active')

    async with outer_agent.iter('outer') as outer_run:
        async with inner_agent.iter('inner') as inner_run:
            child_task = asyncio.create_task(run_after_inner())
            async for _ in inner_run:
                pass

        async for _ in outer_run:
            pass
        inner_finished.set()
        with pytest.raises(UserError, match=NESTED_RUN_SYNC_ERROR):
            await child_task
