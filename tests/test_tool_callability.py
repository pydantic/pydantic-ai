"""Execution eligibility is independent of model-facing schema disclosure.

FunctionModel deliberately calls unadvertised names; real models cannot reliably reproduce that
choice, so these tests pin dispatch, validation, approval and history through the public agent API.
"""

from __future__ import annotations

from typing import Any, Literal

import pytest

from pydantic_ai import Agent
from pydantic_ai.capabilities import AbstractCapability, Capability, HandleDeferredToolCalls
from pydantic_ai.exceptions import ApprovalRequired, ModelRetry, ToolRetryError
from pydantic_ai.messages import (
    ModelMessage,
    ModelResponse,
    TextPart,
    ToolAvailabilityDeltaPart,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.tool_manager import ToolManager
from pydantic_ai.tools import (
    DeferredToolRequests,
    DeferredToolResults,
    RunContext,
    ToolApproved,
    ToolDefinition,
    ToolDenied,
)
from pydantic_ai.toolsets import FunctionToolset, WrapperToolset
from pydantic_ai.toolsets.abstract import ToolsetTool
from pydantic_ai.usage import RunUsage


@pytest.mark.parametrize('owned', [False, True])
@pytest.mark.parametrize('deferred', [False, True])
async def test_known_tool_executes_without_implicit_disclosure(owned: bool, deferred: bool):
    calls: list[int] = []
    toolset = FunctionToolset[None]()

    @toolset.tool(defer_loading=deferred)
    def lookup(ctx: RunContext[None], key: int) -> int:
        assert ctx.is_tool_callable('lookup')
        assert 'lookup' in ctx.callable_tool_names
        assert ctx.is_tool_available('lookup') is not deferred
        assert ctx.discovered_tool_names == set()
        calls.append(key)
        return key * 2

    def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        assert ('lookup' in {tool.name for tool in info.function_tools}) is not deferred
        for message in messages:
            for part in message.parts:
                if isinstance(part, ToolReturnPart) and part.tool_name == 'lookup':
                    assert part.outcome == 'success'
                    return ModelResponse(parts=[TextPart(str(part.content))])
        return ModelResponse(parts=[ToolCallPart('lookup', {'key': 3})])

    agent = Agent(
        FunctionModel(model),
        deps_type=type(None),
        toolsets=[] if owned else [toolset],
        capabilities=[Capability(id='owner', toolsets=[toolset])] if owned else [],
    )
    # A fresh run can reuse the same known name without restoring discovery history.
    for _ in range(2):
        result = await agent.run('Run saved lookup')
        assert result.output == '6'
        assert result.usage.tool_calls == 1
        assert not any(
            isinstance(part, ToolAvailabilityDeltaPart) for msg in result.all_messages() for part in msg.parts
        )
    assert calls == [3, 3]


async def test_nested_manager_resolves_its_own_registry():
    """A nested manager's context can refer to a parent that does not contain the inner tool."""
    inner = FunctionToolset[None]()

    @inner.tool_plain(defer_loading=True)
    def lookup(key: int) -> int:
        return key * 2

    ctx = RunContext(deps=None, model=TestModel(), usage=RunUsage())
    ctx.tool_manager = await ToolManager(FunctionToolset[None]()).for_run_step(ctx)
    manager = ToolManager(inner, ctx=ctx, tools=await inner.get_tools(ctx))
    assert not ctx.is_tool_callable('lookup')
    assert await manager.handle_call(ToolCallPart('lookup', {'key': 4})) == 8
    assert ctx.usage.tool_calls == 1
    assert ctx.discovered_tool_names == set()


@pytest.mark.parametrize('prepared', [None, False, True])
async def test_callability_queries_require_current_membership(prepared: bool | None):
    ctx = RunContext(deps=None, model=TestModel(), usage=RunUsage(), discovered_tool_names={'gone'})
    if prepared is not None:
        ctx.tool_manager = ToolManager(FunctionToolset[None]())
        if prepared:
            ctx.tool_manager = await ctx.tool_manager.for_run_step(ctx)
    assert ctx.callable_tool_names == set()
    assert not ctx.is_tool_callable('gone')
    assert ctx.is_tool_callable(ToolDefinition(name='supplied', defer_loading=True))
    assert not ctx.is_tool_callable(ToolDefinition(name='supplied', capability_id='inactive'))


async def test_hidden_tool_still_validates_arguments():
    calls: list[int] = []
    toolset = FunctionToolset[None]()

    @toolset.tool_plain(defer_loading=True)
    def lookup(key: int) -> int:
        calls.append(key)
        return key

    def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        returns = [part for msg in messages for part in msg.parts if isinstance(part, ToolReturnPart)]
        if not returns:
            return ModelResponse(parts=[ToolCallPart('lookup', {'key': 'invalid'})])
        if len(returns) == 1:
            assert returns[0].outcome == 'retried'
            assert 'int_parsing' in str(returns[0].content)
            return ModelResponse(parts=[ToolCallPart('lookup', {'key': 5})])
        return ModelResponse(parts=[TextPart(str(returns[-1].content))])

    result = await Agent(FunctionModel(model), deps_type=type(None), toolsets=[toolset]).run('go')
    assert result.output == '5'
    assert calls == [5]
    assert result.usage.tool_calls == 1


@pytest.mark.parametrize('nested', [False, True])
@pytest.mark.parametrize('declarative', [False, True])
@pytest.mark.parametrize('decision', ['approve', 'deny', 'override', 'invalid'])
async def test_hidden_tool_approval_and_override_validation(
    nested: bool, declarative: bool, decision: Literal['approve', 'deny', 'override', 'invalid']
):
    calls: list[int] = []
    approvals: list[str] = []
    toolset = FunctionToolset[None]()

    @toolset.tool(defer_loading=True, requires_approval=declarative)
    def lookup(ctx: RunContext[None], key: int) -> int:
        if not ctx.tool_call_approved:
            raise ApprovalRequired
        calls.append(key)
        return key

    async def approve(ctx: RunContext[None], requests: DeferredToolRequests) -> DeferredToolResults:
        assert calls == []
        assert ctx.discovered_tool_names == set()
        assert len(requests.approvals) == 1
        call = requests.approvals[0]
        assert call.tool_name == 'lookup'
        approvals.append(call.tool_name)
        result: bool | ToolApproved = decision != 'deny'
        if decision in ('override', 'invalid'):
            result = ToolApproved(override_args={'key': 7 if decision == 'override' else 'invalid'})
        return DeferredToolResults(approvals={call.tool_call_id: result})

    def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        assert 'lookup' not in {tool.name for tool in info.function_tools}
        returns = [part for msg in messages for part in msg.parts if isinstance(part, ToolReturnPart)]
        if returns:
            part = returns[-1]
            if not nested:
                expected = 'denied' if decision == 'deny' else 'retried' if decision == 'invalid' else 'success'
                assert part.outcome == expected
            return ModelResponse(parts=[TextPart(str(part.content))])
        return ModelResponse(parts=[ToolCallPart('caller' if nested else 'lookup', {} if nested else {'key': 3})])

    agent = Agent(
        FunctionModel(model),
        deps_type=type(None),
        toolsets=[toolset],
        capabilities=[HandleDeferredToolCalls(handler=approve)],
    )
    if nested:

        @agent.tool
        async def caller(ctx: RunContext[None]) -> str:
            assert ctx.tool_manager is not None
            try:
                result = await ctx.tool_manager.handle_call(ToolCallPart('lookup', {'key': 3}))
            except ToolRetryError:
                assert decision == 'invalid'
                return 'invalid override'
            if isinstance(result, ToolDenied):
                assert decision == 'deny'
                return 'denied'
            return str(result)

    result = await agent.run('Run saved call')
    assert approvals == ['lookup']
    assert calls == ([7] if decision == 'override' else [3] if decision == 'approve' else [])
    assert result.usage.tool_calls == len(calls) + int(nested)
    assert not any(isinstance(part, ToolAvailabilityDeltaPart) for msg in result.all_messages() for part in msg.parts)


@pytest.mark.parametrize('reject_at', [None, 'validator', 'hook'])
async def test_hidden_tool_keeps_validator_hooks_and_final_wrapper(reject_at: str | None):
    events: list[str] = []
    toolset = FunctionToolset[None]()

    def validate(ctx: RunContext[None], key: int) -> None:
        events.append('validator')
        if reject_at == 'validator':
            raise ModelRetry('validator rejected')

    @toolset.tool(defer_loading=True, args_validator=validate)
    def lookup(ctx: RunContext[None], key: int) -> int:
        events.append('tool')
        return key

    class Policy(AbstractCapability[None]):
        async def before_tool_execute(
            self, ctx: RunContext[None], *, call: ToolCallPart, tool_def: ToolDefinition, args: dict[str, Any]
        ) -> dict[str, Any]:
            events.append('hook')
            if reject_at == 'hook':
                raise ModelRetry('hook rejected')
            return args

    class Wrapper(WrapperToolset[None]):
        async def call_tool(
            self, name: str, tool_args: dict[str, Any], ctx: RunContext[None], tool: ToolsetTool[None]
        ) -> Any:
            events.append('wrapper')
            return await super().call_tool(name, tool_args, ctx, tool)

    def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        assert info.function_tools == []
        returns = [part for msg in messages for part in msg.parts if isinstance(part, ToolReturnPart)]
        if returns:
            assert returns[-1].outcome == ('retried' if reject_at else 'success')
            return ModelResponse(parts=[TextPart('done')])
        return ModelResponse(parts=[ToolCallPart('lookup', {'key': 3})])

    result = await Agent(
        FunctionModel(model), deps_type=type(None), toolsets=[Wrapper(toolset)], capabilities=[Policy()]
    ).run('go')
    assert events == (
        ['validator']
        if reject_at == 'validator'
        else ['validator', 'hook']
        if reject_at == 'hook'
        else ['validator', 'hook', 'wrapper', 'tool']
    )
    assert result.usage.tool_calls == int(reject_at is None)


async def test_saved_name_uses_current_prepared_registry():
    calls: list[int] = []
    toolset = FunctionToolset[bool]()

    async def prepare(ctx: RunContext[bool], tool_def: ToolDefinition) -> ToolDefinition | None:
        return tool_def if ctx.deps else None

    @toolset.tool_plain(defer_loading=True, prepare=prepare)
    def lookup(key: int) -> int:
        calls.append(key)
        return key

    def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        assert info.function_tools == []
        returns = [part for msg in messages for part in msg.parts if isinstance(part, ToolReturnPart)]
        if returns:
            return ModelResponse(parts=[TextPart(str(returns[-1].content))])
        return ModelResponse(parts=[ToolCallPart('lookup', {'key': 3})])

    agent = Agent(FunctionModel(model), deps_type=bool, toolsets=[toolset])
    assert (await agent.run('saved call', deps=True)).output == '3'
    denied = await agent.run('same saved call', deps=False)
    assert 'Unknown tool name' in denied.output
    assert calls == [3]
    assert denied.usage.tool_calls == 0
