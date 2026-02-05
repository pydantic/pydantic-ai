"""Test ApprovalRequired and CallDeferred within code mode execution.

Flow being tested:
1. LLM generates code that calls tools
2. Tools raise ApprovalRequired/CallDeferred
3. Monty bundles interrupted calls into CodeInterruptedError with checkpoint
4. Agent returns DeferredToolRequests with inner tool calls for user to approve
5. User approves and sends DeferredToolResults
6. Agent resumes from checkpoint, tools execute, result returned
"""

from __future__ import annotations

import pytest

from pydantic_ai import Agent
from pydantic_ai._run_context import RunContext
from pydantic_ai.exceptions import ApprovalRequired, CallDeferred
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.tools import DeferredToolRequests, DeferredToolResults
from pydantic_ai.toolsets.code_mode import CodeModeToolset
from pydantic_ai.toolsets.function import FunctionToolset

pytestmark = pytest.mark.anyio


@pytest.mark.xfail(reason='Code mode approval flow not yet implemented')
async def test_code_mode_mixed_approval_and_deferred():
    """Test both ApprovalRequired and CallDeferred in parallel within code mode."""

    def sensitive_action(ctx: RunContext[None], value: int) -> int:
        if not ctx.tool_call_approved:
            raise ApprovalRequired()
        return value * 10

    def external_service(data: str) -> str:
        raise CallDeferred()

    toolset: FunctionToolset[None] = FunctionToolset()
    toolset.add_function(sensitive_action, takes_ctx=True)
    toolset.add_function(external_service, takes_ctx=False)
    code_toolset = CodeModeToolset(wrapped=toolset)

    call_count = 0

    def llm(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            # Fire both in parallel (no await until both fired)
            code = """f1 = sensitive_action(value=42)
f2 = external_service(data="test")
r1 = await f1
r2 = await f2
{"sensitive": r1, "external": r2}"""
            return ModelResponse(parts=[ToolCallPart('run_code', {'code': code}, tool_call_id='rc1')])
        return ModelResponse(parts=[TextPart('Both done!')])

    agent: Agent[None, str | DeferredToolRequests] = Agent(FunctionModel(llm), output_type=[str, DeferredToolRequests])

    # Run 1
    async with code_toolset:
        result = await agent.run('Do both', toolsets=[code_toolset])

    assert isinstance(result.output, DeferredToolRequests)
    assert len(result.output.approvals) + len(result.output.calls) == 2
    assert result.output.context

    # Run 2: approve and provide external result
    approvals = {c.tool_call_id: True for c in result.output.approvals}
    calls = {c.tool_call_id: 'ext_result' for c in result.output.calls}

    async with code_toolset:
        result = await agent.run(
            message_history=result.all_messages(),
            deferred_tool_results=DeferredToolResults(
                approvals=approvals,  # pyright: ignore[reportArgumentType]
                calls=calls,
                context=result.output.context,
            ),
            toolsets=[code_toolset],
        )

    assert result.output == 'Both done!'
