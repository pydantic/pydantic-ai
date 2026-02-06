"""Test ApprovalRequired and CallDeferred within code mode execution.

Flow being tested:
1. LLM generates code that calls tools
2. Tools raise ApprovalRequired/CallDeferred
3. Monty bundles interrupted calls into CodeInterruptedError with checkpoint
4. Code mode catches it and raises ApprovalRequired with checkpoint in context
5. Agent returns DeferredToolRequests with run_code in approvals, nested call details in context
6. User approves run_code and provides nested call results in context['results']
7. Agent resumes from checkpoint, approved tools execute, result returned
"""

from __future__ import annotations

import pytest
from inline_snapshot import snapshot

from pydantic_ai import Agent
from pydantic_ai._run_context import RunContext
from pydantic_ai.exceptions import ApprovalRequired, CallDeferred, UserError
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.tools import DeferredToolRequests, DeferredToolResults, ToolApproved
from pydantic_ai.toolsets.code_mode import CodeModeContext, CodeModeToolset
from pydantic_ai.toolsets.function import FunctionToolset

from ..conftest import IsBytes, IsStr

pytestmark = pytest.mark.anyio


async def test_code_mode_mixed_approval_and_deferred():
    """Test both ApprovalRequired and CallDeferred in parallel within code mode.

    The run_code tool ends up in approvals, with nested call details in context.
    User approves via approvals and provides nested results in context['results'].
    """

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

    # Run 1: Code mode runs, nested tools need approval/external execution
    async with code_toolset:
        result = await agent.run('Do both', toolsets=[code_toolset])

    # Snapshot shows the full structure of the deferred result
    assert result.output == snapshot(
        DeferredToolRequests(
            calls=[],
            approvals=[
                ToolCallPart(
                    tool_name='run_code',
                    args={
                        'code': """\
f1 = sensitive_action(value=42)
f2 = external_service(data="test")
r1 = await f1
r2 = await f2
{"sensitive": r1, "external": r2}"""
                    },
                    tool_call_id='rc1',
                )
            ],
            context={
                'rc1': {
                    'checkpoint': IsBytes(),
                    'interrupted_calls': [
                        {
                            'call_id': IsStr(),
                            'tool_name': 'sensitive_action',
                            'args': (),
                            'kwargs': {'value': 42},
                            'type': 'approval',
                        },
                        {
                            'call_id': IsStr(),
                            'tool_name': 'external_service',
                            'args': (),
                            'kwargs': {'data': 'test'},
                            'type': 'external',
                        },
                    ],
                }
            },
            metadata={},
        )
    )

    # Extract values needed for resume flow
    assert isinstance(result.output, DeferredToolRequests)
    run_code_call_id = result.output.approvals[0].tool_call_id
    ctx: CodeModeContext = result.output.context[run_code_call_id]  # type: ignore[assignment]

    # Build results for nested calls - type-safe access to interrupted_calls
    nested_results: dict[str, object] = {}
    for ic in ctx['interrupted_calls']:
        if ic['type'] == 'approval':
            nested_results[ic['call_id']] = ToolApproved()
        else:
            nested_results[ic['call_id']] = 'ext_result'

    # Build resume context: original context + results (type-safe)
    resume_ctx: CodeModeContext = {
        'checkpoint': ctx['checkpoint'],
        'interrupted_calls': ctx['interrupted_calls'],
        'results': nested_results,
    }
    # No cast needed - Mapping accepts TypedDict
    resume_context = {run_code_call_id: resume_ctx}

    # Run 2: Resume with approval and nested results
    async with code_toolset:
        result = await agent.run(
            message_history=result.all_messages(),
            deferred_tool_results=DeferredToolResults(
                approvals={run_code_call_id: ToolApproved()},
                context=resume_context,
            ),
            toolsets=[code_toolset],
        )

    assert result.output == snapshot('Both done!')


async def test_code_mode_resume_missing_context_error():
    """Test that resuming without context raises a clear error."""

    def sensitive_action(ctx: RunContext[None], value: int) -> int:
        if not ctx.tool_call_approved:
            raise ApprovalRequired()
        return value * 10

    toolset: FunctionToolset[None] = FunctionToolset()
    toolset.add_function(sensitive_action, takes_ctx=True)
    code_toolset = CodeModeToolset(wrapped=toolset)

    call_count = 0

    def llm(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            code = 'await sensitive_action(value=42)'
            return ModelResponse(parts=[ToolCallPart('run_code', {'code': code}, tool_call_id='rc1')])
        return ModelResponse(parts=[TextPart('Done!')])

    agent: Agent[None, str | DeferredToolRequests] = Agent(FunctionModel(llm), output_type=[str, DeferredToolRequests])

    # Run 1: Get deferred result
    async with code_toolset:
        result = await agent.run('Do it', toolsets=[code_toolset])

    # Snapshot shows the full structure of the deferred result
    assert result.output == snapshot(
        DeferredToolRequests(
            calls=[],
            approvals=[
                ToolCallPart(
                    tool_name='run_code',
                    args={'code': 'await sensitive_action(value=42)'},
                    tool_call_id='rc1',
                )
            ],
            context={
                'rc1': {
                    'checkpoint': IsBytes(),
                    'interrupted_calls': [
                        {
                            'call_id': IsStr(),
                            'tool_name': 'sensitive_action',
                            'args': (),
                            'kwargs': {'value': 42},
                            'type': 'approval',
                        },
                    ],
                }
            },
            metadata={},
        )
    )

    # Extract values needed for resume flow
    assert isinstance(result.output, DeferredToolRequests)
    run_code_call_id = result.output.approvals[0].tool_call_id

    # Run 2: Try to resume WITHOUT context - should error
    async with code_toolset:
        with pytest.raises(UserError) as exc_info:
            await agent.run(
                message_history=result.all_messages(),
                deferred_tool_results=DeferredToolResults(
                    approvals={run_code_call_id: ToolApproved()},
                    # Missing context!
                ),
                toolsets=[code_toolset],
            )

    assert str(exc_info.value) == snapshot(
        'Code mode resume requires context with checkpoint. Pass back the original DeferredToolRequests.context[tool_call_id] with an added "results" key mapping call_id to ToolApproved(), ToolDenied(), or external result.'
    )


async def test_code_mode_resume_missing_results_error():
    """Test that resuming without results for interrupted calls raises a clear error."""

    def sensitive_action(ctx: RunContext[None], value: int) -> int:
        if not ctx.tool_call_approved:
            raise ApprovalRequired()
        return value * 10

    toolset: FunctionToolset[None] = FunctionToolset()
    toolset.add_function(sensitive_action, takes_ctx=True)
    code_toolset = CodeModeToolset(wrapped=toolset)

    call_count = 0

    def llm(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            code = 'await sensitive_action(value=42)'
            return ModelResponse(parts=[ToolCallPart('run_code', {'code': code}, tool_call_id='rc1')])
        return ModelResponse(parts=[TextPart('Done!')])

    agent: Agent[None, str | DeferredToolRequests] = Agent(FunctionModel(llm), output_type=[str, DeferredToolRequests])

    # Run 1: Get deferred result
    async with code_toolset:
        result = await agent.run('Do it', toolsets=[code_toolset])

    # Snapshot shows the full structure of the deferred result
    assert result.output == snapshot(
        DeferredToolRequests(
            calls=[],
            approvals=[
                ToolCallPart(
                    tool_name='run_code',
                    args={'code': 'await sensitive_action(value=42)'},
                    tool_call_id='rc1',
                )
            ],
            context={
                'rc1': {
                    'checkpoint': IsBytes(),
                    'interrupted_calls': [
                        {
                            'call_id': IsStr(),
                            'tool_name': 'sensitive_action',
                            'args': (),
                            'kwargs': {'value': 42},
                            'type': 'approval',
                        },
                    ],
                }
            },
            metadata={},
        )
    )

    # Extract values needed for resume flow
    assert isinstance(result.output, DeferredToolRequests)
    run_code_call_id = result.output.approvals[0].tool_call_id

    # Get the original context (has checkpoint and interrupted_calls)
    original_context = result.output.context[run_code_call_id]

    # Run 2: Try to resume with context but MISSING results - should error
    async with code_toolset:
        with pytest.raises(UserError) as exc_info:
            await agent.run(
                message_history=result.all_messages(),
                deferred_tool_results=DeferredToolResults(
                    approvals={run_code_call_id: ToolApproved()},
                    context={
                        run_code_call_id: {
                            # Include checkpoint but missing 'results'!
                            **original_context,
                        }
                    },
                ),
                toolsets=[code_toolset],
            )

    assert str(exc_info.value) == snapshot(
        'Code mode resume requires context with results for nested calls. Add a "results" key to the context mapping call_id to ToolApproved(), ToolDenied(), or the external result.'
    )
