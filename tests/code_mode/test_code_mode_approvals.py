"""Test ApprovalRequired and CallDeferred within code mode execution.

Parameterized across all CodeRuntime implementations (Monty, stdio subprocess).

Flow being tested:
1. LLM generates code that calls tools
2. Tools raise ApprovalRequired/CallDeferred
3. Runtime bundles interrupted calls into CodeInterruptedError with checkpoint
4. Code mode catches it and raises ApprovalRequired with checkpoint in context
5. Agent returns DeferredToolRequests with run_code in approvals, nested call details in context
6. User approves run_code and provides nested call results in context['results']
7. Agent resumes from checkpoint, approved tools execute, result returned
"""

from __future__ import annotations

from typing import Any

import pytest
from inline_snapshot import snapshot

from pydantic_ai import Agent
from pydantic_ai._run_context import RunContext
from pydantic_ai.exceptions import ApprovalRequired, CallDeferred, UserError
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.runtime.abstract import CodeRuntime
from pydantic_ai.tools import DeferredToolRequests, DeferredToolResults, ToolApproved, ToolDenied
from pydantic_ai.toolsets.code_mode import CodeModeContext, CodeModeToolset, InterruptedCall
from pydantic_ai.toolsets.function import FunctionToolset

from ..conftest import IsBytes, IsList, IsStr

pytestmark = pytest.mark.anyio


# --- Shared tool functions ---


def sensitive_action(ctx: RunContext[None], value: int) -> int:
    """Tool that requires approval before execution."""
    if not ctx.tool_call_approved:
        raise ApprovalRequired()
    return value * 10


def external_service(data: str) -> str:
    """Tool that is always deferred to external execution."""
    raise CallDeferred()


# --- Test helpers ---


def _make_code_agent(
    code: str,
    final_text: str = 'Done!',
) -> Agent[None, str | DeferredToolRequests]:
    """Build an agent where the LLM sends run_code on first call, text after."""
    call_count = 0

    def llm(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return ModelResponse(parts=[ToolCallPart('run_code', {'code': code}, tool_call_id='rc1')])
        return ModelResponse(parts=[TextPart(final_text)])

    return Agent(FunctionModel(llm), output_type=[str, DeferredToolRequests])


async def _run_to_interrupt(
    agent: Agent[None, str | DeferredToolRequests],
    code_toolset: CodeModeToolset[None],
) -> tuple[Any, str, CodeModeContext]:
    """Run the agent expecting interruption -> DeferredToolRequests.

    Returns (result, run_code_call_id, context).
    """
    async with code_toolset:
        result = await agent.run('Go', toolsets=[code_toolset])
    assert isinstance(result.output, DeferredToolRequests)
    call_id = result.output.approvals[0].tool_call_id
    ctx: CodeModeContext = result.output.context[call_id]  # type: ignore[assignment]
    return result, call_id, ctx


async def _resume(
    agent: Agent[None, str | DeferredToolRequests],
    code_toolset: CodeModeToolset[None],
    prev_result: Any,
    call_id: str,
    ctx: CodeModeContext,
    nested_results: dict[str, object],
) -> Any:
    """Resume from checkpoint with the given nested results."""
    resume_ctx: CodeModeContext = {**ctx, 'results': nested_results}
    async with code_toolset:
        return await agent.run(
            message_history=prev_result.all_messages(),
            deferred_tool_results=DeferredToolResults(
                approvals={call_id: ToolApproved()},
                context={call_id: resume_ctx},
            ),
            toolsets=[code_toolset],
        )


def _approve_all(interrupted_calls: list[InterruptedCall]) -> dict[str, object]:
    """Build results dict: ToolApproved for approvals, 'ext_result' for externals."""
    return {
        ic['call_id']: ToolApproved() if ic['type'] == 'approval' else 'ext_result'
        for ic in interrupted_calls
    }


# --- Tests ---


async def test_code_mode_mixed_approval_and_deferred(code_runtime: CodeRuntime):
    """Test both ApprovalRequired and CallDeferred in parallel within code mode.

    The run_code tool ends up in approvals, with nested call details in context.
    User approves via approvals and provides nested results in context['results'].
    """
    toolset: FunctionToolset[None] = FunctionToolset()
    toolset.add_function(sensitive_action, takes_ctx=True)
    toolset.add_function(external_service, takes_ctx=False)
    code_toolset = CodeModeToolset(wrapped=toolset, runtime=code_runtime)

    code = """f1 = sensitive_action(value=42)
f2 = external_service(data="test")
r1 = await f1
r2 = await f2
{"sensitive": r1, "external": r2}"""

    agent = _make_code_agent(code, 'Both done!')
    result, call_id, ctx = await _run_to_interrupt(agent, code_toolset)

    # Sort interrupted_calls by tool_name for deterministic snapshot comparison
    ctx['interrupted_calls'] = sorted(ctx['interrupted_calls'], key=lambda ic: ic['tool_name'])

    assert result.output == snapshot(
        DeferredToolRequests(
            approvals=[
                ToolCallPart(
                    tool_name='run_code',
                    args={
                        'code': """\
f1 = sensitive_action(value=42)
f2 = external_service(data="test")
r1 = await f1
r2 = await f2
{"sensitive": r1, "external": r2}\
"""
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
                            'tool_name': 'external_service',
                            'args': (),
                            'kwargs': {'data': 'test'},
                            'type': 'external',
                        },
                        {
                            'call_id': IsStr(),
                            'tool_name': 'sensitive_action',
                            'args': (),
                            'kwargs': {'value': 42},
                            'type': 'approval',
                        },
                    ],
                    'code': IsStr(),
                    'functions': IsList(IsStr(), length=2),
                    'signatures': IsList(IsStr(), length=2),
                }
            },
        )
    )

    result = await _resume(agent, code_toolset, result, call_id, ctx, _approve_all(ctx['interrupted_calls']))
    assert result.output == snapshot('Both done!')


async def _setup_resume_error(
    code_runtime: CodeRuntime,
) -> tuple[Agent[None, str | DeferredToolRequests], CodeModeToolset[None], Any, str, CodeModeContext]:
    """Shared setup for resume validation error tests."""
    toolset: FunctionToolset[None] = FunctionToolset()
    toolset.add_function(sensitive_action, takes_ctx=True)
    code_toolset = CodeModeToolset(wrapped=toolset, runtime=code_runtime)
    agent = _make_code_agent('await sensitive_action(value=42)')
    result, call_id, ctx = await _run_to_interrupt(agent, code_toolset)
    return agent, code_toolset, result, call_id, ctx


async def test_code_mode_resume_missing_context(code_runtime: CodeRuntime):
    """Test that resuming without context raises a clear error."""
    agent, code_toolset, result, call_id, _ = await _setup_resume_error(code_runtime)

    async with code_toolset:
        with pytest.raises(UserError) as exc_info:
            await agent.run(
                message_history=result.all_messages(),
                deferred_tool_results=DeferredToolResults(approvals={call_id: ToolApproved()}),
                toolsets=[code_toolset],
            )

    assert str(exc_info.value) == snapshot(
        'Code mode resume requires context with checkpoint. Pass back the original DeferredToolRequests.context[tool_call_id] with an added "results" key mapping call_id to ToolApproved(), ToolDenied(), or external result.'
    )


async def test_code_mode_resume_missing_results(code_runtime: CodeRuntime):
    """Test that resuming without results in context raises a clear error."""
    agent, code_toolset, result, call_id, ctx = await _setup_resume_error(code_runtime)

    async with code_toolset:
        with pytest.raises(UserError) as exc_info:
            await agent.run(
                message_history=result.all_messages(),
                deferred_tool_results=DeferredToolResults(
                    approvals={call_id: ToolApproved()}, context={call_id: {**ctx}}
                ),
                toolsets=[code_toolset],
            )

    assert str(exc_info.value) == snapshot(
        'Code mode resume requires context with results for nested calls. Add a "results" key to the context mapping call_id to ToolApproved(), ToolDenied(), or the external result.'
    )


async def test_code_mode_mixed_success_and_interrupted(code_runtime: CodeRuntime):
    """Test that calls which succeed before an interruption are NOT re-executed on resume.

    3 parallel tools: 1 succeeds immediately (returning complex nested data),
    1 needs approval, 1 is deferred.
    On resume the successful call's result is fed back via completed_results,
    avoiding double execution. The complex dict must round-trip through checkpoint
    serialize/deserialize for resume to succeed.
    """
    call_counts: dict[str, int] = {'fast': 0, 'sensitive': 0, 'external': 0}

    # Local tools with closures for call counting
    def fast_lookup(key: str) -> dict[str, Any]:
        call_counts['fast'] += 1
        return {
            'items': [{'id': 1, 'name': 'widget', 'tags': ['a', 'b']}, {'id': 2, 'name': 'gadget', 'tags': ['c']}],
            'metadata': {'total': 2, 'key': key},
        }

    def counted_sensitive(ctx: RunContext[None], value: int) -> int:
        call_counts['sensitive'] += 1
        if not ctx.tool_call_approved:
            raise ApprovalRequired()
        return value * 10

    def counted_external(data: str) -> str:
        call_counts['external'] += 1
        raise CallDeferred()

    toolset: FunctionToolset[None] = FunctionToolset()
    toolset.add_function(fast_lookup, takes_ctx=False)
    toolset.add_function(counted_sensitive, takes_ctx=True)
    toolset.add_function(counted_external, takes_ctx=False)
    code_toolset = CodeModeToolset(wrapped=toolset, runtime=code_runtime)

    code = """\
f1 = fast_lookup(key="abc")
f2 = counted_sensitive(value=42)
f3 = counted_external(data="test")
r1 = await f1
r2 = await f2
r3 = await f3
{"fast": r1, "sensitive": r2, "external": r3}"""

    agent = _make_code_agent(code, 'All three done!')
    result, call_id, ctx = await _run_to_interrupt(agent, code_toolset)

    assert call_counts == {'fast': 1, 'sensitive': 1, 'external': 1}
    assert len(ctx['interrupted_calls']) == 2
    assert {ic['tool_name'] for ic in ctx['interrupted_calls']} == {'counted_sensitive', 'counted_external'}

    result = await _resume(agent, code_toolset, result, call_id, ctx, _approve_all(ctx['interrupted_calls']))
    assert result.output == snapshot('All three done!')

    assert call_counts['fast'] == 1  # NOT re-executed
    assert call_counts['sensitive'] == 2  # once for ApprovalRequired, once approved
    assert call_counts['external'] == 1  # deferred, result provided directly


async def test_code_mode_nested_approvals(code_runtime: CodeRuntime):
    """Test that a resumed execution can trigger a second round of approvals.

    Flow:
      Run 1: action_1 needs approval -> interrupted
      Run 2: resume with action_1 approved -> code continues -> action_2 needs approval
      Run 3: resume with action_2 approved -> code completes
    """

    def action_1(ctx: RunContext[None], value: int) -> int:
        if not ctx.tool_call_approved:
            raise ApprovalRequired()
        return value * 10

    def action_2(ctx: RunContext[None], value: int) -> int:
        if not ctx.tool_call_approved:
            raise ApprovalRequired()
        return value + 1

    toolset: FunctionToolset[None] = FunctionToolset()
    toolset.add_function(action_1, takes_ctx=True)
    toolset.add_function(action_2, takes_ctx=True)
    code_toolset = CodeModeToolset(wrapped=toolset, runtime=code_runtime)

    code = """\
r1 = await action_1(value=5)
r2 = await action_2(value=r1)
{"first": r1, "second": r2}"""

    agent = _make_code_agent(code, 'All done!')

    # Run 1: action_1 needs approval
    result, call_id, ctx1 = await _run_to_interrupt(agent, code_toolset)
    assert ctx1['interrupted_calls'][0]['tool_name'] == 'action_1'

    # Run 2: approve action_1 -> hits action_2 interruption
    result = await _resume(
        agent, code_toolset, result, call_id, ctx1,
        {ctx1['interrupted_calls'][0]['call_id']: ToolApproved()},
    )
    assert isinstance(result.output, DeferredToolRequests)
    call_id_2 = result.output.approvals[0].tool_call_id
    ctx2: CodeModeContext = result.output.context[call_id_2]  # type: ignore[assignment]
    assert ctx2['interrupted_calls'][0]['tool_name'] == 'action_2'
    assert ctx2['interrupted_calls'][0]['kwargs'] == {'value': 50}

    # Run 3: approve action_2 -> completes
    result = await _resume(
        agent, code_toolset, result, call_id_2, ctx2,
        {ctx2['interrupted_calls'][0]['call_id']: ToolApproved()},
    )
    assert result.output == snapshot('All done!')


async def test_code_mode_tool_denied_on_resume(code_runtime: CodeRuntime):
    """Test that ToolDenied propagates as a ModelRetry so the LLM sees the denial message."""
    toolset: FunctionToolset[None] = FunctionToolset()
    toolset.add_function(sensitive_action, takes_ctx=True)
    code_toolset = CodeModeToolset(wrapped=toolset, runtime=code_runtime)

    agent = _make_code_agent('await sensitive_action(value=42)', 'Action was denied.')
    result, call_id, ctx = await _run_to_interrupt(agent, code_toolset)

    result = await _resume(
        agent, code_toolset, result, call_id, ctx,
        {ctx['interrupted_calls'][0]['call_id']: ToolDenied(message='Not allowed')},
    )
    assert result.output == snapshot('Action was denied.')
