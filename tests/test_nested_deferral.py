"""Tests for nested deferred tool calls.

When a tool raises `CallDeferred(deferred_tool_requests=...)`, the framework
should surface the nested deferred calls with composite IDs (`parent::child`)
and reconstruct per-parent `DeferredToolResults` on resume.
"""

from __future__ import annotations

from pydantic_ai import Agent, RunContext
from pydantic_ai.exceptions import ApprovalRequired, CallDeferred
from pydantic_ai.messages import (
    ModelMessage,
    ModelResponse,
    TextPart,
    ToolCallPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.tools import DeferredToolRequests, DeferredToolResults, ToolApproved, ToolDenied


def test_nested_single_approval():
    """A parent tool raises CallDeferred with one nested approval request.

    Flow: first run produces composite IDs, second run approves and resumes.
    """
    call_count = 0

    def llm(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return ModelResponse(
                parts=[ToolCallPart('run_subagent', {'task': 'do something'}, tool_call_id='parent_call')]
            )
        else:
            return ModelResponse(parts=[TextPart('All done!')])

    agent = Agent(FunctionModel(llm), output_type=[str, DeferredToolRequests])

    @agent.tool
    def run_subagent(ctx: RunContext[None], task: str) -> str:
        if ctx.deferred_tool_results is not None:
            # On resume, we have the nested results — use them
            approved = ctx.deferred_tool_results.approvals.get('child_approval')
            if isinstance(approved, ToolApproved):
                return f'Subagent completed: {task} (approved)'
            return f'Subagent completed: {task} (denied)'

        # First call: the subagent needs approval for an inner tool
        nested = DeferredToolRequests(
            approvals=[ToolCallPart('dangerous_action', {'target': 'db'}, tool_call_id='child_approval')],
        )
        raise CallDeferred(deferred_tool_requests=nested)

    # First run: should get composite IDs
    result = agent.run_sync('Do it')
    assert isinstance(result.output, DeferredToolRequests)
    assert len(result.output.calls) == 0
    assert len(result.output.approvals) == 1
    assert result.output.approvals[0].tool_call_id == 'parent_call::child_approval'
    assert result.output.approvals[0].tool_name == 'dangerous_action'

    # Second run: approve the nested call
    messages = result.all_messages()
    result = agent.run_sync(
        message_history=messages,
        deferred_tool_results=DeferredToolResults(
            approvals={'parent_call::child_approval': ToolApproved()},
        ),
    )
    assert result.output == 'All done!'


def test_nested_multiple_approvals():
    """A parent tool raises CallDeferred with multiple nested approval requests."""
    call_count = 0

    def llm(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return ModelResponse(
                parts=[ToolCallPart('run_subagent', {'task': 'cleanup'}, tool_call_id='parent')]
            )
        else:
            return ModelResponse(parts=[TextPart('Done!')])

    agent = Agent(FunctionModel(llm), output_type=[str, DeferredToolRequests])

    @agent.tool
    def run_subagent(ctx: RunContext[None], task: str) -> str:
        if ctx.deferred_tool_results is not None:
            results = []
            for child_id, approval in ctx.deferred_tool_results.approvals.items():
                if isinstance(approval, ToolApproved):
                    results.append(f'{child_id}: approved')
                elif isinstance(approval, ToolDenied):
                    results.append(f'{child_id}: denied')
            return ', '.join(results)

        nested = DeferredToolRequests(
            approvals=[
                ToolCallPart('delete_file', {'path': '/tmp/a'}, tool_call_id='child_1'),
                ToolCallPart('delete_file', {'path': '/tmp/b'}, tool_call_id='child_2'),
            ],
        )
        raise CallDeferred(deferred_tool_requests=nested)

    result = agent.run_sync('Cleanup')
    assert isinstance(result.output, DeferredToolRequests)
    assert len(result.output.approvals) == 2
    approval_ids = {a.tool_call_id for a in result.output.approvals}
    assert approval_ids == {'parent::child_1', 'parent::child_2'}

    messages = result.all_messages()
    result = agent.run_sync(
        message_history=messages,
        deferred_tool_results=DeferredToolResults(
            approvals={
                'parent::child_1': ToolApproved(),
                'parent::child_2': ToolDenied('Not allowed'),
            },
        ),
    )
    assert result.output == 'Done!'


def test_nested_mixed_external_and_approval():
    """A parent tool raises CallDeferred with both external calls and approval requests."""
    call_count = 0

    def llm(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return ModelResponse(
                parts=[ToolCallPart('orchestrate', {'plan': 'execute'}, tool_call_id='parent')]
            )
        else:
            return ModelResponse(parts=[TextPart('Completed!')])

    agent = Agent(FunctionModel(llm), output_type=[str, DeferredToolRequests])

    @agent.tool
    def orchestrate(ctx: RunContext[None], plan: str) -> str:
        if ctx.deferred_tool_results is not None:
            ext_result = ctx.deferred_tool_results.calls.get('ext_call')
            approval = ctx.deferred_tool_results.approvals.get('approval_call')
            return f'external={ext_result}, approved={isinstance(approval, ToolApproved)}'

        nested = DeferredToolRequests(
            calls=[ToolCallPart('background_task', {'input': 'data'}, tool_call_id='ext_call')],
            approvals=[ToolCallPart('risky_action', {'level': 'high'}, tool_call_id='approval_call')],
        )
        raise CallDeferred(deferred_tool_requests=nested)

    result = agent.run_sync('Execute plan')
    assert isinstance(result.output, DeferredToolRequests)
    assert len(result.output.calls) == 1
    assert len(result.output.approvals) == 1
    assert result.output.calls[0].tool_call_id == 'parent::ext_call'
    assert result.output.approvals[0].tool_call_id == 'parent::approval_call'

    messages = result.all_messages()
    result = agent.run_sync(
        message_history=messages,
        deferred_tool_results=DeferredToolResults(
            calls={'parent::ext_call': 'task_result_42'},
            approvals={'parent::approval_call': ToolApproved()},
        ),
    )
    assert result.output == 'Completed!'


def test_nested_metadata_propagation():
    """Metadata is correctly propagated through composite IDs."""
    call_count = 0

    def llm(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return ModelResponse(
                parts=[ToolCallPart('wrapper', {'x': 1}, tool_call_id='parent')]
            )
        else:
            return ModelResponse(parts=[TextPart('Done!')])

    agent = Agent(FunctionModel(llm), output_type=[str, DeferredToolRequests])

    @agent.tool
    def wrapper(ctx: RunContext[None], x: int) -> str:
        if ctx.deferred_tool_results is not None:
            child_meta = ctx.deferred_tool_results.metadata.get('child')
            return f'child_meta={child_meta}'

        nested = DeferredToolRequests(
            approvals=[ToolCallPart('inner', {'y': 2}, tool_call_id='child')],
            metadata={'child': {'reason': 'needs approval'}},
        )
        raise CallDeferred(metadata={'parent_info': 'wrapper'}, deferred_tool_requests=nested)

    result = agent.run_sync('Go')
    assert isinstance(result.output, DeferredToolRequests)

    # Parent metadata should be keyed by parent ID
    assert result.output.metadata.get('parent') == {'parent_info': 'wrapper'}
    # Child metadata should be keyed by composite ID
    assert result.output.metadata.get('parent::child') == {'reason': 'needs approval'}

    messages = result.all_messages()
    result = agent.run_sync(
        message_history=messages,
        deferred_tool_results=DeferredToolResults(
            approvals={'parent::child': ToolApproved()},
            metadata={'parent::child': {'user_note': 'ok'}},
        ),
    )
    assert result.output == 'Done!'


def test_nested_with_regular_tools():
    """Nested deferred calls work alongside regular (non-deferred) tools in the same response."""
    call_count = 0

    def llm(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return ModelResponse(
                parts=[
                    ToolCallPart('regular_tool', {'x': 10}, tool_call_id='regular'),
                    ToolCallPart('deferred_wrapper', {'task': 'inner'}, tool_call_id='deferred'),
                ]
            )
        elif call_count == 2:
            # After first run with deferred, we get the deferred result back
            return ModelResponse(parts=[TextPart('Final answer')])
        else:
            return ModelResponse(parts=[TextPart('Unexpected')])

    agent = Agent(FunctionModel(llm), output_type=[str, DeferredToolRequests])

    @agent.tool_plain
    def regular_tool(x: int) -> int:
        return x * 2

    @agent.tool
    def deferred_wrapper(ctx: RunContext[None], task: str) -> str:
        if ctx.deferred_tool_results is not None:
            return f'inner done: {ctx.deferred_tool_results.approvals}'

        nested = DeferredToolRequests(
            approvals=[ToolCallPart('needs_approval', {'z': 1}, tool_call_id='inner')],
        )
        raise CallDeferred(deferred_tool_requests=nested)

    # First run: regular_tool executes, deferred_wrapper creates nested deferral
    result = agent.run_sync('Go')
    assert isinstance(result.output, DeferredToolRequests)
    assert len(result.output.approvals) == 1
    assert result.output.approvals[0].tool_call_id == 'deferred::inner'

    # Second run: provide nested approval
    messages = result.all_messages()
    result = agent.run_sync(
        message_history=messages,
        deferred_tool_results=DeferredToolResults(
            approvals={'deferred::inner': ToolApproved()},
        ),
    )
    assert result.output == 'Final answer'


def test_nested_denial():
    """When a nested approval is denied, the parent tool receives the denial."""
    call_count = 0

    def llm(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return ModelResponse(
                parts=[ToolCallPart('wrapper', {'x': 1}, tool_call_id='parent')]
            )
        else:
            return ModelResponse(parts=[TextPart('Done!')])

    agent = Agent(FunctionModel(llm), output_type=[str, DeferredToolRequests])

    @agent.tool
    def wrapper(ctx: RunContext[None], x: int) -> str:
        if ctx.deferred_tool_results is not None:
            approval = ctx.deferred_tool_results.approvals.get('child')
            if isinstance(approval, ToolDenied):
                return f'Child denied: {approval.message}'
            return 'Child approved'

        nested = DeferredToolRequests(
            approvals=[ToolCallPart('inner_action', {'a': 1}, tool_call_id='child')],
        )
        raise CallDeferred(deferred_tool_requests=nested)

    result = agent.run_sync('Go')
    assert isinstance(result.output, DeferredToolRequests)

    messages = result.all_messages()
    result = agent.run_sync(
        message_history=messages,
        deferred_tool_results=DeferredToolResults(
            approvals={'parent::child': ToolDenied('Not safe')},
        ),
    )
    assert result.output == 'Done!'


def test_nested_context_round_trip():
    """Context is correctly propagated through nested deferred calls and available on resume."""
    call_count = 0

    def llm(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return ModelResponse(
                parts=[ToolCallPart('run_subagent', {'task': 'process'}, tool_call_id='parent')]
            )
        else:
            return ModelResponse(parts=[TextPart('Done!')])

    agent = Agent(FunctionModel(llm), output_type=[str, DeferredToolRequests])

    @agent.tool
    def run_subagent(ctx: RunContext[None], task: str) -> str:
        if ctx.deferred_tool_results is not None:
            # On resume, the context from the nested requests should be available
            child_ctx = ctx.deferred_tool_results.context.get('child')
            return f'resumed with context: {child_ctx}'

        # First call: subagent needs approval, and we attach context (e.g. message history)
        nested = DeferredToolRequests(
            approvals=[ToolCallPart('dangerous_action', {'target': 'db'}, tool_call_id='child')],
            context={'child': {'conversation_state': [1, 2, 3]}},
        )
        raise CallDeferred(
            context={'parent_state': 'in_progress'},
            deferred_tool_requests=nested,
        )

    # First run: should get composite IDs with context
    result = agent.run_sync('Do it')
    assert isinstance(result.output, DeferredToolRequests)
    # Parent context should be keyed by parent ID
    assert result.output.context.get('parent') == {'parent_state': 'in_progress'}
    # Child context should be keyed by composite ID
    assert result.output.context.get('parent::child') == {'conversation_state': [1, 2, 3]}

    # Second run: pass context back unchanged
    messages = result.all_messages()
    result = agent.run_sync(
        message_history=messages,
        deferred_tool_results=DeferredToolResults(
            approvals={'parent::child': ToolApproved()},
            context={'parent::child': {'conversation_state': [1, 2, 3]}},
        ),
    )
    assert result.output == 'Done!'


def test_context_on_approval_required():
    """Context flows correctly through ApprovalRequired exceptions."""
    call_count = 0

    def llm(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return ModelResponse(
                parts=[ToolCallPart('risky_action', {'x': 1}, tool_call_id='call_1')]
            )
        else:
            return ModelResponse(parts=[TextPart('Done!')])

    agent = Agent(FunctionModel(llm), output_type=[str, DeferredToolRequests])

    @agent.tool
    def risky_action(ctx: RunContext[None], x: int) -> str:
        if not ctx.tool_call_approved:
            raise ApprovalRequired(
                metadata={'reason': 'dangerous'},
                context={'state': 'pending', 'value': x},
            )
        # On approval, context should be available
        assert ctx.tool_call_context == {'state': 'pending', 'value': 1}
        return f'executed with x={x}'

    result = agent.run_sync('Do it')
    assert isinstance(result.output, DeferredToolRequests)
    assert result.output.context.get('call_1') == {'state': 'pending', 'value': 1}

    messages = result.all_messages()
    result = agent.run_sync(
        message_history=messages,
        deferred_tool_results=DeferredToolResults(
            approvals={'call_1': ToolApproved()},
            context={'call_1': {'state': 'pending', 'value': 1}},
        ),
    )
    assert result.output == 'Done!'


def test_context_on_call_deferred():
    """Context flows correctly through CallDeferred exceptions (non-nested)."""
    call_count = 0

    def llm(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return ModelResponse(
                parts=[ToolCallPart('slow_task', {'input': 'data'}, tool_call_id='call_1')]
            )
        else:
            return ModelResponse(parts=[TextPart('Done!')])

    agent = Agent(FunctionModel(llm), output_type=[str, DeferredToolRequests])

    @agent.tool
    def slow_task(ctx: RunContext[None], input: str) -> str:
        raise CallDeferred(
            metadata={'task_id': 'bg_1'},
            context={'progress': 0, 'queue_position': 5},
        )

    result = agent.run_sync('Process data')
    assert isinstance(result.output, DeferredToolRequests)
    assert result.output.context.get('call_1') == {'progress': 0, 'queue_position': 5}

    messages = result.all_messages()
    result = agent.run_sync(
        message_history=messages,
        deferred_tool_results=DeferredToolResults(
            calls={'call_1': 'result_value'},
        ),
    )
    assert result.output == 'Done!'
