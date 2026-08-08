"""Tests for the `HandleDeferredToolCalls` capability.

Split out of `test_capabilities.py`, which had grown past the repository's file-size limit.
"""

from __future__ import annotations

from typing import Any

import pytest

from pydantic_ai._run_context import RunContext
from pydantic_ai.agent import Agent
from pydantic_ai.capabilities import (
    HandleDeferredToolCalls,
    ToolSearch,
)
from pydantic_ai.capabilities.hooks import Hooks
from pydantic_ai.exceptions import (
    ApprovalRequired,
    CallDeferred,
    ModelRetry,
    ToolFailed,
    UserError,
)
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturn,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import DeferredToolRequests, DeferredToolResults, ToolApproved, ToolDenied
from pydantic_ai.toolsets import FunctionToolset
from pydantic_ai.usage import RequestUsage, RunUsage

from ._inline_snapshot import snapshot
from .conftest import IsDatetime, IsStr, iter_message_parts

_SEARCH_TOOLS_NAME = ToolSearch.function_tool_name

pytestmark = [
    pytest.mark.anyio,
]


def _build_run_context(deps: Any = None) -> RunContext[Any]:
    return RunContext(deps=deps, model=TestModel(), usage=RunUsage(), run_step=0)

# region HandleDeferredToolCalls


async def test_deferred_tool_handler_approve():
    """HandleDeferredToolCalls capability auto-approves a requires_approval tool inline."""

    def llm(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(parts=[ToolCallPart('my_tool', {'x': 5}, tool_call_id='call1')])
        return ModelResponse(parts=[TextPart('Done!')])

    async def handle_deferred(ctx: RunContext, requests: DeferredToolRequests) -> DeferredToolResults:
        return DeferredToolResults(approvals={call.tool_call_id: True for call in requests.approvals})

    agent = Agent(
        FunctionModel(llm),
        capabilities=[HandleDeferredToolCalls(handler=handle_deferred)],
    )

    @agent.tool
    def my_tool(ctx: RunContext, x: int) -> int:
        if not ctx.tool_call_approved:
            raise ApprovalRequired
        return x * 10

    result = await agent.run('Hello')
    assert result.output == 'Done!'
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[UserPromptPart(content='Hello', timestamp=IsDatetime())],
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[ToolCallPart(tool_name='my_tool', args={'x': 5}, tool_call_id='call1')],
                usage=RequestUsage(input_tokens=51, output_tokens=4),
                model_name='function:llm:',
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='my_tool',
                        content=50,
                        tool_call_id='call1',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='Done!')],
                usage=RequestUsage(input_tokens=52, output_tokens=5),
                model_name='function:llm:',
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
        ]
    )


async def test_deferred_tool_handler_deny():
    """HandleDeferredToolCalls capability denies a requires_approval tool inline, producing a `ToolReturnPart(outcome='denied')`."""

    def llm(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(parts=[ToolCallPart('my_tool', {'x': 5}, tool_call_id='call1')])
        return ModelResponse(parts=[TextPart('Understood, denied.')])

    async def handle_deferred(ctx: RunContext, requests: DeferredToolRequests) -> DeferredToolResults:
        return DeferredToolResults(
            approvals={call.tool_call_id: ToolDenied('Not allowed.') for call in requests.approvals}
        )

    agent = Agent(
        FunctionModel(llm),
        capabilities=[HandleDeferredToolCalls(handler=handle_deferred)],
    )

    @agent.tool
    def my_tool(ctx: RunContext, x: int) -> int:
        if not ctx.tool_call_approved:
            raise ApprovalRequired
        return x * 10  # pragma: no cover

    result = await agent.run('Hello')
    assert result.output == 'Understood, denied.'
    # The denial must surface in message history as outcome='denied', not a successful return.
    tool_returns = list(iter_message_parts(result.all_messages(), ModelRequest, ToolReturnPart))
    assert len(tool_returns) == 1
    assert tool_returns[0].tool_call_id == 'call1'
    assert tool_returns[0].outcome == 'denied'
    assert tool_returns[0].content == 'Not allowed.'


async def test_deferred_tool_handler_no_output_type_needed():
    """When handler resolves all deferred calls, DeferredToolRequests is not needed in output type."""

    def llm(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(parts=[ToolCallPart('my_tool', {'x': 3}, tool_call_id='call1')])
        return ModelResponse(parts=[TextPart('Result received.')])

    async def handle_deferred(ctx: RunContext, requests: DeferredToolRequests) -> DeferredToolResults:
        return DeferredToolResults(approvals={call.tool_call_id: True for call in requests.approvals})

    # Note: output_type is just str, no DeferredToolRequests
    agent = Agent(
        FunctionModel(llm),
        output_type=str,
        capabilities=[HandleDeferredToolCalls(handler=handle_deferred)],
    )

    @agent.tool
    def my_tool(ctx: RunContext, x: int) -> int:
        if not ctx.tool_call_approved:
            raise ApprovalRequired
        return x * 100

    result = await agent.run('Hello')
    assert result.output == 'Result received.'


async def test_deferred_tool_handler_none_fallback():
    """When no handler is present, deferred tools bubble up as DeferredToolRequests output."""

    agent = Agent(TestModel(), output_type=[str, DeferredToolRequests])

    @agent.tool_plain
    def my_tool(x: int) -> int:
        raise ApprovalRequired

    result = await agent.run('Hello')
    assert isinstance(result.output, DeferredToolRequests)
    assert len(result.output.approvals) == 1


async def test_deferred_tool_handler_partial_resolution():
    """Handler resolves some calls, remaining bubble up as DeferredToolRequests output."""

    def llm(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(
            parts=[
                ToolCallPart('tool_a', {}, tool_call_id='a1'),
                ToolCallPart('tool_b', {}, tool_call_id='b1'),
            ]
        )

    async def handle_deferred(ctx: RunContext, requests: DeferredToolRequests) -> DeferredToolResults:
        # Only approve tool_a, leave tool_b unresolved
        results = DeferredToolResults()
        for call in requests.approvals:
            if call.tool_name == 'tool_a':
                results.approvals[call.tool_call_id] = True
        return results

    agent = Agent(
        FunctionModel(llm),
        output_type=[str, DeferredToolRequests],
        capabilities=[HandleDeferredToolCalls(handler=handle_deferred)],
    )

    @agent.tool
    def tool_a(ctx: RunContext) -> str:
        if not ctx.tool_call_approved:
            raise ApprovalRequired
        return 'a done'

    @agent.tool
    def tool_b(ctx: RunContext) -> str:
        if not ctx.tool_call_approved:
            raise ApprovalRequired
        return 'b done'  # pragma: no cover

    result = await agent.run('Hello')
    assert isinstance(result.output, DeferredToolRequests)
    assert len(result.output.approvals) == 1
    assert result.output.approvals[0].tool_name == 'tool_b'


async def test_deferred_tool_handler_sync_handler():
    """HandleDeferredToolCalls works with a sync handler function."""

    def llm(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(parts=[ToolCallPart('my_tool', {}, tool_call_id='call1')])
        return ModelResponse(parts=[TextPart('OK')])

    def handle_deferred_sync(ctx: RunContext, requests: DeferredToolRequests) -> DeferredToolResults:
        return DeferredToolResults(approvals={call.tool_call_id: True for call in requests.approvals})

    agent = Agent(
        FunctionModel(llm),
        capabilities=[HandleDeferredToolCalls(handler=handle_deferred_sync)],
    )

    @agent.tool
    def my_tool(ctx: RunContext) -> str:
        if not ctx.tool_call_approved:
            raise ApprovalRequired
        return 'done'

    result = await agent.run('Hello')
    assert result.output == 'OK'


async def test_deferred_tool_handler_accumulation():
    """Two capabilities each resolve different deferred calls."""

    def llm(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(
                parts=[
                    ToolCallPart('tool_a', {}, tool_call_id='a1'),
                    ToolCallPart('tool_b', {}, tool_call_id='b1'),
                ]
            )
        return ModelResponse(parts=[TextPart('Both done.')])

    def handler_a(ctx: RunContext, requests: DeferredToolRequests) -> DeferredToolResults:
        results = DeferredToolResults()
        for call in requests.approvals:
            if call.tool_name == 'tool_a':
                results.approvals[call.tool_call_id] = True
        return results

    def handler_b(ctx: RunContext, requests: DeferredToolRequests) -> DeferredToolResults:
        # handler_a resolved tool_a, so we only see tool_b
        return DeferredToolResults(approvals={call.tool_call_id: True for call in requests.approvals})

    agent = Agent(
        FunctionModel(llm),
        capabilities=[
            HandleDeferredToolCalls(handler=handler_a),
            HandleDeferredToolCalls(handler=handler_b),
        ],
    )

    @agent.tool
    def tool_a(ctx: RunContext) -> str:
        if not ctx.tool_call_approved:
            raise ApprovalRequired
        return 'a result'

    @agent.tool
    def tool_b(ctx: RunContext) -> str:
        if not ctx.tool_call_approved:
            raise ApprovalRequired
        return 'b result'

    result = await agent.run('Hello')
    assert result.output == 'Both done.'


async def test_deferred_tool_handler_unresolved_no_output_type_error():
    """Unresolved deferred calls without DeferredToolRequests in output type raises UserError."""

    def llm(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[ToolCallPart('my_tool', {}, tool_call_id='call1')])

    # Handler returns None → does not resolve
    async def handle_deferred(ctx: RunContext, requests: DeferredToolRequests) -> DeferredToolResults:
        return DeferredToolResults()  # Empty results → nothing resolved

    agent = Agent(
        FunctionModel(llm),
        output_type=str,
        capabilities=[HandleDeferredToolCalls(handler=handle_deferred)],
    )

    @agent.tool
    def my_tool(ctx: RunContext) -> str:
        if not ctx.tool_call_approved:
            raise ApprovalRequired
        return 'done'  # pragma: no cover

    with pytest.raises(UserError, match='DeferredToolRequests'):
        await agent.run('Hello')


async def test_deferred_tool_handler_external_call():
    """HandleDeferredToolCalls capability resolves an externally-executed tool."""

    def llm(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(parts=[ToolCallPart('my_tool', {'x': 3}, tool_call_id='call1')])
        return ModelResponse(parts=[TextPart('Got it.')])

    from pydantic_ai.exceptions import CallDeferred

    async def handle_deferred(ctx: RunContext, requests: DeferredToolRequests) -> DeferredToolResults:
        # Simulate external execution: return a ToolReturn with metadata
        return DeferredToolResults(
            calls={
                call.tool_call_id: ToolReturn(return_value='external result', metadata={'source': 'ext'})
                for call in requests.calls
            }
        )

    agent = Agent(
        FunctionModel(llm),
        capabilities=[HandleDeferredToolCalls(handler=handle_deferred)],
    )

    @agent.tool_plain
    def my_tool(x: int) -> str:
        raise CallDeferred

    result = await agent.run('Hello')
    assert result.output == 'Got it.'


async def test_deferred_tool_handler_via_handle_call():
    """handle_call(resolve_deferred=True) resolves deferred tools inline via ToolManager."""

    async def handle_deferred(ctx: RunContext, requests: DeferredToolRequests) -> DeferredToolResults:
        return DeferredToolResults(approvals={call.tool_call_id: True for call in requests.approvals})

    def llm(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(parts=[ToolCallPart('outer_tool', {}, tool_call_id='outer1')])
        return ModelResponse(parts=[TextPart('All done.')])

    agent = Agent(
        FunctionModel(llm),
        capabilities=[HandleDeferredToolCalls(handler=handle_deferred)],
    )

    @agent.tool
    async def outer_tool(ctx: RunContext) -> str:
        """A tool that internally calls another tool via ToolManager.handle_call."""
        assert ctx.tool_manager is not None
        inner_call = ToolCallPart(tool_name='inner_tool', args={}, tool_call_id='inner1')
        result = await ctx.tool_manager.handle_call(inner_call)
        return f'inner returned: {result}'

    @agent.tool
    def inner_tool(ctx: RunContext) -> str:
        if not ctx.tool_call_approved:
            raise ApprovalRequired
        return 'approved inner result'

    result = await agent.run('Hello')
    assert result.output == 'All done.'


async def test_deferred_tool_handler_via_handle_call_wrap_validation_errors_false():
    """`wrap_validation_errors=False` propagates through deferred-tool resolution.

    Regression for the case where a sandboxed caller (`handle_call(wrap_validation_errors=False)`)
    invokes a tool that requires approval: after the handler approves, the re-execution must
    keep the raw-error contract — `ModelRetry` from the approved tool body should propagate
    as-is, not wrapped as `ToolRetryError`.
    """

    async def handle_deferred(ctx: RunContext, requests: DeferredToolRequests) -> DeferredToolResults:
        return DeferredToolResults(approvals={call.tool_call_id: True for call in requests.approvals})

    def llm(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(parts=[ToolCallPart('outer_tool', {}, tool_call_id='outer1')])
        return ModelResponse(parts=[TextPart('Done.')])

    agent = Agent(
        FunctionModel(llm),
        capabilities=[HandleDeferredToolCalls(handler=handle_deferred)],
    )

    @agent.tool
    async def outer_tool(ctx: RunContext) -> str:
        assert ctx.tool_manager is not None
        inner_call = ToolCallPart(tool_name='inner_tool', args={}, tool_call_id='inner1')
        try:
            await ctx.tool_manager.handle_call(inner_call, wrap_validation_errors=False)
        except ModelRetry as e:
            return f'raw ModelRetry: {e}'
        return 'no error'  # pragma: no cover

    @agent.tool
    def inner_tool(ctx: RunContext) -> str:
        if not ctx.tool_call_approved:
            raise ApprovalRequired
        raise ModelRetry('post-approval retry')

    result = await agent.run('Hello')
    assert result.output == 'Done.'
    # outer_tool caught the raw ModelRetry from the approved inner_tool body and surfaced it
    # in its return value; if wrap_validation_errors hadn't been forwarded through
    # _resolve_single_deferred, outer_tool would have seen a ToolRetryError instead.
    inner_message = next(
        msg
        for msg in result.all_messages()
        if isinstance(msg, ModelRequest)
        and any(isinstance(part, ToolReturnPart) and part.tool_name == 'outer_tool' for part in msg.parts)
    )
    outer_return = next(
        part for part in inner_message.parts if isinstance(part, ToolReturnPart) and part.tool_name == 'outer_tool'
    )
    assert outer_return.content == 'raw ModelRetry: post-approval retry'


async def test_deferred_tool_handler_via_handle_call_no_handler():
    """handle_call(resolve_deferred=True) re-raises when no handler is available."""

    # inner_tool is only available via ToolManager, not as a top-level agent tool
    inner_toolset = FunctionToolset()

    @inner_toolset.tool
    def inner_tool(ctx: RunContext) -> str:
        if not ctx.tool_call_approved:
            raise ApprovalRequired
        return 'approved inner result'  # pragma: no cover

    def llm(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(parts=[ToolCallPart('outer_tool', {}, tool_call_id='outer1')])
        return ModelResponse(parts=[TextPart('OK')])

    agent = Agent(FunctionModel(llm), toolsets=[inner_toolset])

    @agent.tool
    async def outer_tool(ctx: RunContext) -> str:
        """A tool that internally calls another tool via ToolManager.handle_call."""
        assert ctx.tool_manager is not None
        inner_call = ToolCallPart(tool_name='inner_tool', args={}, tool_call_id='inner1')
        try:
            result = await ctx.tool_manager.handle_call(inner_call)
            return f'inner returned: {result}'  # pragma: no cover
        except ApprovalRequired:
            return 'inner needs approval'

    result = await agent.run('Hello')
    assert result.output == 'OK'


async def test_deferred_tool_handler_build_results_helper():
    """DeferredToolRequests.build_results() creates a DeferredToolResults."""

    def llm(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(parts=[ToolCallPart('my_tool', {}, tool_call_id='call1')])
        return ModelResponse(parts=[TextPart('Done.')])

    async def handle_deferred(ctx: RunContext, requests: DeferredToolRequests) -> DeferredToolResults:
        return requests.build_results(approvals={call.tool_call_id: True for call in requests.approvals})

    agent = Agent(
        FunctionModel(llm),
        capabilities=[HandleDeferredToolCalls(handler=handle_deferred)],
    )

    @agent.tool
    def my_tool(ctx: RunContext) -> str:
        if not ctx.tool_call_approved:
            raise ApprovalRequired
        return 'done'

    result = await agent.run('Hello')
    assert result.output == 'Done.'


def test_deferred_tool_requests_build_results_validates_ids():
    """build_results rejects result IDs that don't match a pending request of the right kind."""
    requests = DeferredToolRequests(
        approvals=[ToolCallPart('a', {}, tool_call_id='approval_1')],
        calls=[ToolCallPart('b', {}, tool_call_id='call_1')],
    )

    # Mis-routed ID: tool-result provided for something in the approvals list.
    with pytest.raises(ValueError, match=r'calls.*not in.*DeferredToolRequests.calls'):
        requests.build_results(calls={'approval_1': 'oops'})

    # Unknown ID entirely.
    with pytest.raises(ValueError, match=r'approvals.*not in.*DeferredToolRequests.approvals'):
        requests.build_results(approvals={'unknown_id': True})

    # Happy path still works.
    results = requests.build_results(approvals={'approval_1': True}, calls={'call_1': 'result'})
    assert results.approvals == {'approval_1': True}
    assert results.calls == {'call_1': 'result'}


def test_deferred_tool_requests_build_results_approve_all():
    """approve_all=True approves every pending approval not explicitly specified."""
    requests = DeferredToolRequests(
        approvals=[
            ToolCallPart('a', {}, tool_call_id='approval_1'),
            ToolCallPart('b', {}, tool_call_id='approval_2'),
            ToolCallPart('c', {}, tool_call_id='approval_3'),
        ],
    )

    # Explicit deny wins; the other two get auto-approved.
    results = requests.build_results(
        approvals={'approval_1': False},
        approve_all=True,
    )
    assert results.approvals['approval_1'] is False
    assert isinstance(results.approvals['approval_2'], ToolApproved)
    assert isinstance(results.approvals['approval_3'], ToolApproved)


async def test_deferred_tool_handler_wrapper_capability():
    """HandleDeferredToolCalls works through WrapperCapability delegation."""

    def llm(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(parts=[ToolCallPart('my_tool', {}, tool_call_id='call1')])
        return ModelResponse(parts=[TextPart('Done.')])

    async def handle_deferred(ctx: RunContext, requests: DeferredToolRequests) -> DeferredToolResults:
        return DeferredToolResults(approvals={call.tool_call_id: True for call in requests.approvals})

    # PrefixTools wraps HandleDeferredToolCalls — tests WrapperCapability delegation
    inner = HandleDeferredToolCalls(handler=handle_deferred)
    agent = Agent(
        FunctionModel(llm),
        capabilities=[inner.prefix_tools('ns')],
    )

    @agent.tool
    def my_tool(ctx: RunContext) -> str:
        if not ctx.tool_call_approved:
            raise ApprovalRequired
        return 'done'

    result = await agent.run('Hello')
    assert result.output == 'Done.'


async def test_deferred_tool_handler_external_call_plain_value():
    """HandleDeferredToolCalls resolves an external call with a plain value (not ToolReturn)."""

    def llm(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(parts=[ToolCallPart('my_tool', {}, tool_call_id='call1')])
        return ModelResponse(parts=[TextPart('Got it.')])

    from pydantic_ai.exceptions import CallDeferred

    async def handle_deferred(ctx: RunContext, requests: DeferredToolRequests) -> DeferredToolResults:
        return DeferredToolResults(calls={call.tool_call_id: 'plain string result' for call in requests.calls})

    agent = Agent(FunctionModel(llm), capabilities=[HandleDeferredToolCalls(handler=handle_deferred)])

    @agent.tool_plain
    def my_tool() -> str:
        raise CallDeferred

    result = await agent.run('Hello')
    assert result.output == 'Got it.'


async def test_deferred_tool_handler_re_deferred_with_metadata():
    """When an approved tool re-raises ApprovalRequired, it stays unresolved with metadata."""

    call_count = 0

    def llm(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[ToolCallPart('my_tool', {}, tool_call_id='call1')])

    async def handle_deferred(ctx: RunContext, requests: DeferredToolRequests) -> DeferredToolResults:
        return DeferredToolResults(approvals={call.tool_call_id: True for call in requests.approvals})

    agent = Agent(
        FunctionModel(llm),
        output_type=[str, DeferredToolRequests],
        capabilities=[HandleDeferredToolCalls(handler=handle_deferred)],
    )

    @agent.tool
    def my_tool(ctx: RunContext) -> str:
        nonlocal call_count
        call_count += 1
        # Always requires approval — even when approved, raises again with metadata
        raise ApprovalRequired(metadata={'attempt': call_count})

    result = await agent.run('Hello')
    # Tool re-raised after approval → goes to remaining → becomes output
    assert isinstance(result.output, DeferredToolRequests)
    assert len(result.output.approvals) == 1
    assert result.output.metadata.get('call1') == {'attempt': 2}


async def test_deferred_tool_handler_denied_via_batch():
    """Batch path deny via handler produces a `ToolReturnPart(outcome='denied')` in message history."""

    def llm(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(parts=[ToolCallPart('my_tool', {}, tool_call_id='call1')])
        return ModelResponse(parts=[TextPart('Understood.')])

    async def handle_deferred(ctx: RunContext, requests: DeferredToolRequests) -> DeferredToolResults:
        return DeferredToolResults(
            approvals={call.tool_call_id: ToolDenied('Policy denied.') for call in requests.approvals}
        )

    agent = Agent(FunctionModel(llm), capabilities=[HandleDeferredToolCalls(handler=handle_deferred)])

    @agent.tool
    def my_tool(ctx: RunContext) -> str:
        if not ctx.tool_call_approved:
            raise ApprovalRequired
        return 'done'  # pragma: no cover

    result = await agent.run('Hello')
    assert result.output == 'Understood.'
    tool_returns = list(iter_message_parts(result.all_messages(), ModelRequest, ToolReturnPart))
    assert len(tool_returns) == 1
    assert tool_returns[0].outcome == 'denied'
    assert tool_returns[0].content == 'Policy denied.'


async def test_deferred_tool_handler_batch_deny_via_bool_and_default():
    """Batch path: covers `approvals[id] = False` AND default `ToolDenied()` as separate calls."""

    def llm(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(
                parts=[
                    ToolCallPart('needs_approval', {'x': 1}, tool_call_id='bool_false'),
                    ToolCallPart('needs_approval', {'x': 2}, tool_call_id='default_denied'),
                ]
            )
        return ModelResponse(parts=[TextPart('ok')])

    async def handle_deferred(ctx: RunContext, requests: DeferredToolRequests) -> DeferredToolResults:
        return DeferredToolResults(
            approvals={
                'bool_false': False,
                'default_denied': ToolDenied(),  # no custom message
            }
        )

    agent = Agent(FunctionModel(llm), capabilities=[HandleDeferredToolCalls(handler=handle_deferred)])

    @agent.tool
    def needs_approval(ctx: RunContext, x: int) -> int:
        if not ctx.tool_call_approved:
            raise ApprovalRequired
        return x  # pragma: no cover

    result = await agent.run('go')
    assert result.output == 'ok'
    tool_returns = {p.tool_call_id: p for p in iter_message_parts(result.all_messages(), ModelRequest, ToolReturnPart)}
    assert tool_returns['bool_false'].outcome == 'denied'
    assert tool_returns['bool_false'].content == ToolDenied().message
    assert tool_returns['default_denied'].outcome == 'denied'
    assert tool_returns['default_denied'].content == ToolDenied().message


async def test_deferred_tool_handler_batch_approve_via_tool_approved_default():
    """Batch path: covers `approvals[id] = ToolApproved()` (default, no override_args)."""
    from pydantic_ai.tools import ToolApproved

    def llm(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(parts=[ToolCallPart('needs_approval', {'x': 7}, tool_call_id='c1')])
        return ModelResponse(parts=[TextPart('done')])

    async def handle_deferred(ctx: RunContext, requests: DeferredToolRequests) -> DeferredToolResults:
        return DeferredToolResults(approvals={call.tool_call_id: ToolApproved() for call in requests.approvals})

    agent = Agent(FunctionModel(llm), capabilities=[HandleDeferredToolCalls(handler=handle_deferred)])

    @agent.tool
    def needs_approval(ctx: RunContext, x: int) -> int:
        if not ctx.tool_call_approved:
            raise ApprovalRequired
        return x * 2

    result = await agent.run('go')
    assert result.output == 'done'
    tool_returns = list(iter_message_parts(result.all_messages(), ModelRequest, ToolReturnPart))
    assert len(tool_returns) == 1
    assert tool_returns[0].outcome != 'denied'
    assert tool_returns[0].content == 14


async def test_deferred_tool_handler_batch_external_tool_return_metadata():
    """Batch path: handler-supplied external `ToolReturn(value, metadata)` lands on the return part."""
    from pydantic_ai.messages import ToolReturn as _ToolReturn

    def llm(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(parts=[ToolCallPart('external_tool', {}, tool_call_id='c1')])
        return ModelResponse(parts=[TextPart('done')])

    async def handle_deferred(ctx: RunContext, requests: DeferredToolRequests) -> DeferredToolResults:
        return DeferredToolResults(
            calls={
                call.tool_call_id: _ToolReturn(
                    return_value='computed', metadata={'source': 'external'}, content='user extra'
                )
                for call in requests.calls
            }
        )

    agent = Agent(FunctionModel(llm), capabilities=[HandleDeferredToolCalls(handler=handle_deferred)])

    @agent.tool
    def external_tool(ctx: RunContext) -> str:
        raise CallDeferred

    result = await agent.run('go')
    assert result.output == 'done'
    messages = result.all_messages()
    tool_returns = list(iter_message_parts(messages, ModelRequest, ToolReturnPart))
    assert len(tool_returns) == 1
    assert tool_returns[0].content == 'computed'
    assert tool_returns[0].metadata == {'source': 'external'}
    # The `content` field on ToolReturn becomes a UserPromptPart.
    from pydantic_ai.messages import UserPromptPart

    user_extras = [p for p in iter_message_parts(messages, ModelRequest, UserPromptPart) if p.content == 'user extra']
    assert len(user_extras) == 1


async def test_deferred_tool_handler_batch_external_model_retry():
    """Batch path: handler-supplied `ModelRetry` in `calls` surfaces as a `RetryPromptPart`, not a tool return."""
    call_count = 0

    def llm(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return ModelResponse(parts=[ToolCallPart('external_tool', {}, tool_call_id='c1')])
        return ModelResponse(parts=[TextPart('retried')])

    async def handle_deferred(ctx: RunContext, requests: DeferredToolRequests) -> DeferredToolResults:
        return DeferredToolResults(calls={call.tool_call_id: ModelRetry('try again') for call in requests.calls})

    agent = Agent(FunctionModel(llm), capabilities=[HandleDeferredToolCalls(handler=handle_deferred)])

    @agent.tool
    def external_tool(ctx: RunContext) -> str:
        raise CallDeferred

    result = await agent.run('go')
    assert result.output == 'retried'
    messages = result.all_messages()
    retry_parts = list(iter_message_parts(messages, ModelRequest, RetryPromptPart))
    assert len(retry_parts) == 1
    assert retry_parts[0].tool_call_id == 'c1'
    assert retry_parts[0].content == 'try again'
    tool_returns = [p for p in iter_message_parts(messages, ModelRequest, ToolReturnPart) if p.tool_call_id == 'c1']
    assert tool_returns == []


async def test_deferred_tool_handler_batch_external_retry_prompt_part():
    """Batch path: handler-supplied `RetryPromptPart` in `calls` surfaces as a retry (names stamped from the deferred call)."""
    call_count = 0

    def llm(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return ModelResponse(parts=[ToolCallPart('external_tool', {}, tool_call_id='c1')])
        return ModelResponse(parts=[TextPart('retried')])

    async def handle_deferred(ctx: RunContext, requests: DeferredToolRequests) -> DeferredToolResults:
        return DeferredToolResults(
            calls={
                call.tool_call_id: RetryPromptPart(content='retry via part', tool_name='', tool_call_id='')
                for call in requests.calls
            }
        )

    agent = Agent(FunctionModel(llm), capabilities=[HandleDeferredToolCalls(handler=handle_deferred)])

    @agent.tool
    def external_tool(ctx: RunContext) -> str:
        raise CallDeferred

    result = await agent.run('go')
    assert result.output == 'retried'
    retry_parts = list(iter_message_parts(result.all_messages(), ModelRequest, RetryPromptPart))
    assert len(retry_parts) == 1
    assert retry_parts[0].tool_call_id == 'c1'
    assert retry_parts[0].tool_name == 'external_tool'
    assert retry_parts[0].content == 'retry via part'


async def test_deferred_tool_handler_via_handle_call_external_tool_return():
    """Per-call path: handler-supplied external `ToolReturn(value, metadata)` is returned verbatim from handle_call."""
    from pydantic_ai.exceptions import CallDeferred
    from pydantic_ai.messages import ToolReturn as _ToolReturn

    inner_toolset = FunctionToolset()

    @inner_toolset.tool_plain
    def inner_tool() -> str:
        raise CallDeferred

    async def handle_deferred(ctx: RunContext, requests: DeferredToolRequests) -> DeferredToolResults:
        return DeferredToolResults(
            calls={call.tool_call_id: _ToolReturn(return_value='ext', metadata={'k': 'v'}) for call in requests.calls}
        )

    def llm(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(parts=[ToolCallPart('caller_tool', {}, tool_call_id='c1')])
        return ModelResponse(parts=[TextPart('final')])

    agent = Agent(
        FunctionModel(llm),
        toolsets=[inner_toolset],
        capabilities=[HandleDeferredToolCalls(handler=handle_deferred)],
    )

    captured_result: Any = None

    @agent.tool
    async def caller_tool(ctx: RunContext) -> str:
        nonlocal captured_result
        assert ctx.tool_manager is not None
        captured_result = await ctx.tool_manager.handle_call(
            ToolCallPart(tool_name='inner_tool', args={}, tool_call_id='inner_1'),
        )
        return 'done'

    await agent.run('go')
    # Per-call path returns whatever the handler supplied verbatim — full ToolReturn wrapper preserved.
    assert isinstance(captured_result, _ToolReturn)
    assert captured_result.return_value == 'ext'
    assert captured_result.metadata == {'k': 'v'}


async def test_deferred_tool_handler_via_handle_call_tool_failed():
    """Per-call path: handler-supplied `ToolFailed` raises `ToolFailedError`, matching a tool that raises `ToolFailed` in-process."""
    from pydantic_ai.exceptions import CallDeferred, ToolFailedError
    from pydantic_ai.toolsets import FunctionToolset

    inner_toolset = FunctionToolset()

    @inner_toolset.tool_plain
    def inner_tool() -> str:
        raise CallDeferred

    async def handle_deferred(ctx: RunContext[object], requests: DeferredToolRequests) -> DeferredToolResults:
        return DeferredToolResults(
            calls={call.tool_call_id: ToolFailed('backend unavailable') for call in requests.calls}
        )

    def llm(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(parts=[ToolCallPart('caller_tool', {}, tool_call_id='c1')])
        return ModelResponse(parts=[TextPart('final')])

    agent = Agent(
        FunctionModel(llm),
        toolsets=[inner_toolset],
        capabilities=[HandleDeferredToolCalls(handler=handle_deferred)],
    )

    captured_error: Any = None

    @agent.tool
    async def caller_tool(ctx: RunContext[object]) -> str:
        nonlocal captured_error
        assert ctx.tool_manager is not None
        try:
            await ctx.tool_manager.handle_call(
                ToolCallPart(tool_name='inner_tool', args={}, tool_call_id='inner_1'),
            )
        except ToolFailedError as e:
            captured_error = e
        return 'done'

    await agent.run('go')
    assert captured_error is not None
    assert captured_error.tool_failed.tool_name == 'inner_tool'
    assert captured_error.tool_failed.tool_call_id == 'inner_1'
    assert captured_error.tool_failed.content == 'backend unavailable'
    assert captured_error.tool_failed.outcome == 'failed'


def test_deferred_tool_handler_serialization_name():
    """HandleDeferredToolCalls is not spec-constructible."""
    assert HandleDeferredToolCalls.get_serialization_name() is None


async def test_deferred_tool_handler_via_handle_call_with_resolve():
    """handle_call(resolve_deferred=True) goes through _resolve_single_deferred happy path.

    This exercises the per-call resolution path used by CodeMode-style callers.
    """

    inner_toolset = FunctionToolset()

    @inner_toolset.tool
    def inner_tool(ctx: RunContext) -> str:
        if not ctx.tool_call_approved:
            raise ApprovalRequired
        return 'approved result'

    async def handle_deferred(ctx: RunContext, requests: DeferredToolRequests) -> DeferredToolResults:
        return DeferredToolResults(approvals={call.tool_call_id: True for call in requests.approvals})

    def llm(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(parts=[ToolCallPart('caller_tool', {}, tool_call_id='c1')])
        return ModelResponse(parts=[TextPart('final')])

    agent = Agent(
        FunctionModel(llm),
        toolsets=[inner_toolset],
        capabilities=[HandleDeferredToolCalls(handler=handle_deferred)],
    )

    @agent.tool
    async def caller_tool(ctx: RunContext) -> str:
        assert ctx.tool_manager is not None
        # Call inner_tool via handle_call — exercises _resolve_single_deferred
        result = await ctx.tool_manager.handle_call(
            ToolCallPart(tool_name='inner_tool', args={}, tool_call_id='inner_1'),
        )
        # _resolve_single_deferred returns result_part.content
        assert result == 'approved result'
        return f'got: {result}'

    result = await agent.run('go')
    assert result.output == 'final'
    # Verify the inner tool was called (tool return visible in messages)
    tool_returns = [
        p
        for p in iter_message_parts(result.all_messages(), ModelRequest, ToolReturnPart)
        if p.tool_name == 'caller_tool'
    ]
    assert len(tool_returns) == 1
    assert tool_returns[0].content == 'got: approved result'


async def test_deferred_tool_handler_approved_tool_returns_tool_return():
    """Approved tool returning a ToolReturn preserves metadata and user content."""
    from pydantic_ai.messages import ToolReturn as _ToolReturn, UserPromptPart

    def llm(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(parts=[ToolCallPart('my_tool', {}, tool_call_id='call1')])
        return ModelResponse(parts=[TextPart('Done.')])

    async def handle_deferred(ctx: RunContext, requests: DeferredToolRequests) -> DeferredToolResults:
        return DeferredToolResults(approvals={call.tool_call_id: True for call in requests.approvals})

    agent = Agent(FunctionModel(llm), capabilities=[HandleDeferredToolCalls(handler=handle_deferred)])

    @agent.tool
    def my_tool(ctx: RunContext):
        if not ctx.tool_call_approved:
            raise ApprovalRequired
        return _ToolReturn(return_value='result', metadata={'source': 'tool'}, content='user prompt extra')

    result = await agent.run('Hello')
    assert result.output == 'Done.'
    # Verify ToolReturn.metadata preserved
    tool_returns = [
        p for p in iter_message_parts(result.all_messages(), ModelRequest, ToolReturnPart) if p.tool_name == 'my_tool'
    ]
    assert len(tool_returns) == 1
    assert tool_returns[0].metadata == {'source': 'tool'}
    # Verify ToolReturn.content appears as UserPromptPart
    user_parts = [
        p
        for p in iter_message_parts(result.all_messages(), ModelRequest, UserPromptPart)
        if p.content == 'user prompt extra'
    ]
    assert len(user_parts) == 1


async def test_deferred_tool_handler_approved_tool_raises_model_retry():
    """Approved tool that raises ModelRetry produces a RetryPromptPart."""

    def llm(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(parts=[ToolCallPart('my_tool', {}, tool_call_id='call1')])
        return ModelResponse(parts=[TextPart('Retried and done.')])

    async def handle_deferred(ctx: RunContext, requests: DeferredToolRequests) -> DeferredToolResults:
        return DeferredToolResults(approvals={call.tool_call_id: True for call in requests.approvals})

    agent = Agent(FunctionModel(llm), capabilities=[HandleDeferredToolCalls(handler=handle_deferred)])

    @agent.tool
    def my_tool(ctx: RunContext) -> str:
        if not ctx.tool_call_approved:
            raise ApprovalRequired
        raise ModelRetry('try again')

    result = await agent.run('Hello')
    assert result.output == 'Retried and done.'
    # Verify the retry happened
    retry_parts = [
        p for p in iter_message_parts(result.all_messages(), ModelRequest, RetryPromptPart) if p.tool_name == 'my_tool'
    ]
    assert len(retry_parts) == 1


async def test_deferred_tool_handler_approved_tool_override_args():
    """Approved tool with ToolApproved(override_args=...) uses the override."""
    from pydantic_ai.tools import ToolApproved

    received_x = None

    def llm(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(parts=[ToolCallPart('my_tool', {'x': 5}, tool_call_id='call1')])
        return ModelResponse(parts=[TextPart('Done.')])

    async def handle_deferred(ctx: RunContext, requests: DeferredToolRequests) -> DeferredToolResults:
        # Override the args: replace x=5 with x=42
        return DeferredToolResults(
            approvals={call.tool_call_id: ToolApproved(override_args={'x': 42}) for call in requests.approvals}
        )

    agent = Agent(FunctionModel(llm), capabilities=[HandleDeferredToolCalls(handler=handle_deferred)])

    @agent.tool
    def my_tool(ctx: RunContext, x: int) -> int:
        nonlocal received_x
        if not ctx.tool_call_approved:
            raise ApprovalRequired
        received_x = x
        return x * 10

    result = await agent.run('Hello')
    assert result.output == 'Done.'
    assert received_x == 42  # Override was applied


async def test_deferred_tool_handler_via_handle_call_retry():
    """handle_call path: approved tool raising ModelRetry propagates ToolRetryError."""
    from pydantic_ai.exceptions import ToolRetryError

    inner_toolset = FunctionToolset()
    retry_count = 0

    @inner_toolset.tool
    def inner_tool(ctx: RunContext) -> str:
        nonlocal retry_count
        if not ctx.tool_call_approved:
            raise ApprovalRequired
        retry_count += 1
        raise ModelRetry('try again')

    async def handle_deferred(ctx: RunContext, requests: DeferredToolRequests) -> DeferredToolResults:
        return DeferredToolResults(approvals={call.tool_call_id: True for call in requests.approvals})

    def llm(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(parts=[ToolCallPart('caller_tool', {}, tool_call_id='c1')])
        return ModelResponse(parts=[TextPart('final')])

    agent = Agent(
        FunctionModel(llm),
        toolsets=[inner_toolset],
        capabilities=[HandleDeferredToolCalls(handler=handle_deferred)],
    )

    @agent.tool
    async def caller_tool(ctx: RunContext) -> str:
        assert ctx.tool_manager is not None
        try:
            await ctx.tool_manager.handle_call(
                ToolCallPart(tool_name='inner_tool', args={}, tool_call_id='inner_1'),
            )
            return 'no retry'  # pragma: no cover
        except ToolRetryError:
            return 'got retry'

    result = await agent.run('go')
    assert result.output == 'final'
    assert retry_count == 1


async def test_deferred_tool_handler_re_deferred_without_metadata():
    """Approved tool that re-raises without metadata — no metadata added to remaining."""
    call_count = 0

    def llm(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[ToolCallPart('my_tool', {}, tool_call_id='call1')])

    async def handle_deferred(ctx: RunContext, requests: DeferredToolRequests) -> DeferredToolResults:
        return DeferredToolResults(approvals={call.tool_call_id: True for call in requests.approvals})

    agent = Agent(
        FunctionModel(llm),
        output_type=[str, DeferredToolRequests],
        capabilities=[HandleDeferredToolCalls(handler=handle_deferred)],
    )

    @agent.tool
    def my_tool(ctx: RunContext) -> str:
        nonlocal call_count
        call_count += 1
        # No metadata
        raise ApprovalRequired

    result = await agent.run('Hello')
    assert isinstance(result.output, DeferredToolRequests)
    assert len(result.output.approvals) == 1
    # No metadata set (tool raised without metadata both times)
    assert 'call1' not in result.output.metadata


async def test_deferred_tool_handler_mixed_unresolved_and_re_deferred():
    """Handler resolves some, another call is re-deferred — both end up in remaining."""

    def llm(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(
            parts=[
                ToolCallPart('re_raising_tool', {}, tool_call_id='re1'),
                ToolCallPart('unhandled_tool', {}, tool_call_id='un1'),
            ]
        )

    async def handle_deferred(ctx: RunContext, requests: DeferredToolRequests) -> DeferredToolResults:
        # Only approve the re-raising one; leave unhandled_tool unresolved
        return DeferredToolResults(
            approvals={call.tool_call_id: True for call in requests.approvals if call.tool_name == 're_raising_tool'}
        )

    agent = Agent(
        FunctionModel(llm),
        output_type=[str, DeferredToolRequests],
        capabilities=[HandleDeferredToolCalls(handler=handle_deferred)],
    )

    @agent.tool
    def re_raising_tool(ctx: RunContext) -> str:
        # Always raises — even after approval
        raise ApprovalRequired

    @agent.tool
    def unhandled_tool(ctx: RunContext) -> str:
        if not ctx.tool_call_approved:
            raise ApprovalRequired
        return 'done'  # pragma: no cover

    result = await agent.run('Hello')
    assert isinstance(result.output, DeferredToolRequests)
    # Both calls in remaining: unhandled_tool (never resolved) + re_raising_tool (re-deferred after approval)
    approval_ids = {call.tool_call_id for call in result.output.approvals}
    assert 're1' in approval_ids
    assert 'un1' in approval_ids


async def test_deferred_tool_handler_re_deferred_as_call_deferred():
    """Approved tool that re-raises CallDeferred (not ApprovalRequired) stays in remaining.calls."""
    from pydantic_ai.exceptions import CallDeferred

    call_count = 0

    def llm(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[ToolCallPart('my_tool', {}, tool_call_id='call1')])

    async def handle_deferred(ctx: RunContext, requests: DeferredToolRequests) -> DeferredToolResults:
        return DeferredToolResults(approvals={call.tool_call_id: True for call in requests.approvals})

    agent = Agent(
        FunctionModel(llm),
        output_type=[str, DeferredToolRequests],
        capabilities=[HandleDeferredToolCalls(handler=handle_deferred)],
    )

    @agent.tool
    def my_tool(ctx: RunContext) -> str:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise ApprovalRequired
        # After approval, raise CallDeferred (external execution needed)
        raise CallDeferred(metadata={'reason': 'external'})

    result = await agent.run('Hello')
    assert isinstance(result.output, DeferredToolRequests)
    # Should be in calls (external), not approvals
    assert len(result.output.calls) == 1
    assert len(result.output.approvals) == 0
    assert result.output.metadata == {'call1': {'reason': 'external'}}


async def test_deferred_tool_handler_via_handle_call_preserves_tool_return():
    """handle_call(resolve_deferred=True) preserves `ToolReturn` wrapper (metadata, user content).

    The non-deferred `handle_call` path returns whatever the tool returned verbatim.
    The deferred path should do the same — critical for CodeMode-style callers that
    check `isinstance(result, ToolReturn)` to preserve metadata on nested return parts.
    """
    from pydantic_ai.messages import ToolReturn as _ToolReturn

    inner_toolset = FunctionToolset()

    @inner_toolset.tool
    def inner_tool(ctx: RunContext):
        if not ctx.tool_call_approved:
            raise ApprovalRequired
        return _ToolReturn(return_value='actual result', metadata={'source': 'inner'}, content='user extra')

    async def handle_deferred(ctx: RunContext, requests: DeferredToolRequests) -> DeferredToolResults:
        return DeferredToolResults(approvals={call.tool_call_id: True for call in requests.approvals})

    def llm(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(parts=[ToolCallPart('caller_tool', {}, tool_call_id='c1')])
        return ModelResponse(parts=[TextPart('final')])

    agent = Agent(
        FunctionModel(llm),
        toolsets=[inner_toolset],
        capabilities=[HandleDeferredToolCalls(handler=handle_deferred)],
    )

    captured_result: Any = None

    @agent.tool
    async def caller_tool(ctx: RunContext) -> str:
        nonlocal captured_result
        assert ctx.tool_manager is not None
        result = await ctx.tool_manager.handle_call(
            ToolCallPart(tool_name='inner_tool', args={}, tool_call_id='inner_1'),
        )
        captured_result = result
        return 'done'

    await agent.run('go')
    # handle_call returned the ToolReturn wrapper verbatim, not the unwrapped content
    assert isinstance(captured_result, _ToolReturn)
    assert captured_result.return_value == 'actual result'
    assert captured_result.metadata == {'source': 'inner'}
    assert captured_result.content == 'user extra'


async def test_deferred_tool_handler_via_handle_call_denied_via_bool():
    """When a handler denies via `approvals[id] = False`, handle_call returns `ToolDenied()` with the default denial message."""

    inner_toolset = FunctionToolset()

    @inner_toolset.tool
    def inner_tool(ctx: RunContext) -> str:
        if not ctx.tool_call_approved:
            raise ApprovalRequired
        return 'never'  # pragma: no cover

    async def handle_deferred(ctx: RunContext, requests: DeferredToolRequests) -> DeferredToolResults:
        return DeferredToolResults(approvals={call.tool_call_id: False for call in requests.approvals})

    def llm(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(parts=[ToolCallPart('caller_tool', {}, tool_call_id='c1')])
        return ModelResponse(parts=[TextPart('final')])

    agent = Agent(
        FunctionModel(llm),
        toolsets=[inner_toolset],
        capabilities=[HandleDeferredToolCalls(handler=handle_deferred)],
    )

    captured: Any = None

    @agent.tool
    async def caller_tool(ctx: RunContext) -> str:
        nonlocal captured
        assert ctx.tool_manager is not None
        captured = await ctx.tool_manager.handle_call(
            ToolCallPart(tool_name='inner_tool', args={}, tool_call_id='inner_1'),
        )
        return 'caught' if isinstance(captured, ToolDenied) else 'no denial'

    await agent.run('go')
    assert isinstance(captured, ToolDenied)
    assert captured == ToolDenied()


async def test_deferred_tool_handler_via_handle_call_override_args():
    """When a handler approves with override_args, handle_call executes the tool with those args."""
    from pydantic_ai.tools import ToolApproved

    inner_toolset = FunctionToolset()

    @inner_toolset.tool
    def inner_tool(ctx: RunContext, x: int) -> str:
        if not ctx.tool_call_approved:
            raise ApprovalRequired
        return f'x={x}'

    async def handle_deferred(ctx: RunContext, requests: DeferredToolRequests) -> DeferredToolResults:
        return DeferredToolResults(
            approvals={call.tool_call_id: ToolApproved(override_args={'x': 42}) for call in requests.approvals}
        )

    def llm(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(parts=[ToolCallPart('caller_tool', {}, tool_call_id='c1')])
        return ModelResponse(parts=[TextPart('final')])

    agent = Agent(
        FunctionModel(llm),
        toolsets=[inner_toolset],
        capabilities=[HandleDeferredToolCalls(handler=handle_deferred)],
    )

    captured_result: Any = None

    @agent.tool
    async def caller_tool(ctx: RunContext) -> str:
        nonlocal captured_result
        assert ctx.tool_manager is not None
        captured_result = await ctx.tool_manager.handle_call(
            ToolCallPart(tool_name='inner_tool', args={'x': 1}, tool_call_id='inner_1'),
        )
        return 'done'

    await agent.run('go')
    assert captured_result == 'x=42'


async def test_deferred_tool_handler_via_handle_call_external_plain_value():
    """When a handler supplies an external-call plain value, handle_call returns it verbatim."""
    from pydantic_ai.exceptions import CallDeferred

    inner_toolset = FunctionToolset()

    @inner_toolset.tool_plain
    def inner_tool() -> str:
        raise CallDeferred

    async def handle_deferred(ctx: RunContext, requests: DeferredToolRequests) -> DeferredToolResults:
        return DeferredToolResults(calls={call.tool_call_id: 'external value' for call in requests.calls})

    def llm(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(parts=[ToolCallPart('caller_tool', {}, tool_call_id='c1')])
        return ModelResponse(parts=[TextPart('final')])

    agent = Agent(
        FunctionModel(llm),
        toolsets=[inner_toolset],
        capabilities=[HandleDeferredToolCalls(handler=handle_deferred)],
    )

    captured_result: Any = None

    @agent.tool
    async def caller_tool(ctx: RunContext) -> str:
        nonlocal captured_result
        assert ctx.tool_manager is not None
        captured_result = await ctx.tool_manager.handle_call(
            ToolCallPart(tool_name='inner_tool', args={}, tool_call_id='inner_1'),
        )
        return 'done'

    await agent.run('go')
    assert captured_result == 'external value'


async def test_deferred_tool_handler_via_handle_call_external_model_retry():
    """When a handler supplies a `ModelRetry` external-call result, handle_call raises `ToolRetryError`."""
    from pydantic_ai.exceptions import CallDeferred, ModelRetry, ToolRetryError

    inner_toolset = FunctionToolset()

    @inner_toolset.tool_plain
    def inner_tool() -> str:
        raise CallDeferred

    async def handle_deferred(ctx: RunContext, requests: DeferredToolRequests) -> DeferredToolResults:
        return DeferredToolResults(calls={call.tool_call_id: ModelRetry('retry please') for call in requests.calls})

    def llm(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(parts=[ToolCallPart('caller_tool', {}, tool_call_id='c1')])
        return ModelResponse(parts=[TextPart('final')])

    agent = Agent(
        FunctionModel(llm),
        toolsets=[inner_toolset],
        capabilities=[HandleDeferredToolCalls(handler=handle_deferred)],
    )

    caught: ToolRetryError | None = None

    @agent.tool
    async def caller_tool(ctx: RunContext) -> str:
        nonlocal caught
        assert ctx.tool_manager is not None
        try:
            await ctx.tool_manager.handle_call(
                ToolCallPart(tool_name='inner_tool', args={}, tool_call_id='inner_1'),
            )
            return 'no raise'  # pragma: no cover
        except ToolRetryError as e:
            caught = e
            return 'caught'

    await agent.run('go')
    assert caught is not None
    assert caught.tool_retry.content == 'retry please'
    assert caught.tool_retry.tool_name == 'inner_tool'
    assert caught.tool_retry.tool_call_id == 'inner_1'


async def test_deferred_tool_handler_via_handle_call_external_retry_prompt_part():
    """When a handler supplies a `RetryPromptPart` external-call result, handle_call raises `ToolRetryError` with the part."""
    from pydantic_ai.exceptions import CallDeferred, ToolRetryError

    inner_toolset = FunctionToolset()

    @inner_toolset.tool_plain
    def inner_tool() -> str:
        raise CallDeferred

    async def handle_deferred(ctx: RunContext, requests: DeferredToolRequests) -> DeferredToolResults:
        return DeferredToolResults(
            calls={
                call.tool_call_id: RetryPromptPart(content='retry via part', tool_name='', tool_call_id='')
                for call in requests.calls
            }
        )

    def llm(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(parts=[ToolCallPart('caller_tool', {}, tool_call_id='c1')])
        return ModelResponse(parts=[TextPart('final')])

    agent = Agent(
        FunctionModel(llm),
        toolsets=[inner_toolset],
        capabilities=[HandleDeferredToolCalls(handler=handle_deferred)],
    )

    caught: ToolRetryError | None = None

    @agent.tool
    async def caller_tool(ctx: RunContext) -> str:
        nonlocal caught
        assert ctx.tool_manager is not None
        try:
            await ctx.tool_manager.handle_call(
                ToolCallPart(tool_name='inner_tool', args={}, tool_call_id='inner_1'),
            )
            return 'no raise'  # pragma: no cover
        except ToolRetryError as e:
            caught = e
            return 'caught'

    await agent.run('go')
    assert caught is not None
    assert caught.tool_retry.content == 'retry via part'
    # The helper stamps the real tool name / id onto the prompt part.
    assert caught.tool_retry.tool_name == 'inner_tool'
    assert caught.tool_retry.tool_call_id == 'inner_1'


async def test_deferred_tool_handler_via_handle_call_denied_returns_message():
    """When a handler denies a deferred call, handle_call returns the custom `ToolDenied` value verbatim."""

    inner_toolset = FunctionToolset()

    @inner_toolset.tool
    def inner_tool(ctx: RunContext) -> str:
        if not ctx.tool_call_approved:
            raise ApprovalRequired
        return 'never'  # pragma: no cover

    async def handle_deferred(ctx: RunContext, requests: DeferredToolRequests) -> DeferredToolResults:
        return DeferredToolResults(
            approvals={call.tool_call_id: ToolDenied(message='not today') for call in requests.approvals}
        )

    def llm(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(parts=[ToolCallPart('caller_tool', {}, tool_call_id='c1')])
        return ModelResponse(parts=[TextPart('final')])

    agent = Agent(
        FunctionModel(llm),
        toolsets=[inner_toolset],
        capabilities=[HandleDeferredToolCalls(handler=handle_deferred)],
    )

    captured: Any = None

    @agent.tool
    async def caller_tool(ctx: RunContext) -> str:
        nonlocal captured
        assert ctx.tool_manager is not None
        captured = await ctx.tool_manager.handle_call(
            ToolCallPart(tool_name='inner_tool', args={}, tool_call_id='inner_1'),
        )
        return 'caught' if isinstance(captured, ToolDenied) else 'no denial'

    await agent.run('go')
    assert isinstance(captured, ToolDenied)
    assert captured == ToolDenied(message='not today')


async def test_deferred_tool_handler_via_handle_call_re_raises_new_exception():
    """After approval, if tool re-raises CallDeferred (not ApprovalRequired), the new exception type is propagated."""
    from pydantic_ai.exceptions import CallDeferred

    inner_toolset = FunctionToolset()
    call_count = 0

    @inner_toolset.tool
    def inner_tool(ctx: RunContext) -> str:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise ApprovalRequired
        # After approval, raise a *different* deferral type with new metadata
        raise CallDeferred(metadata={'reason': 'external-after-approval'})

    async def handle_deferred(ctx: RunContext, requests: DeferredToolRequests) -> DeferredToolResults:
        return DeferredToolResults(approvals={call.tool_call_id: True for call in requests.approvals})

    def llm(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(parts=[ToolCallPart('caller_tool', {}, tool_call_id='c1')])
        return ModelResponse(parts=[TextPart('final')])

    agent = Agent(
        FunctionModel(llm),
        toolsets=[inner_toolset],
        capabilities=[HandleDeferredToolCalls(handler=handle_deferred)],
    )

    caught_exc_type: type | None = None
    caught_metadata: dict[str, Any] | None = None

    @agent.tool
    async def caller_tool(ctx: RunContext) -> str:
        nonlocal caught_exc_type, caught_metadata
        assert ctx.tool_manager is not None
        try:
            await ctx.tool_manager.handle_call(
                ToolCallPart(tool_name='inner_tool', args={}, tool_call_id='inner_1'),
            )
            return 'no raise'  # pragma: no cover
        except (CallDeferred, ApprovalRequired) as e:
            caught_exc_type = type(e)
            caught_metadata = e.metadata
            return 'caught'

    result = await agent.run('go')
    assert result.output == 'final'
    # The new CallDeferred exception should surface, not the original ApprovalRequired
    assert caught_exc_type is CallDeferred
    assert caught_metadata == {'reason': 'external-after-approval'}


async def test_deferred_tool_handler_via_handle_call_handler_resolves_wrong_id():
    """handle_call path: handler returns results for wrong ID → remaining non-empty → raises original exc."""

    inner_toolset = FunctionToolset()

    @inner_toolset.tool
    def inner_tool(ctx: RunContext) -> str:
        if not ctx.tool_call_approved:
            raise ApprovalRequired
        return 'done'  # pragma: no cover

    async def handle_deferred(ctx: RunContext, requests: DeferredToolRequests) -> DeferredToolResults:
        # Resolve a non-existent ID — our tool's ID stays in remaining
        return DeferredToolResults(approvals={'wrong_id': True})

    def llm(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(parts=[ToolCallPart('caller_tool', {}, tool_call_id='c1')])
        return ModelResponse(parts=[TextPart('final')])

    agent = Agent(
        FunctionModel(llm),
        toolsets=[inner_toolset],
        capabilities=[HandleDeferredToolCalls(handler=handle_deferred)],
    )

    @agent.tool
    async def caller_tool(ctx: RunContext) -> str:
        assert ctx.tool_manager is not None
        try:
            await ctx.tool_manager.handle_call(
                ToolCallPart(tool_name='inner_tool', args={}, tool_call_id='inner_1'),
            )
            return 'no raise'  # pragma: no cover
        except ApprovalRequired:
            return 'caught'

    result = await agent.run('go')
    assert result.output == 'final'


async def test_deferred_tool_handler_via_hooks_decorator():
    """`@hooks.on.deferred_tool_calls` resolves deferred calls inline."""

    def llm(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(parts=[ToolCallPart('my_tool', {'x': 5}, tool_call_id='call1')])
        return ModelResponse(parts=[TextPart('Done!')])

    hooks = Hooks()

    @hooks.on.deferred_tool_calls
    async def handler(ctx: RunContext, *, requests: DeferredToolRequests) -> DeferredToolResults:
        return DeferredToolResults(approvals={call.tool_call_id: True for call in requests.approvals})

    agent = Agent(FunctionModel(llm), capabilities=[hooks])

    @agent.tool
    def my_tool(ctx: RunContext, x: int) -> int:
        if not ctx.tool_call_approved:
            raise ApprovalRequired
        return x * 10

    result = await agent.run('Hello')
    assert result.output == 'Done!'


async def test_deferred_tool_handler_via_hooks_constructor_kwarg_and_accumulation():
    """`Hooks(deferred_tool_calls=...)` accumulates results across registered handlers."""

    def llm(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(
                parts=[
                    ToolCallPart('tool_a', {}, tool_call_id='a1'),
                    ToolCallPart('tool_b', {}, tool_call_id='b1'),
                    ToolCallPart('tool_c', {}, tool_call_id='c1'),
                ]
            )
        return ModelResponse(parts=[TextPart('All done.')])

    def handle_a(ctx: RunContext, *, requests: DeferredToolRequests) -> DeferredToolResults | None:
        results = DeferredToolResults()
        for call in requests.approvals:
            if call.tool_name == 'tool_a':
                results.approvals[call.tool_call_id] = True
        return results

    hooks = Hooks(deferred_tool_calls=handle_a)

    @hooks.on.deferred_tool_calls
    async def handle_rest(ctx: RunContext, *, requests: DeferredToolRequests) -> DeferredToolResults | None:
        # tool_a was already resolved by handle_a; this handler sees only tool_b and tool_c
        return DeferredToolResults(approvals={call.tool_call_id: True for call in requests.approvals})

    @hooks.on.deferred_tool_calls
    async def never_called(  # pragma: no cover
        ctx: RunContext, *, requests: DeferredToolRequests
    ) -> DeferredToolResults | None:
        # All calls should already be resolved by the previous handler — this is the early-break path
        raise AssertionError('Should not be called: all requests already resolved')

    agent = Agent(FunctionModel(llm), capabilities=[hooks])

    @agent.tool
    def tool_a(ctx: RunContext) -> str:
        if not ctx.tool_call_approved:
            raise ApprovalRequired
        return 'a'

    @agent.tool
    def tool_b(ctx: RunContext) -> str:
        if not ctx.tool_call_approved:
            raise ApprovalRequired
        return 'b'

    @agent.tool
    def tool_c(ctx: RunContext) -> str:
        if not ctx.tool_call_approved:
            raise ApprovalRequired
        return 'c'

    result = await agent.run('Hello')
    assert result.output == 'All done.'


async def test_deferred_tool_handler_via_hooks_returns_none_when_unhandled():
    """`Hooks` returns None from the deferred-tool-calls hook when no registered handler resolves anything."""

    def llm(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[ToolCallPart('my_tool', {}, tool_call_id='call1')])

    hooks = Hooks()

    @hooks.on.deferred_tool_calls
    async def declines(ctx: RunContext, *, requests: DeferredToolRequests) -> DeferredToolResults | None:
        return None

    @hooks.on.deferred_tool_calls
    async def empty(ctx: RunContext, *, requests: DeferredToolRequests) -> DeferredToolResults | None:
        # Empty results count as "didn't handle"
        return DeferredToolResults()

    agent = Agent(
        FunctionModel(llm),
        output_type=[str, DeferredToolRequests],
        capabilities=[hooks],
    )

    @agent.tool
    def my_tool(ctx: RunContext) -> str:
        if not ctx.tool_call_approved:
            raise ApprovalRequired
        return 'done'  # pragma: no cover

    result = await agent.run('Hello')
    # Falls through to bubble-up since no handler resolved anything
    assert isinstance(result.output, DeferredToolRequests)
    assert len(result.output.approvals) == 1


