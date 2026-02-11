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

import pytest
from inline_snapshot import snapshot

from pydantic_ai import Agent
from pydantic_ai._run_context import RunContext
from pydantic_ai.exceptions import ApprovalRequired, CallDeferred, UserError
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.runtime.abstract import CodeRuntime
from pydantic_ai.tools import DeferredToolRequests, DeferredToolResults, ToolApproved
from pydantic_ai.toolsets.code_mode import CodeModeContext, CodeModeToolset
from pydantic_ai.toolsets.function import FunctionToolset

from ..conftest import IsBytes, IsList, IsStr

pytestmark = pytest.mark.anyio


async def test_code_mode_mixed_approval_and_deferred(code_runtime: CodeRuntime):
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
    code_toolset = CodeModeToolset(wrapped=toolset, runtime=code_runtime)

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
    assert isinstance(result.output, DeferredToolRequests)
    run_code_call_id = result.output.approvals[0].tool_call_id
    ctx: CodeModeContext = result.output.context[run_code_call_id]  # type: ignore[assignment]

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

    # Build results for nested calls - type-safe access to interrupted_calls
    nested_results: dict[str, object] = {}
    for ic in ctx['interrupted_calls']:
        if ic['type'] == 'approval':
            nested_results[ic['call_id']] = ToolApproved()
        else:
            nested_results[ic['call_id']] = 'ext_result'

    # Build resume context: original context + results
    resume_ctx: CodeModeContext = {
        'checkpoint': ctx['checkpoint'],
        'interrupted_calls': ctx['interrupted_calls'],
        'code': ctx['code'],
        'functions': ctx['functions'],
        'signatures': ctx['signatures'],
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


async def test_code_mode_resume_missing_context_error(code_runtime: CodeRuntime):
    """Test that resuming without context raises a clear error."""

    def sensitive_action(ctx: RunContext[None], value: int) -> int:
        if not ctx.tool_call_approved:
            raise ApprovalRequired()
        return value * 10

    toolset: FunctionToolset[None] = FunctionToolset()
    toolset.add_function(sensitive_action, takes_ctx=True)
    code_toolset = CodeModeToolset(wrapped=toolset, runtime=code_runtime)

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
                    'code': 'await sensitive_action(value=42)',
                    'functions': ['sensitive_action'],
                    'signatures': IsList(IsStr(), length=1),
                }
            },
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


async def test_code_mode_resume_missing_results_error(code_runtime: CodeRuntime):
    """Test that resuming without results for interrupted calls raises a clear error."""

    def sensitive_action(ctx: RunContext[None], value: int) -> int:
        if not ctx.tool_call_approved:
            raise ApprovalRequired()
        return value * 10

    toolset: FunctionToolset[None] = FunctionToolset()
    toolset.add_function(sensitive_action, takes_ctx=True)
    code_toolset = CodeModeToolset(wrapped=toolset, runtime=code_runtime)

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
                    'code': 'await sensitive_action(value=42)',
                    'functions': ['sensitive_action'],
                    'signatures': IsList(IsStr(), length=1),
                }
            },
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


async def test_code_mode_mixed_success_and_interrupted(code_runtime: CodeRuntime):
    """Test that calls which succeed before an interruption are NOT re-executed on resume.

    3 parallel tools: 1 succeeds immediately, 1 needs approval, 1 is deferred.
    On resume the successful call's result is fed back via completed_results,
    avoiding double execution.
    """
    call_counts: dict[str, int] = {'fast': 0, 'sensitive': 0, 'external': 0}

    def fast_lookup(key: str) -> str:
        call_counts['fast'] += 1
        return f'found:{key}'

    def sensitive_action(ctx: RunContext[None], value: int) -> int:
        call_counts['sensitive'] += 1
        if not ctx.tool_call_approved:
            raise ApprovalRequired()
        return value * 10

    def external_service(data: str) -> str:
        call_counts['external'] += 1
        raise CallDeferred()

    toolset: FunctionToolset[None] = FunctionToolset()
    toolset.add_function(fast_lookup, takes_ctx=False)
    toolset.add_function(sensitive_action, takes_ctx=True)
    toolset.add_function(external_service, takes_ctx=False)
    code_toolset = CodeModeToolset(wrapped=toolset, runtime=code_runtime)

    call_count = 0

    def llm(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            # Fire all three in parallel, then await
            code = """\
f1 = fast_lookup(key="abc")
f2 = sensitive_action(value=42)
f3 = external_service(data="test")
r1 = await f1
r2 = await f2
r3 = await f3
{"fast": r1, "sensitive": r2, "external": r3}"""
            return ModelResponse(parts=[ToolCallPart('run_code', {'code': code}, tool_call_id='rc1')])
        return ModelResponse(parts=[TextPart('All three done!')])

    agent: Agent[None, str | DeferredToolRequests] = Agent(FunctionModel(llm), output_type=[str, DeferredToolRequests])

    # Run 1: fast_lookup succeeds, the other two are interrupted
    async with code_toolset:
        result = await agent.run('Do all three', toolsets=[code_toolset])

    assert isinstance(result.output, DeferredToolRequests)
    assert call_counts == {'fast': 1, 'sensitive': 1, 'external': 1}

    # Context should have interrupted_calls for the two interrupted tools
    # (completed_results for fast_lookup are bundled inside the checkpoint)
    run_code_call_id = result.output.approvals[0].tool_call_id
    ctx: CodeModeContext = result.output.context[run_code_call_id]  # type: ignore[assignment]

    assert len(ctx['interrupted_calls']) == 2

    interrupted_names = {ic['tool_name'] for ic in ctx['interrupted_calls']}
    assert interrupted_names == {'sensitive_action', 'external_service'}

    # Build results for interrupted calls
    nested_results: dict[str, object] = {}
    for ic in ctx['interrupted_calls']:
        if ic['type'] == 'approval':
            nested_results[ic['call_id']] = ToolApproved()
        else:
            nested_results[ic['call_id']] = 'ext_result'

    # Build resume context — completed_results are inside the checkpoint,
    # so we just pass the original context fields + results
    resume_ctx: CodeModeContext = {
        'checkpoint': ctx['checkpoint'],
        'interrupted_calls': ctx['interrupted_calls'],
        'code': ctx['code'],
        'functions': ctx['functions'],
        'signatures': ctx['signatures'],
        'results': nested_results,
    }
    resume_context = {run_code_call_id: resume_ctx}

    # Run 2: Resume — fast_lookup should NOT be called again
    async with code_toolset:
        result = await agent.run(
            message_history=result.all_messages(),
            deferred_tool_results=DeferredToolResults(
                approvals={run_code_call_id: ToolApproved()},
                context=resume_context,
            ),
            toolsets=[code_toolset],
        )

    assert result.output == snapshot('All three done!')

    # Critical: fast_lookup was only called once (not re-executed on resume)
    assert call_counts['fast'] == 1
    # sensitive_action called twice: once to get ApprovalRequired, once approved
    assert call_counts['sensitive'] == 2
    # external_service called once (deferred, result provided directly)
    assert call_counts['external'] == 1


async def test_code_mode_nested_approvals(code_runtime: CodeRuntime):
    """Test that a resumed execution can trigger a second round of approvals.

    Scenario: sequential code where a later tool call depends on an earlier
    approved result. The second tool also needs approval, so resume hits a
    second interruption which must propagate as a fresh ApprovalRequired.

    Flow:
      Run 1: action_1 needs approval → interrupted
      Run 2: resume with action_1 approved → executes → code continues →
              action_2 needs approval → second interruption
      Run 3: resume with action_2 approved → executes → code completes
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

    call_count = 0

    def llm(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            # Sequential: action_2 depends on action_1's result
            code = """\
r1 = await action_1(value=5)
r2 = await action_2(value=r1)
{"first": r1, "second": r2}"""
            return ModelResponse(parts=[ToolCallPart('run_code', {'code': code}, tool_call_id='rc1')])
        return ModelResponse(parts=[TextPart('All done!')])

    agent: Agent[None, str | DeferredToolRequests] = Agent(FunctionModel(llm), output_type=[str, DeferredToolRequests])

    # --- Run 1: action_1 needs approval ---
    async with code_toolset:
        result = await agent.run('Do sequential', toolsets=[code_toolset])

    assert isinstance(result.output, DeferredToolRequests)
    run_code_call_id = result.output.approvals[0].tool_call_id
    ctx1: CodeModeContext = result.output.context[run_code_call_id]  # type: ignore[assignment]

    assert len(ctx1['interrupted_calls']) == 1
    assert ctx1['interrupted_calls'][0]['tool_name'] == 'action_1'
    assert ctx1['interrupted_calls'][0]['type'] == 'approval'

    # --- Run 2: resume with action_1 approved → should hit action_2 interruption ---
    nested_results_1: dict[str, object] = {
        ctx1['interrupted_calls'][0]['call_id']: ToolApproved(),
    }
    resume_ctx_1: CodeModeContext = {
        **ctx1,
        'results': nested_results_1,
    }

    async with code_toolset:
        result = await agent.run(
            message_history=result.all_messages(),
            deferred_tool_results=DeferredToolResults(
                approvals={run_code_call_id: ToolApproved()},
                context={run_code_call_id: resume_ctx_1},
            ),
            toolsets=[code_toolset],
        )

    # Should get a SECOND DeferredToolRequests for action_2
    assert isinstance(result.output, DeferredToolRequests)
    run_code_call_id_2 = result.output.approvals[0].tool_call_id
    ctx2: CodeModeContext = result.output.context[run_code_call_id_2]  # type: ignore[assignment]

    assert len(ctx2['interrupted_calls']) == 1
    assert ctx2['interrupted_calls'][0]['tool_name'] == 'action_2'
    assert ctx2['interrupted_calls'][0]['type'] == 'approval'
    # action_2 receives action_1's result (5 * 10 = 50)
    assert ctx2['interrupted_calls'][0]['kwargs'] == {'value': 50}

    # --- Run 3: resume with action_2 approved → should complete ---
    nested_results_2: dict[str, object] = {
        ctx2['interrupted_calls'][0]['call_id']: ToolApproved(),
    }
    resume_ctx_2: CodeModeContext = {
        **ctx2,
        'results': nested_results_2,
    }

    async with code_toolset:
        result = await agent.run(
            message_history=result.all_messages(),
            deferred_tool_results=DeferredToolResults(
                approvals={run_code_call_id_2: ToolApproved()},
                context={run_code_call_id_2: resume_ctx_2},
            ),
            toolsets=[code_toolset],
        )

    assert result.output == snapshot('All done!')
