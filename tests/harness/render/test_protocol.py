from __future__ import annotations

import re

import pytest
from pydantic import TypeAdapter
from render.workflows import Workflows

from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.capabilities import AbstractCapability, durable_operation
from pydantic_ai.exceptions import (
    ApprovalRequired,
    CallDeferred,
    SkipModelRequest,
    SkipToolExecution,
    SkipToolValidation,
    ToolFailed,
    UserError,
)
from pydantic_ai.messages import ModelMessage, ModelResponse, RetryPromptPart, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai_harness import RenderWorkflows

from .conftest import Tamper, TaskBoundary, is_envelope, mcp_toolset, rewrite_tool_context, run_agent_in_task

_ENVELOPE = TypeAdapter(dict[str, object])


def _set(key: str, value: object) -> Tamper:
    return lambda envelope: envelope.__setitem__(key, value)


def _drop(key: str) -> Tamper:
    return lambda envelope: envelope.__delitem__(key)


def _in_error(tamper: Tamper) -> Tamper:
    """Apply a rewrite to the nested error object of a control-flow result."""

    def rewrite(envelope: dict[str, object]) -> None:
        error = _ENVELOPE.validate_python(envelope['error'], strict=True)
        tamper(error)
        envelope['error'] = error

    return rewrite


class Audit(AbstractCapability[None]):
    """A capability whose durable operation runs as a Render child task."""

    id = 'audit'

    def __init__(self, failure: Exception | None = None) -> None:
        self.failure = failure

    @durable_operation(name='record')
    async def record(self, ctx: RunContext[None], message: str) -> str:
        del ctx
        if self.failure is not None:
            raise self.failure
        return f'recorded:{message}'


def build_agent() -> tuple[Agent[None, str], RenderWorkflows[None]]:
    runtime = RenderWorkflows[None](Workflows())
    agent = Agent[None, str](TestModel(), name='support', deps_type=type(None), capabilities=[runtime])
    return agent, runtime


def build_audited_agent(
    failure: Exception | None = None, *, capture: bool = True
) -> tuple[Agent[None, str], RenderWorkflows[None], list[Exception]]:
    """Build an agent that calls a capability operation from the workflow side of the boundary.

    With `capture`, whatever the operation call raises is collected instead of ending the run, so
    a test can assert on the exception the protocol recreated rather than on how the agent then
    reacted to it. Without it, the error ends the run the way an unhandled one would.
    """
    runtime = RenderWorkflows[None](Workflows())
    audit = Audit(failure)
    agent = Agent[None, str](TestModel(), name='audited', deps_type=type(None), capabilities=[runtime, audit])
    recreated: list[Exception] = []

    @agent.instructions
    async def audited_instructions(ctx: RunContext[None]) -> str:
        if not capture:
            return await audit.record(ctx, 'hello')
        try:
            return await audit.record(ctx, 'hello')
        except Exception as exc:
            recreated.append(exc)
            return 'audit unavailable'

    return agent, runtime, recreated


def _error_kind(result: dict[str, object]) -> object:
    error = result['error']
    assert is_envelope(error)
    return error['kind']


def _assert_completed_with_permanent_error(context: TaskBoundary, *, task: str, kind: str) -> None:
    """Assert the child task ran and finished with exactly one permanent-error result."""
    assert context.started == [task]
    assert [result['status'] for result in context.results] == ['error']
    assert _error_kind(context.results[0]) == kind


def _assert_control_flow_payload(kind: str, error: Exception) -> None:
    """Assert the semantic value reconstructed for every supported control-flow kind."""
    if kind == 'model-retry':
        assert isinstance(error, ModelRetry)
        assert error.message == 'again'
    elif kind == 'tool-failed':
        assert isinstance(error, ToolFailed)
        assert error.message == 'nope'
    elif kind == 'approval-required':
        assert isinstance(error, ApprovalRequired)
        assert error.metadata == {'reason': 'review'}
    elif kind == 'call-deferred':
        assert isinstance(error, CallDeferred)
        assert error.metadata == {'queue': 'later'}
    elif kind == 'skip-model-request':
        assert isinstance(error, SkipModelRequest)
        assert error.response.parts == [TextPart('skipped')]
    elif kind == 'skip-tool-validation':
        assert isinstance(error, SkipToolValidation)
        assert error.validated_args == {'value': 1}
    else:
        assert kind == 'skip-tool-execution'
        assert isinstance(error, SkipToolExecution)
        assert error.result == 'done'


async def test_agent_result_survives_the_json_round_trip() -> None:
    agent, runtime = build_agent()
    context = TaskBoundary()

    assert await run_agent_in_task(agent, runtime, context) == 'success (no tool calls)'
    assert context.requests
    assert context.results


@pytest.mark.parametrize(
    'tamper',
    [
        pytest.param(_set('operation', 'support__model.compact_messages'), id='wrong-operation'),
        pytest.param(_drop('payload'), id='missing-payload'),
    ],
)
async def test_malformed_request_completes_the_child_with_an_invalid_request_error(tamper: Tamper) -> None:
    agent, runtime = build_agent()
    context = TaskBoundary(tamper_request=tamper)

    # Retrying cannot repair persisted request bytes, so the worker completes the task with an
    # error result and the workflow side is what raises once it reads that result back.
    with pytest.raises(UserError, match='Render operation invalid request'):
        await run_agent_in_task(agent, runtime, context)

    _assert_completed_with_permanent_error(context, task='support__model.request', kind='invalid-request')


@pytest.mark.parametrize(
    'tamper',
    [
        pytest.param(_set('version', 3), id='unsupported-version'),
        pytest.param(_drop('payload'), id='missing-payload'),
    ],
)
async def test_malformed_result_envelope_is_rejected_workflow_side(tamper: Tamper) -> None:
    agent, runtime = build_agent()

    with pytest.raises(ValueError):
        await run_agent_in_task(agent, runtime, TaskBoundary(tamper_result=tamper))


@pytest.mark.parametrize(
    ('failure', 'kind', 'expected'),
    [
        pytest.param(ModelRetry('again'), 'model-retry', ModelRetry, id='model-retry'),
        pytest.param(ToolFailed('nope'), 'tool-failed', ToolFailed, id='tool-failed'),
        pytest.param(
            ApprovalRequired(metadata={'reason': 'review'}), 'approval-required', ApprovalRequired, id='approval'
        ),
        pytest.param(CallDeferred(metadata={'queue': 'later'}), 'call-deferred', CallDeferred, id='call-deferred'),
        pytest.param(
            SkipModelRequest(ModelResponse(parts=[TextPart('skipped')])),
            'skip-model-request',
            SkipModelRequest,
            id='skip-model-request',
        ),
        pytest.param(
            SkipToolValidation({'value': 1}), 'skip-tool-validation', SkipToolValidation, id='skip-tool-validation'
        ),
        pytest.param(SkipToolExecution('done'), 'skip-tool-execution', SkipToolExecution, id='skip-tool-execution'),
    ],
)
async def test_expected_control_flow_crosses_as_a_result_and_is_recreated(
    failure: Exception, kind: str, expected: type[Exception]
) -> None:
    agent, runtime, recreated = build_audited_agent(failure)
    context = TaskBoundary()

    await run_agent_in_task(agent, runtime, context)

    # The child task returned a result rather than failing, so Render keeps the journal entry
    # and the workflow side turns that result back into the same control flow.
    control_flow = [result for result in context.results if result['status'] == 'control-flow']
    assert [_error_kind(result) for result in control_flow] == [kind]
    assert [type(exc) for exc in recreated] == [expected]
    _assert_control_flow_payload(kind, recreated[0])


async def test_unknown_control_flow_result_is_rejected_workflow_side() -> None:
    agent, runtime, _ = build_audited_agent(ModelRetry('again'), capture=False)

    with pytest.raises(ValueError):
        await run_agent_in_task(agent, runtime, TaskBoundary(tamper_result=_in_error(_set('kind', 'nap'))))


async def test_malformed_control_flow_metadata_is_rejected_workflow_side() -> None:
    agent, runtime, _ = build_audited_agent(ApprovalRequired(metadata={'reason': 'review'}), capture=False)

    with pytest.raises(ValueError, match='Approval metadata must be a JSON object'):
        await run_agent_in_task(agent, runtime, TaskBoundary(tamper_result=_in_error(_set('metadata', 'review'))))


async def test_control_flow_payload_that_cannot_be_encoded_completes_the_child_with_an_invalid_result_error() -> None:
    agent, runtime, _ = build_audited_agent(ApprovalRequired(metadata={'reviewer': object()}), capture=False)
    context = TaskBoundary()

    with pytest.raises(UserError, match='Approval metadata must be JSON serializable'):
        await run_agent_in_task(agent, runtime, context)

    _assert_completed_with_permanent_error(context, task='audited__capability__audit.record', kind='invalid-result')


async def test_unexpected_handler_error_leaves_the_task_failed_rather_than_encoded() -> None:
    agent, runtime, _ = build_audited_agent(RuntimeError('provider exploded'), capture=False)
    context = TaskBoundary()

    with pytest.raises(RuntimeError, match='provider exploded'):
        await run_agent_in_task(agent, runtime, context)

    # Unlike Pydantic ModelRetry control flow, this exception fails the child task.
    assert 'audited__capability__audit.record' in context.started
    assert all(result['status'] == 'ok' for result in context.results)


async def test_function_tool_model_retry_is_completed_control_flow_and_agent_continues() -> None:
    calls = 0

    def retry_then_finish(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        del info
        for message in reversed(messages):
            for part in reversed(message.parts):
                if isinstance(part, ToolReturnPart):
                    return ModelResponse(parts=[TextPart('continued after retry')])
                if isinstance(part, RetryPromptPart):
                    return ModelResponse(parts=[ToolCallPart('unstable_tool', {}, tool_call_id='retry-call')])
        return ModelResponse(parts=[ToolCallPart('unstable_tool', {}, tool_call_id='initial-call')])

    async def unstable_tool() -> str:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ModelRetry('again')
        return 'recovered'

    runtime = RenderWorkflows[None](Workflows())
    agent = Agent[None, str](
        FunctionModel(retry_then_finish),
        name='retrying-agent',
        deps_type=type(None),
        tools=[unstable_tool],
        capabilities=[runtime],
    )
    context = TaskBoundary()

    output = await run_agent_in_task(agent, runtime, context)

    assert output == 'continued after retry'
    assert calls == 2
    completed_model_retries: list[dict[str, object]] = []
    for result in context.results:
        payload = result.get('payload')
        if result['status'] == 'ok' and is_envelope(payload) and payload.get('kind') == 'model_retry':
            completed_model_retries.append(payload)
    assert len(completed_model_retries) == 1
    assert completed_model_retries[0]['message'] == 'again'
    assert context.started.count('retrying-agent__function_toolset__<agent>.call_tool') == 2


async def test_protocol_enforces_four_mib_request_limit_through_public_agent() -> None:
    agent, runtime = build_agent()

    with pytest.raises(ValueError, match='4194304-byte limit') as raised:
        await run_agent_in_task(agent, runtime, TaskBoundary(), prompt='x' * (4 * 1024 * 1024))

    # The limit applies to the serialized task arguments, not to the prompt alone, so a prompt of
    # exactly 4 MiB is already over it once the envelope and JSON punctuation are counted.
    reported = re.search(r'arguments are (\d+) bytes', str(raised.value))
    assert reported is not None
    assert int(reported.group(1)) > 4194304


@pytest.mark.parametrize(
    'fields',
    [
        {'status': 'future'},
        {'effects': {'events': [{'event_kind': 'unknown'}]}},
    ],
)
async def test_foreign_worker_results_reject_invalid_envelope_and_effects(fields: dict[str, object]) -> None:
    agent, runtime = build_agent()
    with pytest.raises(ValueError):
        await run_agent_in_task(agent, runtime, TaskBoundary(tamper_result=lambda result: result.update(fields)))


@pytest.mark.parametrize('error', [{'kind': 'future', 'message': 'x'}, {'kind': 'invalid-result', 'message': 1}])
async def test_foreign_worker_permanent_errors_are_validated(error: dict[str, object]) -> None:
    agent, runtime = build_agent()

    def replace_result(result: dict[str, object]) -> None:
        result.clear()
        result.update(version=2, status='error', error=error)

    with pytest.raises(ValueError):
        await run_agent_in_task(agent, runtime, TaskBoundary(tamper_result=replace_result))


@pytest.mark.parametrize('failure', [ApprovalRequired(), CallDeferred()])
async def test_control_flow_without_metadata_is_preserved(failure: Exception) -> None:
    agent, runtime, recreated = build_audited_agent(failure)
    await run_agent_in_task(agent, runtime, TaskBoundary())
    assert len(recreated) == 1
    assert isinstance(recreated[0], ApprovalRequired | CallDeferred)
    assert recreated[0].metadata is None


async def test_successful_capability_operation_result_crosses_json() -> None:
    agent, runtime, errors = build_audited_agent()
    context = TaskBoundary()
    await run_agent_in_task(agent, runtime, context)
    assert errors == []
    assert any(
        is_envelope(payload := result.get('payload')) and payload.get('value') == 'recorded:hello'
        for result in context.results
    )


@pytest.mark.parametrize(
    ('key', 'value', 'message'),
    [
        ('version', 0, 'run-context version'),
        ('context', [], 'JSON object'),
        ('deps', 'wrong', 'validation error'),
        ('tracer_enabled', 'yes', 'boolean'),
        ('_model_id', 42, 'string or null'),
    ],
)
async def test_worker_rejects_malformed_run_context(key: str, value: object, message: str) -> None:
    agent, runtime = build_agent()

    def rewrite(request: dict[str, object]) -> None:
        payload = _ENVELOPE.validate_python(request['payload'])
        context = _ENVELOPE.validate_python(payload['run_context'])
        if key in {'tracer_enabled', '_model_id'}:
            fields = _ENVELOPE.validate_python(context['context'])
            fields[key] = value
            context['context'] = fields
        else:
            context[key] = value
        payload['run_context'] = context
        request['payload'] = payload

    with pytest.raises(UserError, match=message):
        await run_agent_in_task(agent, runtime, TaskBoundary(tamper_request=rewrite))


@pytest.mark.parametrize('mode', ['missing-definition', 'unknown-original-name', 'unknown-name'])
async def test_worker_rebuilds_or_rejects_a_changed_function_definition(mode: str) -> None:
    async def lookup() -> str:
        return 'found'

    runtime = RenderWorkflows[None](Workflows())
    agent = Agent(TestModel(), name='changed-tool', deps_type=type(None), tools=[lookup], capabilities=[runtime])

    def rewrite(request: dict[str, object]) -> None:
        if not str(request['operation']).endswith('.call_tool'):
            return
        payload = _ENVELOPE.validate_python(request['payload'])
        if mode == 'unknown-original-name':
            payload['original_name'] = 'missing'
        else:
            payload['tool_def'] = None
            if mode == 'unknown-name':
                payload['name'] = 'missing'
        request['payload'] = payload

    context = TaskBoundary(tamper_request=rewrite)
    if mode == 'missing-definition':
        assert 'found' in await run_agent_in_task(agent, runtime, context)
    else:
        with pytest.raises(UserError, match='not found in toolset'):
            await run_agent_in_task(agent, runtime, context)


async def test_unserializable_operation_result_finishes_with_permanent_error() -> None:
    class InvalidResult(AbstractCapability[None]):
        id = 'invalid-result'

        @durable_operation(name='produce')
        async def produce(self, ctx: RunContext[None]) -> object:
            return object()

    capability = InvalidResult()
    runtime = RenderWorkflows[None](Workflows())
    agent = Agent(TestModel(), name='invalid-output', deps_type=type(None), capabilities=[capability, runtime])

    @agent.instructions
    async def instructions(ctx: RunContext[None]) -> str:
        await capability.produce(ctx)
        pytest.fail(
            'an invalid operation result must fail before the agent continues'
        )  # pragma: no cover - invocation is outside this registration or rejection contract

    context = TaskBoundary()
    with pytest.raises(UserError, match='invalid result'):
        await run_agent_in_task(agent, runtime, context)
    assert context.results[0]['status'] == 'error'


async def test_worker_rejects_missing_dependencies() -> None:
    agent, runtime = build_agent()

    def rewrite(request: dict[str, object]) -> None:
        payload = _ENVELOPE.validate_python(request['payload'])
        context = _ENVELOPE.validate_python(payload['run_context'])
        del context['deps']
        payload['run_context'] = context
        request['payload'] = payload

    with pytest.raises(UserError, match='requires `deps`'):
        await run_agent_in_task(agent, runtime, TaskBoundary(tamper_request=rewrite))


@pytest.mark.parametrize(
    'missing', [None, 'usage', 'available_tool_names', 'active_capability_ids', '_deferred_capability_ids']
)
async def test_context_snapshots_and_missing_field_guards(missing: str | None) -> None:
    from pydantic_ai.tools import ToolDefinition

    runtime = RenderWorkflows[None](Workflows())
    agent = Agent(TestModel(), name='context-fields', deps_type=type(None), capabilities=[runtime])

    @agent.tool
    async def inspect_context(ctx: RunContext[None]) -> str:
        if missing == 'available_tool_names':
            assert ctx.available_tool_names == set()
        elif missing == 'active_capability_ids':
            with pytest.raises(UserError, match=r'capabilities.*not available'):
                _ = ctx.active_capability_ids
        elif missing == '_deferred_capability_ids':
            with pytest.raises(UserError, match=r'capabilities.*not available'):
                ctx.is_tool_available(ToolDefinition(name='deferred', defer_loading=True, capability_id='deferred'))
        elif missing is None:
            assert not ctx.is_tool_available(
                ToolDefinition(name='deferred', defer_loading=True, capability_id='deferred')
            )
        return 'checked'

    context = TaskBoundary(tamper_request=rewrite_tool_context(_drop(missing)) if missing else None)
    assert 'checked' in await run_agent_in_task(agent, runtime, context)


async def test_missing_worker_workspace_provider_has_an_actionable_error() -> None:
    from pydantic_ai.workspaces import WorkspaceUnavailableError

    runtime = RenderWorkflows[None](Workflows())
    agent = Agent(TestModel(), name='missing-workspace', deps_type=type(None), capabilities=[runtime])

    @agent.tool
    async def read_file(ctx: RunContext[None]) -> str:
        with pytest.raises(WorkspaceUnavailableError, match='same workspace capabilities'):
            await ctx.workspace.read_text('note.txt')
        return 'unavailable'

    context = TaskBoundary(
        tamper_request=rewrite_tool_context(_set('workspace_ref', {'id': 'absent', 'provider': 'missing'}))
    )
    assert 'unavailable' in await run_agent_in_task(agent, runtime, context)


async def test_mcp_worker_requires_a_serialized_tool_definition() -> None:
    toolset, calls = mcp_toolset()
    runtime = RenderWorkflows[None](Workflows())
    agent = Agent(
        TestModel(call_tools=['remote_lookup']),
        name='missing-mcp-definition',
        deps_type=type(None),
        toolsets=[toolset],
        capabilities=[runtime],
    )

    def rewrite(request: dict[str, object]) -> None:
        if str(request['operation']).endswith('.call_tool'):
            payload = _ENVELOPE.validate_python(request['payload'])
            payload['tool_def'] = None
            request['payload'] = payload

    with pytest.raises(UserError, match='no serialized definition'):
        await run_agent_in_task(agent, runtime, TaskBoundary(tamper_request=rewrite))
    assert calls == []
