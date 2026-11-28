from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Literal

import pytest

from ..conftest import try_import

with try_import() as imports_successful:
    from pydantic_ai import Agent, ModelMessage, ModelResponse, ModelRetry, TextPart, ToolCallPart
    from pydantic_ai.capabilities import Instrumentation
    from pydantic_ai.models.function import AgentInfo, FunctionModel
    from pydantic_ai.models.test import TestModel
    from pydantic_evals import Case, Dataset
    from pydantic_evals.evaluators import EvaluationReason, EvaluatorContext, ToolPolicyCompliance
    from pydantic_evals.otel import SpanTreeRecordingError
    from pydantic_evals.otel.span_tree import SpanNode, SpanStatus, SpanTree

with try_import() as logfire_import_successful:
    from logfire.testing import CaptureLogfire

pytestmark = [pytest.mark.skipif(not imports_successful(), reason='pydantic-evals not installed')]
needs_logfire = pytest.mark.skipif(not logfire_import_successful(), reason='logfire not installed')

_EPOCH = datetime(2025, 1, 1, tzinfo=UTC)


def _make_span(
    *,
    name: str,
    span_id: int,
    parent_span_id: int | None = None,
    attributes: dict[str, Any] | None = None,
    start_offset: float = 0.0,
    duration: float = 0.01,
    trace_id: int = 1,
    status: SpanStatus = 'unset',
) -> SpanNode:
    return SpanNode(
        name=name,
        trace_id=trace_id,
        span_id=span_id,
        parent_span_id=parent_span_id,
        start_timestamp=_EPOCH + timedelta(seconds=start_offset),
        end_timestamp=_EPOCH + timedelta(seconds=start_offset + duration),
        attributes=dict(attributes or {}),
        status=status,
    )


def _v2_tool_span(*, name: str, span_id: int, args: str | None, start_offset: float) -> SpanNode:
    attrs: dict[str, Any] = {
        'gen_ai.tool.name': name,
        'logfire.msg': f'running tool: {name}',
    }
    if args is not None:
        attrs['tool_arguments'] = args
    return _make_span(name='running tool', span_id=span_id, attributes=attrs, start_offset=start_offset)


def _v3_tool_span(
    *,
    name: str,
    span_id: int,
    args: str | None,
    start_offset: float,
    status: SpanStatus = 'unset',
) -> SpanNode:
    attrs: dict[str, Any] = {
        'gen_ai.tool.name': name,
        'logfire.msg': f'running tool: {name}',
    }
    if args is not None:
        attrs['gen_ai.tool.call.arguments'] = args
    return _make_span(
        name=f'execute_tool {name}',
        span_id=span_id,
        attributes=attrs,
        start_offset=start_offset,
        status=status,
    )


def _deferred_tool_span(*, name: str, span_id: int, start_offset: float) -> SpanNode:
    return _make_span(
        name=f'execute_tool {name}',
        span_id=span_id,
        attributes={
            'gen_ai.tool.name': name,
            'logfire.msg': f'running tool: {name}',
            'gen_ai.tool.call.arguments': '{}',
            'pydantic_ai.tool.deferral.name': 'ApprovalRequired',
        },
        start_offset=start_offset,
    )


def _build_tree(nodes: list[SpanNode]) -> SpanTree:
    tree = SpanTree()
    tree.add_spans(nodes)
    return tree


def _ctx(*, tree: SpanTree | SpanTreeRecordingError) -> EvaluatorContext[Any, Any, Any]:
    inputs: dict[str, Any] = {}
    return EvaluatorContext(
        name='test',
        inputs=inputs,
        metadata=None,
        expected_output=None,
        output=None,
        duration=0.0,
        _span_tree=tree,
        attributes={},
        metrics={},
    )


# ---------------------------------------------------------------------------
# Constructor validation & metadata
# ---------------------------------------------------------------------------


def test_negative_max_calls_raises():
    with pytest.raises(ValueError, match=r'must be non-negative, got -1'):
        ToolPolicyCompliance(max_calls_per_tool={'send_email': -1})


def test_invalid_regex_raises():
    with pytest.raises(ValueError, match=r"Invalid regex pattern for tool 'fetch_url', argument 'url'"):
        ToolPolicyCompliance(forbidden_arg_patterns={'fetch_url': {'url': '[unclosed'}})


def test_evaluation_name_default_and_override():
    assert ToolPolicyCompliance().get_default_evaluation_name() == 'ToolPolicyCompliance'
    assert ToolPolicyCompliance(evaluation_name='security_audit').get_default_evaluation_name() == 'security_audit'


def test_empty_policy_passes():
    tree = _build_tree([_v3_tool_span(name='search', span_id=1, args='{"q": "ok"}', start_offset=0.0)])
    assert ToolPolicyCompliance().evaluate(_ctx(tree=tree)) == EvaluationReason(value=True)


def test_no_span_tree_degrades_gracefully():
    result = ToolPolicyCompliance(forbidden_tools=['delete_user']).evaluate(
        _ctx(tree=SpanTreeRecordingError('spans were not recorded'))
    )
    assert result.value is False
    assert result.reason is not None
    assert 'logfire' in result.reason


# ---------------------------------------------------------------------------
# Span filtering & ordering (v2, v3+, output functions, deferrals, invalid)
# ---------------------------------------------------------------------------


def test_span_filtering_excludes_non_tools_output_functions_and_deferrals():
    tree = _build_tree(
        [
            _make_span(name='logfire', span_id=1, start_offset=0.0),
            _make_span(name='running tool', span_id=2, attributes={'gen_ai.tool.name': 42}, start_offset=0.01),
            _make_span(
                name='running output function',
                span_id=3,
                attributes={'gen_ai.tool.name': 'delete_user'},
                start_offset=0.02,
            ),
            _make_span(
                name='execute_tool delete_user',
                span_id=4,
                attributes={
                    'gen_ai.tool.name': 'delete_user',
                    'logfire.msg': 'running output function: delete_user',
                },
                start_offset=0.03,
            ),
            _make_span(
                name='custom_span',
                span_id=5,
                attributes={'gen_ai.tool.name': 'delete_user'},
                start_offset=0.04,
            ),
            _deferred_tool_span(name='delete_user', span_id=6, start_offset=0.05),
            _v2_tool_span(name='read_profile', span_id=7, args='{"id": 1}', start_offset=0.06),
        ]
    )
    policy = ToolPolicyCompliance(forbidden_tools=['delete_user'])
    assert policy.evaluate(_ctx(tree=tree)) == EvaluationReason(value=True)


# ---------------------------------------------------------------------------
# forbidden_tools
# ---------------------------------------------------------------------------


def test_forbidden_tools_detects_all_violations():
    tree = _build_tree(
        [
            _v2_tool_span(name='search', span_id=1, args='{}', start_offset=0.0),
            _v3_tool_span(name='delete_user', span_id=2, args='{"id": 7}', start_offset=0.1),
            _v3_tool_span(name='execute_shell', span_id=3, args='{"cmd": "whoami"}', start_offset=0.2),
        ]
    )
    result = ToolPolicyCompliance(forbidden_tools=['delete_user', 'execute_shell']).evaluate(_ctx(tree=tree))
    assert result.value is False
    assert result.reason is not None
    assert "call #2 invoked forbidden tool 'delete_user'" in result.reason
    assert "call #3 invoked forbidden tool 'execute_shell'" in result.reason


# ---------------------------------------------------------------------------
# max_calls_per_tool
# ---------------------------------------------------------------------------


def test_max_calls_per_tool_boundary_and_zero_limit():
    tree = _build_tree(
        [
            _v3_tool_span(name='send_email', span_id=1, args='{}', start_offset=0.0),
            _v3_tool_span(name='send_email', span_id=2, args='{}', start_offset=0.1),
            _v3_tool_span(name='write_db', span_id=3, args='{}', start_offset=0.2),
        ]
    )
    assert ToolPolicyCompliance(max_calls_per_tool={'send_email': 2, 'write_db': 1}).evaluate(
        _ctx(tree=tree)
    ) == EvaluationReason(value=True)

    result = ToolPolicyCompliance(max_calls_per_tool={'send_email': 1, 'write_db': 0}).evaluate(_ctx(tree=tree))
    assert result.value is False
    assert result.reason is not None
    assert "tool 'send_email' called 2 time(s), max allowed is 1" in result.reason
    assert "tool 'write_db' called 1 time(s), max allowed is 0" in result.reason


# ---------------------------------------------------------------------------
# require_before & failed-attempt semantics
# ---------------------------------------------------------------------------


def test_require_before_satisfied_in_chronological_order():
    # Insert out of order to prove start_timestamp sorting is honored.
    tree = _build_tree(
        [
            _v3_tool_span(name='transfer_funds', span_id=3, args='{"amount": 100}', start_offset=0.3),
            _v2_tool_span(name='verify_identity', span_id=1, args='{"user": "alice"}', start_offset=0.1),
            _v3_tool_span(name='check_balance', span_id=2, args='{}', start_offset=0.2),
        ]
    )
    policy = ToolPolicyCompliance(require_before={'transfer_funds': ['verify_identity', 'check_balance']})
    assert policy.evaluate(_ctx(tree=tree)) == EvaluationReason(value=True)


def test_require_before_rejects_missing_and_out_of_order_prerequisites():
    tree = _build_tree(
        [
            _v3_tool_span(name='transfer_funds', span_id=1, args='{}', start_offset=0.0),
            _v3_tool_span(name='verify_identity', span_id=2, args='{}', start_offset=0.1),
        ]
    )
    policy = ToolPolicyCompliance(require_before={'transfer_funds': ['verify_identity', 'check_balance']})
    result = policy.evaluate(_ctx(tree=tree))
    assert result.value is False
    assert result.reason is not None
    assert "call #1 to 'transfer_funds' missing prerequisite(s): 'verify_identity', 'check_balance'" in result.reason


def test_require_before_does_not_accept_failed_prerequisite():
    # Security invariant: even though include_failed=True audits failed calls,
    # a failed `verify_identity` must NEVER satisfy `require_before`.
    tree = _build_tree(
        [
            _v3_tool_span(name='verify_identity', span_id=1, args='{}', start_offset=0.0, status='error'),
            _v3_tool_span(name='transfer_funds', span_id=2, args='{}', start_offset=0.1),
        ]
    )
    policy = ToolPolicyCompliance(require_before={'transfer_funds': ['verify_identity']})
    result = policy.evaluate(_ctx(tree=tree))
    assert result.value is False
    assert result.reason is not None
    assert "call #2 to 'transfer_funds' missing prerequisite(s): 'verify_identity'" in result.reason


def test_include_failed_controls_auditing_of_failed_attempts():
    tree = _build_tree(
        [
            _v3_tool_span(name='delete_user', span_id=1, args='{"id": 1}', start_offset=0.0, status='error'),
        ]
    )
    # Default include_failed=True catches the failed injection attempt.
    default_res = ToolPolicyCompliance(forbidden_tools=['delete_user']).evaluate(_ctx(tree=tree))
    assert default_res.value is False
    assert default_res.reason is not None
    assert "call #1 invoked forbidden tool 'delete_user'" in default_res.reason

    # Opting out ignores failed attempts.
    opt_out_res = ToolPolicyCompliance(forbidden_tools=['delete_user'], include_failed=False).evaluate(_ctx(tree=tree))
    assert opt_out_res == EvaluationReason(value=True)


# ---------------------------------------------------------------------------
# forbidden_arg_patterns & fail-closed JSON handling
# ---------------------------------------------------------------------------


def test_forbidden_arg_patterns_matches_strings_and_structured_values():
    tree = _build_tree(
        [
            _v2_tool_span(name='fetch_url', span_id=1, args='{"url": "https://example.com"}', start_offset=0.0),
            _v3_tool_span(
                name='fetch_url',
                span_id=2,
                args='{"url": "http://internal.corp.local/admin"}',
                start_offset=0.1,
            ),
            _v3_tool_span(
                name='run_sql',
                span_id=3,
                args='{"params": {"table": "users", "action": "DROP TABLE"}}',
                start_offset=0.2,
            ),
            _v3_tool_span(name='run_sql', span_id=4, args='{"other": "safe"}', start_offset=0.3),
        ]
    )
    policy = ToolPolicyCompliance(
        forbidden_arg_patterns={
            'fetch_url': {'url': r'(internal\.corp\.local|^file://)'},
            'run_sql': {'params': r'(?i)\bDROP\s+TABLE\b'},
            'unused_tool': {},
        }
    )
    result = policy.evaluate(_ctx(tree=tree))
    assert result.value is False
    assert result.reason is not None
    assert "call #2 to 'fetch_url' argument 'url' matched forbidden pattern" in result.reason
    assert "call #3 to 'run_sql' argument 'params' matched forbidden pattern" in result.reason
    assert 'call #1' not in result.reason
    assert 'call #4' not in result.reason


@pytest.mark.parametrize(
    ('args', 'expected_substring'),
    [
        pytest.param(None, 'include_content', id='missing-args'),
        pytest.param('not-json', 'could not be parsed as JSON', id='invalid-json'),
        pytest.param('{"url": ' + '1' * 5_000 + '}', 'could not be parsed as JSON', id='oversized-int'),
        pytest.param('[1, 2, 3]', 'are not a JSON object', id='non-object-json'),
    ],
)
def test_forbidden_arg_patterns_fail_closed_on_uninspectable_args(args: str | None, expected_substring: str):
    tree = _build_tree([_v3_tool_span(name='fetch_url', span_id=1, args=args, start_offset=0.0)])
    policy = ToolPolicyCompliance(forbidden_arg_patterns={'fetch_url': {'url': r'internal\.corp\.local'}})
    result = policy.evaluate(_ctx(tree=tree))
    assert result.value is False
    assert result.reason is not None
    assert expected_substring in result.reason


def test_forbidden_arg_patterns_deeply_nested_json_does_not_raise():
    tree = _build_tree(
        [_v3_tool_span(name='fetch_url', span_id=1, args='[' * 100_000 + ']' * 100_000, start_offset=0.0)]
    )
    policy = ToolPolicyCompliance(forbidden_arg_patterns={'fetch_url': {'url': r'evil'}})
    result = policy.evaluate(_ctx(tree=tree))
    assert result.value is False
    assert result.reason is not None
    assert 'could not be parsed as JSON' in result.reason or 'are not a JSON object' in result.reason


# ---------------------------------------------------------------------------
# Collect-all multi-rule audit
# ---------------------------------------------------------------------------


def test_collect_all_reports_every_violation_together():
    tree = _build_tree(
        [
            _v3_tool_span(name='transfer_funds', span_id=1, args='{"to": "attacker"}', start_offset=0.0),
            _v3_tool_span(name='send_email', span_id=2, args='{"body": "secret"}', start_offset=0.1),
            _v3_tool_span(name='send_email', span_id=3, args='{"body": "spam"}', start_offset=0.2),
            _v3_tool_span(name='delete_user', span_id=4, args='{"id": 9}', start_offset=0.3),
        ]
    )
    policy = ToolPolicyCompliance(
        forbidden_tools=['delete_user'],
        max_calls_per_tool={'send_email': 1},
        require_before={'transfer_funds': ['verify_identity']},
        forbidden_arg_patterns={'send_email': {'body': r'secret'}},
    )
    result = policy.evaluate(_ctx(tree=tree))
    assert result.value is False
    assert result.reason is not None
    assert "call #4 invoked forbidden tool 'delete_user'" in result.reason
    assert "tool 'send_email' called 2 time(s), max allowed is 1" in result.reason
    assert "call #1 to 'transfer_funds' missing prerequisite(s): 'verify_identity'" in result.reason
    assert "call #2 to 'send_email' argument 'body' matched forbidden pattern 'secret'" in result.reason


# ---------------------------------------------------------------------------
# Dataset YAML/JSON serialization round-trip
# ---------------------------------------------------------------------------


@pytest.mark.parametrize('fmt', ['yaml', 'json'])
def test_dataset_file_round_trip(tmp_path: Path, fmt: Literal['yaml', 'json']):
    evaluator = ToolPolicyCompliance(
        forbidden_tools=['delete_user'],
        max_calls_per_tool={'send_email': 1},
        require_before={'transfer_funds': ['verify_identity']},
        forbidden_arg_patterns={'fetch_url': {'url': r'^file://'}},
        include_failed=False,
        evaluation_name='prod_guardrail',
    )
    dataset = Dataset[str, str, str](
        name='policy-round-trip',
        cases=[Case(inputs='hi', evaluators=[evaluator])],
        evaluators=[evaluator],
    )
    path = tmp_path / f'dataset.{fmt}'
    dataset.to_file(path)
    loaded = Dataset[str, str, str].from_file(path)

    assert loaded.evaluators == [evaluator]
    assert loaded.cases[0].evaluators == [evaluator]
    # Verify compiled patterns were rebuilt during deserialization.
    reloaded = loaded.evaluators[0]
    assert isinstance(reloaded, ToolPolicyCompliance)
    tree = _build_tree(
        [_v3_tool_span(name='fetch_url', span_id=1, args='{"url": "file:///restricted/config.ini"}', start_offset=0.0)]
    )
    assert reloaded.evaluate(_ctx(tree=tree)).value is False


# ---------------------------------------------------------------------------
# End-to-end offline Agent + Dataset integration (sync & async)
# ---------------------------------------------------------------------------


def _build_demo_agent(model: FunctionModel | TestModel) -> Agent[None, str]:
    agent = Agent(model, capabilities=[Instrumentation()])

    @agent.tool_plain
    def verify_identity(user: str) -> str:
        if user == 'bad':
            raise ModelRetry('verification failed')
        return f'verified:{user}'

    @agent.tool_plain
    def transfer_funds(amount: int) -> str:
        return f'transferred:{amount}'

    @agent.tool_plain
    def fetch_url(url: str) -> str:
        return f'fetched:{url}'

    return agent


@needs_logfire
async def test_end_to_end_async_dataset_with_function_model(capfire: CaptureLogfire):
    def safe_script(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(parts=[ToolCallPart('verify_identity', {'user': 'alice'})])
        if len(messages) == 3:
            return ModelResponse(parts=[ToolCallPart('transfer_funds', {'amount': 50})])
        return ModelResponse(parts=[TextPart('done')])

    def unsafe_script(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(
                parts=[
                    ToolCallPart('transfer_funds', {'amount': 999}),
                    ToolCallPart('fetch_url', {'url': 'http://internal.corp.local/admin'}),
                ]
            )
        return ModelResponse(parts=[TextPart('done')])

    safe_agent = _build_demo_agent(FunctionModel(safe_script))
    unsafe_agent = _build_demo_agent(FunctionModel(unsafe_script))

    async def run_task(mode: str) -> str:
        agent = safe_agent if mode == 'safe' else unsafe_agent
        result = await agent.run(mode)
        return result.output

    policy = ToolPolicyCompliance(
        require_before={'transfer_funds': ['verify_identity']},
        forbidden_arg_patterns={'fetch_url': {'url': r'internal\.corp\.local'}},
    )
    dataset = Dataset[str, str, Any](
        name='async-tool-policy',
        cases=[Case(name='safe_flow', inputs='safe'), Case(name='unsafe_flow', inputs='unsafe')],
        evaluators=[policy],
    )
    report = await dataset.evaluate(run_task, progress=False)
    by_name = {case.name: case for case in report.cases}

    assert by_name['safe_flow'].assertions['ToolPolicyCompliance'].value is True
    unsafe_assertion = by_name['unsafe_flow'].assertions['ToolPolicyCompliance']
    assert unsafe_assertion.value is False
    assert unsafe_assertion.reason is not None
    assert "missing prerequisite(s): 'verify_identity'" in unsafe_assertion.reason
    assert "argument 'url' matched forbidden pattern" in unsafe_assertion.reason


@needs_logfire
def test_end_to_end_sync_dataset_with_test_model(capfire: CaptureLogfire):
    agent = _build_demo_agent(TestModel(call_tools=['fetch_url']))

    def run_sync_task(prompt: str) -> str:
        return agent.run_sync(prompt).output

    dataset = Dataset[str, str, Any](
        name='sync-tool-policy',
        cases=[Case(name='sync_case', inputs='hello')],
        evaluators=[ToolPolicyCompliance(forbidden_tools=['transfer_funds'], max_calls_per_tool={'fetch_url': 1})],
    )
    report = dataset.evaluate_sync(run_sync_task, progress=False)
    assert report.cases[0].assertions['ToolPolicyCompliance'].value is True
