"""Tests for the agentic span-based evaluators.

These are unit tests over hand-built `SpanNode`/`SpanTree` fixtures rather than
VCR/public-API agent runs: the evaluators' contract is defined in terms of the
instrumentation span shapes (v2 and v3+), and building the spans directly lets
us pin both naming schemes — plus malformed/edge-case spans a live run can't
reliably produce — deterministically and without model access. All assertions
go through the public evaluator API.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from ..conftest import try_import

with try_import() as imports_successful:
    from pydantic_evals.evaluators import (
        ArgumentCorrectness,
        EvaluationReason,
        EvaluatorContext,
        MaxModelRequests,
        MaxToolCalls,
        ToolCorrectness,
        TrajectoryMatch,
    )
    from pydantic_evals.otel import SpanTreeRecordingError
    from pydantic_evals.otel.span_tree import SpanNode, SpanStatus, SpanTree


pytestmark = [pytest.mark.skipif(not imports_successful(), reason='pydantic-evals not installed'), pytest.mark.anyio]


_EPOCH = datetime(2025, 1, 1, tzinfo=timezone.utc)


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
    """Build a `SpanNode` directly for test fixtures.

    Using explicit IDs and start offsets keeps the tree deterministic without
    requiring a live agent run or an OTel SDK.
    """
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
    """Build a v2-style tool call span (`running tool` with `tool_arguments`)."""
    attrs: dict[str, Any] = {
        'gen_ai.tool.name': name,
        'logfire.msg': f'running tool: {name}',
    }
    if args is not None:
        attrs['tool_arguments'] = args
    return _make_span(
        name='running tool',
        span_id=span_id,
        attributes=attrs,
        start_offset=start_offset,
    )


def _v3_tool_span(*, name: str, span_id: int, args: str | None, start_offset: float) -> SpanNode:
    """Build a v3+-style tool call span (`execute_tool {name}` with `gen_ai.tool.call.arguments`)."""
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
    )


def _v2_output_function_span(*, name: str, span_id: int, start_offset: float) -> SpanNode:
    return _make_span(
        name='running output function',
        span_id=span_id,
        attributes={
            'gen_ai.tool.name': name,
            'logfire.msg': f'running output function: {name}',
        },
        start_offset=start_offset,
    )


def _v3_output_function_span(*, name: str, span_id: int, start_offset: float) -> SpanNode:
    return _make_span(
        name=f'execute_tool {name}',
        span_id=span_id,
        attributes={
            'gen_ai.tool.name': name,
            'logfire.msg': f'running output function: {name}',
        },
        start_offset=start_offset,
    )


def _failed_tool_span(*, name: str, span_id: int, args: str | None, start_offset: float) -> SpanNode:
    """A tool span whose attempt ended in an error (raised exception or `ModelRetry`)."""
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
        status='error',
    )


def _deferred_tool_span(*, name: str, span_id: int, start_offset: float) -> SpanNode:
    """A tool span whose call was deferred (`ApprovalRequired`/`CallDeferred`), not executed."""
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


def _model_request_span(*, span_id: int, start_offset: float) -> SpanNode:
    return _make_span(
        name='chat',
        span_id=span_id,
        attributes={
            'gen_ai.request.model': 'gpt-5',
            'gen_ai.operation.name': 'chat',
        },
        start_offset=start_offset,
    )


def _build_tree(nodes: list[SpanNode]) -> SpanTree:
    tree = SpanTree()
    tree.add_spans(nodes)
    return tree


def _ctx(
    *,
    tree: SpanTree | SpanTreeRecordingError,
    metrics: dict[str, int | float] | None = None,
) -> EvaluatorContext[Any, Any, Any]:
    return EvaluatorContext(
        name='test',
        inputs={},
        metadata=None,
        expected_output=None,
        output=None,
        duration=0.0,
        _span_tree=tree,
        attributes={},
        metrics=metrics or {},
    )


# ---------------------------------------------------------------------------
# Tool-call span detection: v2 vs v3+, output functions, deferrals, ordering
# ---------------------------------------------------------------------------


def test_tool_spans_v2_and_v3_both_detected_in_start_order():
    # Insert out of order; the trajectory must reflect start timestamps, and
    # both v2 and v3+ span shapes must be detected.
    tree = _build_tree(
        [
            _v3_tool_span(name='rerank', span_id=2, args='{"top_k": 3}', start_offset=0.1),
            _v2_tool_span(name='search', span_id=1, args='{"q": "cats"}', start_offset=0.0),
        ]
    )
    result = TrajectoryMatch(expected_trajectory=['search', 'rerank'], order='exact').evaluate(_ctx(tree=tree))
    assert result.value == 1.0


def test_tool_spans_ignore_output_function_spans():
    # With allow_extra defaulting to False, passing proves the output-function
    # spans were not counted as tool calls.
    tree = _build_tree(
        [
            _v2_tool_span(name='search', span_id=1, args='{}', start_offset=0.0),
            _v2_output_function_span(name='format_answer', span_id=2, start_offset=0.1),
            _v3_output_function_span(name='final_answer', span_id=3, start_offset=0.2),
        ]
    )
    result = ToolCorrectness(expected_tools=['search']).evaluate(_ctx(tree=tree))
    assert result == EvaluationReason(value=True)


def test_tool_spans_ignore_deferred_tool_calls():
    # A deferred call (ApprovalRequired/CallDeferred) never executed, so it
    # must not count as a tool call.
    tree = _build_tree(
        [
            _v2_tool_span(name='search', span_id=1, args='{}', start_offset=0.0),
            _deferred_tool_span(name='delete_account', span_id=2, start_offset=0.1),
        ]
    )
    result = ToolCorrectness(expected_tools=['search']).evaluate(_ctx(tree=tree))
    assert result == EvaluationReason(value=True)
    # ...and the deferred call also doesn't satisfy an expectation for it.
    result = ToolCorrectness(expected_tools=['search', 'delete_account']).evaluate(_ctx(tree=tree))
    assert result.value is False
    assert result.reason is not None
    assert "missing tools: 'delete_account' (x1)" in result.reason


def test_failed_attempts_excluded_by_default():
    # A tool raised ModelRetry with bad args, then the model retried
    # successfully: one errored span and one successful span. By default only
    # the successful call counts, so the trajectory is the logical one.
    tree = _build_tree(
        [
            _failed_tool_span(name='search', span_id=1, args='{"q": "bad"}', start_offset=0.0),
            _v3_tool_span(name='search', span_id=2, args='{"q": "good"}', start_offset=0.1),
        ]
    )
    assert ToolCorrectness(expected_tools=['search']).evaluate(_ctx(tree=tree)) == EvaluationReason(value=True)
    result = TrajectoryMatch(expected_trajectory=['search'], order='exact').evaluate(_ctx(tree=tree))
    assert result.value == 1.0
    # ArgumentCorrectness picks the successful attempt's arguments.
    args_result = ArgumentCorrectness(tool_name='search', expected_arguments={'q': 'good'}).evaluate(_ctx(tree=tree))
    assert args_result.value is True


def test_failed_attempts_included_when_requested():
    tree = _build_tree(
        [
            _failed_tool_span(name='search', span_id=1, args='{"q": "bad"}', start_offset=0.0),
            _v3_tool_span(name='search', span_id=2, args='{"q": "good"}', start_offset=0.1),
        ]
    )
    # Both attempts count: expected multiset of one 'search' now has an extra.
    result = ToolCorrectness(expected_tools=['search'], include_failed=True).evaluate(_ctx(tree=tree))
    assert result.value is False
    assert result.reason is not None
    assert "unexpected tools: 'search' (x1)" in result.reason
    # The trajectory likewise contains both attempts.
    trajectory_result = TrajectoryMatch(
        expected_trajectory=['search', 'search'],
        order='exact',
        include_failed=True,
    ).evaluate(_ctx(tree=tree))
    assert trajectory_result.value == 1.0
    # The first occurrence is now the failed attempt.
    args_result = ArgumentCorrectness(
        tool_name='search',
        expected_arguments={'q': 'bad'},
        include_failed=True,
    ).evaluate(_ctx(tree=tree))
    assert args_result.value is True


def test_max_tool_calls_counts_failed_attempts_by_default():
    tree = _build_tree(
        [
            _failed_tool_span(name='search', span_id=1, args=None, start_offset=0.0),
            _v3_tool_span(name='search', span_id=2, args='{}', start_offset=0.1),
        ]
    )
    # Both attempts consume budget by default...
    result = MaxToolCalls(max_calls=1).evaluate(_ctx(tree=tree))
    assert result.value is False
    assert result.reason is not None
    assert '2 tool call(s)' in result.reason
    # ...but only successful calls count when include_failed=False.
    result = MaxToolCalls(max_calls=1, include_failed=False).evaluate(_ctx(tree=tree))
    assert result.value is True


def test_tool_spans_ignore_unrelated_spans():
    tree = _build_tree(
        [
            _make_span(name='logfire', span_id=1),
            _make_span(name='chat', span_id=2, attributes={'gen_ai.request.model': 'gpt-5'}),
            # A span with `gen_ai.tool.name` but an unrelated name: should
            # not be treated as a tool call (e.g. an unknown/future span type
            # that happens to carry the attribute).
            _make_span(
                name='some_custom_span',
                span_id=3,
                attributes={'gen_ai.tool.name': 'search'},
                start_offset=0.3,
            ),
            _v2_tool_span(name='search', span_id=4, args='{}', start_offset=0.5),
        ]
    )
    result = ToolCorrectness(expected_tools=['search']).evaluate(_ctx(tree=tree))
    assert result == EvaluationReason(value=True)


def test_tool_spans_skip_non_string_tool_name():
    # Defensive branch: if `gen_ai.tool.name` is somehow set to a non-string
    # value, the span should be skipped rather than crashing. Expecting no
    # tools (with the strict default) passes only if the span was skipped.
    tree = _build_tree(
        [
            _make_span(
                name='running tool',
                span_id=1,
                attributes={'gen_ai.tool.name': 123, 'logfire.msg': 'running tool: 123'},
            ),
        ]
    )
    result = ToolCorrectness(expected_tools=[]).evaluate(_ctx(tree=tree))
    assert result == EvaluationReason(value=True)


def test_tool_spans_v2_output_function_with_no_logfire_msg():
    """A v2 output-function span (span name `running output function`) is excluded.

    This exercises the code path that skips v2 output-function spans by name
    alone, independent of the `logfire.msg` attribute.
    """
    tree = _build_tree(
        [
            _make_span(
                name='running output function',
                span_id=1,
                attributes={'gen_ai.tool.name': 'format_answer'},
                start_offset=0.0,
            ),
            _v2_tool_span(name='search', span_id=2, args='{}', start_offset=0.1),
        ]
    )
    result = ToolCorrectness(expected_tools=['search']).evaluate(_ctx(tree=tree))
    assert result == EvaluationReason(value=True)


# ---------------------------------------------------------------------------
# ToolCorrectness
# ---------------------------------------------------------------------------


def test_tool_correctness_happy_path():
    tree = _build_tree(
        [
            _v2_tool_span(name='search', span_id=1, args='{}', start_offset=0.0),
            _v3_tool_span(name='format', span_id=2, args='{}', start_offset=0.1),
        ]
    )
    evaluator = ToolCorrectness(expected_tools=['search', 'format'])
    assert evaluator.evaluate(_ctx(tree=tree)) == EvaluationReason(value=True)


def test_tool_correctness_multiset_requires_duplicates():
    tree = _build_tree(
        [
            _v2_tool_span(name='search', span_id=1, args='{}', start_offset=0.0),
        ]
    )
    # expected requires search twice; only one call => fail
    evaluator = ToolCorrectness(expected_tools=['search', 'search'])
    result = evaluator.evaluate(_ctx(tree=tree))
    assert result.value is False
    assert result.reason is not None
    assert "'search' (x1)" in result.reason


def test_tool_correctness_missing_tool():
    tree = _build_tree([_v2_tool_span(name='search', span_id=1, args='{}', start_offset=0.0)])
    evaluator = ToolCorrectness(expected_tools=['search', 'format'])
    result = evaluator.evaluate(_ctx(tree=tree))
    assert result.value is False
    assert result.reason is not None
    assert "missing tools: 'format' (x1)" in result.reason


def test_tool_correctness_extra_fails_by_default():
    tree = _build_tree(
        [
            _v2_tool_span(name='search', span_id=1, args='{}', start_offset=0.0),
            _v2_tool_span(name='extra', span_id=2, args='{}', start_offset=0.1),
        ]
    )
    evaluator = ToolCorrectness(expected_tools=['search'])
    result = evaluator.evaluate(_ctx(tree=tree))
    assert result.value is False
    assert result.reason is not None
    assert "unexpected tools: 'extra' (x1)" in result.reason


def test_tool_correctness_allow_extra_permits_extras():
    tree = _build_tree(
        [
            _v2_tool_span(name='search', span_id=1, args='{}', start_offset=0.0),
            _v2_tool_span(name='extra', span_id=2, args='{}', start_offset=0.1),
        ]
    )
    evaluator = ToolCorrectness(expected_tools=['search'], allow_extra=True)
    assert evaluator.evaluate(_ctx(tree=tree)) == EvaluationReason(value=True)


def test_tool_correctness_both_missing_and_extra_reported():
    tree = _build_tree(
        [
            _v2_tool_span(name='unexpected', span_id=1, args='{}', start_offset=0.0),
        ]
    )
    evaluator = ToolCorrectness(expected_tools=['wanted'])
    result = evaluator.evaluate(_ctx(tree=tree))
    assert result.value is False
    assert result.reason is not None
    assert 'missing tools:' in result.reason
    assert 'unexpected tools:' in result.reason


def test_tool_correctness_no_span_tree():
    ctx = _ctx(tree=SpanTreeRecordingError('spans were not recorded'))
    result = ToolCorrectness(expected_tools=['x']).evaluate(ctx)
    assert result.value is False
    assert result.reason is not None
    assert 'logfire' in result.reason


# ---------------------------------------------------------------------------
# TrajectoryMatch
# ---------------------------------------------------------------------------


def test_trajectory_match_exact_pass():
    tree = _build_tree(
        [
            _v2_tool_span(name='a', span_id=1, args='{}', start_offset=0.0),
            _v2_tool_span(name='b', span_id=2, args='{}', start_offset=0.1),
        ]
    )
    result = TrajectoryMatch(expected_trajectory=['a', 'b'], order='exact').evaluate(_ctx(tree=tree))
    assert result.value == 1.0


def test_trajectory_match_exact_fail():
    tree = _build_tree(
        [
            _v2_tool_span(name='a', span_id=1, args='{}', start_offset=0.0),
            _v2_tool_span(name='c', span_id=2, args='{}', start_offset=0.1),
        ]
    )
    result = TrajectoryMatch(expected_trajectory=['a', 'b'], order='exact').evaluate(_ctx(tree=tree))
    assert result.value == 0.0
    assert result.reason is not None
    assert 'does not equal' in result.reason


def test_trajectory_match_in_order_perfect():
    tree = _build_tree(
        [
            _v2_tool_span(name='a', span_id=1, args='{}', start_offset=0.0),
            _v2_tool_span(name='b', span_id=2, args='{}', start_offset=0.1),
        ]
    )
    result = TrajectoryMatch(expected_trajectory=['a', 'b'], order='in_order').evaluate(_ctx(tree=tree))
    assert result.value == 1.0
    assert result.reason is not None
    assert 'LCS=2' in result.reason
    assert 'F1=1.000' in result.reason


def test_trajectory_match_in_order_partial():
    # Hand calculation:
    #   actual   = ['a', 'x', 'b']
    #   expected = ['a', 'b', 'c']
    #   LCS(['a','x','b'], ['a','b','c']) = 2 (subsequence 'a','b')
    #   precision = 2/3 ≈ 0.6667
    #   recall    = 2/3 ≈ 0.6667
    #   F1        = 2 * 0.6667 * 0.6667 / (0.6667 + 0.6667) = 0.6667
    tree = _build_tree(
        [
            _v2_tool_span(name='a', span_id=1, args='{}', start_offset=0.0),
            _v2_tool_span(name='x', span_id=2, args='{}', start_offset=0.1),
            _v2_tool_span(name='b', span_id=3, args='{}', start_offset=0.2),
        ]
    )
    result = TrajectoryMatch(expected_trajectory=['a', 'b', 'c'], order='in_order').evaluate(_ctx(tree=tree))
    assert isinstance(result.value, float)
    assert abs(result.value - 2 / 3) < 1e-9
    assert result.reason is not None
    assert 'LCS=2' in result.reason
    assert '2/3' in result.reason
    assert 'F1=0.667' in result.reason


def test_trajectory_match_in_order_second_example():
    # Hand calculation:
    #   actual   = ['search', 'search', 'format', 'format']
    #   expected = ['search', 'format']
    #   LCS = 2
    #   precision = 2/4 = 0.5
    #   recall    = 2/2 = 1.0
    #   F1        = 2 * 0.5 * 1.0 / (0.5 + 1.0) = 2/3 ≈ 0.6667
    tree = _build_tree(
        [
            _v2_tool_span(name='search', span_id=1, args='{}', start_offset=0.0),
            _v2_tool_span(name='search', span_id=2, args='{}', start_offset=0.1),
            _v2_tool_span(name='format', span_id=3, args='{}', start_offset=0.2),
            _v2_tool_span(name='format', span_id=4, args='{}', start_offset=0.3),
        ]
    )
    result = TrajectoryMatch(expected_trajectory=['search', 'format'], order='in_order').evaluate(_ctx(tree=tree))
    assert isinstance(result.value, float)
    assert abs(result.value - 2 / 3) < 1e-9
    assert result.reason is not None
    assert 'precision=2/4=0.500' in result.reason
    assert 'recall=2/2=1.000' in result.reason


def test_trajectory_match_in_order_interleaved_extras():
    # LCS must skip over interleaved extras:
    #   actual   = ['a', 'x', 'b', 'y', 'c']
    #   expected = ['a', 'b', 'c']
    #   LCS = 3, precision = 3/5, recall = 3/3 = 1.0
    #   F1 = 2 * (3/5) * 1.0 / (3/5 + 1.0) = 0.75
    tree = _build_tree(
        [
            _v2_tool_span(name='a', span_id=1, args='{}', start_offset=0.0),
            _v2_tool_span(name='x', span_id=2, args='{}', start_offset=0.1),
            _v2_tool_span(name='b', span_id=3, args='{}', start_offset=0.2),
            _v2_tool_span(name='y', span_id=4, args='{}', start_offset=0.3),
            _v2_tool_span(name='c', span_id=5, args='{}', start_offset=0.4),
        ]
    )
    result = TrajectoryMatch(expected_trajectory=['a', 'b', 'c'], order='in_order').evaluate(_ctx(tree=tree))
    assert isinstance(result.value, float)
    assert abs(result.value - 0.75) < 1e-9
    assert result.reason is not None
    assert 'LCS=3' in result.reason


def test_trajectory_match_in_order_no_match():
    tree = _build_tree([_v2_tool_span(name='x', span_id=1, args='{}', start_offset=0.0)])
    result = TrajectoryMatch(expected_trajectory=['y'], order='in_order').evaluate(_ctx(tree=tree))
    assert result.value == 0.0


def test_trajectory_match_in_order_both_empty():
    tree = _build_tree([])
    result = TrajectoryMatch(expected_trajectory=[], order='in_order').evaluate(_ctx(tree=tree))
    assert result.value == 1.0


def test_trajectory_match_in_order_actual_empty_expected_nonempty():
    tree = _build_tree([])
    result = TrajectoryMatch(expected_trajectory=['a'], order='in_order').evaluate(_ctx(tree=tree))
    assert result.value == 0.0


def test_trajectory_match_in_order_expected_empty_actual_nonempty():
    tree = _build_tree([_v2_tool_span(name='a', span_id=1, args='{}', start_offset=0.0)])
    result = TrajectoryMatch(expected_trajectory=[], order='in_order').evaluate(_ctx(tree=tree))
    assert result.value == 0.0


def test_trajectory_match_any_order_full_overlap():
    tree = _build_tree(
        [
            _v2_tool_span(name='b', span_id=1, args='{}', start_offset=0.0),
            _v2_tool_span(name='a', span_id=2, args='{}', start_offset=0.1),
        ]
    )
    result = TrajectoryMatch(expected_trajectory=['a', 'b'], order='any_order').evaluate(_ctx(tree=tree))
    assert result.value == 1.0


def test_trajectory_match_any_order_partial_overlap():
    # Hand calculation:
    #   actual   = ['a', 'x']
    #   expected = ['a', 'b', 'c']
    #   overlap  = 1 ('a')
    #   precision = 1/2 = 0.5
    #   recall    = 1/3 ≈ 0.3333
    #   F1        = 2 * 0.5 * (1/3) / (0.5 + 1/3) = 0.4
    tree = _build_tree(
        [
            _v2_tool_span(name='a', span_id=1, args='{}', start_offset=0.0),
            _v2_tool_span(name='x', span_id=2, args='{}', start_offset=0.1),
        ]
    )
    result = TrajectoryMatch(expected_trajectory=['a', 'b', 'c'], order='any_order').evaluate(_ctx(tree=tree))
    assert isinstance(result.value, float)
    assert abs(result.value - 0.4) < 1e-9
    assert result.reason is not None
    assert 'overlap=1' in result.reason
    assert 'F1=0.400' in result.reason


def test_trajectory_match_any_order_extras_reduce_score():
    # Extra calls reduce precision even though order is ignored:
    #   actual   = ['a', 'b', 'x', 'y']
    #   expected = ['a', 'b']
    #   overlap = 2, precision = 2/4 = 0.5, recall = 2/2 = 1.0, F1 = 2/3
    tree = _build_tree(
        [
            _v2_tool_span(name='a', span_id=1, args='{}', start_offset=0.0),
            _v2_tool_span(name='b', span_id=2, args='{}', start_offset=0.1),
            _v2_tool_span(name='x', span_id=3, args='{}', start_offset=0.2),
            _v2_tool_span(name='y', span_id=4, args='{}', start_offset=0.3),
        ]
    )
    result = TrajectoryMatch(expected_trajectory=['a', 'b'], order='any_order').evaluate(_ctx(tree=tree))
    assert isinstance(result.value, float)
    assert abs(result.value - 2 / 3) < 1e-9


def test_trajectory_match_any_order_multiset_semantics():
    # actual has one 'a'; expected requires two 'a's. Overlap counts each
    # 'a' only once because it's a multiset intersection:
    #   overlap = 1, precision = 1/1 = 1.0, recall = 1/2 = 0.5, F1 = 2/3
    tree = _build_tree([_v2_tool_span(name='a', span_id=1, args='{}', start_offset=0.0)])
    result = TrajectoryMatch(expected_trajectory=['a', 'a'], order='any_order').evaluate(_ctx(tree=tree))
    assert isinstance(result.value, float)
    assert abs(result.value - 2 / 3) < 1e-9


def test_trajectory_match_any_order_both_empty():
    tree = _build_tree([])
    result = TrajectoryMatch(expected_trajectory=[], order='any_order').evaluate(_ctx(tree=tree))
    assert result.value == 1.0


def test_trajectory_match_any_order_expected_empty_actual_nonempty():
    tree = _build_tree([_v2_tool_span(name='a', span_id=1, args='{}', start_offset=0.0)])
    result = TrajectoryMatch(expected_trajectory=[], order='any_order').evaluate(_ctx(tree=tree))
    assert result.value == 0.0


def test_trajectory_match_no_span_tree_returns_float_zero():
    # The degraded path must stay a score (float), not become an assertion
    # (bool), so span-less cases don't silently vanish from score averages.
    result = TrajectoryMatch(expected_trajectory=['a']).evaluate(_ctx(tree=SpanTreeRecordingError('x')))
    assert type(result.value) is float
    assert result.value == 0.0
    assert result.reason is not None
    assert 'logfire' in result.reason


# ---------------------------------------------------------------------------
# ArgumentCorrectness
# ---------------------------------------------------------------------------


def test_argument_correctness_subset_pass():
    tree = _build_tree(
        [
            _v3_tool_span(
                name='search',
                span_id=1,
                args='{"q": "cats", "limit": 5}',
                start_offset=0.0,
            ),
        ]
    )
    evaluator = ArgumentCorrectness(
        tool_name='search',
        expected_arguments={'q': 'cats'},
    )
    assert evaluator.evaluate(_ctx(tree=tree)) == EvaluationReason(value=True)


def test_argument_correctness_exact_fail_on_extra_keys():
    tree = _build_tree(
        [
            _v3_tool_span(
                name='search',
                span_id=1,
                args='{"q": "cats", "limit": 5}',
                start_offset=0.0,
            ),
        ]
    )
    evaluator = ArgumentCorrectness(
        tool_name='search',
        expected_arguments={'q': 'cats'},
        match_mode='exact',
    )
    result = evaluator.evaluate(_ctx(tree=tree))
    assert result.value is False
    assert result.reason is not None
    assert "unexpected key 'limit'" in result.reason


def test_argument_correctness_value_mismatch():
    tree = _build_tree(
        [
            _v3_tool_span(
                name='search',
                span_id=1,
                args='{"q": "dogs"}',
                start_offset=0.0,
            ),
        ]
    )
    evaluator = ArgumentCorrectness(
        tool_name='search',
        expected_arguments={'q': 'cats'},
    )
    result = evaluator.evaluate(_ctx(tree=tree))
    assert result.value is False
    assert result.reason is not None
    assert "expected 'cats'" in result.reason
    assert "got 'dogs'" in result.reason


def test_argument_correctness_missing_key():
    tree = _build_tree([_v3_tool_span(name='search', span_id=1, args='{"q": "cats"}', start_offset=0.0)])
    evaluator = ArgumentCorrectness(
        tool_name='search',
        expected_arguments={'limit': 5},
    )
    result = evaluator.evaluate(_ctx(tree=tree))
    assert result.value is False
    assert result.reason is not None
    assert "missing key 'limit'" in result.reason


def test_argument_correctness_tool_never_called():
    tree = _build_tree([_v2_tool_span(name='other', span_id=1, args='{}', start_offset=0.0)])
    evaluator = ArgumentCorrectness(tool_name='search', expected_arguments={'q': 'cats'})
    result = evaluator.evaluate(_ctx(tree=tree))
    assert result.value is False
    assert result.reason is not None
    assert "No calls to tool 'search'" in result.reason


def test_argument_correctness_occurrence_first():
    tree = _build_tree(
        [
            _v3_tool_span(name='search', span_id=1, args='{"q": "first"}', start_offset=0.0),
            _v3_tool_span(name='search', span_id=2, args='{"q": "second"}', start_offset=0.1),
        ]
    )
    evaluator = ArgumentCorrectness(
        tool_name='search',
        expected_arguments={'q': 'first'},
        occurrence='first',
    )
    assert evaluator.evaluate(_ctx(tree=tree)).value is True


def test_argument_correctness_occurrence_last():
    tree = _build_tree(
        [
            _v3_tool_span(name='search', span_id=1, args='{"q": "first"}', start_offset=0.0),
            _v3_tool_span(name='search', span_id=2, args='{"q": "second"}', start_offset=0.1),
        ]
    )
    evaluator = ArgumentCorrectness(
        tool_name='search',
        expected_arguments={'q': 'second'},
        occurrence='last',
    )
    assert evaluator.evaluate(_ctx(tree=tree)).value is True


def test_argument_correctness_occurrence_integer_index():
    tree = _build_tree(
        [
            _v3_tool_span(name='search', span_id=1, args='{"q": "a"}', start_offset=0.0),
            _v3_tool_span(name='search', span_id=2, args='{"q": "b"}', start_offset=0.1),
            _v3_tool_span(name='search', span_id=3, args='{"q": "c"}', start_offset=0.2),
        ]
    )
    evaluator = ArgumentCorrectness(
        tool_name='search',
        expected_arguments={'q': 'b'},
        occurrence=1,
    )
    assert evaluator.evaluate(_ctx(tree=tree)).value is True


@pytest.mark.parametrize('occurrence', [5, -1])
def test_argument_correctness_occurrence_out_of_range(occurrence: int):
    tree = _build_tree([_v3_tool_span(name='search', span_id=1, args='{"q": "a"}', start_offset=0.0)])
    evaluator = ArgumentCorrectness(
        tool_name='search',
        expected_arguments={'q': 'a'},
        occurrence=occurrence,
    )
    result = evaluator.evaluate(_ctx(tree=tree))
    assert result.value is False
    assert result.reason is not None
    assert f'occurrence={occurrence!r} does not select any of them' in result.reason
    assert 'negative ints are not supported' in result.reason


@pytest.mark.parametrize('span_builder', [_v2_tool_span, _v3_tool_span])
def test_argument_correctness_include_content_false(span_builder: Any):
    """When `include_content=False`, the arguments string isn't recorded (v2 and v3+ spans)."""
    tree = _build_tree([span_builder(name='search', span_id=1, args=None, start_offset=0.0)])
    evaluator = ArgumentCorrectness(
        tool_name='search',
        expected_arguments={'q': 'cats'},
    )
    result = evaluator.evaluate(_ctx(tree=tree))
    assert result.value is False
    assert result.reason is not None
    assert 'include_content' in result.reason


def test_argument_correctness_invalid_json():
    tree = _build_tree([_v3_tool_span(name='search', span_id=1, args='not-json', start_offset=0.0)])
    evaluator = ArgumentCorrectness(
        tool_name='search',
        expected_arguments={'q': 'cats'},
    )
    result = evaluator.evaluate(_ctx(tree=tree))
    assert result.value is False
    assert result.reason is not None
    assert 'could not be parsed as JSON' in result.reason


def test_argument_correctness_non_object_json():
    tree = _build_tree([_v3_tool_span(name='search', span_id=1, args='[1, 2, 3]', start_offset=0.0)])
    evaluator = ArgumentCorrectness(
        tool_name='search',
        expected_arguments={'q': 'cats'},
    )
    result = evaluator.evaluate(_ctx(tree=tree))
    assert result.value is False
    assert result.reason is not None
    assert 'not a JSON object' in result.reason


def test_argument_correctness_v2_span_also_works():
    tree = _build_tree(
        [
            _v2_tool_span(
                name='search',
                span_id=1,
                args='{"q": "cats"}',
                start_offset=0.0,
            ),
        ]
    )
    evaluator = ArgumentCorrectness(
        tool_name='search',
        expected_arguments={'q': 'cats'},
    )
    assert evaluator.evaluate(_ctx(tree=tree)).value is True


def test_argument_correctness_nested_values_compared_by_equality():
    # Subset matching applies only to top-level keys: a nested-dict expected
    # value must equal the actual value in full.
    tree = _build_tree(
        [
            _v3_tool_span(
                name='search',
                span_id=1,
                args='{"filters": {"status": "open", "priority": "high"}}',
                start_offset=0.0,
            ),
        ]
    )
    evaluator = ArgumentCorrectness(
        tool_name='search',
        expected_arguments={'filters': {'status': 'open'}},
    )
    result = evaluator.evaluate(_ctx(tree=tree))
    assert result.value is False
    assert result.reason is not None
    assert "key 'filters'" in result.reason


def test_argument_correctness_no_span_tree():
    result = ArgumentCorrectness(tool_name='x', expected_arguments={'a': 1}).evaluate(
        _ctx(tree=SpanTreeRecordingError('x'))
    )
    assert result.value is False


# ---------------------------------------------------------------------------
# MaxToolCalls
# ---------------------------------------------------------------------------


def test_max_tool_calls_under_budget():
    tree = _build_tree(
        [
            _v2_tool_span(name='a', span_id=1, args='{}', start_offset=0.0),
            _v2_tool_span(name='b', span_id=2, args='{}', start_offset=0.1),
        ]
    )
    result = MaxToolCalls(max_calls=3).evaluate(_ctx(tree=tree))
    assert result.value is True
    assert result.reason is not None
    assert '2 tool call(s)' in result.reason


def test_max_tool_calls_over_budget():
    tree = _build_tree(
        [
            _v2_tool_span(name='a', span_id=1, args='{}', start_offset=0.0),
            _v2_tool_span(name='b', span_id=2, args='{}', start_offset=0.1),
            _v2_tool_span(name='c', span_id=3, args='{}', start_offset=0.2),
        ]
    )
    result = MaxToolCalls(max_calls=2).evaluate(_ctx(tree=tree))
    assert result.value is False
    assert result.reason is not None
    assert 'budget=2' in result.reason


def test_max_tool_calls_no_span_tree():
    result = MaxToolCalls(max_calls=2).evaluate(_ctx(tree=SpanTreeRecordingError('x')))
    assert result.value is False
    assert result.reason is not None
    assert 'logfire' in result.reason


# ---------------------------------------------------------------------------
# MaxModelRequests
# ---------------------------------------------------------------------------


def test_max_model_requests_from_metrics():
    # Nothing in the tree; metrics provide the count.
    tree = _build_tree([])
    result = MaxModelRequests(max_requests=3).evaluate(_ctx(tree=tree, metrics={'requests': 2}))
    assert result.value is True
    assert result.reason is not None
    assert 'ctx.metrics' in result.reason


def test_max_model_requests_falls_back_to_span_count():
    tree = _build_tree(
        [
            _model_request_span(span_id=1, start_offset=0.0),
            _model_request_span(span_id=2, start_offset=0.1),
            # Not a model request; must not be counted.
            _v2_tool_span(name='search', span_id=3, args='{}', start_offset=0.2),
        ]
    )
    result = MaxModelRequests(max_requests=1).evaluate(_ctx(tree=tree))
    assert result.value is False
    assert result.reason is not None
    assert 'from span tree' in result.reason


def test_max_model_requests_span_count_ignores_non_chat_spans():
    tree = _build_tree(
        [
            _model_request_span(span_id=1, start_offset=0.0),
            # Has the model attribute but is not a chat operation.
            _make_span(
                name='embeddings',
                span_id=2,
                attributes={'gen_ai.request.model': 'text-embedding-3-small', 'gen_ai.operation.name': 'embeddings'},
                start_offset=0.1,
            ),
        ]
    )
    result = MaxModelRequests(max_requests=1).evaluate(_ctx(tree=tree))
    assert result.value is True


def test_max_model_requests_no_span_tree():
    result = MaxModelRequests(max_requests=3).evaluate(_ctx(tree=SpanTreeRecordingError('x')))
    assert result.value is False
    assert result.reason is not None
    assert 'logfire' in result.reason


# ---------------------------------------------------------------------------
# evaluation_name
# ---------------------------------------------------------------------------


def test_evaluation_name_default_and_override():
    evaluators = [
        ToolCorrectness(expected_tools=['x']),
        TrajectoryMatch(expected_trajectory=['x']),
        ArgumentCorrectness(tool_name='x', expected_arguments={}),
        MaxToolCalls(max_calls=1),
        MaxModelRequests(max_requests=1),
    ]
    assert [e.get_default_evaluation_name() for e in evaluators] == [
        'ToolCorrectness',
        'TrajectoryMatch',
        'ArgumentCorrectness',
        'MaxToolCalls',
        'MaxModelRequests',
    ]

    named = [
        ToolCorrectness(expected_tools=['x'], evaluation_name='rag_tools'),
        TrajectoryMatch(expected_trajectory=['x'], evaluation_name='rag_trajectory'),
        ArgumentCorrectness(tool_name='x', expected_arguments={}, evaluation_name='rag_args'),
        MaxToolCalls(max_calls=1, evaluation_name='tool_budget'),
        MaxModelRequests(max_requests=1, evaluation_name='request_budget'),
    ]
    assert [e.get_default_evaluation_name() for e in named] == [
        'rag_tools',
        'rag_trajectory',
        'rag_args',
        'tool_budget',
        'request_budget',
    ]
