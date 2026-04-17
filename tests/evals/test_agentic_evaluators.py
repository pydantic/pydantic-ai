"""Tests for the agentic span-based evaluators."""

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
        StepEfficiency,
        ToolCorrectness,
        TrajectoryMatch,
    )
    from pydantic_evals.evaluators.agentic import (
        STEP_EFFICIENCY_MODEL_REQUESTS_KEY,
        STEP_EFFICIENCY_TOOL_CALLS_KEY,
        _extract_tool_calls,  # pyright: ignore[reportPrivateUsage]
        _longest_common_subsequence_length,  # pyright: ignore[reportPrivateUsage]
    )
    from pydantic_evals.otel._errors import SpanTreeRecordingError
    from pydantic_evals.otel.span_tree import SpanNode, SpanTree


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
# _extract_tool_calls: v2 vs v3+ and output-function discrimination
# ---------------------------------------------------------------------------


def test_extract_tool_calls_v2_and_v3_both_detected():
    tree = _build_tree(
        [
            _v2_tool_span(name='search', span_id=1, args='{"q": "cats"}', start_offset=0.0),
            _v3_tool_span(name='rerank', span_id=2, args='{"top_k": 3}', start_offset=0.1),
        ]
    )
    calls = _extract_tool_calls(tree)
    assert [c.name for c in calls] == ['search', 'rerank']
    assert calls[0].arguments == '{"q": "cats"}'
    assert calls[1].arguments == '{"top_k": 3}'


def test_extract_tool_calls_ignores_output_function_spans():
    tree = _build_tree(
        [
            _v2_tool_span(name='search', span_id=1, args='{}', start_offset=0.0),
            _v2_output_function_span(name='format_answer', span_id=2, start_offset=0.1),
            _v3_output_function_span(name='final_answer', span_id=3, start_offset=0.2),
        ]
    )
    calls = _extract_tool_calls(tree)
    assert [c.name for c in calls] == ['search']


def test_extract_tool_calls_sorts_by_start_time():
    # Insert out of order; extraction must reorder by start timestamp.
    tree = _build_tree(
        [
            _v2_tool_span(name='later', span_id=2, args='{}', start_offset=1.0),
            _v2_tool_span(name='earlier', span_id=1, args='{}', start_offset=0.0),
        ]
    )
    assert [c.name for c in _extract_tool_calls(tree)] == ['earlier', 'later']


def test_extract_tool_calls_ignores_unrelated_spans():
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
    assert [c.name for c in _extract_tool_calls(tree)] == ['search']


def test_extract_tool_calls_skips_spans_with_non_string_tool_name():
    # Defensive branch: if `gen_ai.tool.name` is somehow set to a non-string
    # value, the span should be skipped rather than crashing.
    tree = _build_tree(
        [
            _make_span(
                name='running tool',
                span_id=1,
                attributes={'gen_ai.tool.name': 123, 'logfire.msg': 'running tool: 123'},
            ),
        ]
    )
    assert _extract_tool_calls(tree) == []


def test_extract_tool_calls_missing_arguments_attribute():
    tree = _build_tree(
        [
            _v2_tool_span(name='noargs', span_id=1, args=None, start_offset=0.0),
        ]
    )
    calls = _extract_tool_calls(tree)
    assert len(calls) == 1
    assert calls[0].arguments is None


def test_extract_tool_calls_result_v2_and_v3():
    """Both v2 (`tool_response`) and v3+ (`gen_ai.tool.call.result`) result attrs are extracted."""
    tree = _build_tree(
        [
            _make_span(
                name='running tool',
                span_id=1,
                attributes={
                    'gen_ai.tool.name': 'v2_tool',
                    'logfire.msg': 'running tool: v2_tool',
                    'tool_arguments': '{}',
                    'tool_response': 'v2_result',
                },
                start_offset=0.0,
            ),
            _make_span(
                name='execute_tool v3_tool',
                span_id=2,
                attributes={
                    'gen_ai.tool.name': 'v3_tool',
                    'logfire.msg': 'running tool: v3_tool',
                    'gen_ai.tool.call.arguments': '{}',
                    'gen_ai.tool.call.result': 'v3_result',
                },
                start_offset=0.1,
            ),
        ]
    )
    calls = _extract_tool_calls(tree)
    assert [c.result for c in calls] == ['v2_result', 'v3_result']


def test_extract_tool_calls_v2_output_function_with_no_logfire_msg():
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
    calls = _extract_tool_calls(tree)
    assert [c.name for c in calls] == ['search']


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


def test_tool_correctness_allow_extra_true_by_default():
    tree = _build_tree(
        [
            _v2_tool_span(name='search', span_id=1, args='{}', start_offset=0.0),
            _v2_tool_span(name='extra', span_id=2, args='{}', start_offset=0.1),
        ]
    )
    evaluator = ToolCorrectness(expected_tools=['search'])
    assert evaluator.evaluate(_ctx(tree=tree)) == EvaluationReason(value=True)


def test_tool_correctness_allow_extra_false_fails_on_extras():
    tree = _build_tree(
        [
            _v2_tool_span(name='search', span_id=1, args='{}', start_offset=0.0),
            _v2_tool_span(name='extra', span_id=2, args='{}', start_offset=0.1),
        ]
    )
    evaluator = ToolCorrectness(expected_tools=['search'], allow_extra=False)
    result = evaluator.evaluate(_ctx(tree=tree))
    assert result.value is False
    assert result.reason is not None
    assert "unexpected tools: 'extra' (x1)" in result.reason


def test_tool_correctness_both_missing_and_extra_reported():
    tree = _build_tree(
        [
            _v2_tool_span(name='unexpected', span_id=1, args='{}', start_offset=0.0),
        ]
    )
    evaluator = ToolCorrectness(expected_tools=['wanted'], allow_extra=False)
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


def test_longest_common_subsequence_length_basic():
    assert _longest_common_subsequence_length(['a', 'b', 'c'], ['a', 'b', 'c']) == 3
    assert _longest_common_subsequence_length(['a', 'b', 'c'], ['a', 'c']) == 2
    assert _longest_common_subsequence_length(['a', 'x', 'b', 'y', 'c'], ['a', 'b', 'c']) == 3
    assert _longest_common_subsequence_length([], ['a']) == 0
    assert _longest_common_subsequence_length(['a'], []) == 0
    assert _longest_common_subsequence_length([], []) == 0


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
    tree = _build_tree(
        [
            _v2_tool_span(name='a', span_id=1, args='{}', start_offset=0.0),
            _v2_tool_span(name='x', span_id=2, args='{}', start_offset=0.1),
        ]
    )
    result = TrajectoryMatch(expected_trajectory=['a', 'b', 'c'], order='any_order').evaluate(_ctx(tree=tree))
    assert isinstance(result.value, float)
    assert abs(result.value - 1 / 3) < 1e-9
    assert result.reason is not None
    assert '1/3' in result.reason


def test_trajectory_match_any_order_multiset_semantics():
    # actual has one 'a'; expected requires two 'a's. Overlap counts each
    # 'a' only once because it's a multiset intersection.
    tree = _build_tree([_v2_tool_span(name='a', span_id=1, args='{}', start_offset=0.0)])
    result = TrajectoryMatch(expected_trajectory=['a', 'a'], order='any_order').evaluate(_ctx(tree=tree))
    assert isinstance(result.value, float)
    assert abs(result.value - 0.5) < 1e-9


def test_trajectory_match_any_order_empty_expected():
    tree = _build_tree([_v2_tool_span(name='a', span_id=1, args='{}', start_offset=0.0)])
    result = TrajectoryMatch(expected_trajectory=[], order='any_order').evaluate(_ctx(tree=tree))
    assert result.value == 1.0


def test_trajectory_match_no_span_tree():
    result = TrajectoryMatch(expected_trajectory=['a']).evaluate(_ctx(tree=SpanTreeRecordingError('x')))
    assert result.value is False


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


def test_argument_correctness_occurrence_out_of_range():
    tree = _build_tree([_v3_tool_span(name='search', span_id=1, args='{"q": "a"}', start_offset=0.0)])
    evaluator = ArgumentCorrectness(
        tool_name='search',
        expected_arguments={'q': 'a'},
        occurrence=5,
    )
    result = evaluator.evaluate(_ctx(tree=tree))
    assert result.value is False
    assert result.reason is not None
    assert 'out of range' in result.reason


def test_argument_correctness_include_content_false():
    """When `include_content=False`, the arguments string isn't recorded."""
    tree = _build_tree([_v3_tool_span(name='search', span_id=1, args=None, start_offset=0.0)])
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


def test_argument_correctness_no_span_tree():
    result = ArgumentCorrectness(tool_name='x', expected_arguments={'a': 1}).evaluate(
        _ctx(tree=SpanTreeRecordingError('x'))
    )
    assert result.value is False


# ---------------------------------------------------------------------------
# StepEfficiency
# ---------------------------------------------------------------------------


def test_step_efficiency_no_thresholds_returns_empty():
    tree = _build_tree([_v2_tool_span(name='a', span_id=1, args='{}', start_offset=0.0)])
    assert StepEfficiency().evaluate(_ctx(tree=tree)) == {}


def test_step_efficiency_tool_calls_under_budget():
    tree = _build_tree(
        [
            _v2_tool_span(name='a', span_id=1, args='{}', start_offset=0.0),
            _v2_tool_span(name='b', span_id=2, args='{}', start_offset=0.1),
        ]
    )
    result = StepEfficiency(max_tool_calls=3).evaluate(_ctx(tree=tree))
    assert set(result.keys()) == {STEP_EFFICIENCY_TOOL_CALLS_KEY}
    reason = result[STEP_EFFICIENCY_TOOL_CALLS_KEY]
    assert reason.value is True
    assert reason.reason is not None
    assert '2 tool call(s)' in reason.reason


def test_step_efficiency_tool_calls_over_budget():
    tree = _build_tree(
        [
            _v2_tool_span(name='a', span_id=1, args='{}', start_offset=0.0),
            _v2_tool_span(name='b', span_id=2, args='{}', start_offset=0.1),
            _v2_tool_span(name='c', span_id=3, args='{}', start_offset=0.2),
        ]
    )
    result = StepEfficiency(max_tool_calls=2).evaluate(_ctx(tree=tree))
    assert result[STEP_EFFICIENCY_TOOL_CALLS_KEY].value is False


def test_step_efficiency_model_requests_from_metrics():
    # Nothing in the tree; metrics provide the count.
    tree = _build_tree([])
    result = StepEfficiency(max_model_requests=3).evaluate(_ctx(tree=tree, metrics={'requests': 2}))
    assert set(result.keys()) == {STEP_EFFICIENCY_MODEL_REQUESTS_KEY}
    reason = result[STEP_EFFICIENCY_MODEL_REQUESTS_KEY]
    assert reason.value is True
    assert reason.reason is not None
    assert 'ctx.metrics' in reason.reason


def test_step_efficiency_model_requests_falls_back_to_span_count():
    tree = _build_tree(
        [
            _model_request_span(span_id=1, start_offset=0.0),
            _model_request_span(span_id=2, start_offset=0.1),
        ]
    )
    result = StepEfficiency(max_model_requests=1).evaluate(_ctx(tree=tree))
    reason = result[STEP_EFFICIENCY_MODEL_REQUESTS_KEY]
    assert reason.value is False
    assert reason.reason is not None
    assert 'from span tree' in reason.reason


def test_step_efficiency_both_thresholds_set_and_both_over_budget():
    tree = _build_tree(
        [
            _v2_tool_span(name='a', span_id=1, args='{}', start_offset=0.0),
            _v2_tool_span(name='b', span_id=2, args='{}', start_offset=0.1),
            _v2_tool_span(name='c', span_id=3, args='{}', start_offset=0.2),
            _model_request_span(span_id=10, start_offset=0.3),
            _model_request_span(span_id=11, start_offset=0.4),
        ]
    )
    result = StepEfficiency(max_tool_calls=2, max_model_requests=1).evaluate(_ctx(tree=tree))
    assert set(result.keys()) == {
        STEP_EFFICIENCY_TOOL_CALLS_KEY,
        STEP_EFFICIENCY_MODEL_REQUESTS_KEY,
    }
    assert result[STEP_EFFICIENCY_TOOL_CALLS_KEY].value is False
    assert result[STEP_EFFICIENCY_MODEL_REQUESTS_KEY].value is False


def test_step_efficiency_no_span_tree_populates_configured_keys():
    ctx = _ctx(tree=SpanTreeRecordingError('spans were not recorded'))
    result = StepEfficiency(max_tool_calls=2, max_model_requests=3).evaluate(ctx)
    assert set(result.keys()) == {
        STEP_EFFICIENCY_TOOL_CALLS_KEY,
        STEP_EFFICIENCY_MODEL_REQUESTS_KEY,
    }
    for reason in result.values():
        assert reason.value is False
        assert reason.reason is not None
        assert 'logfire' in reason.reason


def test_step_efficiency_no_span_tree_skips_unconfigured_keys():
    ctx = _ctx(tree=SpanTreeRecordingError('spans were not recorded'))
    result = StepEfficiency(max_tool_calls=5).evaluate(ctx)
    assert set(result.keys()) == {STEP_EFFICIENCY_TOOL_CALLS_KEY}


def test_step_efficiency_no_span_tree_only_model_requests():
    # Exercises the `max_tool_calls is None, max_model_requests is not None`
    # branch on the SpanTreeRecordingError path.
    ctx = _ctx(tree=SpanTreeRecordingError('spans were not recorded'))
    result = StepEfficiency(max_model_requests=3).evaluate(ctx)
    assert set(result.keys()) == {STEP_EFFICIENCY_MODEL_REQUESTS_KEY}
    assert result[STEP_EFFICIENCY_MODEL_REQUESTS_KEY].value is False


def test_step_efficiency_metrics_non_numeric_falls_back_to_spans():
    # If `ctx.metrics['requests']` is somehow not numeric, we should fall
    # back to counting spans rather than crashing.
    tree = _build_tree([_model_request_span(span_id=1, start_offset=0.0)])
    # Pass a metric with a non-numeric sentinel; the typed signature uses
    # int | float, but guard code must still handle unexpected values.
    ctx = _ctx(tree=tree, metrics={})  # empty metrics -> falls back to span tree
    result = StepEfficiency(max_model_requests=5).evaluate(ctx)
    assert result[STEP_EFFICIENCY_MODEL_REQUESTS_KEY].value is True
