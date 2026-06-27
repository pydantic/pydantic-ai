"""Span-based evaluators for agentic workflows.

These evaluators compare the tool-call trajectory captured in OpenTelemetry
spans against expectations. They are deterministic and require no LLM calls,
so they are cheap to run and produce the same score for the same trace.

All four evaluators ([`ToolCorrectness`][pydantic_evals.evaluators.ToolCorrectness],
[`TrajectoryMatch`][pydantic_evals.evaluators.TrajectoryMatch],
[`ArgumentCorrectness`][pydantic_evals.evaluators.ArgumentCorrectness], and
[`StepEfficiency`][pydantic_evals.evaluators.StepEfficiency]) read from
`ctx.span_tree` and gracefully degrade to a `False`-valued
[`EvaluationReason`][pydantic_evals.evaluators.EvaluationReason] if spans were
not captured (e.g. Logfire isn't configured).

!!! note "Locally-executed tools only"
    These evaluators only see tools whose execution produces a local span
    (i.e. tools Pydantic AI calls itself). Provider-native or server-side
    builtin tools — such as OpenAI's file search or Anthropic's web search —
    do not create local spans and will not be counted.
"""

from __future__ import annotations as _annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any, Literal, cast

from ..otel._errors import SpanTreeRecordingError
from ..otel.span_tree import SpanNode, SpanTree
from .context import EvaluatorContext
from .evaluator import EvaluationReason, Evaluator

__all__ = (
    'ToolCorrectness',
    'TrajectoryMatch',
    'ArgumentCorrectness',
    'StepEfficiency',
)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

# These constants are duplicated (rather than imported) from
# `pydantic_ai._instrumentation.InstrumentationNames` because that module is
# private and its set of versions/naming will keep changing as the
# instrumentation spec evolves. Keeping the small set of constants we depend on
# here lets `pydantic-evals` keep working across multiple instrumentation
# versions without a hard dependency on the private module.
_GEN_AI_TOOL_NAME_ATTR = 'gen_ai.tool.name'
_LOGFIRE_MSG_ATTR = 'logfire.msg'
# v2 span names
_V2_TOOL_SPAN_NAME = 'running tool'
_V2_OUTPUT_FUNCTION_SPAN_NAME = 'running output function'
# v3+ span names are of the form `execute_tool {tool_name}`
_V3_TOOL_SPAN_PREFIX = 'execute_tool '
# v2/v3 attribute names for arguments/result
_V2_TOOL_ARGUMENTS_ATTR = 'tool_arguments'
_V3_TOOL_ARGUMENTS_ATTR = 'gen_ai.tool.call.arguments'
_V2_TOOL_RESULT_ATTR = 'tool_response'
_V3_TOOL_RESULT_ATTR = 'gen_ai.tool.call.result'
# v3+ marker for output-function spans: `logfire.msg` starts with this
_OUTPUT_FUNCTION_MSG_PREFIX = 'running output function:'
# attribute that marks a span as a model request (chat) — same as
# `dataset._extract_span_tree_metrics` uses to count `requests`
_GEN_AI_REQUEST_MODEL_ATTR = 'gen_ai.request.model'
_GEN_AI_OPERATION_NAME_ATTR = 'gen_ai.operation.name'


@dataclass(frozen=True)
class _ToolCallInfo:
    """A single tool-call observation extracted from the span tree.

    This is deliberately private — the shape of instrumentation spans is still
    evolving and we don't want to commit to a public data model yet.
    """

    name: str
    arguments: str | None
    """The JSON-encoded arguments string, or `None` if `include_content=False`."""
    result: str | None
    """The string/JSON result, or `None` if the result isn't recorded."""
    duration: timedelta


def _is_tool_call_span(node: SpanNode) -> bool:
    """Return True if this span represents a locally-executed tool call.

    Output-function spans share the `gen_ai.tool.name` attribute and (in v3+)
    the `execute_tool ...` span name with regular tool spans, so they're
    discriminated by either having the dedicated v2 span name or a
    `logfire.msg` attribute starting with `'running output function:'`.
    """
    tool_name = node.attributes.get(_GEN_AI_TOOL_NAME_ATTR)
    if not isinstance(tool_name, str):
        return False
    # v2: tool calls live under `running tool`; output functions under
    # `running output function`.
    if node.name == _V2_OUTPUT_FUNCTION_SPAN_NAME:
        return False
    if node.name == _V2_TOOL_SPAN_NAME:
        return True
    # v3+: both tool calls and output functions use `execute_tool {name}`,
    # distinguished by `logfire.msg`.
    if not node.name.startswith(_V3_TOOL_SPAN_PREFIX):
        return False
    msg = node.attributes.get(_LOGFIRE_MSG_ATTR)
    if isinstance(msg, str) and msg.startswith(_OUTPUT_FUNCTION_MSG_PREFIX):
        return False
    return True


def _is_model_request_span(node: SpanNode) -> bool:
    """Return True if this span represents an LLM chat request."""
    if _GEN_AI_REQUEST_MODEL_ATTR not in node.attributes:
        return False
    return node.attributes.get(_GEN_AI_OPERATION_NAME_ATTR) == 'chat'


def _extract_tool_call_info(node: SpanNode) -> _ToolCallInfo:
    tool_name = node.attributes.get(_GEN_AI_TOOL_NAME_ATTR)
    assert isinstance(tool_name, str)  # guaranteed by _is_tool_call_span

    # Prefer v3+ attribute, fall back to v2.
    arguments = node.attributes.get(_V3_TOOL_ARGUMENTS_ATTR)
    if arguments is None:
        arguments = node.attributes.get(_V2_TOOL_ARGUMENTS_ATTR)
    result = node.attributes.get(_V3_TOOL_RESULT_ATTR)
    if result is None:
        result = node.attributes.get(_V2_TOOL_RESULT_ATTR)

    return _ToolCallInfo(
        name=tool_name,
        arguments=arguments if isinstance(arguments, str) else None,
        result=result if isinstance(result, str) else None,
        duration=node.duration,
    )


def _extract_tool_calls(span_tree: SpanTree) -> list[_ToolCallInfo]:
    """Return all locally-executed tool calls in the tree, ordered by start time."""
    tool_spans = [node for node in span_tree if _is_tool_call_span(node)]
    tool_spans.sort(key=lambda n: n.start_timestamp)
    return [_extract_tool_call_info(node) for node in tool_spans]


def _count_model_requests(span_tree: SpanTree) -> int:
    """Count LLM chat-request spans in the tree."""
    return sum(1 for node in span_tree if _is_model_request_span(node))


_NO_SPAN_TREE_REASON = 'No span tree available \u2014 ensure logfire/instrumentation is configured.'


# ---------------------------------------------------------------------------
# ToolCorrectness
# ---------------------------------------------------------------------------


@dataclass(repr=False)
class ToolCorrectness(Evaluator[object, object, object]):
    """Assert that the agent called a specific multiset of tools.

    This compares the names of tools actually invoked (as a multiset) against
    `expected_tools`. Repeated names require repeated calls — for example,
    `expected_tools=['search', 'search']` passes only if `search` was called
    at least twice.

    Args:
        expected_tools: The tool names the agent is expected to call. Order
            does not matter; duplicates are significant.
        allow_extra: If `True` (the default), calling tools not listed in
            `expected_tools` is still a pass. If `False`, any unexpected
            tool call fails the check.
        evaluation_name: Optional override for the reported evaluation name.

    Returns `EvaluationReason` with a `bool` value.
    """

    expected_tools: list[str]
    allow_extra: bool = True
    evaluation_name: str | None = field(default=None)

    def evaluate(self, ctx: EvaluatorContext[object, object, object]) -> EvaluationReason:
        try:
            span_tree = ctx.span_tree
        except SpanTreeRecordingError:
            return EvaluationReason(value=False, reason=_NO_SPAN_TREE_REASON)

        actual = Counter(call.name for call in _extract_tool_calls(span_tree))
        expected = Counter(self.expected_tools)

        missing = expected - actual
        extra = actual - expected

        problems: list[str] = []
        if missing:
            missing_desc = ', '.join(f'{name!r} (x{count})' for name, count in sorted(missing.items()))
            problems.append(f'missing tools: {missing_desc}')
        if extra and not self.allow_extra:
            extra_desc = ', '.join(f'{name!r} (x{count})' for name, count in sorted(extra.items()))
            problems.append(f'unexpected tools: {extra_desc}')

        if problems:
            return EvaluationReason(value=False, reason='; '.join(problems))
        return EvaluationReason(value=True)


# ---------------------------------------------------------------------------
# TrajectoryMatch
# ---------------------------------------------------------------------------

TrajectoryOrder = Literal['exact', 'in_order', 'any_order']
"""How to compare the actual tool sequence to `expected_trajectory`.

- `'exact'`: actual must equal expected (1.0) or not (0.0).
- `'in_order'`: F1 score combining precision and recall of the longest
  common subsequence.
- `'any_order'`: multiset overlap, i.e. `|multiset(actual) \u2229 multiset(expected)| / |expected|`.
"""


def _longest_common_subsequence_length(a: list[str], b: list[str]) -> int:
    """Standard dynamic-programming LCS length."""
    if not a or not b:
        return 0
    # Rolling 1-D DP: `prev[j]` holds LCS length for a[:i-1] vs b[:j].
    prev = [0] * (len(b) + 1)
    for i in range(1, len(a) + 1):
        curr = [0] * (len(b) + 1)
        for j in range(1, len(b) + 1):
            if a[i - 1] == b[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev = curr
    return prev[len(b)]


@dataclass(repr=False)
class TrajectoryMatch(Evaluator[object, object, object]):
    """Compare the agent's tool-call trajectory to an expected one.

    Args:
        expected_trajectory: The expected ordered list of tool names.
        order: How strictly to compare — see [`TrajectoryOrder`][pydantic_evals.evaluators.agentic.TrajectoryOrder].
        evaluation_name: Optional override for the reported evaluation name.

    Returns `EvaluationReason` with a `float` value in `[0.0, 1.0]`. For
    `order='in_order'`, the `reason` text shows the precision, recall and F1
    numbers so the score can be reproduced from the reported mismatch.
    """

    expected_trajectory: list[str]
    order: TrajectoryOrder = 'in_order'
    evaluation_name: str | None = field(default=None)

    def evaluate(self, ctx: EvaluatorContext[object, object, object]) -> EvaluationReason:
        try:
            span_tree = ctx.span_tree
        except SpanTreeRecordingError:
            return EvaluationReason(value=False, reason=_NO_SPAN_TREE_REASON)

        actual = [call.name for call in _extract_tool_calls(span_tree)]
        expected = list(self.expected_trajectory)

        if self.order == 'exact':
            if actual == expected:
                return EvaluationReason(value=1.0, reason=f'actual trajectory matches expected: {actual!r}')
            return EvaluationReason(
                value=0.0,
                reason=f'actual trajectory {actual!r} does not equal expected {expected!r}',
            )

        if self.order == 'any_order':
            expected_counter = Counter(expected)
            actual_counter = Counter(actual)
            if not expected_counter:
                # Nothing expected: perfect match by convention.
                return EvaluationReason(value=1.0, reason='expected trajectory is empty')
            overlap = sum((expected_counter & actual_counter).values())
            score = overlap / len(expected)
            return EvaluationReason(
                value=score,
                reason=(
                    f'multiset overlap {overlap}/{len(expected)} = {score:.3f} '
                    f'(expected: {expected!r}, actual: {actual!r})'
                ),
            )

        # order == 'in_order'
        lcs = _longest_common_subsequence_length(actual, expected)
        if not actual and not expected:
            return EvaluationReason(value=1.0, reason='both actual and expected trajectories are empty')
        precision = lcs / len(actual) if actual else 0.0
        recall = lcs / len(expected) if expected else 0.0
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)
        reason = (
            f'LCS={lcs}, precision={lcs}/{len(actual) or 0}={precision:.3f}, '
            f'recall={lcs}/{len(expected) or 0}={recall:.3f}, F1={f1:.3f} '
            f'(expected: {expected!r}, actual: {actual!r})'
        )
        return EvaluationReason(value=f1, reason=reason)


# ---------------------------------------------------------------------------
# ArgumentCorrectness
# ---------------------------------------------------------------------------

ArgumentMatchMode = Literal['exact', 'subset']
"""How to compare actual tool arguments to `expected_arguments`.

- `'exact'`: actual must deep-equal expected.
- `'subset'`: every key/value in expected must be present (and equal) in actual.
"""

ArgumentOccurrence = Literal['first', 'last']
"""Which occurrence of a tool call to inspect when a tool is called multiple times."""


@dataclass(repr=False)
class ArgumentCorrectness(Evaluator[object, object, object]):
    """Assert that a specific tool call received particular arguments.

    Finds all local spans for `tool_name` in the run, picks the requested
    occurrence, parses the recorded JSON arguments, and compares them to
    `expected_arguments`.

    Args:
        tool_name: The tool whose arguments should be checked.
        expected_arguments: Expected argument keys/values.
        match_mode: `'subset'` (default) checks that every expected
            key/value is present in the actual arguments. `'exact'` requires
            deep equality.
        occurrence: Which invocation of the tool to inspect when the tool is
            called multiple times: `'first'`, `'last'`, or a 0-based integer
            index. A negative int is not supported.
        evaluation_name: Optional override for the reported evaluation name.

    Returns `EvaluationReason` with a `bool` value. Fails gracefully with a
    descriptive reason if the tool was never called, the requested occurrence
    doesn't exist, or arguments weren't recorded (e.g. `include_content=False`).
    """

    tool_name: str
    expected_arguments: dict[str, Any]
    match_mode: ArgumentMatchMode = 'subset'
    occurrence: ArgumentOccurrence | int = 'first'
    evaluation_name: str | None = field(default=None)

    def evaluate(self, ctx: EvaluatorContext[object, object, object]) -> EvaluationReason:
        try:
            span_tree = ctx.span_tree
        except SpanTreeRecordingError:
            return EvaluationReason(value=False, reason=_NO_SPAN_TREE_REASON)

        matches = [call for call in _extract_tool_calls(span_tree) if call.name == self.tool_name]
        if not matches:
            return EvaluationReason(value=False, reason=f'No calls to tool {self.tool_name!r} were recorded.')

        selected = self._select(matches)
        if selected is None:
            return EvaluationReason(
                value=False,
                reason=(
                    f'Tool {self.tool_name!r} was called {len(matches)} time(s); '
                    f'occurrence={self.occurrence!r} is out of range.'
                ),
            )

        if selected.arguments is None:
            return EvaluationReason(
                value=False,
                reason=(
                    f'Tool {self.tool_name!r} arguments not available in span (`include_content` may be disabled).'
                ),
            )

        try:
            actual_args = json.loads(selected.arguments)
        except json.JSONDecodeError as e:
            return EvaluationReason(
                value=False,
                reason=f'Tool {self.tool_name!r} arguments could not be parsed as JSON: {e}',
            )

        if not isinstance(actual_args, dict):
            return EvaluationReason(
                value=False,
                reason=f'Tool {self.tool_name!r} arguments are not a JSON object: {actual_args!r}',
            )

        diffs = _diff_arguments(cast(dict[str, Any], actual_args), self.expected_arguments, self.match_mode)
        if diffs:
            return EvaluationReason(
                value=False,
                reason=f'Tool {self.tool_name!r} argument mismatch: ' + '; '.join(diffs),
            )
        return EvaluationReason(value=True)

    def _select(self, matches: list[_ToolCallInfo]) -> _ToolCallInfo | None:
        if self.occurrence == 'first':
            return matches[0]
        if self.occurrence == 'last':
            return matches[-1]
        index = self.occurrence
        if 0 <= index < len(matches):
            return matches[index]
        return None


def _diff_arguments(actual: dict[str, Any], expected: dict[str, Any], match_mode: ArgumentMatchMode) -> list[str]:
    """Return a list of human-readable mismatch descriptions; empty = match."""
    diffs: list[str] = []
    for key, expected_value in expected.items():
        if key not in actual:
            diffs.append(f'missing key {key!r}')
        elif actual[key] != expected_value:
            diffs.append(f'key {key!r}: expected {expected_value!r}, got {actual[key]!r}')
    if match_mode == 'exact':
        for key in actual:
            if key not in expected:
                diffs.append(f'unexpected key {key!r} with value {actual[key]!r}')
    return diffs


# ---------------------------------------------------------------------------
# StepEfficiency
# ---------------------------------------------------------------------------

# Keys used for the mapping returned by `StepEfficiency`. These are documented
# and considered part of the evaluator's public contract so that reports
# render them consistently.
STEP_EFFICIENCY_TOOL_CALLS_KEY = 'tool_calls_under_budget'
STEP_EFFICIENCY_MODEL_REQUESTS_KEY = 'model_requests_under_budget'


@dataclass(repr=False)
class StepEfficiency(Evaluator[object, object, object]):
    """Assert the agent stayed within tool-call and/or model-request budgets.

    Returns a mapping with any of these stable keys (only keys whose budget
    is configured are returned):

    - `'tool_calls_under_budget'`: set when `max_tool_calls` is provided.
    - `'model_requests_under_budget'`: set when `max_model_requests` is provided.

    Each value is an `EvaluationReason` whose `value` is `True` if the count
    is within the budget. The stable key names make these results easy to
    reference consistently in evals reports.

    Args:
        max_tool_calls: Maximum allowed locally-executed tool calls. `None`
            disables the check.
        max_model_requests: Maximum allowed model (chat) requests. `None`
            disables the check. Prefers `ctx.metrics['requests']` when
            available, otherwise counts LLM request spans directly.
        evaluation_name: Unused — results are returned under stable keys.
    """

    max_tool_calls: int | None = None
    max_model_requests: int | None = None
    evaluation_name: str | None = field(default=None)

    def evaluate(self, ctx: EvaluatorContext[object, object, object]) -> dict[str, EvaluationReason]:
        results: dict[str, EvaluationReason] = {}
        if self.max_tool_calls is None and self.max_model_requests is None:
            return results

        try:
            span_tree = ctx.span_tree
        except SpanTreeRecordingError:
            if self.max_tool_calls is not None:
                results[STEP_EFFICIENCY_TOOL_CALLS_KEY] = EvaluationReason(value=False, reason=_NO_SPAN_TREE_REASON)
            if self.max_model_requests is not None:
                results[STEP_EFFICIENCY_MODEL_REQUESTS_KEY] = EvaluationReason(value=False, reason=_NO_SPAN_TREE_REASON)
            return results

        if self.max_tool_calls is not None:
            tool_count = len(_extract_tool_calls(span_tree))
            within = tool_count <= self.max_tool_calls
            results[STEP_EFFICIENCY_TOOL_CALLS_KEY] = EvaluationReason(
                value=within,
                reason=f'{tool_count} tool call(s), budget={self.max_tool_calls}',
            )

        if self.max_model_requests is not None:
            metric = ctx.metrics.get('requests')
            if isinstance(metric, int | float):
                request_count = int(metric)
                source = "ctx.metrics['requests']"
            else:
                request_count = _count_model_requests(span_tree)
                source = 'span tree'
            within = request_count <= self.max_model_requests
            results[STEP_EFFICIENCY_MODEL_REQUESTS_KEY] = EvaluationReason(
                value=within,
                reason=(f'{request_count} model request(s) (from {source}), budget={self.max_model_requests}'),
            )

        return results
