"""Span-based evaluators for agentic workflows.

These evaluators compare the tool-call trajectory captured in OpenTelemetry
spans against expectations. They are deterministic and require no LLM calls,
so they are cheap to run and produce the same score for the same trace.

All of the evaluators in this module ([`ToolCorrectness`][pydantic_evals.evaluators.ToolCorrectness],
[`TrajectoryMatch`][pydantic_evals.evaluators.TrajectoryMatch],
[`ArgumentCorrectness`][pydantic_evals.evaluators.ArgumentCorrectness],
[`MaxToolCalls`][pydantic_evals.evaluators.MaxToolCalls], and
[`MaxModelRequests`][pydantic_evals.evaluators.MaxModelRequests]) read from
`ctx.span_tree` and gracefully degrade to a failing
[`EvaluationReason`][pydantic_evals.evaluators.EvaluationReason] if spans were
not captured (e.g. Logfire isn't configured).

!!! note "Locally-executed tools only"
    These evaluators only see tools whose execution produces a local span
    (i.e. tools Pydantic AI calls itself). Provider-native or server-side
    builtin tools — such as OpenAI's file search or Anthropic's web search —
    do not create local spans and will not be counted.

!!! note "What counts as a tool call"
    Every execution *attempt* produces a span, discriminated as follows:

    - An attempt that ended in an error — the tool body raised an exception,
      or requested a retry via `ModelRetry` — is **not** counted by default;
      pass `include_failed=True` to count every attempt. The exception:
      [`MaxToolCalls`][pydantic_evals.evaluators.MaxToolCalls] counts failed
      attempts by default (they still consume budget); pass
      `include_failed=False` there to count only successful calls.
    - A deferred call (`ApprovalRequired` / `CallDeferred`) is **never**
      counted: it did not execute in this run.
    - All matching spans in the captured tree are counted, including tool
      calls made by nested sub-agents (agent-as-tool delegation).
"""

from __future__ import annotations as _annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Literal, cast

from ..otel._errors import SpanTreeRecordingError
from ..otel.span_tree import SpanNode, SpanTree
from .context import EvaluatorContext
from .evaluator import EvaluationReason, Evaluator

__all__ = (
    'ArgumentCorrectness',
    'ArgumentMatchMode',
    'ArgumentOccurrence',
    'MaxModelRequests',
    'MaxToolCalls',
    'ToolCorrectness',
    'TrajectoryMatch',
    'TrajectoryOrder',
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
# v2/v3 attribute names for arguments
_V2_TOOL_ARGUMENTS_ATTR = 'tool_arguments'
_V3_TOOL_ARGUMENTS_ATTR = 'gen_ai.tool.call.arguments'
# v3+ marker for output-function spans: `logfire.msg` starts with this
_OUTPUT_FUNCTION_MSG_PREFIX = 'running output function:'
# set on tool spans whose call was deferred (`ApprovalRequired`/`CallDeferred`)
# rather than executed
_TOOL_DEFERRAL_NAME_ATTR = 'pydantic_ai.tool.deferral.name'
# attribute that marks a span as a model request (chat) — same criteria as
# `_task_run.extract_span_tree_metrics` uses to count the `requests` metric
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


def _is_tool_call_span(node: SpanNode) -> bool:
    """Return True if this span represents a locally-executed tool call attempt.

    Output-function spans share the `gen_ai.tool.name` attribute and (in v3+)
    the `execute_tool ...` span name with regular tool spans, so they're
    discriminated by either having the dedicated v2 span name or a
    `logfire.msg` attribute starting with `'running output function:'`.

    Deferred calls (`ApprovalRequired`/`CallDeferred`) produce a span with the
    same shape but never execute, so they're excluded via the deferral marker
    attribute.
    """
    tool_name = node.attributes.get(_GEN_AI_TOOL_NAME_ATTR)
    if not isinstance(tool_name, str):
        return False
    # Deferred calls never actually ran; don't count them.
    if _TOOL_DEFERRAL_NAME_ATTR in node.attributes:
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

    return _ToolCallInfo(
        name=tool_name,
        arguments=arguments if isinstance(arguments, str) else None,
    )


def _matches_tool_call(node: SpanNode, *, include_failed: bool) -> bool:
    return _is_tool_call_span(node) and (include_failed or node.status != 'error')


def _extract_tool_calls(span_tree: SpanTree, *, include_failed: bool) -> list[_ToolCallInfo]:
    """Return all locally-executed tool calls in the tree, ordered by start time."""
    tool_spans = [node for node in span_tree if _matches_tool_call(node, include_failed=include_failed)]
    tool_spans.sort(key=lambda n: n.start_timestamp)
    return [_extract_tool_call_info(node) for node in tool_spans]


def _count_tool_calls(span_tree: SpanTree, *, include_failed: bool) -> int:
    """Count locally-executed tool-call spans in the tree."""
    return sum(1 for node in span_tree if _matches_tool_call(node, include_failed=include_failed))


def _count_model_requests(span_tree: SpanTree) -> int:
    """Count LLM chat-request spans in the tree."""
    return sum(1 for node in span_tree if _is_model_request_span(node))


_NO_SPAN_TREE_REASON = 'No span tree available — ensure logfire/instrumentation is configured.'


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
        allow_extra: If `False` (the default), any tool call not listed in
            `expected_tools` fails the check. Set to `True` to only require
            that the expected tools were called, permitting extras.
        include_failed: If `False` (the default), tool-call attempts that
            ended in an error (a raised exception, or a retry requested via
            `ModelRetry`) are not counted. Set to `True` to count every
            attempt.
        evaluation_name: Optional override for the reported evaluation name.

    Returns `EvaluationReason` with a `bool` value.
    """

    expected_tools: list[str]
    allow_extra: bool = False
    include_failed: bool = False
    evaluation_name: str | None = field(default=None)

    def get_default_evaluation_name(self) -> str:
        return self.evaluation_name if isinstance(self.evaluation_name, str) else self.get_serialization_name()

    def evaluate(self, ctx: EvaluatorContext[object, object, object]) -> EvaluationReason:
        try:
            span_tree = ctx.span_tree
        except SpanTreeRecordingError:
            return EvaluationReason(value=False, reason=_NO_SPAN_TREE_REASON)

        actual = Counter(call.name for call in _extract_tool_calls(span_tree, include_failed=self.include_failed))
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
- `'any_order'`: F1 score combining precision and recall of the multiset
  intersection (order is ignored, but extra and missing calls both reduce
  the score).
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


def _f1(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


@dataclass(repr=False)
class TrajectoryMatch(Evaluator[object, object, object]):
    """Compare the agent's tool-call trajectory to an expected one.

    Args:
        expected_trajectory: The expected ordered list of tool names.
        order: How strictly to compare:

            - `'exact'`: actual must equal expected (1.0) or not (0.0).
            - `'in_order'` (default): F1 computed from the longest common
              subsequence (LCS) of the two sequences. Extra calls reduce
              precision; missing calls reduce recall.
            - `'any_order'`: F1 computed from the multiset intersection of the
              two trajectories. Order is ignored, but extra and missing calls
              still reduce the score.
        include_failed: If `False` (the default), tool-call attempts that
            ended in an error (a raised exception, or a retry requested via
            `ModelRetry`) are not part of the trajectory. Set to `True` to
            include every attempt.
        evaluation_name: Optional override for the reported evaluation name.

    Returns `EvaluationReason` with a `float` value in `[0.0, 1.0]` (including
    when no span tree was captured, in which case the value is `0.0`). For the
    F1-based modes, the `reason` text shows the precision, recall and F1
    numbers so the score can be reproduced from the reported mismatch.

    If both the expected and actual trajectories are empty, all modes score
    `1.0`; if only one of them is empty, all modes score `0.0`.
    """

    expected_trajectory: list[str]
    order: TrajectoryOrder = 'in_order'
    include_failed: bool = False
    evaluation_name: str | None = field(default=None)

    def get_default_evaluation_name(self) -> str:
        return self.evaluation_name if isinstance(self.evaluation_name, str) else self.get_serialization_name()

    def evaluate(self, ctx: EvaluatorContext[object, object, object]) -> EvaluationReason:
        try:
            span_tree = ctx.span_tree
        except SpanTreeRecordingError:
            return EvaluationReason(value=0.0, reason=_NO_SPAN_TREE_REASON)

        actual = [call.name for call in _extract_tool_calls(span_tree, include_failed=self.include_failed)]
        expected = list(self.expected_trajectory)

        if self.order == 'exact':
            if actual == expected:
                return EvaluationReason(value=1.0, reason=f'actual trajectory matches expected: {actual!r}')
            return EvaluationReason(
                value=0.0,
                reason=f'actual trajectory {actual!r} does not equal expected {expected!r}',
            )

        if not actual and not expected:
            return EvaluationReason(value=1.0, reason='both actual and expected trajectories are empty')

        if self.order == 'any_order':
            overlap = sum((Counter(expected) & Counter(actual)).values())
            precision = overlap / len(actual) if actual else 0.0
            recall = overlap / len(expected) if expected else 0.0
            f1 = _f1(precision, recall)
            reason = (
                f'multiset overlap={overlap}, precision={overlap}/{len(actual)}={precision:.3f}, '
                f'recall={overlap}/{len(expected)}={recall:.3f}, F1={f1:.3f} '
                f'(expected: {expected!r}, actual: {actual!r})'
            )
            return EvaluationReason(value=f1, reason=reason)

        # order == 'in_order'
        lcs = _longest_common_subsequence_length(actual, expected)
        precision = lcs / len(actual) if actual else 0.0
        recall = lcs / len(expected) if expected else 0.0
        f1 = _f1(precision, recall)
        reason = (
            f'LCS={lcs}, precision={lcs}/{len(actual)}={precision:.3f}, '
            f'recall={lcs}/{len(expected)}={recall:.3f}, F1={f1:.3f} '
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
            deep equality. Note that the subset comparison applies only to
            top-level keys: an expected *value* (including a nested dict) must
            compare equal to the actual value in full.
        occurrence: Which invocation of the tool to inspect when the tool is
            called multiple times: `'first'`, `'last'`, or a 0-based integer
            index. A negative int is not supported.
        include_failed: If `False` (the default), tool-call attempts that
            ended in an error (a raised exception, or a retry requested via
            `ModelRetry`) are not considered. Set to `True` to consider every
            attempt; each attempt then counts as a separate occurrence, so
            `'first'` may select an attempt that was subsequently retried.
        evaluation_name: Optional override for the reported evaluation name.

    Returns `EvaluationReason` with a `bool` value. Fails gracefully with a
    descriptive reason if the tool was never called, the requested occurrence
    doesn't exist, or arguments weren't recorded (e.g. `include_content=False`).
    """

    tool_name: str
    expected_arguments: dict[str, Any]
    match_mode: ArgumentMatchMode = 'subset'
    occurrence: ArgumentOccurrence | int = 'first'
    include_failed: bool = False
    evaluation_name: str | None = field(default=None)

    def get_default_evaluation_name(self) -> str:
        return self.evaluation_name if isinstance(self.evaluation_name, str) else self.get_serialization_name()

    def evaluate(self, ctx: EvaluatorContext[object, object, object]) -> EvaluationReason:
        try:
            span_tree = ctx.span_tree
        except SpanTreeRecordingError:
            return EvaluationReason(value=False, reason=_NO_SPAN_TREE_REASON)

        tool_calls = _extract_tool_calls(span_tree, include_failed=self.include_failed)
        matches = [call for call in tool_calls if call.name == self.tool_name]
        if not matches:
            return EvaluationReason(value=False, reason=f'No calls to tool {self.tool_name!r} were recorded.')

        selected = self._select(matches)
        if selected is None:
            return EvaluationReason(
                value=False,
                reason=(
                    f'Tool {self.tool_name!r} was called {len(matches)} time(s); '
                    f'occurrence={self.occurrence!r} does not select any of them '
                    f"(must be 'first', 'last', or a 0-based index; negative ints are not supported)."
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
        if not isinstance(self.occurrence, int):  # runtime guard: plain dataclasses don't validate the annotation
            return None
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
# MaxToolCalls / MaxModelRequests
# ---------------------------------------------------------------------------


@dataclass(repr=False)
class MaxToolCalls(Evaluator[object, object, object]):
    """Assert that the agent made at most `max_calls` locally-executed tool calls.

    Args:
        max_calls: Maximum allowed locally-executed tool calls.
        include_failed: If `True` (the default), tool-call attempts that ended
            in an error (a raised exception, or a retry requested via
            `ModelRetry`) count against the budget — they still consumed time
            and tokens. Set to `False` to count only successful calls.
        evaluation_name: Optional override for the reported evaluation name.

    Returns `EvaluationReason` with a `bool` value.
    """

    max_calls: int
    include_failed: bool = True
    evaluation_name: str | None = field(default=None)

    def get_default_evaluation_name(self) -> str:
        return self.evaluation_name if isinstance(self.evaluation_name, str) else self.get_serialization_name()

    def evaluate(self, ctx: EvaluatorContext[object, object, object]) -> EvaluationReason:
        try:
            span_tree = ctx.span_tree
        except SpanTreeRecordingError:
            return EvaluationReason(value=False, reason=_NO_SPAN_TREE_REASON)

        tool_count = _count_tool_calls(span_tree, include_failed=self.include_failed)
        return EvaluationReason(
            value=tool_count <= self.max_calls,
            reason=f'{tool_count} tool call(s), budget={self.max_calls}',
        )


@dataclass(repr=False)
class MaxModelRequests(Evaluator[object, object, object]):
    """Assert that the agent made at most `max_requests` model (chat) requests.

    Prefers the `requests` value from `ctx.metrics` when available, otherwise
    counts LLM request spans in the span tree directly (both use the same
    criteria, so the two sources agree whenever both are populated).

    Args:
        max_requests: Maximum allowed model requests.
        evaluation_name: Optional override for the reported evaluation name.

    Returns `EvaluationReason` with a `bool` value.
    """

    max_requests: int
    evaluation_name: str | None = field(default=None)

    def get_default_evaluation_name(self) -> str:
        return self.evaluation_name if isinstance(self.evaluation_name, str) else self.get_serialization_name()

    def evaluate(self, ctx: EvaluatorContext[object, object, object]) -> EvaluationReason:
        try:
            span_tree = ctx.span_tree
        except SpanTreeRecordingError:
            return EvaluationReason(value=False, reason=_NO_SPAN_TREE_REASON)

        metric = ctx.metrics.get('requests')
        if isinstance(metric, int | float):
            request_count = int(metric)
            source = "ctx.metrics['requests']"
        else:
            request_count = _count_model_requests(span_tree)
            source = 'span tree'
        return EvaluationReason(
            value=request_count <= self.max_requests,
            reason=f'{request_count} model request(s) (from {source}), budget={self.max_requests}',
        )
