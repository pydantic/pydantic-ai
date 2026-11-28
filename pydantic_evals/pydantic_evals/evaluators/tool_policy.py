"""Declarative security and policy-compliance evaluator for agent tool-call trajectories.

[`ToolPolicyCompliance`][pydantic_evals.evaluators.ToolPolicyCompliance] audits the
locally-executed tool calls captured in `ctx.span_tree` against negative and
relational policy invariants:

- **Forbidden tools** (`forbidden_tools`): tools that must never be called.
- **Per-tool call caps** (`max_calls_per_tool`): upper bounds on invocations of
  specific tools.
- **Prerequisite ordering** (`require_before`): tools that must have *succeeded*
  earlier in the trajectory before a gated tool is invoked.
- **Forbidden argument patterns** (`forbidden_arg_patterns`): regular-expression
  patterns that must not match recorded tool arguments (fail-closed when
  arguments are missing or malformed).

Unlike functional trajectory evaluators such as
[`ToolCorrectness`][pydantic_evals.evaluators.ToolCorrectness],
`ToolPolicyCompliance` includes failed tool-call attempts by default
(`include_failed=True`) because an attempted call to a forbidden tool or with a
disallowed argument is still a policy violation even if the tool raises an error
or requests a retry. Failed calls, however, never satisfy a `require_before`
prerequisite.
"""

from __future__ import annotations as _annotations

import json
import re
from collections import Counter
from collections.abc import Mapping
from dataclasses import KW_ONLY, dataclass, field
from typing import Any, cast

from ..otel._errors import SpanTreeRecordingError
from ..otel.span_tree import SpanNode, SpanTree
from .context import EvaluatorContext
from .evaluator import EvaluationReason, Evaluator

__all__ = ('ToolPolicyCompliance',)

# Instrumentation span constants (mirrored from `agentic.py` to avoid coupling
# `pydantic-evals` to private `pydantic_ai._instrumentation` internals).
_GEN_AI_TOOL_NAME_ATTR = 'gen_ai.tool.name'
_LOGFIRE_MSG_ATTR = 'logfire.msg'
_V2_TOOL_SPAN_NAME = 'running tool'
_V2_OUTPUT_FUNCTION_SPAN_NAME = 'running output function'
_V3_TOOL_SPAN_PREFIX = 'execute_tool '
_V2_TOOL_ARGUMENTS_ATTR = 'tool_arguments'
_V3_TOOL_ARGUMENTS_ATTR = 'gen_ai.tool.call.arguments'
_OUTPUT_FUNCTION_MSG_PREFIX = 'running output function:'
_TOOL_DEFERRAL_NAME_ATTR = 'pydantic_ai.tool.deferral.name'
_NO_SPAN_TREE_REASON = 'No span tree available — ensure logfire/instrumentation is configured.'


@dataclass(frozen=True)
class _ObservedToolCall:
    """A locally-executed tool-call attempt extracted from the span tree."""

    index: int
    """1-based chronological position in the audited trajectory."""
    name: str
    arguments: str | None
    """JSON-encoded arguments string, or `None` if `include_content=False`."""
    succeeded: bool
    """`True` when the span did not end with `status='error'`."""


def _is_tool_call_span(node: SpanNode) -> bool:
    """Return `True` if `node` represents a locally-executed tool call attempt."""
    tool_name = node.attributes.get(_GEN_AI_TOOL_NAME_ATTR)
    if not isinstance(tool_name, str):
        return False
    if _TOOL_DEFERRAL_NAME_ATTR in node.attributes:
        return False
    if node.name == _V2_OUTPUT_FUNCTION_SPAN_NAME:
        return False
    if node.name == _V2_TOOL_SPAN_NAME:
        return True
    if not node.name.startswith(_V3_TOOL_SPAN_PREFIX):
        return False
    msg = node.attributes.get(_LOGFIRE_MSG_ATTR)
    if isinstance(msg, str) and msg.startswith(_OUTPUT_FUNCTION_MSG_PREFIX):
        return False
    return True


def _extract_observed_calls(span_tree: SpanTree, *, include_failed: bool) -> list[_ObservedToolCall]:
    """Extract locally-executed tool calls ordered by start timestamp."""
    tool_spans = [node for node in span_tree if _is_tool_call_span(node) and (include_failed or node.status != 'error')]
    tool_spans.sort(key=lambda n: n.start_timestamp)
    observed: list[_ObservedToolCall] = []
    for idx, node in enumerate(tool_spans, start=1):
        tool_name = node.attributes.get(_GEN_AI_TOOL_NAME_ATTR)
        assert isinstance(tool_name, str)
        arguments = node.attributes.get(_V3_TOOL_ARGUMENTS_ATTR)
        if arguments is None:
            arguments = node.attributes.get(_V2_TOOL_ARGUMENTS_ATTR)
        observed.append(
            _ObservedToolCall(
                index=idx,
                name=tool_name,
                arguments=arguments if isinstance(arguments, str) else None,
                succeeded=node.status != 'error',
            )
        )
    return observed


def _compile_arg_patterns(
    forbidden_arg_patterns: Mapping[str, Mapping[str, str]],
) -> dict[str, dict[str, re.Pattern[str]]]:
    compiled: dict[str, dict[str, re.Pattern[str]]] = {}
    for tool_name, arg_patterns in forbidden_arg_patterns.items():
        compiled_args: dict[str, re.Pattern[str]] = {}
        for arg_name, pattern in arg_patterns.items():
            try:
                compiled_args[arg_name] = re.compile(pattern)
            except re.error as e:
                raise ValueError(
                    f'Invalid regex pattern for tool {tool_name!r}, argument {arg_name!r}: {pattern!r} ({e})'
                ) from e
        compiled[tool_name] = compiled_args
    return compiled


def _check_forbidden_tools(calls: list[_ObservedToolCall], forbidden_tools: set[str]) -> list[str]:
    if not forbidden_tools:
        return []
    violations: list[str] = []
    for call in calls:
        if call.name in forbidden_tools:
            violations.append(f'call #{call.index} invoked forbidden tool {call.name!r}')
    return violations


def _check_max_calls_per_tool(calls: list[_ObservedToolCall], max_calls_per_tool: Mapping[str, int]) -> list[str]:
    if not max_calls_per_tool:
        return []
    counts = Counter(call.name for call in calls)
    violations: list[str] = []
    for tool_name, limit in sorted(max_calls_per_tool.items()):
        actual = counts.get(tool_name, 0)
        if actual > limit:
            violations.append(f'tool {tool_name!r} called {actual} time(s), max allowed is {limit}')
    return violations


def _check_require_before(calls: list[_ObservedToolCall], require_before: Mapping[str, list[str]]) -> list[str]:
    if not require_before:
        return []
    succeeded_tools: set[str] = set()
    violations: list[str] = []
    for call in calls:
        prerequisites = require_before.get(call.name, ())
        missing = [req for req in prerequisites if req not in succeeded_tools]
        if missing:
            missing_desc = ', '.join(repr(req) for req in missing)
            violations.append(f'call #{call.index} to {call.name!r} missing prerequisite(s): {missing_desc}')
        if call.succeeded:
            succeeded_tools.add(call.name)
    return violations


def _parse_call_arguments(call: _ObservedToolCall) -> tuple[dict[str, Any] | None, str | None]:
    """Parse a tool call's JSON arguments, returning `(parsed_dict, error_reason)`."""
    if call.arguments is None:
        return (
            None,
            f'call #{call.index} to {call.name!r} arguments not available in span (`include_content` may be disabled)',
        )
    try:
        parsed: Any = json.loads(call.arguments)
    except (ValueError, RecursionError) as e:
        return None, f'call #{call.index} to {call.name!r} arguments could not be parsed as JSON: {e}'
    if not isinstance(parsed, dict):
        return None, f'call #{call.index} to {call.name!r} arguments are not a JSON object'
    return cast(dict[str, Any], parsed), None


def _match_call_arg_patterns(
    call: _ObservedToolCall,
    parsed_args: Mapping[str, Any],
    tool_patterns: Mapping[str, re.Pattern[str]],
) -> list[str]:
    violations: list[str] = []
    for arg_name, pattern in tool_patterns.items():
        if arg_name not in parsed_args:
            continue
        raw_value = parsed_args[arg_name]
        text = raw_value if isinstance(raw_value, str) else json.dumps(raw_value, sort_keys=True)
        if pattern.search(text):
            violations.append(
                f'call #{call.index} to {call.name!r} argument {arg_name!r} '
                f'matched forbidden pattern {pattern.pattern!r}'
            )
    return violations


def _check_forbidden_arg_patterns(
    calls: list[_ObservedToolCall],
    compiled_patterns: Mapping[str, Mapping[str, re.Pattern[str]]],
) -> list[str]:
    if not compiled_patterns:
        return []
    violations: list[str] = []
    for call in calls:
        tool_patterns = compiled_patterns.get(call.name)
        if not tool_patterns:
            continue
        parsed_args, error_reason = _parse_call_arguments(call)
        if error_reason is not None:
            violations.append(error_reason)
            continue
        assert parsed_args is not None
        violations.extend(_match_call_arg_patterns(call, parsed_args, tool_patterns))
    return violations


@dataclass(repr=False)
class ToolPolicyCompliance(Evaluator[object, object, object]):
    """Audit an agent's tool-call trajectory against declarative security and policy rules.

    Evaluates all locally-executed tool calls captured in `ctx.span_tree` and
    collects every violation across the configured policy rules, returning an
    [`EvaluationReason`][pydantic_evals.evaluators.EvaluationReason] with a
    `bool` value.

    Args:
        forbidden_tools: Tool names that must never be called during the run.
        max_calls_per_tool: Mapping of tool name to maximum allowed invocations
            (must be non-negative).
        require_before: Mapping of gated tool name to prerequisite tool names
            that must have *succeeded* earlier in the trajectory before the
            gated tool is invoked.
        forbidden_arg_patterns: Mapping of tool name to `{arg_name: regex}`
            patterns that must not match the corresponding argument value on
            any invocation of that tool. Fails closed if arguments are missing
            or cannot be parsed as a JSON object.
        include_failed: If `True` (the default), failed tool-call attempts
            (raised exceptions or `ModelRetry`) are audited for policy
            violations. Failed calls never satisfy `require_before`
            prerequisites. Set to `False` to audit only successful calls.
        evaluation_name: Optional override for the reported evaluation name.
    """

    _: KW_ONLY

    forbidden_tools: list[str] = field(default_factory=list[str])
    max_calls_per_tool: dict[str, int] = field(default_factory=dict[str, int])
    require_before: dict[str, list[str]] = field(default_factory=dict[str, list[str]])
    forbidden_arg_patterns: dict[str, dict[str, str]] = field(default_factory=dict[str, dict[str, str]])
    include_failed: bool = True
    evaluation_name: str | None = field(default=None)

    _compiled_patterns: dict[str, dict[str, re.Pattern[str]]] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        for tool_name, limit in self.max_calls_per_tool.items():
            if limit < 0:
                raise ValueError(f'`max_calls_per_tool` limit for {tool_name!r} must be non-negative, got {limit}')
        self._compiled_patterns = _compile_arg_patterns(self.forbidden_arg_patterns)

    def get_default_evaluation_name(self) -> str:
        return self.evaluation_name if isinstance(self.evaluation_name, str) else self.get_serialization_name()

    def evaluate(self, ctx: EvaluatorContext[object, object, object]) -> EvaluationReason:
        try:
            span_tree = ctx.span_tree
        except SpanTreeRecordingError:
            return EvaluationReason(value=False, reason=_NO_SPAN_TREE_REASON)

        calls = _extract_observed_calls(span_tree, include_failed=self.include_failed)
        violations = [
            *_check_forbidden_tools(calls, set(self.forbidden_tools)),
            *_check_max_calls_per_tool(calls, self.max_calls_per_tool),
            *_check_require_before(calls, self.require_before),
            *_check_forbidden_arg_patterns(calls, self._compiled_patterns),
        ]
        if violations:
            return EvaluationReason(value=False, reason='; '.join(violations))
        return EvaluationReason(value=True)
