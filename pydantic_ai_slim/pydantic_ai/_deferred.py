"""Deferred tool types.

The types here were originally in `tools.py`, but `tools.py` transitively
imports `_function_schema → _run_context → messages`, which means
`messages.py` cannot import from `tools.py` at module-load time.  This
module only depends on `messages`, `exceptions`, and `_utils`, so
`messages.py` can safely late-import from here (same pattern as
`_tool_search.py`).

`tools.py` re-exports every public name so that the external API
(`pydantic_ai.tools.DeferredToolRequests`, etc.) is unchanged.
"""

from __future__ import annotations as _annotations

from dataclasses import KW_ONLY, dataclass, field
from typing import Annotated, Any, Literal, TypeAlias, cast

from pydantic import (
    Discriminator,
    SerializerFunctionWrapHandler,
    Tag,
    TypeAdapter,
    ValidatorFunctionWrapHandler,
    WrapSerializer,
    WrapValidator,
)

from . import _utils
from .exceptions import ModelRetry, ToolFailed
from .messages import RetryPromptPart, ToolCallPart, ToolReturn


@dataclass(kw_only=True)
class DeferredToolRequests:
    """Tool calls that require approval or external execution.

    This can be used as an agent's `output_type` and will be used as the output of the agent run if the model called any deferred tools.

    Results can be passed to the next agent run using a [`DeferredToolResults`][pydantic_ai.tools.DeferredToolResults] object with the same tool call IDs.

    See [deferred tools docs](../deferred-tools.md#deferred-tools) for more information.
    """

    calls: list[ToolCallPart] = field(default_factory=list[ToolCallPart])
    """Tool calls that require external execution."""
    approvals: list[ToolCallPart] = field(default_factory=list[ToolCallPart])
    """Tool calls that require human-in-the-loop approval."""
    metadata: dict[str, dict[str, Any]] = field(default_factory=dict[str, dict[str, Any]])
    """Metadata for deferred tool calls, keyed by `tool_call_id`."""

    def build_results(
        self,
        *,
        approvals: dict[str, bool | DeferredToolApprovalResult] | None = None,
        calls: dict[str, DeferredToolCallResult | Any] | None = None,
        metadata: dict[str, dict[str, Any]] | None = None,
        approve_all: bool = False,
    ) -> DeferredToolResults:
        """Create a [`DeferredToolResults`][pydantic_ai.tools.DeferredToolResults] for these requests.

        Args:
            approvals: Results for tool calls that required approval. Keys must match
                `tool_call_id`s in `self.approvals`.
            calls: Results for tool calls that required external execution. Keys must
                match `tool_call_id`s in `self.calls`.
            metadata: Per-call metadata, keyed by `tool_call_id`.
            approve_all: If `True`, every approval-requesting call not already listed in
                `approvals` is approved (with default `ToolApproved()`).

        Raises:
            ValueError: If a key in `approvals`/`calls` doesn't match a pending request of
                the appropriate kind.
        """
        approvals = dict(approvals) if approvals else {}
        calls = dict(calls) if calls else {}

        approval_ids = {c.tool_call_id for c in self.approvals}
        call_ids = {c.tool_call_id for c in self.calls}

        if extra_approvals := set(approvals) - approval_ids:
            raise ValueError(
                f'`approvals` contains tool call IDs not in this `DeferredToolRequests.approvals`: {sorted(extra_approvals)}'
            )
        if extra_calls := set(calls) - call_ids:
            raise ValueError(
                f'`calls` contains tool call IDs not in this `DeferredToolRequests.calls`: {sorted(extra_calls)}'
            )

        if approve_all:
            for tool_call_id in approval_ids - set(approvals):
                approvals[tool_call_id] = ToolApproved()

        return DeferredToolResults(approvals=approvals, calls=calls, metadata=metadata or {})

    def remaining(self, results: DeferredToolResults) -> DeferredToolRequests | None:
        """Return unresolved requests after applying results, or `None` if all resolved."""
        resolved_ids = set(results.approvals) | set(results.calls)
        remaining = DeferredToolRequests(
            calls=[c for c in self.calls if c.tool_call_id not in resolved_ids],
            approvals=[c for c in self.approvals if c.tool_call_id not in resolved_ids],
            metadata={k: v for k, v in self.metadata.items() if k not in resolved_ids},
        )
        return remaining if remaining.calls or remaining.approvals else None


@dataclass(kw_only=True)
class ToolApproved:
    """Indicates that a tool call has been approved and that the tool function should be executed."""

    override_args: dict[str, Any] | None = None
    """Optional tool call arguments to use instead of the original arguments."""

    kind: Literal['tool-approved'] = 'tool-approved'


@dataclass
class ToolDenied:
    """Indicates that a tool call has been denied and that a denial message should be returned to the model."""

    message: str = 'The tool call was denied.'
    """The message to return to the model."""

    _: KW_ONLY

    kind: Literal['tool-denied'] = 'tool-denied'


def _deferred_tool_call_result_discriminator(x: Any) -> str | None:
    if isinstance(x, ToolFailed):
        return 'tool-failed'
    elif isinstance(x, ModelRetry):
        return 'model-retry'
    elif isinstance(x, ToolReturn):
        return 'tool-return'
    elif isinstance(x, RetryPromptPart):
        return 'retry-prompt'
    elif isinstance(x, dict):
        x_dict = cast(dict[str, Any], x)
        kind = x_dict.get('kind')
        if kind in ('tool-failed', 'model-retry') and 'message' in x_dict and x_dict.keys() <= {'kind', 'message'}:
            return kind
        elif (
            kind == 'tool-return'
            and 'return_value' in x_dict
            and x_dict.keys()
            <= {
                'return_value',
                'content',
                'metadata',
                'kind',
            }
        ):
            return 'tool-return'
        elif (
            x_dict.get('part_kind') == 'retry-prompt'
            and 'content' in x_dict
            and x_dict.keys() <= {'content', 'tool_name', 'tool_call_id', 'timestamp', 'part_kind'}
        ):
            return 'retry-prompt'
    return None


DeferredToolApprovalResult: TypeAlias = Annotated[ToolApproved | ToolDenied, Discriminator('kind')]
"""Result for a tool call that required human-in-the-loop approval."""
DeferredToolCallResult: TypeAlias = Annotated[
    Annotated[ToolReturn, Tag('tool-return')]
    | Annotated[ToolFailed, Tag('tool-failed')]
    | Annotated[ModelRetry, Tag('model-retry')]
    | Annotated[RetryPromptPart, Tag('retry-prompt')],
    Discriminator(_deferred_tool_call_result_discriminator),
]
"""Result for a tool call that required external execution."""
_deferred_tool_call_result_adapter: TypeAdapter[DeferredToolCallResult] = TypeAdapter(DeferredToolCallResult)
DeferredToolResult = DeferredToolApprovalResult | DeferredToolCallResult
"""Result for a tool call that required approval or external execution."""


_DEFERRED_TOOL_RESULT_ESCAPE_KEY = '__pydantic_ai_deferred_tool_result__'


def _is_deferred_tool_result_escape(value: Any) -> bool:
    return (
        _utils.is_str_dict(value)
        and value.keys() == {_DEFERRED_TOOL_RESULT_ESCAPE_KEY, 'value'}
        and value[_DEFERRED_TOOL_RESULT_ESCAPE_KEY] == 'arbitrary'
    )


def _serialize_deferred_tool_calls(
    value: dict[str, DeferredToolCallResult | Any], handler: SerializerFunctionWrapHandler
) -> dict[str, Any]:
    escaped: dict[str, DeferredToolCallResult | Any] = {}
    for tool_call_id, result in value.items():
        if _utils.is_str_dict(result) and (
            _deferred_tool_call_result_discriminator(result) is not None or _is_deferred_tool_result_escape(result)
        ):
            result = {_DEFERRED_TOOL_RESULT_ESCAPE_KEY: 'arbitrary', 'value': result}
        escaped[tool_call_id] = result
    serialized = handler(escaped)
    assert _utils.is_str_dict(serialized)
    return serialized


def _validate_deferred_tool_calls(value: Any, handler: ValidatorFunctionWrapHandler) -> Any:
    if not _utils.is_str_dict(value):
        return handler(value)

    to_validate = value.copy()
    escaped: dict[str, Any] = {}
    control_results: dict[str, dict[str, Any]] = {}
    for tool_call_id, result in value.items():
        if _is_deferred_tool_result_escape(result):
            # A bare exact control dict must keep its historical typed meaning. The escape envelope
            # is therefore the only unambiguous wire form for an arbitrary dict with the same shape.
            escaped[tool_call_id] = result['value']
            to_validate[tool_call_id] = None
        elif _utils.is_str_dict(result) and _deferred_tool_call_result_discriminator(result) is not None:
            control_results[tool_call_id] = result

    validated = handler(to_validate)
    assert _utils.is_str_dict(validated)
    for tool_call_id, result in control_results.items():
        validated[tool_call_id] = _deferred_tool_call_result_adapter.validate_python(result)
    validated.update(escaped)
    return validated


@dataclass(kw_only=True)
class DeferredToolResults:
    """Results for deferred tool calls from a previous run that required approval or external execution.

    The tool call IDs need to match those from the [`DeferredToolRequests`][pydantic_ai.tools.DeferredToolRequests] output object from the previous run.

    See [deferred tools docs](../deferred-tools.md#deferred-tools) for more information.
    """

    calls: Annotated[
        dict[str, DeferredToolCallResult | Any],
        WrapValidator(_validate_deferred_tool_calls),
        WrapSerializer(_serialize_deferred_tool_calls),
    ] = field(default_factory=dict[str, DeferredToolCallResult | Any])
    """Map of tool call IDs to results for tool calls that required external execution."""
    approvals: dict[str, bool | DeferredToolApprovalResult] = field(
        default_factory=dict[str, bool | DeferredToolApprovalResult]
    )
    """Map of tool call IDs to results for tool calls that required human-in-the-loop approval."""
    metadata: dict[str, dict[str, Any]] = field(default_factory=dict[str, dict[str, Any]])
    """Metadata for deferred tool calls, keyed by `tool_call_id`. Each value will be available in the tool's RunContext as `tool_call_metadata`."""

    def update(self, other: DeferredToolResults) -> None:
        """Update this `DeferredToolResults` with entries from another, in-place."""
        self.approvals.update(other.approvals)
        self.calls.update(other.calls)
        self.metadata.update(other.metadata)

    def to_tool_call_results(self) -> dict[str, DeferredToolResult]:
        """Convert results into the internal per-call format used by the tool-execution pipeline.

        Normalizes `True`/`False` approvals to `ToolApproved`/`ToolDenied`, and wraps
        plain external-call values in `ToolReturn`.
        """
        tool_call_results: dict[str, DeferredToolResult] = {}
        for tool_call_id, approval in self.approvals.items():
            if approval is True:
                approval = ToolApproved()
            elif approval is False:
                approval = ToolDenied()
            tool_call_results[tool_call_id] = approval

        call_result_types = _utils.get_union_args(DeferredToolCallResult)
        for tool_call_id, call_result in self.calls.items():
            if not isinstance(call_result, call_result_types):
                call_result = ToolReturn(call_result)
            tool_call_results[tool_call_id] = call_result
        return tool_call_results
