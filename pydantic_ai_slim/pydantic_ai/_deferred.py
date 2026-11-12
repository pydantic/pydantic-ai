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

from collections.abc import Mapping
from dataclasses import KW_ONLY, dataclass, field
from typing import Annotated, Any, Literal, TypeAlias, cast

from pydantic import Discriminator, Field, StrictBool, Tag

from . import _utils
from .exceptions import ModelRetry, ToolFailed, UserError

# TODO(v3): remove `RetryPromptPart`
from .messages import RetryPromptPart, ToolCallPart, ToolReturn  # pyright: ignore[reportDeprecated]

DeferredToolKind: TypeAlias = Literal['approval', 'question', 'external']
"""What a deferred tool call waits for.

- `'approval'`: consent to run a call the model made, from [`ApprovalRequired`][pydantic_ai.exceptions.ApprovalRequired]
  or `requires_approval=True`. Answered in `DeferredToolResults.approvals`.
- `'question'`: content the model asked for, from `CallDeferred(kind='question')`, such as Pydantic AI
  Harness's `AskUser` question. Answered in `DeferredToolResults.calls`.
- `'external'`: the result of a call run outside the agent, from [`CallDeferred`][pydantic_ai.exceptions.CallDeferred]
  or an external tool. Answered in `DeferredToolResults.calls`.
"""


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
    # Internal, not part of the public API: read a call's kind with `kind()`. The calls in `calls` that are
    # questions, keyed by `tool_call_id`. Serialized as `_kinds` only when there are any, so a
    # `DeferredToolRequests` stored, journaled or recorded in telemetry keeps its questions, and one
    # without questions serializes as it did before; left out of the repr.
    _kinds: Annotated[dict[str, DeferredToolKind], Field(exclude_if=lambda kinds: not kinds)] = field(
        default_factory=dict[str, DeferredToolKind], repr=False
    )

    def kind(self, tool_call_id: str) -> DeferredToolKind:
        """What the pending call with this ID waits for: an approval, an answer to a question, or an external result.

        A call in `approvals` is an `'approval'`, and one in `calls` is a `'question'` when the tool deferred
        it with `CallDeferred(kind='question')`, and otherwise `'external'`.

        Raises:
            ValueError: No pending call has this ID.
        """
        if any(call.tool_call_id == tool_call_id for call in self.approvals):
            return 'approval'
        if not any(call.tool_call_id == tool_call_id for call in self.calls):
            raise ValueError(f'No pending deferred tool call has ID {tool_call_id!r}')
        return self._kinds.get(tool_call_id, 'external')

    def build_results(
        self,
        *,
        approvals: Mapping[str, bool | DeferredToolApprovalResult] | None = None,
        calls: dict[str, DeferredToolCallResult | Any] | None = None,
        metadata: dict[str, dict[str, Any]] | None = None,
        answered_by: dict[str, str] | None = None,
        approve_all: bool = False,
    ) -> DeferredToolResults:
        """Create a [`DeferredToolResults`][pydantic_ai.tools.DeferredToolResults] for these requests.

        Args:
            approvals: Results for tool calls that required approval. Keys must match
                `tool_call_id`s in `self.approvals`.
            calls: Results for tool calls that required external execution. Keys must
                match `tool_call_id`s in `self.calls`.
            metadata: Per-call metadata, keyed by `tool_call_id`.
            answered_by: Who answered each call, keyed by `tool_call_id`.
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

        return DeferredToolResults(
            approvals=approvals, calls=calls, metadata=metadata or {}, answered_by=answered_by or {}
        )

    def remaining(self, results: DeferredToolResults) -> DeferredToolRequests | None:
        """Return unresolved requests after applying results, or `None` if all resolved."""
        results = filter_deferred_results(self, results)
        call_result_ids = set(results.calls)
        approval_result_ids = set(results.approvals)
        resolved_ids = call_result_ids | approval_result_ids
        remaining = DeferredToolRequests(
            calls=[c for c in self.calls if c.tool_call_id not in call_result_ids],
            approvals=[c for c in self.approvals if c.tool_call_id not in approval_result_ids],
            metadata={k: v for k, v in self.metadata.items() if k not in resolved_ids},
            _kinds={k: v for k, v in self._kinds.items() if k not in resolved_ids},
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
    elif isinstance(x, dict):
        x_dict = cast(dict[str, Any], x)
        if 'kind' in x_dict:
            return cast(str, x_dict['kind'])
        elif 'part_kind' in x_dict:
            return cast(str, x_dict['part_kind'])
    else:
        if hasattr(x, 'kind'):
            return cast(str, x.kind)
        elif hasattr(x, 'part_kind'):
            return cast(str, x.part_kind)
    return None


DeferredToolApprovalResult: TypeAlias = Annotated[ToolApproved | ToolDenied, Discriminator('kind')]
"""Result for a tool call that required human-in-the-loop approval."""
DeferredToolCallResult: TypeAlias = Annotated[
    Annotated[ToolReturn, Tag('tool-return')]
    | Annotated[ToolFailed, Tag('tool-failed')]
    | Annotated[ModelRetry, Tag('model-retry')]
    # TODO(v3): remove `RetryPromptPart`. A handler still answering with one gets the same result as
    # `ModelRetry`: the retry is bound to the call being resolved and travels as its own return.
    | Annotated[RetryPromptPart, Tag('retry-prompt')],  # pyright: ignore[reportDeprecated]
    Discriminator(_deferred_tool_call_result_discriminator),
]
"""Result for a tool call that required external execution."""
DeferredToolResult = DeferredToolApprovalResult | DeferredToolCallResult
"""Result for a tool call that required approval or external execution."""


@dataclass(kw_only=True)
class DeferredToolResults:
    """Results for deferred tool calls from a previous run that required approval or external execution.

    The tool call IDs need to match those from the [`DeferredToolRequests`][pydantic_ai.tools.DeferredToolRequests] output object from the previous run.

    See [deferred tools docs](../deferred-tools.md#deferred-tools) for more information.
    """

    calls: dict[str, DeferredToolCallResult | Any] = field(default_factory=dict[str, DeferredToolCallResult | Any])
    """Map of tool call IDs to results for tool calls that required external execution."""
    approvals: dict[str, StrictBool | DeferredToolApprovalResult] = field(
        default_factory=dict[str, StrictBool | DeferredToolApprovalResult]
    )
    """Map of tool call IDs to results for tool calls that required human-in-the-loop approval."""
    metadata: dict[str, dict[str, Any]] = field(default_factory=dict[str, dict[str, Any]])
    """Metadata for deferred tool calls, keyed by `tool_call_id`. Each value will be available in the tool's RunContext as `tool_call_metadata`."""
    answered_by: dict[str, str] = field(default_factory=dict[str, str])
    """Who answered each call, keyed by `tool_call_id`: a person's ID, a bot or a policy name.

    Reported as given in the [`DeferredToolResultsEvent`][pydantic_ai.messages.DeferredToolResultsEvent]
    of calls a handler resolves during the run; not yet recorded in the message history, so results
    passed to a new run with `deferred_tool_results` don't report it. Pydantic AI doesn't check it, so
    authorize whoever answers before you record who they are.
    """

    def update(self, other: DeferredToolResults) -> None:
        """Update this `DeferredToolResults` with entries from another, in-place."""
        self.approvals.update(other.approvals)
        self.calls.update(other.calls)
        self.metadata.update(other.metadata)
        self.answered_by.update(other.answered_by)

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
            elif not isinstance(approval, (ToolApproved, ToolDenied)):
                raise UserError(
                    f'Invalid approval result for tool call {tool_call_id!r}: '
                    'expected `bool`, `ToolApproved`, or `ToolDenied`'
                )
            tool_call_results[tool_call_id] = approval

        call_result_types = _utils.get_union_args(DeferredToolCallResult)
        for tool_call_id, call_result in self.calls.items():
            if not isinstance(call_result, call_result_types):
                call_result = ToolReturn(call_result)
            tool_call_results[tool_call_id] = call_result
        return tool_call_results


def filter_deferred_results(requests: DeferredToolRequests, results: DeferredToolResults) -> DeferredToolResults:
    """Keep results whose IDs belong to the matching request category."""
    approval_ids = {call.tool_call_id for call in requests.approvals}
    call_ids = {call.tool_call_id for call in requests.calls}
    return DeferredToolResults(
        approvals={
            tool_call_id: result for tool_call_id, result in results.approvals.items() if tool_call_id in approval_ids
        },
        calls={tool_call_id: result for tool_call_id, result in results.calls.items() if tool_call_id in call_ids},
        metadata=results.metadata,
        answered_by={
            tool_call_id: answerer
            for tool_call_id, answerer in results.answered_by.items()
            if tool_call_id in approval_ids | call_ids
        },
    )
