"""AG-UI interrupt-aware run lifecycle: import gate, stubs, and `DeferredTool*` ↔ interrupt mapping.

The interrupt types (`Interrupt`, `ResumeEntry`, `RunFinishedInterruptOutcome`,
`RunFinishedSuccessOutcome`) and `RunAgentInput.resume` were added in ag-ui-protocol 0.1.19
([#1569](https://github.com/ag-ui-protocol/ag-ui/pull/1569)). Our floor stays at `>=0.1.10`
(see `pydantic_ai/ui/CLAUDE.md`), so this module gates the new types behind a single import
check — `HAS_INTERRUPTS` — with no-op stubs for older SDKs, and owns the two-directional
translation between Pydantic AI `DeferredTool*` and AG-UI interrupts that `_event_stream`
(outbound) and `_adapter` (inbound) consume. `_ResumePayload` is the single source of
truth for the approval resume payload: its generated JSON schema is advertised outbound
on `Interrupt.response_schema` and it validates `ResumeEntry.payload` inbound.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, StrictBool, ValidationError

from ...exceptions import UserError
from ...messages import ToolCallPart
from ...tools import DeferredToolApprovalResult, ToolApproved, ToolDenied
from ._utils import INTERRUPT_ID_PREFIX

if TYPE_CHECKING:
    from ag_ui.core import (
        Interrupt,
        ResumeEntry,
        RunFinishedInterruptOutcome,
        RunFinishedSuccessOutcome,
    )

    HAS_INTERRUPTS = True
else:
    try:
        from ag_ui.core import (
            Interrupt,
            ResumeEntry,
            RunFinishedInterruptOutcome,
            RunFinishedSuccessOutcome,
        )

        HAS_INTERRUPTS = True
    except ImportError:
        HAS_INTERRUPTS = False

        class Interrupt:
            """Stub for ag-ui-protocol < 0.1.19 — no instances are constructed when `HAS_INTERRUPTS` is False."""

        class ResumeEntry:
            """Stub for ag-ui-protocol < 0.1.19 — no instances are constructed when `HAS_INTERRUPTS` is False."""

        class RunFinishedInterruptOutcome:
            """Stub for ag-ui-protocol < 0.1.19."""

        class RunFinishedSuccessOutcome:
            """Stub for ag-ui-protocol < 0.1.19."""


__all__ = [
    'HAS_INTERRUPTS',
    'Interrupt',
    'ResumeEntry',
    'RunFinishedInterruptOutcome',
    'RunFinishedSuccessOutcome',
    'approval_to_interrupt',
    'interrupt_id_to_tool_call_id',
    'resume_entry_to_approval',
]


class _ResumePayload(BaseModel):
    """Wire shape of the `ResumeEntry.payload` a client sends to resolve a tool-approval interrupt.

    Single source of truth for both interrupt legs
    ([#5878](https://github.com/pydantic/pydantic-ai/issues/5878)): its generated JSON
    schema (`_RESUME_RESPONSE_SCHEMA`) is advertised outbound on `Interrupt.response_schema`,
    and `model_validate` parses the payload inbound in `resume_entry_to_approval`, so the
    advertised schema and the accepted shape cannot drift.

    `approved` is a `StrictBool` on purpose: lax-mode coercion would turn payloads like
    `{'approved': 1}` or `{'approved': 'true'}` into approvals, while the deny-by-default
    stance requires such ambiguous payloads to fail validation and deny. It is also
    required (no default) so the generated schema keeps `required: ['approved']`, matching
    the AG-UI recommended approve-with-edits pattern; a payload without `approved` fails
    validation and denies all the same.

    Field names mirror the AG-UI wire keys 1:1, hence the camelCase `editedArgs`. Unknown
    payload keys are ignored (Pydantic's default), matching the previous hand-rolled
    parsing and the `extra='allow'` posture of the AG-UI SDK's own models.
    """

    approved: StrictBool
    editedArgs: dict[str, Any] | None = None
    reason: str | None = None


def _build_resume_response_schema() -> dict[str, Any]:
    """Derive `Interrupt.response_schema` from `_ResumePayload`, stripping cosmetic keys.

    The `title` keys would leak the private class name and auto-generated field labels
    into client-rendered approval forms, and the top-level `description` would leak the
    class docstring, which is written for maintainers rather than end users.
    """
    schema = _ResumePayload.model_json_schema()
    schema.pop('title', None)
    schema.pop('description', None)
    for prop in schema.get('properties', {}).values():
        prop.pop('title', None)
    return schema


_RESUME_RESPONSE_SCHEMA = _build_resume_response_schema()


def approval_to_interrupt(call: ToolCallPart, metadata: dict[str, dict[str, Any]]) -> Interrupt:
    """Build an AG-UI `Interrupt` from a pending approval `ToolCallPart` (outbound).

    The `response_schema` describes the shape clients must put in `ResumeEntry.payload`;
    it is generated from `_ResumePayload`, the same model that validates the payload on
    resume. `editedArgs`, when present, replaces the proposed `ToolCallPart.args`
    (see `ToolApproved.override_args`).
    """
    return Interrupt(
        id=f'{INTERRUPT_ID_PREFIX}{call.tool_call_id}',
        reason='tool_call',
        tool_call_id=call.tool_call_id,
        message=f'Approve {call.tool_name}({call.args_as_json_str()})?',
        response_schema=_RESUME_RESPONSE_SCHEMA,
        metadata=metadata.get(call.tool_call_id),
    )


def interrupt_id_to_tool_call_id(interrupt_id: str) -> str:
    """Reverse the `INTERRUPT_ID_PREFIX` convention applied in `approval_to_interrupt` (inbound)."""
    if not interrupt_id.startswith(INTERRUPT_ID_PREFIX):
        raise UserError(
            f'ResumeEntry.interrupt_id {interrupt_id!r} does not start with the expected '
            f'{INTERRUPT_ID_PREFIX!r} prefix; cannot map it back to a tool call id.'
        )
    return interrupt_id[len(INTERRUPT_ID_PREFIX) :]


def resume_entry_to_approval(entry: ResumeEntry) -> DeferredToolApprovalResult:
    """Translate one `ResumeEntry` payload into `ToolApproved` / `ToolDenied` (inbound).

    The payload is validated against `_ResumePayload`, the same model whose JSON schema
    was advertised on `Interrupt.response_schema`. Approval requires a payload that
    validates with `approved=True`; anything that fails validation (missing, `False`,
    `null`, or non-bool `approved`, a non-dict payload, a non-dict `editedArgs`, or a
    non-string `reason`) is treated as a denial.
    This deny-by-default stance is intentional: this code only runs when a tool was
    declared `requires_approval=True`, so any ambiguity in the client's response must
    not silently execute the call. In particular, an approval carrying a malformed
    `editedArgs` denies the whole payload rather than approving with the original
    arguments the user visibly tried to change.

    `payload.editedArgs` (when `approved=True`) feeds into `ToolApproved.override_args`,
    fully replacing the originally proposed call arguments before the agent re-executes the tool.
    """
    if entry.status == 'cancelled':
        return ToolDenied(message='Cancelled by user.')

    try:
        payload = _ResumePayload.model_validate(entry.payload)
    except ValidationError:
        return ToolDenied()

    if payload.approved:
        if payload.editedArgs is not None:
            return ToolApproved(override_args=payload.editedArgs)
        return ToolApproved()

    # An empty-string reason is treated as absent, keeping ToolDenied's default message.
    if payload.reason:
        return ToolDenied(message=payload.reason)
    return ToolDenied()
