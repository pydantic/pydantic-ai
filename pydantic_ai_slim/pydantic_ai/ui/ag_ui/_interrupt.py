"""AG-UI interrupt-aware run lifecycle: import gate, stubs, and `DeferredTool*` ↔ interrupt mapping.

The interrupt types (`Interrupt`, `ResumeEntry`, `RunFinishedInterruptOutcome`,
`RunFinishedSuccessOutcome`) and `RunAgentInput.resume` were added in ag-ui-protocol 0.1.19
([#1569](https://github.com/ag-ui-protocol/ag-ui/pull/1569)). Our floor stays at `>=0.1.10`
(see `pydantic_ai/ui/CLAUDE.md`), so this module gates the new types behind a single import
check — `HAS_INTERRUPTS` — with no-op stubs for older SDKs, and owns the two-directional
translation between Pydantic AI `DeferredTool*` and AG-UI interrupts that `_event_stream`
(outbound) and `_adapter` (inbound) consume.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ..._utils import is_str_dict
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


def approval_to_interrupt(call: ToolCallPart, metadata: dict[str, dict[str, Any]]) -> Interrupt:
    """Build an AG-UI `Interrupt` from a pending approval `ToolCallPart` (outbound).

    The `response_schema` describes the shape clients must put in `ResumeEntry.payload`:
    `{ approved: bool, editedArgs?: dict }`. `editedArgs`, when present, replaces the
    proposed `ToolCallPart.args` (see `ToolApproved.override_args`).
    """
    return Interrupt(
        id=f'{INTERRUPT_ID_PREFIX}{call.tool_call_id}',
        reason='tool_call',
        tool_call_id=call.tool_call_id,
        message=f'Approve {call.tool_name}({call.args_as_json_str()})?',
        response_schema={
            'type': 'object',
            'properties': {
                'approved': {'type': 'boolean'},
                'editedArgs': {'type': 'object'},
                'reason': {'type': 'string'},
            },
            'required': ['approved'],
        },
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

    Approval requires an explicit `payload.approved == True` — any other shape
    (`False`, missing, `null`, non-bool, or a non-dict payload) is treated as a denial.
    This deny-by-default stance is intentional: this code only runs when a tool was
    declared `requires_approval=True`, so any ambiguity in the client's response must
    not silently execute the call.

    `payload.editedArgs` (when `approved=True`) feeds into `ToolApproved.override_args`,
    fully replacing the originally proposed call arguments before the agent re-executes the tool.
    """
    if entry.status == 'cancelled':
        return ToolDenied(message='Cancelled by user.')

    payload = entry.payload
    if not is_str_dict(payload):
        return ToolDenied()

    if payload.get('approved') is True:
        edited_args = payload.get('editedArgs')
        if is_str_dict(edited_args):
            return ToolApproved(override_args=edited_args)
        return ToolApproved()

    denial_message = payload.get('reason')
    if isinstance(denial_message, str) and denial_message:
        return ToolDenied(message=denial_message)
    return ToolDenied()
