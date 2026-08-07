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

from copy import deepcopy
from typing import TYPE_CHECKING, Annotated, Any

from pydantic import BaseModel, ConfigDict, StrictBool, ValidationError
from pydantic.alias_generators import to_camel
from pydantic.json_schema import WithJsonSchema

from ...exceptions import UserError
from ...messages import ToolCallPart
from ...tools import DeferredToolApprovalResult, GenerateToolJsonSchema, ToolApproved, ToolDenied
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
    field set clients are told about and the field set we accept cannot drift.

    The two optional fields carry an explicit `WithJsonSchema` because the advertised schema
    is a wire contract predating this model: generating it would render them as
    `anyOf: [..., {'type': 'null'}]`, and an AG-UI client that renders its approval form by
    reading `properties.editedArgs.type` — the shape every example in the AG-UI docs uses —
    would stop recognising them. The override keeps the flat pre-existing shape, which is
    deliberately *narrower* than what validation accepts: `null` and omission are both
    accepted for either field even though the advertised schema names only the value type,
    exactly as it did before this model existed.

    `approved` is a `StrictBool` on purpose: lax-mode coercion would turn payloads like
    `{'approved': 1}` or `{'approved': 'true'}` into approvals, while the deny-by-default
    stance requires such ambiguous payloads to fail validation and deny. It is also
    required (no default) so the generated schema keeps `required: ['approved']`, matching
    the AG-UI recommended approve-with-edits pattern. A payload without `approved` still
    denies, but via `ValidationError` rather than the `approved is False` branch, so its
    `reason` never reaches the denial message — spec-conforming clients always send
    `approved`, and `test_resume_deny_by_default_for_ambiguous_payload` pins the outcome.

    The wire keys are camelCase, so `to_camel` supplies them as aliases. Validation is
    deliberately by alias only — no `populate_by_name` — so the accepted shape stays
    exactly the advertised one and cannot drift from the schema clients are handed.
    Unknown payload keys are ignored (Pydantic's default), matching the `extra='allow'`
    posture of the AG-UI SDK's own models.
    """

    model_config = ConfigDict(alias_generator=to_camel)

    approved: StrictBool
    edited_args: Annotated[dict[str, Any] | None, WithJsonSchema({'type': 'object'})] = None
    reason: Annotated[str | None, WithJsonSchema({'type': 'string'})] = None


# `GenerateToolJsonSchema` drops the per-property `title`s the same way it does for tool
# parameter schemas; the top-level `title`/`description` would leak the private class name
# and the maintainer-facing docstring into client-rendered approval forms. The per-property
# `default: null` goes too: it carries no validation meaning and the advertised schema did
# not have it, and `test_run_finished_interrupt_outcome_for_pending_approval` pins
# the result against that contract.
_RESUME_RESPONSE_SCHEMA = _ResumePayload.model_json_schema(schema_generator=GenerateToolJsonSchema)
_RESUME_RESPONSE_SCHEMA.pop('title', None)
_RESUME_RESPONSE_SCHEMA.pop('description', None)
for _property in _RESUME_RESPONSE_SCHEMA['properties'].values():
    _property.pop('default', None)


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
        # Copied per interrupt: pydantic only rebuilds the outer dict for a `dict[str, Any]`
        # field, so sharing the constant would let a consumer mutating one interrupt's schema
        # alter every other interrupt's in the same process.
        response_schema=deepcopy(_RESUME_RESPONSE_SCHEMA),
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

    A denial `reason` only survives when the rest of the payload validates; a payload that
    fails validation denies with `ToolDenied`'s default message even if it carried one.

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
        if payload.edited_args is not None:
            return ToolApproved(override_args=payload.edited_args)
        return ToolApproved()

    # An empty-string reason is treated as absent, keeping ToolDenied's default message.
    if payload.reason:
        return ToolDenied(message=payload.reason)
    return ToolDenied()
