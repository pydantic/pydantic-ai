"""Progress events a workflow script emits through `log()` and `phase()`."""

from __future__ import annotations

from dataclasses import dataclass

from pydantic_ai import CapabilityEvent

DYNAMIC_WORKFLOW_EVENTS = 'dynamic_workflow'
"""Namespace of the `DynamicWorkflow` event family."""


@dataclass(kw_only=True)
class WorkflowLogEvent(CapabilityEvent, namespace=DYNAMIC_WORKFLOW_EVENTS, name='log'):
    """A workflow script called `log(message)`."""

    message: str
    """The logged text."""

    workflow: str | None = None
    """The saved workflow that logged it, or `None` for an ad-hoc script."""


@dataclass(kw_only=True)
class WorkflowPhaseEvent(CapabilityEvent, namespace=DYNAMIC_WORKFLOW_EVENTS, name='phase'):
    """A workflow script entered a phase, through `phase(title)` or `agent(..., phase=title)`."""

    title: str
    """The phase title."""

    workflow: str | None = None
    """The saved workflow that entered it, or `None` for an ad-hoc script."""
