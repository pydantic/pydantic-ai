"""Committed checkpoint notifications. The store, not event delivery, owns durability."""

from dataclasses import dataclass

from pydantic_ai import CapabilityEvent

from ._types import SnapshotState


@dataclass(kw_only=True)
class SnapshotSaved(CapabilityEvent, namespace='step_persistence', name='snapshot_saved'):
    """A checkpoint write completed. Replayed runs may deliver this notification again."""

    persistence_run_id: str
    conversation_id: str | None
    step_index: int
    state: SnapshotState
