"""History sources: where the search corpus comes from.

The search layer never persists anything itself. It consumes a `HistorySource` --
"enumerate persisted runs, yield each run's durable message record" -- and ships
`SnapshotHistorySource`, an adapter that recovers that record from the snapshots a
`pydantic_ai_harness.step_persistence.StepPersistence` capability already writes.

The adapter exists because today's step-persistence substrate stores per-boundary
full-history snapshots, and compaction strategies that persist their edits (e.g.
`SummarizingCompaction`) carry those edits into later snapshots: the latest snapshot
is then post-compaction, but earlier snapshots of the same run still hold the
originals, so reconciling each snapshot's carried-forward prefix against the
accumulated history and excluding compaction artifacts recovers the durable record.
The overlap is reconciled by sequence position, so byte-identical messages at
distinct positions remain in the record. A substrate that keeps an append-only
entry log (the session-tree direction of pydantic-ai-harness#321) can implement
`HistorySource` directly via replay and replace the adapter without touching the
search layer.
"""

from __future__ import annotations

import hashlib
from typing import Protocol, runtime_checkable

from pydantic_ai.messages import (
    ModelMessage,
    ModelMessagesTypeAdapter,
    ModelRequest,
    SystemPromptPart,
)

from pydantic_ai_harness.step_persistence import ContinuableSnapshot, RunRecord

SUMMARY_PREFIX = 'Summary of previous conversation:\n\n'
"""The exact prefix a `SummarizingCompaction` writes into the summary artifact it inserts.

Byte-for-byte mirror of `pydantic_ai_harness.compaction._summarizing_compaction._SUMMARY_PREFIX`,
including the blank line: that is the marker compaction itself matches on to recognize its own
prior summaries (`_extract_previous_summary`), and `SystemPromptPart` carries no metadata field
to mark artifacts with instead. Matching the full literal keeps a user-authored system prompt
that merely opens with the same sentence inside the corpus.

Kept as a local literal rather than an import so this capability does not couple to compaction
internals: the corpus holds the originals a summary replaced, never the derived summary, so
snapshots taken after compaction must contribute only what the earlier snapshots did not
already carry.
"""


@runtime_checkable
class HistorySource(Protocol):
    """A source of persisted conversation history for the search corpus.

    This is the seam between the search layer and whatever substrate persists
    history. `SnapshotHistorySource` implements it over step-persistence
    snapshot stores; an event-sourced substrate can implement it via replay.
    """

    async def list_runs(self) -> list[RunRecord]:
        """Return all persisted runs, sorted by `started_at` ascending."""
        ...  # pragma: no cover

    async def run_history(self, *, run_id: str) -> list[ModelMessage]:
        """Return one run's durable message record, in message order.

        The record contains the original messages, including any that compaction
        later dropped from the live context, and excludes derived compaction
        artifacts (e.g. summary messages). An unknown `run_id` yields `[]`.
        """
        ...  # pragma: no cover


@runtime_checkable
class SnapshotStore(Protocol):
    """The narrow read surface `SnapshotHistorySource` needs from a snapshot store.

    A structural subset of the step-persistence stores: `InMemoryStepStore`,
    `FileStepStore`, `SqliteStepStore`, and `MongoStepStore` all satisfy it.
    `list_snapshots` is not part of the `StepStore` protocol yet -- the shipped
    stores implement it as a plain method; promoting it into the protocol is
    proposed alongside the session-tree evolution (pydantic-ai-harness#321).
    """

    async def list_runs(
        self,
        *,
        parent_run_id: str | None = None,
        conversation_id: str | None = None,
    ) -> list[RunRecord]: ...  # pragma: no cover

    async def list_snapshots(self, *, run_id: str) -> list[ContinuableSnapshot]: ...  # pragma: no cover


def message_hash(message: ModelMessage) -> str:
    """Return a stable content hash of a single message.

    Dedup keys off serialized content, not object identity: consecutive snapshots
    re-serialize the same growing history, and durable executors (Temporal, DBOS)
    re-instantiate messages between steps, so identity-based dedup would re-append.
    The hash is computed over `ModelMessagesTypeAdapter` bytes, so it is stable
    across snapshot round-trips and replay.
    """
    return hashlib.sha256(ModelMessagesTypeAdapter.dump_json([message])).hexdigest()


def _overlap_length(history: list[str], snapshot: list[str]) -> int:
    """Return the longest suffix of `history` matching a prefix of `snapshot`."""
    for length in range(min(len(history), len(snapshot)), 0, -1):
        if history[-length:] == snapshot[:length]:
            return length
    return 0


def is_summary_artifact(message: ModelMessage) -> bool:
    """Return whether a message is a compaction summary artifact (never indexed)."""
    if not isinstance(message, ModelRequest):
        return False
    return any(isinstance(part, SystemPromptPart) and part.content.startswith(SUMMARY_PREFIX) for part in message.parts)


class SnapshotHistorySource:
    """Recover each run's durable message record from its persisted snapshots.

    Reads the same store instance a `StepPersistence` capability writes to. Each
    snapshot holds the full live history at one step boundary; compaction edits
    persist forward, so later snapshots may have replaced early originals with a
    summary. Iterating snapshots in write order, skipping summary artifacts, and
    removing only the overlap between accumulated history's suffix and each
    snapshot's prefix yields the originals plus everything compaction never touched.
    Repeated byte-identical messages at distinct sequence positions are preserved.

    The shipped stores' `list_snapshots` defaults to `complete` snapshots only
    (mirroring `latest_snapshot`), so `interrupted` captures -- which can carry
    unsettled tool work and synthesized tool returns -- stay out of the corpus.
    """

    def __init__(self, store: SnapshotStore) -> None:
        # Fail at construction, not mid-search, when a store lacks the read seam.
        # `list_snapshots` is not part of the `StepStore` protocol, so a
        # third-party store can satisfy `StepStore` without it; without this
        # check the missing seam surfaces as an obscure `AttributeError` deep
        # inside a tool call.
        if not isinstance(store, SnapshotStore):  # pyright: ignore[reportUnnecessaryIsInstance]
            raise TypeError(
                f'{type(store).__name__} is not a supported search substrate: SnapshotHistorySource '
                'needs a store providing both `list_runs` and `list_snapshots`. The shipped '
                'InMemoryStepStore, FileStepStore, SqliteStepStore, and MongoStepStore satisfy this.'
            )
        self._store = store

    async def list_runs(self) -> list[RunRecord]:
        """Return all persisted runs, sorted by `started_at` ascending."""
        return await self._store.list_runs()

    async def run_history(self, *, run_id: str) -> list[ModelMessage]:
        """Union one run's snapshots into its durable message record."""
        history: list[ModelMessage] = []
        history_hashes: list[str] = []
        for snapshot in await self._store.list_snapshots(run_id=run_id):
            messages = [message for message in snapshot.messages if not is_summary_artifact(message)]
            snapshot_hashes = [message_hash(message) for message in messages]
            overlap = _overlap_length(history_hashes, snapshot_hashes)
            history.extend(messages[overlap:])
            history_hashes.extend(snapshot_hashes[overlap:])
        return history
