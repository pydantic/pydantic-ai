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
The overlap is the longest suffix/prefix match, so byte-identical messages at
distinct positions remain in the record, except where a snapshot that dropped a prefix
also repeats the exchange that ends the accumulated history. A substrate that keeps an append-only
entry log (the session-tree direction of pydantic/pydantic-ai#9001) can implement
`HistorySource` directly via replay and replace the adapter without touching the
search layer.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import replace
from typing import Protocol, runtime_checkable

from pydantic_ai.messages import (
    InstructionBaselinePart,
    InstructionDeltaPart,
    ModelMessage,
    ModelMessagesTypeAdapter,
    ModelRequest,
    ModelRequestPart,
    SystemPromptPart,
)
from pydantic_ai_harness.compaction._summary import without_summaries
from pydantic_ai_harness.step_persistence import ContinuableSnapshot, RunRecord

SUMMARY_PREFIX = 'Summary of previous conversation:\n\n'
"""The exact prefix an earlier `SummarizingCompaction` wrote into the summary system prompt it inserted.

Summaries are now a `CompactionPart`; this only recognizes ones in histories saved before that.

Byte-for-byte mirror of `pydantic_ai_harness.compaction._summarizing_compaction.SUMMARY_PREFIX`,
including the blank line: that is the marker compaction itself matches on to recognize its own
prior summaries (`extract_previous_summary`), and `SystemPromptPart` carries no metadata field
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
    `FileStepStore`, `SqliteStepStore`, `MongoStepStore`, and `PostgresStepStore` all
    satisfy it.
    `list_snapshots` is not part of the `StepStore` protocol yet -- the shipped
    stores implement it as a plain method; promoting it into the protocol is
    proposed alongside the session-tree evolution (pydantic/pydantic-ai#9001).
    """

    async def list_runs(
        self,
        *,
        parent_run_id: str | None = None,
        conversation_id: str | None = None,
    ) -> list[RunRecord]: ...  # pragma: no cover

    async def list_snapshots(self, *, run_id: str) -> list[ContinuableSnapshot]: ...  # pragma: no cover


def _canonical_message(message: ModelMessage) -> ModelMessage:
    """Return a copy of `message` with framework-stamped fields cleared.

    pydantic-ai stamps `timestamp`, `run_id`, and `conversation_id` onto a
    message (and fills `instructions` on requests) after the boundary snapshot
    that first carries it is serialized, so the same logical message has
    different bytes before and after stamping. It likewise appends instruction
    bookkeeping to a request that an earlier snapshot already holds: an
    `InstructionDeltaPart` when an `on_change='append'` block changes, an
    `InstructionBaselinePart` when an instruction window opens or closes.
    Clearing exactly those makes the dedup key stable across
    the boundary; every other field keeps participating in the key, so two
    genuinely distinct messages sharing parts stay distinct.
    """
    if isinstance(message, ModelRequest):
        parts = [part for part in message.parts if not isinstance(part, InstructionDeltaPart | InstructionBaselinePart)]
        return replace(message, parts=parts, timestamp=None, run_id=None, conversation_id=None, instructions=None)
    # `ModelMessage` is exactly the `ModelRequest | ModelResponse` union, so any
    # other message is a `ModelResponse`.
    return replace(message, timestamp=None, run_id=None, conversation_id=None)


def message_hash(message: ModelMessage) -> str:
    """Return a stable content hash of a single message.

    Dedup keys off serialized content, not object identity: consecutive snapshots
    re-serialize the same growing history, and durable executors (Temporal, DBOS)
    re-instantiate messages between steps, so identity-based dedup would re-append.
    The hash is computed over `ModelMessagesTypeAdapter` bytes, so it is stable
    across snapshot round-trips and replay. Framework-stamped metadata is cleared
    first, see `_canonical_message`, so the overlap survives the stamping that
    pydantic-ai applies after the boundary snapshot is saved.
    """
    return hashlib.sha256(ModelMessagesTypeAdapter.dump_json([_canonical_message(message)])).hexdigest()


def _overlap_length(history: list[str], snapshot: list[str]) -> int:
    """Return the longest suffix of `history` matching a prefix of `snapshot`."""
    for length in range(min(len(history), len(snapshot)), 0, -1):
        if history[-length:] == snapshot[:length]:
            return length
    return 0


def without_summary_artifacts(messages: Sequence[ModelMessage]) -> list[ModelMessage]:
    """*messages* without compaction summary artifacts, which are never indexed.

    A summary is the part `SummarizingCompaction` marks as its own; a `CompactionPart` an
    application wrote itself stays searchable. In a history saved before `SummarizingCompaction` wrote
    one, the summary is a system prompt opening with `SUMMARY_PREFIX`. Only the summary part is removed,
    since continuing a history merges the summary's request with the real turns next to it; a request
    left with nothing else is dropped.
    """
    out: list[ModelMessage] = []
    for message in without_summaries(messages):
        if isinstance(message, ModelRequest) and any(_is_legacy_summary(part) for part in message.parts):
            parts = [part for part in message.parts if not _is_legacy_summary(part)]
            if not parts:
                continue
            message = replace(message, parts=parts)
        out.append(message)
    return out


def _is_legacy_summary(part: ModelRequestPart) -> bool:
    """Whether *part* is the summary system prompt an earlier `SummarizingCompaction` wrote."""
    return isinstance(part, SystemPromptPart) and part.content.startswith(SUMMARY_PREFIX)


class SnapshotHistorySource:
    """Recover each run's durable message record from its persisted snapshots.

    Reads the same store instance a `StepPersistence` capability writes to. Each
    snapshot holds the full live history at one step boundary; compaction edits
    persist forward, so later snapshots may have replaced early originals with a
    summary. Iterating snapshots in write order, skipping summary artifacts, and
    removing only the overlap between accumulated history's suffix and each
    snapshot's prefix yields the originals plus everything compaction never touched.
    Repeated byte-identical messages at distinct sequence positions are preserved, unless
    a snapshot that dropped a prefix also repeats the exchange that ends the accumulated
    history: the longest overlap then absorbs the repeat.

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
        if not isinstance(store, SnapshotStore):
            raise TypeError(
                f'{type(store).__name__} is not a supported search substrate: SnapshotHistorySource '
                'needs a store providing both `list_runs` and `list_snapshots`. The shipped '
                'InMemoryStepStore, FileStepStore, SqliteStepStore, MongoStepStore, and '
                'PostgresStepStore satisfy this.'
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
            messages = without_summary_artifacts(snapshot.messages)
            snapshot_hashes = [message_hash(message) for message in messages]
            overlap = _overlap_length(history_hashes, snapshot_hashes)
            # Keep the later copies of the overlap: they carry what core added after the
            # earlier snapshot, such as an instruction change, which the hash leaves out.
            history[len(history) - overlap :] = messages[:overlap]
            history.extend(messages[overlap:])
            history_hashes.extend(snapshot_hashes[overlap:])
        return history
