"""The `run_code` REPL state saved beside each conversation, so a session keeps its sandbox between prompts.

Harness `CodeMode` starts every agent run with a fresh REPL, and each prompt is a run. `ReplStates`
is the `CodeModeReplStore` CLAI gives it: one file per saved conversation, in a `sessions.repl`
directory beside `sessions.db`, so `/resume` and `--session-id` bring the REPL back with the history.
Forks get their own copy and deleting a session removes its file.

Only the session's own run keeps its REPL. A delegated sub-agent runs under another conversation
ID, a random one when it is one-shot, so its REPL stays with that run rather than leaving a file
behind for every delegation.
"""

import logging
import os
import re
import tempfile
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path

from anyio.to_thread import run_sync

from pydantic_ai import RunContext

MAX_STATE_BYTES = 32 * 1024 * 1024
"""A larger REPL is not saved: the dump holds every live value, and a runaway one should not fill the disk."""

_ID = re.compile(r'[0-9a-z_-]+')
"""Conversation IDs that are safe as file names, lowercase so a case-insensitive file system cannot
merge two of them. CLAI's own IDs are lowercase UUIDs; any other ID keeps its REPL to the run."""

_logger = logging.getLogger(__name__)


@dataclass(frozen=True, kw_only=True)
class ReplStates:
    """REPL dumps in `directory`, one per conversation, for the conversation `session_id` names."""

    directory: Path
    session_id: Callable[[], str | None]
    """The saved conversation running now; a run under any other conversation ID is not kept."""
    _deletions: dict[str, int] = field(default_factory=dict[str, int], init=False, repr=False, compare=False)
    """How often each ID was deleted, so a write already abandoned to its thread cannot bring a deleted REPL back."""

    def path(self, conversation_id: str) -> Path | None:
        """Where `conversation_id`'s REPL is kept, or `None` when its ID cannot name a file."""
        return self.directory / f'{conversation_id}.bin' if _ID.fullmatch(conversation_id) else None

    def keeps(self, ctx: RunContext[object]) -> bool:
        """Whether this run is the session's own, whose REPL is saved and restored."""
        return self._own(ctx) is not None

    def _own(self, ctx: RunContext[object]) -> Path | None:
        conversation_id = ctx.conversation_id
        if conversation_id is None or conversation_id != self.session_id():
            return None
        return self.path(conversation_id)

    async def load(self, ctx: RunContext[object]) -> bytes | None:
        """The REPL saved for this session; an unreadable file starts a fresh REPL."""
        path = self._own(ctx)
        if path is None:
            return None
        try:
            return await run_sync(path.read_bytes, abandon_on_cancel=True)
        except FileNotFoundError:
            return None
        except OSError as exc:
            _logger.warning('Could not read the saved REPL state %s: %s', path, exc)
            return None

    async def save(self, ctx: RunContext[object], state: bytes | None) -> None:
        """Replace this session's REPL, or forget it when `state` is `None` or too large to keep."""
        path = self._own(ctx)
        if path is None:
            return
        if state is not None and len(state) > MAX_STATE_BYTES:
            _logger.warning(
                'Not saving the REPL state: %d bytes is over the %d byte limit', len(state), MAX_STATE_BYTES
            )
            state = None
        try:
            if state is None:
                await run_sync(partial(path.unlink, missing_ok=True), abandon_on_cancel=True)
            else:
                # Abandoned rather than awaited when harness gives up on the save; the write is atomic.
                deletions = self._deletions.get(path.stem, 0)
                await run_sync(self._write_unless_deleted, path, state, deletions, abandon_on_cancel=True)
        except OSError as exc:
            _logger.warning('Could not save the REPL state %s: %s', path, exc)

    def copy(self, *, source: str, target: str) -> None:
        """Give `target` the REPL `source` has now, for a fork.

        A hard link costs nothing and is a true copy, since every save replaces the file rather than
        writing into it. Where linking fails, such as on a file system without hard links, the fork
        starts with a fresh REPL.
        """
        source_path, target_path = self.path(source), self.path(target)
        if source_path is None or target_path is None:
            return
        try:
            # Cleared even when `source` has no REPL, so a file left under `target` cannot reach the fork.
            target_path.unlink(missing_ok=True)
            if source_path.exists():
                os.link(source_path, target_path)
        except OSError as exc:
            _logger.warning('Could not copy the REPL state to %s: %s', target_path, exc)

    def delete(self, conversation_id: str) -> None:
        """Remove `conversation_id`'s REPL with the session, including one a write still in flight would leave."""
        path = self.path(conversation_id)
        if path is None:
            return
        self._deletions[conversation_id] = self._deletions.get(conversation_id, 0) + 1
        try:
            path.unlink(missing_ok=True)
        except OSError as exc:
            _logger.warning('Could not delete the REPL state %s: %s', path, exc)

    def _write_unless_deleted(self, path: Path, state: bytes, deletions: int) -> None:
        """Write `state`, then remove it again if `delete` ran meanwhile.

        Either `delete` counted before the check below, which then removes the file, or it unlinks
        after this replaced the file; both orders leave no REPL behind.
        """
        _write(path, state)
        if self._deletions.get(path.stem, 0) != deletions:
            path.unlink(missing_ok=True)


def _write(path: Path, state: bytes) -> None:
    """Replace `path` atomically, so a reader or a hard-linked fork never sees a partial file."""
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    # `mkstemp` creates the file owner-only, like `sessions.db`: the state holds whatever the REPL did.
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=f'.{path.name}.')
    try:
        with os.fdopen(descriptor, 'wb') as stream:
            stream.write(state)
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise
