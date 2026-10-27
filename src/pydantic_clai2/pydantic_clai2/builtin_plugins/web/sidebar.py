"""The sidebar's saved sessions: each one's row, and the checkouts of a project."""

import os
import sys
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from pathlib import Path, PurePath

from pydantic_ai_harness.step_persistence.conversations import (
    ConversationConflict,
    ConversationSummary,
    ensure_inactive,
)
from pydantic_clai2.builtin_plugins.web.protocol import SessionEntry, SessionState, Worktree
from pydantic_clai2.plugins import LiveSession
from pydantic_clai2.runtime.project_identity import ProjectIdentity


def session_entry(summary: ConversationSummary, *, live: LiveSession | None, busy: bool, current: bool) -> SessionEntry:
    """A saved session's row: `running` from the live session holding it; `busy` only for another process's turn."""
    state: SessionState = summary.outcome
    if live is not None:
        state = 'running' if live.running else 'ready' if summary.outcome == 'running' else summary.outcome
    elif summary.outcome == 'running':
        state = 'busy' if busy else 'cancelled'
    return SessionEntry(
        id=summary.id,
        title=summary.title,
        workspace=summary.workspace,
        updated_at=summary.updated_at,
        created_at=summary.created_at or summary.updated_at,
        state=state,
        pid=summary.owner_pid if state == 'busy' else None,
        messages=summary.message_count,
        model=summary.model,
        current=current,
        archived=summary.archived_at is not None,
    )


def worktrees(
    found: Mapping[str, tuple[ProjectIdentity, list[SessionEntry]]], *, here: ProjectIdentity
) -> list[Worktree]:
    """A project's checkouts, the current one first, each labelled by its branch or else its folder.

    A label two of them share, such as two removed folders of one name, shows the folder's path instead.
    """
    labels = {root: identity.checkout or PurePath(root).name or root for root, (identity, _) in found.items()}
    worktrees = [
        Worktree(
            root=root,
            label=labels[root] if list(labels.values()).count(labels[root]) == 1 else root,
            current=(identity.key, root) == (here.key, here.root),
            missing=identity.missing,
            sessions=sessions,
        )
        for root, (identity, sessions) in found.items()
    ]
    return sorted(worktrees, key=lambda worktree: not worktree.current)


CLOCK_SLACK = timedelta(seconds=2)
"""How much later than a turn's save its process may seem to start: `/proc` times it in whole seconds and ticks."""


def busy_elsewhere(summary: ConversationSummary) -> bool:
    """Whether another live CLAI process runs this saved session now, rather than one that died mid-turn.

    A process that started after the turn was saved reuses the PID of the one that ran it, which is gone.
    """
    if summary.owner_pid is None or summary.owner_pid == os.getpid():
        # Not live here, so this process's turn in it ended without saving its outcome.
        return False
    try:
        ensure_inactive(summary)
    except ConversationConflict:
        started = process_started(summary.owner_pid)
        return started is None or started <= summary.updated_at + CLOCK_SLACK
    return False


def process_started(pid: int, proc: Path = Path('/proc')) -> datetime | None:
    """When process `pid` started, from Linux's `/proc`; `None` elsewhere or when it cannot be read."""
    if sys.platform != 'linux':  # pragma: no cover
        return None
    try:
        stat = (proc / str(pid) / 'stat').read_text()
        boot = next(
            int(line.split()[1]) for line in (proc / 'stat').read_text().splitlines() if line.startswith('btime ')
        )
    except (OSError, StopIteration):
        return None
    # The command name, in parentheses, may hold spaces; `starttime` is the 22nd field, the 20th after it.
    ticks = int(stat.rpartition(')')[2].split()[19])
    return datetime.fromtimestamp(boot + ticks / os.sysconf('SC_CLK_TCK'), UTC)
