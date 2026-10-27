"""The sidebar's saved sessions: each one's row, and the checkouts of a project."""

from collections.abc import Mapping
from pathlib import PurePath

from pydantic_ai_harness.step_persistence.conversations import ConversationSummary
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
        state=state,
        messages=summary.message_count,
        model=summary.model,
        current=current,
        archived=summary.archived_at is not None,
    )


def worktrees(
    found: Mapping[str, tuple[ProjectIdentity, list[SessionEntry]]], *, here: ProjectIdentity
) -> list[Worktree]:
    """A project's checkouts, the current one first and removed ones last, each labelled by its branch or else its folder.

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
    return sorted(worktrees, key=lambda worktree: (not worktree.current, worktree.missing))
