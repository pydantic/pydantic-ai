"""The orchestrator's session tools: find saved sessions of every project, and archive or unarchive them.

They work over `host.sessions`, the store the sidebar and `/resume` list. Archiving hides a session and deletes
nothing; the orchestrator has no tool that deletes a session or removes a worktree. Only the orchestrator's own
agent gets them (`web.orchestrator`), never the session's.
"""

from collections.abc import Awaitable, Callable, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path

from anyio import to_thread

from pydantic_ai_harness.step_persistence.conversations import ConversationSummary
from pydantic_clai2.builtin_plugins.web.protocol import ServerMessage, SessionsChanged
from pydantic_clai2.builtin_plugins.web.workspace_tools import reported
from pydantic_clai2.plugins import PluginHost, SessionCatalog
from pydantic_clai2.runtime.project_identity import ProjectIdentity, placements, project_identity

SCAN = 5000
"""The most saved sessions one `sessions_find` reads, newest first, before filtering by project and age."""

INSTRUCTIONS = """\
You also see the saved sessions of every project, the ones the sidebar and `/resume` list. `sessions_find` \
searches them by text, project or worktree, and age; archived ones only with `include_archived`. To clean up \
sessions, archive them with `sessions_archive`: it hides them from the sidebar and `/resume` and deletes \
nothing, and `sessions_unarchive` lists them again, so archiving needs no confirmation. Report exactly which \
sessions were archived and which were refused. No tool deletes a session, and archiving removes no folder or \
worktree; `workspace_destroy` removes only the checkout of a workspace you manage, as described above. When \
asked to delete sessions or remove other worktrees, archive their sessions instead and say that removing those \
worktrees is not available here."""


class SessionTools:
    """The tools, over the host's saved sessions; each change tells the sidebar to list them again."""

    def __init__(self, host: PluginHost[None], *, publish: Callable[[ServerMessage], None]) -> None:
        self.host = host
        self.publish = publish
        self._identities: dict[str, ProjectIdentity] = {}
        """What Git said of each workspace, read once per workspace."""

    def catalog(self) -> SessionCatalog:
        if self.host.sessions is None:
            raise RuntimeError('This CLAI does not save sessions.')
        return self.host.sessions

    def tools(self) -> list[Callable[..., Awaitable[str]]]:
        """Every tool, `reported`, for an agent."""
        return [reported(self.sessions_find), reported(self.sessions_archive), reported(self.sessions_unarchive)]

    async def sessions_find(
        self,
        query: str = '',
        project: str = '',
        older_than_days: float | None = None,
        include_archived: bool = False,
        limit: int = 50,
    ) -> str:
        """Saved sessions of every project, newest first: ID, title, project, worktree, last active, archived.

        `query` matches titles and conversation text; `project` a project's name, or a worktree's branch or folder;
        `older_than_days` keeps only sessions last active longer ago than that.
        """
        catalog = self.catalog()
        entries = await catalog.listing(query=query, limit=SCAN, archived=None if include_archived else False)
        now = datetime.now(UTC)
        if older_than_days is not None:
            cutoff = now - timedelta(days=older_than_days)
            entries = [entry for entry in entries if entry.updated_at < cutoff]
        places = await to_thread.run_sync(self._places, entries)
        needle = project.casefold()
        placed = [(entry, places[entry.workspace]) for entry in entries]
        found = [
            (entry, place)
            for entry, place in placed
            if needle in f'{place.name}\n{place.checkout}\n{entry.workspace}'.casefold()
        ]
        if not found:
            return 'No saved sessions match.'
        shown = found[: max(limit, 1)]
        current = self.host.conversation.conversation_id
        lines = [
            f'{len(shown)} of {len(found)} matching sessions, newest first.',
            '| id | title | project | worktree | last active | archived |',
            '|---|---|---|---|---|---|',
        ]
        lines += [
            f'| {entry.id} | {_cell(entry.title)}{" (current session)" if entry.id == current else ""} '
            f'| {_cell(place.name)} | {_cell(place.checkout or entry.workspace)} | {_active(entry.updated_at, now)} '
            f'| {"yes" if entry.archived_at is not None else "no"} |'
            for entry, place in shown
        ]
        return '\n'.join(lines)

    async def sessions_archive(self, ids: list[str]) -> str:
        """Archive saved sessions by ID: hidden from the sidebar and `/resume`, nothing deleted.

        `sessions_unarchive` undoes it. The current session and running ones are refused, with the reason.
        """
        return await self._change(ids, archived=True)

    async def sessions_unarchive(self, ids: list[str]) -> str:
        """List archived sessions again, by ID."""
        return await self._change(ids, archived=False)

    async def _change(self, ids: Sequence[str], *, archived: bool) -> str:
        catalog = self.catalog()
        done: list[str] = []
        refused: list[str] = []
        for conversation_id in dict.fromkeys(ids):
            try:
                summary = await (catalog.archive(conversation_id) if archived else catalog.unarchive(conversation_id))
            except (LookupError, ValueError) as exc:
                refused.append(f'- {conversation_id}: {exc}')
            else:
                done.append(f'- {summary.id} ({_cell(summary.title)})')
        if done:
            self.publish(SessionsChanged())
        verb = 'Archived' if archived else 'Unarchived'
        lines = [f'{verb} {len(done)} session{"" if len(done) == 1 else "s"}' + (':' if done else '.'), *done]
        if refused:
            lines += [f'Refused {len(refused)}:', *refused]
        return '\n'.join(lines)

    def _places(self, entries: Sequence[ConversationSummary]) -> dict[str, ProjectIdentity]:
        """Each workspace's project and worktree, as the sidebar groups them; reads Git, so call it off the loop."""
        for workspace in {entry.workspace for entry in entries}:
            # A folder removed or recreated since it was last read resolves again, as the sidebar does.
            known = self._identities.get(workspace)
            if known is None or Path(workspace).exists() == known.missing:
                self._identities[workspace] = project_identity(workspace)
        return placements(entries, identities=self._identities)


def _cell(text: str) -> str:
    """Text for one table cell: one line, without the column separator."""
    return ' '.join(text.split()).replace('|', '/')


def _active(moment: datetime, now: datetime) -> str:
    days = (now - moment).days
    age = 'today' if days < 1 else f'{days} day{"" if days == 1 else "s"} ago'
    return f'{moment.astimezone():%Y-%m-%d %H:%M} ({age})'
