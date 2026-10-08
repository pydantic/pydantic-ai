"""`/memory` for a Logfire-managed clai2 (hackathon): what the agent remembers here, and where it comes from."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from pydantic_clai2.builtin_plugins.fleet_memory import (
    PendingNote,
    PendingNotes,
    RepoNote,
    SharedMode,
    note_problem,
    personal_dirs,
)

USAGE = 'Usage: /memory [open FILE | edit [global] | forget FILE | propose FILE]'
MAIN = 'MEMORY.md'
_EXCERPT_LINES = 8
_MODES: dict[SharedMode, str] = {
    'review': 'review: an admin accepts each shared note in Logfire',
    'corroborate': "corroborate: a note is shared once a teammate's agent confirms it",
    'auto': 'auto: proposed notes are shared with everyone in the repository',
}


def _size(content: str) -> str:
    size = len(content.encode())
    return f'{size} B' if size < 1024 else f'{size / 1024:.1f} KB'


def _excerpt(content: str) -> list[str]:
    lines = [line for line in content.strip().splitlines() if line.strip()]
    shown = [f'      {line[:110]}' for line in lines[:_EXCERPT_LINES]]
    if len(lines) > _EXCERPT_LINES:
        shown.append(f'      … {len(lines) - _EXCERPT_LINES} more lines')
    return shown


@dataclass(kw_only=True)
class MemoryCommand:
    """The notebooks this session uses: personal (this repository's and every repository's), repo notes, pending."""

    directory: Path
    repo: Callable[[], str | None]
    notes: Callable[[], Sequence[RepoNote]]
    """The shared repo notes for this repository, as Logfire serves them."""
    mode: Callable[[], SharedMode]
    pending: PendingNotes
    propose: Callable[[str, str, str], str]
    withdraw: Callable[[PendingNote], None]
    """Records that a pending proposal was withdrawn, so the fleet miner can mark it stale."""
    edit: Callable[[str, str], Awaitable[str | None]]
    """`(text, title)` to the edited text, or `None` when the user cancelled: the shared multi-line editor."""
    link: str | None = None
    """Logfire's Memory tab, where admins edit repo notes."""
    synced: bool = False
    """Whether personal notes are synced through Logfire, so they follow the user to other machines."""

    async def __call__(self, args: list[str]) -> str:
        if not args:
            return self.overview()
        action, rest = args[0], args[1:]
        if action == 'open' and len(rest) == 1:
            return self.open(rest[0])
        if action == 'edit' and rest in ([], ['global']):
            return await self.edit_notebook(every_repo=bool(rest))
        if action == 'forget' and len(rest) == 1:
            return self.forget(rest[0])
        if action == 'propose' and len(rest) == 1:
            return self.propose_file(rest[0])
        raise ValueError(USAGE)

    def _dirs(self) -> tuple[Path, Path]:
        return personal_dirs(self.directory, self.repo())

    def overview(self) -> str:
        repo = self.repo()
        here, everywhere = self._dirs()
        lines = [f'Memory in this session. Shared notes are published by {_MODES[self.mode()]}.']
        for title, folder in (
            (f'Personal, {repo or "this directory"}', here),
            ('Personal, every repository', everywhere),
        ):
            where = 'synced via Logfire' if self.synced else 'only on this machine'
            lines.append(f'{title} ({where}; /memory edit{"" if folder == here else " global"}):')
            lines.extend(self._folder(folder) or ['  (empty)'])
        notes = self.notes()
        where = f' Edit them in Logfire: {self.link}' if self.link else ''
        lines.append(f'Repo notes, shared via Logfire (read-only here).{where}')
        for note in notes:
            provenance = f'  {note.provenance}' if note.provenance else ''
            lines.append(f'  {note.path}  {_size(note.content)}{provenance}')
            if note.path == MAIN:
                lines.extend(_excerpt(note.content))
        if not notes:
            lines.append(
                '  (none for this repository)'
                if repo
                else '  (no repository: repo notes need a GitHub or GitLab origin)'
            )
        pending = self.pending.pending(repo)
        if pending:
            lines.append('Pending, only you see these until they are shared:')
            for note in pending:
                lines.append(
                    f'  {note.path}  {_size(note.content)}  proposed {note.proposed_at[:10]}, awaiting {self.mode()}'
                )
        lines.append('The model sees each MEMORY.md (as excerpted) and the other files by name.')
        return '\n'.join(lines)

    def _folder(self, folder: Path) -> list[str]:
        lines: list[str] = []
        files = sorted(folder.glob('*.md')) if folder.is_dir() else []
        for path in files:
            content = path.read_text(encoding='utf-8', errors='replace')
            lines.append(f'  {path.name}  {_size(content)}')
            if path.name == MAIN:
                lines.extend(_excerpt(content))
        return lines

    def _personal(self, name: str) -> Path | None:
        here, everywhere = self._dirs()
        if name.startswith('global/'):
            candidates = [everywhere / name.removeprefix('global/')]
        else:
            candidates = [here / name, everywhere / name]
        return next((path for path in candidates if path.name.endswith('.md') and path.is_file()), None)

    def open(self, name: str) -> str:
        """A personal file (`global/NAME` for the every-repository notebook), a pending note, or a repo note."""
        if (path := self._personal(name)) is not None:
            return f'{path}\n\n{path.read_text(encoding="utf-8", errors="replace")}'
        if (
            pending := next((note for note in self.pending.pending(self.repo()) if note.path == name), None)
        ) is not None:
            return f'{name} (pending: only you see this)\n\n{pending.content}'
        if (note := next((note for note in self.notes() if note.path == name), None)) is not None:
            return f'{name} (repo note{f", {note.provenance}" if note.provenance else ""})\n\n{note.content}'
        raise ValueError(f'No memory file {name}. /memory lists them.')

    async def edit_notebook(self, *, every_repo: bool) -> str:
        here, everywhere = self._dirs()
        path = (everywhere if every_repo else here) / MAIN
        current = path.read_text(encoding='utf-8') if path.is_file() else ''
        edited = await self.edit(
            current, f'Your notes ({"every repository" if every_repo else self.repo() or "this directory"})'
        )
        if edited is None or edited == current:
            return f'No changes to {path}.'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(edited, encoding='utf-8')
        return f'Saved {path}. The agent sees it from the next model request.'

    def forget(self, name: str) -> str:
        """Delete a personal file, or withdraw a pending proposal."""
        repo = self.repo()
        if repo is not None and (withdrawn := self.pending.withdraw(repo=repo, path=name)) is not None:
            self.withdraw(withdrawn)
            return f'Withdrew your proposal {name}; it no longer applies to your sessions.'
        if (path := self._personal(name)) is not None:
            path.unlink()
            return f'Deleted {path}.'
        if any(note.path == name for note in self.notes()):
            raise ValueError(f'{name} is a shared repo note; ask an admin to change it in Logfire.')
        raise ValueError(f'No personal memory file or pending note {name}. /memory lists them.')

    def propose_file(self, name: str) -> str:
        """Share a personal file with the repository, through the same review as the agent's proposals."""
        path = self._personal(name)
        if path is None:
            raise ValueError(f'No personal memory file {name}. /memory lists them.')
        content = path.read_text(encoding='utf-8', errors='replace')
        shared_name = path.name
        if (problem := note_problem(shared_name, content)) is not None:
            raise ValueError(problem)
        return self.propose(shared_name, content, f'Shared by hand from personal memory ({name}) with /memory propose.')
