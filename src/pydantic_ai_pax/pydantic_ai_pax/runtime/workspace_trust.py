"""Whether you trust the folder a session works in, decided once per repository and kept in your settings.

Every folder starts untrusted. The interactive shell asks the first time it starts in one (`ui/menus/trust_prompt.py`),
and `/trust` changes the answer later. The answer sets the `authority` of the session's `LocalWorkspace`: trusted, the
folder's files are `'operator'`, as yours; untrusted, they are `'none'`, the repository's. Either way the agent reads,
edits and runs code there as before: trust decides only the voice the folder's own files reach the model in.

The answer is kept in the user's settings database, never in the repository, so a repository cannot trust itself;
and nothing the repository holds is read to decide whether to ask. Inside a Git repository it is kept for the
repository's main checkout, so its worktrees and subdirectories share it, while a repository nested inside it (a
submodule, a clone) gets its own. Outside a repository it is kept for the folder and covers its subfolders. In your
home folder, or the filesystem root, it holds for the session only, so Pax asks again next time.

Harness `RepoContext` and `Skills` read `Workspace.authority`: an untrusted folder's instruction files and skills
reach the model as the repository's, not in the instructions. TODO(#11839): disk sub-agents don't read it yet, so an
untrusted folder's agents still load as before. TODO: a `/sandbox` conversation's workspace comes from the sandbox
plugin, which doesn't take this decision yet.
"""

import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Self

from pydantic_ai.messages import Authority
from pydantic_ai_pax.config import TrustLevel
from pydantic_ai_pax.config.project_settings import PROJECT_FILE, repository_directories
from pydantic_ai_pax.config.settings_store import SettingsStore
from pydantic_ai_pax.ui.rendering.tool_output import terminal_text

_INSTRUCTION_FILES = ('AGENTS.md', 'CLAUDE.md')
"""What `repo_context` loads from each folder: the first of these it finds there."""
_SKILL_FOLDERS = (Path('.agents') / 'skills', Path('.claude') / 'skills')
_AGENT_FOLDERS = (Path('.agents') / 'agents', Path('.claude') / 'agents', Path('.codex') / 'agents')
_APPROVED_SEPARATELY = (
    (PROJECT_FILE, '/project approve, /plugins enable'),
    (Path('.clai') / 'mcp_servers.json', '/mcp trust'),
    (Path('.mcp.json'), '/mcp trust'),
)
"""Repository files that wait for their own approval, trusted or not."""


@dataclass(frozen=True, kw_only=True)
class TrustScope:
    """Which folder a trust decision is kept for, and whether it is kept beyond the session."""

    key: Path
    """The repository's main checkout inside Git, else the folder the session works in; resolved."""
    repository: bool
    """Whether `key` is a Git repository, whose decision covers exactly it; otherwise it covers subfolders too."""
    remembered: bool
    """`False` for your home folder and the filesystem root: a decision there holds for the session only."""


def trust_scope(directory: Path) -> TrustScope:
    """The scope of the trust decision for a session working in `directory`."""
    start = directory.resolve()
    if _session_only(start):
        return TrustScope(key=start, repository=False, remembered=False)
    root = _repository_root_for_trust(start)
    key = start if root is None else _main_checkout(root)
    return TrustScope(key=key, repository=root is not None, remembered=not _session_only(key))


def _repository_root_for_trust(start: Path) -> Path | None:
    """The nearest repository root at or above `start`, stopping before `$HOME` and `/`, which never cover subfolders."""
    for folder in (start, *start.parents):
        if _session_only(folder):
            return None
        if os.path.lexists(folder / '.git'):
            return folder
    return None  # pragma: no cover -- every resolved path reaches `/`, which is session-only.


def _main_checkout(root: Path) -> Path:
    """The main checkout of the repository at `root`, for a linked worktree; otherwise `root` itself.

    A `.git` folder makes `root` its own repository. A `.git` file names a Git directory elsewhere, which a
    downloaded folder can point at a trusted repository's, so `root` takes that repository's decision only when
    that repository's `worktrees/` entry points back to `root / '.git'`. A submodule has no worktree `commondir`,
    so it gets its own decision.
    """
    dot_git = root / '.git'
    if dot_git.is_symlink() or not dot_git.is_file():
        return root
    target = _gitdir_target(dot_git)
    if target is None:
        return root
    wt_gitdir = (root / target).resolve()
    back = _read_first_line(wt_gitdir / 'gitdir')
    common_raw = _read_first_line(wt_gitdir / 'commondir')
    if back is None or common_raw is None:
        return root
    back_path = wt_gitdir / back
    if back_path.name != '.git' or back_path.parent.resolve() != root:
        return root
    common = (wt_gitdir / common_raw).resolve()
    if wt_gitdir.parent != (common / 'worktrees').resolve():
        return root
    return common.parent if _is_main_git_dir(common) else root


def _gitdir_target(dot_git: Path) -> str | None:
    line = _read_first_line(dot_git)
    if line is None or not line.startswith('gitdir:'):
        return None
    return line.removeprefix('gitdir:').strip() or None


def _read_first_line(path: Path) -> str | None:
    try:
        if not path.is_file():
            return None
        with path.open(encoding='utf-8') as file:
            first = file.readline(4096)
    except (OSError, UnicodeDecodeError):
        return None
    return first.strip() or None


def _is_main_git_dir(common: Path) -> bool:
    main_dot_git = common.parent / '.git'
    return (
        common.name == '.git'
        and not main_dot_git.is_symlink()
        and main_dot_git.is_dir()
        and main_dot_git.resolve() == common
    )


def _session_only(key: Path) -> bool:
    if key.parent == key:
        return True
    try:
        return key == Path.home().resolve()
    except RuntimeError:  # No resolvable home directory.
        return False


def saved_trust(store: SettingsStore, scope: TrustScope) -> tuple[bool, Path] | None:
    """The decision in effect for `scope` and the folder it was made for; `None` when you never decided.

    Outside a repository, the nearest folder at or above `scope.key` with a decision gives it, stopping
    below your home folder and the filesystem root, whose decisions are never kept.
    """
    if not scope.remembered:
        return None
    folders = (scope.key,) if scope.repository else (scope.key, *scope.key.parents)
    for folder in folders:
        if _session_only(folder):
            break
        if (trusted := store.workspace_trust(folder)) is not None:
            return trusted, folder
    return None


@dataclass(kw_only=True)
class WorkspaceTrust:
    """One session's trust in its folder: the decision in effect, and where it came from."""

    scope: TrustScope
    trusted: bool = False
    decided_for: Path | None = None
    """The folder the decision in effect was made for; `None` while you have not decided."""

    @classmethod
    def from_scope(cls, store: SettingsStore, scope: TrustScope) -> Self:
        """The saved trust for `scope`; untrusted when you never decided."""
        trust = cls(scope=scope)
        trust.refresh(store)
        return trust

    @classmethod
    def saved(cls, store: SettingsStore, directory: Path) -> Self:
        """The trust for a session in `directory`, as you decided it before; untrusted when you never did."""
        return cls.from_scope(store, trust_scope(directory))

    def refresh(self, store: SettingsStore) -> None:
        """Reload this scope's saved decision from `store`, if one exists for it or an ancestor."""
        self.apply_saved(saved_trust(store, self.scope))

    def apply_saved(self, saved: tuple[bool, Path] | None) -> None:
        """Apply a `saved_trust` lookup result to this instance."""
        if saved is None:
            self.trusted = False
            self.decided_for = None
            return
        self.trusted, self.decided_for = saved

    @property
    def decided(self) -> bool:
        """Whether a decision is in effect, made now or in an earlier session."""
        return self.decided_for is not None

    @property
    def level(self) -> TrustLevel:
        return 'trusted' if self.trusted else 'untrusted'

    @property
    def authority(self) -> Authority:
        """The `authority` the session's `LocalWorkspace` gives the folder's files."""
        return 'operator' if self.trusted else 'none'

    def decide(self, store: SettingsStore, *, trusted: bool) -> None:
        """Trust the folder or not, keeping the decision in `store` unless the scope holds it for the session only."""
        if self.scope.remembered:
            store.set_workspace_trust(self.scope.key, trusted=trusted)
        self.trusted = trusted
        self.decided_for = self.scope.key


@dataclass(frozen=True, kw_only=True)
class RepositoryFiles:
    """What the repository around a folder holds that trust is about, by path relative to that folder."""

    trusted: tuple[str, ...] = ()
    """Instruction files, skill folders and agent folders: what reaches the model as yours once trusted."""
    approved_separately: tuple[tuple[str, str], ...] = ()
    """Each file that waits for its own approval, trusted or not, with the command that approves it."""


def repository_files(directory: Path) -> RepositoryFiles:
    """The `RepositoryFiles` of the folders from `directory` up to the repository root, as `repo_context` reads them.

    Only names are checked, never contents.
    """
    start = directory.resolve()
    trusted: list[str] = []
    approved: list[tuple[str, str]] = []
    for folder in repository_directories(start):
        # Each folder contributes its first instruction file, as `repo_context` loads it.
        name = next((name for name in _INSTRUCTION_FILES if os.path.isfile(folder / name)), None)
        trusted += [] if name is None else [_label(folder / name, start)]
        trusted += [
            _label(folder / sub, start) for sub in (*_SKILL_FOLDERS, *_AGENT_FOLDERS) if os.path.isdir(folder / sub)
        ]
        approved += [
            (_label(folder / sub, start), how) for sub, how in _APPROVED_SEPARATELY if os.path.isfile(folder / sub)
        ]
    return RepositoryFiles(trusted=tuple(trusted), approved_separately=tuple(approved))


def _label(path: Path, start: Path) -> str:
    return Path(os.path.relpath(path, start)).as_posix()


def describe(trust: WorkspaceTrust, directory: Path) -> str:
    """What trusting the folder covers and changes, for the trust prompt and `/trust`."""
    scope = trust.scope
    covers = 'this repository and its worktrees' if scope.repository else 'this folder and its subfolders'
    lines = [f'{terminal_text(str(scope.key), keep="")}: {covers}.']
    if not scope.remembered:
        lines.append('This is your home folder or the filesystem root, so the answer holds for this session only.')
    files = repository_files(directory)
    if files.trusted:
        lines.append(
            f'Found here: {_names(files.trusted)} (agent folders still reach the model as yours either way, for now).'
        )
    lines.append(
        'Untrusted, the agent can still read, edit and run code here, '
        "and instruction files and skills reach the model as the repository's, not yours."
    )
    if files.approved_separately:
        waiting = ', '.join(f'{_names((name,))} ({how})' for name, how in files.approved_separately)
        lines.append(f'Either way, these still wait for their own approval: {waiting}.')
    return '\n'.join(lines)


def _names(names: Sequence[str]) -> str:
    return ', '.join(terminal_text(name, keep='') for name in names)


def trust_command(trust: WorkspaceTrust, store: SettingsStore, directory: Path, args: list[str]) -> str:
    """`/trust [yes|no]`: show whether this session's folder is trusted, or trust it or not."""
    if not args:
        if trust.decided_for is None:
            source = ', as every folder is until you decide'
        elif trust.decided_for != trust.scope.key:
            source = f', as you decided for {terminal_text(str(trust.decided_for), keep="")}'
        else:
            source = ''
        return (
            f'This folder is {trust.level}{source}.\n{describe(trust, directory)}\n'
            '/trust yes trusts it; /trust no keeps it untrusted.'
        )
    if args not in (['yes'], ['no']):
        raise ValueError('Usage: /trust [yes|no]')
    trust.decide(store, trusted=args == ['yes'])
    held = '' if trust.scope.remembered else ' for this session'
    key = terminal_text(str(trust.scope.key), keep='')
    if trust.trusted:
        return f'Trusted {key}{held}. Applies from the next prompt.'
    return (
        f'Untrusted {key}{held}. Applies from the next prompt; '
        'what the model has already read stays in this conversation, and /clear starts a new one.'
    )


def undecided_notice(trust: WorkspaceTrust | None) -> str | None:
    """For a session nobody is asked in, as with `pax -p` or `pax --web`: that its folder stays untrusted, or `None`."""
    if trust is None or trust.decided:
        return None
    return (
        f'{terminal_text(str(trust.scope.key), keep="")} is untrusted, as you have not decided whether to trust it. '
        'Start Pax there and answer its question, or enter /trust yes, to trust it.'
    )
