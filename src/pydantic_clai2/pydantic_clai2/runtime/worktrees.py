"""Git worktrees for isolated CLI workspaces."""

import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4


@dataclass(frozen=True, kw_only=True)
class Worktree:
    """The linked checkout CLAI starts in, as it was at launch."""

    path: Path
    branch: str
    head: str
    """The commit checked out at launch; a different one on exit counts as a change."""
    created: bool
    """Whether this launch added the checkout."""
    new_branch: bool = False
    """Whether this launch also created `branch`, so exit deletes it with an unchanged checkout."""
    remove_if_unchanged: bool = False
    """Whether an interactive exit removes the checkout without asking when nothing changed."""

    @property
    def notice(self) -> str:
        """The launch line naming the checkout and its branch."""
        opened = 'Worktree' if self.created else 'Reopened worktree'
        on_exit = (
            'Removed on exit if unchanged.' if self.remove_if_unchanged else 'Kept unless removal is confirmed on exit.'
        )
        return f'{opened}: {self.path} (branch: {self.branch}). {on_exit}'


def open_worktree(*, name: str) -> Worktree:
    """Reopen `.worktrees/NAME`, or check it out on `clai-NAME`, reusing that branch if it exists."""
    name = name or f'worktree-{uuid4().hex[:8]}'
    if re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*', name) is None:
        raise ValueError('Worktree names must start with a letter or digit and contain only letters, digits, - or _.')
    try:
        root = Path(_git('rev-parse', '--show-toplevel'))
        exclude = Path(_git('rev-parse', '--git-path', 'info/exclude'))
        contents = exclude.read_bytes() if exclude.exists() else b''
        path = root / '.worktrees' / name
        registered = os.path.realpath(path) in _registered_worktrees()
        if registered and path.exists():
            branch = _git('-C', str(path), 'branch', '--show-current') or 'detached HEAD'
            worktree = Worktree(
                path=path, branch=branch, head=_git('-C', str(path), 'rev-parse', 'HEAD'), created=False
            )
        elif path.exists():
            raise ValueError(f'Cannot create worktree: {path} exists but is not a Git worktree. Pick another name.')
        else:
            if registered:
                _git('worktree', 'prune')  # The checkout was deleted by hand; Git still lists it until pruned.
            worktree = _check_out(path=path, name=name)
        # Also when reopening: a worktree made with plain `git worktree add` has no exclude entry yet.
        if b'/.worktrees/' not in contents.splitlines():
            try:
                exclude.parent.mkdir(parents=True, exist_ok=True)
                with exclude.open('ab') as file:
                    file.write(b'\n/.worktrees/\n')
            except OSError as exc:
                raise ValueError(f'Worktree kept at {path}, but could not update Git excludes: {exc}') from exc
    except subprocess.CalledProcessError as exc:
        raise ValueError(f'Cannot create worktree: {exc.stderr.strip()}') from exc
    except OSError as exc:
        raise ValueError(f'Cannot create worktree: {exc}') from exc
    return worktree


def _check_out(*, path: Path, name: str) -> Worktree:
    """Add the checkout on `clai-NAME`, creating that branch from `HEAD` only if it is missing."""
    # Flat on purpose: Git refs are paths, so a nested `clai/NAME` cannot coexist with a user's `clai` branch.
    branch = f'clai-{name}'
    new_branch = not _git('branch', '--list', branch)
    if new_branch:
        _git('branch', branch, 'HEAD')
    try:
        _git('worktree', 'add', '--', str(path), branch)
    except (OSError, subprocess.CalledProcessError) as exc:
        if not new_branch:
            raise
        try:
            _git('branch', '-d', '--', branch)
        except (OSError, subprocess.CalledProcessError) as cleanup:
            raise ValueError(
                f'Cannot create worktree at {path}: {exc}. Branch {branch} could not be removed: {cleanup}'
            ) from cleanup
        raise
    head = _git('-C', str(path), 'rev-parse', 'HEAD')
    return Worktree(path=path, branch=branch, head=head, created=True, new_branch=new_branch)


def _registered_worktrees() -> set[str]:
    # `realpath`, not `Path.resolve`: it never raises, even on a symlink loop planted at `.worktrees/NAME`.
    listing = _git('worktree', 'list', '--porcelain').splitlines()
    return {os.path.realpath(line.removeprefix('worktree ')) for line in listing if line.startswith('worktree ')}


def current_worktree() -> Worktree | None:
    """The linked worktree the working directory is in, or `None` in a main checkout or outside Git."""
    try:
        # One Git call on every interactive launch; `--abbrev-ref HEAD` prints the branch, or `HEAD` when detached.
        root, common, git_dir, head, branch = _git(
            'rev-parse', '--show-toplevel', '--git-common-dir', '--absolute-git-dir', 'HEAD', '--abbrev-ref', 'HEAD'
        ).splitlines()
    except (OSError, subprocess.CalledProcessError):
        return None
    if Path(git_dir).resolve() == Path(common).resolve():
        return None
    branch = 'detached HEAD' if branch == 'HEAD' else branch
    return Worktree(path=Path(root).resolve(), branch=branch, head=head, created=False)


def offer_worktree_cleanup(*, worktree: Worktree | None) -> None:
    """After interactive shutdown, remove an unchanged checkout marked `remove_if_unchanged`, or offer to remove a changed one.

    Uncommitted or untracked files, or a commit other than the one at launch, count as changes. Other
    unchanged checkouts are kept without asking. A confirmed removal keeps the branch and dirty files.
    """
    if worktree is None or not sys.stdin.isatty():
        return
    root = str(worktree.path)
    try:
        changed = (
            bool(_git('-C', root, 'status', '--porcelain')) or _git('-C', root, 'rev-parse', 'HEAD') != worktree.head
        )
    except (OSError, subprocess.CalledProcessError):
        return
    if not changed:
        if worktree.remove_if_unchanged:
            _remove(worktree, unchanged=True)
        return
    try:
        answer = input(f'Remove worktree {root}? The branch will be kept. [y/N] ')
    except (EOFError, KeyboardInterrupt):
        answer = ''
        print()
    if answer.strip().lower() not in ('y', 'yes'):
        print(f'Worktree kept at {root}.')
        return
    if _remove(worktree, unchanged=False):
        print(f'Removed worktree {root}. Branch kept.')


def _remove(worktree: Worktree, *, unchanged: bool) -> bool:
    """Remove the checkout without `--force`, and an unchanged branch this launch created; report a refusal."""
    root = str(worktree.path)
    original = Path.cwd()
    try:
        common = (worktree.path / _git('-C', root, 'rev-parse', '--git-common-dir')).resolve()
        # Run outside the checkout so successful removal leaves a valid working directory.
        os.chdir(common.parent)
        _git('-C', str(common), 'worktree', 'remove', '--', root)
    except (OSError, subprocess.CalledProcessError) as exc:
        os.chdir(original)
        detail = exc.stderr.strip() if isinstance(exc, subprocess.CalledProcessError) else str(exc)
        print(f'Worktree kept at {root}: {detail}', file=sys.stderr)
        return False
    if unchanged and worktree.new_branch:
        try:
            # `-d`, not `-D`: Git keeps a branch that is not merged into the main checkout's `HEAD`.
            _git('-C', str(common), 'branch', '-d', '--', worktree.branch)
        except (OSError, subprocess.CalledProcessError):
            pass  # A kept branch loses nothing; it is `git branch -d` away.
    return True


def _git(*args: str) -> str:
    return subprocess.run(['git', *args], check=True, capture_output=True, text=True).stdout.removesuffix('\n')
