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
    """The checkout CLAI starts in, and whether this launch created it."""

    path: Path
    branch: str
    created: bool


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
        if os.path.realpath(path) in _registered_worktrees():
            branch = _git('-C', str(path), 'branch', '--show-current') or 'detached HEAD'
            worktree = Worktree(path=path, branch=branch, created=False)
        elif path.exists():
            raise ValueError(f'Cannot create worktree: {path} exists but is not a Git worktree. Pick another name.')
        else:
            worktree = Worktree(path=path, branch=_check_out(path=path, name=name), created=True)
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


def _check_out(*, path: Path, name: str) -> str:
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
    return branch


def _registered_worktrees() -> set[str]:
    # `realpath`, not `Path.resolve`: it never raises, even on a symlink loop planted at `.worktrees/NAME`.
    listing = _git('worktree', 'list', '--porcelain').splitlines()
    return {os.path.realpath(line.removeprefix('worktree ')) for line in listing if line.startswith('worktree ')}


def offer_worktree_cleanup() -> None:
    """Offer removal after interactive shutdown, keeping the branch and dirty files."""
    if not sys.stdin.isatty():
        return
    try:
        root = Path(_git('rev-parse', '--show-toplevel')).resolve()
        common = Path(_git('rev-parse', '--git-common-dir')).resolve()
        git_dir = Path(_git('rev-parse', '--absolute-git-dir')).resolve()
    except (OSError, subprocess.CalledProcessError):
        return
    if git_dir == common:
        return
    try:
        answer = input(f'Remove worktree {root}? The branch will be kept. [y/N] ')
    except (EOFError, KeyboardInterrupt):
        answer = ''
        print()
    if answer.strip().lower() not in ('y', 'yes'):
        print(f'Worktree kept at {root}.')
        return
    original = Path.cwd()
    try:
        # Run outside the checkout so successful removal leaves a valid working directory.
        os.chdir(common.parent)
        _git('-C', str(common), 'worktree', 'remove', '--', str(root))
    except (OSError, subprocess.CalledProcessError) as exc:
        os.chdir(original)
        detail = exc.stderr.strip() if isinstance(exc, subprocess.CalledProcessError) else str(exc)
        print(f'Worktree kept at {root}: {detail}', file=sys.stderr)
    else:
        print(f'Removed worktree {root}. Branch kept.')


def _git(*args: str) -> str:
    return subprocess.run(['git', *args], check=True, capture_output=True, text=True).stdout.removesuffix('\n')
