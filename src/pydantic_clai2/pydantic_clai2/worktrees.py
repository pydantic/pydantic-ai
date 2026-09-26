"""Git worktrees for isolated CLI workspaces."""

import os
import re
import subprocess
import sys
from pathlib import Path
from uuid import uuid4


def create_worktree(*, name: str) -> Path:
    """Create a checkout in `.worktrees` on a new `clai/<name>` branch."""
    name = name or f'worktree-{uuid4().hex[:8]}'
    if re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*', name) is None:
        raise ValueError('Worktree names must start with a letter or digit and contain only letters, digits, - or _.')
    try:
        root = Path(_git('rev-parse', '--show-toplevel'))
        exclude = Path(_git('rev-parse', '--git-path', 'info/exclude'))
        contents = exclude.read_bytes() if exclude.exists() else b''
        path = root / '.worktrees' / name
        branch = f'clai/{name}'
        _git('branch', branch, 'HEAD')
        try:
            _git('worktree', 'add', '--', str(path), branch)
        except (OSError, subprocess.CalledProcessError) as exc:
            try:
                _git('branch', '-d', '--', branch)
            except (OSError, subprocess.CalledProcessError) as cleanup:
                raise ValueError(
                    f'Cannot create worktree at {path}: {exc}. Branch {branch} could not be removed: {cleanup}'
                ) from cleanup
            raise
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
    return path


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
