"""Copy the local checkout into a new sandbox: its branch, its commits, and its uncommitted changes.

`snapshot` reads the checkout without changing it: the uncommitted files go into a commit made with a
temporary index, which no branch points to. `seed` clones a GitHub checkout in the sandbox, signed in as
the user, then sends only the commits that clone lacks, in a Git bundle; any other
checkout travels whole. It then rebuilds the same branch and files in the sandbox.
"""

import os
import re
import shlex
import subprocess
import tempfile
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from anyio import to_thread

from pydantic_ai.workspaces import SupportsCommands, Workspace, WorkspaceBackend

_SEED_REFS = 'refs/clai-seed/'
"""Where `_bundle` names the commit it bundles, one ref per call, so concurrent snapshots of linked worktrees never share one."""
_GITHUB = re.compile(
    r'^(?:https://github\.com/|git@github\.com:|ssh://git@github\.com/)([\w.-]+/[\w.-]+?)(?:\.git)?/?$'
)
GITHUB_TOKEN_FILE = '.config/clai/github-token'
"""Where, under the sandbox's home, a provider without a proxy to sign requests puts the GitHub token."""
_SEED_TIMEOUT = 900.0
"""Seconds the sandbox may take to clone and rebuild the checkout; a large repository clones slowly."""


class SandboxBackend(WorkspaceBackend, SupportsCommands, Protocol):
    """A sandbox's workspace: it runs commands. Files go through `Workspace`, which uses the shell when it has no file API."""


@dataclass(frozen=True, kw_only=True)
class Snapshot:
    """What `seed` rebuilds in the sandbox; all `None` for a folder outside Git."""

    clone_url: str | None = None
    """The `https://github.com/...` URL the sandbox clones, from `origin`; `None` when it is not on GitHub."""
    branch: str | None = None
    """The checked-out branch; `None` when `HEAD` is detached."""
    head: str | None = None
    """The commit checked out; `None` in a repository with no commits yet."""
    work: str | None = None
    """The commit holding the uncommitted files too; `head` when there are none."""
    root: Path | None = field(default=None, compare=False)
    """The checkout's top folder, where `seed` bundles the commits the sandbox lacks."""
    author_name: str | None = None
    """Git's `user.name` here, for the sandbox's commits."""
    author_email: str | None = None
    """Git's `user.email` here, for the sandbox's commits."""

    @property
    def changed(self) -> bool:
        """Whether there were uncommitted changes, untracked files included."""
        return self.work != self.head


def github_clone_url(remote: str) -> str | None:
    """The HTTPS URL of a GitHub remote, as the proxy authenticates only HTTPS; `None` for any other remote."""
    found = _GITHUB.match(remote.strip())
    return f'https://github.com/{found.group(1)}.git' if found else None


def _git(root: Path, *args: str, env: dict[str, str] | None = None) -> str:
    result = subprocess.run(
        ['git', '-C', str(root), *args],
        capture_output=True,
        text=True,
        encoding='utf-8',
        env=env,
        stdin=subprocess.DEVNULL,
        check=False,
    )
    if result.returncode != 0:
        raise ValueError(f'git {args[0]} failed: {result.stderr.strip() or result.stdout.strip()}')
    return result.stdout.strip()


def _try_git(root: Path, *args: str) -> str | None:
    try:
        return _git(root, *args) or None
    except ValueError:
        return None


def snapshot(directory: Path) -> Snapshot:
    """Read the checkout at `directory`, or an empty `Snapshot` outside Git; `ValueError` when Git fails."""
    top = _try_git(directory, 'rev-parse', '--show-toplevel')
    if top is None:
        return Snapshot()
    root = Path(top)
    head = _try_git(root, 'rev-parse', '--verify', '--quiet', 'HEAD^{commit}')
    remote = _try_git(root, 'remote', 'get-url', 'origin')
    clone_url = github_clone_url(remote) if remote else None
    branch = _try_git(root, 'symbolic-ref', '--short', '-q', 'HEAD')
    name, email = _try_git(root, 'config', 'user.name'), _try_git(root, 'config', 'user.email')
    work = _work_commit(root, head)
    return Snapshot(
        clone_url=clone_url, branch=branch, head=head, work=work, root=root, author_name=name, author_email=email
    )


def _work_commit(root: Path, head: str | None) -> str | None:
    """`head`, or a commit on top of it holding every file Git would add, without touching the real index."""
    with tempfile.TemporaryDirectory() as scratch:
        env = {
            **os.environ,
            'GIT_INDEX_FILE': str(Path(scratch) / 'index'),
            'GIT_AUTHOR_NAME': 'clai',
            'GIT_AUTHOR_EMAIL': 'clai@localhost',
            'GIT_COMMITTER_NAME': 'clai',
            'GIT_COMMITTER_EMAIL': 'clai@localhost',
        }
        if head is not None:
            _git(root, 'read-tree', head, env=env)
        _git(root, 'add', '--all', env=env)
        tree = _git(root, 'write-tree', env=env)
        if head is not None and tree == _git(root, 'rev-parse', f'{head}^{{tree}}'):
            return head
        if head is None and not _git(root, 'ls-files', env=env):
            return None
        parent = ('-p', head) if head is not None else ()
        return _git(root, 'commit-tree', tree, *parent, '-m', 'Uncommitted changes', env=env)


def _bundle(root: Path, work: str, have: list[str]) -> tuple[str, bytes] | None:
    """The bundle of `work`'s history less the commits in `have`, and the ref it names; `None` when that is nothing."""
    present = _present(root, have)
    exclude = ('--not', *present) if present else ()
    ref = f'{_SEED_REFS}{uuid.uuid4().hex}'
    _git(root, 'update-ref', ref, work)
    try:
        if _git(root, 'rev-list', '--count', ref, *exclude) == '0':
            return None
        with tempfile.TemporaryDirectory() as scratch:
            path = Path(scratch) / 'seed.bundle'
            _git(root, 'bundle', 'create', '--quiet', str(path), ref, *exclude)
            return ref, path.read_bytes()
    finally:
        _git(root, 'update-ref', '-d', ref)


def _present(root: Path, commits: list[str]) -> list[str]:
    """The `commits` this repository has: a bundle can only leave out commits it knows."""
    if not commits:
        return []
    result = subprocess.run(
        ['git', '-C', str(root), 'cat-file', '--batch-check=%(objectname) %(objecttype)'],
        input='\n'.join(commits),
        capture_output=True,
        text=True,
        encoding='utf-8',
        check=True,
    )
    return [line.split()[0] for line in result.stdout.splitlines() if line.endswith(' commit')]


def clone_script(snapshot: Snapshot) -> str:
    """The shell script that sets up Git in the sandbox's empty working directory and clones `snapshot`'s origin.

    It prints the commits the clone's branches point to, which `seed` leaves out of the bundle.
    """
    q = shlex.quote
    lines = [
        'set -eu',
        # The GitHub login signs HTTPS requests only; SSH ones would need a key the sandbox does not have.
        'git config --global url.https://github.com/.insteadOf git@github.com:',
        'git config --global --add url.https://github.com/.insteadOf ssh://git@github.com/',
        # A proxy provider sets `GH_TOKEN` to a stand-in its proxy replaces, or replaces the whole header; any other
        # leaves `GH_TOKEN` unset, and `/sandbox` writes the token to `GITHUB_TOKEN_FILE`.
        'git config --global credential.https://github.com.helper '
        + shlex.quote(
            '!f() { echo username=x-access-token; '
            f'echo "password=${{GH_TOKEN:-$(cat "$HOME/{GITHUB_TOKEN_FILE}" 2>/dev/null)}}"; }}; f'
        ),
    ]
    if snapshot.author_name:
        lines.append(f'git config --global user.name {q(snapshot.author_name)}')
    if snapshot.author_email:
        lines.append(f'git config --global user.email {q(snapshot.author_email)}')
    if snapshot.clone_url is not None:
        lines += [
            f'git clone --quiet --no-checkout {q(snapshot.clone_url)} .',
            "git for-each-ref --format='%(objectname)' refs/remotes/origin",
        ]
    else:
        lines.append('git init --quiet')
    return '\n'.join(lines)


def checkout_script(snapshot: Snapshot, *, bundled: str | None) -> str:
    """The shell script that puts `snapshot`'s branch and files back, after `clone_script` and the bundle."""
    lines = ['set -eu']
    if bundled is not None:
        # Into FETCH_HEAD only: the checkout below names commits, and no ref is left behind.
        path = _bundle_path(bundled)
        lines += [f'git fetch --quiet {path} {bundled}', f'rm -f {path}']
    lines += _checkout(snapshot)
    if snapshot.changed and snapshot.work is not None:
        # The uncommitted files, as they were: changed in the working tree, not staged.
        lines += (
            [f'git read-tree -u -m HEAD {snapshot.work}', 'git reset --quiet']
            if snapshot.head is not None
            else [f'git read-tree -u --reset {snapshot.work}', 'git read-tree --empty']
        )
    return '\n'.join(lines)


def _checkout(snapshot: Snapshot) -> list[str]:
    """The commands that put `snapshot`'s branch and commit back."""
    branch = shlex.quote(snapshot.branch) if snapshot.branch else None
    if snapshot.head is None:
        return [f'git symbolic-ref HEAD refs/heads/{branch}'] if branch else []
    if branch is None:
        return [f'git checkout --quiet --detach {snapshot.head}']
    lines = [f'git checkout --quiet -B {branch} {snapshot.head}']
    if snapshot.clone_url is not None:
        lines.append(f'git branch --quiet --set-upstream-to=origin/{branch} 2>/dev/null || true')
    return lines


async def seed(backend: SandboxBackend, snapshot: Snapshot) -> None:
    """Rebuild `snapshot` in `backend`'s working directory; `ValueError` with Git's output when that fails."""
    have = (await _run(backend, clone_script(snapshot))).split()
    bundle = None
    if snapshot.root is not None and snapshot.work is not None:
        # Checked against the clone, not `origin`'s refs here, which may name commits GitHub no longer has.
        bundle = await to_thread.run_sync(_bundle, snapshot.root, snapshot.work, have)
    if bundle is not None:
        await Workspace(backend).write_bytes(_bundle_path(bundle[0]), bundle[1])
    await _run(backend, checkout_script(snapshot, bundled=bundle[0] if bundle is not None else None))


def _bundle_path(ref: str) -> str:
    """Where the bundle for `ref` goes in the sandbox, named after the ref so each copy has its own."""
    return f'/tmp/clai-seed-{ref.removeprefix(_SEED_REFS)}.bundle'


async def _run(backend: SandboxBackend, script: str) -> str:
    result = await backend.run(['bash', '-c', script], timeout=_SEED_TIMEOUT)
    if result.exit_code != 0:
        raise ValueError(f'Could not copy the checkout into the sandbox: {(result.stderr or result.stdout).strip()}')
    return result.stdout
