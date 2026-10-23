"""Exercise the dynamic-versioning fetch step against local Git remotes."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
import yaml

SHARED_STEPS = Path(__file__).parent.parent / 'workflows' / 'shared' / 'pre-agent-steps.md'
VERSION_TAG = 'v2.1.0'
BLEEDING_TAG = 'clai2-bleeding'


def _git(*args: str, cwd: Path | None = None) -> str:
    result = subprocess.run(['git', *args], cwd=cwd, capture_output=True, text=True, check=True)
    return result.stdout.strip()


def _fetch_script() -> str:
    frontmatter = SHARED_STEPS.read_text(encoding='utf-8').split('---', 2)[1]
    steps = yaml.safe_load(frontmatter)
    return next(
        step['run']
        for step in steps['pre-agent-steps']
        if step.get('name') == 'Restore full history and tags for dynamic versioning'
    )


def _git_repositories(tmp_path: Path) -> tuple[Path, Path, str]:
    origin = tmp_path / 'origin.git'
    seed = tmp_path / 'seed'
    origin.mkdir()
    _git('init', '--bare', str(origin))
    seed.mkdir()
    _git('init', '-b', 'main', cwd=seed)
    _git('config', 'user.name', 'Tag fetch test', cwd=seed)
    _git('config', 'user.email', 'tag-fetch@example.com', cwd=seed)
    (seed / 'history.txt').write_text('release\n', encoding='utf-8')
    _git('add', 'history.txt', cwd=seed)
    _git('commit', '-m', 'release', cwd=seed)
    release_commit = _git('rev-parse', 'HEAD', cwd=seed)
    _git('tag', VERSION_TAG, cwd=seed)
    _git('tag', BLEEDING_TAG, cwd=seed)
    _git('remote', 'add', 'origin', str(origin), cwd=seed)
    _git('push', '-u', 'origin', 'main', cwd=seed)
    _git('push', 'origin', '--tags', cwd=seed)
    _git('symbolic-ref', 'HEAD', 'refs/heads/main', cwd=origin)
    return origin, seed, release_commit


@pytest.mark.parametrize('shallow', [True, False], ids=['shallow', 'full'])
@pytest.mark.parametrize('remote_has_bleeding_tag', [True, False], ids=['moved-tag', 'no-remote-tag'])
def test_dynamic_version_fetch_handles_mutable_bleeding_tag(
    tmp_path: Path, shallow: bool, remote_has_bleeding_tag: bool
):
    """Fetch history and immutable version tags when the bleeding tag moves or disappears."""
    origin, seed, release_commit = _git_repositories(tmp_path)
    clone = tmp_path / 'checkout'
    clone_args = ['clone']
    if shallow:
        clone_args.extend(['--depth=1', origin.as_uri()])
    else:
        clone_args.append(str(origin))
    clone_args.append(str(clone))
    _git(*clone_args)

    # Mutate the remote only after cloning so checkout retains the old tag value.
    moved_commit: str | None = None
    if remote_has_bleeding_tag:
        (seed / 'history.txt').write_text('release\nnext\n', encoding='utf-8')
        _git('commit', '-am', 'advance main', cwd=seed)
        moved_commit = _git('rev-parse', 'HEAD', cwd=seed)
        _git('tag', '-f', BLEEDING_TAG, moved_commit, cwd=seed)
        _git('push', 'origin', 'main', cwd=seed)
        _git('push', '--force', 'origin', f'refs/tags/{BLEEDING_TAG}', cwd=seed)
    else:
        _git('tag', '-d', BLEEDING_TAG, cwd=seed)
        _git('push', 'origin', f':refs/tags/{BLEEDING_TAG}', cwd=seed)

    script = tmp_path / 'fetch-tags.sh'
    script.write_text(_fetch_script(), encoding='utf-8')
    result = subprocess.run(['bash', str(script)], cwd=clone, capture_output=True, text=True)

    assert result.returncode == 0, result.stderr
    assert _git('describe', '--tags', '--match', 'v*', cwd=clone).startswith(VERSION_TAG)
    assert _git('rev-parse', VERSION_TAG, cwd=clone) == release_commit
    git_dir = Path(_git('rev-parse', '--git-dir', cwd=clone))
    assert not (clone / git_dir / 'shallow').exists()
    if remote_has_bleeding_tag:
        assert moved_commit is not None
        assert _git('rev-parse', BLEEDING_TAG, cwd=clone) == moved_commit
    else:
        assert _git('tag', '--list', BLEEDING_TAG, cwd=clone) == ''
