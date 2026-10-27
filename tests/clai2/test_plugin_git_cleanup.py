"""Incomplete checkouts are removable even with Windows' read-only Git object files."""

import asyncio
import os
import stat
from collections.abc import Callable
from pathlib import Path
from types import TracebackType

import pytest

from pydantic_clai2.plugins import _git
from pydantic_clai2.plugins._git import remove_checkout

from .test_plugin_loader import Harness


@pytest.fixture
def windows_unlink(monkeypatch: pytest.MonkeyPatch) -> None:
    """Model Windows' read-only unlink failure while using real files and rmtree."""
    unlink = os.unlink

    def unlink_readonly(
        path: str | bytes | os.PathLike[str] | os.PathLike[bytes], *, dir_fd: int | None = None
    ) -> None:
        mode = os.stat(path, dir_fd=dir_fd, follow_symlinks=False).st_mode
        if not mode & stat.S_IWRITE:
            raise PermissionError('read-only file')
        unlink(path, dir_fd=dir_fd)

    monkeypatch.setattr(os, 'unlink', unlink_readonly)


def readonly_pack(destination: Path) -> Path:
    pack = destination / '.git' / 'objects' / 'pack' / 'objects.pack'
    pack.parent.mkdir(parents=True, exist_ok=True)
    pack.write_bytes(b'Git pack data')
    pack.chmod(stat.S_IREAD)
    return pack


@pytest.mark.usefixtures('windows_unlink')
def test_remove_checkout_clears_readonly_git_objects(tmp_path: Path) -> None:
    destination = tmp_path / 'checkout'
    readonly_pack(destination)
    remove_checkout(destination, windows=True)
    assert not destination.exists()


@pytest.mark.usefixtures('windows_unlink')
async def test_invalid_clone_rolls_back_readonly_git_objects(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    async def clone(url: str, destination: Path) -> tuple[int, bytes]:
        readonly_pack(destination)
        return 0, b''

    def remove_windows(destination: Path) -> None:
        remove_checkout(destination, windows=True)

    monkeypatch.setattr(_git, 'clone_repository', clone)
    monkeypatch.setattr(_git, 'remove_checkout', remove_windows)
    harness = Harness(tmp_path)
    with pytest.raises(ValueError, match=r'regular __init__\.py or plugin\.py'):
        await harness.loader.command(['add', 'https://example.invalid/plugin.git'])
    assert harness.store.plugins() == []
    assert not (harness.store.plugins_dir / '_git' / 'plugin').exists()


@pytest.mark.parametrize('cancel', [False, True])
async def test_cleanup_failure_preserves_original_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cancel: bool
) -> None:
    async def clone(url: str, destination: Path) -> tuple[int, bytes]:
        if cancel:
            raise asyncio.CancelledError
        return 1, b'original clone failure'

    def remove(destination: Path) -> None:
        raise PermissionError('checkout still locked')

    monkeypatch.setattr(_git, 'clone_repository', clone)
    monkeypatch.setattr(_git, 'remove_checkout', remove)
    harness = Harness(tmp_path)
    expected = asyncio.CancelledError if cancel else ValueError
    with pytest.raises(expected, match=None if cancel else 'original clone failure'):
        await harness.loader.command(['add', 'https://example.invalid/plugin.git'])
    assert harness.store.plugins() == []
    assert (harness.store.plugins_dir / '_git' / 'plugin').exists()


@pytest.mark.parametrize(
    'case',
    [
        'other-error',
        'other-operation',
        pytest.param('symlink', marks=pytest.mark.skipif(os.name == 'nt', reason='requires creating a file symlink')),
    ],
)
def test_readonly_retry_preserves_other_errors_and_symlink_targets(tmp_path: Path, case: str) -> None:
    target = tmp_path / 'outside-checkout'
    target.write_text('keep its permissions')
    mode = target.stat().st_mode
    path = tmp_path / 'link'
    if case == 'symlink':
        path.symlink_to(target)
    error = OSError('original') if case == 'other-error' else PermissionError('original')
    operation = os.stat if case == 'other-operation' else os.unlink
    with pytest.raises(OSError) as captured:
        _git.retry_readonly(operation, str(path), error)
    assert captured.value is error
    assert target.stat().st_mode == mode


@pytest.mark.usefixtures('windows_unlink')
def test_legacy_readonly_callback(tmp_path: Path) -> None:
    pack = readonly_pack(tmp_path)
    try:
        raise PermissionError('read-only file')
    except PermissionError as error:
        assert error.__traceback__ is not None
        _git.retry_readonly_legacy(os.unlink, str(pack), (type(error), error, error.__traceback__))
    assert not pack.exists()


def test_legacy_rmtree_uses_error_adapter(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[Path] = []

    def remove(
        path: Path,
        *,
        onerror: Callable[
            [Callable[[str], object], str, tuple[type[BaseException], BaseException, TracebackType]], None
        ],
    ) -> None:
        assert onerror is _git.retry_readonly_legacy
        calls.append(path)

    with monkeypatch.context() as patch:
        patch.setattr(_git.sys, 'version_info', (3, 11))
        patch.setattr(_git.shutil, 'rmtree', remove)
        remove_checkout(tmp_path, windows=True)
    assert calls == [tmp_path]
