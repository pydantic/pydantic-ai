import os
from pathlib import Path

import pytest


def tools_path(directory: Path, *, exclude: frozenset[str] = frozenset({'rg'})) -> str:
    """A `PATH` directory with the system tools except `exclude`."""
    # GitHub's Ubuntu runners install rg in /usr/bin, so `PATH=/usr/bin:/bin` alone does not hide it.
    directory.mkdir(parents=True, exist_ok=True)
    for source in (Path('/usr/bin'), Path('/bin')):
        for tool in source.iterdir():
            link = directory / tool.name
            if tool.name not in exclude and not os.path.lexists(link):
                link.symlink_to(tool)
    return str(directory)


@pytest.fixture(scope='session')
def no_rg_path(tmp_path_factory: pytest.TempPathFactory) -> str:
    """A `PATH` with the system tools but not `rg`, to exercise the POSIX search fallback."""
    return tools_path(tmp_path_factory.mktemp('no-rg-bin'))


@pytest.fixture(scope='session')
def no_rg_git_path(tmp_path_factory: pytest.TempPathFactory) -> str:
    """A `PATH` without `rg` or `git`, to exercise the POSIX search's `find` enumeration."""
    return tools_path(tmp_path_factory.mktemp('no-rg-git-bin'), exclude=frozenset({'rg', 'git'}))
