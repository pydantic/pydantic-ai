import os
from pathlib import Path

import pytest


@pytest.fixture(scope='session')
def no_rg_path(tmp_path_factory: pytest.TempPathFactory) -> str:
    """A `PATH` with the system tools but not `rg`, to exercise the POSIX search fallback."""
    # GitHub's Ubuntu runners install rg in /usr/bin, so `PATH=/usr/bin:/bin` alone does not hide it.
    directory = tmp_path_factory.mktemp('no-rg-bin')
    for source in (Path('/usr/bin'), Path('/bin')):
        for tool in source.iterdir():
            link = directory / tool.name
            if tool.name != 'rg' and not os.path.lexists(link):
                link.symlink_to(tool)
    return str(directory)
