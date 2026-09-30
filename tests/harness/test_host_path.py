"""Tests for the `PATH` filter that the host-side scripts of the SSH and bubblewrap workspaces share."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from pydantic_ai_harness._host_path import TRUSTED_PATH_FUNCTIONS

pytestmark = pytest.mark.skipif(os.name != 'posix', reason='POSIX shell functions')


@pytest.mark.parametrize('shell', [path for name in ('dash', 'bash', 'sh') if (path := shutil.which(name))])
def test_entries_a_sandboxed_command_could_write_are_left_out(shell: str, tmp_path: Path) -> None:
    wd = tmp_path / 'wd'
    (wd / 'bin').mkdir(parents=True)
    outside = tmp_path / 'outside'
    outside.mkdir()
    alias = tmp_path / 'alias'
    alias.symlink_to(wd / 'bin', target_is_directory=True)
    # A canonical directory with a `:` would split into a relative entry.
    (tmp_path / 'a:b').mkdir()
    (tmp_path / 'colon').symlink_to(tmp_path / 'a:b', target_is_directory=True)
    entries = ['relative', '', str(wd), str(wd / 'bin'), str(alias), str(tmp_path / 'missing'), str(tmp_path / 'colon')]
    path = os.pathsep.join([*entries, str(outside)])

    def trusted(directory: str) -> str:
        script = f'{TRUSTED_PATH_FUNCTIONS}__pai_trusted_path "$1"'
        result = subprocess.run(
            [shell, '-c', script, shell, directory], env={'PATH': path}, capture_output=True, text=True
        )
        return result.stdout

    wd_bin = str((wd / 'bin').resolve())
    assert trusted(str(wd.resolve())) == f'{outside.resolve()}\n'
    assert trusted('/') == '\n'
    assert trusted('') == f'{wd.resolve()}:{wd_bin}:{wd_bin}:{outside.resolve()}\n'
